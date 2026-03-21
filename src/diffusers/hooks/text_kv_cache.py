# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass

import torch

from .hooks import HookRegistry, ModelHook


_TEXT_KV_CACHE_HOOK = "text_kv_cache"


@dataclass
class TextKVCacheConfig:
    """Enable exact (lossless) text K/V caching for transformer models.

    Pre-computes per-block text key and value projections once before the
    denoising loop and reuses them across all steps. The cached values are keyed by
    the ``data_ptr()`` of the ``encoder_hidden_states`` tensor so that both the positive
    and negative prompts (when ``true_cfg_scale > 1``) are handled correctly.
    """

    pass  # no hyperparameters needed — cache is always exact


class TextKVCacheHook(ModelHook):
    """Block-level hook that caches (txt_key, txt_value) per unique prompt."""

    _is_stateful = True

    def __init__(self):
        super().__init__()
        # Maps encoder_hidden_states.data_ptr() → (txt_key, txt_value)
        self.kv_cache: dict[int, tuple] = {}

    def new_forward(self, module: torch.nn.Module, *args, **kwargs):
        from ..models.transformers.transformer_nucleusmoe_image import _apply_rotary_emb_nucleus

        # --- extract encoder_hidden_states ---
        if "encoder_hidden_states" in kwargs:
            encoder_hidden_states = kwargs["encoder_hidden_states"]
        else:
            # positional: (hidden_states, encoder_hidden_states, temb, ...)
            encoder_hidden_states = args[1]

        # --- extract image_rotary_emb ---
        if "image_rotary_emb" in kwargs:
            image_rotary_emb = kwargs.get("image_rotary_emb")
        elif len(args) > 3:
            image_rotary_emb = args[3]
        else:
            image_rotary_emb = None

        ptr = encoder_hidden_states.data_ptr()

        if ptr not in self.kv_cache:
            context = module.encoder_proj(encoder_hidden_states)

            attn = module.attn
            head_dim = attn.inner_dim // attn.heads
            num_kv_heads = attn.inner_kv_dim // head_dim

            txt_key = attn.add_k_proj(context).unflatten(-1, (num_kv_heads, -1))
            txt_value = attn.add_v_proj(context).unflatten(-1, (num_kv_heads, -1))

            if attn.norm_added_k is not None:
                txt_key = attn.norm_added_k(txt_key)

            if image_rotary_emb is not None:
                _, txt_freqs = image_rotary_emb
                txt_key = _apply_rotary_emb_nucleus(txt_key, txt_freqs, use_real=False)

            self.kv_cache[ptr] = (txt_key, txt_value)

        txt_key, txt_value = self.kv_cache[ptr]

        # Inject cached k/v — block sees cached_txt_key and skips encoder_proj too
        attn_kwargs = kwargs.get("attention_kwargs") or {}
        attn_kwargs["cached_txt_key"] = txt_key
        attn_kwargs["cached_txt_value"] = txt_value
        kwargs["attention_kwargs"] = attn_kwargs

        return self.fn_ref.original_forward(*args, **kwargs)

    def reset_state(self, module: torch.nn.Module):
        self.kv_cache.clear()
        return module


def apply_text_kv_cache(module: torch.nn.Module, config: TextKVCacheConfig) -> None:
    from ..models.transformers.transformer_nucleusmoe_image import NucleusMoEImageTransformerBlock

    for _, submodule in module.named_modules():
        if isinstance(submodule, NucleusMoEImageTransformerBlock):
            hook = TextKVCacheHook()
            registry = HookRegistry.check_if_exists_or_initialize(submodule)
            registry.register_hook(hook, _TEXT_KV_CACHE_HOOK)
