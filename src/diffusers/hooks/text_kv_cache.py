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

from .hooks import BaseState, HookRegistry, ModelHook, StateManager


_TEXT_KV_CACHE_TRANSFORMER_HOOK = "text_kv_cache_transformer"
_TEXT_KV_CACHE_BLOCK_HOOK = "text_kv_cache_block"


@dataclass
class TextKVCacheConfig:
    """Enable exact (lossless) text K/V caching for transformer models.

    Pre-computes per-block text key and value projections once before the denoising loop and reuses them across all
    steps. Positive and negative prompts are distinguished via a stable cache key captured by a transformer-level hook
    before any intermediate tensor allocations.
    """

    pass


class TextKVCacheState(BaseState):
    """Shared state between the transformer-level and block-level hooks.

    The transformer hook writes the stable ``encoder_hidden_states`` ``data_ptr()`` (captured *before* ``txt_norm``) so
    that block hooks can use it as a reliable cache key across denoising steps.
    """

    def __init__(self):
        self.key: int | None = None

    def reset(self):
        self.key = None


class TextKVCacheBlockState(BaseState):
    """Per-block state holding cached text key/value projections."""

    def __init__(self):
        self.kv_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def reset(self):
        self.kv_cache.clear()


class TextKVCacheTransformerHook(ModelHook):
    """Captures ``encoder_hidden_states.data_ptr()`` before ``txt_norm``
    and writes it to shared state for the block hooks to read."""

    _is_stateful = True

    def __init__(self, state_manager: StateManager):
        super().__init__()
        self.state_manager = state_manager

    def new_forward(self, module: torch.nn.Module, *args, **kwargs):
        if self.state_manager._current_context is None:
            self.state_manager.set_context("inference")

        encoder_hidden_states = kwargs.get("encoder_hidden_states")
        if encoder_hidden_states is not None:
            state: TextKVCacheState = self.state_manager.get_state()
            state.key = encoder_hidden_states.data_ptr()
        return self.fn_ref.original_forward(*args, **kwargs)

    def reset_state(self, module: torch.nn.Module):
        self.state_manager.reset()
        return module


class TextKVCacheBlockHook(ModelHook):
    """Caches ``(txt_key, txt_value)`` per block per unique prompt using
    the stable cache key from the shared state."""

    _is_stateful = True

    def __init__(self, state_manager: StateManager, block_state_manager: StateManager):
        super().__init__()
        self.state_manager = state_manager
        self.block_state_manager = block_state_manager

    def new_forward(self, module: torch.nn.Module, *args, **kwargs):
        from ..models.transformers.transformer_nucleusmoe_image import _apply_rotary_emb_nucleus

        if self.state_manager._current_context is None:
            self.state_manager.set_context("inference")

        if self.block_state_manager._current_context is None:
            self.block_state_manager.set_context("inference")

        if "encoder_hidden_states" in kwargs:
            encoder_hidden_states = kwargs["encoder_hidden_states"]
        else:
            encoder_hidden_states = args[1]

        if "image_rotary_emb" in kwargs:
            image_rotary_emb = kwargs["image_rotary_emb"]
        elif len(args) > 3:
            image_rotary_emb = args[3]
        else:
            image_rotary_emb = None

        state: TextKVCacheState = self.state_manager.get_state()
        cache_key = state.key

        block_state: TextKVCacheBlockState = self.block_state_manager.get_state()

        if cache_key not in block_state.kv_cache:
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

            block_state.kv_cache[cache_key] = (txt_key, txt_value)

        txt_key, txt_value = block_state.kv_cache[cache_key]

        attn_kwargs = kwargs.get("attention_kwargs") or {}
        attn_kwargs["cached_txt_key"] = txt_key
        attn_kwargs["cached_txt_value"] = txt_value
        kwargs["attention_kwargs"] = attn_kwargs

        return self.fn_ref.original_forward(*args, **kwargs)

    def reset_state(self, module: torch.nn.Module):
        self.block_state_manager.reset()
        return module


def apply_text_kv_cache(module: torch.nn.Module, config: TextKVCacheConfig) -> None:
    from ..models.transformers.transformer_nucleusmoe_image import NucleusMoEImageTransformerBlock

    HookRegistry.check_if_exists_or_initialize(module)

    state_manager = StateManager(TextKVCacheState)

    transformer_hook = TextKVCacheTransformerHook(state_manager)
    registry = HookRegistry.check_if_exists_or_initialize(module)
    registry.register_hook(transformer_hook, _TEXT_KV_CACHE_TRANSFORMER_HOOK)

    for _, submodule in module.named_modules():
        if isinstance(submodule, NucleusMoEImageTransformerBlock):
            block_state_manager = StateManager(TextKVCacheBlockState)
            hook = TextKVCacheBlockHook(state_manager, block_state_manager)
            block_registry = HookRegistry.check_if_exists_or_initialize(submodule)
            block_registry.register_hook(hook, _TEXT_KV_CACHE_BLOCK_HOOK)
