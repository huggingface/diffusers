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

from ..models.attention_processor import Attention
from ..utils import get_logger
from ..utils.torch_utils import unwrap_module
from ._common import _ALL_TRANSFORMER_BLOCK_IDENTIFIERS
from ._helpers import TransformerBlockRegistry
from .hooks import HookRegistry, ModelHook


logger = get_logger(__name__)  # pylint: disable=invalid-name

ROLLING_KV_CACHE_HOOK = "rolling_kv_cache_hook"


@dataclass
class RollingKVCacheConfig:
    local_attn_size: int = -1
    num_sink_tokens: int = 1
    frame_seq_length: int = 128
    batch_size: int = 1
    num_layers: int = None
    max_seq_length: int = 32760


# One hook per each attention layer
class RollingKVCachekHook(ModelHook):
    _is_stateful = True

    def __init__(
        self,
        batch_size: int,
        max_seq_length: int,
        num_sink_tokens: int,
        frame_seq_length: int,
        num_layers: int,
        local_attn_size: int,
        layer_idx: int = None,
    ):
        self.batch_size = batch_size
        self.num_sink_tokens = num_sink_tokens
        self.num_layers = num_layers
        self.layer_idx = layer_idx
        if local_attn_size != -1:
            self.max_seq_length = local_attn_size * frame_seq_length
        else:
            self.max_seq_length = max_seq_length
        self._metadata = None
        self.cache_initialized = False

    def initialize_hook(self, module):
        unwrapped_module = unwrap_module(module)
        self._metadata = TransformerBlockRegistry.get(unwrapped_module.__class__)

        # No access to config anymore from each transformer block? Would be great to get dims from config
        self.self_attn_kv_shape = [
            self.batch_size,
            self.max_seq_length,
            module.num_heads,
            module.dim // module.num_heads,
        ]
        self.cross_attn_kv_shape = [
            self.batch_size,
            module.encoder_dim,
            module.num_heads,
            module.dim // module.num_heads,
        ]
        return module

    def lazy_initialize_cache(self, device: str, dtype: torch.dtype):
        """
        Initialize a Per-GPU KV cache for the Wan model.
        """
        if not self.cache_initialized:
            self.cache = CacheLayer(
                self.num_sink_tokens,
                self.self_attn_kv_shape,
                self.cross_attn_kv_shape,
                self.layer_idx,
                device=device,
                dtype=dtype,
            )
            self.cache_initialized = True
        return self.cache

    def new_forward(self, module: Attention, *args, **kwargs):
        original_hidden_states = self._metadata._get_parameter_from_args_kwargs("hidden_states", args, kwargs)
        current_cache = self.lazy_initialize_cache(original_hidden_states.device, original_hidden_states.dtype)
        kwargs["kv_cache"] = current_cache
        output = self.fn_ref.original_forward(*args, **kwargs)
        return output

    def reset_state(self, module):
        if self.cache_initialized:
            self.cache.reset()
        return module


class CacheLayer:
    def __init__(self, num_sink_tokens, self_attn_kv_shape, cross_attn_kv_shape, layer_idx, device, dtype) -> None:
        self.key_cache = torch.zeros(self_attn_kv_shape, device=device, dtype=dtype)
        self.value_cache = torch.zeros(self_attn_kv_shape, device=device, dtype=dtype)
        # self.cross_key_cache = torch.zeros(cross_attn_kv_shape, device=device, dtype=dtype)
        # self.cross_value_cache = torch.zeros(cross_attn_kv_shape, device=device, dtype=dtype)
        self.layer_idx = layer_idx
        self.local_start_index = 0
        self.num_sink_tokens = num_sink_tokens
        self.max_seq_length = self_attn_kv_shape[1]

    def reset(self):
        self.key_cache.zero_()
        self.value_cache.zero_()
        # self.cross_key_cache.zero_()
        # self.cross_value_cache.zero_()
        self.local_start_index = 0

    @torch.compiler.disable
    def update(
        self, key_states: torch.Tensor, value_states: torch.Tensor, rotary_emb: tuple[torch.Tensor, torch.Tensor]
    ) -> bool:
        num_new_tokens = key_states.shape[1]
        # Skip `sink_tokens` and `evicted_tokens`. Roll back cache by removing evicted tokens
        if num_new_tokens + self.local_start_index > self.max_seq_length:
            num_evicted_tokens = (num_new_tokens + self.local_start_index) - self.max_seq_length

            keys_to_keep = self.key_cache[:, self.num_sink_tokens + num_evicted_tokens :]
            keys_to_keep = self.rerotate_key_rotary_pos_emb(keys_to_keep, *rotary_emb, num_evicted_tokens)
            self.key_cache[:, self.num_sink_tokens : -num_evicted_tokens].copy_(keys_to_keep)

            values_to_keep = self.value_cache[:, self.num_sink_tokens + num_evicted_tokens :]
            self.value_cache[:, self.num_sink_tokens : -num_evicted_tokens].copy_(values_to_keep)
            self.local_start_index = self.local_start_index - num_evicted_tokens

        # Assign new keys/values directly up to current_end and update running cache positions
        end_index = self.local_start_index + key_states.shape[1]
        self.key_cache[:, self.local_start_index : end_index].copy_(key_states)
        self.value_cache[:, self.local_start_index : end_index].copy_(value_states)
        return self.key_cache[:, :end_index], self.value_cache[:, :end_index]

    def recompute_indices(self, num_new_tokens: int):
        self.local_start_index = self.local_start_index + num_new_tokens

    @staticmethod
    def _apply_rope(hidden_states, freqs_cos, freqs_sin):
        x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
        cos = freqs_cos[..., 0::2]
        sin = freqs_sin[..., 1::2]
        out = torch.empty_like(hidden_states)
        out[..., 0::2] = x1 * cos - x2 * sin
        out[..., 1::2] = x1 * sin + x2 * cos
        return out.type_as(hidden_states)

    def rerotate_key_rotary_pos_emb(
        self, key_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, num_evicted_tokens: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Compute the cos and sin required for back- and forward-rotating to `num_evicted_tokens` position earlier in the sequence
        original_cos = cos[
            :,
            self.num_sink_tokens + num_evicted_tokens : self.num_sink_tokens
            + num_evicted_tokens
            + key_states.shape[1],
        ]
        shifted_cos = cos[:, self.num_sink_tokens : self.num_sink_tokens + key_states.shape[1]]
        original_sin = sin[
            :,
            self.num_sink_tokens + num_evicted_tokens : self.num_sink_tokens
            + num_evicted_tokens
            + key_states.shape[1],
        ]
        shifted_sin = sin[:, self.num_sink_tokens : self.num_sink_tokens + key_states.shape[1]]
        rerotation_cos = original_cos * shifted_cos + original_sin * shifted_sin
        rerotation_sin = -original_sin * shifted_cos + original_cos * shifted_sin

        # rerotation_cos = cos[:, num_evicted_tokens]
        # rerotation_sin = -sin[:, num_evicted_tokens]
        rotated_key_states = self._apply_rope(key_states, rerotation_cos, rerotation_sin)
        return rotated_key_states


def apply_rolling_kv_cache(module: torch.nn.Module, config: RollingKVCacheConfig) -> None:
    for name, submodule in module.named_children():
        if name not in _ALL_TRANSFORMER_BLOCK_IDENTIFIERS or not isinstance(submodule, torch.nn.ModuleList):
            continue
        for i, block in enumerate(submodule):
            registry = HookRegistry.check_if_exists_or_initialize(block)
            hook = RollingKVCachekHook(
                batch_size=config.batch_size,
                max_seq_length=config.max_seq_length,
                num_sink_tokens=config.num_sink_tokens,
                frame_seq_length=config.frame_seq_length,
                num_layers=config.num_layers,
                local_attn_size=config.local_attn_size,
                layer_idx=i,
            )
            registry.register_hook(hook, ROLLING_KV_CACHE_HOOK)
