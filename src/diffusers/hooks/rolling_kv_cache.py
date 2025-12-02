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
    ):
        self.batch_size = batch_size
        self.num_sink_tokens = num_sink_tokens
        self.num_layers = num_layers
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
                self.num_sink_tokens, self.self_attn_kv_shape, self.cross_attn_kv_shape, device=device, dtype=dtype
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
    def __init__(self, num_sink_tokens, self_attn_kv_shape, cross_attn_kv_shape, device, dtype) -> None:
        self.key_cache = torch.zeros(self_attn_kv_shape, device=device, dtype=dtype)
        self.value_cache = torch.zeros(self_attn_kv_shape, device=device, dtype=dtype)
        # self.cross_key_cache = torch.zeros(cross_attn_kv_shape, device=device, dtype=dtype)
        # self.cross_value_cache = torch.zeros(cross_attn_kv_shape, device=device, dtype=dtype)
        self.global_end_index = 0
        self.local_end_index = 0
        self.local_start_index = 0
        self.num_sink_tokens = num_sink_tokens
        self.max_seq_length = self_attn_kv_shape[1]

    def reset(self):
        self.key_cache.zero_()
        self.value_cache.zero_()
        # self.cross_key_cache.zero_()
        # self.cross_value_cache.zero_()
        self.global_end_index = 0
        self.local_end_index = 0
        self.local_start_index = 0

    @torch.compiler.disable
    def update(self, key_states: torch.Tensor, value_states: torch.Tensor) -> bool:
        # Assign new keys/values directly up to current_end
        start_idx, end_idx = self.maybe_roll_back(key_states.shape[1])
        self.key_cache[:, start_idx:end_idx] = key_states
        self.value_cache[:, start_idx:end_idx] = value_states
        # self.local_start_index += key_states.shape[1]
        return key_states, value_states

    @torch.compiler.disable
    def maybe_roll_back(self, num_new_tokens: int):
        if num_new_tokens + self.local_end_index > self.max_seq_length:
            num_evicted_tokens = (num_new_tokens + self.local_end_index) - self.max_seq_length
        else:
            num_evicted_tokens = 0

        # Skip `sink_tokens` and `evicted_tokens`. Roll back cache by removing the evicted tokens
        num_tokens_to_skip = self.num_sink_tokens + num_evicted_tokens
        # self.key_cache[:, self.num_sink_tokens : self.num_sink_tokens + num_tokens_to_skip] = self.key_cache[:, num_tokens_to_skip:].clone()
        # self.value_cache[:, self.num_sink_tokens : self.num_sink_tokens + num_tokens_to_skip] = self.value_cache[:, num_tokens_to_skip:].clone()
        self.key_cache.roll(-num_tokens_to_skip, dims=1)
        self.value_cache.roll(-num_tokens_to_skip, dims=1)

        self.local_start_index = self.local_start_index - num_evicted_tokens
        self.local_end_index = self.local_start_index + num_new_tokens
        return self.local_start_index, self.local_end_index


def apply_rolling_kv_cache(module: torch.nn.Module, config: RollingKVCacheConfig) -> None:
    for name, submodule in module.named_children():
        if name not in _ALL_TRANSFORMER_BLOCK_IDENTIFIERS or not isinstance(submodule, torch.nn.ModuleList):
            continue
        for block in submodule:
            registry = HookRegistry.check_if_exists_or_initialize(block)
            hook = RollingKVCachekHook(
                batch_size=config.batch_size,
                max_seq_length=config.max_seq_length,
                num_sink_tokens=config.num_sink_tokens,
                frame_seq_length=config.frame_seq_length,
                num_layers=config.num_layers,
                local_attn_size=config.local_attn_size,
            )
            registry.register_hook(hook, ROLLING_KV_CACHE_HOOK)
