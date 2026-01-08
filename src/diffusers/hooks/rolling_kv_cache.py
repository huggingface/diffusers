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

import torch


class RollingKVCache:
    def __init__(
        self,
        module,
        batch_size: int,
        max_seq_length: int,
        num_sink_tokens: int,
    ):
        encoder_dim = 0
        self_attn_kv_shape = [
            batch_size,
            max_seq_length,
            module.config.num_attention_heads,
            module.config.attention_head_dim,
        ]
        cross_attn_kv_shape = [
            batch_size,
            encoder_dim,
            module.config.num_attention_heads,
            module.config.attention_head_dim,
        ]

        self.cache_layers = []
        for layer_idx in range(module.config.num_layers):
            self.cache_layers.append(
                CacheLayer(
                    num_sink_tokens,
                    self_attn_kv_shape,
                    cross_attn_kv_shape,
                    layer_idx=layer_idx,
                )
            )

    def __getitem__(self, layer_idx: int):
        return self.cache_layers[layer_idx]

    def recompute_indices(self, num_new_tokens: int):
        for cache in self.cache_layers:
            cache.local_start_index = cache.local_start_index + num_new_tokens

    def reset_state(self):
        for cache in self.cache_layers:
            cache.reset()


class CacheLayer:
    def __init__(self, num_sink_tokens, self_attn_kv_shape, cross_attn_kv_shape, layer_idx) -> None:
        self.self_attn_kv_shape = self_attn_kv_shape
        self.cross_attn_kv_shape = cross_attn_kv_shape
        self.layer_idx = layer_idx
        self.local_start_index = 0
        self.num_sink_tokens = num_sink_tokens
        self.max_seq_length = self_attn_kv_shape[1]
        self.cache_initialized = False

    def lazy_initialize(self, device: str, dtype: torch.dtype):
        if not self.cache_initialized:
            self.key_cache = torch.zeros(self.self_attn_kv_shape, device=device, dtype=dtype)
            self.value_cache = torch.zeros(self.self_attn_kv_shape, device=device, dtype=dtype)
            self.cross_key_cache = torch.zeros(self.cross_attn_kv_shape, device=device, dtype=dtype)
            self.cross_value_cache = torch.zeros(self.cross_attn_kv_shape, device=device, dtype=dtype)
            self.cache_initialized = True

    def reset(self):
        self.key_cache.zero_()
        self.value_cache.zero_()
        self.cross_key_cache.zero_()
        self.cross_value_cache.zero_()
        self.local_start_index = 0

    @torch.compiler.disable
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor],
        is_cross_attn: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.lazy_initialize(device=key_states.device, dtype=key_states.dtype)
        num_new_tokens = key_states.shape[1]

        if is_cross_attn:
            self.cross_key_cache = key_states
            self.cross_value_cache = value_states
            return self.cross_key_cache, self.cross_value_cache

        # Skip `sink_tokens` and `evicted_tokens`. Roll back cache by removing evicted tokens
        if num_new_tokens + self.local_start_index > self.max_seq_length:
            num_evicted_tokens = (num_new_tokens + self.local_start_index) - self.max_seq_length

            keys_to_keep = self.key_cache[:, self.num_sink_tokens + num_evicted_tokens :]
            keys_to_keep = self.rerotate_key_rotary_pos_emb(keys_to_keep, *rotary_emb, num_evicted_tokens)
            self.key_cache[:, self.num_sink_tokens : -num_evicted_tokens] = keys_to_keep

            values_to_keep = self.value_cache[:, self.num_sink_tokens + num_evicted_tokens :]
            self.value_cache[:, self.num_sink_tokens : -num_evicted_tokens] = values_to_keep.clone()
            self.local_start_index = self.local_start_index - num_evicted_tokens

        # Assign new keys/values directly up to current_end and update running cache positions
        end_index = self.local_start_index + key_states.shape[1]
        self.key_cache[:, self.local_start_index : end_index] = key_states
        self.value_cache[:, self.local_start_index : end_index] = value_states
        return self.key_cache[:, :end_index], self.value_cache[:, :end_index]

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
