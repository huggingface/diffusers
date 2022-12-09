# Copyright 2022 The HuggingFace Team. All rights reserved.
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

from ..utils.import_utils import is_xformers_available


if is_xformers_available():
    import xformers
    import xformers.ops
else:
    xformers = None


class CrossAttentionProcMixin:

    def __init__(self, head_size, upcast_attention):
        self.head_size = head_size
        self.upcast_attention = upcast_attention

    def __call__(self, hidden_states, query_proj, key_proj, value_proj, context=None):
        raise NotImplementedError("Make sure this method is overwritten in the subclass.")

    def batch_to_head_dim(self, tensor, head_size):
        batch_size, seq_len, dim = tensor.shape
        tensor = tensor.reshape(batch_size // self.head_size, self.head_size, seq_len, dim)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size // self.head_size, seq_len, dim * self.head_size)
        return tensor

    def head_to_batch_dim(self, tensor, head_size):
        batch_size, seq_len, dim = tensor.shape
        tensor = tensor.reshape(batch_size, seq_len, self.head_size, dim // self.head_size)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size * self.head_size, seq_len, dim // self.head_size)
        return tensor

    def get_attention_scores(self, query, key):
        dtype = query.dtype
        if self.upcast_attention:
            query = query.float()
            key = key.float()

        attention_scores = torch.baddbmm(
            torch.empty(query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device),
            query,
            key.transpose(-1, -2),
            beta=0,
            alpha=self.scale,
        )
        attention_probs = attention_scores.softmax(dim=-1)
        attention_probs = attention_probs.to(dtype)

        return attention_probs


class CrossAttentionProc(CrossAttentionProcMixin):

    def __call__(self, hidden_states, query_proj, key_proj, value_proj, context=None):
        batch_size, sequence_length, _ = hidden_states.shape
        query = query_proj(hidden_states)

        context = context if context is not None else hidden_states
        key = key_proj(context)
        value = self.value_proj(context)

        query = self.head_to_batch_dim(query, self.head_size)
        key = self.head_to_batch_dim(key, self.head_size)
        value = self.head_to_batch_dim(value, self.head_size)

        attention_probs = self.get_attention_scores(query, key)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = self.batch_to_head_dim(hidden_states)

        return hidden_states


class XFormersCrossAttentionProc(CrossAttentionProcMixin):

    def __call__(self, hidden_states, query_proj, key_proj, value_proj, context=None):
        batch_size, sequence_length, _ = hidden_states.shape
        query = query_proj(hidden_states)

        context = context if context is not None else hidden_states
        key = key_proj(context)
        value = self.value_proj(context)

        query = self.head_to_batch_dim(query, self.head_size).contiguous()
        key = self.head_to_batch_dim(key, self.head_size).contiguous()
        value = self.head_to_batch_dim(value, self.head_size).contiguous()

        hidden_states = xformers.ops.memory_efficient_attention(query, key, value, attn_bias=None)
        hidden_states = self.batch_to_head_dim(hidden_states)

        return hidden_states


class SlicedAttentionProc(CrossAttentionProcMixin):

    def __init__(self, head_size, upcast_attention, slice_size):
        super().__init__(head_size=head_size, upcast_attention=upcast_attention)

        self.slice_size = self.slice_size

    def __call__(self, hidden_states, query_proj, key_proj, value_proj, context=None):
        batch_size, sequence_length, _ = hidden_states.shape
        query = query_proj(hidden_states)

        dim = query.shape[-1]

        context = context if context is not None else hidden_states
        key = key_proj(context)
        value = self.value_proj(context)

        query = self.head_to_batch_dim(query, self.head_size)
        key = self.head_to_batch_dim(key, self.head_size)
        value = self.head_to_batch_dim(value, self.head_size)

        batch_size_attention = query.shape[0]
        hidden_states = torch.zeros(
            (batch_size_attention, sequence_length, dim // self.heads), device=query.device, dtype=query.dtype
        )

        for i in range(hidden_states.shape[0] // self.slice_size):
            start_idx = i * self.slice_size
            end_idx = (i + 1) * self.slice_size

            query_slice = query[start_idx:end_idx]
            key_slice = key[start_idx:end_idx]

            attn_slice = self.get_attention_scores(query_slice, key_slice)

            attn_slice = torch.bmm(attn_slice, value[start_idx:end_idx])

            hidden_states[start_idx:end_idx] = attn_slice

        hidden_states = self.batch_to_head_dim(hidden_states)

        return hidden_states
