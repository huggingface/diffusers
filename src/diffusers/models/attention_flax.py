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

import flax.linen as nn
import jax.numpy as jnp


class FlaxAttentionBlock(nn.Module):
    query_dim: int
    heads: int = 8
    dim_head: int = 64
    dropout: float = 0.0
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        inner_dim = self.dim_head * self.heads
        self.scale = self.dim_head**-0.5

        self.to_q = nn.Dense(inner_dim, use_bias=False, dtype=self.dtype)
        self.to_k = nn.Dense(inner_dim, use_bias=False, dtype=self.dtype)
        self.to_v = nn.Dense(inner_dim, use_bias=False, dtype=self.dtype)

        self.to_out = nn.Dense(self.query_dim, dtype=self.dtype)

    def reshape_heads_to_batch_dim(self, tensor):
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(batch_size, seq_len, head_size, dim // head_size)
        tensor = jnp.transpose(tensor, (0, 2, 1, 3))
        tensor = tensor.reshape(batch_size * head_size, seq_len, dim // head_size)
        return tensor

    def reshape_batch_dim_to_heads(self, tensor):
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
        tensor = jnp.transpose(tensor, (0, 2, 1, 3))
        tensor = tensor.reshape(batch_size // head_size, seq_len, dim * head_size)
        return tensor

    def __call__(self, hidden_states, context=None, deterministic=True):
        context = hidden_states if context is None else context

        q = self.to_q(hidden_states)
        k = self.to_k(context)
        v = self.to_v(context)

        q = self.reshape_heads_to_batch_dim(q)
        k = self.reshape_heads_to_batch_dim(k)
        v = self.reshape_heads_to_batch_dim(v)

        # compute attentions
        attn_weights = jnp.einsum("b i d, b j d->b i j", q, k)
        attn_weights = attn_weights * self.scale
        attn_weights = nn.softmax(attn_weights, axis=2)

        # attend to values
        hidden_states = jnp.einsum("b i j, b j d -> b i d", attn_weights, v)
        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
        hidden_states = self.to_out(hidden_states)
        return hidden_states


class FlaxBasicTransformerBlock(nn.Module):
    dim: int
    n_heads: int
    d_head: int
    dropout: float = 0.0
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        # self attention
        self.self_attn = FlaxAttentionBlock(self.dim, self.n_heads, self.d_head, self.dropout, dtype=self.dtype)
        # cross attention
        self.cross_attn = FlaxAttentionBlock(self.dim, self.n_heads, self.d_head, self.dropout, dtype=self.dtype)
        self.ff = FlaxGluFeedForward(dim=self.dim, dropout=self.dropout, dtype=self.dtype)
        self.norm1 = nn.LayerNorm(epsilon=1e-5, dtype=self.dtype)
        self.norm2 = nn.LayerNorm(epsilon=1e-5, dtype=self.dtype)
        self.norm3 = nn.LayerNorm(epsilon=1e-5, dtype=self.dtype)

    def __call__(self, hidden_states, context, deterministic=True):
        # self attention
        residual = hidden_states
        hidden_states = self.self_attn(self.norm1(hidden_states))
        hidden_states = hidden_states + residual

        # cross attention
        residual = hidden_states
        hidden_states = self.cross_attn(self.norm2(hidden_states), context)
        hidden_states = hidden_states + residual

        # feed forward
        residual = hidden_states
        hidden_states = self.ff(self.norm3(hidden_states))
        hidden_states = hidden_states + residual

        return hidden_states


class FlaxSpatialTransformer(nn.Module):
    in_channels: int
    n_heads: int
    d_head: int
    depth: int = 1
    dropout: float = 0.0
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.norm = nn.GroupNorm(num_groups=32, epsilon=1e-5)

        inner_dim = self.n_heads * self.d_head
        self.proj_in = nn.Conv(
            inner_dim,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="VALID",
            dtype=self.dtype,
        )

        self.transformer_blocks = [
            FlaxBasicTransformerBlock(inner_dim, self.n_heads, self.d_head, dropout=self.dropout, dtype=self.dtype)
            for _ in range(self.depth)
        ]

        self.proj_out = nn.Conv(
            inner_dim,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="VALID",
            dtype=self.dtype,
        )

    def __call__(self, hidden_states, context, deterministic=True):
        batch, height, width, channels = hidden_states.shape
        # import ipdb; ipdb.set_trace()
        residual = hidden_states
        hidden_states = self.norm(hidden_states)
        hidden_states = self.proj_in(hidden_states)

        # hidden_states = jnp.transpose(hidden_states, (0, 2, 3, 1))
        hidden_states = hidden_states.reshape(batch, height * width, channels)

        for transformer_block in self.transformer_blocks:
            hidden_states = transformer_block(hidden_states, context)

        hidden_states = hidden_states.reshape(batch, height, width, channels)
        # hidden_states = jnp.transpose(hidden_states, (0, 3, 1, 2))

        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states + residual

        return hidden_states


class FlaxGluFeedForward(nn.Module):
    dim: int
    dropout: float = 0.0
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        inner_dim = self.dim * 4
        self.dense1 = nn.Dense(inner_dim * 2, dtype=self.dtype)
        self.dense2 = nn.Dense(self.dim, dtype=self.dtype)

    def __call__(self, hidden_states, deterministic=True):
        hidden_states = self.dense1(hidden_states)
        hidden_linear, hidden_gelu = jnp.split(hidden_states, 2, axis=2)
        hidden_states = hidden_linear * nn.gelu(hidden_gelu)
        hidden_states = self.dense2(hidden_states)
        return hidden_states
