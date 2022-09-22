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
import math

import flax.linen as nn
import jax.numpy as jnp


# This is like models.embeddings.get_timestep_embedding (PyTorch) but
# less general (only handles the case we currently need).
def get_sinusoidal_embeddings(timesteps, embedding_dim, freq_shift: float = 1):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models: Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D tensor of N indices, one per batch element.
                      These may be fractional.
    :param embedding_dim: the dimension of the output. :param max_period: controls the minimum frequency of the
    embeddings. :return: an [N x dim] tensor of positional embeddings.
    """
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - freq_shift)
    emb = jnp.exp(jnp.arange(half_dim) * -emb)
    emb = timesteps[:, None] * emb[None, :]
    emb = jnp.concatenate([jnp.cos(emb), jnp.sin(emb)], -1)
    return emb


class FlaxTimestepEmbedding(nn.Module):
    time_embed_dim: int = 32
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, temb):
        temb = nn.Dense(self.time_embed_dim, dtype=self.dtype, name="linear_1")(temb)
        temb = nn.silu(temb)
        temb = nn.Dense(self.time_embed_dim, dtype=self.dtype, name="linear_2")(temb)
        return temb


class FlaxTimesteps(nn.Module):
    dim: int = 32
    freq_shift: float = 1

    @nn.compact
    def __call__(self, timesteps):
        return get_sinusoidal_embeddings(timesteps, self.dim, freq_shift=self.freq_shift)
