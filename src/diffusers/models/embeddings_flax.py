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
import math

import flax.linen as nn
import jax.numpy as jnp

from ..utils import logging


logger = logging.get_logger(__name__)


def get_sinusoidal_embeddings(
    timesteps: jnp.ndarray,
    embedding_dim: int,
    freq_shift: float = 1,
    min_timescale: float = 1,
    max_timescale: float = 1.0e4,
    flip_sin_to_cos: bool = False,
    scale: float = 1.0,
) -> jnp.ndarray:
    """Returns the positional encoding (same as Tensor2Tensor).

    Args:
        timesteps (`jnp.ndarray` of shape `(N,)`):
            A 1-D array of N indices, one per batch element. These may be fractional.
        embedding_dim (`int`):
            The number of output channels.
        freq_shift (`float`, *optional*, defaults to `1`):
            Shift applied to the frequency scaling of the embeddings.
        min_timescale (`float`, *optional*, defaults to `1`):
            The smallest time unit used in the sinusoidal calculation (should probably be 0.0).
        max_timescale (`float`, *optional*, defaults to `1.0e4`):
            The largest time unit used in the sinusoidal calculation.
        flip_sin_to_cos (`bool`, *optional*, defaults to `False`):
            Whether to flip the order of sinusoidal components to cosine first.
        scale (`float`, *optional*, defaults to `1.0`):
            A scaling factor applied to the positional embeddings.

    Returns:
        a Tensor of timing signals [N, num_channels]
    """
    assert timesteps.ndim == 1, "Timesteps should be a 1d-array"
    assert embedding_dim % 2 == 0, f"Embedding dimension {embedding_dim} should be even"
    num_timescales = float(embedding_dim // 2)
    log_timescale_increment = math.log(max_timescale / min_timescale) / (num_timescales - freq_shift)
    inv_timescales = min_timescale * jnp.exp(jnp.arange(num_timescales, dtype=jnp.float32) * -log_timescale_increment)
    emb = jnp.expand_dims(timesteps, 1) * jnp.expand_dims(inv_timescales, 0)

    # scale embeddings
    scaled_time = scale * emb

    if flip_sin_to_cos:
        signal = jnp.concatenate([jnp.cos(scaled_time), jnp.sin(scaled_time)], axis=1)
    else:
        signal = jnp.concatenate([jnp.sin(scaled_time), jnp.cos(scaled_time)], axis=1)
    signal = jnp.reshape(signal, [jnp.shape(timesteps)[0], embedding_dim])
    return signal


class FlaxTimestepEmbedding(nn.Module):
    r"""
    Time step Embedding Module. Learns embeddings for input time steps.

    Args:
        time_embed_dim (`int`, *optional*, defaults to `32`):
            Time step embedding dimension.
        dtype (`jnp.dtype`, *optional*, defaults to `jnp.float32`):
            The data type for the embedding parameters.
    """

    logger.warning(
        "Flax classes are deprecated and will be removed in Diffusers v1.0.0. We "
        "recommend migrating to PyTorch classes or pinning your version of Diffusers."
    )

    time_embed_dim: int = 32
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, temb):
        temb = nn.Dense(self.time_embed_dim, dtype=self.dtype, name="linear_1")(temb)
        temb = nn.silu(temb)
        temb = nn.Dense(self.time_embed_dim, dtype=self.dtype, name="linear_2")(temb)
        return temb


class FlaxTimesteps(nn.Module):
    r"""
    Wrapper Module for sinusoidal Time step Embeddings as described in https://huggingface.co/papers/2006.11239

    Args:
        dim (`int`, *optional*, defaults to `32`):
            Time step embedding dimension.
        flip_sin_to_cos (`bool`, *optional*, defaults to `False`):
            Whether to flip the sinusoidal function from sine to cosine.
        freq_shift (`float`, *optional*, defaults to `1`):
            Frequency shift applied to the sinusoidal embeddings.
    """

    dim: int = 32
    flip_sin_to_cos: bool = False
    freq_shift: float = 1

    logger.warning(
        "Flax classes are deprecated and will be removed in Diffusers v1.0.0. We "
        "recommend migrating to PyTorch classes or pinning your version of Diffusers."
    )

    @nn.compact
    def __call__(self, timesteps):
        return get_sinusoidal_embeddings(
            timesteps, embedding_dim=self.dim, flip_sin_to_cos=self.flip_sin_to_cos, freq_shift=self.freq_shift
        )
