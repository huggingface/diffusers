# Copyright 2024 The HuggingFace Team. All rights reserved.
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
from typing import Optional, Tuple, Union

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict

from ..configuration_utils import ConfigMixin, flax_register_to_config
from ..utils import BaseOutput
from .embeddings_flax import FlaxTimestepEmbedding, FlaxTimesteps
from .modeling_flax_utils import FlaxModelMixin
from .unets.unet_2d_blocks_flax import (
    FlaxCrossAttnDownBlock2D,
    FlaxDownBlock2D,
    FlaxUNetMidBlock2DCrossAttn,
)


@flax.struct.dataclass
class FlaxControlNetOutput(BaseOutput):
    """
    The output of [`FlaxControlNetModel`].

    Args:
        down_block_res_samples (`jnp.ndarray`):
        mid_block_res_sample (`jnp.ndarray`):
    """

    down_block_res_samples: jnp.ndarray
    mid_block_res_sample: jnp.ndarray


class FlaxControlNetConditioningEmbedding(nn.Module):
    conditioning_embedding_channels: int
    block_out_channels: Tuple[int, ...] = (16, 32, 96, 256)
    dtype: jnp.dtype = jnp.float32

    def setup(self) -> None:
        self.conv_in = nn.Conv(
            self.block_out_channels[0],
            kernel_size=(3, 3),
            padding=((1, 1), (1, 1)),
            dtype=self.dtype,
        )

        blocks = []
        for i in range(len(self.block_out_channels) - 1):
            channel_in = self.block_out_channels[i]
            channel_out = self.block_out_channels[i + 1]
            conv1 = nn.Conv(
                channel_in,
                kernel_size=(3, 3),
                padding=((1, 1), (1, 1)),
                dtype=self.dtype,
            )
            blocks.append(conv1)
            conv2 = nn.Conv(
                channel_out,
                kernel_size=(3, 3),
                strides=(2, 2),
                padding=((1, 1), (1, 1)),
                dtype=self.dtype,
            )
            blocks.append(conv2)
        self.blocks = blocks

        self.conv_out = nn.Conv(
            self.conditioning_embedding_channels,
            kernel_size=(3, 3),
            padding=((1, 1), (1, 1)),
            kernel_init=nn.initializers.zeros_init(),
            bias_init=nn.initializers.zeros_init(),
            dtype=self.dtype,
        )

    def __call__(self, conditioning: jnp.ndarray) -> jnp.ndarray:
        embedding = self.conv_in(conditioning)
        embedding = nn.silu(embedding)

        for block in self.blocks:
            embedding = block(embedding)
            embedding = nn.silu(embedding)

        embedding = self.conv_out(embedding)

        return embedding


@flax_register_to_config
class FlaxControlNetModel(nn.Module, FlaxModelMixin, ConfigMixin):
    r"""
    A ControlNet model.

    This model inherits from [`FlaxModelMixin`]. Check the superclass documentation for it’s generic methods
    implemented for all models (such as downloading or saving).

    This model is also a Flax Linen [`flax.linen.Module`](https://flax.readthedocs.io/en/latest/flax.linen.html#module)
    subclass. Use it as a regular Flax Linen module and refer to the Flax documentation for all matters related to its
    general usage and behavior.

    Inherent JAX features such as the following are supported:

    - [Just-In-Time (JIT) compilation](https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit)
    - [Automatic Differentiation](https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation)
    - [Vectorization](https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap)
    - [Parallelization](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap)

    Parameters:
        sample_size (`int`, *optional*):
            The size of the input sample.
        in_channels (`int`, *optional*, defaults to 4):
            The number of channels in the input sample.
        down_block_types (`Tuple[str]`, *optional*, defaults to `("FlaxCrossAttnDownBlock2D", "FlaxCrossAttnDownBlock2D", "FlaxCrossAttnDownBlock2D", "FlaxDownBlock2D")`):
            The tuple of downsample blocks to use.
        block_out_channels (`Tuple[int]`, *optional*, defaults to `(320, 640, 1280, 1280)`):
            The tuple of output channels for each block.
        layers_per_block (`int`, *optional*, defaults to 2):
            The number of layers per block.
        attention_head_dim (`int` or `Tuple[int]`, *optional*, defaults to 8):
            The dimension of the attention heads.
        num_attention_heads (`int` or `Tuple[int]`, *optional*):
            The number of attention heads.
        cross_attention_dim (`int`, *optional*, defaults to 768):
            The dimension of the cross attention features.
        dropout (`float`, *optional*, defaults to 0):
            Dropout probability for down, up and bottleneck blocks.
        flip_sin_to_cos (`bool`, *optional*, defaults to `True`):
            Whether to flip the sin to cos in the time embedding.
        freq_shift (`int`, *optional*, defaults to 0): The frequency shift to apply to the time embedding.
        controlnet_conditioning_channel_order (`str`, *optional*, defaults to `rgb`):
            The channel order of conditional image. Will convert to `rgb` if it's `bgr`.
        conditioning_embedding_out_channels (`tuple`, *optional*, defaults to `(16, 32, 96, 256)`):
            The tuple of output channel for each block in the `conditioning_embedding` layer.
    """

    sample_size: int = 32
    in_channels: int = 4
    down_block_types: Tuple[str, ...] = (
        "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D",
        "DownBlock2D",
    )
    only_cross_attention: Union[bool, Tuple[bool, ...]] = False
    block_out_channels: Tuple[int, ...] = (320, 640, 1280, 1280)
    layers_per_block: int = 2
    attention_head_dim: Union[int, Tuple[int, ...]] = 8
    num_attention_heads: Optional[Union[int, Tuple[int, ...]]] = None
    cross_attention_dim: int = 1280
    dropout: float = 0.0
    use_linear_projection: bool = False
    dtype: jnp.dtype = jnp.float32
    flip_sin_to_cos: bool = True
    freq_shift: int = 0
    controlnet_conditioning_channel_order: str = "rgb"
    conditioning_embedding_out_channels: Tuple[int, ...] = (16, 32, 96, 256)

    def init_weights(self, rng: jax.Array) -> FrozenDict:
        # init input tensors
        sample_shape = (1, self.in_channels, self.sample_size, self.sample_size)
        sample = jnp.zeros(sample_shape, dtype=jnp.float32)
        timesteps = jnp.ones((1,), dtype=jnp.int32)
        encoder_hidden_states = jnp.zeros((1, 1, self.cross_attention_dim), dtype=jnp.float32)
        controlnet_cond_shape = (1, 3, self.sample_size * 8, self.sample_size * 8)
        controlnet_cond = jnp.zeros(controlnet_cond_shape, dtype=jnp.float32)

        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}

        return self.init(rngs, sample, timesteps, encoder_hidden_states, controlnet_cond)["params"]

    def setup(self) -> None:
        block_out_channels = self.block_out_channels
        time_embed_dim = block_out_channels[0] * 4

        # If `num_attention_heads` is not defined (which is the case for most models)
        # it will default to `attention_head_dim`. This looks weird upon first reading it and it is.
        # The reason for this behavior is to correct for incorrectly named variables that were introduced
        # when this library was created. The incorrect naming was only discovered much later in https://github.com/huggingface/diffusers/issues/2011#issuecomment-1547958131
        # Changing `attention_head_dim` to `num_attention_heads` for 40,000+ configurations is too backwards breaking
        # which is why we correct for the naming here.
        num_attention_heads = self.num_attention_heads or self.attention_head_dim

        # input
        self.conv_in = nn.Conv(
            block_out_channels[0],
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            dtype=self.dtype,
        )

        # time
        self.time_proj = FlaxTimesteps(
            block_out_channels[0], flip_sin_to_cos=self.flip_sin_to_cos, freq_shift=self.config.freq_shift
        )
        self.time_embedding = FlaxTimestepEmbedding(time_embed_dim, dtype=self.dtype)

        self.controlnet_cond_embedding = FlaxControlNetConditioningEmbedding(
            conditioning_embedding_channels=block_out_channels[0],
            block_out_channels=self.conditioning_embedding_out_channels,
        )

        only_cross_attention = self.only_cross_attention
        if isinstance(only_cross_attention, bool):
            only_cross_attention = (only_cross_attention,) * len(self.down_block_types)

        if isinstance(num_attention_heads, int):
            num_attention_heads = (num_attention_heads,) * len(self.down_block_types)

        # down
        down_blocks = []
        controlnet_down_blocks = []

        output_channel = block_out_channels[0]

        controlnet_block = nn.Conv(
            output_channel,
            kernel_size=(1, 1),
            padding="VALID",
            kernel_init=nn.initializers.zeros_init(),
            bias_init=nn.initializers.zeros_init(),
            dtype=self.dtype,
        )
        controlnet_down_blocks.append(controlnet_block)

        for i, down_block_type in enumerate(self.down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            if down_block_type == "CrossAttnDownBlock2D":
                down_block = FlaxCrossAttnDownBlock2D(
                    in_channels=input_channel,
                    out_channels=output_channel,
                    dropout=self.dropout,
                    num_layers=self.layers_per_block,
                    num_attention_heads=num_attention_heads[i],
                    add_downsample=not is_final_block,
                    use_linear_projection=self.use_linear_projection,
                    only_cross_attention=only_cross_attention[i],
                    dtype=self.dtype,
                )
            else:
                down_block = FlaxDownBlock2D(
                    in_channels=input_channel,
                    out_channels=output_channel,
                    dropout=self.dropout,
                    num_layers=self.layers_per_block,
                    add_downsample=not is_final_block,
                    dtype=self.dtype,
                )

            down_blocks.append(down_block)

            for _ in range(self.layers_per_block):
                controlnet_block = nn.Conv(
                    output_channel,
                    kernel_size=(1, 1),
                    padding="VALID",
                    kernel_init=nn.initializers.zeros_init(),
                    bias_init=nn.initializers.zeros_init(),
                    dtype=self.dtype,
                )
                controlnet_down_blocks.append(controlnet_block)

            if not is_final_block:
                controlnet_block = nn.Conv(
                    output_channel,
                    kernel_size=(1, 1),
                    padding="VALID",
                    kernel_init=nn.initializers.zeros_init(),
                    bias_init=nn.initializers.zeros_init(),
                    dtype=self.dtype,
                )
                controlnet_down_blocks.append(controlnet_block)

        self.down_blocks = down_blocks
        self.controlnet_down_blocks = controlnet_down_blocks

        # mid
        mid_block_channel = block_out_channels[-1]
        self.mid_block = FlaxUNetMidBlock2DCrossAttn(
            in_channels=mid_block_channel,
            dropout=self.dropout,
            num_attention_heads=num_attention_heads[-1],
            use_linear_projection=self.use_linear_projection,
            dtype=self.dtype,
        )

        self.controlnet_mid_block = nn.Conv(
            mid_block_channel,
            kernel_size=(1, 1),
            padding="VALID",
            kernel_init=nn.initializers.zeros_init(),
            bias_init=nn.initializers.zeros_init(),
            dtype=self.dtype,
        )

    def __call__(
        self,
        sample: jnp.ndarray,
        timesteps: Union[jnp.ndarray, float, int],
        encoder_hidden_states: jnp.ndarray,
        controlnet_cond: jnp.ndarray,
        conditioning_scale: float = 1.0,
        return_dict: bool = True,
        train: bool = False,
    ) -> Union[FlaxControlNetOutput, Tuple[Tuple[jnp.ndarray, ...], jnp.ndarray]]:
        r"""
        Args:
            sample (`jnp.ndarray`): (batch, channel, height, width) noisy inputs tensor
            timestep (`jnp.ndarray` or `float` or `int`): timesteps
            encoder_hidden_states (`jnp.ndarray`): (batch_size, sequence_length, hidden_size) encoder hidden states
            controlnet_cond (`jnp.ndarray`): (batch, channel, height, width) the conditional input tensor
            conditioning_scale (`float`, *optional*, defaults to `1.0`): the scale factor for controlnet outputs
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`models.unets.unet_2d_condition_flax.FlaxUNet2DConditionOutput`] instead of
                a plain tuple.
            train (`bool`, *optional*, defaults to `False`):
                Use deterministic functions and disable dropout when not training.

        Returns:
            [`~models.unets.unet_2d_condition_flax.FlaxUNet2DConditionOutput`] or `tuple`:
                [`~models.unets.unet_2d_condition_flax.FlaxUNet2DConditionOutput`] if `return_dict` is True, otherwise
                a `tuple`. When returning a tuple, the first element is the sample tensor.
        """
        channel_order = self.controlnet_conditioning_channel_order
        if channel_order == "bgr":
            controlnet_cond = jnp.flip(controlnet_cond, axis=1)

        # 1. time
        if not isinstance(timesteps, jnp.ndarray):
            timesteps = jnp.array([timesteps], dtype=jnp.int32)
        elif isinstance(timesteps, jnp.ndarray) and len(timesteps.shape) == 0:
            timesteps = timesteps.astype(dtype=jnp.float32)
            timesteps = jnp.expand_dims(timesteps, 0)

        t_emb = self.time_proj(timesteps)
        t_emb = self.time_embedding(t_emb)

        # 2. pre-process
        sample = jnp.transpose(sample, (0, 2, 3, 1))
        sample = self.conv_in(sample)

        controlnet_cond = jnp.transpose(controlnet_cond, (0, 2, 3, 1))
        controlnet_cond = self.controlnet_cond_embedding(controlnet_cond)
        sample += controlnet_cond

        # 3. down
        down_block_res_samples = (sample,)
        for down_block in self.down_blocks:
            if isinstance(down_block, FlaxCrossAttnDownBlock2D):
                sample, res_samples = down_block(sample, t_emb, encoder_hidden_states, deterministic=not train)
            else:
                sample, res_samples = down_block(sample, t_emb, deterministic=not train)
            down_block_res_samples += res_samples

        # 4. mid
        sample = self.mid_block(sample, t_emb, encoder_hidden_states, deterministic=not train)

        # 5. contronet blocks
        controlnet_down_block_res_samples = ()
        for down_block_res_sample, controlnet_block in zip(down_block_res_samples, self.controlnet_down_blocks):
            down_block_res_sample = controlnet_block(down_block_res_sample)
            controlnet_down_block_res_samples += (down_block_res_sample,)

        down_block_res_samples = controlnet_down_block_res_samples

        mid_block_res_sample = self.controlnet_mid_block(sample)

        # 6. scaling
        down_block_res_samples = [sample * conditioning_scale for sample in down_block_res_samples]
        mid_block_res_sample *= conditioning_scale

        if not return_dict:
            return (down_block_res_samples, mid_block_res_sample)

        return FlaxControlNetOutput(
            down_block_res_samples=down_block_res_samples, mid_block_res_sample=mid_block_res_sample
        )
