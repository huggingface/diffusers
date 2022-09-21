from typing import Tuple, Union

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict

from ..configuration_utils import ConfigMixin, flax_register_to_config
from ..modeling_flax_utils import FlaxModelMixin
from ..utils import BaseOutput
from .embeddings_flax import FlaxTimestepEmbedding, FlaxTimesteps
from .unet_blocks_flax import (
    FlaxCrossAttnDownBlock2D,
    FlaxCrossAttnUpBlock2D,
    FlaxDownBlock2D,
    FlaxUNetMidBlock2DCrossAttn,
    FlaxUpBlock2D,
)


@flax.struct.dataclass
class FlaxUNet2DConditionOutput(BaseOutput):
    """
    Args:
        sample (`jnp.ndarray` of shape `(batch_size, num_channels, height, width)`):
            Hidden states conditioned on `encoder_hidden_states` input. Output of last layer of model.
    """

    sample: jnp.ndarray


@flax_register_to_config
class FlaxUNet2DConditionModel(nn.Module, FlaxModelMixin, ConfigMixin):
    r"""
    FlaxUNet2DConditionModel is a conditional 2D UNet model that takes in a noisy sample, conditional state, and a
    timestep and returns sample shaped output.

    This model inherits from [`FlaxModelMixin`]. Check the superclass documentation for the generic methods the library
    implements for all the models (such as downloading or saving, etc.)

    Parameters:
        sample_size (`int`, *optional*): The size of the input sample.
        in_channels (`int`, *optional*, defaults to 4): The number of channels in the input sample.
        out_channels (`int`, *optional*, defaults to 4): The number of channels in the output.
        down_block_types (`Tuple[str]`, *optional*, defaults to `("CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D")`):
            The tuple of downsample blocks to use. The corresponding class names will be: "FlaxCrossAttnDownBlock2D",
            "FlaxCrossAttnDownBlock2D", "FlaxCrossAttnDownBlock2D", "FlaxDownBlock2D"
        up_block_types (`Tuple[str]`, *optional*, defaults to `("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D",)`):
            The tuple of upsample blocks to use. The corresponding class names will be: "FlaxUpBlock2D",
            "FlaxCrossAttnUpBlock2D", "FlaxCrossAttnUpBlock2D", "FlaxCrossAttnUpBlock2D"
        block_out_channels (`Tuple[int]`, *optional*, defaults to `(320, 640, 1280, 1280)`):
            The tuple of output channels for each block.
        layers_per_block (`int`, *optional*, defaults to 2): The number of layers per block.
        attention_head_dim (`int`, *optional*, defaults to 8): The dimension of the attention heads.
        cross_attention_dim (`int`, *optional*, defaults to 768): The dimension of the cross attention features.
        dropout (`float`, *optional*, defaults to 0): Dropout probability for down, up and bottleneck blocks.
    """

    sample_size: int = 32
    in_channels: int = 4
    out_channels: int = 4
    down_block_types: Tuple[str] = (
        "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D",
        "DownBlock2D",
    )
    up_block_types: Tuple[str] = ("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D")
    block_out_channels: Tuple[int] = (320, 640, 1280, 1280)
    layers_per_block: int = 2
    attention_head_dim: int = 8
    cross_attention_dim: int = 1280
    dropout: float = 0.0
    dtype: jnp.dtype = jnp.float32
    freq_shift: int = 0

    def init_weights(self, rng: jax.random.PRNGKey) -> FrozenDict:
        # init input tensors
        sample_shape = (1, self.in_channels, self.sample_size, self.sample_size)
        sample = jnp.zeros(sample_shape, dtype=jnp.float32)
        timesteps = jnp.ones((1,), dtype=jnp.int32)
        encoder_hidden_states = jnp.zeros((1, 1, self.cross_attention_dim), dtype=jnp.float32)

        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}

        return self.init(rngs, sample, timesteps, encoder_hidden_states)["params"]

    def setup(self):
        block_out_channels = self.block_out_channels
        time_embed_dim = block_out_channels[0] * 4

        # input
        self.conv_in = nn.Conv(
            block_out_channels[0],
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            dtype=self.dtype,
        )

        # time
        self.time_proj = FlaxTimesteps(block_out_channels[0], freq_shift=self.config.freq_shift)
        self.time_embedding = FlaxTimestepEmbedding(time_embed_dim, dtype=self.dtype)

        # down
        down_blocks = []
        output_channel = block_out_channels[0]
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
                    attn_num_head_channels=self.attention_head_dim,
                    add_downsample=not is_final_block,
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
        self.down_blocks = down_blocks

        # mid
        self.mid_block = FlaxUNetMidBlock2DCrossAttn(
            in_channels=block_out_channels[-1],
            dropout=self.dropout,
            attn_num_head_channels=self.attention_head_dim,
            dtype=self.dtype,
        )

        # up
        up_blocks = []
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(self.up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]

            is_final_block = i == len(block_out_channels) - 1

            if up_block_type == "CrossAttnUpBlock2D":
                up_block = FlaxCrossAttnUpBlock2D(
                    in_channels=input_channel,
                    out_channels=output_channel,
                    prev_output_channel=prev_output_channel,
                    num_layers=self.layers_per_block + 1,
                    attn_num_head_channels=self.attention_head_dim,
                    add_upsample=not is_final_block,
                    dropout=self.dropout,
                    dtype=self.dtype,
                )
            else:
                up_block = FlaxUpBlock2D(
                    in_channels=input_channel,
                    out_channels=output_channel,
                    prev_output_channel=prev_output_channel,
                    num_layers=self.layers_per_block + 1,
                    add_upsample=not is_final_block,
                    dropout=self.dropout,
                    dtype=self.dtype,
                )

            up_blocks.append(up_block)
            prev_output_channel = output_channel
        self.up_blocks = up_blocks

        # out
        self.conv_norm_out = nn.GroupNorm(num_groups=32, epsilon=1e-5)
        self.conv_out = nn.Conv(
            self.out_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            dtype=self.dtype,
        )

    def __call__(
        self,
        sample,
        timesteps,
        encoder_hidden_states,
        return_dict: bool = True,
        train: bool = False,
    ) -> Union[FlaxUNet2DConditionOutput, Tuple]:
        """r
        Args:
            sample (`jnp.ndarray`): (channel, height, width) noisy inputs tensor
            timestep (`jnp.ndarray` or `float` or `int`): timesteps
            encoder_hidden_states (`jnp.ndarray`): (channel, height, width) encoder hidden states
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`models.unet_2d_condition_flax.FlaxUNet2DConditionOutput`] instead of a
                plain tuple.
            train (`bool`, *optional*, defaults to `False`):
                Use deterministic functions and disable dropout when not training.

        Returns:
            [`~models.unet_2d_condition_flax.FlaxUNet2DConditionOutput`] or `tuple`:
            [`~models.unet_2d_condition_flax.FlaxUNet2DConditionOutput`] if `return_dict` is True, otherwise a `tuple`.
            When returning a tuple, the first element is the sample tensor.
        """
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

        # 5. up
        for up_block in self.up_blocks:
            res_samples = down_block_res_samples[-(self.layers_per_block + 1) :]
            down_block_res_samples = down_block_res_samples[: -(self.layers_per_block + 1)]
            if isinstance(up_block, FlaxCrossAttnUpBlock2D):
                sample = up_block(
                    sample,
                    temb=t_emb,
                    encoder_hidden_states=encoder_hidden_states,
                    res_hidden_states_tuple=res_samples,
                    deterministic=not train,
                )
            else:
                sample = up_block(sample, temb=t_emb, res_hidden_states_tuple=res_samples, deterministic=not train)

        # 6. post-process
        sample = self.conv_norm_out(sample)
        sample = nn.silu(sample)
        sample = self.conv_out(sample)
        sample = jnp.transpose(sample, (0, 3, 1, 2))

        if not return_dict:
            return (sample,)

        return FlaxUNet2DConditionOutput(sample=sample)
