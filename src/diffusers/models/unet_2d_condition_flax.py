from typing import Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
# from flax.core.frozen_dict import FrozenDict

from ..configuration_utils import ConfigMixin, flax_register_to_config
from ..modeling_flax_utils import FlaxModelMixin
from .embeddings_flax import FlaxTimestepEmbedding, FlaxTimesteps
from .unet_blocks_flax import (
    FlaxDownBlock2D,
    FlaxCrossAttnDownBlock2D,
    FlaxUNetMidBlock2DCrossAttn,
    FlaxUpBlock2D,
    FlaxCrossAttnUpBlock2D,
)


@flax_register_to_config
class FlaxUNet2DConditionModel(nn.Module, FlaxModelMixin, ConfigMixin):
    sample_size=32
    in_channels=4
    out_channels=4
    down_block_types=("CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D")
    up_block_types=("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D")
    block_out_channels=(224, 448, 672, 896)
    layers_per_block=2
    attention_head_dim=8
    cross_attention_dim=768
    dropout=0.1
    dtype: jnp.dtype = jnp.float32

    # model args
    # input_shape: Tuple = (1, 32, 32, 4)
    # seed: int = 0

    # # Note: input_shape is ignored
    # def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple) -> FrozenDict:
    #     # init input tensors
    #     sample_shape = (1, self.module.sample_size, self.module.sample_size, self.module.in_channels)
    #     sample = jnp.zeros(sample_shape, dtype=jnp.float32)
    #     timesteps = jnp.ones((1,), dtype=jnp.int32)
    #     encoder_hidden_states = jnp.zeros((1, 1, self.module.cross_attention_dim), dtype=jnp.float32)

    #     params_rng, dropout_rng = jax.random.split(rng)
    #     rngs = {"params": params_rng, "dropout": dropout_rng}

    #     return self.module.init(rngs, sample, timesteps, encoder_hidden_states)["params"]


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
        self.time_proj = FlaxTimesteps(block_out_channels[0])
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
        # params: dict = None,
        # dropout_rng: jax.random.PRNGKey = None,
        train: bool = False,
    ):
        # Handle any PRNG if needed
        # rngs = {"dropout": dropout_rng} if dropout_rng is not None else {}

        # 1. time
        # broadcast to batch dimension
        # timesteps = jnp.broadcast_to(timesteps, (sample.shape[0],) + timesteps.shape)
        t_emb = self.time_proj(timesteps)
        t_emb = self.time_embedding(t_emb)

        # 2. pre-process
        sample = self.conv_in(sample)

        # 3. down
        down_block_res_samples = (sample,)
        for down_block in self.down_blocks:
            if isinstance(down_block, FlaxCrossAttnDownBlock2D):
                sample, res_samples = down_block(sample, t_emb, encoder_hidden_states)
            else:
                sample, res_samples = down_block(sample, t_emb)
            down_block_res_samples += res_samples

        # 4. mid
        sample = self.mid_block(sample, t_emb, encoder_hidden_states)

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
                )
            else:
                sample = up_block(sample, temb=t_emb, res_hidden_states_tuple=res_samples)

        # 6. post-process
        sample = self.conv_norm_out(sample)
        sample = nn.silu(sample)
        sample = self.conv_out(sample)

        return sample
