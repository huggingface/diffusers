# JAX implementation of VQGAN from taming-transformers https://github.com/CompVis/taming-transformers

import math
from functools import partial
from typing import Tuple

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict

from ..configuration_utils import ConfigMixin, flax_register_to_config
from ..modeling_flax_utils import FlaxModelMixin
from ..utils import BaseOutput


@flax.struct.dataclass
class FlaxDecoderOutput(BaseOutput):
    """
    Output of decoding method.

    Args:
        sample (`jnp.ndarray` of shape `(batch_size, num_channels, height, width)`):
            Decoded output sample of the model. Output of the last layer of the model.
    """

    sample: jnp.ndarray


@flax.struct.dataclass
class FlaxAutoencoderKLOutput(BaseOutput):
    """
    Output of AutoencoderKL encoding method.

    Args:
        latent_dist (`DiagonalGaussianDistribution`):
            Encoded outputs of `Encoder` represented as the mean and logvar of `DiagonalGaussianDistribution`.
            `DiagonalGaussianDistribution` allows for sampling latents from the distribution.
    """

    latent_dist: "DiagonalGaussianDistribution"


class Upsample2D(nn.Module):
    in_channels: int
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.conv = nn.Conv(
            self.in_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            dtype=self.dtype,
        )

    def __call__(self, hidden_states):
        batch, height, width, channels = hidden_states.shape
        hidden_states = jax.image.resize(
            hidden_states,
            shape=(batch, height * 2, width * 2, channels),
            method="nearest",
        )
        hidden_states = self.conv(hidden_states)
        return hidden_states


class Downsample2D(nn.Module):
    in_channels: int
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.conv = nn.Conv(
            self.in_channels,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="VALID",
            dtype=self.dtype,
        )

    def __call__(self, hidden_states):
        pad = ((0, 0), (0, 1), (0, 1), (0, 0))  # pad height and width dim
        hidden_states = jnp.pad(hidden_states, pad_width=pad)
        hidden_states = self.conv(hidden_states)
        return hidden_states


class ResnetBlock2D(nn.Module):
    in_channels: int
    out_channels: int = None
    dropout_prob: float = 0.0
    use_nin_shortcut: bool = None
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        out_channels = self.in_channels if self.out_channels is None else self.out_channels

        self.norm1 = nn.GroupNorm(num_groups=32, epsilon=1e-6)
        self.conv1 = nn.Conv(
            out_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            dtype=self.dtype,
        )

        self.norm2 = nn.GroupNorm(num_groups=32, epsilon=1e-6)
        self.dropout = nn.Dropout(self.dropout_prob)
        self.conv2 = nn.Conv(
            out_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            dtype=self.dtype,
        )

        use_nin_shortcut = self.in_channels != out_channels if self.use_nin_shortcut is None else self.use_nin_shortcut

        self.conv_shortcut = None
        if use_nin_shortcut:
            self.conv_shortcut = nn.Conv(
                out_channels,
                kernel_size=(1, 1),
                strides=(1, 1),
                padding="VALID",
                dtype=self.dtype,
            )

    def __call__(self, hidden_states, deterministic=True):
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = nn.swish(hidden_states)
        hidden_states = self.conv1(hidden_states)

        hidden_states = self.norm2(hidden_states)
        hidden_states = nn.swish(hidden_states)
        hidden_states = self.dropout(hidden_states, deterministic)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            residual = self.conv_shortcut(residual)

        return hidden_states + residual


class AttentionBlock(nn.Module):
    channels: int
    num_head_channels: int = None
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.num_heads = self.channels // self.num_head_channels if self.num_head_channels is not None else 1

        dense = partial(nn.Dense, self.channels, dtype=self.dtype)

        self.group_norm = nn.GroupNorm(num_groups=32, epsilon=1e-6)
        self.query, self.key, self.value = dense(), dense(), dense()
        self.proj_attn = dense()

    def transpose_for_scores(self, projection):
        new_projection_shape = projection.shape[:-1] + (self.num_heads, -1)
        # move heads to 2nd position (B, T, H * D) -> (B, T, H, D)
        new_projection = projection.reshape(new_projection_shape)
        # (B, T, H, D) -> (B, H, T, D)
        new_projection = jnp.transpose(new_projection, (0, 2, 1, 3))
        return new_projection

    def __call__(self, hidden_states):
        residual = hidden_states
        batch, height, width, channels = hidden_states.shape

        hidden_states = self.group_norm(hidden_states)

        hidden_states = hidden_states.reshape((batch, height * width, channels))

        query = self.query(hidden_states)
        key = self.key(hidden_states)
        value = self.value(hidden_states)

        # transpose
        query = self.transpose_for_scores(query)
        key = self.transpose_for_scores(key)
        value = self.transpose_for_scores(value)

        # compute attentions
        scale = 1 / math.sqrt(math.sqrt(self.channels / self.num_heads))
        attn_weights = jnp.einsum("...qc,...kc->...qk", query * scale, key * scale)
        attn_weights = nn.softmax(attn_weights, axis=-1)

        # attend to values
        hidden_states = jnp.einsum("...kc,...qk->...qc", value, attn_weights)

        hidden_states = jnp.transpose(hidden_states, (0, 2, 1, 3))
        new_hidden_states_shape = hidden_states.shape[:-2] + (self.channels,)
        hidden_states = hidden_states.reshape(new_hidden_states_shape)

        hidden_states = self.proj_attn(hidden_states)
        hidden_states = hidden_states.reshape((batch, height, width, channels))
        hidden_states = hidden_states + residual
        return hidden_states


class DownEncoderBlock2D(nn.Module):
    in_channels: int
    out_channels: int
    dropout: float = 0.0
    num_layers: int = 1
    add_downsample: bool = True
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        resnets = []
        for i in range(self.num_layers):
            in_channels = self.in_channels if i == 0 else self.out_channels

            res_block = ResnetBlock2D(
                in_channels=in_channels,
                out_channels=self.out_channels,
                dropout_prob=self.dropout,
                dtype=self.dtype,
            )
            resnets.append(res_block)
        self.resnets = resnets

        if self.add_downsample:
            self.downsample = Downsample2D(self.out_channels, dtype=self.dtype)

    def __call__(self, hidden_states, deterministic=True):
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, deterministic=deterministic)

        if self.add_downsample:
            hidden_states = self.downsample(hidden_states)

        return hidden_states


class UpEncoderBlock2D(nn.Module):
    in_channels: int
    out_channels: int
    dropout: float = 0.0
    num_layers: int = 1
    add_upsample: bool = True
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        resnets = []
        for i in range(self.num_layers):
            in_channels = self.in_channels if i == 0 else self.out_channels
            res_block = ResnetBlock2D(
                in_channels=in_channels,
                out_channels=self.out_channels,
                dropout_prob=self.dropout,
                dtype=self.dtype,
            )
            resnets.append(res_block)

        self.resnets = resnets

        if self.add_upsample:
            self.upsample = Upsample2D(self.out_channels, dtype=self.dtype)

    def __call__(self, hidden_states, deterministic=True):
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, deterministic=deterministic)

        if self.add_upsample:
            hidden_states = self.upsample(hidden_states)

        return hidden_states


class UNetMidBlock2D(nn.Module):
    in_channels: int
    dropout: float = 0.0
    num_layers: int = 1
    attn_num_head_channels: int = 1
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        # there is always at least one resnet
        resnets = [
            ResnetBlock2D(
                in_channels=self.in_channels,
                out_channels=self.in_channels,
                dropout_prob=self.dropout,
                dtype=self.dtype,
            )
        ]

        attentions = []

        for _ in range(self.num_layers):
            attn_block = AttentionBlock(
                channels=self.in_channels, num_head_channels=self.attn_num_head_channels, dtype=self.dtype
            )
            attentions.append(attn_block)

            res_block = ResnetBlock2D(
                in_channels=self.in_channels,
                out_channels=self.in_channels,
                dropout_prob=self.dropout,
                dtype=self.dtype,
            )
            resnets.append(res_block)

        self.resnets = resnets
        self.attentions = attentions

    def __call__(self, hidden_states, deterministic=True):
        hidden_states = self.resnets[0](hidden_states, deterministic=deterministic)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            hidden_states = attn(hidden_states)
            hidden_states = resnet(hidden_states, deterministic=deterministic)

        return hidden_states


class Encoder(nn.Module):
    in_channels: int = 3
    out_channels: int = 3
    down_block_types: Tuple[str] = ("DownEncoderBlock2D",)
    block_out_channels: Tuple[int] = (64,)
    layers_per_block: int = 2
    norm_num_groups: int = 32
    act_fn: str = "silu"
    double_z: bool = False
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        block_out_channels = self.block_out_channels
        # in
        self.conv_in = nn.Conv(
            block_out_channels[0],
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            dtype=self.dtype,
        )

        # downsampling
        down_blocks = []
        output_channel = block_out_channels[0]
        for i, _ in enumerate(self.down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = DownEncoderBlock2D(
                in_channels=input_channel,
                out_channels=output_channel,
                num_layers=self.layers_per_block,
                add_downsample=not is_final_block,
                dtype=self.dtype,
            )
            down_blocks.append(down_block)
        self.down_blocks = down_blocks

        # middle
        self.mid_block = UNetMidBlock2D(
            in_channels=block_out_channels[-1], attn_num_head_channels=None, dtype=self.dtype
        )

        # end
        conv_out_channels = 2 * self.out_channels if self.double_z else self.out_channels
        self.conv_norm_out = nn.GroupNorm(num_groups=32, epsilon=1e-6)
        self.conv_out = nn.Conv(
            conv_out_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            dtype=self.dtype,
        )

    def __call__(self, sample, deterministic: bool = True):
        # in
        sample = self.conv_in(sample)

        # downsampling
        for block in self.down_blocks:
            sample = block(sample, deterministic=deterministic)

        # middle
        sample = self.mid_block(sample, deterministic=deterministic)

        # end
        sample = self.conv_norm_out(sample)
        sample = nn.swish(sample)
        sample = self.conv_out(sample)

        return sample


class Decoder(nn.Module):
    dtype: jnp.dtype = jnp.float32
    in_channels: int = 3
    out_channels: int = 3
    up_block_types: Tuple[str] = ("UpDecoderBlock2D",)
    block_out_channels: int = (64,)
    layers_per_block: int = 2
    norm_num_groups: int = 32
    act_fn: str = "silu"

    def setup(self):
        block_out_channels = self.block_out_channels

        # z to block_in
        self.conv_in = nn.Conv(
            block_out_channels[-1],
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            dtype=self.dtype,
        )

        # middle
        self.mid_block = UNetMidBlock2D(
            in_channels=block_out_channels[-1], attn_num_head_channels=None, dtype=self.dtype
        )

        # upsampling
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        up_blocks = []
        for i, _ in enumerate(self.up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]

            is_final_block = i == len(block_out_channels) - 1

            up_block = UpEncoderBlock2D(
                in_channels=prev_output_channel,
                out_channels=output_channel,
                num_layers=self.layers_per_block + 1,
                add_upsample=not is_final_block,
                dtype=self.dtype,
            )
            up_blocks.append(up_block)
            prev_output_channel = output_channel

        self.up_blocks = up_blocks

        # end
        self.conv_norm_out = nn.GroupNorm(num_groups=32, epsilon=1e-6)
        self.conv_out = nn.Conv(
            self.out_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            dtype=self.dtype,
        )

    def __call__(self, sample, deterministic: bool = True):
        # z to block_in
        sample = self.conv_in(sample)

        # middle
        sample = self.mid_block(sample, deterministic=deterministic)

        # upsampling
        for block in self.up_blocks:
            sample = block(sample, deterministic=deterministic)

        sample = self.conv_norm_out(sample)
        sample = nn.swish(sample)
        sample = self.conv_out(sample)

        return sample


class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False):
        # Last axis to account for channels-last
        self.mean, self.logvar = jnp.split(parameters, 2, axis=-1)
        self.logvar = jnp.clip(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = jnp.exp(0.5 * self.logvar)
        self.var = jnp.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = jnp.zeros_like(self.mean)

    def sample(self, key):
        return self.mean + self.std * jax.random.normal(key, self.mean.shape)

    def kl(self, other=None):
        if self.deterministic:
            return jnp.array([0.0])

        if other is None:
            return 0.5 * jnp.sum(self.mean**2 + self.var - 1.0 - self.logvar, axis=[1, 2, 3])

        return 0.5 * jnp.sum(
            jnp.square(self.mean - other.mean) / other.var + self.var / other.var - 1.0 - self.logvar + other.logvar,
            axis=[1, 2, 3],
        )

    def nll(self, sample, axis=[1, 2, 3]):
        if self.deterministic:
            return jnp.array([0.0])

        logtwopi = jnp.log(2.0 * jnp.pi)
        return 0.5 * jnp.sum(logtwopi + self.logvar + jnp.square(sample - self.mean) / self.var, axis=axis)

    def mode(self):
        return self.mean


@flax_register_to_config
class FlaxAutoencoderKL(nn.Module, FlaxModelMixin, ConfigMixin):
    in_channels: int = 3
    out_channels: int = 3
    down_block_types: Tuple[str] = ("DownEncoderBlock2D",)
    up_block_types: Tuple[str] = ("UpDecoderBlock2D",)
    block_out_channels: Tuple[int] = (64,)
    layers_per_block: int = 1
    act_fn: str = "silu"
    latent_channels: int = 4
    norm_num_groups: int = 32
    sample_size: int = 32
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.encoder = Encoder(
            in_channels=self.config.in_channels,
            out_channels=self.config.latent_channels,
            down_block_types=self.config.down_block_types,
            block_out_channels=self.config.block_out_channels,
            layers_per_block=self.config.layers_per_block,
            act_fn=self.config.act_fn,
            norm_num_groups=self.config.norm_num_groups,
            double_z=True,
            dtype=self.dtype,
        )
        self.decoder = Decoder(
            in_channels=self.config.latent_channels,
            out_channels=self.config.out_channels,
            up_block_types=self.config.up_block_types,
            block_out_channels=self.config.block_out_channels,
            layers_per_block=self.config.layers_per_block,
            norm_num_groups=self.config.norm_num_groups,
            act_fn=self.config.act_fn,
            dtype=self.dtype,
        )
        self.quant_conv = nn.Conv(
            2 * self.config.latent_channels,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="VALID",
            dtype=self.dtype,
        )
        self.post_quant_conv = nn.Conv(
            self.config.latent_channels,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="VALID",
            dtype=self.dtype,
        )

    def init_weights(self, rng: jax.random.PRNGKey) -> FrozenDict:
        # init input tensors
        sample_shape = (1, self.in_channels, self.sample_size, self.sample_size)
        sample = jnp.zeros(sample_shape, dtype=jnp.float32)

        params_rng, dropout_rng, gaussian_rng = jax.random.split(rng, 3)
        rngs = {"params": params_rng, "dropout": dropout_rng, "gaussian": gaussian_rng}

        return self.init(rngs, sample)["params"]

    def encode(self, sample, deterministic: bool = True, return_dict: bool = True):
        sample = jnp.transpose(sample, (0, 2, 3, 1))

        hidden_states = self.encoder(sample, deterministic=deterministic)
        moments = self.quant_conv(hidden_states)
        posterior = DiagonalGaussianDistribution(moments)

        if not return_dict:
            return (posterior,)

        return FlaxAutoencoderKLOutput(latent_dist=posterior)

    def decode(self, latents, deterministic: bool = True, return_dict: bool = True):
        if latents.shape[-1] != self.config.latent_channels:
            latents = jnp.transpose(latents, (0, 2, 3, 1))

        hidden_states = self.post_quant_conv(latents)
        hidden_states = self.decoder(hidden_states, deterministic=deterministic)

        hidden_states = jnp.transpose(hidden_states, (0, 3, 1, 2))

        if not return_dict:
            return (hidden_states,)

        return FlaxDecoderOutput(sample=hidden_states)

    def __call__(self, sample, sample_posterior=False, deterministic: bool = True, return_dict: bool = True):
        posterior = self.encode(sample, deterministic=deterministic, return_dict=return_dict)
        if sample_posterior:
            rng = self.make_rng("gaussian")
            hidden_states = posterior.latent_dist.sample(rng)
        else:
            hidden_states = posterior.latent_dist.mode()
        hidden_states = self.decode(hidden_states, return_dict=return_dict).sample

        if not return_dict:
            return (sample,)

        return FlaxDecoderOutput(sample=sample)
