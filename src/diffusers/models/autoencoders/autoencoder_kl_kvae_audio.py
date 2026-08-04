# Copyright 2026 The Kandinsky Lab Team and The HuggingFace Team. All rights reserved.
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
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm

from ...configuration_utils import ConfigMixin, register_to_config
from ...utils.accelerate_utils import apply_forward_hook
from ..attention_processor import Attention
from ..modeling_outputs import AutoencoderKLOutput
from ..modeling_utils import ModelMixin
from .vae import AutoencoderMixin, DecoderOutput, DiagonalGaussianDistribution


class Snake1d(nn.Module):
    """
    A 1-dimensional Snake activation function module.
    """

    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, hidden_states):
        shape = hidden_states.shape
        hidden_states = hidden_states.reshape(shape[0], shape[1], -1)
        hidden_states = hidden_states + (self.alpha + 1e-9).reciprocal() * torch.sin(self.alpha * hidden_states).pow(2)
        hidden_states = hidden_states.reshape(shape)
        return hidden_states


# Copied from diffusers.models.autoencoders.autoencoder_oobleck.OobleckResidualUnit with Oobleck->KVAEAudio
class KVAEAudioResidualUnit(nn.Module):
    """
    A residual unit composed of Snake1d and weight-normalized Conv1d layers with dilations.
    """

    def __init__(self, dimension: int = 16, dilation: int = 1):
        super().__init__()
        pad = ((7 - 1) * dilation) // 2

        self.snake1 = Snake1d(dimension)
        self.conv1 = weight_norm(nn.Conv1d(dimension, dimension, kernel_size=7, dilation=dilation, padding=pad))
        self.snake2 = Snake1d(dimension)
        self.conv2 = weight_norm(nn.Conv1d(dimension, dimension, kernel_size=1))

    def forward(self, hidden_state):
        """
        Forward pass through the residual unit.

        Args:
            hidden_state (`torch.Tensor` of shape `(batch_size, channels, time_steps)`):
                Input tensor .

        Returns:
            output_tensor (`torch.Tensor` of shape `(batch_size, channels, time_steps)`)
                Input tensor after passing through the residual unit.
        """
        output_tensor = hidden_state
        output_tensor = self.conv1(self.snake1(output_tensor))
        output_tensor = self.conv2(self.snake2(output_tensor))

        padding = (hidden_state.shape[-1] - output_tensor.shape[-1]) // 2
        if padding > 0:
            hidden_state = hidden_state[..., padding:-padding]
        output_tensor = hidden_state + output_tensor
        return output_tensor


# Copied from diffusers.models.autoencoders.autoencoder_oobleck.OobleckEncoderBlock with Oobleck->KVAEAudio
class KVAEAudioEncoderBlock(nn.Module):
    """Encoder block used in KVAEAudio encoder."""

    def __init__(self, input_dim, output_dim, stride: int = 1):
        super().__init__()

        self.res_unit1 = KVAEAudioResidualUnit(input_dim, dilation=1)
        self.res_unit2 = KVAEAudioResidualUnit(input_dim, dilation=3)
        self.res_unit3 = KVAEAudioResidualUnit(input_dim, dilation=9)
        self.snake1 = Snake1d(input_dim)
        self.conv1 = weight_norm(
            nn.Conv1d(input_dim, output_dim, kernel_size=2 * stride, stride=stride, padding=math.ceil(stride / 2))
        )

    def forward(self, hidden_state):
        hidden_state = self.res_unit1(hidden_state)
        hidden_state = self.res_unit2(hidden_state)
        hidden_state = self.snake1(self.res_unit3(hidden_state))
        hidden_state = self.conv1(hidden_state)

        return hidden_state


class KVAEAudioDecoderBlock(nn.Module):
    """Decoder block used in KVAEAudio decoder."""

    def __init__(self, input_dim, output_dim, stride: int = 1):
        super().__init__()

        self.snake1 = Snake1d(input_dim)
        # odd strides need output_padding=1 to invert the encoder's downsampling exactly, or decode() silently truncates
        self.conv_t1 = weight_norm(
            nn.ConvTranspose1d(
                input_dim,
                output_dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
                output_padding=stride % 2,
            )
        )
        self.res_unit1 = KVAEAudioResidualUnit(output_dim, dilation=1)
        self.res_unit2 = KVAEAudioResidualUnit(output_dim, dilation=3)
        self.res_unit3 = KVAEAudioResidualUnit(output_dim, dilation=9)

    def forward(self, hidden_state):
        hidden_state = self.snake1(hidden_state)
        hidden_state = self.conv_t1(hidden_state)
        hidden_state = self.res_unit1(hidden_state)
        hidden_state = self.res_unit2(hidden_state)
        hidden_state = self.res_unit3(hidden_state)

        return hidden_state


class KVAEAudioEncoder(nn.Module):
    """KVAEAudio encoder: strided Conv1d downsampling with dilated residual blocks."""

    def __init__(self, encoder_dim: int, encoder_rates: list[int], latent_dim: int, num_channels: int):
        super().__init__()

        self.conv1 = weight_norm(nn.Conv1d(num_channels, encoder_dim, kernel_size=7, padding=3))

        d_model = encoder_dim
        blocks = []
        for stride in encoder_rates:
            input_dim = d_model
            d_model *= 2
            blocks.append(KVAEAudioEncoderBlock(input_dim, d_model, stride=stride))
        self.block = nn.ModuleList(blocks)

        self.snake1 = Snake1d(d_model)
        self.conv2 = weight_norm(nn.Conv1d(d_model, latent_dim, kernel_size=3, padding=1))

    def forward(self, hidden_states):
        hidden_states = self.conv1(hidden_states)

        for block in self.block:
            hidden_states = block(hidden_states)

        hidden_states = self.snake1(hidden_states)
        hidden_states = self.conv2(hidden_states)
        return hidden_states


class KVAEAudioDecoder(nn.Module):
    """KVAEAudio decoder: strided ConvTranspose1d upsampling with dilated residual blocks."""

    def __init__(self, latent_dim: int, decoder_dim: int, decoder_rates: list[int], num_channels: int):
        super().__init__()

        self.conv1 = weight_norm(nn.Conv1d(latent_dim, decoder_dim, kernel_size=7, padding=3))

        d_model = decoder_dim
        blocks = []
        for stride in decoder_rates:
            input_dim = d_model
            d_model = d_model // 2
            blocks.append(KVAEAudioDecoderBlock(input_dim, d_model, stride=stride))
        self.block = nn.ModuleList(blocks)

        self.snake1 = Snake1d(d_model)
        self.conv2 = weight_norm(nn.Conv1d(d_model, num_channels, kernel_size=7, padding=3))
        self.tanh = nn.Tanh()

    def forward(self, hidden_states):
        hidden_states = self.conv1(hidden_states)

        for block in self.block:
            hidden_states = block(hidden_states)

        hidden_states = self.snake1(hidden_states)
        hidden_states = self.conv2(hidden_states)
        hidden_states = self.tanh(hidden_states)
        return hidden_states


class AutoencoderKLKVAEAudio(ModelMixin, AutoencoderMixin, ConfigMixin):
    r"""
    A 1D convolutional autoencoder for encoding raw audio waveforms into continuous latents and decoding them back into
    waveforms. Introduced in [KVAE-Audio](https://huggingface.co/kandinskylab/KVAE-Audio).

    This model inherits from [`ModelMixin`]. Check the superclass documentation for it's generic methods implemented
    for all models (such as downloading or saving).

    Parameters:
        encoder_dim (`int`, *optional*, defaults to 64):
            Base channel dimension for the encoder; doubled at every downsampling stage.
        encoder_rates (`list[int]`, *optional*, defaults to `[2, 3, 4, 5, 8]`):
            Strides for downsampling in the encoder. Used in reverse order for upsampling in the decoder.
        latent_dim (`int`, *optional*):
            Channel dimension of the encoder output (before the `in_proj` bottleneck). Defaults to `encoder_dim * 2 **
            len(encoder_rates)` when not provided.
        codebook_dim (`int`, *optional*, defaults to 64):
            Channel dimension of the posterior distribution's latent space (after the `in_proj` bottleneck).
        decoder_dim (`int`, *optional*, defaults to 1536):
            Base channel dimension for the decoder; halved at every upsampling stage.
        decoder_rates (`list[int]`, *optional*, defaults to `[8, 5, 4, 3, 2]`):
            Strides for upsampling in the decoder.
        sample_rate (`int`, *optional*, defaults to 48000):
            The sampling rate, in Hz, that the model was trained on.
        num_channels (`int`, *optional*, defaults to 1):
            Number of channels in the audio data (1 for mono).
        use_attn (`bool`, *optional*, defaults to `False`):
            Whether to apply a global self-attention block to the encoder output before the `in_proj` bottleneck.
        attn_num_heads (`int`, *optional*, defaults to 8):
            Number of attention heads used when `use_attn=True`.
    """

    _supports_gradient_checkpointing = False
    _supports_group_offloading = False

    @register_to_config
    def __init__(
        self,
        encoder_dim: int = 64,
        encoder_rates: list[int] = [2, 3, 4, 5, 8],
        latent_dim: Optional[int] = None,
        codebook_dim: int = 64,
        decoder_dim: int = 1536,
        decoder_rates: list[int] = [8, 5, 4, 3, 2],
        sample_rate: int = 48000,
        num_channels: int = 1,
        use_attn: bool = False,
        attn_num_heads: int = 8,
    ):
        super().__init__()

        if latent_dim is None:
            latent_dim = encoder_dim * 2 ** len(encoder_rates)
            self.register_to_config(latent_dim=latent_dim)

        self.hop_length = int(np.prod(encoder_rates))

        self.encoder = KVAEAudioEncoder(encoder_dim, encoder_rates, latent_dim, num_channels)
        self.in_proj = weight_norm(nn.Conv1d(latent_dim, codebook_dim * 2, kernel_size=1))
        self.out_proj = weight_norm(nn.Conv1d(codebook_dim, latent_dim, kernel_size=1))
        self.decoder = KVAEAudioDecoder(latent_dim, decoder_dim, decoder_rates, num_channels)

        self.use_attn = use_attn
        if use_attn:
            self.attn = Attention(
                query_dim=latent_dim,
                heads=attn_num_heads,
                dim_head=latent_dim // attn_num_heads,
                bias=True,
                out_bias=True,
            )

        self.use_slicing = False

    def _pad_to_hop_length(self, audio_data: torch.Tensor) -> torch.Tensor:
        length = audio_data.shape[-1]
        right_pad = math.ceil(length / self.hop_length) * self.hop_length - length
        return nn.functional.pad(audio_data, (0, right_pad))

    def _encode(self, audio_data: torch.Tensor) -> torch.Tensor:
        audio_data = self._pad_to_hop_length(audio_data)
        hidden_states = self.encoder(audio_data)

        if self.use_attn:
            hidden_states = hidden_states.transpose(1, 2)
            hidden_states = self.attn(hidden_states)
            hidden_states = hidden_states.transpose(1, 2)

        return self.in_proj(hidden_states)

    @apply_forward_hook
    def encode(
        self, audio_data: torch.Tensor, sample_rate: Optional[int] = None, return_dict: bool = True
    ) -> AutoencoderKLOutput | tuple[DiagonalGaussianDistribution]:
        """
        Encode a batch of audio waveforms into latents.

        Args:
            audio_data (`torch.Tensor` of shape `(batch_size, num_channels, num_samples)`):
                Input batch of raw audio waveforms.
            sample_rate (`int`, *optional*):
                Sample rate of `audio_data`, in Hz. If given, it must match `self.config.sample_rate`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`~models.autoencoder_kl.AutoencoderKLOutput`] instead of a plain tuple.

        Returns:
                The latent representations of the encoded audio. If `return_dict` is True, a
                [`~models.autoencoder_kl.AutoencoderKLOutput`] is returned, otherwise a plain `tuple` is returned.
        """
        if sample_rate is not None and sample_rate != self.config.sample_rate:
            raise ValueError(
                f"`sample_rate` ({sample_rate}) does not match the model's configured sample rate "
                f"({self.config.sample_rate})."
            )

        if self.use_slicing and audio_data.shape[0] > 1:
            encoded_slices = [self._encode(x_slice) for x_slice in audio_data.split(1)]
            moments = torch.cat(encoded_slices)
        else:
            moments = self._encode(audio_data)

        posterior = DiagonalGaussianDistribution(moments)

        if not return_dict:
            return (posterior,)
        return AutoencoderKLOutput(latent_dist=posterior)

    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        z = self.out_proj(z)
        return self.decoder(z)

    @apply_forward_hook
    def decode(self, z: torch.Tensor, return_dict: bool = True) -> DecoderOutput | torch.Tensor:
        """
        Decode a batch of latents into audio waveforms.

        Args:
            z (`torch.Tensor`): Input batch of latent vectors.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`~models.autoencoders.vae.DecoderOutput`] instead of a plain tuple.

        Returns:
            [`~models.autoencoders.vae.DecoderOutput`] or `tuple`:
                If return_dict is True, a [`~models.autoencoders.vae.DecoderOutput`] is returned, otherwise a plain
                `tuple` is returned.
        """
        if self.use_slicing and z.shape[0] > 1:
            decoded_slices = [self._decode(z_slice) for z_slice in z.split(1)]
            decoded = torch.cat(decoded_slices)
        else:
            decoded = self._decode(z)

        if not return_dict:
            return (decoded,)
        return DecoderOutput(sample=decoded)

    def forward(
        self,
        sample: torch.Tensor,
        sample_rate: Optional[int] = None,
        sample_posterior: bool = False,
        return_dict: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> DecoderOutput | torch.Tensor:
        r"""
        Args:
            sample (`torch.Tensor` of shape `(batch_size, num_channels, num_samples)`):
                Input batch of raw audio waveforms.
            sample_rate (`int`, *optional*):
                Sample rate of `sample`, in Hz. If given, it must match `self.config.sample_rate`.
            sample_posterior (`bool`, *optional*, defaults to `False`):
                Whether to sample from the posterior.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.autoencoders.vae.DecoderOutput`] instead of a plain tuple.
            generator (`torch.Generator`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make sampling
                deterministic.

        Returns:
            [`~models.autoencoders.vae.DecoderOutput`] or `tuple`:
                If `return_dict` is True, a [`~models.autoencoders.vae.DecoderOutput`] is returned, otherwise a plain
                `tuple` is returned.
        """
        length = sample.shape[-1]

        posterior = self.encode(sample, sample_rate=sample_rate).latent_dist
        if sample_posterior:
            z = posterior.sample(generator=generator)
        else:
            z = posterior.mode()
        decoded = self.decode(z).sample[..., :length]

        if not return_dict:
            return (decoded,)
        return DecoderOutput(sample=decoded)
