# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Cosmos3 AVAE Audio Tokenizer — decoder-only implementation.

The decoder reuses the Oobleck architecture (Snake1d activations + weight-norm
convs + residual units), inlined here instead of imported so the audio module
is self-contained. The corresponding encoder is intentionally not inlined:
upstream Cosmos3 uses a spec-convnext encoder whose tensor layout doesn't map
onto Oobleck's encoder.
"""

import math

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

from ...configuration_utils import ConfigMixin, register_to_config
from ...utils.accelerate_utils import apply_forward_hook
from ..modeling_utils import ModelMixin


# Copied from diffusers.models.autoencoders.autoencoder_oobleck.Snake1d
class Snake1d(nn.Module):
    """
    A 1-dimensional Snake activation function module.
    """

    def __init__(self, hidden_dim, logscale=True):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(1, hidden_dim, 1))
        self.beta = nn.Parameter(torch.zeros(1, hidden_dim, 1))

        self.alpha.requires_grad = True
        self.beta.requires_grad = True
        self.logscale = logscale

    def forward(self, hidden_states):
        shape = hidden_states.shape

        alpha = self.alpha if not self.logscale else torch.exp(self.alpha)
        beta = self.beta if not self.logscale else torch.exp(self.beta)

        hidden_states = hidden_states.reshape(shape[0], shape[1], -1)
        hidden_states = hidden_states + (beta + 1e-9).reciprocal() * torch.sin(alpha * hidden_states).pow(2)
        hidden_states = hidden_states.reshape(shape)
        return hidden_states


# Copied from diffusers.models.autoencoders.autoencoder_oobleck.OobleckResidualUnit with Oobleck->Cosmos3Audio
class Cosmos3AudioResidualUnit(nn.Module):
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


"""
Copied from diffusers.models.autoencoders.autoencoder_oobleck.OobleckDecoderBlock with Oobleck->Cosmos3Audio
with output_padding enabled.
"""
class Cosmos3AudioDecoderBlock(nn.Module):
    """Decoder block used in Cosmos3Audio decoder."""

    def __init__(self, input_dim, output_dim, stride: int = 1, output_padding: int = 0):
        super().__init__()

        self.snake1 = Snake1d(input_dim)
        self.conv_t1 = weight_norm(
            nn.ConvTranspose1d(
                input_dim,
                output_dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
                output_padding=output_padding,
            )
        )
        self.res_unit1 = Cosmos3AudioResidualUnit(output_dim, dilation=1)
        self.res_unit2 = Cosmos3AudioResidualUnit(output_dim, dilation=3)
        self.res_unit3 = Cosmos3AudioResidualUnit(output_dim, dilation=9)

    def forward(self, hidden_state):
        hidden_state = self.snake1(hidden_state)
        hidden_state = self.conv_t1(hidden_state)
        hidden_state = self.res_unit1(hidden_state)
        hidden_state = self.res_unit2(hidden_state)
        hidden_state = self.res_unit3(hidden_state)

        return hidden_state


"""
Copied from diffusers.models.autoencoders.autoencoder_oobleck.OobleckDecoder with Oobleck->Cosmos3Audio and
one change of adding "output_padding=stride % 2,"
"""
class Cosmos3AudioDecoder(nn.Module):
    """Cosmos3Audio Decoder"""

    def __init__(self, channels, input_channels, audio_channels, upsampling_ratios, channel_multiples):
        super().__init__()

        strides = upsampling_ratios
        channel_multiples = [1] + channel_multiples

        # Add first conv layer
        self.conv1 = weight_norm(nn.Conv1d(input_channels, channels * channel_multiples[-1], kernel_size=7, padding=3))

        # Add upsampling + MRF blocks
        block = []
        for stride_index, stride in enumerate(strides):
            block += [
                Cosmos3AudioDecoderBlock(
                    input_dim=channels * channel_multiples[len(strides) - stride_index],
                    output_dim=channels * channel_multiples[len(strides) - stride_index - 1],
                    stride=stride,
                    output_padding=stride % 2,
                )
            ]

        self.block = nn.ModuleList(block)
        output_dim = channels
        self.snake1 = Snake1d(output_dim)
        self.conv2 = weight_norm(nn.Conv1d(channels, audio_channels, kernel_size=7, padding=3, bias=False))

    def forward(self, hidden_state):
        hidden_state = self.conv1(hidden_state)

        for layer in self.block:
            hidden_state = layer(hidden_state)

        hidden_state = self.snake1(hidden_state)
        hidden_state = self.conv2(hidden_state)

        return hidden_state


class Cosmos3AVAEAudioTokenizer(ModelMixin, ConfigMixin):
    """Decoder-only audio tokenizer for Cosmos3 sound generation.

    Wraps the Cosmos3Audio decoder (an inlined copy of Oobleck) used in the AVAE
    (Audio VAE) component of the Cosmos3 omni model. Provides the interface
    expected by ``Cosmos3OmniDiffusersPipeline`` when ``enable_sound=True``.

    For now encoder part of the Tokenizer is not supported.
    The encoder support will be added in the future.

    Parameters:
        sampling_rate (`int`, defaults to `48000`): Audio sample rate in Hz.
        vocoder_input_dim (`int`, defaults to `64`): Latent channel count fed into the decoder
            (``== transformer sound_dim``).
        dec_dim (`int`, defaults to `320`): Base decoder channel count.
        dec_c_mults (`tuple[int, ...]`, defaults to `(1, 2, 4, 8, 16)`): Channel multipliers.
        dec_strides (`tuple[int, ...]`, defaults to `(2, 4, 5, 6, 8)`): Upsampling strides.
        dec_out_channels (`int`, defaults to `2`): Output audio channels (2 = stereo).
    """

    _supports_gradient_checkpointing = False
    _supports_group_offloading = False

    @register_to_config
    def __init__(
        self,
        sampling_rate: int = 48000,
        vocoder_input_dim: int = 64,
        dec_dim: int = 320,
        dec_c_mults: tuple = (1, 2, 4, 8, 16),
        dec_strides: tuple = (2, 4, 5, 6, 8),
        dec_out_channels: int = 2,
    ):
        super().__init__()

        self.decoder = Cosmos3AudioDecoder(
            channels=dec_dim,
            input_channels=vocoder_input_dim,
            audio_channels=dec_out_channels,
            upsampling_ratios=list(reversed(dec_strides)),
            channel_multiples=list(dec_c_mults),
        )

        self._hop_size: int = math.prod(dec_strides)

    @property
    def sample_rate(self) -> int:
        """Audio sample rate in Hz."""
        return self.config.sampling_rate

    @apply_forward_hook
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode sound latents into an audio waveform.

        Args:
            latents: ``[B, C, T]`` or ``[C, T]`` tensor of diffusion-model latents.

        Returns:
            Waveform tensor ``[B, audio_channels, N]`` or ``[audio_channels, N]``.
        """
        squeeze = latents.ndim == 2
        if squeeze:
            latents = latents.unsqueeze(0)
        audio = self.decoder(latents).clamp(-1.0, 1.0)
        return audio.squeeze(0) if squeeze else audio
