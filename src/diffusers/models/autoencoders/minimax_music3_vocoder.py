# Copyright 2026 The MiniMax Team and The HuggingFace Team. All rights reserved.
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

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

from ...configuration_utils import ConfigMixin, register_to_config
from ..modeling_utils import ModelMixin


class MiniMaxMusic3Snake1d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        shape = hidden_states.shape
        hidden_states = hidden_states.reshape(shape[0], shape[1], -1)
        hidden_states = hidden_states + (self.alpha + 1e-9).reciprocal() * torch.sin(self.alpha * hidden_states).pow(2)
        return hidden_states.reshape(shape)


class MiniMaxMusic3VocoderResidualUnit(nn.Module):
    def __init__(self, dim: int, dilation: int):
        super().__init__()
        pad = (7 - 1) * dilation // 2
        self.snake1 = MiniMaxMusic3Snake1d(dim)
        self.conv1 = weight_norm(nn.Conv1d(dim, dim, kernel_size=7, dilation=dilation, padding=pad))
        self.snake2 = MiniMaxMusic3Snake1d(dim)
        self.conv2 = weight_norm(nn.Conv1d(dim, dim, kernel_size=1))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = self.conv2(self.snake2(self.conv1(self.snake1(hidden_states))))
        return hidden_states + residual


class MiniMaxMusic3VocoderBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, stride: int):
        super().__init__()
        self.snake1 = MiniMaxMusic3Snake1d(input_dim)
        self.conv_t1 = weight_norm(
            nn.ConvTranspose1d(
                input_dim, output_dim, kernel_size=2 * stride, stride=stride, padding=math.ceil(stride / 2)
            )
        )
        self.res_unit1 = MiniMaxMusic3VocoderResidualUnit(output_dim, dilation=1)
        self.res_unit2 = MiniMaxMusic3VocoderResidualUnit(output_dim, dilation=3)
        self.res_unit3 = MiniMaxMusic3VocoderResidualUnit(output_dim, dilation=9)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.conv_t1(self.snake1(hidden_states))
        hidden_states = self.res_unit1(hidden_states)
        hidden_states = self.res_unit2(hidden_states)
        return self.res_unit3(hidden_states)


class MiniMaxMusic3Vocoder(ModelMixin, ConfigMixin):
    r"""
    The Flow-VAE waveform decoder of MiniMax Music 3 (a DAC-style decoder). It decodes flow-matched latents of shape
    `(batch, latent_channels, length)` into stereo waveforms at `sampling_rate`; the two audio channels are decoded as
    two folded `latent_channels // 2` streams.
    """

    @register_to_config
    def __init__(
        self,
        latent_channels: int = 128,
        decoder_input_dim: int = 1024,
        decoder_hidden_dim: int = 1536,
        upsampling_ratios: tuple = (8, 8, 4, 2),
        sampling_rate: int = 44100,
    ):
        super().__init__()
        self.dec_in_proj = nn.Conv1d(latent_channels // 2, decoder_input_dim, kernel_size=1)
        self.conv_in = weight_norm(nn.Conv1d(decoder_input_dim, decoder_hidden_dim, kernel_size=7, padding=3))
        blocks = []
        output_dim = decoder_hidden_dim
        for index, stride in enumerate(upsampling_ratios):
            input_dim = decoder_hidden_dim // (2**index)
            output_dim = decoder_hidden_dim // (2 ** (index + 1))
            blocks.append(MiniMaxMusic3VocoderBlock(input_dim, output_dim, stride))
        self.blocks = nn.ModuleList(blocks)
        self.snake_out = MiniMaxMusic3Snake1d(output_dim)
        self.conv_out = weight_norm(nn.Conv1d(output_dim, 1, kernel_size=7, padding=3))

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            latents (`torch.Tensor` of shape `(batch, latent_channels, length)`):
                Flow-matched Flow-VAE latents.

        Returns:
            `torch.Tensor` of shape `(batch, 2, samples)`: the stereo waveform in `[-1, 1]`.
        """
        batch_size, _, length = latents.shape
        hidden_states = latents.reshape(batch_size * 2, self.config.latent_channels // 2, length)
        hidden_states = self.conv_in(self.dec_in_proj(hidden_states))
        for block in self.blocks:
            hidden_states = block(hidden_states)
        waveform = torch.tanh(self.conv_out(self.snake_out(hidden_states)))
        return waveform.reshape(batch_size, 2, -1)
