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

Latents are passed directly to the Oobleck decoder (no latent normalisation).
"""

import math

import torch

from ...configuration_utils import ConfigMixin, register_to_config
from ...utils.accelerate_utils import apply_forward_hook
from ..modeling_utils import ModelMixin
from .autoencoder_oobleck import OobleckDecoder


class Cosmos3AVAEAudioTokenizer(ModelMixin, ConfigMixin):
    """Decoder-only audio tokenizer for Cosmos3 sound generation.

    Wraps the Oobleck decoder used in the AVAE (Audio VAE) component of the Cosmos3
    omni model.  Provides the interface expected by ``Cosmos3OmniDiffusersPipeline``
    when ``enable_sound=True``.

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

        self.decoder = OobleckDecoder(
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
