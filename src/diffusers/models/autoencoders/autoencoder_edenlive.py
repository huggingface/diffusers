# Copyright 2026 The VAST-AI Team. All rights reserved.
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

from dataclasses import dataclass, field

import torch
from torch import nn
from torch.nn import functional as F

from ...configuration_utils import ConfigMixin, register_to_config
from ...utils.accelerate_utils import apply_forward_hook
from ..modeling_utils import ModelMixin
from .vae import DecoderOutput


@dataclass
class _EdenLiveTAEDecodeState:
    memory: dict[int, torch.Tensor] = field(default_factory=dict)


class EdenLiveTAEClamp(nn.Module):
    def forward(self, hidden_states) -> torch.Tensor:
        return torch.tanh(hidden_states / 3) * 3


class EdenLiveTAEMemoryBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
        self.skip = nn.Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(self, hidden_states, state, batch_size) -> torch.Tensor:
        batch_frames, channels, height, width = hidden_states.shape
        frames = hidden_states.reshape(batch_size, -1, channels, height, width)
        previous = state.memory.get(id(self))
        if previous is None:
            previous = torch.zeros_like(frames[:, :1])
        past = torch.cat([previous, frames[:, :-1]], dim=1).reshape(batch_frames, channels, height, width)
        # Copy the final frame so a short tail does not retain the whole previous chunk.
        state.memory[id(self)] = frames[:, -1:].clone()
        return self.act(self.conv(torch.cat([hidden_states, past], dim=1)) + self.skip(hidden_states))


class EdenLiveTAETemporalUpsample(nn.Module):
    def __init__(self, channels, stride):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels * stride, 1, bias=False)

    def forward(self, hidden_states) -> torch.Tensor:
        _, channels, height, width = hidden_states.shape
        return self.conv(hidden_states).reshape(-1, channels, height, width)


class EdenLiveLightTAEDecoder(ModelMixin, ConfigMixin):
    """Decoder-only LightTAE for EdenLive's Wan latents.

    Input latents use the VAE's raw (unnormalized) scale and `[batch, channels, frames, height, width]` layout. Decoded
    RGB samples use the same layout, range `[-1, 1]`, and `(frames - 1) * 4 + 1` frames. Temporal state belongs to each
    `decode` call and is never retained on the model.

    Parameters:
        latent_channels (`int`, *optional*, defaults to 16):
            Latent channels: 16 for Wan 2.1 / EdenLive 14B, or 48 for Wan 2.2 / EdenLive 5B.
        patch_size (`int`, *optional*, defaults to 1):
            Output pixel shuffle factor: 1 for 14B or 2 for 5B.
        block_out_channels (`tuple[int]`, *optional*, defaults to `(256, 128, 64, 64)`):
            Decoder widths at its four spatial resolutions.
    """

    _no_split_modules = ["EdenLiveTAEMemoryBlock", "EdenLiveTAETemporalUpsample"]

    @register_to_config
    def __init__(self, latent_channels=16, patch_size=1, block_out_channels=(256, 128, 64, 64)):
        super().__init__()
        if latent_channels < 1 or patch_size not in (1, 2):
            raise ValueError("LightTAE requires positive latent channels and patch_size of 1 or 2.")
        if len(block_out_channels) != 4 or any(channels < 1 for channels in block_out_channels):
            raise ValueError("`block_out_channels` must contain four positive widths.")
        widths = block_out_channels
        layers = [EdenLiveTAEClamp(), nn.Conv2d(latent_channels, widths[0], 3, padding=1), nn.ReLU(inplace=True)]
        for stage in range(3):
            layers.extend(EdenLiveTAEMemoryBlock(widths[stage]) for _ in range(3))
            layers.extend(
                [
                    nn.Upsample(scale_factor=2),
                    EdenLiveTAETemporalUpsample(widths[stage], 1 if stage == 0 else 2),
                    nn.Conv2d(widths[stage], widths[stage + 1], 3, padding=1, bias=False),
                ]
            )
        layers.extend([nn.ReLU(inplace=True), nn.Conv2d(widths[3], 3 * patch_size**2, 3, padding=1)])
        self.decoder = nn.Sequential(*layers)

    @property
    def scale_factor_temporal(self):
        return 4

    @property
    def scale_factor_spatial(self):
        return 8 * self.config.patch_size

    def _decode_chunk(self, latents, state):
        first_chunk = not state.memory
        batch_size, _, num_frames, height, width = latents.shape
        hidden_states = latents.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, -1, height, width)
        for layer in self.decoder:
            if isinstance(layer, EdenLiveTAEMemoryBlock):
                hidden_states = layer(hidden_states, state, batch_size)
            else:
                hidden_states = layer(hidden_states)
        hidden_states = hidden_states.clamp(0, 1)
        if self.config.patch_size > 1:
            hidden_states = F.pixel_shuffle(hidden_states, self.config.patch_size)
        frames = hidden_states.reshape(batch_size, -1, *hidden_states.shape[1:])
        if first_chunk:
            frames = frames[:, self.scale_factor_temporal - 1 :]
        return frames.permute(0, 2, 1, 3, 4) * 2 - 1

    @apply_forward_hook
    def decode(self, latents: torch.Tensor, return_dict: bool = True) -> DecoderOutput | tuple[torch.Tensor]:
        """Decode a complete video from raw VAE latents.

        Args:
            latents (`torch.Tensor`):
                Unnormalized VAE latents with shape `(B, C, T_latent, H_latent, W_latent)` on the decoder device and in
                its dtype. `C` must equal `config.latent_channels`, and at least one latent frame is required. Pipeline
                outputs with `output_type="latent"` must first be unnormalized with the VAE mean and standard
                deviation. The pipeline performs this conversion automatically when decoding frames.
            return_dict (`bool`, defaults to `True`):
                Return a [`DecoderOutput`] when true; otherwise return a one-element tuple containing the sample.

        Returns:
            [`DecoderOutput`] or `tuple[torch.Tensor]`:
                RGB sample in `(B, 3, T, H, W)` layout and [-1, 1] range. The output has `(T_latent - 1) *
                scale_factor_temporal + 1` frames and spatial dimensions multiplied by `scale_factor_spatial`. Internal
                chunks share temporal state, including a short final chunk.
        """
        if latents.ndim != 5 or latents.shape[1] != self.config.latent_channels or latents.shape[2] < 1:
            raise ValueError("Expected nonempty NCTHW latents with the configured latent channel count.")
        # Keep one state object across hooks that rebuild dictionary and list arguments when moving tensors.
        state = _EdenLiveTAEDecodeState()
        sample = torch.cat([self._decode_chunk(chunk, state) for chunk in latents.split(3, dim=2)], dim=2)
        if not return_dict:
            return (sample,)
        return DecoderOutput(sample=sample)

    def forward(self, latents: torch.Tensor, return_dict: bool = True) -> DecoderOutput | tuple[torch.Tensor]:
        """
        Args:
            latents (`torch.Tensor`):
                Raw video latents in `[batch, channels, frames, height, width]` layout.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`DecoderOutput`] instead of a tuple containing the decoded sample.

        Returns:
            [`DecoderOutput`] or `tuple[torch.Tensor]`:
                The decoded RGB video in NCTHW layout and `[-1, 1]` range.
        """
        return self.decode(latents, return_dict=return_dict)
