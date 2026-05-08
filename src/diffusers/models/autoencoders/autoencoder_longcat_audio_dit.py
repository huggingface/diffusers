# Copyright 2026 MeiTuan LongCat-AudioDiT Team and The HuggingFace Team. All rights reserved.
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

# Adapted from the LongCat-AudioDiT reference implementation:
# https://github.com/meituan-longcat/LongCat-AudioDiT

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

from ...configuration_utils import ConfigMixin, register_to_config
from ...utils import BaseOutput
from ...utils.accelerate_utils import apply_forward_hook
from ...utils.torch_utils import randn_tensor
from ..modeling_utils import ModelMixin
from .vae import AutoencoderMixin


def _wn_conv1d(in_channels, out_channels, kernel_size, stride=1, dilation=1, padding=0, bias=True):
    return weight_norm(nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias))


def _wn_conv_transpose1d(*args, **kwargs):
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))


class Snake1d(nn.Module):
    def __init__(self, channels: int, alpha_logscale: bool = True):
        super().__init__()
        self.alpha_logscale = alpha_logscale
        self.alpha = nn.Parameter(torch.zeros(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha[None, :, None]
        beta = self.beta[None, :, None]
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
            beta = torch.exp(beta)
        return hidden_states + (1.0 / (beta + 1e-9)) * torch.sin(hidden_states * alpha).pow(2)


def _get_vae_activation(name: str, channels: int = 0) -> nn.Module:
    if name == "elu":
        act = nn.ELU()
    elif name == "snake":
        act = Snake1d(channels)
    else:
        raise ValueError(f"Unknown activation: {name}")
    return act


def _pixel_shuffle_1d(hidden_states: torch.Tensor, factor: int) -> torch.Tensor:
    batch, channels, width = hidden_states.size()
    return (
        hidden_states.view(batch, channels // factor, factor, width)
        .permute(0, 1, 3, 2)
        .contiguous()
        .view(batch, channels // factor, width * factor)
    )


class DownsampleShortcut(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, factor: int):
        super().__init__()
        self.factor = factor
        self.group_size = in_channels * factor // out_channels
        self.out_channels = out_channels

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch, channels, width = hidden_states.shape
        hidden_states = (
            hidden_states.view(batch, channels, width // self.factor, self.factor)
            .permute(0, 1, 3, 2)
            .contiguous()
            .view(batch, channels * self.factor, width // self.factor)
        )
        return hidden_states.view(batch, self.out_channels, self.group_size, width // self.factor).mean(dim=2)


class UpsampleShortcut(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, factor: int):
        super().__init__()
        self.factor = factor
        self.repeats = out_channels * factor // in_channels

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.repeat_interleave(self.repeats, dim=1)
        return _pixel_shuffle_1d(hidden_states, self.factor)


class VaeResidualUnit(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, dilation: int, kernel_size: int = 7, act_fn: str = "snake"
    ):
        super().__init__()
        padding = (dilation * (kernel_size - 1)) // 2
        self.layers = nn.Sequential(
            _get_vae_activation(act_fn, channels=out_channels),
            _wn_conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=padding),
            _get_vae_activation(act_fn, channels=out_channels),
            _wn_conv1d(out_channels, out_channels, kernel_size=1),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states + self.layers(hidden_states)


class VaeEncoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        act_fn: str = "snake",
        downsample_shortcut: str = "none",
    ):
        super().__init__()
        layers = [
            VaeResidualUnit(in_channels, in_channels, dilation=1, act_fn=act_fn),
            VaeResidualUnit(in_channels, in_channels, dilation=3, act_fn=act_fn),
            VaeResidualUnit(in_channels, in_channels, dilation=9, act_fn=act_fn),
        ]
        layers.append(_get_vae_activation(act_fn, channels=in_channels))
        layers.append(
            _wn_conv1d(in_channels, out_channels, kernel_size=2 * stride, stride=stride, padding=math.ceil(stride / 2))
        )
        self.layers = nn.Sequential(*layers)
        self.residual = (
            DownsampleShortcut(in_channels, out_channels, stride) if downsample_shortcut == "averaging" else None
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        output_hidden_states = self.layers(hidden_states)
        if self.residual is not None:
            residual = self.residual(hidden_states)
            output_hidden_states = output_hidden_states + residual
        return output_hidden_states


class VaeDecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        act_fn: str = "snake",
        upsample_shortcut: str = "none",
    ):
        super().__init__()
        layers = [
            _get_vae_activation(act_fn, channels=in_channels),
            _wn_conv_transpose1d(
                in_channels, out_channels, kernel_size=2 * stride, stride=stride, padding=math.ceil(stride / 2)
            ),
            VaeResidualUnit(out_channels, out_channels, dilation=1, act_fn=act_fn),
            VaeResidualUnit(out_channels, out_channels, dilation=3, act_fn=act_fn),
            VaeResidualUnit(out_channels, out_channels, dilation=9, act_fn=act_fn),
        ]
        self.layers = nn.Sequential(*layers)
        self.residual = (
            UpsampleShortcut(in_channels, out_channels, stride) if upsample_shortcut == "duplicating" else None
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        output_hidden_states = self.layers(hidden_states)
        if self.residual is not None:
            residual = self.residual(hidden_states)
            output_hidden_states = output_hidden_states + residual
        return output_hidden_states


class AudioDiTVaeEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        channels: int = 128,
        c_mults: list[int] | None = None,
        strides: list[int] | None = None,
        latent_dim: int = 64,
        encoder_latent_dim: int = 128,
        act_fn: str = "snake",
        downsample_shortcut: str = "averaging",
        out_shortcut: str = "averaging",
    ):
        super().__init__()
        c_mults = [1] + (c_mults or [1, 2, 4, 8, 16])
        strides = list(strides or [2] * (len(c_mults) - 1))
        if len(strides) < len(c_mults) - 1:
            strides.extend([strides[-1] if strides else 2] * (len(c_mults) - 1 - len(strides)))
        else:
            strides = strides[: len(c_mults) - 1]
        channels_base = channels
        layers = [_wn_conv1d(in_channels, c_mults[0] * channels_base, kernel_size=7, padding=3)]
        for idx in range(len(c_mults) - 1):
            layers.append(
                VaeEncoderBlock(
                    c_mults[idx] * channels_base,
                    c_mults[idx + 1] * channels_base,
                    strides[idx],
                    act_fn=act_fn,
                    downsample_shortcut=downsample_shortcut,
                )
            )
        layers.append(_wn_conv1d(c_mults[-1] * channels_base, encoder_latent_dim, kernel_size=3, padding=1))
        self.layers = nn.Sequential(*layers)
        self.shortcut = (
            DownsampleShortcut(c_mults[-1] * channels_base, encoder_latent_dim, 1)
            if out_shortcut == "averaging"
            else None
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.layers[:-1](hidden_states)
        output_hidden_states = self.layers[-1](hidden_states)
        if self.shortcut is not None:
            shortcut = self.shortcut(hidden_states)
            output_hidden_states = output_hidden_states + shortcut
        return output_hidden_states


class AudioDiTVaeDecoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        channels: int = 128,
        c_mults: list[int] | None = None,
        strides: list[int] | None = None,
        latent_dim: int = 64,
        act_fn: str = "snake",
        in_shortcut: str = "duplicating",
        final_tanh: bool = False,
        upsample_shortcut: str = "duplicating",
    ):
        super().__init__()
        c_mults = [1] + (c_mults or [1, 2, 4, 8, 16])
        strides = list(strides or [2] * (len(c_mults) - 1))
        if len(strides) < len(c_mults) - 1:
            strides.extend([strides[-1] if strides else 2] * (len(c_mults) - 1 - len(strides)))
        else:
            strides = strides[: len(c_mults) - 1]
        channels_base = channels

        self.shortcut = (
            UpsampleShortcut(latent_dim, c_mults[-1] * channels_base, 1) if in_shortcut == "duplicating" else None
        )

        layers = [_wn_conv1d(latent_dim, c_mults[-1] * channels_base, kernel_size=7, padding=3)]
        for idx in range(len(c_mults) - 1, 0, -1):
            layers.append(
                VaeDecoderBlock(
                    c_mults[idx] * channels_base,
                    c_mults[idx - 1] * channels_base,
                    strides[idx - 1],
                    act_fn=act_fn,
                    upsample_shortcut=upsample_shortcut,
                )
            )
        layers.append(_get_vae_activation(act_fn, channels=c_mults[0] * channels_base))
        layers.append(_wn_conv1d(c_mults[0] * channels_base, in_channels, kernel_size=7, padding=3, bias=False))
        layers.append(nn.Tanh() if final_tanh else nn.Identity())
        self.layers = nn.Sequential(*layers)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.shortcut is None:
            return self.layers(hidden_states)
        hidden_states = self.shortcut(hidden_states) + self.layers[0](hidden_states)
        return self.layers[1:](hidden_states)


@dataclass
class LongCatAudioDiTVaeEncoderOutput(BaseOutput):
    latents: torch.Tensor


@dataclass
class LongCatAudioDiTVaeDecoderOutput(BaseOutput):
    sample: torch.Tensor


class LongCatAudioDiTVae(ModelMixin, AutoencoderMixin, ConfigMixin):
    _supports_group_offloading = False

    @register_to_config
    def __init__(
        self,
        in_channels: int = 1,
        channels: int = 128,
        c_mults: list[int] | None = None,
        strides: list[int] | None = None,
        latent_dim: int = 64,
        encoder_latent_dim: int = 128,
        act_fn: str | None = None,
        use_snake: bool | None = None,
        downsample_shortcut: str = "averaging",
        upsample_shortcut: str = "duplicating",
        out_shortcut: str = "averaging",
        in_shortcut: str = "duplicating",
        final_tanh: bool = False,
        downsampling_ratio: int = 2048,
        sample_rate: int = 24000,
        scale: float = 0.71,
    ):
        super().__init__()
        if act_fn is None:
            if use_snake is None:
                act_fn = "snake"
            else:
                act_fn = "snake" if use_snake else "elu"
        self.encoder = AudioDiTVaeEncoder(
            in_channels=in_channels,
            channels=channels,
            c_mults=c_mults,
            strides=strides,
            latent_dim=latent_dim,
            encoder_latent_dim=encoder_latent_dim,
            act_fn=act_fn,
            downsample_shortcut=downsample_shortcut,
            out_shortcut=out_shortcut,
        )
        self.decoder = AudioDiTVaeDecoder(
            in_channels=in_channels,
            channels=channels,
            c_mults=c_mults,
            strides=strides,
            latent_dim=latent_dim,
            act_fn=act_fn,
            in_shortcut=in_shortcut,
            final_tanh=final_tanh,
            upsample_shortcut=upsample_shortcut,
        )

    @apply_forward_hook
    def encode(
        self,
        sample: torch.Tensor,
        sample_posterior: bool = True,
        return_dict: bool = True,
        generator: torch.Generator | None = None,
    ) -> LongCatAudioDiTVaeEncoderOutput | tuple[torch.Tensor]:
        encoder_dtype = next(self.encoder.parameters()).dtype
        if sample.dtype != encoder_dtype:
            sample = sample.to(encoder_dtype)
        encoded = self.encoder(sample)
        mean, scale_param = encoded.chunk(2, dim=1)
        std = F.softplus(scale_param) + 1e-4
        if sample_posterior:
            noise = randn_tensor(mean.shape, generator=generator, device=mean.device, dtype=mean.dtype)
            latents = mean + std * noise
        else:
            latents = mean
        latents = latents / self.config.scale
        if encoder_dtype != torch.float32:
            latents = latents.float()
        if not return_dict:
            return (latents,)
        return LongCatAudioDiTVaeEncoderOutput(latents=latents)

    @apply_forward_hook
    def decode(
        self, latents: torch.Tensor, return_dict: bool = True
    ) -> LongCatAudioDiTVaeDecoderOutput | tuple[torch.Tensor]:
        decoder_dtype = next(self.decoder.parameters()).dtype
        latents = latents * self.config.scale
        if latents.dtype != decoder_dtype:
            latents = latents.to(decoder_dtype)
        decoded = self.decoder(latents)
        if decoder_dtype != torch.float32:
            decoded = decoded.float()
        if not return_dict:
            return (decoded,)
        return LongCatAudioDiTVaeDecoderOutput(sample=decoded)

    def forward(
        self,
        sample: torch.Tensor,
        sample_posterior: bool = False,
        return_dict: bool = True,
        generator: torch.Generator | None = None,
    ) -> LongCatAudioDiTVaeDecoderOutput | tuple[torch.Tensor]:
        latents = self.encode(sample, sample_posterior=sample_posterior, return_dict=True, generator=generator).latents
        decoded = self.decode(latents, return_dict=True).sample
        if not return_dict:
            return (decoded,)
        return LongCatAudioDiTVaeDecoderOutput(sample=decoded)
