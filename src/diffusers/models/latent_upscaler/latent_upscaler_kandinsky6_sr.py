# Copyright 2025 The Kandinsky Team and The HuggingFace Team. All rights reserved.
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

"""Latent upscalers used by the Kandinsky 6 video super-resolution pipeline."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional

from ...configuration_utils import ConfigMixin, register_to_config
from ..autoencoders.vae import DecoderOutput
from ..modeling_utils import ModelMixin


class Kandinsky6SRLatentUpscalerConv3d(nn.Conv3d):
    """`Conv3d` that replicates the edge frame along time and zero-pads height and width, matching the K-VAE
    latents this model operates on: repeating the boundary frame avoids a zero "hole" next to frame 0, which already
    encodes a single pixel frame while every later latent frame aggregates several.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        super().__init__(in_channels, out_channels, kernel_size, padding=(0, kernel_size // 2, kernel_size // 2))
        self.temporal_pad = kernel_size // 2

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = functional.pad(
            hidden_states, (0, 0, 0, 0, self.temporal_pad, self.temporal_pad), mode="replicate"
        )
        return super().forward(hidden_states)


class Kandinsky6SRLatentUpscalerRMSNorm(nn.Module):
    """Channel-first RMS normalization with a learnable gain, computed in float32."""

    def __init__(self, num_channels: int) -> None:
        super().__init__()
        self.scale = num_channels**0.5
        self.gamma = nn.Parameter(torch.ones(num_channels, 1, 1, 1))

    def forward(self, hidden_states: Tensor) -> Tensor:
        normalized = functional.normalize(hidden_states.float(), dim=1).to(hidden_states.dtype)
        return normalized * self.scale * self.gamma


class Kandinsky6SRLatentUpscalerModulatedNorm(nn.Module):
    """RMS norm followed by a FiLM modulation computed from the input latent `zq`, nearest-upsampled to the feature
    grid so the same conditioning serves every resolution of the cascade."""

    def __init__(self, num_channels: int, zq_channels: int) -> None:
        super().__init__()
        self.norm = Kandinsky6SRLatentUpscalerRMSNorm(num_channels)
        self.conv_y = nn.Conv3d(zq_channels, num_channels, kernel_size=1)
        self.conv_b = nn.Conv3d(zq_channels, num_channels, kernel_size=1)

    def forward(self, hidden_states: Tensor, zq: Tensor) -> Tensor:
        if zq.shape[2:] != hidden_states.shape[2:]:
            zq = functional.interpolate(zq, size=hidden_states.shape[2:], mode="nearest")
        return self.norm(hidden_states) * self.conv_y(zq) + self.conv_b(zq)


class Kandinsky6SRLatentUpscalerResidualBlock(nn.Module):
    """Pre-activation residual block: `norm -> SiLU -> conv3x3x3 -> norm -> SiLU -> conv3x3x3`, with a 1x1 shortcut
    when the width changes."""

    def __init__(self, in_channels: int, out_channels: int, zq_channels: int) -> None:
        super().__init__()
        self.norm1 = Kandinsky6SRLatentUpscalerModulatedNorm(in_channels, zq_channels)
        self.conv1 = Kandinsky6SRLatentUpscalerConv3d(in_channels, out_channels, kernel_size=3)
        self.norm2 = Kandinsky6SRLatentUpscalerModulatedNorm(out_channels, zq_channels)
        self.conv2 = Kandinsky6SRLatentUpscalerConv3d(out_channels, out_channels, kernel_size=3)
        self.shortcut = (
            nn.Identity() if in_channels == out_channels else nn.Conv3d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, hidden_states: Tensor, zq: Tensor) -> Tensor:
        residual = self.conv1(functional.silu(self.norm1(hidden_states, zq)))
        residual = self.conv2(functional.silu(self.norm2(residual, zq)))
        return self.shortcut(hidden_states) + residual


class Kandinsky6SRLatentUpscalerUpsample(nn.Module):
    """Spatial 2x: nearest upsample plus a per-frame convolutional residual, mixed by a pointwise convolution."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.spatial_conv = nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.linear = nn.Conv3d(channels, channels, kernel_size=1)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = functional.interpolate(hidden_states, scale_factor=(1, 2, 2), mode="nearest")
        return self.linear(hidden_states + self.spatial_conv(hidden_states))


class Kandinsky6SRLatentUpscalerOutputHead(nn.Module):
    """`modulated norm -> SiLU -> conv3x3x3` projection back to the latent channels, threading `zq` into the norm."""

    def __init__(self, in_channels: int, out_channels: int, zq_channels: int) -> None:
        super().__init__()
        self.norm = Kandinsky6SRLatentUpscalerModulatedNorm(in_channels, zq_channels)
        self.activation = nn.SiLU()
        self.conv = Kandinsky6SRLatentUpscalerConv3d(in_channels, out_channels, kernel_size=3)

    def forward(self, hidden_states: Tensor, zq: Tensor) -> Tensor:
        return self.conv(self.activation(self.norm(hidden_states, zq)))


class Kandinsky6SRLatentUpscalerX2Branch(nn.Module):
    """The x2 upscaler tail: adapter blocks and a finisher on the input grid, then one 2x stage."""

    def __init__(
        self,
        in_channels: int,
        stage_channels: tuple[int, int, int],
        num_adapter_blocks: int,
        num_mid_blocks: int,
        num_post_blocks: int,
    ) -> None:
        super().__init__()
        width_1, width_2, width_3 = stage_channels
        self.adapter = nn.ModuleList(
            [Kandinsky6SRLatentUpscalerResidualBlock(width_1, width_1, in_channels) for _ in range(num_adapter_blocks)]
        )
        self.finisher = nn.Module()
        self.finisher.spatial_conv = nn.Conv3d(width_1, width_1, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.finisher.linear = nn.Conv3d(width_1, width_1, kernel_size=1)
        mid_widths = [width_1] + [width_2] * num_mid_blocks
        self.mid_blocks = nn.ModuleList(
            [
                Kandinsky6SRLatentUpscalerResidualBlock(mid_widths[index], mid_widths[index + 1], in_channels)
                for index in range(num_mid_blocks)
            ]
        )
        self.upsample = Kandinsky6SRLatentUpscalerUpsample(width_2)
        post_widths = [width_2] + [width_3] * num_post_blocks
        self.blocks = nn.ModuleList(
            [
                Kandinsky6SRLatentUpscalerResidualBlock(post_widths[index], post_widths[index + 1], in_channels)
                for index in range(num_post_blocks)
            ]
        )
        self.output_proj = Kandinsky6SRLatentUpscalerOutputHead(width_3, in_channels, in_channels)

    def forward(self, hidden_states: Tensor, zq: Tensor) -> Tensor:
        for block in self.adapter:
            hidden_states = block(hidden_states, zq)
        hidden_states = self.finisher.linear(hidden_states + self.finisher.spatial_conv(hidden_states))
        for block in self.mid_blocks:
            hidden_states = block(hidden_states, zq)
        hidden_states = self.upsample(hidden_states)
        for block in self.blocks:
            hidden_states = block(hidden_states, zq)
        return self.output_proj(hidden_states, zq)


class Kandinsky6SRLatentUpscaler(nn.Module):
    """One latent upscaler of the bank: a two-stage 2x+2x cascade for `scale=4`, or its single-stage x2 variant.

    Both share the same widths and the same `input_proj -> pre_blocks -> upsample_1 -> mid_blocks -> upsample_2 ->
    post_blocks -> output_proj` backbone (plus an auxiliary `mid_output_head`, a deep-supervision head the checkpoint
    trains with zero loss weight -- see below). The x2 model additionally runs the
    [`Kandinsky6SRLatentUpscalerX2Branch`] tail; the x4 model's `forward` is just its backbone.

    Released checkpoints train the x2 entry's backbone jointly with its [`Kandinsky6SRLatentUpscalerX2Branch`] tail
    (the `x2_adapter_sources` config field names which backbone activations the tail's `adapter` reads from), but
    `forward` here only runs the tail, matching this class's behavior before the backbone was known to exist: the
    backbone and `mid_output_head` are declared so their trained weights load from the checkpoint instead of raising
    "unused weights"/leaving parameters meta/randomly-initialized, but neither is wired into `forward`, so numerically
    this is unchanged from before. Wiring the backbone into the x2 tail's forward pass needs the original training code
    to confirm the exact tap points first -- guessing would risk silently wrong output.
    """

    def __init__(
        self,
        scale: int,
        in_channels: int,
        stage_channels: tuple[int, int, int],
        num_pre_blocks: int,
        num_mid_blocks: int,
        num_post_blocks: int,
        num_x2_adapter_blocks: int,
    ) -> None:
        super().__init__()
        if scale not in (2, 4):
            raise ValueError(f"`scale` must be 2 or 4, got {scale}")
        self.scale = scale
        width_1, width_2, width_3 = stage_channels

        self.input_proj = nn.Sequential(Kandinsky6SRLatentUpscalerConv3d(in_channels, width_1, kernel_size=3))
        self.pre_blocks = nn.ModuleList(
            [Kandinsky6SRLatentUpscalerResidualBlock(width_1, width_1, in_channels) for _ in range(num_pre_blocks)]
        )
        self.upsample_1 = Kandinsky6SRLatentUpscalerUpsample(width_1)
        mid_widths = [width_1] + [width_2] * num_mid_blocks
        self.mid_blocks = nn.ModuleList(
            [
                Kandinsky6SRLatentUpscalerResidualBlock(mid_widths[index], mid_widths[index + 1], in_channels)
                for index in range(num_mid_blocks)
            ]
        )
        self.mid_output_head = nn.Sequential(
            Kandinsky6SRLatentUpscalerRMSNorm(width_2),
            nn.SiLU(),
            Kandinsky6SRLatentUpscalerConv3d(width_2, in_channels, kernel_size=3),
        )
        self.upsample_2 = Kandinsky6SRLatentUpscalerUpsample(width_2)
        post_widths = [width_2] + [width_3] * num_post_blocks
        self.post_blocks = nn.ModuleList(
            [
                Kandinsky6SRLatentUpscalerResidualBlock(post_widths[index], post_widths[index + 1], in_channels)
                for index in range(num_post_blocks)
            ]
        )
        self.output_proj = Kandinsky6SRLatentUpscalerOutputHead(width_3, in_channels, in_channels)

        if scale == 2:
            self.mid_input_proj = nn.Sequential(Kandinsky6SRLatentUpscalerConv3d(in_channels, width_1, kernel_size=3))
            self.x2_branch = Kandinsky6SRLatentUpscalerX2Branch(
                in_channels, stage_channels, num_x2_adapter_blocks, num_mid_blocks, num_post_blocks
            )

    def forward(self, latents: Tensor) -> Tensor:
        zq = latents
        if self.scale == 2:
            return self.x2_branch(self.mid_input_proj(latents), zq)

        hidden_states = self.input_proj(latents)
        for block in self.pre_blocks:
            hidden_states = block(hidden_states, zq)
        hidden_states = self.upsample_1(hidden_states)
        for block in self.mid_blocks:
            hidden_states = block(hidden_states, zq)
        hidden_states = self.upsample_2(hidden_states)
        for block in self.post_blocks:
            hidden_states = block(hidden_states, zq)
        return self.output_proj(hidden_states, zq)


class Kandinsky6SRLatentUpscalerBank(ModelMixin, ConfigMixin):
    r"""
    Bank of latent upscalers used by [`Kandinsky6SRPipeline`]: one [`Kandinsky6SRLatentUpscaler`] per supported spatial
    scale, operating on K-VAE latents.

    Args:
        in_channels (`int`, defaults to `64`):
            Number of latent channels.
        stage_channels (`tuple[int, int, int]`, defaults to `(2048, 1024, 512)`):
            Feature widths of the three stages of the cascade.
        num_pre_blocks (`int`, defaults to `5`):
            Residual blocks before the first upsample of the x4 model.
        num_mid_blocks (`int`, defaults to `3`):
            Residual blocks between the two upsamples.
        num_post_blocks (`int`, defaults to `3`):
            Residual blocks after the last upsample.
        num_x2_adapter_blocks (`int`, defaults to `2`):
            Residual blocks of the x2 model's adapter.
        scales (`tuple[int, ...]`, defaults to `(2, 4)`):
            Spatial scales the bank provides an upscaler for.
        scaling_factor (`float`, defaults to `0.910344`):
            Scale the input latents are expected to carry (the K-VAE `scaling_factor`).
    """

    _no_split_modules = ["Kandinsky6SRLatentUpscaler"]

    @register_to_config
    def __init__(
        self,
        in_channels: int = 64,
        stage_channels: tuple[int, int, int] = (2048, 1024, 512),
        num_pre_blocks: int = 5,
        num_mid_blocks: int = 3,
        num_post_blocks: int = 3,
        num_x2_adapter_blocks: int = 2,
        scales: tuple[int, ...] = (2, 4),
        scaling_factor: float = 0.910344004631042,
    ) -> None:
        super().__init__()
        self._models = nn.ModuleList(
            [
                Kandinsky6SRLatentUpscaler(
                    scale=scale,
                    in_channels=in_channels,
                    stage_channels=stage_channels,
                    num_pre_blocks=num_pre_blocks,
                    num_mid_blocks=num_mid_blocks,
                    num_post_blocks=num_post_blocks,
                    num_x2_adapter_blocks=num_x2_adapter_blocks,
                )
                for scale in scales
            ]
        )

    def forward(self, latents: Tensor, scale: int, return_dict: bool = True) -> DecoderOutput | tuple[Tensor]:
        r"""
        Args:
            latents (`torch.Tensor` of shape `(batch_size, in_channels, num_frames, height, width)`):
                K-VAE latents scaled by `scaling_factor`.
            scale (`int`):
                Spatial upscale factor; one of `scales`.
            return_dict (`bool`, defaults to `True`):
                Whether to return a [`~models.autoencoder_kl.DecoderOutput`] instead of a plain tuple.

        Returns:
            The upscaled latents of shape `(batch_size, in_channels, num_frames, height * scale, width * scale)`.
        """
        if scale not in self.config.scales:
            raise ValueError(f"No latent upscaler for scale {scale}; available scales: {list(self.config.scales)}")
        upscaled = self._models[self.config.scales.index(scale)](latents)
        if not return_dict:
            return (upscaled,)
        return DecoderOutput(sample=upscaled)
