# Copyright 2025 The Lightricks team and The HuggingFace Team.
# All rights reserved.
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

from collections import namedtuple
from typing import Optional, Set, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...configuration_utils import ConfigMixin, register_to_config
from ...utils.accelerate_utils import apply_forward_hook
from ..modeling_utils import ModelMixin
from .vae import AutoencoderMixin, DecoderOutput


LATENT_DOWNSAMPLE_FACTOR = 4
SUPPORTED_CAUSAL_AXES = {"none", "width", "height", "width-compatibility"}


AudioLatentShape = namedtuple(
    "AudioLatentShape",
    [
        "batch",
        "channels",
        "frames",
        "mel_bins",
    ],
)


def _resolve_causality_axis(causality_axis: Optional[str] = None) -> Optional[str]:
    normalized = "none" if causality_axis is None else str(causality_axis).lower()
    if normalized not in SUPPORTED_CAUSAL_AXES:
        raise NotImplementedError(
            f"Unsupported causality_axis '{causality_axis}'. Supported: {sorted(SUPPORTED_CAUSAL_AXES)}"
        )
    return None if normalized == "none" else normalized


def make_conv2d(
    in_channels: int,
    out_channels: int,
    kernel_size: Union[int, Tuple[int, int]],
    stride: int = 1,
    padding: Optional[Tuple[int, int, int, int]] = None,
    dilation: int = 1,
    groups: int = 1,
    bias: bool = True,
    causality_axis: Optional[str] = None,
) -> nn.Module:
    if causality_axis is not None:
        return LTX2AudioCausalConv2d(
            in_channels, out_channels, kernel_size, stride, dilation, groups, bias, causality_axis
        )
    if padding is None:
        padding = kernel_size // 2 if isinstance(kernel_size, int) else tuple(k // 2 for k in kernel_size)

    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        groups,
        bias,
    )


class LTX2AudioCausalConv2d(nn.Module):
    """
    A causal 2D convolution that pads asymmetrically along the causal axis.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: int = 1,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
        causality_axis: str = "height",
    ) -> None:
        super().__init__()

        self.causality_axis = causality_axis
        kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        dilation = (dilation, dilation) if isinstance(dilation, int) else dilation


        pad_h = (kernel_size[0] - 1) * dilation[0]
        pad_w = (kernel_size[1] - 1) * dilation[1]

        if self.causality_axis == "none":
            padding = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)
        elif self.causality_axis in {"width", "width-compatibility"}:
            padding = (pad_w, 0, pad_h // 2, pad_h - pad_h // 2)
        elif self.causality_axis == "height":
            padding = (pad_w // 2, pad_w - pad_w // 2, pad_h, 0)
        else:
            raise ValueError(f"Invalid causality_axis: {causality_axis}")

        self.padding = padding
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, self.padding)
        return self.conv(x)


class LTX2AudioPixelNorm(nn.Module):
    """
    Per-pixel (per-location) RMS normalization layer.
    """

    def __init__(self, dim: int = 1, eps: float = 1e-8) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean_sq = torch.mean(x**2, dim=self.dim, keepdim=True)
        rms = torch.sqrt(mean_sq + self.eps)
        return x / rms


def build_normalization_layer(in_channels: int, *, num_groups: int = 32, normtype: str = "group") -> nn.Module:
    if normtype == "group":
        return nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)
    if normtype == "pixel":
        return LTX2AudioPixelNorm(dim=1, eps=1e-6)
    raise ValueError(f"Invalid normalization type: {normtype}")


class LTX2AudioAttnBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        norm_type: str = "group",
    ) -> None:
        super().__init__()
        self.in_channels = in_channels

        self.norm = build_normalization_layer(in_channels, normtype=norm_type)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_ = self.norm(x)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        batch, channels, height, width = q.shape
        q = q.reshape(batch, channels, height * width).permute(0, 2, 1).contiguous()
        k = k.reshape(batch, channels, height * width).contiguous()
        attn = torch.bmm(q, k) * (int(channels) ** (-0.5))
        attn = torch.nn.functional.softmax(attn, dim=2)

        v = v.reshape(batch, channels, height * width)
        attn = attn.permute(0, 2, 1).contiguous()
        h_ = torch.bmm(v, attn).reshape(batch, channels, height, width)

        h_ = self.proj_out(h_)
        return x + h_


class LTX2AudioResnetBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        conv_shortcut: bool = False,
        dropout: float = 0.0,
        temb_channels: int = 512,
        norm_type: str = "group",
        causality_axis: str = "height",
    ) -> None:
        super().__init__()
        self.causality_axis = causality_axis

        if self.causality_axis is not None and self.causality_axis != "none" and norm_type == "group":
            raise ValueError("Causal ResnetBlock with GroupNorm is not supported.")
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = build_normalization_layer(in_channels, normtype=norm_type)
        self.non_linearity = nn.SiLU()
        self.conv1 = make_conv2d(in_channels, out_channels, kernel_size=3, stride=1, causality_axis=causality_axis)
        if temb_channels > 0:
            self.temb_proj = nn.Linear(temb_channels, out_channels)
        self.norm2 = build_normalization_layer(out_channels, normtype=norm_type)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = make_conv2d(out_channels, out_channels, kernel_size=3, stride=1, causality_axis=causality_axis)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = make_conv2d(
                    in_channels, out_channels, kernel_size=3, stride=1, causality_axis=causality_axis
                )
            else:
                self.nin_shortcut = make_conv2d(
                    in_channels, out_channels, kernel_size=1, stride=1, causality_axis=causality_axis
                )

    def forward(
        self,
        x: torch.Tensor,
        temb: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        h = self.norm1(x)
        h = self.non_linearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(self.non_linearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = self.non_linearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.conv_shortcut(x) if self.use_conv_shortcut else self.nin_shortcut(x)

        return x + h


class LTX2AudioUpsample(nn.Module):
    def __init__(
        self,
        in_channels: int,
        with_conv: bool,
        causality_axis: Optional[str] = "height"
    ) -> None:
        super().__init__()
        self.with_conv = with_conv
        self.causality_axis = causality_axis
        if self.with_conv:
            self.conv = make_conv2d(in_channels, in_channels, kernel_size=3, stride=1, causality_axis=causality_axis)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
            if self.causality_axis is None or self.causality_axis == "none":
                pass
            elif self.causality_axis == "height":
                x = x[:, :, 1:, :]
            elif self.causality_axis == "width":
                x = x[:, :, :, 1:]
            elif self.causality_axis == "width-compatibility":
                pass
            else:
                raise ValueError(f"Invalid causality_axis: {self.causality_axis}")

        return x


class LTX2AudioPerChannelStatistics(nn.Module):
    """
    Per-channel statistics for normalizing and denormalizing the latent representation. This statics is computed over
    the entire dataset and stored in model's checkpoint under AudioVAE state_dict
    """

    def __init__(self, latent_channels: int = 128) -> None:
        super().__init__()
        # Sayak notes: `empty` always causes problems in CI. Should we consider using `torch.ones`?
        self.register_buffer("std-of-means", torch.empty(latent_channels))
        self.register_buffer("mean-of-means", torch.empty(latent_channels))

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x * self.get_buffer("std-of-means").to(x)) + self.get_buffer("mean-of-means").to(x)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.get_buffer("mean-of-means").to(x)) / self.get_buffer("std-of-means").to(x)


class LTX2AudioAudioPatchifier:
    """
    Patchifier for spectrogram/audio latents.
    """

    def __init__(
        self,
        patch_size: int,
        sample_rate: int = 16000,
        hop_length: int = 160,
        audio_latent_downsample_factor: int = 4,
        is_causal: bool = True,
    ):
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        self.audio_latent_downsample_factor = audio_latent_downsample_factor
        self.is_causal = is_causal
        self._patch_size = (1, patch_size, patch_size)

    def patchify(self, audio_latents: torch.Tensor) -> torch.Tensor:
        batch, channels, time, freq = audio_latents.shape
        return audio_latents.permute(0, 2, 1, 3).reshape(batch, time, channels * freq)

    def unpatchify(
        self,
        audio_latents: torch.Tensor,
        output_shape: AudioLatentShape
    ) -> torch.Tensor:
        batch, time, _ = audio_latents.shape
        channels = output_shape.channels
        freq = output_shape.mel_bins
        return audio_latents.view(batch, time, channels, freq).permute(0, 2, 1, 3)

    @property
    def patch_size(self) -> Tuple[int, int, int]:
        return self._patch_size


class LTX2AudioDecoder(nn.Module):
    """
    Symmetric decoder that reconstructs audio spectrograms from latent features.

    The decoder mirrors the encoder structure with configurable channel multipliers, attention resolutions, and causal
    convolutions.
    """

    def __init__(
        self,
        base_channels: int,
        output_channels: int,
        num_res_blocks: int,
        attn_resolutions: Set[int],
        in_channels: int,
        resolution: int,
        latent_channels: int,
        ch_mult: Tuple[int, ...] = (1, 2, 4, 8),
        norm_type: str = "group",
        causality_axis: Optional[str] = "width",
        dropout: float = 0.0,
        mid_block_add_attention: bool = True,
        sample_rate: int = 16000,
        mel_hop_length: int = 160,
        is_causal: bool = True,
        mel_bins: Optional[int] = None,
    ) -> None:
        super().__init__()

        resolved_causality_axis = _resolve_causality_axis(causality_axis)

        self.per_channel_statistics = LTX2AudioPerChannelStatistics(latent_channels=base_channels)
        self.sample_rate = sample_rate
        self.mel_hop_length = mel_hop_length
        self.is_causal = is_causal
        self.mel_bins = mel_bins
        self.patchifier = LTX2AudioAudioPatchifier(
            patch_size=1,
            audio_latent_downsample_factor=LATENT_DOWNSAMPLE_FACTOR,
            sample_rate=sample_rate,
            hop_length=mel_hop_length,
            is_causal=is_causal,
        )

        self.base_channels = base_channels
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.out_ch = output_channels
        self.give_pre_end = False
        self.tanh_out = False
        self.norm_type = norm_type
        self.latent_channels = latent_channels
        self.channel_multipliers = ch_mult
        self.attn_resolutions = attn_resolutions
        self.causality_axis = resolved_causality_axis

        base_block_channels = base_channels * self.channel_multipliers[-1]
        base_resolution = resolution // (2 ** (self.num_resolutions - 1))
        self.z_shape = (1, latent_channels, base_resolution, base_resolution)

        self.conv_in = make_conv2d(
            latent_channels, base_block_channels, kernel_size=3, stride=1, causality_axis=self.causality_axis
        )
        self.non_linearity = nn.SiLU()
        self.mid = self._build_mid_layers(base_block_channels, dropout, mid_block_add_attention)
        self.up, final_block_channels = self._build_up_path(
            initial_block_channels=base_block_channels,
            dropout=dropout,
            resamp_with_conv=True,
        )

        self.norm_out = build_normalization_layer(final_block_channels, normtype=self.norm_type)
        self.conv_out = make_conv2d(
            final_block_channels, output_channels, kernel_size=3, stride=1, causality_axis=self.causality_axis
        )

    def _adjust_output_shape(
        self,
        decoded_output: torch.Tensor,
        target_shape: AudioLatentShape
    ) -> torch.Tensor:
        _, _, current_time, current_freq = decoded_output.shape
        target_channels = target_shape.channels
        target_time = target_shape.frames
        target_freq = target_shape.mel_bins

        decoded_output = decoded_output[
            :, :target_channels, : min(current_time, target_time), : min(current_freq, target_freq)
        ]

        time_padding_needed = target_time - decoded_output.shape[2]
        freq_padding_needed = target_freq - decoded_output.shape[3]

        if time_padding_needed > 0 or freq_padding_needed > 0:
            padding = (
                0,
                max(freq_padding_needed, 0),
                0,
                max(time_padding_needed, 0),
            )
            decoded_output = F.pad(decoded_output, padding)

        decoded_output = decoded_output[:, :target_channels, :target_time, :target_freq]

        return decoded_output

    def forward(
        self,
        sample: torch.Tensor,
    ) -> torch.Tensor:
        latent_shape = AudioLatentShape(
            batch=sample.shape[0],
            channels=sample.shape[1],
            frames=sample.shape[2],
            mel_bins=sample.shape[3],
        )

        sample_patched = self.patchifier.patchify(sample)
        sample_denormalized = self.per_channel_statistics.denormalize(sample_patched)
        sample = self.patchifier.unpatchify(sample_denormalized, latent_shape)

        target_frames = latent_shape.frames * LATENT_DOWNSAMPLE_FACTOR

        if self.causality_axis is not None:
            target_frames = max(target_frames - (LATENT_DOWNSAMPLE_FACTOR - 1), 1)

        target_shape = AudioLatentShape(
            batch=latent_shape.batch,
            channels=self.out_ch,
            frames=target_frames,
            mel_bins=self.mel_bins if self.mel_bins is not None else latent_shape.mel_bins,
        )

        hidden_features = self.conv_in(sample)
        hidden_features = self._run_mid_layers(hidden_features)
        hidden_features = self._run_upsampling_path(hidden_features)
        decoded_output = self._finalize_output(hidden_features)

        decoded_output = self._adjust_output_shape(decoded_output, target_shape)

        return decoded_output

    def _build_mid_layers(self, channels: int, dropout: float, add_attention: bool) -> nn.Module:
        mid = nn.Module()
        mid.block_1 = LTX2AudioResnetBlock(
            in_channels=channels,
            out_channels=channels,
            temb_channels=self.temb_ch,
            dropout=dropout,
            norm_type=self.norm_type,
            causality_axis=self.causality_axis,
        )
        mid.attn_1 = LTX2AudioAttnBlock(channels, norm_type=self.norm_type) if add_attention else nn.Identity()
        mid.block_2 = LTX2AudioResnetBlock(
            in_channels=channels,
            out_channels=channels,
            temb_channels=self.temb_ch,
            dropout=dropout,
            norm_type=self.norm_type,
            causality_axis=self.causality_axis,
        )
        return mid

    def _build_up_path(
        self,
        initial_block_channels: int,
        dropout: float,
        resamp_with_conv: bool
    ) -> tuple[nn.ModuleList, int]:
        up_modules = nn.ModuleList()
        block_in = initial_block_channels
        curr_res = self.resolution // (2 ** (self.num_resolutions - 1))

        for level in reversed(range(self.num_resolutions)):
            stage = nn.Module()
            stage.block = nn.ModuleList()
            stage.attn = nn.ModuleList()
            block_out = self.base_channels * self.channel_multipliers[level]

            for _ in range(self.num_res_blocks + 1):
                stage.block.append(
                    LTX2AudioResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                        norm_type=self.norm_type,
                        causality_axis=self.causality_axis,
                    )
                )
                block_in = block_out
                if self.attn_resolutions:
                    if curr_res in self.attn_resolutions:
                        stage.attn.append(LTX2AudioAttnBlock(block_in, norm_type=self.norm_type))

            if level != 0:
                stage.upsample = LTX2AudioUpsample(block_in, resamp_with_conv, causality_axis=self.causality_axis)
                curr_res *= 2

            up_modules.insert(0, stage)

        return up_modules, block_in

    def _run_mid_layers(self, features: torch.Tensor) -> torch.Tensor:
        features = self.mid.block_1(features, temb=None)
        features = self.mid.attn_1(features)
        return self.mid.block_2(features, temb=None)

    def _run_upsampling_path(self, features: torch.Tensor) -> torch.Tensor:
        for level in reversed(range(self.num_resolutions)):
            stage = self.up[level]
            for block_idx, block in enumerate(stage.block):
                features = block(features, temb=None)
                if stage.attn:
                    features = stage.attn[block_idx](features)

            if level != 0 and hasattr(stage, "upsample"):
                features = stage.upsample(features)

        return features

    def _finalize_output(self, features: torch.Tensor) -> torch.Tensor:
        if self.give_pre_end:
            return features

        hidden = self.norm_out(features)
        hidden = self.non_linearity(hidden)
        decoded = self.conv_out(hidden)
        return torch.tanh(decoded) if self.tanh_out else decoded


class AutoencoderKLLTX2Audio(ModelMixin, AutoencoderMixin, ConfigMixin):
    r"""
    LTX2 audio VAE. Currently, only implements the decoder.
    """

    _supports_gradient_checkpointing = False

    @register_to_config
    def __init__(
        self,
        base_channels: int = 128,
        output_channels: int = 2,
        ch_mult: Tuple[int] = (1, 2, 4),
        num_res_blocks: int = 2,
        attn_resolutions: Optional[Tuple[int]] = None,
        in_channels: int = 2,
        resolution: int = 256,
        latent_channels: int = 8,
        norm_type: str = "pixel",
        causality_axis: Optional[str] = "height",
        dropout: float = 0.0,
        mid_block_add_attention: bool = False,
        sample_rate: int = 16000,
        mel_hop_length: int = 160,
        is_causal: bool = True,
        mel_bins: Optional[int] = 64,
    ) -> None:
        super().__init__()

        resolved_causality_axis = _resolve_causality_axis(causality_axis)
        attn_resolution_set = set(attn_resolutions) if attn_resolutions else attn_resolutions

        self.decoder = LTX2AudioDecoder(
            base_channels=base_channels,
            output_channels=output_channels,
            ch_mult=ch_mult,
            num_res_blocks=num_res_blocks,
            attn_resolutions=attn_resolution_set,
            in_channels=in_channels,
            resolution=resolution,
            latent_channels=latent_channels,
            norm_type=norm_type,
            causality_axis=resolved_causality_axis,
            dropout=dropout,
            mid_block_add_attention=mid_block_add_attention,
            sample_rate=sample_rate,
            mel_hop_length=mel_hop_length,
            is_causal=is_causal,
            mel_bins=mel_bins,
        )

        self.use_slicing = False

    @apply_forward_hook
    def encode(
        self,
        x: torch.Tensor,
        return_dict: bool = True
    ):
        raise NotImplementedError("AutoencoderKLLTX2Audio does not implement encoding.")

    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    @apply_forward_hook
    def decode(self, z: torch.Tensor, return_dict: bool = True) -> Union[DecoderOutput, torch.Tensor]:
        if self.use_slicing and z.shape[0] > 1:
            decoded_slices = [self._decode(z_slice) for z_slice in z.split(1)]
            decoded = torch.cat(decoded_slices)
        else:
            decoded = self._decode(z)

        if not return_dict:
            return (decoded,)

        return DecoderOutput(sample=decoded)

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "This model doesn't have an encoder yet so we don't implement its `forward()`. Please use `decode()`."
        )
