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

"""Causal 3D K-VAE used by the Kandinsky 6 video super-resolution pipeline."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...configuration_utils import ConfigMixin, register_to_config
from ...utils.accelerate_utils import apply_forward_hook
from ..modeling_outputs import AutoencoderKLOutput
from ..modeling_utils import ModelMixin
from .vae import DecoderOutput, DiagonalGaussianDistribution


# Number of pixel frames encoded or decoded per causal segment. Segmentation only bounds peak memory: the causal
# convolutions carry their padding state from one segment to the next, so the result matches a single-pass call.
SEGMENT_FRAMES = 16
# Element count above which a convolution input is processed in temporal chunks.
CONV_CHUNK_ELEMENTS = 2 * 10**9


class Kandinsky6SRSafeConv3d(nn.Conv3d):
    """`Conv3d` that splits very large inputs along the time axis and convolves them chunk by chunk.

    Each chunk is padded with the last `kernel_size - 1` frames of the previous chunk before convolving, so the result
    is identical to running the convolution on the whole tensor at once; this only bounds peak memory.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.numel() <= CONV_CHUNK_ELEMENTS:
            return super().forward(x)

        kernel_size = self.kernel_size[0]
        chunks = torch.chunk(x, math.ceil(x.numel() / CONV_CHUNK_ELEMENTS), dim=2)
        if kernel_size > 1 and any(chunk.size(2) < kernel_size for chunk in chunks):
            # Chunks shorter than the kernel: slide one window at a time instead.
            if chunks[0].numel() * (kernel_size / chunks[0].size(2)) >= CONV_CHUNK_ELEMENTS:
                raise ValueError("frames are too big for Conv3d")
            stride = self.stride[0]
            windows = range(0, x.size(2) - kernel_size + 1, stride)
            return torch.cat([super().forward(x[:, :, i : i + kernel_size]) for i in windows], dim=2)

        outputs = []
        for i, chunk in enumerate(chunks):
            if i == 0 or kernel_size == 1:
                carried = chunk
            else:
                carried = torch.cat([carried[:, :, -kernel_size + 1 :], chunk], dim=2)
            outputs.append(super().forward(carried))
        return torch.cat(outputs, dim=2)


class Kandinsky6SRCausalConv3d(nn.Module):
    """Causal 3D convolution whose temporal padding is carried across segments through a `cache` dict.

    Height and width are zero-padded symmetrically. Along time the first segment is padded by repeating its first frame
    `kernel_size - 1` times; later segments are padded with the frames the previous segment left behind in
    `cache["padding"]`, so a video processed segment by segment matches a single pass.

    This caching is not an optional performance knob: it is what lets `Kandinsky6SRVAE.encode`/`decode` process an
    arbitrarily long video in bounded-memory segments (see `SEGMENT_FRAMES`) while reproducing the exact output of a
    single non-causal pass. Without it, each segment would need the raw frames the previous segment already consumed in
    order to rebuild correct padding — a lookback window that grows with network depth, rather than the bounded,
    constant-size cache this class carries instead — which defeats the point of processing the video in segments.

    `cache` is mutated in place: the caller passes the same dict to every segment, and this class writes the padding it
    leaves behind directly into `cache["padding"]` before returning.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int],
        stride: tuple[int, int, int] = (1, 1, 1),
    ) -> None:
        super().__init__()
        if not isinstance(kernel_size, tuple):
            kernel_size = (kernel_size,) * 3
        time_kernel_size, height_kernel_size, width_kernel_size = kernel_size
        if not (height_kernel_size % 2 and width_kernel_size % 2):
            raise ValueError(
                f"height_kernel_size and width_kernel_size must be odd, got {height_kernel_size} and "
                f"{width_kernel_size}"
            )
        self.height_pad = height_kernel_size // 2
        self.width_pad = width_kernel_size // 2
        self.time_pad = time_kernel_size - 1
        self.time_kernel_size = time_kernel_size
        self.time_stride = stride[0]
        self.conv = Kandinsky6SRSafeConv3d(in_channels, out_channels, kernel_size, stride=stride)

    @staticmethod
    def make_cache() -> dict:
        return {"padding": None}

    def forward(self, hidden_states: torch.Tensor, cache: dict) -> torch.Tensor:
        batch_size, _, num_frames, height, width = hidden_states.shape
        hidden_states = F.pad(hidden_states, (self.width_pad, self.width_pad, self.height_pad, self.height_pad))

        if cache["padding"] is None:
            first_frame = hidden_states[:, :, :1]
            padding = first_frame.expand(-1, -1, self.time_pad, -1, -1)
        else:
            padding = cache["padding"]

        stride = self.time_stride
        output_frames = (num_frames + 1) // 2 if stride == 2 else num_frames
        output = torch.empty(
            (batch_size, self.conv.out_channels, output_frames, height, width),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        # The frames that overlap the carried padding are convolved together with it; the rest run on their own.
        offset_out = math.ceil(padding.size(2) / stride)
        offset_in = offset_out * stride - padding.size(2)
        if offset_out > 0:
            padded = torch.cat([padding, hidden_states[:, :, : offset_in + self.time_kernel_size - stride]], dim=2)
            output[:, :, :offset_out] = self.conv(padded)
        if offset_out < output_frames:
            output[:, :, offset_out:] = self.conv(hidden_states[:, :, offset_in:])

        # The frames the next segment's first window still needs.
        pad_offset = (
            offset_in + stride * math.trunc((num_frames - offset_in - self.time_kernel_size) / stride) + stride
        )
        if pad_offset < 0:
            cache["padding"] = torch.cat([padding[:, :, pad_offset:], hidden_states], dim=2)
        else:
            cache["padding"] = hidden_states[:, :, pad_offset:].clone()
        return output


class Kandinsky6VAERMSNorm(nn.Module):
    """RMS normalization over the channel axis of a `(B, C, T, H, W)` tensor, computed in float32.

    Same idea as `WanRMS_norm` (`autoencoder_kl_wan.py`) / `QwenImageRMS_norm` (`autoencoder_kl_qwenimage.py`), but not
    a verbatim copy of either, so it is not marked `# Copied from`: it is hardcoded to the channel-first 5D layout used
    throughout this VAE instead of taking `channel_first`/`images` flags, always upcasts to float32 rather than only
    for fp16/bf16/fp8 inputs, and has no learnable bias term.
    """

    def __init__(self, num_channels: int) -> None:
        super().__init__()
        self.scale = num_channels**0.5
        self.gamma = nn.Parameter(torch.ones(num_channels, 1, 1, 1))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        normalized = F.normalize(hidden_states.float(), dim=1).to(hidden_states.dtype)
        return normalized * self.scale * self.gamma


class Kandinsky6SRSpatialNorm3D(nn.Module):
    """Spatial normalization conditioned on the latent `zq` (https://huggingface.co/papers/2209.09002), used here so
    the decoder can inject the latent back in at every resnet block. Structurally this plays the same role as
    `SpatialNorm` (`attention_processor.py`) / `CogVideoXSpatialNorm3D` (`autoencoder_kl_cogvideox.py`) — normalize the
    features, then scale and shift by convolutions of the upsampled `zq`, carrying a `cache` dict across segments the
    same way `CogVideoXSpatialNorm3D` threads its `conv_cache` — but it is not a `# Copied from` of either: it
    normalizes with `Kandinsky6VAERMSNorm` instead of `GroupNorm` (matching the plain-RMSNorm blocks elsewhere in this
    VAE), and interpolates `zq` in channel chunks to bound peak memory.

    `zq` is nearest-upsampled to the feature grid. In the first segment the first frame is upsampled separately,
    because the temporal upsampler turns `T + 1` latent frames into `2T + 1` pixel frames.
    """

    def __init__(self, num_channels: int, zq_channels: int) -> None:
        super().__init__()
        self.norm_layer = Kandinsky6VAERMSNorm(num_channels)
        self.conv_y = Kandinsky6SRSafeConv3d(zq_channels, num_channels, kernel_size=1)
        self.conv_b = Kandinsky6SRSafeConv3d(zq_channels, num_channels, kernel_size=1)

    @staticmethod
    def make_cache() -> dict:
        return {"is_first_segment": True}

    def forward(self, hidden_states: torch.Tensor, zq: torch.Tensor, cache: dict) -> torch.Tensor:
        if cache["is_first_segment"]:
            zq_first = F.interpolate(zq[:, :, :1], size=hidden_states[:, :, :1].shape[-3:], mode="nearest")
            if zq.size(2) > 1:
                # Interpolate in channel chunks to bound the memory of the upsampled conditioning tensor.
                zq_rest = torch.cat(
                    [
                        F.interpolate(split, size=hidden_states[:, :, 1:].shape[-3:], mode="nearest")
                        for split in torch.split(zq[:, :, 1:], 32, dim=1)
                    ],
                    dim=1,
                )
                zq = torch.cat([zq_first, zq_rest], dim=2)
            else:
                zq = zq_first
        else:
            zq = torch.cat(
                [
                    F.interpolate(split, size=hidden_states.shape[-3:], mode="nearest")
                    for split in torch.split(zq, 32, dim=1)
                ],
                dim=1,
            )
        output = self.norm_layer(hidden_states) * self.conv_y(zq) + self.conv_b(zq)
        cache["is_first_segment"] = False
        return output


class Kandinsky6SRResnetBlock3D(nn.Module):
    """Causal residual block; decoder blocks modulate their norms with the latent `zq`.

    `conv1`/`conv2` (`Kandinsky6SRCausalConv3d`) and, for decoder blocks, `norm1`/`norm2` (`Kandinsky6SRSpatialNorm3D`)
    each mutate the matching key of the `cache` dict this block is given in place.
    """

    def __init__(self, in_channels: int, out_channels: int, zq_channels: int | None = None) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if zq_channels is None:
            self.norm1 = Kandinsky6VAERMSNorm(in_channels)
            self.norm2 = Kandinsky6VAERMSNorm(out_channels)
        else:
            self.norm1 = Kandinsky6SRSpatialNorm3D(in_channels, zq_channels)
            self.norm2 = Kandinsky6SRSpatialNorm3D(out_channels, zq_channels)
        self.conv1 = Kandinsky6SRCausalConv3d(in_channels, out_channels, kernel_size=3)
        self.conv2 = Kandinsky6SRCausalConv3d(out_channels, out_channels, kernel_size=3)
        if in_channels != out_channels:
            self.nin_shortcut = Kandinsky6SRSafeConv3d(in_channels, out_channels, kernel_size=1)

    def make_cache(self) -> dict:
        cache = {"conv1": self.conv1.make_cache(), "conv2": self.conv2.make_cache()}
        if isinstance(self.norm1, Kandinsky6SRSpatialNorm3D):
            cache["norm1"] = self.norm1.make_cache()
            cache["norm2"] = self.norm2.make_cache()
        return cache

    def forward(self, hidden_states: torch.Tensor, cache: dict, zq: torch.Tensor | None = None) -> torch.Tensor:
        residual = hidden_states
        if zq is None:
            hidden_states = self.norm1(hidden_states)
        else:
            hidden_states = self.norm1(hidden_states, zq, cache["norm1"])
        hidden_states = self.conv1(F.silu(hidden_states), cache["conv1"])
        if zq is None:
            hidden_states = self.norm2(hidden_states)
        else:
            hidden_states = self.norm2(hidden_states, zq, cache["norm2"])
        hidden_states = self.conv2(F.silu(hidden_states), cache["conv2"])
        if self.in_channels != self.out_channels:
            residual = self.nin_shortcut(residual)
        return residual + hidden_states


class Kandinsky6SRDownsample(nn.Module):
    """Spatial 2x downsample (strided conv plus pixel-unshuffle average) with an optional causal temporal 2x.

    See `Kandinsky6SRUpsample` for the upsampling counterpart.
    """

    def __init__(self, in_channels: int, compress_time: bool) -> None:
        super().__init__()
        out_channels = 2 * in_channels
        self.compress_time = compress_time
        self.unshuffle = nn.PixelUnshuffle(2)
        self.spatial_conv = Kandinsky6SRSafeConv3d(
            in_channels, out_channels, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)
        )
        if compress_time:
            self.temporal_conv = nn.ModuleList(
                [
                    Kandinsky6SRCausalConv3d(out_channels, out_channels, kernel_size=(2, 1, 1), stride=(1, 1, 1)),
                    Kandinsky6SRCausalConv3d(out_channels, out_channels, kernel_size=(2, 1, 1), stride=(2, 1, 1)),
                ]
            )
        self.linear = Kandinsky6SRSafeConv3d(out_channels, out_channels, kernel_size=1)

    def make_cache(self) -> dict:
        if not self.compress_time:
            return {}
        return {"temporal_conv": [conv.make_cache() for conv in self.temporal_conv]}

    def forward(self, hidden_states: torch.Tensor, cache: dict) -> torch.Tensor:
        batch_size, channels, num_frames, height, width = hidden_states.shape

        # Spatial: pixel-unshuffle, average pairs of sub-pixel channels, add the strided convolution.
        frames = hidden_states.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, channels, height, width)
        frames = self.unshuffle(frames)
        frames = frames.view(batch_size * num_frames, 2 * channels, 2, height // 2, width // 2).mean(dim=2)
        frames = frames.view(batch_size, num_frames, 2 * channels, height // 2, width // 2).permute(0, 2, 1, 3, 4)
        hidden_states = self.spatial_conv(hidden_states) + frames

        if not self.compress_time:
            return self.linear(hidden_states)

        # Temporal: average pairs of frames (the first segment keeps its first frame), add the causal convolutions.
        batch_size, channels, num_frames, height, width = hidden_states.shape
        sequence = hidden_states.permute(0, 3, 4, 1, 2).reshape(-1, channels, num_frames)
        if cache["temporal_conv"][0]["padding"] is None:
            first, rest = sequence[..., :1], sequence[..., 1:]
            pooled = (
                torch.cat([first, F.avg_pool1d(rest, kernel_size=2, stride=2)], dim=-1) if rest.size(-1) else first
            )
        else:
            pooled = F.avg_pool1d(sequence, kernel_size=2, stride=2)
        pooled = pooled.reshape(batch_size, height, width, channels, -1).permute(0, 3, 4, 1, 2)

        conv_out = self.temporal_conv[0](hidden_states, cache["temporal_conv"][0])
        conv_out = self.temporal_conv[1](conv_out, cache["temporal_conv"][1])
        return self.linear(conv_out + pooled)


class Kandinsky6SRUpsample(nn.Module):
    """Spatial 2x nearest upsample with a convolutional residual, preceded by an optional causal temporal 2x.

    See `Kandinsky6SRDownsample` for the downsampling counterpart.
    """

    def __init__(self, channels: int, compress_time: bool) -> None:
        super().__init__()
        self.compress_time = compress_time
        self.spatial_conv = Kandinsky6SRSafeConv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        if compress_time:
            self.temporal_conv = Kandinsky6SRCausalConv3d(channels, channels, kernel_size=(3, 1, 1))
        self.linear = Kandinsky6SRSafeConv3d(channels, channels, kernel_size=1)

    def make_cache(self) -> dict:
        return {"temporal_conv": self.temporal_conv.make_cache()} if self.compress_time else {}

    def forward(self, hidden_states: torch.Tensor, cache: dict) -> torch.Tensor:
        if self.compress_time:
            # `T + 1` frames become `2T + 1`: every frame is repeated and the first segment drops the extra copy of
            # its first frame.
            repeated = hidden_states.repeat_interleave(2, dim=2)
            if cache["temporal_conv"]["padding"] is None:
                repeated = repeated[:, :, 1:]
            conv_out = self.temporal_conv(repeated, cache["temporal_conv"])
            hidden_states = conv_out + repeated

        hidden_states = F.interpolate(hidden_states, scale_factor=(1, 2, 2), mode="nearest")
        hidden_states = hidden_states + self.spatial_conv(hidden_states)
        return self.linear(hidden_states)


class Kandinsky6SREncoder3D(nn.Module):
    """Causal encoder: `conv_in`, downsampling resnet levels, a resnet bottleneck, then `norm_out`/`conv_out`.

    Each submodule mutates the matching key of the `cache` dict it is given in place. Because the caller
    (`Kandinsky6SRVAE.encode`) passes the same `cache` dict to every segment, that in-place mutation is what carries
    the padding state from one segment to the next.
    """

    def __init__(
        self,
        in_channels: int,
        latent_channels: int,
        block_out_channels: tuple[int, ...],
        layers_per_block: int,
        temporal_compression_ratio: int,
        temporal_compression_start_level: int,
    ) -> None:
        super().__init__()
        num_levels = len(block_out_channels)
        temporal_compression_end_level = int(math.log2(temporal_compression_ratio)) + temporal_compression_start_level

        self.conv_in = Kandinsky6SRCausalConv3d(in_channels, block_out_channels[0], kernel_size=3)
        self.down = nn.ModuleList()
        for level in range(num_levels):
            # Every downsample doubles the channel count, so the next level starts at twice the previous width.
            block_in = block_out_channels[0] if level == 0 else 2 * block_out_channels[level - 1]
            block_out = block_out_channels[level]
            blocks = nn.ModuleList()
            for _ in range(layers_per_block):
                blocks.append(Kandinsky6SRResnetBlock3D(block_in, block_out))
                block_in = block_out
            down = nn.Module()
            down.block = blocks
            if level != num_levels - 1:
                compress_time = temporal_compression_start_level <= level < temporal_compression_end_level
                down.downsample = Kandinsky6SRDownsample(block_in, compress_time=compress_time)
            self.down.append(down)

        self.mid = nn.Module()
        self.mid.block_1 = Kandinsky6SRResnetBlock3D(block_in, block_in)
        self.mid.block_2 = Kandinsky6SRResnetBlock3D(block_in, block_in)
        self.norm_out = Kandinsky6VAERMSNorm(block_in)
        self.conv_out = Kandinsky6SRCausalConv3d(block_in, 2 * latent_channels, kernel_size=3)

    def make_cache(self) -> dict:
        return {
            "conv_in": self.conv_in.make_cache(),
            "down": [
                {
                    "block": [block.make_cache() for block in down.block],
                    "downsample": down.downsample.make_cache() if hasattr(down, "downsample") else {},
                }
                for down in self.down
            ],
            "mid_1": self.mid.block_1.make_cache(),
            "mid_2": self.mid.block_2.make_cache(),
            "conv_out": self.conv_out.make_cache(),
        }

    def forward(self, hidden_states: torch.Tensor, cache: dict) -> torch.Tensor:
        hidden_states = self.conv_in(hidden_states, cache["conv_in"])
        for down, level_cache in zip(self.down, cache["down"]):
            for i, block in enumerate(down.block):
                hidden_states = block(hidden_states, level_cache["block"][i])
            if hasattr(down, "downsample"):
                hidden_states = down.downsample(hidden_states, level_cache["downsample"])

        hidden_states = self.mid.block_1(hidden_states, cache["mid_1"])
        hidden_states = self.mid.block_2(hidden_states, cache["mid_2"])

        hidden_states = F.silu(self.norm_out(hidden_states))
        hidden_states = self.conv_out(hidden_states, cache["conv_out"])
        return hidden_states


class Kandinsky6SRDecoder3D(nn.Module):
    """Causal decoder: `conv_in`, a `zq`-conditioned resnet bottleneck, upsampling resnet levels, then
    `norm_out`/`conv_out`.

    Like `Kandinsky6SREncoder3D`, each submodule mutates the matching key of the `cache` dict it is given in place,
    which is what carries the padding state across the segments `Kandinsky6SRVAE.decode` passes it.
    """

    def __init__(
        self,
        out_channels: int,
        latent_channels: int,
        block_out_channels: tuple[int, ...],
        layers_per_block: int,
        temporal_compression_ratio: int,
        temporal_compression_start_level: int,
    ) -> None:
        super().__init__()
        num_levels = len(block_out_channels)
        temporal_compression_end_level = int(math.log2(temporal_compression_ratio)) + temporal_compression_start_level

        block_in = block_out_channels[-1]
        self.conv_in = Kandinsky6SRCausalConv3d(latent_channels, block_in, kernel_size=3)
        self.mid = nn.Module()
        self.mid.block_1 = Kandinsky6SRResnetBlock3D(block_in, block_in, zq_channels=latent_channels)
        self.mid.block_2 = Kandinsky6SRResnetBlock3D(block_in, block_in, zq_channels=latent_channels)

        self.up = nn.ModuleList()
        for level in reversed(range(num_levels)):
            block_out = block_out_channels[level]
            blocks = nn.ModuleList()
            for _ in range(layers_per_block + 1):
                blocks.append(Kandinsky6SRResnetBlock3D(block_in, block_out, zq_channels=latent_channels))
                block_in = block_out
            up = nn.Module()
            up.block = blocks
            if level != 0:
                compress_time = (
                    num_levels - temporal_compression_start_level
                    > level
                    >= num_levels - temporal_compression_end_level
                )
                up.upsample = Kandinsky6SRUpsample(block_in, compress_time=compress_time)
            self.up.insert(0, up)

        self.norm_out = Kandinsky6SRSpatialNorm3D(block_in, latent_channels)
        self.conv_out = Kandinsky6SRCausalConv3d(block_in, out_channels, kernel_size=3)

    def make_cache(self) -> dict:
        return {
            "conv_in": self.conv_in.make_cache(),
            "mid_1": self.mid.block_1.make_cache(),
            "mid_2": self.mid.block_2.make_cache(),
            "up": [
                {
                    "block": [block.make_cache() for block in up.block],
                    "upsample": up.upsample.make_cache() if hasattr(up, "upsample") else {},
                }
                for up in self.up
            ],
            "norm_out": self.norm_out.make_cache(),
            "conv_out": self.conv_out.make_cache(),
        }

    def forward(self, latents: torch.Tensor, cache: dict) -> torch.Tensor:
        hidden_states = self.conv_in(latents, cache["conv_in"])

        hidden_states = self.mid.block_1(hidden_states, cache["mid_1"], zq=latents)
        hidden_states = self.mid.block_2(hidden_states, cache["mid_2"], zq=latents)

        for level in reversed(range(len(self.up))):
            up, level_cache = self.up[level], cache["up"][level]
            for i, block in enumerate(up.block):
                hidden_states = block(hidden_states, level_cache["block"][i], zq=latents)
            if hasattr(up, "upsample"):
                hidden_states = up.upsample(hidden_states, level_cache["upsample"])

        hidden_states = self.norm_out(hidden_states, latents, cache["norm_out"])
        hidden_states = F.silu(hidden_states)
        hidden_states = self.conv_out(hidden_states, cache["conv_out"])
        return hidden_states


class Kandinsky6SRVAE(ModelMixin, ConfigMixin):
    r"""
    Causal 3D K-VAE used by [`Kandinsky6SRPipeline`] to encode and decode video.

    Videos are processed in temporal segments of 16 pixel frames (plus the leading frame). The causal convolutions
    carry their padding state between segments, so the segmentation only bounds peak memory and does not change the
    result.

    Args:
        in_channels (`int`, defaults to `3`):
            Number of pixel channels.
        out_channels (`int`, defaults to `3`):
            Number of reconstructed pixel channels.
        latent_channels (`int`, defaults to `64`):
            Number of latent channels.
        encoder_block_out_channels (`tuple[int, ...]`, defaults to `(16, 128, 256, 512, 1024)`):
            Output width of the residual blocks at each encoder level; every level but the last halves the spatial size
            and doubles the width on the way to the next level.
        decoder_block_out_channels (`tuple[int, ...]`, defaults to `(16, 256, 512, 1024, 2048)`):
            Output width of the residual blocks at each decoder level.
        layers_per_block (`int`, defaults to `2`):
            Number of residual blocks per encoder level; the decoder uses one more per level.
        temporal_compression_ratio (`int`, defaults to `4`):
            Temporal compression factor; `log2` of it consecutive levels also compress time.
        temporal_compression_start_level (`int`, defaults to `1`):
            First level that compresses time.
        scaling_factor (`float`, defaults to `0.910344`):
            Scale applied to the latents before they enter the diffusion transformer.
    """

    _no_split_modules = ["Kandinsky6SREncoder3D", "Kandinsky6SRDecoder3D"]

    @register_to_config
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        latent_channels: int = 64,
        encoder_block_out_channels: tuple[int, ...] = (16, 128, 256, 512, 1024),
        decoder_block_out_channels: tuple[int, ...] = (16, 256, 512, 1024, 2048),
        layers_per_block: int = 2,
        temporal_compression_ratio: int = 4,
        temporal_compression_start_level: int = 1,
        scaling_factor: float = 0.910344004631042,
    ) -> None:
        super().__init__()
        if len(encoder_block_out_channels) != len(decoder_block_out_channels):
            raise ValueError("`encoder_block_out_channels` and `decoder_block_out_channels` must have the same length")

        self.encoder = Kandinsky6SREncoder3D(
            in_channels=in_channels,
            latent_channels=latent_channels,
            block_out_channels=encoder_block_out_channels,
            layers_per_block=layers_per_block,
            temporal_compression_ratio=temporal_compression_ratio,
            temporal_compression_start_level=temporal_compression_start_level,
        )
        self.decoder = Kandinsky6SRDecoder3D(
            out_channels=out_channels,
            latent_channels=latent_channels,
            block_out_channels=decoder_block_out_channels,
            layers_per_block=layers_per_block,
            temporal_compression_ratio=temporal_compression_ratio,
            temporal_compression_start_level=temporal_compression_start_level,
        )

        self.spatial_compression_ratio = 2 ** (len(encoder_block_out_channels) - 1)
        self.temporal_compression_ratio = temporal_compression_ratio

    @apply_forward_hook
    def encode(self, x: torch.Tensor, return_dict: bool = True) -> AutoencoderKLOutput | tuple:
        r"""
        Encode a video into its latent distribution.

        Args:
            x (`torch.Tensor` of shape `(batch_size, channels, num_frames, height, width)`):
                Pixel video in `[-1, 1]`. `num_frames` should be `1 + k * temporal_compression_ratio`.
            return_dict (`bool`, defaults to `True`):
                Whether to return an [`~models.modeling_outputs.AutoencoderKLOutput`] instead of a plain tuple.
        """
        cache = self.encoder.make_cache()
        segment_lengths = [min(SEGMENT_FRAMES + 1, x.size(2))]
        remaining = x.size(2) - segment_lengths[0]
        while remaining > 0:
            segment_lengths.append(min(SEGMENT_FRAMES, remaining))
            remaining -= SEGMENT_FRAMES

        moments = torch.cat(
            [self.encoder(segment, cache) for segment in torch.split(x, segment_lengths, dim=2)], dim=2
        )
        posterior = DiagonalGaussianDistribution(moments)
        if not return_dict:
            return (posterior,)
        return AutoencoderKLOutput(latent_dist=posterior)

    @apply_forward_hook
    def decode(self, z: torch.Tensor, return_dict: bool = True) -> DecoderOutput | tuple:
        r"""
        Decode latents into a video.

        Args:
            z (`torch.Tensor` of shape `(batch_size, latent_channels, num_latent_frames, height, width)`):
                Latents, already divided by `scaling_factor`.
            return_dict (`bool`, defaults to `True`):
                Whether to return a [`~models.autoencoder_kl.DecoderOutput`] instead of a plain tuple.
        """
        cache = self.decoder.make_cache()
        latent_segment = SEGMENT_FRAMES // self.temporal_compression_ratio
        num_latent_frames = z.size(2)
        if num_latent_frames == 1:
            segment_lengths = [1]
        else:
            # The leading latent frame decodes to a single pixel frame; every following latent frame decodes to
            # `temporal_compression_ratio` pixel frames.
            segment_lengths = [latent_segment] * ((num_latent_frames - 1) // latent_segment)
            if (num_latent_frames - 1) % latent_segment:
                segment_lengths.append((num_latent_frames - 1) % latent_segment)
            segment_lengths[0] += 1

        decoded = torch.cat(
            [self.decoder(segment, cache) for segment in torch.split(z, segment_lengths, dim=2)], dim=2
        )
        if not return_dict:
            return (decoded,)
        return DecoderOutput(sample=decoded)

    def forward(
        self,
        sample: torch.Tensor,
        sample_posterior: bool = False,
        return_dict: bool = True,
        generator: torch.Generator | None = None,
    ) -> DecoderOutput | tuple:
        r"""
        Args:
            sample (`torch.Tensor` of shape `(batch_size, channels, num_frames, height, width)`):
                Pixel video in `[-1, 1]`. `num_frames` should be `1 + k * temporal_compression_ratio`.
            sample_posterior (`bool`, *optional*, defaults to `False`):
                Whether to sample from the latent posterior instead of using its mode.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`~models.autoencoder_kl.DecoderOutput`] instead of a plain tuple.
            generator (`torch.Generator`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make sampling
                deterministic.

        Returns:
            [`~models.autoencoder_kl.DecoderOutput`] or `tuple`:
                If `return_dict` is True, a [`~models.autoencoder_kl.DecoderOutput`] is returned, otherwise a plain
                `tuple` is returned. Its `sample` is the reconstructed video.
        """
        posterior = self.encode(sample).latent_dist
        z = posterior.sample(generator=generator) if sample_posterior else posterior.mode()
        decoded = self.decode(z).sample
        if not return_dict:
            return (decoded,)
        return DecoderOutput(sample=decoded)
