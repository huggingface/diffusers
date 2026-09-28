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

"""Kandinsky 6 SR KVAE Diffusers component."""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...configuration_utils import ConfigMixin, register_to_config
from ...utils.accelerate_utils import apply_forward_hook
from ..modeling_utils import ModelMixin
from .vae import DiagonalGaussianDistribution


class SafeConv3d(nn.Conv3d):
    """Conv3d that splits its input along the time axis when it would otherwise materialize more than ~2B
    elements, running the convolution chunk by chunk instead. Each chunk is padded with the last
    ``kernel_size - 1`` frames of the previous chunk before convolving, so the result is identical to running
    the convolution on the whole tensor at once; this only bounds peak memory.
    """

    def forward(self, x, write_to=None, transform=None):
        if transform is None:

            def transform(x):
                return x

        memory_count = x.numel() / (10**9)
        if memory_count > 2:
            kernel_size = self.kernel_size[0]
            part_num = math.ceil(memory_count / 2)
            input_chunks = torch.chunk(x, part_num, dim=2)  # NCTHW

            if any(ch.size(2) < kernel_size for ch in input_chunks) and kernel_size > 1:
                if input_chunks[0].numel() * (kernel_size / input_chunks[0].size(2)) >= (2 * 10**9):
                    raise ValueError("frames are too big for Conv3d")

                t_stride, output = self.stride[0], []
                for i in range(0, x.size(2) - kernel_size + 1, t_stride):
                    chunk = transform(x[:, :, i : i + kernel_size])
                    output.append(super(SafeConv3d, self).forward(chunk))
                output = torch.cat(output, dim=2)
                return output

            if write_to is None:
                output = []
                for i, chunk in enumerate(input_chunks):
                    if i == 0 or kernel_size == 1:
                        z = torch.clone(chunk)
                    else:
                        z = torch.cat([z[:, :, -kernel_size + 1 :], chunk], dim=2)
                    output.append(super(SafeConv3d, self).forward(transform(z)))
                output = torch.cat(output, dim=2)
                return output
            else:
                time_offset = 0
                for i, chunk in enumerate(input_chunks):
                    if i == 0 or kernel_size == 1:
                        z = torch.clone(chunk)
                    else:
                        z = torch.cat([z[:, :, -kernel_size + 1 :], chunk], dim=2)
                    z_time = z.size(2) - (kernel_size - 1)
                    write_to[:, :, time_offset : time_offset + z_time] = super(SafeConv3d, self).forward(transform(z))
                    time_offset += z_time
                return write_to
        else:
            if write_to is None:
                return super(SafeConv3d, self).forward(transform(x))
            else:
                write_to[...] = super(SafeConv3d, self).forward(transform(x))
                return write_to


class CausalConv3d(nn.Module):
    def __init__(
        self,
        chan_in,
        chan_out,
        kernel_size: int | tuple[int, int, int],
        stride=(1, 1, 1),
        dilation=(1, 1, 1),
        padding_mode=None,
        **kwargs,
    ):
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
        self.temporal_dim = 2

        self.stride = stride
        self.conv = SafeConv3d(chan_in, chan_out, kernel_size, stride=stride, dilation=dilation, **kwargs)
        self.cache_padding = None
        self.padding_mode = padding_mode

    def forward(self, input_):
        input_parallel = input_

        padding_3d = (self.width_pad, self.width_pad, self.height_pad, self.height_pad, self.time_pad, 0)
        input_parallel = F.pad(input_parallel, padding_3d, mode=self.padding_mode or "replicate")

        output = self.conv(input_parallel)
        return output


class Kandinsky6VAERMSNorm(nn.Module):
    r"""
    RMS normalization over the channel dimension of a video tensor (`N, C, T, H, W`).

    `forward` accepts and ignores extra positional/keyword arguments (e.g. the causal-cache dict the
    encoder/decoder pass to every normalization layer) so this stays a drop-in alternative to
    `CachedGroupNorm`; construction likewise ignores `zq_ch`/`add_conv`, which only apply to the
    `CachedSpatialNorm3D` alternative.

    Args:
        in_channels (int): The number of channels to normalize over.
        bias (bool, optional): Whether to include a learnable bias term. Default is False.
    """

    def __init__(self, in_channels: int, bias: bool = False, **kwargs) -> None:
        super().__init__()
        shape = (in_channels, 1, 1, 1)

        self.scale = in_channels**0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.0

    def forward(self, x, *args, **kwargs):
        needs_fp32_normalize = x.dtype in (torch.float16, torch.bfloat16) or any(
            t in str(x.dtype) for t in ("float4_", "float8_")
        )
        normalized = F.normalize(x.float() if needs_fp32_normalize else x, dim=1).to(x.dtype)
        return normalized * self.scale * self.gamma + self.bias


class CachedGroupNorm(nn.GroupNorm):
    """GroupNorm alternative to `Kandinsky6VAERMSNorm`: same call convention (ignores `zq_ch`/`add_conv`,
    accepts a `cache` dict) so the encoder/decoder can pick either one through the same `normalization(...)`
    call. `cache` only ever records whether this layer has run before — `CachedSpatialNorm3D` reads that
    presence flag to decide whether it's normalizing the first causal segment or a later one; the statistics
    themselves are not reused across calls.
    """

    def __init__(self, in_channels, **kwargs):
        super().__init__(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

    def forward(self, x, cache: dict):
        out = super().forward(x)
        if cache.get("mean") is None and cache.get("var") is None:
            cache["mean"] = 1
            cache["var"] = 1
        return out


class CachedCausalConv3d(CausalConv3d):
    def forward(self, input_, cache: dict, fixed_stride=False):
        t_stride = self.stride[0]
        padding_3d = (self.height_pad, self.height_pad, self.width_pad, self.width_pad, 0, 0)
        input_parallel = F.pad(
            input_,
            padding_3d,
            mode="constant" if self.padding_mode == "zeros" else (self.padding_mode or "replicate"),
            value=0 if self.padding_mode in ["constant", "zeros"] else None,
        )

        if cache["padding"] is None:
            first_frame = input_parallel[:, :, :1]
            time_pad_shape = list(first_frame.shape)
            time_pad_shape[2] = self.time_pad
            padding = first_frame.expand(time_pad_shape)
        else:
            padding = cache["padding"]

        out_size = list(input_.shape)
        out_size[1] = self.conv.out_channels
        if t_stride == 2:
            out_size[2] = (input_.size(2) + 1) // 2
        output = torch.empty(tuple(out_size), dtype=input_.dtype, device=input_.device)

        offset_out = math.ceil(
            padding.size(2) / t_stride
        )  # forward on `padding_poisoned` should take exactly this range
        offset_in = offset_out * t_stride - padding.size(
            2
        )  # to make forward on `input_parallel` take slice starting with this index

        if offset_out > 0:
            padding_poisoned = torch.cat(
                [padding, input_parallel[:, :, : offset_in + self.time_kernel_size - t_stride]], dim=2
            )
            output[:, :, :offset_out] = self.conv(padding_poisoned)

        if offset_out < output.size(2):
            output[:, :, offset_out:] = self.conv(input_parallel[:, :, offset_in:])

        if t_stride == 2 and not fixed_stride and cache["padding"] is not None:
            expected_pad_size = padding.size(2) - 1
            offset_out = math.ceil(expected_pad_size / t_stride)
            offset_in = offset_out * t_stride - expected_pad_size

        # exact formula, doesn't depend on size of segments
        pad_offset = (
            offset_in
            + t_stride * math.trunc((input_parallel.size(2) - offset_in - self.time_kernel_size) / t_stride)
            + t_stride
        )

        # this condition will be executed ONLY for old models
        if t_stride == 2 and not fixed_stride:
            pad_offset -= 1

        if pad_offset < 0:  # specific to small chunks (for inference on high resolution videos)
            cache["padding"] = torch.cat([padding[:, :, pad_offset:], input_parallel], dim=2)
        else:
            cache["padding"] = torch.clone(input_parallel[:, :, pad_offset:])

        return output


class CachedCausalResnetBlock3D(nn.Module):
    def __init__(
        self,
        *,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        dropout,
        temb_channels=512,
        zq_ch=None,
        add_conv=False,
        normalization=CachedGroupNorm,
        padding_mode=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = normalization(in_channels, zq_ch=zq_ch, add_conv=add_conv)

        self.conv1 = CachedCausalConv3d(
            chan_in=in_channels, chan_out=out_channels, kernel_size=3, padding_mode=padding_mode
        )
        self.norm2 = normalization(out_channels, zq_ch=zq_ch, add_conv=add_conv)
        self.conv2 = CachedCausalConv3d(
            chan_in=out_channels, chan_out=out_channels, kernel_size=3, padding_mode=padding_mode
        )
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = CachedCausalConv3d(
                    chan_in=in_channels, chan_out=out_channels, kernel_size=3, padding_mode=padding_mode
                )
            else:
                self.nin_shortcut = SafeConv3d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )

    def forward(self, x, temb, layer_cache, zq=None):
        h = x

        if zq is None:
            h = self.norm1(h, cache=layer_cache["norm1"])
        else:
            h = self.norm1(h, zq, cache=layer_cache["norm1"])

        h = F.silu(h, inplace=True)
        h = self.conv1(h, cache=layer_cache["conv1"])

        if zq is None:
            h = self.norm2(h, cache=layer_cache["norm2"])
        else:
            h = self.norm2(h, zq, cache=layer_cache["norm2"])

        h = F.silu(h, inplace=True)
        h = self.conv2(h, cache=layer_cache["conv2"])

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x, cache=layer_cache["conv_shortcut"])
            else:
                x = self.nin_shortcut(x)

        return x + h


class CachedPXSDownsample(nn.Module):
    def __init__(
        self, in_channels: int, compress_time: bool, factor: int = 2, version=1, fixed_stride=False, padding_mode=None
    ):
        super().__init__()
        self.temporal_compress = compress_time
        self.fixed_stride = fixed_stride
        self.factor = factor
        self.unshuffle = nn.PixelUnshuffle(self.factor)
        self.s_pool = nn.AvgPool3d((1, 2, 2), (1, 2, 2))

        self.version = version
        out_channels = in_channels * 2 if version > 1 else in_channels

        self.spatial_conv = SafeConv3d(
            in_channels,
            out_channels,
            kernel_size=(1, 3, 3),
            stride=(1, 2, 2),
            padding=(0, 1, 1),
            padding_mode=padding_mode or "reflect",
        )

        if self.temporal_compress:
            if version == 2:
                self.temporal_conv = nn.Sequential(
                    CachedCausalConv3d(
                        out_channels,
                        out_channels,
                        kernel_size=(2, 1, 1),
                        stride=(1, 1, 1),
                        dilation=(1, 1, 1),
                        padding_mode=padding_mode,
                    ),
                    CachedCausalConv3d(
                        out_channels,
                        out_channels,
                        kernel_size=(2, 1, 1),
                        stride=(2, 1, 1),
                        dilation=(1, 1, 1),
                        padding_mode=padding_mode,
                    ),
                )
            else:  # 1 or 3
                self.temporal_conv = CachedCausalConv3d(
                    out_channels,
                    out_channels,
                    kernel_size=(3, 1, 1),
                    stride=(2, 1, 1),
                    dilation=(1, 1, 1),
                    padding_mode=padding_mode,
                )

        self.linear = SafeConv3d(out_channels, out_channels, kernel_size=1, stride=1)

    def spatial_downsample(self, input_):
        # PixelShuffle part
        pxs_input = input_.permute(0, 2, 1, 3, 4).reshape(-1, input_.shape[1], input_.shape[3], input_.shape[4])
        pxs_interm = self.unshuffle(pxs_input)
        b, c, h, w = pxs_interm.shape
        if self.version > 1:
            pxs_interm_view = pxs_interm.view(b, c // self.factor, self.factor, h, w)
        else:  #
            pxs_interm_view = pxs_interm.view(b, c // self.factor**2, self.factor**2, h, w)
        pxs_out = torch.mean(pxs_interm_view, dim=2)
        pxs_out = pxs_out.reshape(
            input_.shape[0], input_.size(2), pxs_out.shape[1], pxs_out.shape[2], pxs_out.shape[3]
        ).permute(0, 2, 1, 3, 4)

        # Downsampling by 3D-convolution
        conv_out = self.spatial_conv(input_)

        # adding it all together
        return conv_out + pxs_out

    def temporal_downsample(self, input_, cache):
        # Interpolation part
        permuted = input_.permute(0, 3, 4, 1, 2).reshape(-1, input_.shape[1], input_.shape[2])
        if cache[0]["padding"] is None:
            first, rest = permuted[..., :1], permuted[..., 1:]

            if rest.size(-1) > 0:
                rest_interp = F.avg_pool1d(rest, kernel_size=2, stride=2)
                full_interp = torch.cat([first, rest_interp], dim=-1)
            else:
                full_interp = first
        else:
            rest = permuted
            if rest.size(-1) > 0:
                full_interp = F.avg_pool1d(rest, kernel_size=2, stride=2)

        full_interp = full_interp.reshape(
            input_.shape[0], input_.size(-2), input_.size(-1), full_interp.shape[1], full_interp.shape[2]
        ).permute(0, 3, 4, 1, 2)

        # Downsampling by convolution
        if self.version == 1:
            conv_out = self.temporal_conv(input_, cache[0], fixed_stride=self.fixed_stride)
        elif self.version == 2:
            conv_out = self.temporal_conv[0](input_, cache[0], fixed_stride=self.fixed_stride)
            conv_out = self.temporal_conv[1](conv_out, cache[1], fixed_stride=self.fixed_stride)

        return conv_out + full_interp

    def forward(self, x, cache):
        # SPATIAL DOWNSAMPLE
        out = self.spatial_downsample(x)

        if self.temporal_compress:
            # TEMPORAL DOWNSAMPLE
            out = self.temporal_downsample(out, cache=cache)

        return self.linear(out)


class CachedSpatialNorm3D(nn.Module):
    """Normalization used by the decoder's ResnetBlocks: modulates a GroupNorm/RMSNorm output with a
    per-channel scale and shift derived from the latent `zq`, the same "spatial norm" conditioning other VAEs
    in the library (e.g. CogVideoX) use to inject the encoder's output back into the decoder — adapted here
    for causal, segment-by-segment video decoding.
    """

    def __init__(
        self,
        f_channels,
        zq_channels,
        freeze_norm_layer=False,
        add_conv=False,
        padding_mode=None,
        normalization=CachedGroupNorm,
        **norm_layer_params,
    ):
        super().__init__()
        self.norm_layer = normalization(in_channels=f_channels, **norm_layer_params)

        self.add_conv = add_conv
        if add_conv:
            self.conv = CachedCausalConv3d(
                chan_in=zq_channels, chan_out=zq_channels, kernel_size=3, padding_mode=padding_mode
            )

        self.conv_y = SafeConv3d(
            zq_channels,
            f_channels,
            kernel_size=1,
        )
        self.conv_b = SafeConv3d(
            zq_channels,
            f_channels,
            kernel_size=1,
        )

    def forward(self, f, zq, cache):
        if cache["norm"]["mean"] is None and cache["norm"]["var"] is None:
            f_first, f_rest = f[:, :, :1], f[:, :, 1:]
            f_first_size, f_rest_size = f_first.shape[-3:], f_rest.shape[-3:]
            zq_first, zq_rest = zq[:, :, :1], zq[:, :, 1:]

            zq_first = F.interpolate(zq_first, size=f_first_size, mode="nearest")

            if zq.size(2) > 1:
                zq_rest_splits = torch.split(zq_rest, 32, dim=1)
                interpolated_splits = [
                    F.interpolate(split, size=f_rest_size, mode="nearest") for split in zq_rest_splits
                ]

                zq_rest = torch.cat(interpolated_splits, dim=1)
                zq = torch.cat([zq_first, zq_rest], dim=2)
            else:
                zq = zq_first
        else:
            f_size = f.shape[-3:]
            zq_splits = torch.split(zq, 32, dim=1)
            interpolated_splits = [F.interpolate(split, size=f_size, mode="nearest") for split in zq_splits]
            zq = torch.cat(interpolated_splits, dim=1)

        if self.add_conv:
            zq = self.conv(zq, cache["add_conv"])

        norm_f = self.norm_layer(f, cache["norm"])
        norm_f.mul_(self.conv_y(zq))
        norm_f.add_(self.conv_b(zq))

        if cache["norm"]["mean"] is None and cache["norm"]["var"] is None:
            cache["norm"]["mean"] = 1
            cache["norm"]["var"] = 1

        return norm_f


def Normalize3D(in_channels, zq_ch, add_conv, normalization=CachedGroupNorm):
    return CachedSpatialNorm3D(
        in_channels,
        zq_ch,
        freeze_norm_layer=False,
        add_conv=add_conv,
        num_groups=32,
        eps=1e-6,
        affine=True,
        normalization=normalization,
    )


class CachedPXSUpsample(nn.Module):
    def __init__(self, in_channels: int, compress_time: bool, factor: int = 2, padding_mode=None):
        super().__init__()
        self.temporal_compress = compress_time
        self.factor = factor
        self.shuffle = nn.PixelShuffle(self.factor)
        self.spatial_conv = SafeConv3d(
            in_channels,
            in_channels,
            kernel_size=(1, 3, 3),
            stride=(1, 1, 1),
            padding=(0, 1, 1),
            padding_mode=padding_mode or "reflect",
        )

        if self.temporal_compress:
            self.temporal_conv = CachedCausalConv3d(
                in_channels,
                in_channels,
                kernel_size=(3, 1, 1),
                stride=(1, 1, 1),
                dilation=(1, 1, 1),
                padding_mode=padding_mode,
            )

        self.linear = SafeConv3d(in_channels, in_channels, kernel_size=1, stride=1)

    def spatial_upsample_NEW(self, input_):
        def conv_part(x):
            to = torch.empty_like(x)
            out = self.spatial_conv(x, write_to=to)
            return out

        # 5D interpolate keeps channels_last_3d layout intact; merging dims
        # via view() is impossible for channels_last strides (dim 1 is innermost)
        input_interp = F.interpolate(input_, scale_factor=(1, 2, 2), mode="nearest")
        input_interp.add_(conv_part(input_interp))
        return input_interp

    def temporal_upsample(self, input_, cache):
        # input_ : (T + 1) x H x W

        repeated = input_.repeat_interleave(2, dim=2)
        # repeated: (2T + 2) x H x W

        if cache["padding"] is None:
            tail = repeated[..., 1:, :, :]  # tail: (2T + 1) x H x W
        else:
            tail = repeated

        conv_out = self.temporal_conv(tail, cache)
        return conv_out + tail

    def forward(self, x, cache):
        if self.temporal_compress:
            # TEMPORAL UPSAMPLE
            x = self.temporal_upsample(x, cache)

        # SPATIAL UPSAMPLE
        s_out = self.spatial_upsample_NEW(x)
        to = torch.empty_like(s_out)

        lin_out = self.linear(s_out, write_to=to)

        return lin_out


class CachedEncoder3D(nn.Module):
    def __init__(
        self,
        *,
        ch=128,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks,
        dropout=0.0,
        in_channels,
        resolution=0,
        z_channels,
        double_z=True,
        padding_mode=None,
        temporal_compress_times=4,
        fix_pxs=False,
        norm_type="group_norm",
        downsample_version=1,
        temporal_compress_start_level=0,
        skip_last_resolution=False,
    ):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.skip_last_resolution = skip_last_resolution

        # log2 of temporal_compress_times
        temporal_compress_level = int(np.log2(temporal_compress_times)) + temporal_compress_start_level

        in_ch_mult = (ch_mult[0],) + tuple(ch_mult)
        self.conv_in = CachedCausalConv3d(
            chan_in=in_channels, chan_out=round(in_ch_mult[0] * self.ch), kernel_size=3, padding_mode=padding_mode
        )

        normalization = CachedGroupNorm if norm_type == "group_norm" else Kandinsky6VAERMSNorm

        curr_res = resolution
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()

            block_in = round(ch * in_ch_mult[i_level])
            block_out = round(ch * ch_mult[i_level])

            if downsample_version > 1 and i_level > 0:
                block_in *= 2

            for i_block in range(self.num_res_blocks):
                block.append(
                    CachedCausalResnetBlock3D(
                        in_channels=block_in,
                        out_channels=block_out,
                        dropout=dropout,
                        temb_channels=self.temb_ch,
                        normalization=normalization,
                        padding_mode=padding_mode,
                    )
                )
                block_in = block_out
            down = nn.Module()
            down.block = block
            if i_level != self.num_resolutions - 1:
                if temporal_compress_start_level <= i_level < temporal_compress_level:
                    down.downsample = CachedPXSDownsample(
                        block_in,
                        compress_time=True,
                        version=downsample_version,
                        fixed_stride=fix_pxs,
                        padding_mode=padding_mode,
                    )
                else:
                    down.downsample = CachedPXSDownsample(
                        block_in,
                        compress_time=False,
                        version=downsample_version,
                        fixed_stride=fix_pxs,
                        padding_mode=padding_mode,
                    )
                curr_res = curr_res // 2
            if not skip_last_resolution or i_level != self.num_resolutions - 1:
                self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = CachedCausalResnetBlock3D(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
            normalization=normalization,
            padding_mode=padding_mode,
        )

        self.mid.block_2 = CachedCausalResnetBlock3D(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
            normalization=normalization,
            padding_mode=padding_mode,
        )

        # end
        self.norm_out = normalization(block_in)

        self.conv_out = CachedCausalConv3d(
            chan_in=block_in,
            chan_out=2 * z_channels if double_z else z_channels,
            kernel_size=3,
            padding_mode=padding_mode,
        )

    def forward(self, x, cache_dict, use_cp=True):
        # timestep embedding
        temb = None

        # downsampling
        h = self.conv_in(x, cache=cache_dict["conv_in"])
        for i_level in range(self.num_resolutions):
            if not self.skip_last_resolution or i_level != self.num_resolutions - 1:
                for i_block in range(self.num_res_blocks):
                    h = self.down[i_level].block[i_block](h, temb, layer_cache=cache_dict[i_level][i_block])
            if i_level != self.num_resolutions - 1:
                h = self.down[i_level].downsample(h, cache=cache_dict[i_level]["down"])

        # middle
        h = self.mid.block_1(h, temb, layer_cache=cache_dict["mid_1"])
        h = self.mid.block_2(h, temb, layer_cache=cache_dict["mid_2"])

        # end
        h = self.norm_out(h, cache=cache_dict["norm_out"])
        h = F.silu(h)
        h = self.conv_out(h, cache=cache_dict["conv_out"])

        return h


class CachedDecoder3D(nn.Module):
    def __init__(
        self,
        ch=128,
        out_ch=None,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks=2,
        dropout=0.0,
        resolution=0,
        z_channels=16,
        give_pre_end=False,
        zq_ch=None,
        add_conv=False,
        padding_mode=None,
        temporal_compress_times=4,
        norm_type="group_norm",
        temporal_compress_start_level=0,
    ):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.give_pre_end = give_pre_end

        # log2 of temporal_compress_times
        temporal_compress_level = int(np.log2(temporal_compress_times)) + temporal_compress_start_level

        if zq_ch is None:
            zq_ch = z_channels

        # compute in_ch_mult, block_in and curr_res at lowest res
        block_in = round(ch * ch_mult[self.num_resolutions - 1])
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)

        self.conv_in = CachedCausalConv3d(
            chan_in=z_channels, chan_out=block_in, kernel_size=3, padding_mode=padding_mode
        )

        modulated_norm = functools.partial(
            Normalize3D, normalization=CachedGroupNorm if norm_type == "group_norm" else Kandinsky6VAERMSNorm
        )

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = CachedCausalResnetBlock3D(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
            zq_ch=zq_ch,
            add_conv=add_conv,
            normalization=modulated_norm,
            padding_mode=padding_mode,
        )

        self.mid.block_2 = CachedCausalResnetBlock3D(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
            zq_ch=zq_ch,
            add_conv=add_conv,
            normalization=modulated_norm,
            padding_mode=padding_mode,
        )

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            block_out = round(ch * ch_mult[i_level])
            for i_block in range(self.num_res_blocks + 1):
                block.append(
                    CachedCausalResnetBlock3D(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                        zq_ch=zq_ch,
                        add_conv=add_conv,
                        normalization=modulated_norm,
                        padding_mode=padding_mode,
                    )
                )
                block_in = block_out
            up = nn.Module()
            up.block = block
            if i_level != 0:
                if (
                    self.num_resolutions - temporal_compress_start_level
                    > i_level
                    >= self.num_resolutions - temporal_compress_level
                ):
                    up.upsample = CachedPXSUpsample(block_in, compress_time=True, padding_mode=padding_mode)
                else:
                    up.upsample = CachedPXSUpsample(block_in, compress_time=False, padding_mode=padding_mode)
            self.up.insert(0, up)

        self.norm_out = modulated_norm(block_in, zq_ch, add_conv=add_conv)

        self.conv_out = CachedCausalConv3d(chan_in=block_in, chan_out=out_ch, kernel_size=3, padding_mode=padding_mode)

    def forward(self, z, cache_dict):
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        zq = z
        h = self.conv_in(z, cache_dict["conv_in"])

        # middle
        h = self.mid.block_1(h, temb, layer_cache=cache_dict["mid_1"], zq=zq)
        h = self.mid.block_2(h, temb, layer_cache=cache_dict["mid_2"], zq=zq)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb, layer_cache=cache_dict[i_level][i_block], zq=zq)
            if i_level != 0:
                h = self.up[i_level].upsample(h, cache_dict[i_level]["up"])

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h, zq, cache_dict["norm_out"])
        h = F.silu(h)
        h = self.conv_out(h, cache_dict["conv_out"])

        return h


@dataclass
class DecoderOutput:
    sample: torch.Tensor


class Kandinsky6SRVAE(ModelMixin, ConfigMixin):
    """Causal 3D VAE used by the Kandinsky 6 video super-resolution pipeline.

    The encoder and decoder process videos in temporal segments while reusing
    their causal convolution state between segments. This keeps memory usage
    bounded for long videos and preserves the checkpoint layout of the KVAE.

    Args:
        vae_type (`str`): VAE architecture identifier. Must be ``"video-kvae"``.
        decoder_ch (`int`, *optional*): Base channel count for the decoder, if
            different from the encoder's ``ch``. Defaults to ``ch``.
        decoder_ch_mult (`tuple[float, ...]`, *optional*): Per-resolution channel
            multiplier for the decoder, if different from the encoder's
            ``ch_mult``. Defaults to ``ch_mult``.
        scaling_factor (`float`, *optional*, defaults to 1.0): Latent scaling
            factor stored in the component configuration.
        spatial_factor (`int`, *optional*, defaults to 16): Spatial compression
            factor of the VAE.
        temporal_factor (`int`, *optional*, defaults to 4): Temporal compression
            factor of the VAE.
    """

    _no_split_modules = ["CachedEncoder3D", "CachedDecoder3D"]
    _supports_gradient_checkpointing = False

    @staticmethod
    def normalize_data(data):
        """Normalize pixel values to the KVAE input range."""
        return data / 128 - 1.0

    @staticmethod
    def denormalize_data(data):
        """Convert normalized KVAE outputs back to pixel values."""
        return (data + 1) * 128

    @register_to_config
    def __init__(
        self,
        vae_type: str,
        in_channels: int = 3,
        out_channels: int = 3,
        z_channels: int = 16,
        ch: int = 128,
        ch_mult: tuple[float, ...] = (1, 2, 4, 8),
        decoder_ch: int | None = None,
        decoder_ch_mult: tuple[float, ...] | None = None,
        num_res_blocks: int = 2,
        dropout: float = 0.0,
        resolution: int = 0,
        padding_mode: str | None = None,
        temporal_compress_times: int = 4,
        temporal_compress_start_level: int = 0,
        norm_type: str = "group_norm",
        double_z: bool = True,
        downsample_version: int = 1,
        fix_pxs: bool = False,
        skip_last_resolution: bool = False,
        give_pre_end: bool = False,
        zq_ch: int | None = None,
        add_conv: bool = False,
        scaling_factor: float = 1.0,
        spatial_factor: int = 16,
        temporal_factor: int = 4,
    ) -> None:
        super().__init__()
        if vae_type != "video-kvae":
            raise ValueError(f"Kandinsky6SRVAE supports only 'video-kvae', got: {vae_type!r}")
        self.encoder = CachedEncoder3D(
            in_channels=in_channels,
            z_channels=z_channels,
            ch=ch,
            ch_mult=ch_mult,
            num_res_blocks=num_res_blocks,
            dropout=dropout,
            resolution=resolution,
            padding_mode=padding_mode,
            temporal_compress_times=temporal_compress_times,
            temporal_compress_start_level=temporal_compress_start_level,
            norm_type=norm_type,
            double_z=double_z,
            downsample_version=downsample_version,
            fix_pxs=fix_pxs,
            skip_last_resolution=skip_last_resolution,
        )
        self.decoder = CachedDecoder3D(
            out_ch=out_channels,
            z_channels=z_channels,
            ch=ch if decoder_ch is None else decoder_ch,
            ch_mult=ch_mult if decoder_ch_mult is None else decoder_ch_mult,
            num_res_blocks=num_res_blocks,
            dropout=dropout,
            resolution=resolution,
            padding_mode=padding_mode,
            temporal_compress_times=temporal_compress_times,
            temporal_compress_start_level=temporal_compress_start_level,
            norm_type=norm_type,
            give_pre_end=give_pre_end,
            zq_ch=zq_ch,
            add_conv=add_conv,
        )
        self.spatial_factor = int(spatial_factor)
        self.temporal_factor = int(temporal_factor)

    def make_empty_cache(self, block: str):
        """Create empty causal-convolution and normalization caches."""

        def make_dict(name, p=None):
            if name == "conv":
                return {"padding": None}

            layer, module = name.split("_")
            if layer == "norm":
                if module == "enc":
                    return {"mean": None, "var": None}
                else:
                    return {"norm": make_dict("norm_enc"), "add_conv": make_dict("conv")}
            elif layer == "resblock":
                return {
                    "norm1": make_dict(f"norm_{module}"),
                    "norm2": make_dict(f"norm_{module}"),
                    "conv1": make_dict("conv"),
                    "conv2": make_dict("conv"),
                    "conv_shortcut": make_dict("conv"),
                }
            elif layer.isdigit():
                out_dict = {"down": [make_dict("conv"), make_dict("conv")], "up": make_dict("conv")}
                for i in range(p):
                    out_dict[i] = make_dict(f"resblock_{module}")

                return out_dict

        cache = {
            "conv_in": make_dict("conv"),
            "mid_1": make_dict(f"resblock_{block}"),
            "mid_2": make_dict(f"resblock_{block}"),
            "norm_out": make_dict(f"norm_{block}"),
            "conv_out": make_dict("conv"),
        }
        for i in range(len(self.config.ch_mult)):
            cache[i] = make_dict(f"{i}_block", p=self.config.num_res_blocks + 1)
        return cache

    @apply_forward_hook
    def encode(self, x, seg_len=16):
        """Encode a video in temporal segments and return latents and segment sizes.

        Args:
            x (`torch.Tensor`): Video tensor in ``(batch, channels, frames, height, width)`` format.
            seg_len (`int`, *optional*, defaults to 16): Number of non-initial
                frames processed in each segment.

        Returns:
            `tuple[torch.Tensor, list[int]]`: Encoded latents and the pixel-space
            segment sizes needed by :meth:`decode`.
        """
        cache = self.make_empty_cache("enc")

        # Compute segment sizes.
        split_list = [seg_len + 1]
        n_frames = x.size(2) - (seg_len + 1)
        while n_frames > 0:
            split_list.append(seg_len)
            n_frames -= seg_len

        split_list[-1] += n_frames

        # Encode each segment.
        latent = []
        for chunk in torch.split(x, split_list, dim=2):
            l = self.encoder(chunk, cache)
            latent.append(DiagonalGaussianDistribution(l).mode())

        latent = torch.cat(latent, dim=2)
        return latent, split_list

    @apply_forward_hook
    def decode(self, z, split_list=None):
        """Decode latent segments while reusing the causal decoder cache.

        Args:
            z (`torch.Tensor`): Latent tensor in ``(batch, channels, frames, height, width)`` format.
            split_list (`list[int]`, *optional*): Pixel-space segment sizes
                returned by :meth:`encode`.

        Returns:
            `DecoderOutput`: Decoded video in ``sample``.
        """
        cache = self.make_empty_cache("dec")

        # Compute latent segment sizes.
        if split_list is None:
            default_split_size = 16 // self.config.temporal_compress_times
            time_dim = z.shape[2]
            if time_dim == 1:
                # image
                split_list = [1]
            else:
                splits_num = (time_dim - 1) // default_split_size
                split_list = [default_split_size] * splits_num
                if (time_dim - 1) % default_split_size != 0:
                    split_list.append((time_dim - 1) % default_split_size)
                split_list[0] += 1
        else:
            split_list = [math.ceil(size / self.config.temporal_compress_times) for size in split_list]

        # Decode each segment.
        recs = []
        for chunk in torch.split(z, split_list, dim=2):
            out = self.decoder(chunk, cache)
            recs.append(out)

        recs = torch.cat(recs, dim=2)
        return DecoderOutput(sample=recs)

    def forward(self, x, seg_len: int = 16):
        """Encode and decode a video in one call.

        Args:
            x (`torch.Tensor`): Video tensor in ``(batch, channels, frames, height, width)`` format.
            seg_len (`int`, *optional*, defaults to 16): Number of non-initial
                frames processed in each segment.

        Returns:
            `DecoderOutput`: Reconstructed video in ``sample``.
        """
        latent, split_list = self.encode(x, seg_len)
        recs = self.decode(latent, split_list)
        return recs


__all__ = ["Kandinsky6SRVAE"]
