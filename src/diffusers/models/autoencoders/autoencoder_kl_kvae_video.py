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
import functools
import math
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...configuration_utils import ConfigMixin, register_to_config
from ...loaders.single_file_model import FromOriginalModelMixin
from ...utils import logging
from ...utils.accelerate_utils import apply_forward_hook
from ..modeling_outputs import AutoencoderKLOutput
from ..modeling_utils import ModelMixin
from .vae import AutoencoderMixin, DecoderOutput, DiagonalGaussianDistribution


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def nonlinearity(x: torch.Tensor) -> torch.Tensor:
    return F.silu(x)


# =============================================================================
# Base layers
# =============================================================================

class KVAESafeConv3d(nn.Conv3d):
    r"""
    A 3D convolution layer that splits the input tensor into smaller parts to avoid OOM.
    """

    def forward(self, input: torch.Tensor, write_to: torch.Tensor = None) -> torch.Tensor:
        memory_count = input.numel() * input.element_size() / (10**9)

        if memory_count > 3:
            kernel_size = self.kernel_size[0]
            part_num = math.ceil(memory_count / 2)
            input_chunks = torch.chunk(input, part_num, dim=2)

            if write_to is None:
                output = []
                for i, chunk in enumerate(input_chunks):
                    if i == 0 or kernel_size == 1:
                        z = torch.clone(chunk)
                    else:
                        z = torch.cat([z[:, :, -kernel_size + 1:], chunk], dim=2)
                    output.append(super().forward(z))
                return torch.cat(output, dim=2)
            else:
                time_offset = 0
                for i, chunk in enumerate(input_chunks):
                    if i == 0 or kernel_size == 1:
                        z = torch.clone(chunk)
                    else:
                        z = torch.cat([z[:, :, -kernel_size + 1:], chunk], dim=2)
                    z_time = z.size(2) - (kernel_size - 1)
                    write_to[:, :, time_offset:time_offset + z_time] = super().forward(z)
                    time_offset += z_time
                return write_to
        else:
            if write_to is None:
                return super().forward(input)
            else:
                write_to[...] = super().forward(input)
                return write_to


class KVAECausalConv3d(nn.Module):
    r"""
    A 3D causal convolution layer.
    """

    def __init__(
        self,
        chan_in: int,
        chan_out: int,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: Tuple[int, int, int] = (1, 1, 1),
        dilation: Tuple[int, int, int] = (1, 1, 1),
        **kwargs,
    ):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)

        time_kernel_size, height_kernel_size, width_kernel_size = kernel_size

        self.height_pad = height_kernel_size // 2
        self.width_pad = width_kernel_size // 2
        self.time_pad = time_kernel_size - 1
        self.time_kernel_size = time_kernel_size
        self.stride = stride

        self.conv = KVAESafeConv3d(chan_in, chan_out, kernel_size, stride=stride, dilation=dilation, **kwargs)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        padding_3d = (self.width_pad, self.width_pad, self.height_pad, self.height_pad, self.time_pad, 0)
        input_padded = F.pad(input, padding_3d, mode="replicate")
        return self.conv(input_padded)


class KVAECachedCausalConv3d(KVAECausalConv3d):
    r"""
    A 3D causal convolution layer with caching for temporal processing.
    """

    def forward(self, input: torch.Tensor, cache: Dict) -> torch.Tensor:
        t_stride = self.stride[0]
        padding_3d = (self.height_pad, self.height_pad, self.width_pad, self.width_pad, 0, 0)
        input_parallel = F.pad(input, padding_3d, mode="replicate")

        if cache['padding'] is None:
            first_frame = input_parallel[:, :, :1]
            time_pad_shape = list(first_frame.shape)
            time_pad_shape[2] = self.time_pad
            padding = first_frame.expand(time_pad_shape)
        else:
            padding = cache['padding']

        out_size = list(input.shape)
        out_size[1] = self.conv.out_channels
        if t_stride == 2:
            out_size[2] = (input.size(2) + 1) // 2
        output = torch.empty(tuple(out_size), dtype=input.dtype, device=input.device)

        offset_out = math.ceil(padding.size(2) / t_stride)
        offset_in = offset_out * t_stride - padding.size(2)

        if offset_out > 0:
            padding_poisoned = torch.cat([padding, input_parallel[:, :, :offset_in + self.time_kernel_size - t_stride]], dim=2)
            output[:, :, :offset_out] = self.conv(padding_poisoned)

        if offset_out < output.size(2):
            output[:, :, offset_out:] = self.conv(input_parallel[:, :, offset_in:])

        pad_offset = offset_in + t_stride * math.trunc((input_parallel.size(2) - offset_in - self.time_kernel_size) / t_stride) + t_stride
        cache['padding'] = torch.clone(input_parallel[:, :, pad_offset:])

        return output


class KVAECachedGroupNorm(nn.GroupNorm):
    r"""
    GroupNorm with caching support for temporal processing.
    """

    def forward(self, x: torch.Tensor, cache: Dict = None) -> torch.Tensor:
        out = super().forward(x)
        if cache is not None:
            if cache.get('mean') is None and cache.get('var') is None:
                cache['mean'] = 1
                cache['var'] = 1
        return out


def Normalize(in_channels: int, gather: bool = False, **kwargs) -> nn.GroupNorm:
    return KVAECachedGroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


# =============================================================================
# Cached layers
# =============================================================================

class KVAECachedSpatialNorm3D(nn.Module):
    r"""
    Spatially conditioned normalization for decoder with caching.
    """

    def __init__(
        self,
        f_channels: int,
        zq_channels: int,
        add_conv: bool = False,
        normalization = Normalize,
        **norm_layer_params,
    ):
        super().__init__()
        self.norm_layer = normalization(in_channels=f_channels, **norm_layer_params)
        self.add_conv = add_conv

        if add_conv:
            self.conv = KVAECachedCausalConv3d(chan_in=zq_channels, chan_out=zq_channels, kernel_size=3)

        self.conv_y = KVAESafeConv3d(zq_channels, f_channels, kernel_size=1)
        self.conv_b = KVAESafeConv3d(zq_channels, f_channels, kernel_size=1)

    def forward(self, f: torch.Tensor, zq: torch.Tensor, cache: Dict) -> torch.Tensor:
        if cache['norm'].get('mean') is None and cache['norm'].get('var') is None:
            f_first, f_rest = f[:, :, :1], f[:, :, 1:]
            f_first_size, f_rest_size = f_first.shape[-3:], f_rest.shape[-3:]
            zq_first, zq_rest = zq[:, :, :1], zq[:, :, 1:]

            zq_first = F.interpolate(zq_first, size=f_first_size, mode="nearest")

            if zq.size(2) > 1:
                zq_rest_splits = torch.split(zq_rest, 32, dim=1)
                interpolated_splits = [F.interpolate(split, size=f_rest_size, mode="nearest") for split in zq_rest_splits]
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
            zq = self.conv(zq, cache['add_conv'])

        norm_f = self.norm_layer(f, cache['norm'])
        norm_f.mul_(self.conv_y(zq))
        norm_f.add_(self.conv_b(zq))

        return norm_f


def Normalize3D(in_channels: int, zq_ch: int, add_conv: bool, normalization = Normalize):
    return KVAECachedSpatialNorm3D(
        in_channels, zq_ch,
        add_conv=add_conv,
        num_groups=32, eps=1e-6, affine=True,
        normalization=normalization
    )


class KVAECachedResnetBlock3D(nn.Module):
    r"""
    A 3D ResNet block with caching.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        conv_shortcut: bool = False,
        dropout: float = 0.0,
        temb_channels: int = 0,
        zq_ch: Optional[int] = None,
        add_conv: bool = False,
        gather_norm: bool = False,
        normalization = Normalize,
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = normalization(in_channels, zq_ch=zq_ch, add_conv=add_conv)
        self.conv1 = KVAECachedCausalConv3d(chan_in=in_channels, chan_out=out_channels, kernel_size=3)

        if temb_channels > 0:
            self.temb_proj = nn.Linear(temb_channels, out_channels)

        self.norm2 = normalization(out_channels, zq_ch=zq_ch, add_conv=add_conv)
        self.conv2 = KVAECachedCausalConv3d(chan_in=out_channels, chan_out=out_channels, kernel_size=3)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = KVAECachedCausalConv3d(chan_in=in_channels, chan_out=out_channels, kernel_size=3)
            else:
                self.nin_shortcut = KVAESafeConv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor, temb: torch.Tensor, layer_cache: Dict, zq: torch.Tensor = None) -> torch.Tensor:
        h = x

        if zq is None:
            # Encoder path - norm takes cache
            h = self.norm1(h, cache=layer_cache['norm1'])
        else:
            # Decoder path - spatial norm takes zq and cache
            h = self.norm1(h, zq, cache=layer_cache['norm1'])

        h = F.silu(h, inplace=True)
        h = self.conv1(h, cache=layer_cache['conv1'])

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None, None]

        if zq is None:
            h = self.norm2(h, cache=layer_cache['norm2'])
        else:
            h = self.norm2(h, zq, cache=layer_cache['norm2'])

        h = F.silu(h, inplace=True)
        h = self.conv2(h, cache=layer_cache['conv2'])

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x, cache=layer_cache['conv_shortcut'])
            else:
                x = self.nin_shortcut(x)

        return x + h


class KVAECachedPXSDownsample(nn.Module):
    r"""
    A 3D downsampling layer using PixelUnshuffle with caching.
    """

    def __init__(self, in_channels: int, compress_time: bool, factor: int = 2):
        super().__init__()
        self.temporal_compress = compress_time
        self.factor = factor
        self.unshuffle = nn.PixelUnshuffle(self.factor)
        self.s_pool = nn.AvgPool3d((1, 2, 2), (1, 2, 2))

        self.spatial_conv = KVAESafeConv3d(
            in_channels, in_channels, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1),
            padding_mode='reflect'
        )

        if self.temporal_compress:
            self.temporal_conv = KVAECachedCausalConv3d(
                in_channels, in_channels, kernel_size=(3, 1, 1), stride=(2, 1, 1), dilation=(1, 1, 1)
            )

        self.linear = nn.Conv3d(in_channels, in_channels, kernel_size=1, stride=1)

    def spatial_downsample(self, input: torch.Tensor) -> torch.Tensor:
        from einops import rearrange
        pxs_input = rearrange(input, 'b c t h w -> (b t) c h w')
        pxs_interm = self.unshuffle(pxs_input)
        b, c, h, w = pxs_interm.shape
        pxs_interm_view = pxs_interm.view(b, c // self.factor ** 2, self.factor ** 2, h, w)
        pxs_out = torch.mean(pxs_interm_view, dim=2)
        pxs_out = rearrange(pxs_out, '(b t) c h w -> b c t h w', t=input.size(2))
        conv_out = self.spatial_conv(input)
        return conv_out + pxs_out

    def temporal_downsample(self, input: torch.Tensor, cache: list) -> torch.Tensor:
        from einops import rearrange
        permuted = rearrange(input, "b c t h w -> (b h w) c t")

        if cache[0]['padding'] is None:
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

        full_interp = rearrange(full_interp, "(b h w) c t -> b c t h w", h=input.size(-2), w=input.size(-1))
        conv_out = self.temporal_conv(input, cache[0])
        return conv_out + full_interp

    def forward(self, x: torch.Tensor, cache: list) -> torch.Tensor:
        out = self.spatial_downsample(x)

        if self.temporal_compress:
            out = self.temporal_downsample(out, cache=cache)

        return self.linear(out)


class KVAECachedPXSUpsample(nn.Module):
    r"""
    A 3D upsampling layer using PixelShuffle with caching.
    """

    def __init__(self, in_channels: int, compress_time: bool, factor: int = 2):
        super().__init__()
        self.temporal_compress = compress_time
        self.factor = factor
        self.shuffle = nn.PixelShuffle(self.factor)

        self.spatial_conv = KVAESafeConv3d(
            in_channels, in_channels, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1),
            padding_mode='reflect'
        )

        if self.temporal_compress:
            self.temporal_conv = KVAECachedCausalConv3d(
                in_channels, in_channels, kernel_size=(3, 1, 1), stride=(1, 1, 1), dilation=(1, 1, 1)
            )

        self.linear = KVAESafeConv3d(in_channels, in_channels, kernel_size=1, stride=1)

    def spatial_upsample(self, input: torch.Tensor) -> torch.Tensor:
        b, c, t, h, w = input.shape
        input_view = input.permute(0, 2, 1, 3, 4).reshape(b, t * c, h, w)
        input_interp = F.interpolate(input_view, scale_factor=2, mode='nearest')
        input_interp = input_interp.view(b, t, c, 2 * h, 2 * w).permute(0, 2, 1, 3, 4)

        to = torch.empty_like(input_interp)
        out = self.spatial_conv(input_interp, write_to=to)
        input_interp.add_(out)
        return input_interp

    def temporal_upsample(self, input: torch.Tensor, cache: Dict) -> torch.Tensor:
        time_factor = 1.0 + 1.0 * (input.size(2) > 1)
        if isinstance(time_factor, torch.Tensor):
            time_factor = time_factor.item()

        repeated = input.repeat_interleave(int(time_factor), dim=2)

        if cache['padding'] is None:
            tail = repeated[..., int(time_factor - 1):, :, :]
        else:
            tail = repeated

        conv_out = self.temporal_conv(tail, cache)
        return conv_out + tail

    def forward(self, x: torch.Tensor, cache: Dict) -> torch.Tensor:
        if self.temporal_compress:
            x = self.temporal_upsample(x, cache)

        s_out = self.spatial_upsample(x)
        to = torch.empty_like(s_out)
        lin_out = self.linear(s_out, write_to=to)
        return lin_out


# =============================================================================
# Cached Encoder/Decoder
# =============================================================================

class KVAECachedEncoder3D(nn.Module):
    r"""
    Cached 3D Encoder for KVAE.
    """

    def __init__(
        self,
        ch: int = 128,
        ch_mult: Tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        dropout: float = 0.0,
        in_channels: int = 3,
        z_channels: int = 16,
        double_z: bool = True,
        temporal_compress_times: int = 4,
        **ignore_kwargs,
    ):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.in_channels = in_channels
        self.temporal_compress_level = int(np.log2(temporal_compress_times))

        self.conv_in = KVAECachedCausalConv3d(chan_in=in_channels, chan_out=self.ch, kernel_size=3)

        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        block_in = ch

        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()

            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]

            for i_block in range(self.num_res_blocks):
                block.append(
                    KVAECachedResnetBlock3D(
                        in_channels=block_in,
                        out_channels=block_out,
                        dropout=dropout,
                        temb_channels=self.temb_ch,
                        normalization=Normalize,
                    )
                )
                block_in = block_out

            down = nn.Module()
            down.block = block
            down.attn = attn

            if i_level != self.num_resolutions - 1:
                if i_level < self.temporal_compress_level:
                    down.downsample = KVAECachedPXSDownsample(block_in, compress_time=True)
                else:
                    down.downsample = KVAECachedPXSDownsample(block_in, compress_time=False)
            self.down.append(down)

        self.mid = nn.Module()
        self.mid.block_1 = KVAECachedResnetBlock3D(
            in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout, normalization=Normalize
        )
        self.mid.block_2 = KVAECachedResnetBlock3D(
            in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout, normalization=Normalize
        )

        self.norm_out = Normalize(block_in)
        self.conv_out = KVAECachedCausalConv3d(
            chan_in=block_in, chan_out=2 * z_channels if double_z else z_channels, kernel_size=3
        )

    def forward(self, x: torch.Tensor, cache_dict: Dict) -> torch.Tensor:
        temb = None

        h = self.conv_in(x, cache=cache_dict['conv_in'])

        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](h, temb, layer_cache=cache_dict[i_level][i_block])
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
            if i_level != self.num_resolutions - 1:
                h = self.down[i_level].downsample(h, cache=cache_dict[i_level]['down'])

        h = self.mid.block_1(h, temb, layer_cache=cache_dict['mid_1'])
        h = self.mid.block_2(h, temb, layer_cache=cache_dict['mid_2'])

        h = self.norm_out(h, cache=cache_dict['norm_out'])
        h = nonlinearity(h)
        h = self.conv_out(h, cache=cache_dict['conv_out'])

        return h


class KVAECachedDecoder3D(nn.Module):
    r"""
    Cached 3D Decoder for KVAE.
    """

    def __init__(
        self,
        ch: int = 128,
        out_ch: int = 3,
        ch_mult: Tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        dropout: float = 0.0,
        z_channels: int = 16,
        zq_ch: Optional[int] = None,
        add_conv: bool = False,
        temporal_compress_times: int = 4,
        **kwargs,
    ):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.temporal_compress_level = int(np.log2(temporal_compress_times))

        if zq_ch is None:
            zq_ch = z_channels

        block_in = ch * ch_mult[self.num_resolutions - 1]

        self.conv_in = KVAECachedCausalConv3d(chan_in=z_channels, chan_out=block_in, kernel_size=3)

        modulated_norm = functools.partial(Normalize3D, normalization=Normalize)

        self.mid = nn.Module()
        self.mid.block_1 = KVAECachedResnetBlock3D(
            in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch,
            dropout=dropout, zq_ch=zq_ch, add_conv=add_conv, normalization=modulated_norm
        )
        self.mid.block_2 = KVAECachedResnetBlock3D(
            in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch,
            dropout=dropout, zq_ch=zq_ch, add_conv=add_conv, normalization=modulated_norm
        )

        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]

            for i_block in range(self.num_res_blocks + 1):
                block.append(
                    KVAECachedResnetBlock3D(
                        in_channels=block_in, out_channels=block_out, temb_channels=self.temb_ch,
                        dropout=dropout, zq_ch=zq_ch, add_conv=add_conv, normalization=modulated_norm
                    )
                )
                block_in = block_out

            up = nn.Module()
            up.block = block
            up.attn = attn

            if i_level != 0:
                if i_level < self.num_resolutions - self.temporal_compress_level:
                    up.upsample = KVAECachedPXSUpsample(block_in, compress_time=False)
                else:
                    up.upsample = KVAECachedPXSUpsample(block_in, compress_time=True)
            self.up.insert(0, up)

        self.norm_out = modulated_norm(block_in, zq_ch, add_conv=add_conv)
        self.conv_out = KVAECachedCausalConv3d(chan_in=block_in, chan_out=out_ch, kernel_size=3)

    def forward(self, z: torch.Tensor, cache_dict: Dict) -> torch.Tensor:
        temb = None
        zq = z

        h = self.conv_in(z, cache_dict['conv_in'])

        h = self.mid.block_1(h, temb, layer_cache=cache_dict['mid_1'], zq=zq)
        h = self.mid.block_2(h, temb, layer_cache=cache_dict['mid_2'], zq=zq)

        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb, layer_cache=cache_dict[i_level][i_block], zq=zq)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h, zq)
            if i_level != 0:
                h = self.up[i_level].upsample(h, cache_dict[i_level]['up'])

        h = self.norm_out(h, zq, cache_dict['norm_out'])
        h = nonlinearity(h)
        h = self.conv_out(h, cache_dict['conv_out'])

        return h



# =============================================================================
# Main AutoencoderKL class
# =============================================================================

class AutoencoderKLKVAEVideo(ModelMixin, AutoencoderMixin, ConfigMixin, FromOriginalModelMixin):
    r"""
    A VAE model with KL loss for encoding videos into latents and decoding latent representations into videos.
    Used in [KVAE](https://github.com/kandinskylab/kvae-1).

    This model inherits from [`ModelMixin`]. Check the superclass documentation for its generic methods implemented
    for all models (such as downloading or saving).

    Parameters:
        ch (`int`, *optional*, defaults to 128): Base channel count.
        ch_mult (`Tuple[int]`, *optional*, defaults to `(1, 2, 4, 8)`): Channel multipliers per level.
        num_res_blocks (`int`, *optional*, defaults to 2): Number of residual blocks per level.
        in_channels (`int`, *optional*, defaults to 3): Number of input channels.
        out_ch (`int`, *optional*, defaults to 3): Number of output channels.
        z_channels (`int`, *optional*, defaults to 16): Number of latent channels.
        temporal_compress_times (`int`, *optional*, defaults to 4): Temporal compression factor.
    """

    _supports_gradient_checkpointing = True
    _no_split_modules = ["KVAECachedResnetBlock3D"]

    @register_to_config
    def __init__(
        self,
        ch: int = 128,
        ch_mult: Tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        in_channels: int = 3,
        out_ch: int = 3,
        z_channels: int = 16,
        temporal_compress_times: int = 4,
    ):
        super().__init__()

        encoder_params = dict(
            ch=ch,
            ch_mult=ch_mult,
            num_res_blocks=num_res_blocks,
            in_channels=in_channels,
            z_channels=z_channels,
            double_z=True,
            temporal_compress_times=temporal_compress_times,
        )

        decoder_params = dict(
            ch=ch,
            ch_mult=ch_mult,
            num_res_blocks=num_res_blocks,
            out_ch=out_ch,
            z_channels=z_channels,
            temporal_compress_times=temporal_compress_times,
        )

        self.encoder = KVAECachedEncoder3D(**encoder_params)

        self.decoder = KVAECachedDecoder3D(**decoder_params)

        self.use_slicing = False
        self.use_tiling = False

    def _make_encoder_cache(self) -> Dict:
        """Create empty cache for cached encoder."""
        def make_dict(name, p=None):
            if name == 'conv':
                return {'padding': None}

            layer, module = name.split('_')
            if layer == 'norm':
                if module == 'enc':
                    return {'mean': None, 'var': None}
                else:
                    return {'norm': make_dict('norm_enc'), 'add_conv': make_dict('conv')}
            elif layer == 'resblock':
                return {
                    'norm1': make_dict(f'norm_{module}'),
                    'norm2': make_dict(f'norm_{module}'),
                    'conv1': make_dict('conv'),
                    'conv2': make_dict('conv'),
                    'conv_shortcut': make_dict('conv')
                }
            elif layer.isdigit():
                out_dict = {'down': [make_dict('conv'), make_dict('conv')], 'up': make_dict('conv')}
                for i in range(p):
                    out_dict[i] = make_dict(f'resblock_{module}')
                return out_dict

        cache = {
            'conv_in': make_dict('conv'),
            'mid_1': make_dict('resblock_enc'),
            'mid_2': make_dict('resblock_enc'),
            'norm_out': make_dict('norm_enc'),
            'conv_out': make_dict('conv')
        }
        # Encoder uses num_res_blocks per level
        for i in range(len(self.config.ch_mult)):
            cache[i] = make_dict(f'{i}_enc', p=self.config.num_res_blocks)
        return cache

    def _make_decoder_cache(self) -> Dict:
        """Create empty cache for decoder."""
        def make_dict(name, p=None):
            if name == 'conv':
                return {'padding': None}

            layer, module = name.split('_')
            if layer == 'norm':
                if module == 'enc':
                    return {'mean': None, 'var': None}
                else:
                    return {'norm': make_dict('norm_enc'), 'add_conv': make_dict('conv')}
            elif layer == 'resblock':
                return {
                    'norm1': make_dict(f'norm_{module}'),
                    'norm2': make_dict(f'norm_{module}'),
                    'conv1': make_dict('conv'),
                    'conv2': make_dict('conv'),
                    'conv_shortcut': make_dict('conv')
                }
            elif layer.isdigit():
                out_dict = {'down': [make_dict('conv'), make_dict('conv')], 'up': make_dict('conv')}
                for i in range(p):
                    out_dict[i] = make_dict(f'resblock_{module}')
                return out_dict

        cache = {
            'conv_in': make_dict('conv'),
            'mid_1': make_dict('resblock_dec'),
            'mid_2': make_dict('resblock_dec'),
            'norm_out': make_dict('norm_dec'),
            'conv_out': make_dict('conv')
        }
        for i in range(len(self.config.ch_mult)):
            cache[i] = make_dict(f'{i}_dec', p=self.config.num_res_blocks + 1)
        return cache

    def enable_slicing(self) -> None:
        r"""Enable sliced VAE decoding."""
        self.use_slicing = True

    def disable_slicing(self) -> None:
        r"""Disable sliced VAE decoding."""
        self.use_slicing = False

    def _encode(self, x: torch.Tensor, seg_len: int = 16) -> torch.Tensor:
        # Cached encoder processes by segments
        cache = self._make_encoder_cache()

        split_list = [seg_len + 1]
        n_frames = x.size(2) - (seg_len + 1)
        while n_frames > 0:
            split_list.append(seg_len)
            n_frames -= seg_len
        split_list[-1] += n_frames

        latent = []
        for chunk in torch.split(x, split_list, dim=2):
            l = self.encoder(chunk, cache)
            sample, _ = torch.chunk(l, 2, dim=1)
            latent.append(sample)

        return torch.cat(latent, dim=2)

    @apply_forward_hook
    def encode(
        self, x: torch.Tensor, return_dict: bool = True
    ) -> Union[AutoencoderKLOutput, Tuple[DiagonalGaussianDistribution]]:
        """
        Encode a batch of videos into latents.

        Args:
            x (`torch.Tensor`): Input batch of videos with shape (B, C, T, H, W).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`~models.autoencoder_kl.AutoencoderKLOutput`] instead of a plain tuple.

        Returns:
            The latent representations of the encoded videos.
        """
        if self.use_slicing and x.shape[0] > 1:
            encoded_slices = [self._encode(x_slice) for x_slice in x.split(1)]
            h = torch.cat(encoded_slices)
        else:
            h = self._encode(x)

        # For cached encoder, we already did the split in _encode
        h_double = torch.cat([h, torch.zeros_like(h)], dim=1)
        posterior = DiagonalGaussianDistribution(h_double)

        if not return_dict:
            return (posterior,)
        return AutoencoderKLOutput(latent_dist=posterior)

    def _decode(self, z: torch.Tensor, seg_len: int = 16) -> torch.Tensor:
        cache = self._make_decoder_cache()
        temporal_compress = self.config.temporal_compress_times

        split_list = [seg_len + 1]
        n_frames = temporal_compress * (z.size(2) - 1) - seg_len
        while n_frames > 0:
            split_list.append(seg_len)
            n_frames -= seg_len
        split_list[-1] += n_frames
        split_list = [math.ceil(size / temporal_compress) for size in split_list]

        recs = []
        for chunk in torch.split(z, split_list, dim=2):
            out = self.decoder(chunk, cache)
            recs.append(out)

        return torch.cat(recs, dim=2)

    @apply_forward_hook
    def decode(self, z: torch.Tensor, return_dict: bool = True) -> Union[DecoderOutput, torch.Tensor]:
        """
        Decode a batch of videos.

        Args:
            z (`torch.Tensor`): Input batch of latent vectors with shape (B, C, T, H, W).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`~models.vae.DecoderOutput`] instead of a plain tuple.

        Returns:
            [`~models.vae.DecoderOutput`] or `tuple`: Decoded video.
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
        sample_posterior: bool = False,
        return_dict: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> Union[DecoderOutput, torch.Tensor]:
        x = sample
        posterior = self.encode(x).latent_dist
        if sample_posterior:
            z = posterior.sample(generator=generator)
        else:
            z = posterior.mode()
        dec = self.decode(z).sample
        if not return_dict:
            return (dec,)
        return DecoderOutput(sample=dec)
