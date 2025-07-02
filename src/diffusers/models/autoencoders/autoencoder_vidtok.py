# Copyright 2025 The VidTok team, MSRA & Shanghai Jiao Tong University and The HuggingFace Team.
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

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...configuration_utils import ConfigMixin, register_to_config
from ...utils import logging
from ...utils.accelerate_utils import apply_forward_hook
from ..modeling_outputs import AutoencoderKLOutput
from ..modeling_utils import ModelMixin
from .vae import DecoderOutput, DiagonalGaussianDistribution


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class FSQRegularizer(nn.Module):
    r"""
    Finite Scalar Quantization: VQ-VAE Made Simple - https://arxiv.org/abs/2309.15505 Code adapted from
    https://github.com/lucidrains/vector-quantize-pytorch/blob/master/vector_quantize_pytorch/finite_scalar_quantization.py

    Args:
        levels (`List[int]`):
            A list of quantization levels.
        dim (`int`, *optional*, defaults to `None`):
            The dimension of latent codes.
        num_codebooks (`int`, defaults to 1):
            The number of codebooks.
        keep_num_codebooks_dim (`bool`, *optional*, defaults to `None`):
            Whether to keep the number of codebook dim.
    """

    def __init__(
        self,
        levels: List[int],
        dim: Optional[int] = None,
        num_codebooks: int = 1,
        keep_num_codebooks_dim: Optional[bool] = None,
    ):
        super().__init__()

        _levels = torch.tensor(levels, dtype=torch.int32)
        self.register_buffer("_levels", _levels, persistent=False)

        _basis = torch.cumprod(torch.tensor([1] + levels[:-1]), dim=0, dtype=torch.int32)
        self.register_buffer("_basis", _basis, persistent=False)

        codebook_dim = len(levels)
        self.codebook_dim = codebook_dim

        effective_codebook_dim = codebook_dim * num_codebooks
        self.num_codebooks = num_codebooks
        self.effective_codebook_dim = effective_codebook_dim

        if keep_num_codebooks_dim is None:
            keep_num_codebooks_dim = num_codebooks > 1
        self.keep_num_codebooks_dim = keep_num_codebooks_dim
        self.dim = len(_levels) * num_codebooks if dim is None else dim

        has_projections = self.dim != effective_codebook_dim
        self.project_in = nn.Linear(self.dim, effective_codebook_dim) if has_projections else nn.Identity()
        self.project_out = nn.Linear(effective_codebook_dim, self.dim) if has_projections else nn.Identity()
        self.has_projections = has_projections

        self.codebook_size = self._levels.prod().item()

        implicit_codebook = self.indices_to_codes(torch.arange(self.codebook_size), project_out=False)
        self.register_buffer("implicit_codebook", implicit_codebook, persistent=False)
        self.register_buffer("zero", torch.tensor(0.0), persistent=False)

        self.global_codebook_usage = torch.zeros([2**self.codebook_dim, self.num_codebooks], dtype=torch.long)

    def quantize(self, z: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
        r"""Quantizes z, returns quantized zhat, same shape as z."""
        half_l = (self._levels - 1) * (1 + eps) / 2
        offset = torch.where(self._levels % 2 == 0, 0.5, 0.0)
        shift = (offset / half_l).atanh()
        z = (z + shift).tanh() * half_l - offset
        zhat = z.round()
        quantized = z + (zhat - z).detach()
        half_width = self._levels // 2
        return quantized / half_width

    def codes_to_indices(self, zhat: torch.Tensor) -> torch.Tensor:
        r"""Converts a `code` to an index in the codebook."""
        assert zhat.shape[-1] == self.codebook_dim
        half_width = self._levels // 2
        zhat = (zhat * half_width) + half_width
        return (zhat * self._basis).sum(dim=-1).to(torch.int32)

    def indices_to_codes(self, indices: torch.Tensor, project_out: bool = True) -> torch.Tensor:
        r"""Inverse of `codes_to_indices`."""
        is_img_or_video = indices.ndim >= (3 + int(self.keep_num_codebooks_dim))
        indices = indices.unsqueeze(-1)
        codes_non_centered = (indices // self._basis) % self._levels
        half_width = self._levels // 2
        codes = (codes_non_centered - half_width) / half_width
        if self.keep_num_codebooks_dim:
            codes = codes.reshape(*codes.shape[:-2], -1)
        if project_out:
            codes = self.project_out(codes)
        if is_img_or_video:
            codes = codes.permute(0, -1, *range(1, codes.dim() - 1))
        return codes

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        einstein notation b - batch n - sequence (or flattened spatial dimensions) d - feature dimension c - number of
        codebook dim
        """
        is_img_or_video = z.ndim >= 4

        if is_img_or_video:
            if z.ndim == 5:
                b, d, t, h, w = z.shape
                is_video = True
            else:
                b, d, h, w = z.shape
                is_video = False
            z = z.reshape(b, d, -1).permute(0, 2, 1)

        z = self.project_in(z)
        b, n, _ = z.shape
        z = z.reshape(b, n, self.num_codebooks, -1)

        with torch.autocast("cuda", enabled=False):
            orig_dtype = z.dtype
            z = z.float()
            codes = self.quantize(z)
            indices = self.codes_to_indices(codes)
            codes = codes.type(orig_dtype)

        codes = codes.reshape(b, n, -1)
        out = self.project_out(codes)

        # reconstitute image or video dimensions
        if is_img_or_video:
            if is_video:
                out = out.reshape(b, t, h, w, d).permute(0, 4, 1, 2, 3)
                indices = indices.reshape(b, t, h, w, 1)
            else:
                out = out.reshape(b, h, w, d).permute(0, 3, 1, 2)
                indices = indices.reshape(b, h, w, 1)

        if not self.keep_num_codebooks_dim:
            indices = indices.squeeze(-1)

        return out, indices


class VidTokDownsample2D(nn.Module):
    r"""A 2D downsampling layer used in VidTok Model."""

    def __init__(self, in_channels: int):
        super().__init__()

        self.in_channels = in_channels
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad = (0, 1, 0, 1)
        x = F.pad(x, pad, mode="constant", value=0)
        x = self.conv(x)
        return x


class VidTokUpsample2D(nn.Module):
    r"""A 2D upsampling layer used in VidTok Model."""

    def __init__(self, in_channels: int):
        super().__init__()

        self.in_channels = in_channels
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x.to(torch.float32), scale_factor=2.0, mode="nearest").to(x.dtype)
        x = self.conv(x)
        return x


class VidTokLayerNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()

        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 5:
            x = x.permute(0, 2, 3, 4, 1)
            x = self.norm(x)
            x = x.permute(0, 4, 1, 2, 3)
        elif x.dim() == 4:
            x = x.permute(0, 2, 3, 1)
            x = self.norm(x)
            x = x.permute(0, 3, 1, 2)
        else:
            x = x.permute(0, 2, 1)
            x = self.norm(x)
            x = x.permute(0, 2, 1)
        return x


class VidTokCausalConv1d(nn.Module):
    r"""A 1D causal convolution layer that pads the input tensor to ensure causality in VidTok Model."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        padding: int = 0,
    ):
        super().__init__()

        self.time_pad = dilation * (kernel_size - 1) + (1 - stride)

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation)

        self.is_first_chunk = True
        self.causal_cache = None
        self.cache_offset = 0

    def forward(self, x):
        r"""The forward method of the `VidTokCausalConv1d` class."""
        if self.is_first_chunk:
            first_frame_pad = x[:, :, :1].repeat((1, 1, self.time_pad))
        else:
            first_frame_pad = self.causal_cache
            if self.time_pad != 0:
                first_frame_pad = first_frame_pad[:, :, -self.time_pad :]
            else:
                first_frame_pad = first_frame_pad[:, :, 0:0]
        x = torch.concatenate((first_frame_pad, x), dim=2)
        if self.cache_offset == 0:
            self.causal_cache = x.clone()
        else:
            self.causal_cache = x[:, :, : -self.cache_offset].clone()
        return self.conv(x)


class VidTokCausalConv3d(nn.Module):
    r"""A 3D causal convolution layer that pads the input tensor to ensure causality in VidTok Model."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: Union[int, Tuple[int, int, int]] = 1,
        dilation: Union[int, Tuple[int, int, int]] = 1,
        padding: Union[int, Tuple[int, int, int]] = 0,
        pad_mode: str = "constant",
    ):
        super().__init__()
        self.pad_mode = pad_mode
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3
        if isinstance(dilation, int):
            dilation = (dilation,) * 3
        if isinstance(stride, int):
            stride = (stride,) * 3
        time_kernel_size, height_kernel_size, width_kernel_size = kernel_size
        time_pad = dilation[0] * (time_kernel_size - 1) + (1 - stride[0])
        height_pad = dilation[1] * (height_kernel_size - 1) + (1 - stride[1])
        width_pad = dilation[2] * (width_kernel_size - 1) + (1 - stride[2])

        self.time_pad = time_pad
        self.spatial_padding = (
            width_pad // 2,
            width_pad - width_pad // 2,
            height_pad // 2,
            height_pad - height_pad // 2,
            0,
            0,
        )
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation)

        self.is_first_chunk = True
        self.causal_cache = None
        self.cache_offset = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""The forward method of the `VidTokCausalConv3d` class."""
        if self.is_first_chunk:
            first_frame_pad = x[:, :, :1, :, :].repeat((1, 1, self.time_pad, 1, 1))
        else:
            first_frame_pad = self.causal_cache
            if self.time_pad != 0:
                first_frame_pad = first_frame_pad[:, :, -self.time_pad :]
            else:
                first_frame_pad = first_frame_pad[:, :, 0:0]
        x = torch.concatenate((first_frame_pad, x), dim=2)
        if self.cache_offset == 0:
            self.causal_cache = x.clone()
        else:
            self.causal_cache = x[:, :, : -self.cache_offset].clone()
        x = F.pad(x, self.spatial_padding, mode=self.pad_mode)
        return self.conv(x)


class VidTokDownsample3D(nn.Module):
    r"""A 3D downsampling layer used in VidTok Model."""

    def __init__(self, in_channels: int, out_channels: int, mix_factor: float = 2.0, is_causal: bool = True):
        super().__init__()
        self.is_causal = is_causal
        self.kernel_size = (3, 3, 3)
        self.avg_pool = nn.AvgPool3d((3, 1, 1), stride=(2, 1, 1))
        make_conv_cls = VidTokCausalConv3d if self.is_causal else nn.Conv3d
        self.conv = make_conv_cls(in_channels, out_channels, 3, stride=(2, 1, 1), padding=(0, 1, 1))
        self.mix_factor = nn.Parameter(torch.Tensor([mix_factor]))
        if self.is_causal:
            self.is_first_chunk = True
            self.causal_cache = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""The forward method of the `VidTokDownsample3D` class."""
        alpha = torch.sigmoid(self.mix_factor)
        if self.is_causal:
            pad = (0, 0, 0, 0, 1, 0)
            if self.is_first_chunk:
                x_pad = torch.nn.functional.pad(x, pad, mode="replicate")
            else:
                x_pad = torch.concatenate((self.causal_cache, x), dim=2)
            self.causal_cache = x_pad[:, :, -1:].clone()
            x1 = self.avg_pool(x_pad)
        else:
            pad = (0, 0, 0, 0, 0, 1)
            x = F.pad(x, pad, mode="constant", value=0)
            x1 = self.avg_pool(x)
        x2 = self.conv(x)
        return alpha * x1 + (1 - alpha) * x2


class VidTokUpsample3D(nn.Module):
    r"""A 3D upsampling layer used in VidTok Model."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mix_factor: float = 2.0,
        num_temp_upsample: int = 1,
        is_causal: bool = True,
    ):
        super().__init__()
        make_conv_cls = VidTokCausalConv3d if is_causal else nn.Conv3d
        self.conv = make_conv_cls(in_channels, out_channels, 3, padding=1)
        self.mix_factor = nn.Parameter(torch.Tensor([mix_factor]))

        self.is_causal = is_causal
        if self.is_causal:
            self.enable_cached = True
            self.interpolation_mode = "trilinear"
            self.is_first_chunk = True
            self.causal_cache = None
            self.num_temp_upsample = num_temp_upsample
        else:
            self.enable_cached = False
            self.interpolation_mode = "nearest"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""The forward method of the `VidTokUpsample3D` class."""
        alpha = torch.sigmoid(self.mix_factor)
        if not self.is_causal:
            xlst = [
                F.interpolate(
                    sx.unsqueeze(0).to(torch.float32), scale_factor=[2.0, 1.0, 1.0], mode=self.interpolation_mode
                ).to(x.dtype)
                for sx in x
            ]
            x = torch.cat(xlst, dim=0)
        else:
            if not self.enable_cached:
                x = F.interpolate(x.to(torch.float32), scale_factor=[2.0, 1.0, 1.0], mode=self.interpolation_mode).to(
                    x.dtype
                )
            elif not self.is_first_chunk:
                x = torch.cat([self.causal_cache, x], dim=2)
                self.causal_cache = x[:, :, -2 * self.num_temp_upsample : -self.num_temp_upsample].clone()
                x = F.interpolate(x.to(torch.float32), scale_factor=[2.0, 1.0, 1.0], mode=self.interpolation_mode).to(
                    x.dtype
                )
                x = x[:, :, 2 * self.num_temp_upsample :]
            else:
                self.causal_cache = x[:, :, -self.num_temp_upsample :].clone()
                x, _x = x[:, :, : self.num_temp_upsample], x[:, :, self.num_temp_upsample :]
                x = F.interpolate(x.to(torch.float32), scale_factor=[2.0, 1.0, 1.0], mode=self.interpolation_mode).to(
                    x.dtype
                )
                if _x.shape[-3] > 0:
                    _x = F.interpolate(
                        _x.to(torch.float32), scale_factor=[2.0, 1.0, 1.0], mode=self.interpolation_mode
                    ).to(_x.dtype)
                    x = torch.concat([x, _x], dim=2)
        x_ = self.conv(x)
        return alpha * x + (1 - alpha) * x_


class VidTokAttnBlock(nn.Module):
    r"""A 2D self-attention block used in VidTok Model."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.norm = VidTokLayerNorm(dim=in_channels, eps=1e-6)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def attention(self, hidden_states: torch.Tensor) -> torch.Tensor:
        r"""Implement self-attention."""
        hidden_states = self.norm(hidden_states)
        q = self.q(hidden_states)
        k = self.k(hidden_states)
        v = self.v(hidden_states)
        b, c, h, w = q.shape
        q, k, v = [x.permute(0, 2, 3, 1).reshape(b, -1, c).unsqueeze(1).contiguous() for x in [q, k, v]]
        hidden_states = F.scaled_dot_product_attention(q, k, v)  # scale is dim ** -0.5 per default
        return hidden_states.squeeze(1).reshape(b, h, w, c).permute(0, 3, 1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""The forward method of the `VidTokAttnBlock` class."""
        hidden_states = x
        hidden_states = self.attention(hidden_states)
        hidden_states = self.proj_out(hidden_states)
        return x + hidden_states


class VidTokAttnBlockWrapper(VidTokAttnBlock):
    r"""A 3D self-attention block used in VidTok Model."""

    def __init__(self, in_channels: int, is_causal: bool = True):
        super().__init__(in_channels)
        make_conv_cls = VidTokCausalConv3d if is_causal else nn.Conv3d
        self.q = make_conv_cls(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = make_conv_cls(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = make_conv_cls(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = make_conv_cls(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def attention(self, hidden_states: torch.Tensor) -> torch.Tensor:
        r"""Implement self-attention."""
        hidden_states = self.norm(hidden_states)
        q = self.q(hidden_states)
        k = self.k(hidden_states)
        v = self.v(hidden_states)
        b, c, t, h, w = q.shape
        q, k, v = [x.permute(0, 2, 3, 4, 1).reshape(b, t, -1, c).contiguous() for x in [q, k, v]]
        hidden_states = F.scaled_dot_product_attention(q, k, v)  # scale is dim ** -0.5 per default
        return hidden_states.reshape(b, t, h, w, c).permute(0, 4, 1, 2, 3)


class VidTokResnetBlock(nn.Module):
    r"""A versatile ResNet block used in VidTok Model."""

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        conv_shortcut: bool = False,
        dropout: float = 0.0,
        temb_channels: int = 512,
        btype: str = "3d",
        is_causal: bool = True,
    ):
        super().__init__()
        assert btype in ["1d", "2d", "3d"], f"Invalid btype: {btype}"
        if btype == "2d":
            make_conv_cls = nn.Conv2d
        elif btype == "1d":
            make_conv_cls = VidTokCausalConv1d if is_causal else nn.Conv1d
        else:
            make_conv_cls = VidTokCausalConv3d if is_causal else nn.Conv3d

        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.nonlinearity = nn.SiLU()

        self.norm1 = VidTokLayerNorm(dim=in_channels, eps=1e-6)
        self.conv1 = make_conv_cls(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if temb_channels > 0:
            self.temb_proj = nn.Linear(temb_channels, out_channels)
        self.norm2 = VidTokLayerNorm(dim=out_channels, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = make_conv_cls(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = make_conv_cls(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            else:
                self.nin_shortcut = make_conv_cls(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor, temb: Optional[torch.Tensor]) -> torch.Tensor:
        r"""The forward method of the `VidTokResnetBlock` class."""
        hidden_states = x
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv1(hidden_states)

        if temb is not None:
            hidden_states = hidden_states + self.temb_proj(self.nonlinearity(temb))[:, :, None, None]

        hidden_states = self.norm2(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)
        return x + hidden_states


class VidTokEncoder3D(nn.Module):
    r"""
    The `VidTokEncoder3D` layer of a variational autoencoder that encodes its input into a latent representation.

    Args:
        in_channels (`int`):
            The number of input channels.
        ch (`int`):
            The number of the basic channel.
        ch_mult (`List[int]`, defaults to `[1, 2, 4, 8]`):
            The multiple of the basic channel for each block.
        num_res_blocks (`int`, defaults to 2):
            The number of resblocks.
        dropout (`float`, defaults to 0.0):
            Dropout rate.
        z_channels (`int`, defaults to 4):
            The number of latent channels.
        double_z (`bool`, defaults to `True`):
            Whether or not to double the z_channels.
        spatial_ds (`List`, *optional*, defaults to `None`):
            Spatial downsample layers.
        tempo_ds (`List`, *optional*, defaults to `None`):
            Temporal downsample layers.
        is_causal (`bool`, defaults to `True`):
            Whether it is a causal module.
    """

    def __init__(
        self,
        in_channels: int,
        ch: int,
        ch_mult: List[int] = [1, 2, 4, 8],
        num_res_blocks: int = 2,
        dropout: float = 0.0,
        z_channels: int = 4,
        double_z: bool = True,
        spatial_ds: Optional[List] = None,
        tempo_ds: Optional[List] = None,
        is_causal: bool = True,
    ):
        super().__init__()
        self.is_causal = is_causal

        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.in_channels = in_channels
        self.nonlinearity = nn.SiLU()

        make_conv_cls = VidTokCausalConv3d if self.is_causal else nn.Conv3d

        self.conv_in = make_conv_cls(in_channels, self.ch, kernel_size=3, stride=1, padding=1)

        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.spatial_ds = list(range(0, self.num_resolutions - 1)) if spatial_ds is None else spatial_ds
        self.tempo_ds = [self.num_resolutions - 2, self.num_resolutions - 3] if tempo_ds is None else tempo_ds
        self.down = nn.ModuleList()
        self.down_temporal = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]

            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_temporal = nn.ModuleList()
            attn_temporal = nn.ModuleList()

            for i_block in range(self.num_res_blocks):
                block.append(
                    VidTokResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                        btype="2d",
                    )
                )
                block_temporal.append(
                    VidTokResnetBlock(
                        in_channels=block_out,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                        btype="1d",
                        is_causal=self.is_causal,
                    )
                )
                block_in = block_out

            down = nn.Module()
            down.block = block
            down.attn = attn

            down_temporal = nn.Module()
            down_temporal.block = block_temporal
            down_temporal.attn = attn_temporal

            if i_level in self.spatial_ds:
                down.downsample = VidTokDownsample2D(block_in)
                if i_level in self.tempo_ds:
                    down_temporal.downsample = VidTokDownsample3D(block_in, block_in, is_causal=self.is_causal)

            self.down.append(down)
            self.down_temporal.append(down_temporal)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = VidTokResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
            btype="3d",
            is_causal=self.is_causal,
        )
        self.mid.attn_1 = VidTokAttnBlockWrapper(block_in, is_causal=self.is_causal)
        self.mid.block_2 = VidTokResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
            btype="3d",
            is_causal=self.is_causal,
        )

        # end
        self.norm_out = VidTokLayerNorm(dim=block_in, eps=1e-6)
        self.conv_out = make_conv_cls(
            block_in,
            2 * z_channels if double_z else z_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.gradient_checkpointing = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""The forward method of the `VidTokEncoder3D` class."""
        temb = None
        B, _, T, H, W = x.shape
        hs = [self.conv_in(x)]

        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for i_level in range(self.num_resolutions):
                for i_block in range(self.num_res_blocks):
                    hidden_states = hs[-1].permute(0, 2, 1, 3, 4).reshape(B * T, -1, H, W)
                    hidden_states = self._gradient_checkpointing_func(
                        self.down[i_level].block[i_block], hidden_states, temb
                    )
                    hidden_states = (
                        hidden_states.reshape(B, T, -1, H, W).permute(0, 3, 4, 2, 1).reshape(B * H * W, -1, T)
                    )
                    hidden_states = self._gradient_checkpointing_func(
                        self.down_temporal[i_level].block[i_block], hidden_states, temb
                    )
                    hidden_states = hidden_states.reshape(B, H, W, -1, T).permute(0, 3, 4, 1, 2)
                    hs.append(hidden_states)

                if i_level in self.spatial_ds:
                    # spatial downsample
                    hidden_states = hs[-1].permute(0, 2, 1, 3, 4).reshape(B * T, -1, H, W)
                    hidden_states = self._gradient_checkpointing_func(self.down[i_level].downsample, hidden_states)
                    hidden_states = hidden_states.reshape(B, T, -1, *hidden_states.shape[-2:]).permute(0, 2, 1, 3, 4)
                    if i_level in self.tempo_ds:
                        # temporal downsample
                        hidden_states = self._gradient_checkpointing_func(
                            self.down_temporal[i_level].downsample, hidden_states
                        )
                    hs.append(hidden_states)
                    B, _, T, H, W = hidden_states.shape
            # middle
            hidden_states = hs[-1]
            hidden_states = self._gradient_checkpointing_func(self.mid.block_1, hidden_states, temb)
            hidden_states = self._gradient_checkpointing_func(self.mid.attn_1, hidden_states)
            hidden_states = self._gradient_checkpointing_func(self.mid.block_2, hidden_states, temb)

        else:
            for i_level in range(self.num_resolutions):
                for i_block in range(self.num_res_blocks):
                    hidden_states = hs[-1].permute(0, 2, 1, 3, 4).reshape(B * T, -1, H, W)
                    hidden_states = self.down[i_level].block[i_block](hidden_states, temb)
                    hidden_states = (
                        hidden_states.reshape(B, T, -1, H, W).permute(0, 3, 4, 2, 1).reshape(B * H * W, -1, T)
                    )
                    hidden_states = self.down_temporal[i_level].block[i_block](hidden_states, temb)
                    hidden_states = hidden_states.reshape(B, H, W, -1, T).permute(0, 3, 4, 1, 2)
                    hs.append(hidden_states)

                if i_level in self.spatial_ds:
                    # spatial downsample
                    hidden_states = hs[-1].permute(0, 2, 1, 3, 4).reshape(B * T, -1, H, W)
                    hidden_states = self.down[i_level].downsample(hidden_states)
                    hidden_states = hidden_states.reshape(B, T, -1, *hidden_states.shape[-2:]).permute(0, 2, 1, 3, 4)
                    if i_level in self.tempo_ds:
                        # temporal downsample
                        hidden_states = self.down_temporal[i_level].downsample(hidden_states)
                    hs.append(hidden_states)
                    B, _, T, H, W = hidden_states.shape
            # middle
            hidden_states = hs[-1]
            hidden_states = self.mid.block_1(hidden_states, temb)
            hidden_states = self.mid.attn_1(hidden_states)
            hidden_states = self.mid.block_2(hidden_states, temb)

        # end
        hidden_states = self.norm_out(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv_out(hidden_states)
        return hidden_states


class VidTokDecoder3D(nn.Module):
    r"""
    The `VidTokDecoder3D` layer of a variational autoencoder that decodes its latent representation into an output
    video.

    Args:
        ch (`int`):
            The number of the basic channel.
        ch_mult (`List[int]`, defaults to `[1, 2, 4, 8]`):
            The multiple of the basic channel for each block.
        num_res_blocks (`int`, defaults to 2):
            The number of resblocks.
        dropout (`float`, defaults to 0.0):
            Dropout rate.
        z_channels (`int`, defaults to 4):
            The number of latent channels.
        out_channels (`int`, defaults to 3):
            The number of output channels.
        spatial_us (`List`, *optional*, defaults to `None`):
            Spatial upsample layers.
        tempo_us (`List`, *optional*, defaults to `None`):
            Temporal upsample layers.
        is_causal (`bool`, defaults to `True`):
            Whether it is a causal module.
    """

    def __init__(
        self,
        ch: int,
        ch_mult: List[int] = [1, 2, 4, 8],
        num_res_blocks: int = 2,
        dropout: float = 0.0,
        z_channels: int = 4,
        out_channels: int = 3,
        spatial_us: Optional[List] = None,
        tempo_us: Optional[List] = None,
        is_causal: bool = True,
    ):
        super().__init__()

        self.is_causal = is_causal
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.nonlinearity = nn.SiLU()

        block_in = ch * ch_mult[self.num_resolutions - 1]

        make_conv_cls = VidTokCausalConv3d if self.is_causal else nn.Conv3d

        self.conv_in = make_conv_cls(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = VidTokResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
            btype="3d",
            is_causal=self.is_causal,
        )
        self.mid.attn_1 = VidTokAttnBlockWrapper(block_in, is_causal=self.is_causal)
        self.mid.block_2 = VidTokResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
            btype="3d",
            is_causal=self.is_causal,
        )

        # upsampling
        self.spatial_us = list(range(1, self.num_resolutions)) if spatial_us is None else spatial_us
        self.tempo_us = [1, 2] if tempo_us is None else tempo_us
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(
                    VidTokResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                        btype="2d",
                    )
                )
                block_in = block_out

            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level in self.spatial_us:
                up.upsample = VidTokUpsample2D(block_in)
            self.up.insert(0, up)

        num_temp_upsample = 1
        self.up_temporal = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(
                    VidTokResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                        btype="1d",
                        is_causal=self.is_causal,
                    )
                )
                block_in = block_out
            up_temporal = nn.Module()
            up_temporal.block = block
            up_temporal.attn = attn
            if i_level in self.tempo_us:
                up_temporal.upsample = VidTokUpsample3D(
                    block_in, block_in, num_temp_upsample=num_temp_upsample, is_causal=self.is_causal
                )
                num_temp_upsample *= 2

            self.up_temporal.insert(0, up_temporal)

        # end
        self.norm_out = VidTokLayerNorm(dim=block_in, eps=1e-6)
        self.conv_out = make_conv_cls(block_in, out_channels, kernel_size=3, stride=1, padding=1)

        self.gradient_checkpointing = False

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        r"""The forward method of the `VidTokDecoder3D` class."""
        temb = None
        B, _, T, H, W = z.shape
        hidden_states = self.conv_in(z)

        if torch.is_grad_enabled() and self.gradient_checkpointing:
            # middle
            hidden_states = self._gradient_checkpointing_func(self.mid.block_1, hidden_states, temb)
            hidden_states = self._gradient_checkpointing_func(self.mid.attn_1, hidden_states)
            hidden_states = self._gradient_checkpointing_func(self.mid.block_2, hidden_states, temb)

            for i_level in reversed(range(self.num_resolutions)):
                for i_block in range(self.num_res_blocks + 1):
                    hidden_states = hidden_states.permute(0, 2, 1, 3, 4).reshape(B * T, -1, H, W)
                    hidden_states = self._gradient_checkpointing_func(
                        self.up[i_level].block[i_block], hidden_states, temb
                    )
                    hidden_states = (
                        hidden_states.reshape(B, T, -1, H, W).permute(0, 3, 4, 2, 1).reshape(B * H * W, -1, T)
                    )
                    hidden_states = self._gradient_checkpointing_func(
                        self.up_temporal[i_level].block[i_block], hidden_states, temb
                    )
                    hidden_states = hidden_states.reshape(B, H, W, -1, T).permute(0, 3, 4, 1, 2)

                if i_level in self.spatial_us:
                    # spatial upsample
                    hidden_states = hidden_states.permute(0, 2, 1, 3, 4).reshape(B * T, -1, H, W)
                    hidden_states = self._gradient_checkpointing_func(self.up[i_level].upsample, hidden_states)
                    hidden_states = hidden_states.reshape(B, T, -1, *hidden_states.shape[-2:]).permute(0, 2, 1, 3, 4)
                    if i_level in self.tempo_us:
                        # temporal upsample
                        hidden_states = self._gradient_checkpointing_func(
                            self.up_temporal[i_level].upsample, hidden_states
                        )
                    B, _, T, H, W = hidden_states.shape

        else:
            # middle
            hidden_states = self.mid.block_1(hidden_states, temb)
            hidden_states = self.mid.attn_1(hidden_states)
            hidden_states = self.mid.block_2(hidden_states, temb)

            for i_level in reversed(range(self.num_resolutions)):
                for i_block in range(self.num_res_blocks + 1):
                    hidden_states = hidden_states.permute(0, 2, 1, 3, 4).reshape(B * T, -1, H, W)
                    hidden_states = self.up[i_level].block[i_block](hidden_states, temb)
                    hidden_states = (
                        hidden_states.reshape(B, T, -1, H, W).permute(0, 3, 4, 2, 1).reshape(B * H * W, -1, T)
                    )
                    hidden_states = self.up_temporal[i_level].block[i_block](hidden_states, temb)
                    hidden_states = hidden_states.reshape(B, H, W, -1, T).permute(0, 3, 4, 1, 2)

                if i_level in self.spatial_us:
                    # spatial upsample
                    hidden_states = hidden_states.permute(0, 2, 1, 3, 4).reshape(B * T, -1, H, W)
                    hidden_states = self.up[i_level].upsample(hidden_states)
                    hidden_states = hidden_states.reshape(B, T, -1, *hidden_states.shape[-2:]).permute(0, 2, 1, 3, 4)
                    if i_level in self.tempo_us:
                        # temporal upsample
                        hidden_states = self.up_temporal[i_level].upsample(hidden_states)
                    B, _, T, H, W = hidden_states.shape

        # end
        hidden_states = self.norm_out(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        out = self.conv_out(hidden_states)
        return out


class AutoencoderVidTok(ModelMixin, ConfigMixin):
    r"""
    A VAE model for encoding videos into latents and decoding latent representations into videos, supporting both
    continuous and discrete latent representations. Used in [VidTok](https://github.com/microsoft/VidTok).

    This model inherits from [`ModelMixin`]. Check the superclass documentation for it's generic methods implemented
    for all models (such as downloading or saving).

    Args:
        in_channels (`int`, defaults to 3):
            The number of input channels.
        out_channels (`int`, defaults to 3):
            The number of output channels.
        ch (`int`, defaults to 128):
            The number of the basic channel.
        ch_mult (`List[int]`, defaults to `[1, 2, 4, 4]`):
            The multiple of the basic channel for each block.
        z_channels (`int`, defaults to 4):
            The number of latent channels.
        double_z (`bool`, defaults to `True`):
            Whether or not to double the z_channels.
        num_res_blocks (`int`, defaults to 2):
            The number of resblocks.
        spatial_ds (`List`, *optional*, defaults to `None`):
            Spatial downsample layers.
        spatial_us (`List`, *optional*, defaults to `None`):
            Spatial upsample layers.
        tempo_ds (`List`, *optional*, defaults to `None`):
            Temporal downsample layers.
        tempo_us (`List`, *optional*, defaults to `None`):
            Temporal upsample layers.
        dropout (`float`, defaults to 0.0):
            Dropout rate.
        regularizer (`str`, defaults to `"kl"`):
            The regularizer type - "kl" for continuous cases and "fsq" for discrete cases.
        codebook_size (`int`, defaults to 262144):
            The codebook size used only in discrete cases.
        is_causal (`bool`, defaults to `True`):
            Whether it is a causal module.
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        ch: int = 128,
        ch_mult: List[int] = [1, 2, 4, 4],
        z_channels: int = 4,
        double_z: bool = True,
        num_res_blocks: int = 2,
        spatial_ds: Optional[List] = None,
        spatial_us: Optional[List] = None,
        tempo_ds: Optional[List] = None,
        tempo_us: Optional[List] = None,
        dropout: float = 0.0,
        regularizer: str = "kl",
        codebook_size: int = 262144,
        is_causal: bool = True,
    ):
        super().__init__()
        self.is_causal = is_causal

        self.encoder = VidTokEncoder3D(
            in_channels=in_channels,
            ch=ch,
            ch_mult=ch_mult,
            num_res_blocks=num_res_blocks,
            dropout=dropout,
            z_channels=z_channels,
            double_z=double_z,
            spatial_ds=spatial_ds,
            tempo_ds=tempo_ds,
            is_causal=self.is_causal,
        )
        self.decoder = VidTokDecoder3D(
            ch=ch,
            ch_mult=ch_mult,
            num_res_blocks=num_res_blocks,
            dropout=dropout,
            z_channels=z_channels,
            out_channels=out_channels,
            spatial_us=spatial_us,
            tempo_us=tempo_us,
            is_causal=self.is_causal,
        )
        self.temporal_compression_ratio = 2 ** len(self.encoder.tempo_ds)

        self.regularizer = regularizer
        assert self.regularizer in ["kl", "fsq"], f"Invalid regularizer: {self.regtype}. Only support 'kl' and 'fsq'."

        if self.regularizer == "fsq":
            assert z_channels == int(math.log(codebook_size, 8)) and double_z is False
            self.regularization = FSQRegularizer(levels=[8] * z_channels)

        self.use_slicing = False
        self.use_tiling = False

        # Decode more latent frames at once
        self.num_sample_frames_batch_size = 16
        self.num_latent_frames_batch_size = self.num_sample_frames_batch_size // self.temporal_compression_ratio

        # We make the minimum height and width of sample for tiling half that of the generally supported
        self.tile_sample_min_height = 256
        self.tile_sample_min_width = 256
        self.tile_latent_min_height = int(self.tile_sample_min_height / (2 ** len(self.encoder.spatial_ds)))
        self.tile_latent_min_width = int(self.tile_sample_min_width / (2 ** len(self.encoder.spatial_ds)))
        self.tile_overlap_factor_height = 0.0  # 1 / 8
        self.tile_overlap_factor_width = 0.0  # 1 / 8

    @staticmethod
    def _pad_at_dim(
        t: torch.Tensor, pad: Tuple[int], dim: int = -1, pad_mode: str = "constant", value: float = 0.0
    ) -> torch.Tensor:
        r"""Pad function. Supported pad_mode: `constant`, `replicate`, `reflect`."""
        dims_from_right = (-dim - 1) if dim < 0 else (t.ndim - dim - 1)
        zeros = (0, 0) * dims_from_right
        if pad_mode == "constant":
            return F.pad(t, (*zeros, *pad), value=value)
        return F.pad(t, (*zeros, *pad), mode=pad_mode)

    def enable_tiling(
        self,
        tile_sample_min_height: Optional[int] = None,
        tile_sample_min_width: Optional[int] = None,
        tile_overlap_factor_height: Optional[float] = None,
        tile_overlap_factor_width: Optional[float] = None,
    ) -> None:
        r"""
        Enable tiled VAE decoding. When this option is enabled, the VAE will split the input tensor into tiles to
        compute decoding and encoding in several steps. This is useful for saving a large amount of memory and to allow
        processing larger images.

        Args:
            tile_sample_min_height (`int`, *optional*, defaults to `None`):
                The minimum height required for a sample to be separated into tiles across the height dimension.
            tile_sample_min_width (`int`, *optional*, defaults to `None`):
                The minimum width required for a sample to be separated into tiles across the width dimension.
            tile_overlap_factor_height (`float`, *optional*, defaults to `None`):
                The minimum amount of overlap between two consecutive vertical tiles. This is to ensure that there are
                no tiling artifacts produced across the height dimension. Must be between 0 and 1. Setting a higher
                value might cause more tiles to be processed leading to slow down of the decoding process.
            tile_overlap_factor_width (`float`, *optional*, defaults to `None`):
                The minimum amount of overlap between two consecutive horizontal tiles. This is to ensure that there
                are no tiling artifacts produced across the width dimension. Must be between 0 and 1. Setting a higher
                value might cause more tiles to be processed leading to slow down of the decoding process.
        """
        self.use_tiling = True
        self.tile_sample_min_height = tile_sample_min_height or self.tile_sample_min_height
        self.tile_sample_min_width = tile_sample_min_width or self.tile_sample_min_width
        self.tile_latent_min_height = int(self.tile_sample_min_height / (2 ** len(self.encoder.spatial_ds)))
        self.tile_latent_min_width = int(self.tile_sample_min_width / (2 ** len(self.encoder.spatial_ds)))
        self.tile_overlap_factor_height = tile_overlap_factor_height or self.tile_overlap_factor_height
        self.tile_overlap_factor_width = tile_overlap_factor_width or self.tile_overlap_factor_width

    def disable_tiling(self) -> None:
        r"""
        Disable tiled VAE decoding. If `enable_tiling` was previously enabled, this method will go back to computing
        decoding in one step.
        """
        self.use_tiling = False

    def enable_slicing(self) -> None:
        r"""
        Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
        compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.use_slicing = True

    def disable_slicing(self) -> None:
        r"""
        Disable sliced VAE decoding. If `enable_slicing` was previously enabled, this method will go back to computing
        decoding in one step.
        """
        self.use_slicing = False

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        self._empty_causal_cached(self.encoder)
        self._set_first_chunk(True)

        if self.use_tiling:
            return self.tiled_encode(x)
        return self.encoder(x)

    @apply_forward_hook
    def encode(self, x: torch.Tensor) -> Union[AutoencoderKLOutput, Tuple[torch.Tensor]]:
        r"""
        Encode a batch of images into latents.

        Args:
            x (`torch.Tensor`): Input batch of images.

        Returns:
            `AutoencoderKLOutput` or `Tuple[torch.Tensor]`:
                The latent representations of the encoded videos. If the regularizer is `kl`, an `AutoencoderKLOutput`
                is returned, otherwise a tuple of `torch.Tensor` is returned.
        """
        if self.use_slicing and x.shape[0] > 1:
            encoded_slices = [self._encode(x_slice) for x_slice in x.split(1)]
            z = torch.cat(encoded_slices)
        else:
            z = self._encode(x)

        if self.regularizer == "kl":
            posterior = DiagonalGaussianDistribution(z)
            return AutoencoderKLOutput(latent_dist=posterior)
        else:
            quant_z, indices = self.regularization(z)
            return quant_z, indices

    def _decode(self, z: torch.Tensor, decode_from_indices: bool = False) -> torch.Tensor:
        self._empty_causal_cached(self.decoder)
        self._set_first_chunk(True)
        if not self.is_causal and z.shape[-3] % self.num_latent_frames_batch_size != 0:
            assert (
                z.shape[-3] >= self.num_latent_frames_batch_size
            ), f"Too short latent frames. At least {self.num_latent_frames_batch_size} frames."
            z = z[..., : (z.shape[-3] // self.num_latent_frames_batch_size * self.num_latent_frames_batch_size), :, :]
        if decode_from_indices:
            z = self.tile_indices_to_latent(z) if self.use_tiling else self.indices_to_latent(z)
        dec = self.tiled_decode(z) if self.use_tiling else self.decoder(z)
        return dec

    @apply_forward_hook
    def decode(self, z: torch.Tensor, decode_from_indices: bool = False) -> torch.Tensor:
        r"""
        Decode a batch of images from latents.

        Args:
            z (`torch.Tensor`): Input batch of latent vectors.
            decode_from_indices (`bool`): If decode from indices or decode from latent code.
        Returns:
            `torch.Tensor`: The decoded images.
        """
        if self.use_slicing and z.shape[0] > 1:
            decoded_slices = [self._decode(z_slice, decode_from_indices=decode_from_indices) for z_slice in z.split(1)]
            decoded = torch.cat(decoded_slices)
        else:
            decoded = self._decode(z, decode_from_indices=decode_from_indices)
        if self.is_causal:
            decoded = decoded[:, :, self.temporal_compression_ratio - 1 :, :, :]
        return decoded

    def blend_v(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        blend_extent = min(a.shape[3], b.shape[3], blend_extent)
        for y in range(blend_extent):
            b[:, :, :, y, :] = a[:, :, :, -blend_extent + y, :] * (1 - y / blend_extent) + b[:, :, :, y, :] * (
                y / blend_extent
            )
        return b

    def blend_h(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        blend_extent = min(a.shape[4], b.shape[4], blend_extent)
        for x in range(blend_extent):
            b[:, :, :, :, x] = a[:, :, :, :, -blend_extent + x] * (1 - x / blend_extent) + b[:, :, :, :, x] * (
                x / blend_extent
            )
        return b

    def build_chunk_start_end(self, t, decoder_mode=False):
        if self.is_causal:
            start_end = [[0, self.temporal_compression_ratio]] if not decoder_mode else [[0, 1]]
            start = start_end[0][-1]
        else:
            start_end, start = [], 0
        end = start
        while True:
            if start >= t:
                break
            end = min(
                t, end + (self.num_latent_frames_batch_size if decoder_mode else self.num_sample_frames_batch_size)
            )
            start_end.append([start, end])
            start = end
        if len(start_end) > (2 if self.is_causal else 1):
            if start_end[-1][1] - start_end[-1][0] < (
                self.num_latent_frames_batch_size if decoder_mode else self.num_sample_frames_batch_size
            ):
                start_end[-2] = [start_end[-2][0], start_end[-1][1]]
                start_end = start_end[:-1]
        return start_end

    def _set_first_chunk(self, is_first_chunk=True):
        for module in self.modules():
            if hasattr(module, "is_first_chunk"):
                module.is_first_chunk = is_first_chunk

    def _empty_causal_cached(self, parent):
        for name, module in parent.named_modules():
            if hasattr(module, "causal_cache"):
                module.causal_cache = None

    def _set_cache_offset(self, modules, cache_offset=0):
        for module in modules:
            for submodule in module.modules():
                if hasattr(submodule, "cache_offset"):
                    submodule.cache_offset = cache_offset

    def tiled_encode(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Encode a batch of images using a tiled encoder.

        When this option is enabled, the VAE will split the input tensor into tiles to compute encoding in several
        steps. This is useful to keep memory use constant regardless of image size. The end result of tiled encoding is
        different from non-tiled encoding because each tile uses a different encoder. To avoid tiling artifacts, the
        tiles overlap and are blended together to form a smooth output. You may still see tile-sized changes in the
        output, but they should be much less noticeable.

        Args:
            x (`torch.Tensor`): Input batch of videos.

        Returns:
            `torch.Tensor`: The latent representation of the encoded videos.
        """
        num_frames, height, width = x.shape[-3:]

        overlap_height = int(self.tile_sample_min_height * (1 - self.tile_overlap_factor_height))
        overlap_width = int(self.tile_sample_min_width * (1 - self.tile_overlap_factor_width))
        blend_extent_height = int(self.tile_latent_min_height * self.tile_overlap_factor_height)
        blend_extent_width = int(self.tile_latent_min_width * self.tile_overlap_factor_width)
        row_limit_height = self.tile_latent_min_height - blend_extent_height
        row_limit_width = self.tile_latent_min_width - blend_extent_width

        # Split x into overlapping tiles and encode them separately.
        # The tiles have an overlap to avoid seams between tiles.
        rows = []
        for i in range(0, height, overlap_height):
            row = []
            for j in range(0, width, overlap_width):
                start_end = self.build_chunk_start_end(num_frames)
                time = []
                for idx, (start_frame, end_frame) in enumerate(start_end):
                    self._set_first_chunk(idx == 0)
                    tile = x[
                        :,
                        :,
                        start_frame:end_frame,
                        i : i + self.tile_sample_min_height,
                        j : j + self.tile_sample_min_width,
                    ]
                    tile = self.encoder(tile)
                    time.append(tile)
                row.append(torch.cat(time, dim=2))
            rows.append(row)

        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                # blend the above tile and the left tile
                # to the current tile and add the current tile to the result row
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_extent_height)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_extent_width)
                result_row.append(tile[:, :, :, :row_limit_height, :row_limit_width])
            result_rows.append(torch.cat(result_row, dim=4))
        enc = torch.cat(result_rows, dim=3)
        return enc

    def indices_to_latent(self, token_indices: torch.Tensor) -> torch.Tensor:
        r"""
        Transform indices to latent code.

        Args:
            token_indices (`torch.Tensor`): Token indices.

        Returns:
            `torch.Tensor`: Latent code corresponding to the input token indices.
        """
        b, t, h, w = token_indices.shape
        token_indices = token_indices.unsqueeze(-1).reshape(b, -1, 1)
        codes = self.regularization.indices_to_codes(token_indices)
        codes = codes.permute(0, 2, 3, 1).reshape(b, codes.shape[2], -1)
        z = self.regularization.project_out(codes)
        return z.reshape(b, t, h, w, -1).permute(0, 4, 1, 2, 3)

    def tile_indices_to_latent(self, token_indices: torch.Tensor) -> torch.Tensor:
        r"""
        Transform indices to latent code with tiling inference.

        Args:
            token_indices (`torch.Tensor`): Token indices.

        Returns:
            `torch.Tensor`: Latent code corresponding to the input token indices.
        """
        num_frames = token_indices.shape[1]
        start_end = self.build_chunk_start_end(num_frames, decoder_mode=True)
        result_z = []
        for start, end in start_end:
            chunk_z = self.indices_to_latent(token_indices[:, start:end, :, :])
            result_z.append(chunk_z.clone())
        return torch.cat(result_z, dim=2)

    def tiled_decode(self, z: torch.Tensor) -> torch.Tensor:
        r"""
        Decode a batch of images using a tiled decoder.

        Args:
            z (`torch.Tensor`): Input batch of latent vectors.

        Returns:
            `torch.Tensor`: Reconstructed batch of videos.
        """
        num_frames, height, width = z.shape[-3:]

        overlap_height = int(self.tile_latent_min_height * (1 - self.tile_overlap_factor_height))
        overlap_width = int(self.tile_latent_min_width * (1 - self.tile_overlap_factor_width))
        blend_extent_height = int(self.tile_sample_min_height * self.tile_overlap_factor_height)
        blend_extent_width = int(self.tile_sample_min_width * self.tile_overlap_factor_width)
        row_limit_height = self.tile_sample_min_height - blend_extent_height
        row_limit_width = self.tile_sample_min_width - blend_extent_width

        # Split z into overlapping tiles and decode them separately.
        # The tiles have an overlap to avoid seams between tiles.
        rows = []
        for i in range(0, height, overlap_height):
            row = []
            for j in range(0, width, overlap_width):
                if self.is_causal:
                    assert self.temporal_compression_ratio in [
                        2,
                        4,
                        8,
                    ], "Only support 2x, 4x or 8x temporal downsampling now."
                    if self.temporal_compression_ratio == 4:
                        self._set_cache_offset([self.decoder], 1)
                        self._set_cache_offset([self.decoder.up_temporal[2].upsample, self.decoder.up_temporal[1]], 2)
                        self._set_cache_offset(
                            [self.decoder.up_temporal[1].upsample, self.decoder.up_temporal[0], self.decoder.conv_out],
                            4,
                        )
                    elif self.temporal_compression_ratio == 2:
                        self._set_cache_offset([self.decoder], 1)
                        self._set_cache_offset(
                            [
                                self.decoder.up_temporal[2].upsample,
                                self.decoder.up_temporal[1],
                                self.decoder.up_temporal[0],
                                self.decoder.conv_out,
                            ],
                            2,
                        )
                    else:
                        self._set_cache_offset([self.decoder], 1)
                        self._set_cache_offset([self.decoder.up_temporal[3].upsample, self.decoder.up_temporal[2]], 2)
                        self._set_cache_offset([self.decoder.up_temporal[2].upsample, self.decoder.up_temporal[1]], 4)
                        self._set_cache_offset(
                            [self.decoder.up_temporal[1].upsample, self.decoder.up_temporal[0], self.decoder.conv_out],
                            8,
                        )

                start_end = self.build_chunk_start_end(num_frames, decoder_mode=True)
                time = []
                for idx, (start_frame, end_frame) in enumerate(start_end):
                    self._set_first_chunk(idx == 0)
                    tile = z[
                        :,
                        :,
                        start_frame : (end_frame + 1 if self.is_causal and end_frame + 1 <= num_frames else end_frame),
                        i : i + self.tile_latent_min_height,
                        j : j + self.tile_latent_min_width,
                    ]
                    tile = self.decoder(tile)
                    if self.is_causal and end_frame + 1 <= num_frames:
                        tile = tile[:, :, : -self.temporal_compression_ratio]
                    time.append(tile)
                row.append(torch.cat(time, dim=2))
            rows.append(row)

        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                # blend the above tile and the left tile
                # to the current tile and add the current tile to the result row
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_extent_height)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_extent_width)
                result_row.append(tile[:, :, :, :row_limit_height, :row_limit_width])
            result_rows.append(torch.cat(result_row, dim=4))

        dec = torch.cat(result_rows, dim=3)
        return dec

    def forward(
        self,
        sample: torch.Tensor,
        sample_posterior: bool = True,
        encoder_mode: bool = False,
        return_dict: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> Union[torch.Tensor, DecoderOutput]:
        r"""The forward method of the `AutoencoderVidTok` class."""
        x = sample
        res = 1 if self.is_causal else 0
        if self.is_causal:
            if x.shape[2] % self.temporal_compression_ratio != res:
                time_padding = self.temporal_compression_ratio - x.shape[2] % self.temporal_compression_ratio + res
                x = self._pad_at_dim(x, (0, time_padding), dim=2, pad_mode="replicate")
            else:
                time_padding = 0
        else:
            if x.shape[2] % self.num_sample_frames_batch_size != res:
                if not encoder_mode:
                    time_padding = (
                        self.num_sample_frames_batch_size - x.shape[2] % self.num_sample_frames_batch_size + res
                    )
                    x = self._pad_at_dim(x, (0, time_padding), dim=2, pad_mode="replicate")
                else:
                    assert (
                        x.shape[2] >= self.num_sample_frames_batch_size
                    ), f"Too short video. At least {self.num_sample_frames_batch_size} frames."
                    x = x[:, :, : x.shape[2] // self.num_sample_frames_batch_size * self.num_sample_frames_batch_size]
            else:
                time_padding = 0

        if self.is_causal:
            x = self._pad_at_dim(x, (self.temporal_compression_ratio - 1, 0), dim=2, pad_mode="replicate")

        if self.regularizer == "kl":
            posterior = self.encode(x).latent_dist
            if sample_posterior:
                z = posterior.sample(generator=generator)
            else:
                z = posterior.mode()
            if encoder_mode:
                return z
        else:
            z, indices = self.encode(x)
            if encoder_mode:
                return z, indices

        dec = self.decode(z)
        if time_padding != 0:
            dec = dec[:, :, :-time_padding, :, :]

        if not return_dict:
            return dec
        return DecoderOutput(sample=dec)
