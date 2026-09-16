"""MMAudio VAE Diffusers component."""

from __future__ import annotations

import json
import math
import os
import shutil
from os import PathLike
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...configuration_utils import ConfigMixin, register_to_config
from ..modeling_utils import ModelMixin
from ...utils.accelerate_utils import apply_forward_hook
from einops import rearrange
from huggingface_hub import PyTorchModelHubMixin, hf_hub_download
from librosa.filters import mel as librosa_mel_fn
from torch import nn, pow, sin
from torch.nn import Conv1d, ConvTranspose1d, Parameter
from torch.nn import functional as F
from torch.nn.utils import weight_norm as legacy_weight_norm
from torch.nn.utils.parametrizations import weight_norm
from torch.nn.utils.parametrize import remove_parametrizations

# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/
"""Improved diffusion model architecture proposed in the paper
"Analyzing and Improving the Training Dynamics of Diffusion Models"."""


# ----------------------------------------------------------------------------
# Variant of constant() that inherits dtype and device from the given
# reference tensor by default.

_constant_cache = dict()


def constant(value, shape=None, dtype=None, device=None, memory_format=None):
    value = np.asarray(value)
    if shape is not None:
        shape = tuple(shape)
    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        device = torch.device("cpu")
    if memory_format is None:
        memory_format = torch.contiguous_format

    key = (value.shape, value.dtype, value.tobytes(), shape, dtype, device, memory_format)
    tensor = _constant_cache.get(key, None)
    if tensor is None:
        tensor = torch.as_tensor(value.copy(), dtype=dtype, device=device)
        if shape is not None:
            tensor, _ = torch.broadcast_tensors(tensor, torch.empty(shape))
        tensor = tensor.contiguous(memory_format=memory_format)
        _constant_cache[key] = tensor
    return tensor


def const_like(ref, value, shape=None, dtype=None, device=None, memory_format=None):
    if dtype is None:
        dtype = ref.dtype
    if device is None:
        device = ref.device
    return constant(value, shape=shape, dtype=dtype, device=device, memory_format=memory_format)


# ----------------------------------------------------------------------------
# Normalize given tensor to unit magnitude with respect to the given
# dimensions. Default = all dimensions except the first.


def normalize(x, dim=None, eps=1e-4):
    if dim is None:
        dim = list(range(1, x.ndim))
    norm = torch.linalg.vector_norm(x, dim=dim, keepdim=True, dtype=torch.float32)
    norm = torch.add(eps, norm, alpha=np.sqrt(norm.numel() / x.numel()))
    return x / norm.to(x.dtype)


class Normalize(torch.nn.Module):
    def __init__(self, dim=None, eps=1e-4):
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x):
        return normalize(x, dim=self.dim, eps=self.eps)


# ----------------------------------------------------------------------------
# Upsample or downsample the given tensor with the given filter,
# or keep it as is.


def resample(x, f=[1, 1], mode="keep"):
    if mode == "keep":
        return x
    f = np.float32(f)
    assert f.ndim == 1 and len(f) % 2 == 0
    pad = (len(f) - 1) // 2
    f = f / f.sum()
    f = np.outer(f, f)[np.newaxis, np.newaxis, :, :]
    f = const_like(x, f)
    c = x.shape[1]
    if mode == "down":
        return torch.nn.functional.conv2d(x, f.tile([c, 1, 1, 1]), groups=c, stride=2, padding=(pad,))
    assert mode == "up"
    return torch.nn.functional.conv_transpose2d(x, (f * 4).tile([c, 1, 1, 1]), groups=c, stride=2, padding=(pad,))


# ----------------------------------------------------------------------------
# Magnitude-preserving SiLU (Equation 81).


def mp_silu(x):
    return torch.nn.functional.silu(x) / 0.596


class MPSiLU(torch.nn.Module):
    def forward(self, x):
        return mp_silu(x)


# ----------------------------------------------------------------------------
# Magnitude-preserving sum (Equation 88).


def mp_sum(a, b, t=0.5):
    return a.lerp(b, t) / np.sqrt((1 - t) ** 2 + t**2)


# ----------------------------------------------------------------------------
# Magnitude-preserving concatenation (Equation 103).


def mp_cat(a, b, dim=1, t=0.5):
    Na = a.shape[dim]
    Nb = b.shape[dim]
    C = np.sqrt((Na + Nb) / ((1 - t) ** 2 + t**2))
    wa = C / np.sqrt(Na) * (1 - t)
    wb = C / np.sqrt(Nb) * t
    return torch.cat([wa * a, wb * b], dim=dim)


# ----------------------------------------------------------------------------
# Magnitude-preserving convolution or fully-connected layer (Equation 47)
# with force weight normalization (Equation 66).


class MPConv1D(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.out_channels = out_channels
        self.weight = torch.nn.Parameter(torch.randn(out_channels, in_channels, kernel_size))

        self.weight_norm_removed = False

    def forward(self, x, gain=1):
        assert self.weight_norm_removed, "call remove_weight_norm() before inference"

        w = self.weight * gain
        if w.ndim == 2:
            return x @ w.t()
        assert w.ndim == 3
        return torch.nn.functional.conv1d(x, w, padding=(w.shape[-1] // 2,))

    def remove_weight_norm(self):
        w = self.weight.to(torch.float32)
        w = normalize(w)  # traditional weight normalization
        w = w / np.sqrt(w[0].numel())
        w = w.to(self.weight.dtype)
        self.weight.data.copy_(w)

        self.weight_norm_removed = True
        return self


class DiagonalGaussianDistribution:
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)

    def sample(self, rng: Optional[torch.Generator] = None):
        # x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.parameters.device)

        r = torch.empty_like(self.mean).normal_(generator=rng)
        x = self.mean + self.std * r

        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.0])
        else:
            if other is None:
                return 0.5 * torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar
            else:
                return 0.5 * (
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var
                    - 1.0
                    - self.logvar
                    + other.logvar
                )

    def nll(self, sample, dims=[1, 2, 3]):
        if self.deterministic:
            return torch.Tensor([0.0])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var, dim=dims)

    def mode(self):
        return self.mean


def nonlinearity(x):
    # swish
    return mp_silu(x)


class ResnetBlock1D(nn.Module):
    def __init__(self, *, in_dim, out_dim=None, conv_shortcut=False, kernel_size=3, use_norm=True):
        super().__init__()
        self.in_dim = in_dim
        out_dim = in_dim if out_dim is None else out_dim
        self.out_dim = out_dim
        self.use_conv_shortcut = conv_shortcut
        self.use_norm = use_norm

        self.conv1 = MPConv1D(in_dim, out_dim, kernel_size=kernel_size)
        self.conv2 = MPConv1D(out_dim, out_dim, kernel_size=kernel_size)
        if self.in_dim != self.out_dim:
            if self.use_conv_shortcut:
                self.conv_shortcut = MPConv1D(in_dim, out_dim, kernel_size=kernel_size)
            else:
                self.nin_shortcut = MPConv1D(in_dim, out_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # pixel norm
        if self.use_norm:
            x = normalize(x, dim=1)

        h = x
        h = nonlinearity(h)
        h = self.conv1(h)

        h = nonlinearity(h)
        h = self.conv2(h)

        if self.in_dim != self.out_dim:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return mp_sum(x, h, t=0.3)


class AttnBlock1D(nn.Module):
    def __init__(self, in_channels, num_heads=1):
        super().__init__()
        self.in_channels = in_channels

        self.num_heads = num_heads
        self.qkv = MPConv1D(in_channels, in_channels * 3, kernel_size=1)
        self.proj_out = MPConv1D(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        h = x
        y = self.qkv(h)
        y = y.reshape(y.shape[0], self.num_heads, -1, 3, y.shape[-1])
        q, k, v = normalize(y, dim=2).unbind(3)

        q = rearrange(q, "b h c l -> b h l c")
        k = rearrange(k, "b h c l -> b h l c")
        v = rearrange(v, "b h c l -> b h l c")

        h = F.scaled_dot_product_attention(q, k, v)
        h = rearrange(h, "b h l c -> b (h c) l")

        h = self.proj_out(h)

        return mp_sum(x, h, t=0.3)


class Upsample1D(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = MPConv1D(in_channels, in_channels, kernel_size=3)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest-exact")  # support 3D tensor(B,C,T)
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample1D(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv1 = MPConv1D(in_channels, in_channels, kernel_size=1)
            self.conv2 = MPConv1D(in_channels, in_channels, kernel_size=1)

    def forward(self, x):

        if self.with_conv:
            x = self.conv1(x)

        x = F.avg_pool1d(x, kernel_size=2, stride=2)

        if self.with_conv:
            x = self.conv2(x)

        return x


# fmt: off
DATA_MEAN_128D = [
    -3.3462, -2.6723, -2.4893, -2.3143, -2.2664, -2.3317, -2.1802, -2.4006, -2.2357, -2.4597,
    -2.3717, -2.4690, -2.5142, -2.4919, -2.6610, -2.5047, -2.7483, -2.5926, -2.7462, -2.7033,
    -2.7386, -2.8112, -2.7502, -2.9594, -2.7473, -3.0035, -2.8891, -2.9922, -2.9856, -3.0157,
    -3.1191, -2.9893, -3.1718, -3.0745, -3.1879, -3.2310, -3.1424, -3.2296, -3.2791, -3.2782,
    -3.2756, -3.3134, -3.3509, -3.3750, -3.3951, -3.3698, -3.4505, -3.4509, -3.5089, -3.4647,
    -3.5536, -3.5788, -3.5867, -3.6036, -3.6400, -3.6747, -3.7072, -3.7279, -3.7283, -3.7795,
    -3.8259, -3.8447, -3.8663, -3.9182, -3.9605, -3.9861, -4.0105, -4.0373, -4.0762, -4.1121,
    -4.1488, -4.1874, -4.2461, -4.3170, -4.3639, -4.4452, -4.5282, -4.6297, -4.7019, -4.7960,
    -4.8700, -4.9507, -5.0303, -5.0866, -5.1634, -5.2342, -5.3242, -5.4053, -5.4927, -5.5712,
    -5.6464, -5.7052, -5.7619, -5.8410, -5.9188, -6.0103, -6.0955, -6.1673, -6.2362, -6.3120,
    -6.3926, -6.4797, -6.5565, -6.6511, -6.8130, -6.9961, -7.1275, -7.2457, -7.3576, -7.4663,
    -7.6136, -7.7469, -7.8815, -8.0132, -8.1515, -8.3071, -8.4722, -8.7418, -9.3975, -9.6628,
    -9.7671, -9.8863, -9.9992, -10.0860, -10.1709, -10.5418, -11.2795, -11.3861
]

DATA_STD_128D = [
    2.3804, 2.4368, 2.3772, 2.3145, 2.2803, 2.2510, 2.2316, 2.2083, 2.1996, 2.1835, 2.1769, 2.1659,
    2.1631, 2.1618, 2.1540, 2.1606, 2.1571, 2.1567, 2.1612, 2.1579, 2.1679, 2.1683, 2.1634, 2.1557,
    2.1668, 2.1518, 2.1415, 2.1449, 2.1406, 2.1350, 2.1313, 2.1415, 2.1281, 2.1352, 2.1219, 2.1182,
    2.1327, 2.1195, 2.1137, 2.1080, 2.1179, 2.1036, 2.1087, 2.1036, 2.1015, 2.1068, 2.0975, 2.0991,
    2.0902, 2.1015, 2.0857, 2.0920, 2.0893, 2.0897, 2.0910, 2.0881, 2.0925, 2.0873, 2.0960, 2.0900,
    2.0957, 2.0958, 2.0978, 2.0936, 2.0886, 2.0905, 2.0845, 2.0855, 2.0796, 2.0840, 2.0813, 2.0817,
    2.0838, 2.0840, 2.0917, 2.1061, 2.1431, 2.1976, 2.2482, 2.3055, 2.3700, 2.4088, 2.4372, 2.4609,
    2.4731, 2.4847, 2.5072, 2.5451, 2.5772, 2.6147, 2.6529, 2.6596, 2.6645, 2.6726, 2.6803, 2.6812,
    2.6899, 2.6916, 2.6931, 2.6998, 2.7062, 2.7262, 2.7222, 2.7158, 2.7041, 2.7485, 2.7491, 2.7451,
    2.7485, 2.7233, 2.7297, 2.7233, 2.7145, 2.6958, 2.6788, 2.6439, 2.6007, 2.4786, 2.2469, 2.1877,
    2.1392, 2.0717, 2.0107, 1.9676, 1.9140, 1.7102, 0.9101, 0.7164
]
# fmt: on


class VAE(nn.Module):
    def __init__(
        self,
        *,
        data_dim: int,
        embed_dim: int,
        hidden_dim: int,
    ):
        super().__init__()

        if data_dim != 128:
            raise ValueError(f"Only 44k audio VAE data is supported, got data_dim={data_dim}")
        self.data_mean = nn.Buffer(torch.tensor(DATA_MEAN_128D, dtype=torch.float32))
        self.data_std = nn.Buffer(torch.tensor(DATA_STD_128D, dtype=torch.float32))

        self.data_mean = self.data_mean.view(1, -1, 1)
        self.data_std = self.data_std.view(1, -1, 1)

        self.encoder = Encoder1D(
            dim=hidden_dim,
            ch_mult=(1, 2, 4),
            num_res_blocks=2,
            attn_layers=[3],
            down_layers=[0],
            in_dim=data_dim,
            embed_dim=embed_dim,
        )
        self.decoder = Decoder1D(
            dim=hidden_dim,
            ch_mult=(1, 2, 4),
            num_res_blocks=2,
            attn_layers=[3],
            down_layers=[0],
            in_dim=data_dim,
            out_dim=data_dim,
            embed_dim=embed_dim,
        )

        self.embed_dim = embed_dim
        # self.quant_conv = nn.Conv1d(2 * embed_dim, 2 * embed_dim, 1)
        # self.post_quant_conv = nn.Conv1d(embed_dim, embed_dim, 1)

    def encode(self, x: torch.Tensor, normalize: bool = True) -> DiagonalGaussianDistribution:
        if normalize:
            x = self.normalize(x)
        moments = self.encoder(x)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z: torch.Tensor, unnormalize: bool = True) -> torch.Tensor:
        dec = self.decoder(z)
        if unnormalize:
            dec = self.unnormalize(dec)
        return dec

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.data_mean) / self.data_std

    def unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.data_std + self.data_mean

    def forward(
        self,
        x: torch.Tensor,
        sample_posterior: bool = True,
        rng: Optional[torch.Generator] = None,
        normalize: bool = True,
        unnormalize: bool = True,
    ) -> tuple[torch.Tensor, DiagonalGaussianDistribution]:

        posterior = self.encode(x, normalize=normalize)
        if sample_posterior:
            z = posterior.sample(rng)
        else:
            z = posterior.mode()
        dec = self.decode(z, unnormalize=unnormalize)
        return dec, posterior

    def load_weights(self, src_dict) -> None:
        self.load_state_dict(src_dict, strict=True)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def remove_weight_norm(self):
        for name, m in self.named_modules():
            if isinstance(m, MPConv1D):
                m.remove_weight_norm()

        return self


class Encoder1D(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        ch_mult: tuple[int] = (1, 2, 4, 8),
        num_res_blocks: int,
        attn_layers: list[int] = [],
        down_layers: list[int] = [],
        resamp_with_conv: bool = True,
        in_dim: int,
        embed_dim: int,
        double_z: bool = True,
        kernel_size: int = 3,
        clip_act: float = 256.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_layers = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.in_channels = in_dim
        self.clip_act = clip_act
        self.down_layers = down_layers
        self.attn_layers = attn_layers
        self.conv_in = MPConv1D(in_dim, self.dim, kernel_size=kernel_size)

        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        # downsampling
        self.down = nn.ModuleList()
        for i_level in range(self.num_layers):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = dim * in_ch_mult[i_level]
            block_out = dim * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock1D(in_dim=block_in, out_dim=block_out, kernel_size=kernel_size, use_norm=True))
                block_in = block_out
                if i_level in attn_layers:
                    attn.append(AttnBlock1D(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level in down_layers:
                down.downsample = Downsample1D(block_in, resamp_with_conv)
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock1D(in_dim=block_in, out_dim=block_in, kernel_size=kernel_size, use_norm=True)
        self.mid.attn_1 = AttnBlock1D(block_in)
        self.mid.block_2 = ResnetBlock1D(in_dim=block_in, out_dim=block_in, kernel_size=kernel_size, use_norm=True)

        # end
        self.conv_out = MPConv1D(block_in, 2 * embed_dim if double_z else embed_dim, kernel_size=kernel_size)

        self.learnable_gain = nn.Parameter(torch.zeros([]))

    def forward(self, x):

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_layers):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1])
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                h = h.clamp(-self.clip_act, self.clip_act)
                hs.append(h)
            if i_level in self.down_layers:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        h = h.clamp(-self.clip_act, self.clip_act)

        # end
        h = nonlinearity(h)
        h = self.conv_out(h, gain=(self.learnable_gain + 1))
        return h


class Decoder1D(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        out_dim: int,
        ch_mult: tuple[int] = (1, 2, 4, 8),
        num_res_blocks: int,
        attn_layers: list[int] = [],
        down_layers: list[int] = [],
        kernel_size: int = 3,
        resamp_with_conv: bool = True,
        in_dim: int,
        embed_dim: int,
        clip_act: float = 256.0,
    ):
        super().__init__()
        self.ch = dim
        self.num_layers = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.in_channels = in_dim
        self.clip_act = clip_act
        self.down_layers = [i + 1 for i in down_layers]  # each downlayer add one

        # compute in_ch_mult, block_in and curr_res at lowest res
        block_in = dim * ch_mult[self.num_layers - 1]

        # z to block_in
        self.conv_in = MPConv1D(embed_dim, block_in, kernel_size=kernel_size)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock1D(in_dim=block_in, out_dim=block_in, use_norm=True)
        self.mid.attn_1 = AttnBlock1D(block_in)
        self.mid.block_2 = ResnetBlock1D(in_dim=block_in, out_dim=block_in, use_norm=True)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_layers)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = dim * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(ResnetBlock1D(in_dim=block_in, out_dim=block_out, use_norm=True))
                block_in = block_out
                if i_level in attn_layers:
                    attn.append(AttnBlock1D(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level in self.down_layers:
                up.upsample = Upsample1D(block_in, resamp_with_conv)
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.conv_out = MPConv1D(block_in, out_dim, kernel_size=kernel_size)
        self.learnable_gain = nn.Parameter(torch.zeros([]))

    def forward(self, z):
        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        h = h.clamp(-self.clip_act, self.clip_act)

        # upsampling
        for i_level in reversed(range(self.num_layers)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
                h = h.clamp(-self.clip_act, self.clip_act)
            if i_level in self.down_layers:
                h = self.up[i_level].upsample(h)

        h = nonlinearity(h)
        h = self.conv_out(h, gain=(self.learnable_gain + 1))
        return h


# Implementation adapted from https://github.com/EdwardDixon/snake under the MIT license.
#   LICENSE is in incl_licenses directory.


class Snake(nn.Module):
    """
    Implementation of a sine-based periodic activation function
    Shape:
        - Input: (B, C, T)
        - Output: (B, C, T), same shape as the input
    Parameters:
        - alpha - trainable parameter
    References:
        - This activation function is from this paper by Liu Ziyin, Tilman Hartwig, Masahito Ueda:
        https://arxiv.org/abs/2006.08195
    Examples:
        >>> a1 = snake(256)
        >>> x = torch.randn(256)
        >>> x = a1(x)
    """

    def __init__(self, in_features, alpha=1.0, alpha_trainable=True, alpha_logscale=False):
        """
        Initialization.
        INPUT:
            - in_features: shape of the input
            - alpha: trainable parameter
            alpha is initialized to 1 by default, higher values = higher-frequency.
            alpha will be trained along with the rest of your model.
        """
        super(Snake, self).__init__()
        self.in_features = in_features

        # Initialize alpha
        self.alpha_logscale = alpha_logscale
        if self.alpha_logscale:  # Log scale alphas initialized to zeros
            self.alpha = Parameter(torch.zeros(in_features) * alpha)
        else:  # Linear scale alphas initialized to ones
            self.alpha = Parameter(torch.ones(in_features) * alpha)

        self.alpha.requires_grad = alpha_trainable

        self.no_div_by_zero = 0.000000001

    def forward(self, x):
        """
        Forward pass of the function.
        Applies the function to the input elementwise.
        Snake ∶= x + 1/a * sin^2 (xa)
        """
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)  # Line up with x to [B, C, T]
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
        x = x + (1.0 / (alpha + self.no_div_by_zero)) * pow(sin(x * alpha), 2)

        return x


class SnakeBeta(nn.Module):
    """
    A modified Snake function which uses separate parameters for the magnitude of the periodic components
    Shape:
        - Input: (B, C, T)
        - Output: (B, C, T), same shape as the input
    Parameters:
        - alpha - trainable parameter that controls frequency
        - beta - trainable parameter that controls magnitude
    References:
        - This activation function is a modified version based on this paper by Liu Ziyin, Tilman Hartwig, Masahito Ueda:
        https://arxiv.org/abs/2006.08195
    Examples:
        >>> a1 = snakebeta(256)
        >>> x = torch.randn(256)
        >>> x = a1(x)
    """

    def __init__(self, in_features, alpha=1.0, alpha_trainable=True, alpha_logscale=False):
        """
        Initialization.
        INPUT:
            - in_features: shape of the input
            - alpha - trainable parameter that controls frequency
            - beta - trainable parameter that controls magnitude
            alpha is initialized to 1 by default, higher values = higher-frequency.
            beta is initialized to 1 by default, higher values = higher-magnitude.
            alpha will be trained along with the rest of your model.
        """
        super(SnakeBeta, self).__init__()
        self.in_features = in_features

        # Initialize alpha
        self.alpha_logscale = alpha_logscale
        if self.alpha_logscale:  # Log scale alphas initialized to zeros
            self.alpha = Parameter(torch.zeros(in_features) * alpha)
            self.beta = Parameter(torch.zeros(in_features) * alpha)
        else:  # Linear scale alphas initialized to ones
            self.alpha = Parameter(torch.ones(in_features) * alpha)
            self.beta = Parameter(torch.ones(in_features) * alpha)

        self.alpha.requires_grad = alpha_trainable
        self.beta.requires_grad = alpha_trainable

        self.no_div_by_zero = 0.000000001

    def forward(self, x):
        """
        Forward pass of the function.
        Applies the function to the input elementwise.
        SnakeBeta ∶= x + 1/b * sin^2 (xa)
        """
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)  # Line up with x to [B, C, T]
        beta = self.beta.unsqueeze(0).unsqueeze(-1)
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
            beta = torch.exp(beta)
        x = x + (1.0 / (beta + self.no_div_by_zero)) * pow(sin(x * alpha), 2)

        return x


# Adapted from https://github.com/junjun3518/alias-free-torch under the Apache License 2.0
#   LICENSE is in incl_licenses directory.


if "sinc" in dir(torch):
    sinc = torch.sinc
else:
    # This code is adopted from adefossez's julius.core.sinc under the MIT License
    # https://adefossez.github.io/julius/julius/core.html
    #   LICENSE is in incl_licenses directory.
    def sinc(x: torch.Tensor):
        """
        Implementation of sinc, i.e. sin(pi * x) / (pi * x)
        __Warning__: Different to julius.sinc, the input is multiplied by `pi`!
        """
        return torch.where(
            x == 0,
            torch.tensor(1.0, device=x.device, dtype=x.dtype),
            torch.sin(math.pi * x) / math.pi / x,
        )


# This code is adopted from adefossez's julius.lowpass.LowPassFilters under the MIT License
# https://adefossez.github.io/julius/julius/lowpass.html
#   LICENSE is in incl_licenses directory.
def kaiser_sinc_filter1d(cutoff, half_width, kernel_size):  # return filter [1,1,kernel_size]
    even = kernel_size % 2 == 0
    half_size = kernel_size // 2

    # For kaiser window
    delta_f = 4 * half_width
    A = 2.285 * (half_size - 1) * math.pi * delta_f + 7.95
    if A > 50.0:
        beta = 0.1102 * (A - 8.7)
    elif A >= 21.0:
        beta = 0.5842 * (A - 21) ** 0.4 + 0.07886 * (A - 21.0)
    else:
        beta = 0.0
    window = torch.kaiser_window(kernel_size, beta=beta, periodic=False)

    # ratio = 0.5/cutoff -> 2 * cutoff = 1 / ratio
    if even:
        time = torch.arange(-half_size, half_size) + 0.5
    else:
        time = torch.arange(kernel_size) - half_size
    if cutoff == 0:
        filter_ = torch.zeros_like(time)
    else:
        filter_ = 2 * cutoff * window * sinc(2 * cutoff * time)
        """
        Normalize filter to have sum = 1, otherwise we will have a small leakage of the constant component in the input signal.
        """
        filter_ /= filter_.sum()
        filter = filter_.view(1, 1, kernel_size)

    return filter


class LowPassFilter1d(nn.Module):
    def __init__(
        self,
        cutoff=0.5,
        half_width=0.6,
        stride: int = 1,
        padding: bool = True,
        padding_mode: str = "replicate",
        kernel_size: int = 12,
    ):
        """
        kernel_size should be even number for stylegan3 setup, in this implementation, odd number is also possible.
        """
        super().__init__()
        if cutoff < -0.0:
            raise ValueError("Minimum cutoff must be larger than zero.")
        if cutoff > 0.5:
            raise ValueError("A cutoff above 0.5 does not make sense.")
        self.kernel_size = kernel_size
        self.even = kernel_size % 2 == 0
        self.pad_left = kernel_size // 2 - int(self.even)
        self.pad_right = kernel_size // 2
        self.stride = stride
        self.padding = padding
        self.padding_mode = padding_mode
        filter = kaiser_sinc_filter1d(cutoff, half_width, kernel_size)
        self.register_buffer("filter", filter)

    # Input [B, C, T]
    def forward(self, x):
        _, C, _ = x.shape

        if self.padding:
            x = F.pad(x, (self.pad_left, self.pad_right), mode=self.padding_mode)
        out = F.conv1d(x, self.filter.expand(C, -1, -1), stride=self.stride, groups=C)

        return out


# Adapted from https://github.com/junjun3518/alias-free-torch under the Apache License 2.0
#   LICENSE is in incl_licenses directory.


class UpSample1d(nn.Module):
    def __init__(self, ratio=2, kernel_size=None):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        self.stride = ratio
        self.pad = self.kernel_size // ratio - 1
        self.pad_left = self.pad * self.stride + (self.kernel_size - self.stride) // 2
        self.pad_right = self.pad * self.stride + (self.kernel_size - self.stride + 1) // 2
        filter = kaiser_sinc_filter1d(cutoff=0.5 / ratio, half_width=0.6 / ratio, kernel_size=self.kernel_size)
        self.register_buffer("filter", filter)

    # x: [B, C, T]
    def forward(self, x):
        _, C, _ = x.shape

        x = F.pad(x, (self.pad, self.pad), mode="replicate")
        x = self.ratio * F.conv_transpose1d(x, self.filter.expand(C, -1, -1), stride=self.stride, groups=C)
        x = x[..., self.pad_left : -self.pad_right]

        return x


class DownSample1d(nn.Module):
    def __init__(self, ratio=2, kernel_size=None):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        self.lowpass = LowPassFilter1d(
            cutoff=0.5 / ratio,
            half_width=0.6 / ratio,
            stride=ratio,
            kernel_size=self.kernel_size,
        )

    def forward(self, x):
        xx = self.lowpass(x)

        return xx


# Adapted from https://github.com/junjun3518/alias-free-torch under the Apache License 2.0
#   LICENSE is in incl_licenses directory.


class Activation1d(nn.Module):
    def __init__(
        self,
        activation,
        up_ratio: int = 2,
        down_ratio: int = 2,
        up_kernel_size: int = 12,
        down_kernel_size: int = 12,
    ):
        super().__init__()
        self.up_ratio = up_ratio
        self.down_ratio = down_ratio
        self.act = activation
        self.upsample = UpSample1d(up_ratio, up_kernel_size)
        self.downsample = DownSample1d(down_ratio, down_kernel_size)

    # x: [B,C,T]
    def forward(self, x):
        x = self.upsample(x)
        x = self.act(x)
        x = self.downsample(x)

        return x


# Adapted from https://github.com/jik876/hifi-gan under the MIT license.
#   LICENSE is in incl_licenses directory.


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def build_env(config, config_name, path):
    t_path = os.path.join(path, config_name)
    if config != t_path:
        os.makedirs(path, exist_ok=True)
        shutil.copyfile(config, os.path.join(path, config_name))


# Adapted from https://github.com/jik876/hifi-gan under the MIT license.
#   LICENSE is in incl_licenses directory.


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def apply_weight_norm(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        legacy_weight_norm(m)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print(f"Loading '{filepath}'")
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


# Copyright (c) 2024 NVIDIA CORPORATION.
#   Licensed under the MIT license.

# Adapted from https://github.com/jik876/hifi-gan under the MIT license.
#   LICENSE is in incl_licenses directory.


def load_hparams_from_json(path) -> AttrDict:
    with open(path) as f:
        data = f.read()
    return AttrDict(json.loads(data))


class AMPBlock1(torch.nn.Module):
    """
    AMPBlock applies Snake / SnakeBeta activation functions with trainable parameters that control periodicity, defined for each layer.
    AMPBlock1 has additional self.convs2 that contains additional Conv1d layers with a fixed dilation=1 followed by each layer in self.convs1

    Args:
        h (AttrDict): Hyperparameters.
        channels (int): Number of convolution channels.
        kernel_size (int): Size of the convolution kernel. Default is 3.
        dilation (tuple): Dilation rates for the convolutions. Each dilation layer has two convolutions. Default is (1, 3, 5).
        activation (str): Activation function type. Should be either 'snake' or 'snakebeta'. Default is None.
    """

    def __init__(
        self,
        h: AttrDict,
        channels: int,
        kernel_size: int = 3,
        dilation: tuple = (1, 3, 5),
        activation: str = None,
    ):
        super().__init__()

        self.h = h

        self.convs1 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        stride=1,
                        dilation=d,
                        padding=get_padding(kernel_size, d),
                    )
                )
                for d in dilation
            ]
        )
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        stride=1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                )
                for _ in range(len(dilation))
            ]
        )
        self.convs2.apply(init_weights)

        self.num_layers = len(self.convs1) + len(self.convs2)  # Total number of conv layers

        # Activation functions
        if activation == "snake":
            self.activations = nn.ModuleList(
                [
                    TorchActivation1d(activation=Snake(channels, alpha_logscale=h.snake_logscale))
                    for _ in range(self.num_layers)
                ]
            )
        elif activation == "snakebeta":
            self.activations = nn.ModuleList(
                [
                    TorchActivation1d(activation=SnakeBeta(channels, alpha_logscale=h.snake_logscale))
                    for _ in range(self.num_layers)
                ]
            )
        else:
            raise NotImplementedError(
                "activation incorrectly specified. check the config file and look for 'activation'."
            )

    def forward(self, x):
        acts1, acts2 = self.activations[::2], self.activations[1::2]
        for c1, c2, a1, a2 in zip(self.convs1, self.convs2, acts1, acts2):
            xt = a1(x)
            xt = c1(xt)
            xt = a2(xt)
            xt = c2(xt)
            x = xt + x

        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_parametrizations(l, "weight")
        for l in self.convs2:
            remove_parametrizations(l, "weight")


class AMPBlock2(torch.nn.Module):
    """
    AMPBlock applies Snake / SnakeBeta activation functions with trainable parameters that control periodicity, defined for each layer.
    Unlike AMPBlock1, AMPBlock2 does not contain extra Conv1d layers with fixed dilation=1

    Args:
        h (AttrDict): Hyperparameters.
        channels (int): Number of convolution channels.
        kernel_size (int): Size of the convolution kernel. Default is 3.
        dilation (tuple): Dilation rates for the convolutions. Each dilation layer has two convolutions. Default is (1, 3, 5).
        activation (str): Activation function type. Should be either 'snake' or 'snakebeta'. Default is None.
    """

    def __init__(
        self,
        h: AttrDict,
        channels: int,
        kernel_size: int = 3,
        dilation: tuple = (1, 3, 5),
        activation: str = None,
    ):
        super().__init__()

        self.h = h

        self.convs = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        stride=1,
                        dilation=d,
                        padding=get_padding(kernel_size, d),
                    )
                )
                for d in dilation
            ]
        )
        self.convs.apply(init_weights)

        self.num_layers = len(self.convs)  # Total number of conv layers

        # Activation functions
        if activation == "snake":
            self.activations = nn.ModuleList(
                [
                    TorchActivation1d(activation=Snake(channels, alpha_logscale=h.snake_logscale))
                    for _ in range(self.num_layers)
                ]
            )
        elif activation == "snakebeta":
            self.activations = nn.ModuleList(
                [
                    TorchActivation1d(activation=SnakeBeta(channels, alpha_logscale=h.snake_logscale))
                    for _ in range(self.num_layers)
                ]
            )
        else:
            raise NotImplementedError(
                "activation incorrectly specified. check the config file and look for 'activation'."
            )

    def forward(self, x):
        for c, a in zip(self.convs, self.activations):
            xt = a(x)
            xt = c(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)


TorchActivation1d = Activation1d


class BigVGANv2(
    torch.nn.Module,
    PyTorchModelHubMixin,
    library_name="bigvgan",
    repo_url="https://github.com/NVIDIA/BigVGAN",
    docs_url="https://github.com/NVIDIA/BigVGAN/blob/main/README.md",
    pipeline_tag="audio-to-audio",
    license="mit",
    tags=["neural-vocoder", "audio-generation", "arxiv:2206.04658"],
):
    """
    BigVGAN is a neural vocoder model that applies anti-aliased periodic activation for residual blocks (resblocks).

    Args:
        h (AttrDict): Hyperparameters.

    Note:
        - Ensure that the activation function is correctly specified in the hyperparameters (h.activation).
    """

    def __init__(self, h: AttrDict):
        super().__init__()
        self.h = h

        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)

        # Pre-conv
        self.conv_pre = weight_norm(Conv1d(h.num_mels, h.upsample_initial_channel, 7, 1, padding=3))

        # Define which AMPBlock to use. BigVGAN uses AMPBlock1 as default
        if h.resblock == "1":
            resblock_class = AMPBlock1
        elif h.resblock == "2":
            resblock_class = AMPBlock2
        else:
            raise ValueError(f"Incorrect resblock class specified in hyperparameters. Got {h.resblock}")

        # Transposed conv-based upsamplers. does not apply anti-aliasing
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            self.ups.append(
                nn.ModuleList(
                    [
                        weight_norm(
                            ConvTranspose1d(
                                h.upsample_initial_channel // (2**i),
                                h.upsample_initial_channel // (2 ** (i + 1)),
                                k,
                                u,
                                padding=(k - u) // 2,
                            )
                        )
                    ]
                )
            )

        # Residual blocks using anti-aliased multi-periodicity composition modules (AMP)
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)):
                self.resblocks.append(resblock_class(h, ch, k, d, activation=h.activation))

        # Post-conv
        activation_post = (
            Snake(ch, alpha_logscale=h.snake_logscale)
            if h.activation == "snake"
            else (SnakeBeta(ch, alpha_logscale=h.snake_logscale) if h.activation == "snakebeta" else None)
        )
        if activation_post is None:
            raise NotImplementedError(
                "activation incorrectly specified. check the config file and look for 'activation'."
            )

        self.activation_post = TorchActivation1d(activation=activation_post)

        # Whether to use bias for the final conv_post. Default to True for backward compatibility
        self.use_bias_at_final = h.get("use_bias_at_final", True)
        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3, bias=self.use_bias_at_final))

        # Weight initialization
        for i in range(len(self.ups)):
            self.ups[i].apply(init_weights)
        self.conv_post.apply(init_weights)

        # Final tanh activation. Defaults to True for backward compatibility
        self.use_tanh_at_final = h.get("use_tanh_at_final", True)

    def forward(self, x):
        # Pre-conv
        x = self.conv_pre(x)

        for i in range(self.num_upsamples):
            # Upsampling
            for i_up in range(len(self.ups[i])):
                x = self.ups[i][i_up](x)
            # AMP blocks
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

        # Post-conv
        x = self.activation_post(x)
        x = self.conv_post(x)
        # Final tanh activation
        if self.use_tanh_at_final:
            x = torch.tanh(x)
        else:
            x = torch.clamp(x, min=-1.0, max=1.0)  # Bound the output to [-1, 1]

        return x

    def remove_weight_norm(self):
        try:
            # print("Removing weight norm...")
            for l in self.ups:
                for l_i in l:
                    remove_parametrizations(l_i, "weight")
            for l in self.resblocks:
                l.remove_weight_norm()
            remove_parametrizations(self.conv_pre, "weight")
            remove_parametrizations(self.conv_post, "weight")
        except ValueError:
            print("[INFO] Model already removed weight norm. Skipping!")
            pass

    # Additional methods for huggingface_hub support
    def _save_pretrained(self, save_directory: Path) -> None:
        """Save weights and config.json from a Pytorch model to a local directory."""

        model_path = save_directory / "bigvgan_generator.pt"
        torch.save({"generator": self.state_dict()}, model_path)

        config_path = save_directory / "config.json"
        with open(config_path, "w") as config_file:
            json.dump(self.h, config_file, indent=4)

    @classmethod
    def _from_pretrained(
        cls,
        *,
        model_id: str,
        revision: str,
        cache_dir: str,
        force_download: bool,
        proxies: Optional[Dict],
        resume_download: bool,
        local_files_only: bool,
        token: Union[str, bool, None],
        map_location: str = "cpu",  # Additional argument
        strict: bool = False,  # Additional argument
        **model_kwargs,
    ):
        """Load Pytorch pretrained weights and return the loaded model."""

        # Download and load hyperparameters (h) used by BigVGAN
        if os.path.isdir(model_id):
            # print("Loading config.json from local directory")
            config_file = os.path.join(model_id, "config.json")
        else:
            config_file = hf_hub_download(
                repo_id=model_id,
                filename="config.json",
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                token=token,
                local_files_only=local_files_only,
            )
        h = load_hparams_from_json(config_file)

        model = cls(h)

        # Download and load pretrained generator weight
        if os.path.isdir(model_id):
            # print("Loading weights from local directory")
            model_file = os.path.join(model_id, "bigvgan_generator.pt")
        else:
            print(f"Loading weights from {model_id}")
            model_file = hf_hub_download(
                repo_id=model_id,
                filename="bigvgan_generator.pt",
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                token=token,
                local_files_only=local_files_only,
            )

        checkpoint_dict = torch.load(model_file, map_location=map_location, weights_only=True)

        try:
            model.load_state_dict(checkpoint_dict["generator"])
        except RuntimeError:
            print(
                f"[INFO] the pretrained checkpoint does not contain weight norm. Loading the checkpoint after removing weight norm!"
            )
            model.remove_weight_norm()
            model.load_state_dict(checkpoint_dict["generator"])

        return model


def _load_bigvgan_v2(
    vocoder_ckpt_path: str | None = None,
    *,
    vocoder_config: dict | None = None,
) -> BigVGANv2:
    """Load BigVGAN-v2 from a local directory (avoids HubMixin/huggingface_hub API drift)."""
    model_file = None if vocoder_ckpt_path is None else os.path.join(vocoder_ckpt_path, "bigvgan_generator.pt")
    if vocoder_config is None:
        if vocoder_ckpt_path is None:
            raise ValueError("Either vocoder_ckpt_path or vocoder_config is required")
        config_file = os.path.join(vocoder_ckpt_path, "config.json")
        if not os.path.isfile(config_file) or not os.path.isfile(model_file):
            raise FileNotFoundError(f"Expected config.json and bigvgan_generator.pt under {vocoder_ckpt_path}")
        h = load_hparams_from_json(config_file)
    else:
        if model_file is not None and not os.path.isfile(model_file):
            raise FileNotFoundError(f"BigVGAN checkpoint does not exist: {model_file}")
        h = AttrDict(vocoder_config)

    model = BigVGANv2(h)
    if vocoder_ckpt_path is not None:
        assert model_file is not None
        checkpoint_dict = torch.load(model_file, map_location="cpu", weights_only=True)
        try:
            model.load_state_dict(checkpoint_dict["generator"])
        except RuntimeError:
            model.remove_weight_norm()
            model.load_state_dict(checkpoint_dict["generator"])
        model.remove_weight_norm()
    else:
        model.remove_weight_norm()
    return model


class AutoEncoderModule(nn.Module):
    def __init__(
        self,
        *,
        vae_ckpt_path: str | None = None,
        vocoder_ckpt_path: str | None = None,
        vocoder_config: dict | None = None,
        mode: Literal["44k"],
        need_vae_encoder: bool = True,
        need_vae_decoder: bool = True,
    ):
        super().__init__()
        if mode != "44k":
            raise ValueError(f"Unknown model: {mode}")
        self.vae: VAE = VAE(data_dim=128, embed_dim=40, hidden_dim=512).eval()
        if vae_ckpt_path is not None:
            vae_state_dict = torch.load(vae_ckpt_path, weights_only=True, map_location="cpu")
            self.vae.load_state_dict(vae_state_dict)
        self.vae.remove_weight_norm()

        if need_vae_decoder:
            self.vocoder = _load_bigvgan_v2(
                vocoder_ckpt_path,
                vocoder_config=vocoder_config,
            ).eval()
        else:
            del self.vae.decoder

        if not need_vae_encoder:
            del self.vae.encoder

        for param in self.parameters():
            param.requires_grad = False

    def encode(self, x: torch.Tensor) -> DiagonalGaussianDistribution:
        return self.vae.encode(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.vae.decode(z)

    def vocode(self, spec: torch.Tensor) -> torch.Tensor:
        return self.vocoder(spec)


# Reference: # https://github.com/bytedance/Make-An-Audio-2


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5, *, norm_fn):
    return norm_fn(torch.clamp(x, min=clip_val) * C)


def spectral_normalize_torch(magnitudes, norm_fn):
    output = dynamic_range_compression_torch(magnitudes, norm_fn=norm_fn)
    return output


class MelConverter(nn.Module):
    def __init__(
        self,
        *,
        sampling_rate: float,
        n_fft: int,
        num_mels: int,
        hop_size: int,
        win_size: int,
        fmin: float,
        fmax: float,
        norm_fn,
    ):
        super().__init__()
        self.sampling_rate = sampling_rate
        self.n_fft = n_fft
        self.num_mels = num_mels
        self.hop_size = hop_size
        self.win_size = win_size
        self.fmin = fmin
        self.fmax = fmax
        self.norm_fn = norm_fn

        mel = librosa_mel_fn(
            sr=self.sampling_rate, n_fft=self.n_fft, n_mels=self.num_mels, fmin=self.fmin, fmax=self.fmax
        )
        mel_basis = torch.from_numpy(mel).float()
        hann_window = torch.hann_window(self.win_size)

        self.register_buffer("mel_basis", mel_basis)
        self.register_buffer("hann_window", hann_window)

    @property
    def device(self):
        return self.mel_basis.device

    def forward(self, waveform: torch.Tensor, center: bool = False) -> torch.Tensor:
        waveform = waveform.clamp(min=-1.0, max=1.0).to(self.device)

        waveform = torch.nn.functional.pad(
            waveform.unsqueeze(1),
            [int((self.n_fft - self.hop_size) / 2), int((self.n_fft - self.hop_size) / 2)],
            mode="reflect",
        )
        waveform = waveform.squeeze(1)

        spec = torch.stft(
            waveform,
            self.n_fft,
            hop_length=self.hop_size,
            win_length=self.win_size,
            window=self.hann_window,
            center=center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )

        spec = torch.view_as_real(spec)
        spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9)).float()
        spec = torch.matmul(self.mel_basis, spec)
        spec = spectral_normalize_torch(spec, self.norm_fn)

        return spec


def get_mel_converter(mode: Literal["44k"]) -> MelConverter:
    return MelConverter(
        sampling_rate=44_100,
        n_fft=2048,
        num_mels=128,
        hop_size=512,
        win_size=2048,
        fmin=0,
        fmax=44100 / 2,
        norm_fn=torch.log,
    )


"""Audio feature / VAE facade."""


class MMAudioVAE(ModelMixin, ConfigMixin):
    ignore_for_config = ["tod_vae_ckpt", "bigvgan_vocoder_ckpt", "device"]

    @register_to_config
    def __init__(
        self,
        *,
        tod_vae_ckpt: str | PathLike[str] | None = None,
        bigvgan_vocoder_ckpt: str | PathLike[str] | None = None,
        vocoder_config: dict[str, Any] | None = None,
        mode: str = "44k",
        need_vae_encoder: bool = True,
        need_vae_decoder: bool = True,
        scaling_factor: float = 1.0,
        sample_rate: int = 44_100,
        downsample_factor: int = 1_024,
        device: str | torch.device | None = None,
    ) -> None:
        super().__init__()
        self.mel_converter = get_mel_converter(mode)
        self.tod = AutoEncoderModule(
            vae_ckpt_path=tod_vae_ckpt,
            vocoder_ckpt_path=bigvgan_vocoder_ckpt,
            vocoder_config=vocoder_config,
            mode=mode,
            need_vae_encoder=need_vae_encoder,
            need_vae_decoder=need_vae_decoder,
        )
        self.scaling_factor = float(scaling_factor)
        self.sample_rate = int(sample_rate)
        self.downsample_factor = int(downsample_factor)
        if device is not None:
            self.to(device)

    def compile(self):
        self.decode = torch.compile(self.decode)
        self.vocode = torch.compile(self.vocode)

    def train(self, mode: bool = True) -> MMAudioVAE:
        return super().train(False)

    def encode_audio(self, x) -> DiagonalGaussianDistribution:
        assert self.tod is not None, "VAE is not loaded"
        mel = self.mel_converter(x)
        return self.tod.encode(mel)

    def vocode(self, mel: torch.Tensor) -> torch.Tensor:
        assert self.tod is not None, "VAE is not loaded"
        return self.tod.vocode(mel)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        assert self.tod is not None, "VAE is not loaded"
        return self.tod.decode(z)

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @apply_forward_hook
    def wrapped_decode(self, z):
        mel_decoded = self.decode(z.to(dtype=self.dtype))
        return self.vocode(mel_decoded)

    @apply_forward_hook
    def wrapped_encode(self, audio):
        dist = self.encode_audio(audio.to(dtype=self.dtype))
        return dist.mean


__all__ = ["MMAudioVAE"]
