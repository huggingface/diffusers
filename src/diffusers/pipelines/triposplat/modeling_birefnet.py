# Copyright 2026 The HuggingFace Team. All rights reserved.
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
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from ...configuration_utils import ConfigMixin, register_to_config
from ...loaders import FromOriginalModelMixin
from ...models.attention import AttentionMixin, AttentionModuleMixin
from ...models.attention_dispatch import dispatch_attention_fn
from ...models.modeling_utils import ModelMixin
from ...utils import BaseOutput, requires_backends


@dataclass
class BiRefNetOutput(BaseOutput):
    """Foreground probabilities predicted by BiRefNet.

    Args:
        sample (`torch.Tensor`):
            Probabilities between zero and one, shaped `(batch, 1, height, width)`.
    """

    sample: torch.Tensor


class BiRefNetAttnProcessor:
    _attention_backend = None
    _parallel_config = None

    def __call__(
        self, attn: "BiRefNetWindowAttention", hidden_states: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        batch, length, channels = hidden_states.shape
        query, key, value = (
            attn.qkv(hidden_states).reshape(batch, length, 3, attn.num_heads, channels // attn.num_heads).unbind(2)
        )
        bias = attn.relative_position_bias_table[attn.relative_position_index.reshape(-1)]
        bias = bias.reshape(length, length, attn.num_heads).permute(2, 0, 1)[None]
        if mask is not None:
            bias = bias + mask[:, None]
            bias = bias.repeat(batch // mask.shape[0], 1, 1, 1)
        hidden_states = dispatch_attention_fn(
            query, key, value, attn_mask=bias, backend=self._attention_backend, parallel_config=self._parallel_config
        )
        return attn.proj(hidden_states.reshape(batch, length, channels))


class BiRefNetSwinMlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class BiRefNetWindowAttention(nn.Module, AttentionModuleMixin):
    _default_processor_cls = BiRefNetAttnProcessor
    _available_processors = [BiRefNetAttnProcessor]
    _supports_qkv_fusion = False

    def __init__(self, dim: int, window_size: tuple[int, int], num_heads: int) -> None:
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )
        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size[0] - 1
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        self.register_buffer("relative_position_index", relative_coords.sum(-1))
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.set_processor(self._default_processor_cls())

    def forward(self, hidden_states: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.processor(self, hidden_states, mask)


class BiRefNetSwinBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, window_size: int, shift_size: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = BiRefNetWindowAttention(dim, (window_size, window_size), num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = BiRefNetSwinMlp(dim, int(dim * mlp_ratio))

    def forward(self, x: torch.Tensor, mask_matrix: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x).view(B, H, W, C)
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        _, Hp, Wp, _ = x.shape
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None
        x_windows = shifted_x.view(
            B, Hp // self.window_size, self.window_size, Wp // self.window_size, self.window_size, C
        )
        x_windows = x_windows.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, self.window_size**2, C)
        attn_windows = self.attn(x_windows, mask=attn_mask).view(-1, self.window_size, self.window_size, C)
        shifted_x = attn_windows.view(
            B, Hp // self.window_size, Wp // self.window_size, self.window_size, self.window_size, C
        )
        shifted_x = shifted_x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, C)
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
        x = x.view(B, H * W, C)
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x


class BiRefNetPatchMerging(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, L, C = x.shape
        x = x.view(B, H, W, C)
        if H % 2 == 1 or W % 2 == 1:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1).view(B, -1, 4 * C)
        return self.reduction(self.norm(x))


class BiRefNetSwinBasicLayer(nn.Module):
    def __init__(
        self, dim: int, depth: int, num_heads: int, window_size: int, mlp_ratio: float = 4.0, downsample: bool = True
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth
        self.blocks = nn.ModuleList(
            [
                BiRefNetSwinBlock(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if i % 2 == 0 else window_size // 2,
                    mlp_ratio=mlp_ratio,
                )
                for i in range(depth)
            ]
        )
        self.downsample = BiRefNetPatchMerging(dim) if downsample else None

    def forward(self, x: torch.Tensor, H: int, W: int) -> tuple[torch.Tensor, int, int, torch.Tensor, int, int]:
        Hp = int(math.ceil(H / self.window_size)) * self.window_size
        Wp = int(math.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mask_windows = img_mask.view(
            1, Hp // self.window_size, self.window_size, Wp // self.window_size, self.window_size, 1
        )
        mask_windows = mask_windows.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, self.window_size**2)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = (
            attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0)).to(x.dtype)
        )
        for blk in self.blocks:
            x = blk(x, attn_mask, H, W)
        if self.downsample is not None:
            x_down = self.downsample(x, H, W)
            Wh, Ww = ((H + 1) // 2, (W + 1) // 2)
            return (x, H, W, x_down, Wh, Ww)
        return (x, H, W, x, H, W)


class BiRefNetSwinPatchEmbed(nn.Module):
    def __init__(self, patch_size: int = 4, in_channels: int = 3, embed_dim: int = 192) -> None:
        super().__init__()
        self.patch_size = (patch_size, patch_size)
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, H, W = x.shape
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))
        x = self.proj(x)
        Wh, Ww = (x.size(2), x.size(3))
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)


class BiRefNetSwinLarge(nn.Module):
    def __init__(self, embed_dim: int, depths: tuple[int, ...], num_heads: tuple[int, ...], window_size: int) -> None:
        super().__init__()
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_embed = BiRefNetSwinPatchEmbed(patch_size=4, in_channels=3, embed_dim=embed_dim)
        self.layers = nn.ModuleList(
            [
                BiRefNetSwinBasicLayer(
                    dim=int(embed_dim * 2**i),
                    depth=depths[i],
                    num_heads=num_heads[i],
                    window_size=window_size,
                    downsample=i < self.num_layers - 1,
                )
                for i in range(self.num_layers)
            ]
        )
        num_features = [int(embed_dim * 2**i) for i in range(self.num_layers)]
        self.num_features = num_features
        for i in range(self.num_layers):
            self.add_module(f"norm{i}", nn.LayerNorm(num_features[i]))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        x = self.patch_embed(x)
        Wh, Ww = (x.size(2), x.size(3))
        x = x.flatten(2).transpose(1, 2)
        outs = []
        for i in range(self.num_layers):
            x_out, H, W, x, Wh, Ww = self.layers[i](x, Wh, Ww)
            norm_layer = getattr(self, f"norm{i}")
            x_out = norm_layer(x_out)
            out = x_out.view(-1, H, W, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
            outs.append(out)
        return tuple(outs)


class BiRefNetDeformableConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int] = 3,
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 1,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.padding = padding
        self.offset_conv = nn.Conv2d(
            in_channels,
            2 * kernel_size[0] * kernel_size[1],
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True,
        )
        self.modulator_conv = nn.Conv2d(
            in_channels,
            1 * kernel_size[0] * kernel_size[1],
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True,
        )
        convolution = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias
        )
        self.weight = convolution.weight
        self.bias = convolution.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from torchvision.ops import deform_conv2d

        offset = self.offset_conv(x)
        modulator = 2.0 * torch.sigmoid(self.modulator_conv(x))
        dtype = x.dtype
        weight, bias = self.weight, self.bias
        if dtype == torch.bfloat16 or (x.device.type == "cpu" and dtype == torch.float16):
            x, offset, modulator, weight = x.float(), offset.float(), modulator.float(), weight.float()
            bias = bias.float() if bias is not None else None
        return deform_conv2d(
            input=x, offset=offset, weight=weight, bias=bias, padding=self.padding, mask=modulator, stride=self.stride
        ).to(dtype)


class BiRefNetASPPModuleDeformable(nn.Module):
    def __init__(self, in_channels: int, planes: int, kernel_size: int, padding: int) -> None:
        super().__init__()
        self.atrous_conv = BiRefNetDeformableConv2d(
            in_channels, planes, kernel_size=kernel_size, stride=1, padding=padding, bias=False
        )
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.atrous_conv(x)))


class BiRefNetASPPDeformable(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        out_channels = in_channels
        inter = 256
        self.aspp1 = BiRefNetASPPModuleDeformable(in_channels, inter, 1, padding=0)
        self.aspp_deforms = nn.ModuleList(
            [BiRefNetASPPModuleDeformable(in_channels, inter, k, padding=k // 2) for k in (1, 3, 7)]
        )
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, inter, 1, stride=1, bias=False),
            nn.BatchNorm2d(inter),
            nn.ReLU(inplace=True),
        )
        self.conv1 = nn.Conv2d(inter * (2 + len(self.aspp_deforms)), out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.aspp1(x)
        x_aspp_deforms = [m(x) for m in self.aspp_deforms]
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x1.size()[2:], mode="bilinear", align_corners=True)
        y = torch.cat((x1, *x_aspp_deforms, x5), dim=1)
        return self.relu(self.bn1(self.conv1(y)))


class BiRefNetBasicDecBlk(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, inter_channels: int = 64) -> None:
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, inter_channels, 3, 1, padding=1)
        self.bn_in = nn.BatchNorm2d(inter_channels)
        self.relu_in = nn.ReLU(inplace=True)
        self.dec_att = BiRefNetASPPDeformable(in_channels=inter_channels)
        self.conv_out = nn.Conv2d(inter_channels, out_channels, 3, 1, padding=1)
        self.bn_out = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu_in(self.bn_in(self.conv_in(x)))
        x = self.dec_att(x)
        x = self.bn_out(self.conv_out(x))
        return x


class BiRefNetBasicLatBlk(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, 1, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class BiRefNetSimpleConvs(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, inter_channels: int = 64) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, inter_channels, 3, 1, 1)
        self.conv_out = nn.Conv2d(inter_channels, out_channels, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_out(self.conv1(x))


def BiRefNetimage2patches(image: torch.Tensor, patch_ref: torch.Tensor) -> torch.Tensor:
    """Stack non-overlapping image patches along the channel axis to match `patch_ref` spatial dimensions."""
    b, c, h_full, w_full = image.shape
    hg, wg = (h_full // patch_ref.shape[-2], w_full // patch_ref.shape[-1])
    h, w = (h_full // hg, w_full // wg)
    return image.view(b, c, hg, h, wg, w).permute(0, 1, 2, 4, 3, 5).reshape(b, c * hg * wg, h, w)


class BiRefNetBiRefNetDecoder(nn.Module):
    def __init__(self, channels: tuple[int, ...] = (3072, 1536, 768, 384)) -> None:
        super().__init__()
        c = channels
        self.ipt_blk5 = BiRefNetSimpleConvs(2**10 * 3, c[0] // 8, inter_channels=64)
        self.ipt_blk4 = BiRefNetSimpleConvs(2**8 * 3, c[0] // 8, inter_channels=64)
        self.ipt_blk3 = BiRefNetSimpleConvs(2**6 * 3, c[1] // 8, inter_channels=64)
        self.ipt_blk2 = BiRefNetSimpleConvs(2**4 * 3, c[2] // 8, inter_channels=64)
        self.ipt_blk1 = BiRefNetSimpleConvs(2**0 * 3, c[3] // 8, inter_channels=64)
        self.decoder_block4 = BiRefNetBasicDecBlk(c[0] + c[0] // 8, c[1])
        self.decoder_block3 = BiRefNetBasicDecBlk(c[1] + c[0] // 8, c[2])
        self.decoder_block2 = BiRefNetBasicDecBlk(c[2] + c[1] // 8, c[3])
        self.decoder_block1 = BiRefNetBasicDecBlk(c[3] + c[2] // 8, c[3] // 2)
        self.conv_out1 = nn.Sequential(nn.Conv2d(c[3] // 2 + c[3] // 8, 1, 1, 1, 0))
        self.lateral_block4 = BiRefNetBasicLatBlk(c[1], c[1])
        self.lateral_block3 = BiRefNetBasicLatBlk(c[2], c[2])
        self.lateral_block2 = BiRefNetBasicLatBlk(c[3], c[3])
        _N = 16

        def _gdt_branch(in_c):
            return nn.Sequential(nn.Conv2d(in_c, _N, 3, 1, 1), nn.BatchNorm2d(_N), nn.ReLU(inplace=True))

        self.gdt_convs_4 = _gdt_branch(c[1])
        self.gdt_convs_3 = _gdt_branch(c[2])
        self.gdt_convs_2 = _gdt_branch(c[3])

        def _head_1x1():
            return nn.Sequential(nn.Conv2d(_N, 1, 1, 1, 0))

        self.gdt_convs_attn_4 = _head_1x1()
        self.gdt_convs_attn_3 = _head_1x1()
        self.gdt_convs_attn_2 = _head_1x1()

    def forward(
        self, x: torch.Tensor, x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor, x4: torch.Tensor
    ) -> torch.Tensor:
        x4 = torch.cat((x4, self.ipt_blk5(BiRefNetimage2patches(x, x4)).to(x4.device)), 1)
        p4 = self.decoder_block4(x4)
        p4 = p4 * self.gdt_convs_attn_4(self.gdt_convs_4(p4)).sigmoid().to(p4.device)
        _p4 = F.interpolate(p4, size=x3.shape[2:], mode="bilinear", align_corners=True)
        _p3 = _p4 + self.lateral_block4(x3).to(_p4.device)
        _p3 = torch.cat((_p3, self.ipt_blk4(BiRefNetimage2patches(x, _p3)).to(_p3.device)), 1)
        p3 = self.decoder_block3(_p3)
        p3 = p3 * self.gdt_convs_attn_3(self.gdt_convs_3(p3)).sigmoid().to(p3.device)
        _p3 = F.interpolate(p3, size=x2.shape[2:], mode="bilinear", align_corners=True)
        _p2 = _p3 + self.lateral_block3(x2).to(_p3.device)
        _p2 = torch.cat((_p2, self.ipt_blk3(BiRefNetimage2patches(x, _p2)).to(_p2.device)), 1)
        p2 = self.decoder_block2(_p2)
        p2 = p2 * self.gdt_convs_attn_2(self.gdt_convs_2(p2)).sigmoid().to(p2.device)
        _p2 = F.interpolate(p2, size=x1.shape[2:], mode="bilinear", align_corners=True)
        _p1 = _p2 + self.lateral_block2(x1).to(_p2.device)
        _p1 = torch.cat((_p1, self.ipt_blk2(BiRefNetimage2patches(x, _p1)).to(_p1.device)), 1)
        _p1 = self.decoder_block1(_p1)
        _p1 = F.interpolate(_p1, size=x.shape[2:], mode="bilinear", align_corners=True)
        _p1 = torch.cat((_p1, self.ipt_blk1(BiRefNetimage2patches(x, _p1)).to(_p1.device)), 1)
        return self.conv_out1(_p1)


class BiRefNetModel(ModelMixin, ConfigMixin, AttentionMixin, FromOriginalModelMixin):
    """Optional BiRefNet foreground segmentation model used by TripoSplat.

    Parameters:
        embed_dim (`int`, defaults to `192`):
            Initial Swin backbone hidden width.
        depths (`tuple[int, ...]`, defaults to `(2, 2, 18, 2)`):
            Number of blocks in each Swin stage.
        num_heads (`tuple[int, ...]`, defaults to `(6, 12, 24, 48)`):
            Number of attention heads in each Swin stage.
        window_size (`int`, defaults to `12`):
            Side length of each attention window.
        sample_size (`int`, defaults to `1024`):
            Image side length used for foreground mask prediction.
    """

    _skip_layerwise_casting_patterns = ["norm", "patch_embed", "bn"]
    _repeated_blocks = ["BiRefNetSwinBlock"]
    _no_split_modules = ["BiRefNetSwinBlock", "BiRefNetBasicDecBlk", "BiRefNetDeformableConv2d"]

    @register_to_config
    def __init__(
        self,
        embed_dim: int = 192,
        depths: tuple[int, ...] = (2, 2, 18, 2),
        num_heads: tuple[int, ...] = (6, 12, 24, 48),
        window_size: int = 12,
        sample_size: int = 1024,
    ) -> None:
        super().__init__()
        requires_backends(self, ["torchvision"])
        self.bb = BiRefNetSwinLarge(embed_dim, depths, num_heads, window_size)
        self._CHANNELS = (embed_dim * 16, embed_dim * 8, embed_dim * 4, embed_dim * 2)
        cxt = list(self._CHANNELS[1:][::-1][-3:])
        self.squeeze_module = nn.Sequential(BiRefNetBasicDecBlk(self._CHANNELS[0] + sum(cxt), self._CHANNELS[0]))
        self.decoder = BiRefNetBiRefNetDecoder(channels=self._CHANNELS)
        self.eval()

    def forward(self, sample: torch.Tensor, return_dict: bool = True) -> BiRefNetOutput | tuple[torch.Tensor]:
        """
        Args:
            sample (`torch.Tensor`):
                Normalized RGB images of shape `(batch, 3, height, width)`.
            return_dict (`bool`, defaults to `True`):
                Whether to return a structured output or a tuple.

        Returns:
            `BiRefNetOutput` or `tuple`: Foreground probabilities of shape `(batch, 1, height, width)`.
        """
        x = sample
        x1, x2, x3, x4 = self.bb(x)
        B, C, H, W = x.shape
        x1_, x2_, x3_, x4_ = self.bb(F.interpolate(x, size=(H // 2, W // 2), mode="bilinear", align_corners=True))
        x1 = torch.cat([x1, F.interpolate(x1_, size=x1.shape[2:], mode="bilinear", align_corners=True)], 1)
        x2 = torch.cat([x2, F.interpolate(x2_, size=x2.shape[2:], mode="bilinear", align_corners=True)], 1)
        x3 = torch.cat([x3, F.interpolate(x3_, size=x3.shape[2:], mode="bilinear", align_corners=True)], 1)
        x4 = torch.cat([x4, F.interpolate(x4_, size=x4.shape[2:], mode="bilinear", align_corners=True)], 1)
        x4 = torch.cat(
            [
                F.interpolate(x1, size=x4.shape[2:], mode="bilinear", align_corners=True).to(x4.device),
                F.interpolate(x2, size=x4.shape[2:], mode="bilinear", align_corners=True).to(x4.device),
                F.interpolate(x3, size=x4.shape[2:], mode="bilinear", align_corners=True).to(x4.device),
                x4,
            ],
            1,
        )
        x4 = self.squeeze_module(x4)
        logits = self.decoder(sample, x1, x2, x3, x4)
        alpha = torch.sigmoid(logits)
        if not return_dict:
            return (alpha,)
        return BiRefNetOutput(sample=alpha)
