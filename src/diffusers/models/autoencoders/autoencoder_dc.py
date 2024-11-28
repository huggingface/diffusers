# Copyright 2024 MIT, Tsinghua University, NVIDIA CORPORATION and The HuggingFace Team.
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

from typing import Any, Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm2d

from ...configuration_utils import ConfigMixin, register_to_config
from ..activations import get_activation
from ..modeling_utils import ModelMixin
from ..normalization import RMSNormNd


def val2tuple(x: list | tuple | Any, min_len: int = 1) -> tuple:
    x = list(x) if isinstance(x, (list, tuple)) else [x]
    # repeat elements if necessary
    if len(x) > 0:
        x.extend([x[-1] for _ in range(min_len - len(x))])
    return tuple(x)


def build_norm(name: Optional[str] = "bn2d", num_features: Optional[int] = None) -> Optional[nn.Module]:
    if name is None:
        norm = None
    elif name == "rms2d":
        norm = RMSNormNd(num_features, eps=1e-5, elementwise_affine=True, bias=True, channel_dim=1)
    elif name == "bn2d":
        norm = BatchNorm2d(num_features=num_features)
    else:
        raise ValueError(f"norm {name} is not supported")
    return norm


class DCConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        dilation=1,
        groups=1,
        use_bias=False,
        dropout=0,
        norm="bn2d",
        act_func="relu",
    ):
        super(DCConv2d, self).__init__()

        padding = kernel_size // 2
        padding *= dilation

        self.dropout = nn.Dropout2d(dropout, inplace=False) if dropout > 0 else None
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride),
            padding=padding,
            dilation=(dilation, dilation),
            groups=groups,
            bias=use_bias,
        )
        self.norm = build_norm(norm, num_features=out_channels)
        self.act = get_activation(act_func) if act_func is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x


class DownsamplePixelUnshuffleChannelAveraging(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        factor: int,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor = factor
        assert in_channels * factor**2 % out_channels == 0
        self.group_size = in_channels * factor**2 // out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pixel_unshuffle(x, self.factor)
        x = x.unflatten(1, (-1, self.group_size))
        x = x.mean(dim=2)
        return x


class UpsampleChannelDuplicatingPixelUnshuffle(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        factor: int,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor = factor
        assert out_channels * factor**2 % in_channels == 0
        self.repeats = out_channels * factor**2 // in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.repeat_interleave(self.repeats, dim=1)
        x = F.pixel_shuffle(x, self.factor)
        return x


class GLUMBConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        mid_channels=None,
        expand_ratio=6,
        use_bias=False,
        norm=(None, None, "ln2d"),
        act_func=("silu", "silu", None),
    ):
        super().__init__()
        use_bias = val2tuple(use_bias, 3)
        norm = val2tuple(norm, 3)
        act_func = val2tuple(act_func, 3)

        mid_channels = round(in_channels * expand_ratio) if mid_channels is None else mid_channels

        self.glu_act = get_activation(act_func[1])
        self.conv_inverted = DCConv2d(
            in_channels,
            mid_channels * 2,
            1,
            use_bias=use_bias[0],
            norm=norm[0],
            act_func=act_func[0],
        )
        self.conv_depth = DCConv2d(
            mid_channels * 2,
            mid_channels * 2,
            kernel_size,
            stride=stride,
            groups=mid_channels * 2,
            use_bias=use_bias[1],
            norm=norm[1],
            act_func=None,
        )
        self.conv_point = DCConv2d(
            mid_channels,
            out_channels,
            1,
            use_bias=use_bias[2],
            norm=norm[2],
            act_func=act_func[2],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv_inverted(x)
        x = self.conv_depth(x)

        x, gate = torch.chunk(x, 2, dim=1)
        gate = self.glu_act(gate)
        x = x * gate

        x = self.conv_point(x)
        return x + residual


class ResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        mid_channels=None,
        expand_ratio=1,
        use_bias=False,
        norm=("bn2d", "bn2d"),
        act_func=("relu6", None),
    ):
        super().__init__()
        use_bias = val2tuple(use_bias, 2)
        norm = val2tuple(norm, 2)
        act_func = val2tuple(act_func, 2)

        mid_channels = round(in_channels * expand_ratio) if mid_channels is None else mid_channels

        self.conv1 = DCConv2d(
            in_channels,
            mid_channels,
            kernel_size,
            stride,
            use_bias=use_bias[0],
            norm=norm[0],
            act_func=act_func[0],
        )
        self.conv2 = DCConv2d(
            mid_channels,
            out_channels,
            kernel_size,
            1,
            use_bias=use_bias[1],
            norm=norm[1],
            act_func=act_func[1],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        return x + residual


class LiteMLA(nn.Module):
    r"""Lightweight multi-scale linear attention"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: Optional[int] = None,
        heads_ratio: float = 1.0,
        dim=8,
        use_bias=False,
        norm=(None, "bn2d"),
        act_func=(None, None),
        kernel_func="relu",
        scales: tuple[int, ...] = (5,),
        eps=1.0e-15,
    ):
        super(LiteMLA, self).__init__()
        self.eps = eps
        heads = int(in_channels // dim * heads_ratio) if heads is None else heads

        total_dim = heads * dim

        use_bias = val2tuple(use_bias, 2)
        norm = val2tuple(norm, 2)
        act_func = val2tuple(act_func, 2)

        self.dim = dim
        self.qkv = DCConv2d(
            in_channels,
            3 * total_dim,
            1,
            use_bias=use_bias[0],
            norm=norm[0],
            act_func=act_func[0],
        )
        self.aggreg = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        3 * total_dim,
                        3 * total_dim,
                        scale,
                        padding=scale // 2,
                        groups=3 * total_dim,
                        bias=use_bias[0],
                    ),
                    nn.Conv2d(3 * total_dim, 3 * total_dim, 1, groups=3 * heads, bias=use_bias[0]),
                )
                for scale in scales
            ]
        )
        self.kernel_func = get_activation(kernel_func)

        self.proj = DCConv2d(
            total_dim * (1 + len(scales)),
            out_channels,
            1,
            use_bias=use_bias[1],
            norm=norm[1],
            act_func=act_func[1],
        )

    def relu_linear_att(self, qkv: torch.Tensor) -> torch.Tensor:
        B, _, H, W = list(qkv.size())

        if qkv.dtype == torch.float16:
            qkv = qkv.float()

        qkv = torch.reshape(
            qkv,
            (
                B,
                -1,
                3 * self.dim,
                H * W,
            ),
        )
        q, k, v = (
            qkv[:, :, 0 : self.dim],
            qkv[:, :, self.dim : 2 * self.dim],
            qkv[:, :, 2 * self.dim :],
        )

        # lightweight linear attention
        q = self.kernel_func(q)
        k = self.kernel_func(k)

        # linear matmul
        trans_k = k.transpose(-1, -2)

        v = F.pad(v, (0, 0, 0, 1), mode="constant", value=1)
        vk = torch.matmul(v, trans_k)
        out = torch.matmul(vk, q)
        if out.dtype == torch.bfloat16:
            out = out.float()
        out = out[:, :, :-1] / (out[:, :, -1:] + self.eps)

        out = torch.reshape(out, (B, -1, H, W))
        return out

    def relu_quadratic_att(self, qkv: torch.Tensor) -> torch.Tensor:
        B, _, H, W = list(qkv.size())

        qkv = torch.reshape(
            qkv,
            (
                B,
                -1,
                3 * self.dim,
                H * W,
            ),
        )
        q, k, v = (
            qkv[:, :, 0 : self.dim],
            qkv[:, :, self.dim : 2 * self.dim],
            qkv[:, :, 2 * self.dim :],
        )

        q = self.kernel_func(q)
        k = self.kernel_func(k)

        att_map = torch.matmul(k.transpose(-1, -2), q)  # b h n n
        original_dtype = att_map.dtype
        if original_dtype in [torch.float16, torch.bfloat16]:
            att_map = att_map.float()
        att_map = att_map / (torch.sum(att_map, dim=2, keepdim=True) + self.eps)  # b h n n
        att_map = att_map.to(original_dtype)
        out = torch.matmul(v, att_map)  # b h d n

        out = torch.reshape(out, (B, -1, H, W))
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        # generate multi-scale q, k, v
        qkv = self.qkv(x)
        multi_scale_qkv = [qkv]
        for op in self.aggreg:
            multi_scale_qkv.append(op(qkv))
        qkv = torch.cat(multi_scale_qkv, dim=1)

        H, W = list(qkv.size())[-2:]
        if H * W > self.dim:
            out = self.relu_linear_att(qkv).to(qkv.dtype)
        else:
            out = self.relu_quadratic_att(qkv)
        out = self.proj(out)

        return out + residual


class EfficientViTBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        heads_ratio: float = 1.0,
        dim=32,
        expand_ratio: float = 4,
        scales: tuple[int, ...] = (5,),
        norm: str = "bn2d",
        act_func: str = "hswish",
        context_module: str = "LiteMLA",
        local_module: str = "MBConv",
    ):
        super().__init__()
        if context_module == "LiteMLA":
            self.context_module = LiteMLA(
                in_channels=in_channels,
                out_channels=in_channels,
                heads_ratio=heads_ratio,
                dim=dim,
                norm=(None, norm),
                scales=scales,
            )
        else:
            raise ValueError(f"context_module {context_module} is not supported")
        if local_module == "GLUMBConv":
            self.local_module = GLUMBConv(
                in_channels=in_channels,
                out_channels=in_channels,
                expand_ratio=expand_ratio,
                use_bias=(True, True, False),
                norm=(None, None, norm),
                act_func=(act_func, act_func, None),
            )
        else:
            raise NotImplementedError(f"local_module {local_module} is not supported")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.context_module(x)
        x = self.local_module(x)
        return x


#################################################################################
#                             Functional Blocks                                 #
#################################################################################


class ResidualBlock(nn.Module):
    def __init__(
        self,
        main: Optional[nn.Module],
        shortcut: Optional[nn.Module],
        post_act=None,
        pre_norm: Optional[nn.Module] = None,
    ):
        super().__init__()

        self.pre_norm = pre_norm
        self.main = main
        self.shortcut = shortcut
        self.post_act = get_activation(post_act) if post_act is not None else None

    def forward_main(self, x: torch.Tensor) -> torch.Tensor:
        if self.pre_norm is None:
            return self.main(x)
        else:
            return self.main(self.pre_norm(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.main is None:
            res = x
        elif self.shortcut is None:
            res = self.forward_main(x)
        else:
            res = self.forward_main(x) + self.shortcut(x)
            if self.post_act:
                res = self.post_act(res)
        return res


def build_stage_main(
    width: int, depth: int, block_type: str | list[str], norm: str, act: str, input_width: int
) -> list[nn.Module]:
    assert isinstance(block_type, str) or (isinstance(block_type, list) and depth == len(block_type))
    stage = []
    for d in range(depth):
        current_block_type = block_type[d] if isinstance(block_type, list) else block_type

        in_channels = width if d > 0 else input_width
        out_channels = width

        if current_block_type == "ResBlock":
            assert in_channels == out_channels
            block = ResBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                use_bias=(True, False),
                norm=(None, norm),
                act_func=(act, None),
            )
        elif current_block_type == "EViT_GLU":
            assert in_channels == out_channels
            block = EfficientViTBlock(in_channels, norm=norm, act_func=act, local_module="GLUMBConv", scales=())
        elif current_block_type == "EViTS5_GLU":
            assert in_channels == out_channels
            block = EfficientViTBlock(in_channels, norm=norm, act_func=act, local_module="GLUMBConv", scales=(5,))
        else:
            raise ValueError(f"block_type {current_block_type} is not supported")

        stage.append(block)
    return stage


class DownsamplePixelUnshuffle(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        factor: int,
    ):
        super().__init__()
        self.factor = factor
        out_ratio = factor**2
        assert out_channels % out_ratio == 0
        self.conv = DCConv2d(
            in_channels=in_channels,
            out_channels=out_channels // out_ratio,
            kernel_size=kernel_size,
            use_bias=True,
            norm=None,
            act_func=None,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = F.pixel_unshuffle(x, self.factor)
        return x


class DCDownBlock2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        downsample: bool = False,
        shortcut: bool = True,
    ) -> None:
        super().__init__()

        self.downsample = downsample
        self.factor = 2
        self.stride = 1 if downsample else 2

        out_ratio = self.factor**2
        if downsample:
            assert out_channels % out_ratio == 0
            out_channels = out_channels // out_ratio

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=self.stride,
            padding=kernel_size // 2,
        )

        self.shortcut = None
        if shortcut:
            self.shortcut = DownsamplePixelUnshuffleChannelAveraging(
                in_channels=in_channels, out_channels=out_channels, factor=2
            )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = self.conv(hidden_states)
        if self.downsample:
            x = F.pixel_unshuffle(x, self.factor)
        if self.shortcut is not None:
            hidden_states = x + self.shortcut(hidden_states)
        else:
            hidden_states = x
        return hidden_states


class DCUpBlock2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        interpolate: bool = False,
        shortcut: bool = True,
        interpolation_mode: str = "nearest",
    ) -> None:
        super().__init__()

        self.interpolate = interpolate
        self.interpolation_mode = interpolation_mode
        self.factor = 2
        self.stride = 1

        out_ratio = self.factor**2
        if not interpolate:
            out_channels = out_channels * out_ratio

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=self.stride,
            padding=kernel_size // 2,
        )

        self.shortcut = None
        if shortcut:
            self.shortcut = UpsampleChannelDuplicatingPixelUnshuffle(
                in_channels=in_channels, out_channels=out_channels, factor=2
            )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.interpolate:
            x = F.interpolate(hidden_states, scale_factor=self.factor, mode=self.interpolation_mode)
            x = self.conv(x)
        else:
            x = self.conv(hidden_states)
            x = F.pixel_shuffle(x, self.factor)

        if self.shortcut is not None:
            hidden_states = x + self.shortcut(hidden_states)
        else:
            hidden_states = x

        return hidden_states


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        latent_channels: int,
        block_out_channels: list[int] = [128, 256, 512, 512, 1024, 1024],
        layers_per_block: list[int] = [2, 2, 2, 2, 2, 2],
        block_type: str | list[str] = "ResBlock",
        downsample_block_type: str = "ConvPixelUnshuffle",
    ):
        super().__init__()
        num_stages = len(block_out_channels)
        self.num_stages = num_stages
        assert len(layers_per_block) == num_stages
        assert len(block_out_channels) == num_stages
        assert isinstance(block_type, str) or (isinstance(block_type, list) and len(block_type) == num_stages)

        factor = 1 if layers_per_block[0] > 0 else 2

        if factor == 1:
            self.conv_in = nn.Conv2d(
                in_channels,
                block_out_channels[0] if layers_per_block[0] > 0 else block_out_channels[1],
                kernel_size=3,
                stride=1,
                padding=1,
            )
        elif factor == 2:
            self.conv_in = DCDownBlock2d(
                in_channels=in_channels,
                out_channels=block_out_channels[0] if layers_per_block[0] > 0 else block_out_channels[1],
                downsample=downsample_block_type == "ConvPixelUnshuffle",
                shortcut=False,
            )
        else:
            raise

        stages = []
        for stage_id, (width, depth) in enumerate(zip(block_out_channels, layers_per_block)):
            stage_block_type = block_type[stage_id] if isinstance(block_type, list) else block_type
            current_stage = build_stage_main(
                width=width, depth=depth, block_type=stage_block_type, norm="rms2d", act="silu", input_width=width
            )
            if stage_id < num_stages - 1 and depth > 0:
                downsample_block = DCDownBlock2d(
                    in_channels=width,
                    out_channels=block_out_channels[stage_id + 1],
                    downsample=downsample_block_type == "ConvPixelUnshuffle",
                    shortcut=True,
                )
                current_stage.append(downsample_block)
            stages.append(nn.Sequential(*current_stage))
        self.stages = nn.ModuleList(stages)

        self.conv_out = nn.Conv2d(
            block_out_channels[-1],
            latent_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.norm_out = DownsamplePixelUnshuffleChannelAveraging(
            in_channels=block_out_channels[-1], out_channels=latent_channels, factor=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(x)
        for stage in self.stages:
            x = stage(x)
        x = self.conv_out(x) + self.norm_out(x)
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        latent_channels: int,
        block_out_channels: list[int] = [128, 256, 512, 512, 1024, 1024],
        layers_per_block: list[int] = [2, 2, 2, 2, 2, 2],
        block_type: str | list[str] = "ResBlock",
        norm: str | list[str] = "rms2d",
        act: str | list[str] = "silu",
        upsample_block_type: str = "ConvPixelShuffle",
        upsample_shortcut: str = "duplicating",
    ):
        super().__init__()
        num_stages = len(block_out_channels)
        self.num_stages = num_stages
        assert len(layers_per_block) == num_stages
        assert len(block_out_channels) == num_stages
        assert isinstance(block_type, str) or (isinstance(block_type, list) and len(block_type) == num_stages)
        assert isinstance(norm, str) or (isinstance(norm, list) and len(norm) == num_stages)
        assert isinstance(act, str) or (isinstance(act, list) and len(act) == num_stages)

        self.conv_in = nn.Conv2d(latent_channels, block_out_channels[-1], kernel_size=3, stride=1, padding=1)
        self.norm_in = UpsampleChannelDuplicatingPixelUnshuffle(
            in_channels=latent_channels, out_channels=block_out_channels[-1], factor=1
        )

        stages = []
        for stage_id, (width, depth) in reversed(list(enumerate(zip(block_out_channels, layers_per_block)))):
            current_stage = []
            if stage_id < num_stages - 1 and depth > 0:
                upsample_block = DCUpBlock2d(
                    block_out_channels[stage_id + 1],
                    width,
                    interpolate=upsample_block_type == "InterpolateConv",
                    shortcut=upsample_shortcut,
                )
                current_stage.append(upsample_block)

            stage_block_type = block_type[stage_id] if isinstance(block_type, list) else block_type
            stage_norm = norm[stage_id] if isinstance(norm, list) else norm
            stage_act = act[stage_id] if isinstance(act, list) else act
            current_stage.extend(
                build_stage_main(
                    width=width,
                    depth=depth,
                    block_type=stage_block_type,
                    norm=stage_norm,
                    act=stage_act,
                    input_width=width,
                )
            )
            stages.insert(0, nn.Sequential(*current_stage))
        self.stages = nn.ModuleList(stages)

        factor = 1 if layers_per_block[0] > 0 else 2

        self.norm_out = RMSNormNd(
            block_out_channels[0] if layers_per_block[0] > 0 else block_out_channels[1],
            eps=1e-5,
            elementwise_affine=True,
            bias=True,
            channel_dim=1,
        )
        self.conv_act = nn.ReLU()
        self.conv_out = None

        if factor == 1:
            self.conv_out = nn.Conv2d(
                block_out_channels[0] if layers_per_block[0] > 0 else block_out_channels[1],
                in_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            )
        else:
            self.conv_out = DCUpBlock2d(
                block_out_channels[0] if layers_per_block[0] > 0 else block_out_channels[1],
                in_channels,
                interpolate=upsample_block_type == "InterpolateConv",
                shortcut=False,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(x) + self.norm_in(x)
        for stage in reversed(self.stages):
            x = stage(x)
        x = self.norm_out(x)
        x = self.conv_act(x)
        x = self.conv_out(x)
        return x


class AutoencoderDC(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 32,
        block_out_channels: Tuple[int, ...] = (128, 256, 512, 512, 1024, 1024),
        encoder_layers_per_block: Tuple[int] = (2, 2, 2, 3, 3, 3),
        decoder_layers_per_block: Tuple[int] = (3, 3, 3, 3, 3, 3),
        encoder_block_type: str | list[str] = "ResBlock",
        downsample_block_type: str = "ConvPixelUnshuffle",
        decoder_block_type: str | list[str] = "ResBlock",
        decoder_norm: str = "rms2d",
        decoder_act: str = "silu",
        upsample_block_type: str = "ConvPixelShuffle",
        scaling_factor: Optional[float] = None,
        **kwargs,
    ):
        super().__init__()
        self.encoder = Encoder(
            in_channels=in_channels,
            latent_channels=latent_channels,
            block_out_channels=block_out_channels,
            layers_per_block=encoder_layers_per_block,
            block_type=encoder_block_type,
            downsample_block_type=downsample_block_type,
        )
        self.decoder = Decoder(
            in_channels=in_channels,
            latent_channels=latent_channels,
            block_out_channels=block_out_channels,
            layers_per_block=decoder_layers_per_block,
            block_type=decoder_block_type,
            norm=decoder_norm,
            act=decoder_act,
            upsample_block_type=upsample_block_type,
        )

    @property
    def spatial_compression_ratio(self) -> int:
        return 2 ** (self.decoder.num_stages - 1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        return x

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.decoder(x)
        return x

    def forward(self, x: torch.Tensor, global_step: int) -> torch.Tensor:
        x = self.encoder(x)
        x = self.decoder(x)
        return x, torch.tensor(0), {}


def dc_ae_f32c32(name: str) -> dict:
    if name in ["dc-ae-f32c32-in-1.0", "dc-ae-f32c32-mix-1.0"]:
        cfg = {
            "latent_channels": 32,
            "encoder_block_type": ["ResBlock", "ResBlock", "ResBlock", "EViT_GLU", "EViT_GLU", "EViT_GLU"],
            "block_out_channels": [128, 256, 512, 512, 1024, 1024],
            "encoder_layers_per_block": [0, 4, 8, 2, 2, 2],
            "decoder_block_type": ["ResBlock", "ResBlock", "ResBlock", "EViT_GLU", "EViT_GLU", "EViT_GLU"],
            "decoder_layers_per_block": [0, 5, 10, 2, 2, 2],
            "decoder_norm": ["bn2d", "bn2d", "bn2d", "rms2d", "rms2d", "rms2d"],
            "decoder_act": ["relu", "relu", "relu", "silu", "silu", "silu"],
        }
    elif name in ["dc-ae-f32c32-sana-1.0"]:
        cfg = {
            "latent_channels": 32,
            "encoder_block_type": ["ResBlock", "ResBlock", "ResBlock", "EViTS5_GLU", "EViTS5_GLU", "EViTS5_GLU"],
            "block_out_channels": [128, 256, 512, 512, 1024, 1024],
            "encoder_layers_per_block": [2, 2, 2, 3, 3, 3],
            "downsample_block_type": "Conv",
            "decoder_block_type": ["ResBlock", "ResBlock", "ResBlock", "EViTS5_GLU", "EViTS5_GLU", "EViTS5_GLU"],
            "decoder_layers_per_block": [3, 3, 3, 3, 3, 3],
            "upsample_block_type": "InterpolateConv",
            "scaling_factor": 0.41407,
        }
    else:
        raise NotImplementedError
    return cfg


def dc_ae_f64c128(
    name: str,
) -> dict:
    if name in ["dc-ae-f64c128-in-1.0", "dc-ae-f64c128-mix-1.0"]:
        cfg = {
            "latent_channels": 128,
            "encoder_block_type": ["ResBlock", "ResBlock", "ResBlock", "EViT_GLU", "EViT_GLU", "EViT_GLU", "EViT_GLU"],
            "block_out_channels": [128, 256, 512, 512, 1024, 1024, 2048],
            "encoder_layers_per_block": [0, 4, 8, 2, 2, 2, 2],
            "decoder_block_type": ["ResBlock", "ResBlock", "ResBlock", "EViT_GLU", "EViT_GLU", "EViT_GLU", "EViT_GLU"],
            "decoder_layers_per_block": [0, 5, 10, 2, 2, 2, 2],
            "decoder_norm": ["bn2d", "bn2d", "bn2d", "rms2d", "rms2d", "rms2d", "rms2d"],
            "decoder_act": ["relu", "relu", "relu", "silu", "silu", "silu", "silu"],
        }
    else:
        raise NotImplementedError
    return cfg


def dc_ae_f128c512(
    name: str,
) -> dict:
    if name in ["dc-ae-f128c512-in-1.0", "dc-ae-f128c512-mix-1.0"]:
        cfg = {
            "latent_channels": 512,
            "encoder_block_type": [
                "ResBlock",
                "ResBlock",
                "ResBlock",
                "EViT_GLU",
                "EViT_GLU",
                "EViT_GLU",
                "EViT_GLU",
                "EViT_GLU",
            ],
            "block_out_channels": [128, 256, 512, 512, 1024, 1024, 2048, 2048],
            "encoder_layers_per_block": [0, 4, 8, 2, 2, 2, 2, 2],
            "decoder_block_type": [
                "ResBlock",
                "ResBlock",
                "ResBlock",
                "EViT_GLU",
                "EViT_GLU",
                "EViT_GLU",
                "EViT_GLU",
                "EViT_GLU",
            ],
            "decoder_layers_per_block": [0, 5, 10, 2, 2, 2, 2, 2],
            "decoder_norm": ["bn2d", "bn2d", "bn2d", "rms2d", "rms2d", "rms2d", "rms2d", "rms2d"],
            "decoder_act": ["relu", "relu", "relu", "silu", "silu", "silu", "silu", "silu"],
        }
    else:
        raise NotImplementedError
    return cfg


REGISTERED_DCAE_MODEL: dict[str, Callable] = {
    "dc-ae-f32c32-in-1.0": dc_ae_f32c32,
    "dc-ae-f64c128-in-1.0": dc_ae_f64c128,
    "dc-ae-f128c512-in-1.0": dc_ae_f128c512,
    #################################################################################################
    "dc-ae-f32c32-mix-1.0": dc_ae_f32c32,
    "dc-ae-f64c128-mix-1.0": dc_ae_f64c128,
    "dc-ae-f128c512-mix-1.0": dc_ae_f128c512,
    #################################################################################################
    "dc-ae-f32c32-sana-1.0": dc_ae_f32c32,
}
