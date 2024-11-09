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

from typing import Any, Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm2d
from huggingface_hub import PyTorchModelHubMixin
import ipdb

from ...configuration_utils import ConfigMixin, register_to_config
from ..modeling_utils import ModelMixin

from ..activations import get_activation
from ..downsampling import ConvPixelUnshuffleDownsample2D, PixelUnshuffleChannelAveragingDownsample2D
from ..upsampling import ConvPixelShuffleUpsample2D, ChannelDuplicatingPixelUnshuffleUpsample2D, Upsample2D


__all__ = ["DCAE", "dc_ae_f32c32", "dc_ae_f64c128", "dc_ae_f128c512"]


def val2tuple(x: list | tuple | Any, min_len: int = 1) -> tuple:
    x = list(x) if isinstance(x, (list, tuple)) else [x]
    # repeat elements if necessary
    if len(x) > 0:
        x.extend([x[-1] for _ in range(min_len - len(x))])
    return tuple(x)


def get_same_padding(kernel_size: int | tuple[int, ...]) -> int | tuple[int, ...]:
    if isinstance(kernel_size, tuple):
        return tuple([get_same_padding(ks) for ks in kernel_size])
    else:
        assert kernel_size % 2 > 0, "kernel size should be odd number"
        return kernel_size // 2


class RMSNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x / torch.sqrt(torch.square(x.float()).mean(dim=1, keepdim=True) + self.eps)).to(x.dtype)
        if self.elementwise_affine:
            x = x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
        return x

def build_norm(name: Optional[str]="bn2d", num_features: Optional[int]=None) -> Optional[nn.Module]:
    if name is None:
        norm = None
    elif name == "rms2d":
        norm = RMSNorm2d(normalized_shape=num_features)
    elif name == "bn2d":
        norm = BatchNorm2d(num_features=num_features)
    else:
        raise ValueError(f"norm {name} is not supported")
    return norm


class ConvLayer(nn.Module):
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
        super(ConvLayer, self).__init__()

        padding = get_same_padding(kernel_size)
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
        self.inverted_conv = ConvLayer(
            in_channels,
            mid_channels * 2,
            1,
            use_bias=use_bias[0],
            norm=norm[0],
            act_func=act_func[0],
        )
        self.depth_conv = ConvLayer(
            mid_channels * 2,
            mid_channels * 2,
            kernel_size,
            stride=stride,
            groups=mid_channels * 2,
            use_bias=use_bias[1],
            norm=norm[1],
            act_func=None,
        )
        self.point_conv = ConvLayer(
            mid_channels,
            out_channels,
            1,
            use_bias=use_bias[2],
            norm=norm[2],
            act_func=act_func[2],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.inverted_conv(x)
        x = self.depth_conv(x)

        x, gate = torch.chunk(x, 2, dim=1)
        gate = self.glu_act(gate)
        x = x * gate

        x = self.point_conv(x)
        return x


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

        self.conv1 = ConvLayer(
            in_channels,
            mid_channels,
            kernel_size,
            stride,
            use_bias=use_bias[0],
            norm=norm[0],
            act_func=act_func[0],
        )
        self.conv2 = ConvLayer(
            mid_channels,
            out_channels,
            kernel_size,
            1,
            use_bias=use_bias[1],
            norm=norm[1],
            act_func=act_func[1],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        return x


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
        self.qkv = ConvLayer(
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
                        padding=get_same_padding(scale),
                        groups=3 * total_dim,
                        bias=use_bias[0],
                    ),
                    nn.Conv2d(3 * total_dim, 3 * total_dim, 1, groups=3 * heads, bias=use_bias[0]),
                )
                for scale in scales
            ]
        )
        self.kernel_func = get_activation(kernel_func)

        self.proj = ConvLayer(
            total_dim * (1 + len(scales)),
            out_channels,
            1,
            use_bias=use_bias[1],
            norm=norm[1],
            act_func=act_func[1],
        )

    @torch.autocast(device_type="cuda", enabled=False)
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

    @torch.autocast(device_type="cuda", enabled=False)
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

        return out


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
        super(EfficientViTBlock, self).__init__()
        if context_module == "LiteMLA":
            self.context_module = ResidualBlock(
                LiteMLA(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    heads_ratio=heads_ratio,
                    dim=dim,
                    norm=(None, norm),
                    scales=scales,
                ),
                nn.Identity(),
            )
        else:
            raise ValueError(f"context_module {context_module} is not supported")
        if local_module == "GLUMBConv":
            self.local_module = ResidualBlock(
                GLUMBConv(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    expand_ratio=expand_ratio,
                    use_bias=(True, True, False),
                    norm=(None, None, norm),
                    act_func=(act_func, act_func, None),
                ),
                nn.Identity(),
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
        super(ResidualBlock, self).__init__()

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


class OpSequential(nn.Module):
    def __init__(self, op_list: list[Optional[nn.Module]]):
        super(OpSequential, self).__init__()
        valid_op_list = []
        for op in op_list:
            if op is not None:
                valid_op_list.append(op)
        self.op_list = nn.ModuleList(valid_op_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for op in self.op_list:
            x = op(x)
        return x



def build_block(
    block_type: str, in_channels: int, out_channels: int, norm: Optional[str], act: Optional[str]
) -> nn.Module:
    if block_type == "ResBlock":
        assert in_channels == out_channels
        main_block = ResBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            use_bias=(True, False),
            norm=(None, norm),
            act_func=(act, None),
        )
        block = ResidualBlock(main_block, nn.Identity())
    elif block_type == "EViT_GLU":
        assert in_channels == out_channels
        block = EfficientViTBlock(in_channels, norm=norm, act_func=act, local_module="GLUMBConv", scales=())
    elif block_type == "EViTS5_GLU":
        assert in_channels == out_channels
        block = EfficientViTBlock(in_channels, norm=norm, act_func=act, local_module="GLUMBConv", scales=(5,))
    else:
        raise ValueError(f"block_type {block_type} is not supported")
    return block


def build_stage_main(
    width: int, depth: int, block_type: str | list[str], norm: str, act: str, input_width: int
) -> list[nn.Module]:
    assert isinstance(block_type, str) or (isinstance(block_type, list) and depth == len(block_type))
    stage = []
    for d in range(depth):
        current_block_type = block_type[d] if isinstance(block_type, list) else block_type
        block = build_block(
            block_type=current_block_type,
            in_channels=width if d > 0 else input_width,
            out_channels=width,
            norm=norm,
            act=act,
        )
        stage.append(block)
    return stage


def build_downsample_block(block_type: str, in_channels: int, out_channels: int, shortcut: Optional[str]) -> nn.Module:
    if block_type == "Conv":
        block = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
        )
    elif block_type == "ConvPixelUnshuffle":
        block = ConvPixelUnshuffleDownsample2D(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, factor=2
        )
    else:
        raise ValueError(f"block_type {block_type} is not supported for downsampling")
    if shortcut is None:
        pass
    elif shortcut == "averaging":
        shortcut_block = PixelUnshuffleChannelAveragingDownsample2D(
            in_channels=in_channels, out_channels=out_channels, factor=2
        )
        block = ResidualBlock(block, shortcut_block)
    else:
        raise ValueError(f"shortcut {shortcut} is not supported for downsample")
    return block


def build_upsample_block(block_type: str, in_channels: int, out_channels: int, shortcut: Optional[str]) -> nn.Module:
    if block_type == "ConvPixelShuffle":
        block = ConvPixelShuffleUpsample2D(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, factor=2
        )
    elif block_type == "InterpolateConv":
        block = Upsample2D(channels=in_channels, use_conv=True, out_channels=out_channels)
    else:
        raise ValueError(f"block_type {block_type} is not supported for upsampling")
    if shortcut is None:
        pass
    elif shortcut == "duplicating":
        shortcut_block = ChannelDuplicatingPixelUnshuffleUpsample2D(
            in_channels=in_channels, out_channels=out_channels, factor=2
        )
        block = ResidualBlock(block, shortcut_block)
    else:
        raise ValueError(f"shortcut {shortcut} is not supported for upsample")
    return block


def build_encoder_project_in_block(in_channels: int, out_channels: int, factor: int, downsample_block_type: str):
    if factor == 1:
        block = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
        )
    elif factor == 2:
        block = build_downsample_block(
            block_type=downsample_block_type, in_channels=in_channels, out_channels=out_channels, shortcut=None
        )
    else:
        raise ValueError(f"downsample factor {factor} is not supported for encoder project in")
    return block


def build_encoder_project_out_block(
    in_channels: int, out_channels: int, norm: Optional[str], act: Optional[str], shortcut: Optional[str]
):
    block = OpSequential(
        [
            build_norm(norm),
            get_activation(act) if act is not None else None,
            ConvLayer(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                use_bias=True,
                norm=None,
                act_func=None,
            ),
        ]
    )
    if shortcut is None:
        pass
    elif shortcut == "averaging":
        shortcut_block = PixelUnshuffleChannelAveragingDownsample2D(
            in_channels=in_channels, out_channels=out_channels, factor=1
        )
        block = ResidualBlock(block, shortcut_block)
    else:
        raise ValueError(f"shortcut {shortcut} is not supported for encoder project out")
    return block


def build_decoder_project_in_block(in_channels: int, out_channels: int, shortcut: Optional[str]):
    block = ConvLayer(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=1,
        use_bias=True,
        norm=None,
        act_func=None,
    )
    if shortcut is None:
        pass
    elif shortcut == "duplicating":
        shortcut_block = ChannelDuplicatingPixelUnshuffleUpsample2D(
            in_channels=in_channels, out_channels=out_channels, factor=1
        )
        block = ResidualBlock(block, shortcut_block)
    else:
        raise ValueError(f"shortcut {shortcut} is not supported for decoder project in")
    return block


def build_decoder_project_out_block(
    in_channels: int, out_channels: int, factor: int, upsample_block_type: str, norm: Optional[str], act: Optional[str]
):
    layers: list[nn.Module] = [
        build_norm(norm, in_channels),
        get_activation(act) if act is not None else None,
    ]
    if factor == 1:
        layers.append(
            ConvLayer(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                use_bias=True,
                norm=None,
                act_func=None,
            )
        )
    elif factor == 2:
        layers.append(
            build_upsample_block(
                block_type=upsample_block_type, in_channels=in_channels, out_channels=out_channels, shortcut=None
            )
        )
    else:
        raise ValueError(f"upsample factor {factor} is not supported for decoder project out")
    return OpSequential(layers)


class Encoder(nn.Module):
    def __init__(
        self, 
        in_channels: int,
        latent_channels: int,
        width_list: list[int] = [128, 256, 512, 512, 1024, 1024],
        depth_list: list[int] = [2, 2, 2, 2, 2, 2],
        block_type: str | list[str] = "ResBlock",
        norm: str = "rms2d",
        act: str = "silu",
        downsample_block_type: str = "ConvPixelUnshuffle",
        downsample_match_channel: bool = True,
        downsample_shortcut: Optional[str] = "averaging",
        out_norm: Optional[str] = None,
        out_act: Optional[str] = None,
        out_shortcut: Optional[str] = "averaging",
        double_latent: bool = False,
    ):
        super().__init__()
        num_stages = len(width_list)
        self.num_stages = num_stages
        assert len(depth_list) == num_stages
        assert len(width_list) == num_stages
        assert isinstance(block_type, str) or (
            isinstance(block_type, list) and len(block_type) == num_stages
        )

        self.project_in = build_encoder_project_in_block(
            in_channels=in_channels,
            out_channels=width_list[0] if depth_list[0] > 0 else width_list[1],
            factor=1 if depth_list[0] > 0 else 2,
            downsample_block_type=downsample_block_type,
        )

        self.stages: list[OpSequential] = []
        for stage_id, (width, depth) in enumerate(zip(width_list, depth_list)):
            stage_block_type = block_type[stage_id] if isinstance(block_type, list) else block_type
            stage = build_stage_main(
                width=width, depth=depth, block_type=stage_block_type, norm=norm, act=act, input_width=width
            )
            if stage_id < num_stages - 1 and depth > 0:
                downsample_block = build_downsample_block(
                    block_type=downsample_block_type,
                    in_channels=width,
                    out_channels=width_list[stage_id + 1] if downsample_match_channel else width,
                    shortcut=downsample_shortcut,
                )
                stage.append(downsample_block)
            self.stages.append(OpSequential(stage))
        self.stages = nn.ModuleList(self.stages)

        self.project_out = build_encoder_project_out_block(
            in_channels=width_list[-1],
            out_channels=2 * latent_channels if double_latent else latent_channels,
            norm=out_norm,
            act=out_act,
            shortcut=out_shortcut,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.project_in(x)
        for stage in self.stages:
            if len(stage.op_list) == 0:
                continue
            x = stage(x)
        x = self.project_out(x)
        return x


class Decoder(nn.Module):
    def __init__(
        self, 
        in_channels: int,
        latent_channels: int,
        in_shortcut: Optional[str] = "duplicating",
        width_list: list[int] = [128, 256, 512, 512, 1024, 1024],
        depth_list: list[int] = [2, 2, 2, 2, 2, 2],
        block_type: str | list[str] = "ResBlock",
        norm: str | list[str] = "rms2d",
        act: str | list[str] = "silu",
        upsample_block_type: str = "ConvPixelShuffle",
        upsample_match_channel: bool = True,
        upsample_shortcut: str = "duplicating",
        out_norm: str = "rms2d",
        out_act: str = "relu",
    ):
        super().__init__()
        num_stages = len(width_list)
        self.num_stages = num_stages
        assert len(depth_list) == num_stages
        assert len(width_list) == num_stages
        assert isinstance(block_type, str) or (
            isinstance(block_type, list) and len(block_type) == num_stages
        )
        assert isinstance(norm, str) or (isinstance(norm, list) and len(norm) == num_stages)
        assert isinstance(act, str) or (isinstance(act, list) and len(act) == num_stages)

        self.project_in = build_decoder_project_in_block(
            in_channels=latent_channels,
            out_channels=width_list[-1],
            shortcut=in_shortcut,
        )

        self.stages: list[OpSequential] = []
        for stage_id, (width, depth) in reversed(list(enumerate(zip(width_list, depth_list)))):
            stage = []
            if stage_id < num_stages - 1 and depth > 0:
                upsample_block = build_upsample_block(
                    block_type=upsample_block_type,
                    in_channels=width_list[stage_id + 1],
                    out_channels=width if upsample_match_channel else width_list[stage_id + 1],
                    shortcut=upsample_shortcut,
                )
                stage.append(upsample_block)

            stage_block_type = block_type[stage_id] if isinstance(block_type, list) else block_type
            stage_norm = norm[stage_id] if isinstance(norm, list) else norm
            stage_act = act[stage_id] if isinstance(act, list) else act
            stage.extend(
                build_stage_main(
                    width=width,
                    depth=depth,
                    block_type=stage_block_type,
                    norm=stage_norm,
                    act=stage_act,
                    input_width=(
                        width if upsample_match_channel else width_list[min(stage_id + 1, num_stages - 1)]
                    ),
                )
            )
            self.stages.insert(0, OpSequential(stage))
        self.stages = nn.ModuleList(self.stages)

        self.project_out = build_decoder_project_out_block(
            in_channels=width_list[0] if depth_list[0] > 0 else width_list[1],
            out_channels=in_channels,
            factor=1 if depth_list[0] > 0 else 2,
            upsample_block_type=upsample_block_type,
            norm=out_norm,
            act=out_act,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.project_in(x)
        for stage in reversed(self.stages):
            if len(stage.op_list) == 0:
                continue
            x = stage(x)
        x = self.project_out(x)
        return x


class DCAE(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 32,
        encoder_block_type: str | list[str] = "ResBlock",
        encoder_width_list: list[int] = [128, 256, 512, 512, 1024, 1024],
        encoder_depth_list: list[int] = [2, 2, 2, 2, 2, 2],
        encoder_norm: str = "rms2d",
        encoder_act: str = "silu",
        downsample_block_type: str = "ConvPixelUnshuffle",
        decoder_block_type: str | list[str] = "ResBlock",
        decoder_width_list: list[int] = [128, 256, 512, 512, 1024, 1024],
        decoder_depth_list: list[int] = [2, 2, 2, 2, 2, 2],
        decoder_norm: str = "rms2d",
        decoder_act: str = "silu",
        upsample_block_type: str = "ConvPixelShuffle",
        scaling_factor: Optional[float] = None,
    ):
        super().__init__()
        self.encoder = Encoder(
            in_channels=in_channels,
            latent_channels=latent_channels,
            width_list=encoder_width_list,
            depth_list=encoder_depth_list,
            block_type=encoder_block_type,
            norm=encoder_norm,
            act=encoder_act,
            downsample_block_type=downsample_block_type,
        )
        self.decoder = Decoder(
            in_channels=in_channels,
            latent_channels=latent_channels,
            width_list=decoder_width_list,
            depth_list=decoder_depth_list,
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
            "encoder_width_list": [128, 256, 512, 512, 1024, 1024],
            "encoder_depth_list": [0, 4, 8, 2, 2, 2],
            "decoder_block_type": ["ResBlock", "ResBlock", "ResBlock", "EViT_GLU", "EViT_GLU", "EViT_GLU"],
            "decoder_width_list": [128, 256, 512, 512, 1024, 1024],
            "decoder_depth_list": [0, 5, 10, 2, 2, 2],
            "decoder_norm": ["bn2d", "bn2d", "bn2d", "rms2d", "rms2d", "rms2d"],
            "decoder_act": ["relu", "relu", "relu", "silu", "silu", "silu"],
        }
    elif name in ["dc-ae-f32c32-sana-1.0"]:
        cfg = {
            "latent_channels": 32,
            "encoder_block_type": ["ResBlock", "ResBlock", "ResBlock", "EViTS5_GLU", "EViTS5_GLU", "EViTS5_GLU"],
            "encoder_width_list": [128, 256, 512, 512, 1024, 1024],
            "encoder_depth_list": [2, 2, 2, 3, 3, 3],
            "downsample_block_type": "Conv",
            "decoder_block_type": ["ResBlock", "ResBlock", "ResBlock", "EViTS5_GLU", "EViTS5_GLU", "EViTS5_GLU"],
            "decoder_width_list": [128, 256, 512, 512, 1024, 1024],
            "decoder_depth_list": [3, 3, 3, 3, 3, 3],
            "upsample_block_type": "InterpolateConv",
            "scaling_factor": 0.41407,
        }
    else:
        raise NotImplementedError
    return cfg


def dc_ae_f64c128(name: str,) -> dict:
    if name in ["dc-ae-f64c128-in-1.0", "dc-ae-f64c128-mix-1.0"]:
        cfg = {
            "latent_channels": 128,
            "encoder_block_type": ["ResBlock", "ResBlock", "ResBlock", "EViT_GLU", "EViT_GLU", "EViT_GLU", "EViT_GLU"],
            "encoder_width_list": [128, 256, 512, 512, 1024, 1024, 2048],
            "encoder_depth_list": [0, 4, 8, 2, 2, 2, 2],
            "decoder_block_type": ["ResBlock", "ResBlock", "ResBlock", "EViT_GLU", "EViT_GLU", "EViT_GLU", "EViT_GLU"],
            "decoder_width_list": [128, 256, 512, 512, 1024, 1024, 2048],
            "decoder_depth_list": [0, 5, 10, 2, 2, 2, 2],
            "decoder_norm": ["bn2d", "bn2d", "bn2d", "rms2d", "rms2d", "rms2d", "rms2d"],
            "decoder_act": ["relu", "relu", "relu", "silu", "silu", "silu", "silu"],
        }
    else:
        raise NotImplementedError
    return cfg


def dc_ae_f128c512(name: str,) -> dict:
    if name in ["dc-ae-f128c512-in-1.0", "dc-ae-f128c512-mix-1.0"]:
        cfg = {
            "latent_channels": 512,
            "encoder_block_type": ["ResBlock", "ResBlock", "ResBlock", "EViT_GLU", "EViT_GLU", "EViT_GLU", "EViT_GLU", "EViT_GLU"],
            "encoder_width_list": [128, 256, 512, 512, 1024, 1024, 2048, 2048],
            "encoder_depth_list": [0, 4, 8, 2, 2, 2, 2, 2],
            "decoder_block_type": ["ResBlock", "ResBlock", "ResBlock", "EViT_GLU", "EViT_GLU", "EViT_GLU", "EViT_GLU", "EViT_GLU"],
            "decoder_width_list": [128, 256, 512, 512, 1024, 1024, 2048, 2048],
            "decoder_depth_list": [0, 5, 10, 2, 2, 2, 2, 2],
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


def create_dc_ae_model_cfg(name: str) -> dict:
    assert name in REGISTERED_DCAE_MODEL, f"{name} is not supported"
    dc_ae_cls = REGISTERED_DCAE_MODEL[name]
    model_cfg = dc_ae_cls(name)
    return model_cfg


class DCAE_HF(PyTorchModelHubMixin, DCAE):
    def __init__(self, model_name: str):
        cfg = create_dc_ae_model_cfg(model_name)
        DCAE.__init__(self, **cfg)


def main():
    from PIL import Image
    import torch
    import torchvision.transforms as transforms
    from torchvision.utils import save_image
    import ipdb

    torch.set_grad_enabled(False)
    device = torch.device("cuda")
    dtype = torch.float32

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    image = Image.open("/home/junyuc/workspace/code/efficientvit/assets/fig/girl.png")
    x = transform(image)[None].to(device=device, dtype=dtype)
    for model_name in REGISTERED_DCAE_MODEL:
        dc_ae = DCAE_HF.from_pretrained(f"mit-han-lab/{model_name}")
        dc_ae = dc_ae.to(device=device, dtype=dtype).eval()
        ipdb.set_trace()
        latent = dc_ae.encode(x)
        print(latent.shape)
        y = dc_ae.decode(latent)
        save_image(y * 0.5 + 0.5, f"demo_{model_name}.png")

if __name__ == "__main__":
    main()

"""
python -m src.diffusers.models.autoencoders.dc_ae
"""