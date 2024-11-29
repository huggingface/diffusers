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

from typing import Any, List, Optional, Tuple

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


class GLUMBConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        hidden_channels = 4 * in_channels

        self.nonlinearity = nn.SiLU()

        self.conv_inverted = nn.Conv2d(in_channels, hidden_channels * 2, 1, 1, 0)
        self.conv_depth = nn.Conv2d(hidden_channels * 2, hidden_channels * 2, 3, 1, 1, groups=hidden_channels * 2)
        self.conv_point = nn.Conv2d(hidden_channels, out_channels, 1, 1, 0, bias=False)
        self.norm = RMSNormNd(out_channels, eps=1e-5, elementwise_affine=True, bias=True, channel_dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        x = self.conv_inverted(x)
        x = self.nonlinearity(x)

        x = self.conv_depth(x)
        x, gate = torch.chunk(x, 2, dim=1)
        x = x * self.nonlinearity(gate)

        x = self.conv_point(x)
        x = self.norm(x)

        return x + residual


class ResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm=("bn2d", "bn2d"),
        act_func=("relu6", None),
    ) -> None:
        super().__init__()

        norm = val2tuple(norm, 2)
        act_func = val2tuple(act_func, 2)

        self.nonlinearity = get_activation(act_func[0]) if act_func[0] is not None else nn.Identity()

        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        self.norm1 = build_norm(norm[0], num_features=in_channels) if norm[0] is not None else nn.Identity()

        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False)
        self.norm2 = build_norm(norm[1], num_features=out_channels) if norm[1] is not None else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.nonlinearity(x)
        x = self.conv2(x)
        x = self.norm2(x)
        return x + residual


class LiteMLA(nn.Module):
    r"""Lightweight multi-scale linear attention"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_attention_heads: Optional[int] = None,
        heads_ratio: float = 1.0,
        dim=8,
        norm=(None, "bn2d"),
        act_func=(None, None),
        scales: tuple[int, ...] = (5,),
        eps: float = 1e-15,
    ):
        super().__init__()

        norm = val2tuple(norm, 2)
        act_func = val2tuple(act_func, 2)

        self.eps = eps
        self.dim = dim

        num_attention_heads = (
            int(in_channels // dim * heads_ratio) if num_attention_heads is None else num_attention_heads
        )
        inner_dim = num_attention_heads * dim

        # TODO(aryan): Convert to nn.linear
        self.qkv = nn.Conv2d(in_channels, 3 * inner_dim, 1, 1, 0, bias=False)

        self.aggreg = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        3 * inner_dim,
                        3 * inner_dim,
                        scale,
                        padding=scale // 2,
                        groups=3 * inner_dim,
                        bias=False,
                    ),
                    nn.Conv2d(3 * inner_dim, 3 * inner_dim, 1, 1, 0, groups=3 * num_attention_heads, bias=False),
                )
                for scale in scales
            ]
        )
        self.kernel_nonlinearity = nn.ReLU()

        self.proj_out = nn.Conv2d(inner_dim * (1 + len(scales)), out_channels, 1, 1, 0, bias=False)
        self.norm_out = build_norm(norm[1], num_features=out_channels) or nn.Identity()
        self.act_out = get_activation(act_func[1]) if act_func[1] is not None else nn.Identity()

    def relu_linear_att(self, qkv: torch.Tensor) -> torch.Tensor:
        B, _, H, W = list(qkv.size())

        qkv = qkv.float()
        qkv = torch.reshape(qkv, (B, -1, 3 * self.dim, H * W))

        query, key, value = (qkv[:, :, 0 : self.dim], qkv[:, :, self.dim : 2 * self.dim], qkv[:, :, 2 * self.dim :])

        # lightweight linear attention
        query = self.kernel_nonlinearity(query)
        key = self.kernel_nonlinearity(key)

        # linear matmul
        k_T = key.transpose(-1, -2)

        value = F.pad(value, (0, 0, 0, 1), mode="constant", value=1)
        vk = torch.matmul(value, k_T)
        out = torch.matmul(vk, query)
        out = out.float()

        out = out[:, :, :-1] / (out[:, :, -1:] + self.eps)
        out = torch.reshape(out, (B, -1, H, W))

        return out

    def relu_quadratic_att(self, qkv: torch.Tensor) -> torch.Tensor:
        B, _, H, W = list(qkv.size())

        qkv = torch.reshape(qkv, (B, -1, 3 * self.dim, H * W))
        q, k, v = (
            qkv[:, :, 0 : self.dim],
            qkv[:, :, self.dim : 2 * self.dim],
            qkv[:, :, 2 * self.dim :],
        )

        q = self.kernel_nonlinearity(q)
        k = self.kernel_nonlinearity(k)

        att_map = torch.matmul(k.transpose(-1, -2), q)  # b h n n

        original_dtype = att_map.dtype
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

        out = self.proj_out(out)
        out = self.norm_out(out)
        out = self.act_out(out)

        return out + residual


class EfficientViTBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        heads_ratio: float = 1.0,
        dim=32,
        scales: tuple[int, ...] = (5,),
        norm: str = "bn2d",
    ):
        super().__init__()

        self.context_module = LiteMLA(
            in_channels=in_channels,
            out_channels=in_channels,
            heads_ratio=heads_ratio,
            dim=dim,
            norm=(None, norm),
            scales=scales,
        )

        self.local_module = GLUMBConv(
            in_channels=in_channels,
            out_channels=in_channels,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.context_module(x)
        x = self.local_module(x)
        return x


def build_stage_main(
    width: int, depth: int, block_type: str | List[str], norm: str, act: str, input_width: int
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
                norm=(None, norm),
                act_func=(act, None),
            )
        elif current_block_type == "EViT_GLU":
            assert in_channels == out_channels
            block = EfficientViTBlock(in_channels, norm=norm, scales=())
        elif current_block_type == "EViTS5_GLU":
            assert in_channels == out_channels
            block = EfficientViTBlock(in_channels, norm=norm, scales=(5,))
        else:
            raise ValueError(f"block_type {current_block_type} is not supported")

        stage.append(block)
    return stage


class DCDownBlock2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, downsample: bool = False, shortcut: bool = True) -> None:
        super().__init__()

        self.downsample = downsample
        self.factor = 2
        self.stride = 1 if downsample else 2
        self.group_size = in_channels * self.factor**2 // out_channels
        self.shortcut = shortcut

        out_ratio = self.factor**2
        if downsample:
            assert out_channels % out_ratio == 0
            out_channels = out_channels // out_ratio

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=self.stride,
            padding=1,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = self.conv(hidden_states)
        if self.downsample:
            x = F.pixel_unshuffle(x, self.factor)

        if self.shortcut:
            y = F.pixel_unshuffle(hidden_states, self.factor)
            y = y.unflatten(1, (-1, self.group_size))
            y = y.mean(dim=2)
            hidden_states = x + y
        else:
            hidden_states = x

        return hidden_states


class DCUpBlock2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        interpolate: bool = False,
        shortcut: bool = True,
        interpolation_mode: str = "nearest",
    ) -> None:
        super().__init__()

        self.interpolate = interpolate
        self.interpolation_mode = interpolation_mode
        self.factor = 2
        out_ratio = self.factor**2

        if not interpolate:
            out_channels = out_channels * out_ratio

        self.repeats = out_channels * self.factor**2 // in_channels

        self.conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1)

        self.shortcut = shortcut

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.interpolate:
            x = F.interpolate(hidden_states, scale_factor=self.factor, mode=self.interpolation_mode)
            x = self.conv(x)
        else:
            x = self.conv(hidden_states)
            x = F.pixel_shuffle(x, self.factor)

        if self.shortcut:
            y = hidden_states.repeat_interleave(self.repeats, dim=1)
            y = F.pixel_shuffle(y, self.factor)
            hidden_states = x + y
        else:
            hidden_states = x

        return hidden_states


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        latent_channels: int,
        block_out_channels: List[int] = [128, 256, 512, 512, 1024, 1024],
        layers_per_block: List[int] = [2, 2, 2, 2, 2, 2],
        block_type: Union[str, List[str]] = "ResBlock",
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
        self.norm_factor = 1
        norm_in_channels = block_out_channels[-1]
        norm_out_channels = latent_channels
        self.norm_group_size = norm_in_channels * self.norm_factor**2 // norm_out_channels

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.conv_in(hidden_states)
        for stage in self.stages:
            hidden_states = stage(hidden_states)

        x = F.pixel_unshuffle(hidden_states, self.norm_factor)
        x = x.unflatten(1, (-1, self.norm_group_size))
        x = x.mean(dim=2)

        hidden_states = self.conv_out(hidden_states) + x
        return hidden_states


class Decoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        latent_channels: int,
        block_out_channels: List[int] = [128, 256, 512, 512, 1024, 1024],
        layers_per_block: List[int] = [2, 2, 2, 2, 2, 2],
        block_type: str | List[str] = "ResBlock",
        norm: str | List[str] = "rms2d",
        act: str | List[str] = "silu",
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

        self.norm_factor = 1
        # TODO(aryan): Make sure this is divisible
        self.norm_repeats = block_out_channels[-1] * self.norm_factor**2 // latent_channels

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

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = hidden_states.repeat_interleave(self.norm_repeats, dim=1)
        x = F.pixel_shuffle(x, self.norm_factor)

        hidden_states = self.conv_in(hidden_states) + x

        for stage in reversed(self.stages):
            hidden_states = stage(hidden_states)

        hidden_states = self.norm_out(hidden_states)
        hidden_states = self.conv_act(hidden_states)
        hidden_states = self.conv_out(hidden_states)
        return hidden_states


class AutoencoderDC(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 32,
        block_out_channels: Tuple[int, ...] = (128, 256, 512, 512, 1024, 1024),
        encoder_layers_per_block: Tuple[int] = (2, 2, 2, 3, 3, 3),
        decoder_layers_per_block: Tuple[int] = (3, 3, 3, 3, 3, 3),
        encoder_block_type: str | List[str] = "ResBlock",
        downsample_block_type: str = "ConvPixelUnshuffle",
        decoder_block_type: str | List[str] = "ResBlock",
        decoder_norm: str = "rms2d",
        decoder_act: str = "silu",
        upsample_block_type: str = "ConvPixelShuffle",
        scaling_factor: Optional[float] = None,
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
