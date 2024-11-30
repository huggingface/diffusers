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

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...configuration_utils import ConfigMixin, register_to_config
from ..activations import get_activation
from ..modeling_utils import ModelMixin
from ..normalization import RMSNormNd


def get_norm_layer(name: Optional[str] = "bn2d", num_features: Optional[int] = None) -> Optional[nn.Module]:
    if name is None:
        norm = None
    elif name == "rms2d":
        norm = RMSNormNd(num_features, eps=1e-5, elementwise_affine=True, bias=True, channel_dim=1)
    elif name == "bn2d":
        norm = nn.BatchNorm2d(num_features=num_features)
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

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states

        hidden_states = self.conv_inverted(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.conv_depth(hidden_states)
        hidden_states, gate = torch.chunk(hidden_states, 2, dim=1)
        hidden_states = hidden_states * self.nonlinearity(gate)

        hidden_states = self.conv_point(hidden_states)
        hidden_states = self.norm(hidden_states)

        return hidden_states + residual


class ResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_type: str = "bn2d",
        act_fn: str = "relu6",
    ) -> None:
        super().__init__()

        self.nonlinearity = get_activation(act_fn) if act_fn is not None else nn.Identity()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False)
        self.norm = get_norm_layer(norm_type, out_channels)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.conv1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv2(hidden_states)
        hidden_states = self.norm(hidden_states)
        return hidden_states + residual


class MLAProjection(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_attention_heads: int,
        kernel_size: int,
    ) -> None:
        super().__init__()

        self.proj_in = nn.Conv2d(
            3 * in_channels,
            3 * in_channels,
            kernel_size,
            padding=kernel_size // 2,
            groups=3 * in_channels,
            bias=False,
        )
        self.proj_out = nn.Conv2d(3 * in_channels, 3 * in_channels, 1, 1, 0, groups=3 * num_attention_heads, bias=False)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.proj_in(hidden_states)
        hidden_states = self.proj_out(hidden_states)
        return hidden_states


class LiteMLA(nn.Module):
    r"""Lightweight multi-scale linear attention"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_attention_heads: Optional[int] = None,
        heads_ratio: float = 1.0,
        attention_head_dim: int = 8,
        norm_type: str = "bn2d",
        kernel_sizes: Tuple[int, ...] = (5,),
        eps: float = 1e-15,
    ):
        super().__init__()

        self.eps = eps
        self.attention_head_dim = attention_head_dim

        num_attention_heads = (
            int(in_channels // attention_head_dim * heads_ratio) if num_attention_heads is None else num_attention_heads
        )
        inner_dim = num_attention_heads * attention_head_dim

        # TODO(aryan): Convert to nn.linear
        # self.qkv = nn.Conv2d(in_channels, 3 * inner_dim, 1, 1, 0, bias=False)
        self.to_q = nn.Linear(in_channels, inner_dim, bias=False)
        self.to_k = nn.Linear(in_channels, inner_dim, bias=False)
        self.to_v = nn.Linear(in_channels, inner_dim, bias=False)

        self.to_qkv_multiscale = nn.ModuleList()
        for kernel_size in kernel_sizes:
            self.to_qkv_multiscale.append(MLAProjection(inner_dim, num_attention_heads, kernel_size))

        self.kernel_nonlinearity = nn.ReLU()

        self.proj_out = nn.Conv2d(inner_dim * (1 + len(kernel_sizes)), out_channels, 1, 1, 0, bias=False)
        self.norm_out = get_norm_layer(norm_type, num_features=out_channels)

    def relu_linear_att(self, qkv: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = qkv.shape

        qkv = qkv.float()
        qkv = torch.reshape(qkv, (batch_size, -1, 3 * self.attention_head_dim, height * width))

        query, key, value = (qkv[:, :, 0 : self.attention_head_dim], qkv[:, :, self.attention_head_dim : 2 * self.attention_head_dim], qkv[:, :, 2 * self.attention_head_dim :])

        # lightweight linear attention
        query = self.kernel_nonlinearity(query)
        key = self.kernel_nonlinearity(key)
        value = F.pad(value, (0, 0, 0, 1), mode="constant", value=1)

        key_T = key.transpose(-1, -2)
        scores = torch.matmul(value, key_T)
        output = torch.matmul(scores, query)
        
        output = output.float()
        output = output[:, :, :-1] / (output[:, :, -1:] + self.eps)
        output = torch.reshape(output, (batch_size, -1, height, width))

        return output

    def relu_quadratic_att(self, qkv: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = list(qkv.size())

        qkv = torch.reshape(qkv, (batch_size, -1, 3 * self.attention_head_dim, height * width))
        query, key, value = (
            qkv[:, :, 0 : self.attention_head_dim],
            qkv[:, :, self.attention_head_dim : 2 * self.attention_head_dim],
            qkv[:, :, 2 * self.attention_head_dim :],
        )

        query = self.kernel_nonlinearity(query)
        key = self.kernel_nonlinearity(key)

        scores = torch.matmul(key.transpose(-1, -2), query)

        original_dtype = scores.dtype
        scores = scores.float()
        scores = scores / (torch.sum(scores, dim=2, keepdim=True) + self.eps)
        scores = scores.to(original_dtype)

        output = torch.matmul(value, scores)
        output = torch.reshape(output, (batch_size, -1, height, width))
        
        return output

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        
        hidden_states = hidden_states.movedim(1, -1)
        query = self.to_q(hidden_states)
        key = self.to_k(hidden_states)
        value = self.to_v(hidden_states)
        qkv = torch.cat([query, key, value], dim=-1)
        qkv = qkv.movedim(-1, 1)

        multi_scale_qkv = [qkv]
        
        for block in self.to_qkv_multiscale:
            multi_scale_qkv.append(block(qkv))

        qkv = torch.cat(multi_scale_qkv, dim=1)

        height, width = qkv.shape[-2:]
        if height * width > self.attention_head_dim:
            hidden_states = self.relu_linear_att(qkv).to(qkv.dtype)
        else:
            hidden_states = self.relu_quadratic_att(qkv)

        hidden_states = self.proj_out(hidden_states)
        hidden_states = self.norm_out(hidden_states)

        return hidden_states + residual


class EfficientViTBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        heads_ratio: float = 1.0,
        dim: int = 32,
        qkv_multiscales: Tuple[int, ...] = (5,),
        norm_type: str = "bn2d",
    ):
        super().__init__()

        self.attn = LiteMLA(
            in_channels=in_channels,
            out_channels=in_channels,
            heads_ratio=heads_ratio,
            attention_head_dim=dim,
            norm_type=norm_type,
            kernel_sizes=qkv_multiscales,
        )

        self.conv_out = GLUMBConv(
            in_channels=in_channels,
            out_channels=in_channels,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attn(x)
        x = self.conv_out(x)
        return x


def get_block_from_block_type(
    block_type: str,
    in_channels: int,
    out_channels: int,
    norm_type: str,
    act_fn: str,
    qkv_mutliscales: Tuple[int] = (),
):
    if block_type == "ResBlock":
        block = ResBlock(in_channels, out_channels, norm_type, act_fn)
    
    elif block_type == "EfficientViTBlock":
        block = EfficientViTBlock(in_channels, norm_type=norm_type, qkv_multiscales=qkv_mutliscales)

    else:
        raise ValueError(f"Block with {block_type=} is not supported.")
    
    return block


def build_stage_main(
    width: int, depth: int, block_type: str | List[str], norm: str, act: str, input_width: int, qkv_multiscales=()
) -> list[nn.Module]:
    stage = []
    for d in range(depth):
        current_block_type = block_type[d] if isinstance(block_type, list) else block_type

        in_channels = width if d > 0 else input_width
        out_channels = width

        block = get_block_from_block_type(current_block_type, in_channels, out_channels, norm_type=norm, act_fn=act, qkv_mutliscales=qkv_multiscales)
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
        self.shortcut = shortcut
        self.factor = 2
        self.repeats = out_channels * self.factor**2 // in_channels

        out_ratio = self.factor**2

        if not interpolate:
            out_channels = out_channels * out_ratio

        self.conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1)

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
        block_type: Union[str, Tuple[str]] = "ResBlock",
        block_out_channels: Tuple[int] = (128, 256, 512, 512, 1024, 1024),
        layers_per_block: Tuple[int] = (2, 2, 2, 2, 2, 2),
        qkv_multiscales: Tuple[Tuple[int, ...], ...] = ((), (), (), (5,), (5,), (5,)),
        downsample_block_type: str = "ConvPixelUnshuffle",
    ):
        super().__init__()
        
        num_stages = len(block_out_channels)

        if isinstance(block_type, str):
            block_type = (block_type,) * num_stages

        if layers_per_block[0] > 0:
            self.conv_in = nn.Conv2d(
                in_channels,
                block_out_channels[0] if layers_per_block[0] > 0 else block_out_channels[1],
                kernel_size=3,
                stride=1,
                padding=1,
            )
        else:
            self.conv_in = DCDownBlock2d(
                in_channels=in_channels,
                out_channels=block_out_channels[0] if layers_per_block[0] > 0 else block_out_channels[1],
                downsample=downsample_block_type == "ConvPixelUnshuffle",
                shortcut=False,
            )

        stages = []
        for stage_id, (width, depth) in enumerate(zip(block_out_channels, layers_per_block)):
            stage_block_type = block_type[stage_id]
            current_stage = build_stage_main(
                width=width, depth=depth, block_type=stage_block_type, norm="rms2d", act="silu", input_width=width, qkv_multiscales=qkv_multiscales[stage_id]
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

        self.conv_out = nn.Conv2d(block_out_channels[-1], latent_channels, 3, 1, 1)
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
        block_type: Union[str, Tuple[str]] = "ResBlock",
        block_out_channels: Tuple[int] = (128, 256, 512, 512, 1024, 1024),
        layers_per_block: Tuple[int] = (2, 2, 2, 2, 2, 2),
        qkv_multiscales: Tuple[Tuple[int, ...], ...] = ((), (), (), (5,), (5,), (5,)),
        norm_type: Union[str, Tuple[str]] = "rms2d",
        act_fn: Union[str, Tuple[str]] = "silu",
        upsample_block_type: str = "ConvPixelShuffle",
        upsample_shortcut: str = "duplicating",
    ):
        super().__init__()
        
        num_stages = len(block_out_channels)

        if isinstance(block_type, str):
            block_type = (block_type,) * num_stages
        if isinstance(norm_type, str):
            norm_type = (norm_type,) * num_stages
        if isinstance(act_fn, str):
            act_fn = (act_fn,) * num_stages

        self.conv_in = nn.Conv2d(latent_channels, block_out_channels[-1], 3, 1, 1)

        self.norm_factor = 1
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

            stage_block_type = block_type[stage_id]
            stage_norm = norm_type[stage_id]
            stage_act = act_fn[stage_id]
            current_stage.extend(
                build_stage_main(
                    width=width,
                    depth=depth,
                    block_type=stage_block_type,
                    norm=stage_norm,
                    act=stage_act,
                    input_width=width,
                    qkv_multiscales=qkv_multiscales[stage_id],
                )
            )
            stages.insert(0, nn.Sequential(*current_stage))
        self.stages = nn.ModuleList(stages)

        channels = block_out_channels[0] if layers_per_block[0] > 0 else block_out_channels[1]

        self.norm_out = RMSNormNd(channels, eps=1e-5, elementwise_affine=True, bias=True, channel_dim=1)
        self.conv_act = nn.ReLU()
        self.conv_out = None

        if layers_per_block[0] > 0:
            self.conv_out = nn.Conv2d(channels, in_channels, 3, 1, 1)
        else:
            self.conv_out = DCUpBlock2d(
                channels, in_channels, interpolate=upsample_block_type == "InterpolateConv", shortcut=False
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
        encoder_block_types: Union[str, Tuple[str]] = "ResBlock",
        decoder_block_types: Union[str, Tuple[str]] = "ResBlock",
        block_out_channels: Tuple[int, ...] = (128, 256, 512, 512, 1024, 1024),
        encoder_layers_per_block: Tuple[int] = (2, 2, 2, 3, 3, 3),
        decoder_layers_per_block: Tuple[int] = (3, 3, 3, 3, 3, 3),
        encoder_qkv_multiscales: Tuple[Tuple[int, ...], ...] = ((), (), (), (5,), (5,), (5,)),
        decoder_qkv_multiscales: Tuple[Tuple[int, ...], ...] = ((), (), (), (5,), (5,), (5,)),
        upsample_block_type: str = "ConvPixelShuffle",
        downsample_block_type: str = "ConvPixelUnshuffle",
        decoder_norm_types: Union[str, Tuple[str]] = "rms2d",
        decoder_act_fns: Union[str, Tuple[str]] = "silu",
        scaling_factor: float = 1.0,
    ) -> None:
        super().__init__()
        
        self.encoder = Encoder(
            in_channels=in_channels,
            latent_channels=latent_channels,
            block_type=encoder_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=encoder_layers_per_block,
            qkv_multiscales=encoder_qkv_multiscales,
            downsample_block_type=downsample_block_type,
        )
        self.decoder = Decoder(
            in_channels=in_channels,
            latent_channels=latent_channels,
            block_type=decoder_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=decoder_layers_per_block,
            qkv_multiscales=decoder_qkv_multiscales,
            norm_type=decoder_norm_types,
            act_fn=decoder_act_fns,
            upsample_block_type=upsample_block_type,
        )

        self.spatial_compression_ratio = 2 ** (len(block_out_channels) - 1)
        self.temporal_compression_ratio = 1

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        return x

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.decoder(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.decoder(x)
        return x
