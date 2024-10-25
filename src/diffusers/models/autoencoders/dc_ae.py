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
from huggingface_hub import PyTorchModelHubMixin

from ...configuration_utils import ConfigMixin, register_to_config
from ...loaders.single_file_model import FromOriginalModelMixin
from ..modeling_utils import ModelMixin

from .dc_ae_blocks.act import build_act
from .dc_ae_blocks.norm import build_norm
from .dc_ae_blocks.ops import (
    ChannelDuplicatingPixelUnshuffleUpSampleLayer,
    ConvLayer,
    ConvPixelShuffleUpSampleLayer,
    ConvPixelUnshuffleDownSampleLayer,
    EfficientViTBlock,
    IdentityLayer,
    InterpolateConvUpSampleLayer,
    OpSequential,
    PixelUnshuffleChannelAveragingDownSampleLayer,
    ResBlock,
    ResidualBlock,
)

__all__ = ["DCAE", "dc_ae_f32c32", "dc_ae_f64c128", "dc_ae_f128c512"]


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
        block = ResidualBlock(main_block, IdentityLayer())
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
        block = ConvLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            use_bias=True,
            norm=None,
            act_func=None,
        )
    elif block_type == "ConvPixelUnshuffle":
        block = ConvPixelUnshuffleDownSampleLayer(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, factor=2
        )
    else:
        raise ValueError(f"block_type {block_type} is not supported for downsampling")
    if shortcut is None:
        pass
    elif shortcut == "averaging":
        shortcut_block = PixelUnshuffleChannelAveragingDownSampleLayer(
            in_channels=in_channels, out_channels=out_channels, factor=2
        )
        block = ResidualBlock(block, shortcut_block)
    else:
        raise ValueError(f"shortcut {shortcut} is not supported for downsample")
    return block


def build_upsample_block(block_type: str, in_channels: int, out_channels: int, shortcut: Optional[str]) -> nn.Module:
    if block_type == "ConvPixelShuffle":
        block = ConvPixelShuffleUpSampleLayer(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, factor=2
        )
    elif block_type == "InterpolateConv":
        block = InterpolateConvUpSampleLayer(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, factor=2
        )
    else:
        raise ValueError(f"block_type {block_type} is not supported for upsampling")
    if shortcut is None:
        pass
    elif shortcut == "duplicating":
        shortcut_block = ChannelDuplicatingPixelUnshuffleUpSampleLayer(
            in_channels=in_channels, out_channels=out_channels, factor=2
        )
        block = ResidualBlock(block, shortcut_block)
    else:
        raise ValueError(f"shortcut {shortcut} is not supported for upsample")
    return block


def build_encoder_project_in_block(in_channels: int, out_channels: int, factor: int, downsample_block_type: str):
    if factor == 1:
        block = ConvLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            use_bias=True,
            norm=None,
            act_func=None,
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
            build_act(act),
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
        shortcut_block = PixelUnshuffleChannelAveragingDownSampleLayer(
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
        shortcut_block = ChannelDuplicatingPixelUnshuffleUpSampleLayer(
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
        build_act(act),
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


class DCAE(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 32,
        encoder_width_list: list[int] = [128, 256, 512, 512, 1024, 1024],
        encoder_depth_list: list[int] = [2, 2, 2, 2, 2, 2],
        encoder_block_type: str | list[str] = "ResBlock",
        encoder_norm: str = "rms2d",
        encoder_act: str = "silu",
        downsample_block_type: str = "ConvPixelUnshuffle",
        decoder_width_list: list[int] = [128, 256, 512, 512, 1024, 1024],
        decoder_depth_list: list[int] = [2, 2, 2, 2, 2, 2],
        decoder_block_type: str | list[str] = "ResBlock",
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
