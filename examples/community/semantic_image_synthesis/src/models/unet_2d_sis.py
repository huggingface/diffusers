# Copyright 2023 The HuggingFace Team. All rights reserved.
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
# Copied from here :
# https://github.com/WeilunWang/semantic-diffusion-model/blob/main/guided_diffusion/unet.py
# https://arxiv.org/abs/2207.00050

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.unet_2d import UNet2DOutput
from diffusers.utils import logging

from .unet_2d_sis_blocks import SISHeadAttnBlock, get_down_block, get_up_block


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# Resolutions 512,256,128,64,32,16,8


class UNet2DSISModel(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cls_count: int,
        img_size: Optional[Union[int, Tuple[int, int]]] = None,
        num_block_encoder: int = 2,
        num_block_decoder: int = 3,
        down_block_types: Tuple[str] = ("ConvBlock", "DownAttnBlock", "DownAttnBlock", "DownAttnBlock"),
        up_block_types: Tuple[str] = ("UpAttnBlock", "UpAttnBlock", "UpAttnBlock", "ConvBlock"),
        block_out_channels: Tuple[int] = (64, 128, 256, 512),
        attention_head_dim=64,
        freq_shift: int = 0,
        flip_sin_to_cos: bool = True,
        **kwargs,
    ):
        super().__init__()
        assert len(down_block_types) == len(
            up_block_types
        ), "Model should be symetric and have the same number of up and down blocks"
        assert len(block_out_channels) == len(
            down_block_types
        ), "Block channels and block names should have the same number of items..."
        self.img_size = img_size
        self.time_emb_dim = block_out_channels[0] * 4
        self.cls_count = cls_count
        # TimeEmbedding
        self.time_proj = Timesteps(block_out_channels[0], flip_sin_to_cos, freq_shift)
        self.time_embedding = TimestepEmbedding(block_out_channels[0], self.time_emb_dim)
        # Encoder
        self.encoder = nn.ModuleList()
        encoder_channels = block_out_channels
        blk_in_channels = in_channels
        for i, blk_out_channels in enumerate(encoder_channels):
            block = get_down_block(
                down_block_type=down_block_types[i],
                in_channels=blk_in_channels,
                out_channels=blk_out_channels,
                embedding_dim=self.time_emb_dim,
                attention_head_dim=attention_head_dim,
                num_res_blocks=num_block_encoder,
            )
            self.encoder.add_module(f"Enc_{i}", block)
            blk_in_channels = blk_out_channels
        self.head = SISHeadAttnBlock(
            blk_in_channels, blk_in_channels, cls_count, self.time_emb_dim, attention_head_dim=attention_head_dim
        )
        # Decoder
        self.decoder = nn.ModuleList()
        decoder_channels = tuple(block_out_channels[::-1][1:]) + (out_channels,)
        for i, blk_out_channels in enumerate(decoder_channels):
            block = get_up_block(
                up_block_type=up_block_types[i],
                in_channels=blk_in_channels,
                out_channels=blk_out_channels,
                label_channels=cls_count,
                embedding_dim=self.time_emb_dim,
                attention_head_dim=attention_head_dim,
                num_res_blocks=num_block_decoder,
            )
            self.decoder.add_module(f"Dec_{i}", block)
            blk_in_channels = blk_out_channels

    def forward(self, x, timesteps, cond=None) -> UNet2DOutput:
        b, _, h, w = x.shape

        # 1. Timesteps Management
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=x.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(x.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps * torch.ones(x.shape[0], dtype=timesteps.dtype, device=timesteps.device)
        temb = self.time_embedding(self.time_proj(timesteps))

        # 2. Condition Management...
        if cond is None:
            cond = torch.zeros([b, self.cls_count, h, w], device=self.device)
        else:
            if cond.shape == torch.Size([b, h, w]):
                # We need to one hot condition size
                cond = F.one_hot(cond.long(), num_classes=self.cls_count).to(self.device).float().permute(0, 3, 1, 2)
            elif cond.shape == torch.Size([h, w]):
                cond = (
                    F.one_hot(cond.unsqueeze(0).long(), num_classes=self.cls_count)
                    .to(self.device)
                    .float()
                    .permute(0, 3, 1, 2)
                )
            elif cond.shape == torch.Size([b, 1, h, w]):
                cond = (
                    F.one_hot(cond.squeeze(1).long(), num_classes=self.cls_count)
                    .to(self.device)
                    .float()
                    .permute(0, 3, 1, 2)
                )
            elif cond.shape == torch.Size([b, self.cls_count, h, w]):
                cond = cond.to(self.device).float()
        assert cond.shape == torch.Size(
            [b, self.cls_count, h, w]
        ), "Condition should be one hot encoding of shape [b,nClasses,h,w]"
        # Encoder...
        y = x
        y_encoder = []
        for layer in self.encoder:
            y = layer(y, temb, cond)
            y_encoder.append(y)
        # BottleNeck
        y = self.head(y, temb, cond)
        # Decoder
        for i, layer in enumerate(self.decoder):
            y = layer(y + y_encoder[::-1][i], temb, cond)
        return UNet2DOutput(y)

    def get_config(img_size: int = 512, in_channels: int = 3, out_channels: int = 3, cls_count: int = None):
        """Generate a configuration for SIS model from image size.

        https://github.com/WeilunWang/semantic-diffusion-model/blob/main/guided_diffusion/script_util.py#L156

        Args:
            img_size (int, optional): image size (should be a power of 2 from 64 to 512). Defaults to 512.
            in_channels (int, optional): number of input channels. Defaults to 3.
            out_channels (int, optional): number of output channels. Defaults to 3.
            cls_count (int): number of classes
        """
        assert cls_count is not None, "cls_count should be an integer"
        SIS_CONFIG_32 = {
            "in_channels": in_channels,
            "out_channels": out_channels,
            "cls_count": cls_count,
            "img_size": 64,
            "down_block_types": ("ConvBlock", "DownAttnBlock", "DownAttnBlock", "DownAttnBlock"),
            "up_block_types": ("UpAttnBlock", "UpAttnBlock", "UpAttnBlock", "ConvBlock"),
            "block_out_channels": (128, 256, 384, 512),
        }
        SIS_CONFIG_48 = {
            "in_channels": in_channels,
            "out_channels": out_channels,
            "cls_count": cls_count,
            "img_size": 48,
            "down_block_types": ("ConvBlock", "DownAttnBlock", "DownAttnBlock", "DownAttnBlock"),
            "up_block_types": ("UpAttnBlock", "UpAttnBlock", "UpAttnBlock", "ConvBlock"),
            "block_out_channels": (128, 256, 384, 512),
        }
        SIS_CONFIG_64 = {
            "in_channels": in_channels,
            "out_channels": out_channels,
            "cls_count": cls_count,
            "img_size": 64,
            "down_block_types": ("ConvBlock", "DownAttnBlock", "DownAttnBlock", "DownAttnBlock"),
            "up_block_types": ("UpAttnBlock", "UpAttnBlock", "UpAttnBlock", "ConvBlock"),
            "block_out_channels": (128, 256, 384, 512),
        }

        SIS_CONFIG_128 = {
            "in_channels": in_channels,
            "out_channels": out_channels,
            "cls_count": cls_count,
            "img_size": 128,
            "down_block_types": ("ConvBlock", "DownBlock", "DownAttnBlock", "DownAttnBlock", "DownAttnBlock"),
            "up_block_types": ("UpAttnBlock", "UpAttnBlock", "UpAttnBlock", "UpBlock", "ConvBlock"),
            "block_out_channels": (128, 128, 256, 384, 512),
        }

        SIS_CONFIG_256 = {
            "in_channels": in_channels,
            "out_channels": out_channels,
            "cls_count": cls_count,
            "img_size": 256,
            "down_block_types": (
                "ConvBlock",
                "DownBlock",
                "DownBlock",
                "DownAttnBlock",
                "DownAttnBlock",
                "DownAttnBlock",
            ),
            "up_block_types": ("UpAttnBlock", "UpAttnBlock", "UpAttnBlock", "UpBlock", "UpBlock", "ConvBlock"),
            "block_out_channels": (128, 128, 256, 256, 512, 512),
        }

        SIS_CONFIG_512 = {
            "in_channels": in_channels,
            "out_channels": out_channels,
            "cls_count": cls_count,
            "img_size": 512,
            "down_block_types": (
                "ConvBlock",
                "DownBlock",
                "DownBlock",
                "DownBlock",
                "DownAttnBlock",
                "DownAttnBlock",
                "DownAttnBlock",
            ),
            "up_block_types": (
                "UpAttnBlock",
                "UpAttnBlock",
                "UpAttnBlock",
                "UpBlock",
                "UpBlock",
                "UpBlock",
                "ConvBlock",
            ),
            "block_out_channels": (64, 128, 128, 256, 256, 512, 512),
        }

        configuration = {"32":SIS_CONFIG_32,"48":SIS_CONFIG_48,"64": SIS_CONFIG_64, "128": SIS_CONFIG_128, "256": SIS_CONFIG_256, "512": SIS_CONFIG_512}
        assert (
            str(img_size) in configuration.keys()
        ), "img_size should be in existing configurations {configuration.keys()}"
        return configuration[str(img_size)]
