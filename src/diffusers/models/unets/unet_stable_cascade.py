# Copyright 2024 The HuggingFace Team. All rights reserved.
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
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn

from ...configuration_utils import ConfigMixin, register_to_config
from ...pipelines.wuerstchen.modeling_wuerstchen_common import AttnBlock, TimestepBlock, WuerstchenLayerNorm
from ...pipelines.wuerstchen.modeling_wuerstchen_diffnext import ResBlockStageB
from ...utils import BaseOutput
from ..modeling_utils import ModelMixin


class UpDownBlock2d(nn.Module):
    def __init__(self, in_channels, out_channels, mode, enabled=True):
        super().__init__()
        if mode not in ["up", "down"]:
            raise ValueError(f"{mode} not supported")
        interpolation = (
            nn.Upsample(scale_factor=2 if mode == "up" else 0.5, mode="bilinear", align_corners=True)
            if enabled
            else nn.Identity()
        )
        mapping = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.blocks = nn.ModuleList([interpolation, mapping] if mode == "up" else [mapping, interpolation])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


@dataclass
class StableCascadeUNetOutput(BaseOutput):
    sample: torch.FloatTensor = None


class StableCascadeUNet(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        in_channels: int = 16,
        out_channels: int = 16,
        timestep_ratio_embedding_dim: int = 64,
        patch_size: int = 1,
        conditioning_dim: int = 2048,
        block_out_channels: List[int] = [2048, 2048],
        num_attention_heads: List[int] = [32, 32],
        down_num_layers_per_block: List[int] = [8, 24],
        up_num_layers_per_block: List[int] = [24, 8],
        block_repeat: List[List[int]] = [[1, 1], [1, 1]],
        down_blocks_repeat_mappers: List[int] = [
            1,
            1,
        ],
        up_blocks_repeat_mappers: List[int] = [1, 1],
        block_types_per_layer: List[List[str]] = [
            ["ResBlockStageB", "TimestepBlock", "AttnBlock"],
            ["ResBlockStageB", "TimestepBlock", "AttnBlock"],
        ],
        clip_text_in_channels: Optional[int] = None,
        clip_text_pooled_in_channels=1280,
        clip_image_in_channels: Optional[int] = None,
        clip_seq=4,
        effnet_in_channels: Optional[int] = None,
        pixel_mapper_in_channels: Optional[int] = None,
        kernel_size=3,
        dropout: List[float] = [0.1, 0.1],
        self_attn: Union[bool, List[bool]] = True,
        timestep_conditioning_type: List[str] = ["sca", "crp"],
        switch_level: Optional[List[bool]] = None,
    ):
        super().__init__()

        if len(block_out_channels) != len(down_num_layers_per_block):
            raise ValueError(
                f"Number of elements in `down_num_layers_per_block` must match the length of `block_out_channels`: {len(block_out_channels)}"
            )

        elif len(block_out_channels) != len(up_num_layers_per_block):
            raise ValueError(
                f"Number of elements in `up_num_layers_per_block` must match the length of `block_out_channels`: {len(block_out_channels)}"
            )

        elif len(block_out_channels) != len(down_blocks_repeat_mappers):
            raise ValueError(
                f"Number of elements in `down_blocks_repeat_mappers` must match the length of `block_out_channels`: {len(block_out_channels)}"
            )

        elif len(block_out_channels) != len(up_blocks_repeat_mappers):
            raise ValueError(
                f"Number of elements in `up_blocks_repeat_mappers` must match the length of `block_out_channels`: {len(block_out_channels)}"
            )

        elif len(block_out_channels) != len(block_types_per_layer):
            raise ValueError(
                f"Number of elements in `block_types_per_layer` must match the length of `block_out_channels`: {len(block_out_channels)}"
            )

        if not isinstance(dropout, list):
            dropout = [dropout] * len(block_out_channels)
        if not isinstance(self_attn, list):
            self_attn = [self_attn] * len(block_out_channels)

        # CONDITIONING
        if effnet_in_channels is not None:
            self.effnet_mapper = nn.Sequential(
                nn.Conv2d(effnet_in_channels, block_out_channels[0] * 4, kernel_size=1),
                nn.GELU(),
                nn.Conv2d(block_out_channels[0] * 4, block_out_channels[0], kernel_size=1),
                WuerstchenLayerNorm(block_out_channels[0], elementwise_affine=False, eps=1e-6),
            )
        if pixel_mapper_in_channels is not None:
            self.pixels_mapper = nn.Sequential(
                nn.Conv2d(pixel_mapper_in_channels, block_out_channels[0] * 4, kernel_size=1),
                nn.GELU(),
                nn.Conv2d(block_out_channels[0] * 4, block_out_channels[0], kernel_size=1),
                WuerstchenLayerNorm(block_out_channels[0], elementwise_affine=False, eps=1e-6),
            )

        self.clip_txt_pooled_mapper = nn.Linear(clip_text_pooled_in_channels, conditioning_dim * clip_seq)
        if clip_text_in_channels is not None:
            self.clip_txt_mapper = nn.Linear(clip_text_in_channels, conditioning_dim)
        if clip_image_in_channels is not None:
            self.clip_img_mapper = nn.Linear(clip_image_in_channels, conditioning_dim * clip_seq)
        self.clip_norm = nn.LayerNorm(conditioning_dim, elementwise_affine=False, eps=1e-6)

        self.embedding = nn.Sequential(
            nn.PixelUnshuffle(patch_size),
            nn.Conv2d(in_channels * (patch_size**2), block_out_channels[0], kernel_size=1),
            WuerstchenLayerNorm(block_out_channels[0], elementwise_affine=False, eps=1e-6),
        )

        def get_block(block_type, in_channels, nhead, c_skip=0, dropout=0, self_attn=True):
            if block_type == "ResBlockStageB":
                return ResBlockStageB(in_channels, c_skip, kernel_size=kernel_size, dropout=dropout)
            elif block_type == "AttnBlock":
                return AttnBlock(in_channels, conditioning_dim, nhead, self_attn=self_attn, dropout=dropout)
            elif block_type == "TimestepBlock":
                return TimestepBlock(in_channels, timestep_ratio_embedding_dim, conds=timestep_conditioning_type)
            else:
                raise ValueError(f"Block type {block_type} not supported")

        # BLOCKS
        # -- down blocks
        self.down_blocks = nn.ModuleList()
        self.down_downscalers = nn.ModuleList()
        self.down_repeat_mappers = nn.ModuleList()
        for i in range(len(block_out_channels)):
            if i > 0:
                self.down_downscalers.append(
                    nn.Sequential(
                        WuerstchenLayerNorm(block_out_channels[i - 1], elementwise_affine=False, eps=1e-6),
                        UpDownBlock2d(
                            block_out_channels[i - 1], block_out_channels[i], mode="down", enabled=switch_level[i - 1]
                        )
                        if switch_level is not None
                        else nn.Conv2d(block_out_channels[i - 1], block_out_channels[i], kernel_size=2, stride=2),
                    )
                )
            else:
                self.down_downscalers.append(nn.Identity())

            down_block = nn.ModuleList()
            for _ in range(down_num_layers_per_block[i]):
                for block_type in block_types_per_layer[i]:
                    block = get_block(
                        block_type,
                        block_out_channels[i],
                        num_attention_heads[i],
                        dropout=dropout[i],
                        self_attn=self_attn[i],
                    )
                    down_block.append(block)
            self.down_blocks.append(down_block)

            if block_repeat is not None:
                block_repeat_mappers = nn.ModuleList()
                for _ in range(down_blocks_repeat_mappers[i] - 1):
                    block_repeat_mappers.append(nn.Conv2d(block_out_channels[i], block_out_channels[i], kernel_size=1))
                self.down_repeat_mappers.append(block_repeat_mappers)

        # -- up blocks
        self.up_blocks = nn.ModuleList()
        self.up_upscalers = nn.ModuleList()
        self.up_repeat_mappers = nn.ModuleList()
        for i in reversed(range(len(block_out_channels))):
            if i > 0:
                self.up_upscalers.append(
                    nn.Sequential(
                        WuerstchenLayerNorm(block_out_channels[i], elementwise_affine=False, eps=1e-6),
                        UpDownBlock2d(
                            block_out_channels[i], block_out_channels[i - 1], mode="up", enabled=switch_level[i - 1]
                        )
                        if switch_level is not None
                        else nn.ConvTranspose2d(
                            block_out_channels[i], block_out_channels[i - 1], kernel_size=2, stride=2
                        ),
                    )
                )
            else:
                self.up_upscalers.append(nn.Identity())

            up_block = nn.ModuleList()
            for j in range(up_num_layers_per_block[::-1][i]):
                for k, block_type in enumerate(block_types_per_layer[i]):
                    c_skip = block_out_channels[i] if i < len(block_out_channels) - 1 and j == k == 0 else 0
                    block = get_block(
                        block_type,
                        block_out_channels[i],
                        num_attention_heads[i],
                        c_skip=c_skip,
                        dropout=dropout[i],
                        self_attn=self_attn[i],
                    )
                    up_block.append(block)
            self.up_blocks.append(up_block)

            if block_repeat is not None:
                block_repeat_mappers = nn.ModuleList()
                for _ in range(up_blocks_repeat_mappers[::-1][i] - 1):
                    block_repeat_mappers.append(nn.Conv2d(block_out_channels[i], block_out_channels[i], kernel_size=1))
                self.up_repeat_mappers.append(block_repeat_mappers)

        # OUTPUT
        self.clf = nn.Sequential(
            WuerstchenLayerNorm(block_out_channels[0], elementwise_affine=False, eps=1e-6),
            nn.Conv2d(block_out_channels[0], out_channels * (patch_size**2), kernel_size=1),
            nn.PixelShuffle(patch_size),
        )

        self.gradient_checkpointing = False

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        nn.init.normal_(self.clip_txt_pooled_mapper.weight, std=0.02)
        nn.init.normal_(self.clip_txt_mapper.weight, std=0.02) if hasattr(self, "clip_txt_mapper") else None
        nn.init.normal_(self.clip_img_mapper.weight, std=0.02) if hasattr(self, "clip_img_mapper") else None

        if hasattr(self, "effnet_mapper"):
            nn.init.normal_(self.effnet_mapper[0].weight, std=0.02)  # conditionings
            nn.init.normal_(self.effnet_mapper[2].weight, std=0.02)  # conditionings

        if hasattr(self, "pixels_mapper"):
            nn.init.normal_(self.pixels_mapper[0].weight, std=0.02)  # conditionings
            nn.init.normal_(self.pixels_mapper[2].weight, std=0.02)  # conditionings

        torch.nn.init.xavier_uniform_(self.embedding[1].weight, 0.02)  # inputs
        nn.init.constant_(self.clf[1].weight, 0)  # outputs

        # blocks
        for level_block in self.down_blocks + self.up_blocks:
            for block in level_block:
                if isinstance(block, ResBlockStageB):
                    block.channelwise[-1].weight.data *= np.sqrt(1 / sum(self.config.blocks[0]))
                elif isinstance(block, TimestepBlock):
                    nn.init.constant_(block.mapper.weight, 0)

    def get_timestep_ratio_embedding(self, timestep_ratio, max_positions=10000):
        r = timestep_ratio * max_positions
        half_dim = self.config.timestep_ratio_embedding_dim // 2

        emb = math.log(max_positions) / (half_dim - 1)
        emb = torch.arange(half_dim, device=r.device).float().mul(-emb).exp()
        emb = r[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=1)

        if self.config.timestep_ratio_embedding_dim % 2 == 1:  # zero pad
            emb = nn.functional.pad(emb, (0, 1), mode="constant")

        return emb.to(dtype=r.dtype)

    def get_clip_embeddings(self, clip_txt_pooled, clip_txt=None, clip_img=None):
        if len(clip_txt_pooled.shape) == 2:
            clip_txt_pool = clip_txt_pooled.unsqueeze(1)
        clip_txt_pool = self.clip_txt_pooled_mapper(clip_txt_pooled).view(
            clip_txt_pooled.size(0), clip_txt_pooled.size(1) * self.config.clip_seq, -1
        )
        if clip_txt is not None and clip_img is not None:
            clip_txt = self.clip_txt_mapper(clip_txt)
            if len(clip_img.shape) == 2:
                clip_img = clip_img.unsqueeze(1)
            clip_img = self.clip_img_mapper(clip_img).view(
                clip_img.size(0), clip_img.size(1) * self.config.clip_seq, -1
            )
            clip = torch.cat([clip_txt, clip_txt_pool, clip_img], dim=1)
        else:
            clip = clip_txt_pool
        return self.clip_norm(clip)

    def _down_encode(self, x, r_embed, clip):
        level_outputs = []
        block_group = zip(self.down_blocks, self.down_downscalers, self.down_repeat_mappers)

        if self.training and self.gradient_checkpointing:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward

            for down_block, downscaler, repmap in block_group:
                x = downscaler(x)
                for i in range(len(repmap) + 1):
                    for block in down_block:
                        if isinstance(block, ResBlockStageB):
                            x = torch.utils.checkpoint.checkpoint(create_custom_forward(block), x, use_reentrant=False)
                        elif isinstance(block, AttnBlock):
                            x = torch.utils.checkpoint.checkpoint(
                                create_custom_forward(block), x, clip, use_reentrant=False
                            )
                        elif isinstance(block, TimestepBlock):
                            x = torch.utils.checkpoint.checkpoint(
                                create_custom_forward(block), x, r_embed, use_reentrant=False
                            )
                        else:
                            x = x = torch.utils.checkpoint.checkpoint(
                                create_custom_forward(block), use_reentrant=False
                            )
                    if i < len(repmap):
                        x = repmap[i](x)
                level_outputs.insert(0, x)
        else:
            for down_block, downscaler, repmap in block_group:
                x = downscaler(x)
                for i in range(len(repmap) + 1):
                    for block in down_block:
                        if isinstance(block, ResBlockStageB):
                            x = block(x)
                        elif isinstance(block, AttnBlock):
                            x = block(x, clip)
                        elif isinstance(block, TimestepBlock):
                            x = block(x, r_embed)
                        else:
                            x = block(x)
                    if i < len(repmap):
                        x = repmap[i](x)
                level_outputs.insert(0, x)
        return level_outputs

    def _up_decode(self, level_outputs, r_embed, clip):
        x = level_outputs[0]
        block_group = zip(self.up_blocks, self.up_upscalers, self.up_repeat_mappers)

        if self.training and self.gradient_checkpointing:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward

            for i, (up_block, upscaler, repmap) in enumerate(block_group):
                for j in range(len(repmap) + 1):
                    for k, block in enumerate(up_block):
                        if isinstance(block, ResBlockStageB):
                            skip = level_outputs[i] if k == 0 and i > 0 else None
                            if skip is not None and (x.size(-1) != skip.size(-1) or x.size(-2) != skip.size(-2)):
                                x = torch.nn.functional.interpolate(
                                    x.float(), skip.shape[-2:], mode="bilinear", align_corners=True
                                )
                            x = torch.utils.checkpoint.checkpoint(
                                create_custom_forward(block), x, skip, use_reentrant=False
                            )
                        elif isinstance(block, AttnBlock):
                            x = torch.utils.checkpoint.checkpoint(
                                create_custom_forward(block), x, clip, use_reentrant=False
                            )
                        elif isinstance(block, TimestepBlock):
                            x = torch.utils.checkpoint.checkpoint(
                                create_custom_forward(block), x, r_embed, use_reentrant=False
                            )
                        else:
                            x = torch.utils.checkpoint.checkpoint(create_custom_forward(block), x, use_reentrant=False)
                    if j < len(repmap):
                        x = repmap[j](x)
                x = upscaler(x)
        else:
            for i, (up_block, upscaler, repmap) in enumerate(block_group):
                for j in range(len(repmap) + 1):
                    for k, block in enumerate(up_block):
                        if isinstance(block, ResBlockStageB):
                            skip = level_outputs[i] if k == 0 and i > 0 else None
                            if skip is not None and (x.size(-1) != skip.size(-1) or x.size(-2) != skip.size(-2)):
                                x = torch.nn.functional.interpolate(
                                    x.float(), skip.shape[-2:], mode="bilinear", align_corners=True
                                )
                            x = block(x, skip)
                        elif isinstance(block, AttnBlock):
                            x = block(x, clip)
                        elif isinstance(block, TimestepBlock):
                            x = block(x, r_embed)
                        else:
                            x = block(x)
                    if j < len(repmap):
                        x = repmap[j](x)
                x = upscaler(x)
        return x

    def forward(
        self,
        sample,
        timestep_ratio,
        clip_text_pooled,
        clip_text=None,
        clip_img=None,
        effnet=None,
        pixels=None,
        sca=None,
        crp=None,
        return_dict=True,
    ):
        if pixels is None:
            pixels = sample.new_zeros(sample.size(0), 3, 8, 8)

        # Process the conditioning embeddings
        timestep_ratio_embed = self.get_timestep_ratio_embedding(timestep_ratio)
        for c in self.config.timestep_conditioning_type:
            if c == "sca":
                cond = sca
            elif c == "crp":
                cond = crp
            else:
                cond = None
            t_cond = cond or torch.zeros_like(timestep_ratio)
            timestep_ratio_embed = torch.cat([timestep_ratio_embed, self.get_timestep_ratio_embedding(t_cond)], dim=1)
        clip = self.get_clip_embeddings(clip_txt_pooled=clip_text_pooled, clip_txt=clip_text, clip_img=clip_img)

        # Model Blocks
        x = self.embedding(sample)
        if hasattr(self, "effnet_mapper") and effnet is not None:
            x = x + self.effnet_mapper(
                nn.functional.interpolate(effnet, size=x.shape[-2:], mode="bilinear", align_corners=True)
            )
        if hasattr(self, "pixels_mapper"):
            x = x + nn.functional.interpolate(
                self.pixels_mapper(pixels), size=x.shape[-2:], mode="bilinear", align_corners=True
            )
        level_outputs = self._down_encode(x, timestep_ratio_embed, clip)
        x = self._up_decode(level_outputs, timestep_ratio_embed, clip)
        sample = self.clf(x)

        if not return_dict:
            return (sample,)
        return StableCascadeUNetOutput(sample=sample)
