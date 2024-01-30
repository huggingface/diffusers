# Copyright (c) 2023 Dominic Rampas MIT License
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
import math

import numpy as np
import torch
import torch.nn as nn
from typing_extensions import List, Optional

from ...configuration_utils import ConfigMixin, register_to_config
from ...models.modeling_utils import ModelMixin
from ..wuerstchen.modeling_wuerstchen_common import AttnBlock, TimestepBlock, WuerstchenLayerNorm
from ..wuerstchen.modeling_wuerstchen_diffnext import ResBlockStageB


class UpDownBlock2d(nn.Module):
    def __init__(self, c_in, c_out, mode, enabled=True):
        super().__init__()
        assert mode in ["up", "down"]
        interpolation = (
            nn.Upsample(scale_factor=2 if mode == "up" else 0.5, mode="bilinear", align_corners=True)
            if enabled
            else nn.Identity()
        )
        mapping = nn.Conv2d(c_in, c_out, kernel_size=1)
        self.blocks = nn.ModuleList([interpolation, mapping] if mode == "up" else [mapping, interpolation])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class WuerstchenV3Unet(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        c_in=16,
        c_out=16,
        c_r=64,
        patch_size=1,
        c_cond=2048,
        c_hidden=[2048, 2048],
        nhead=[32, 32],
        blocks=[[8, 24], [24, 8]],
        block_repeat=[[1, 1], [1, 1]],
        level_config=["CTA", "CTA"],
        c_clip_text: Optional[int] = None,
        c_clip_text_pooled=1280,
        c_clip_img: Optional[int] = None,
        c_clip_seq=4,
        c_effnet: Optional[int] = None,
        c_pixels: Optional[int] = None,
        kernel_size=3,
        dropout: List[float] = [0.1, 0.1],
        self_attn: bool = True,
        t_conds: List[str] = ["sca", "crp"],
        switch_level: Optional[List[bool]] = [False],
    ):
        super().__init__()
        self.c_r = c_r
        self.t_conds = t_conds
        self.c_clip_seq = c_clip_seq
        if not isinstance(dropout, list):
            dropout = [dropout] * len(c_hidden)
        if not isinstance(self_attn, list):
            self_attn = [self_attn] * len(c_hidden)

        # CONDITIONING
        if c_effnet is not None:
            self.effnet_mapper = nn.Sequential(
                nn.Conv2d(c_effnet, c_hidden[0] * 4, kernel_size=1),
                nn.GELU(),
                nn.Conv2d(c_hidden[0] * 4, c_hidden[0], kernel_size=1),
                WuerstchenLayerNorm(c_hidden[0], elementwise_affine=False, eps=1e-6),
            )
        if c_pixels is not None:
            self.pixels_mapper = nn.Sequential(
                nn.Conv2d(c_pixels, c_hidden[0] * 4, kernel_size=1),
                nn.GELU(),
                nn.Conv2d(c_hidden[0] * 4, c_hidden[0], kernel_size=1),
                WuerstchenLayerNorm(c_hidden[0], elementwise_affine=False, eps=1e-6),
            )

        self.clip_txt_pooled_mapper = nn.Linear(c_clip_text_pooled, c_cond * c_clip_seq)
        if c_clip_text is not None:
            self.clip_txt_mapper = nn.Linear(c_clip_text, c_cond)
        if c_clip_img is not None:
            self.clip_img_mapper = nn.Linear(c_clip_img, c_cond * c_clip_seq)
        self.clip_norm = nn.LayerNorm(c_cond, elementwise_affine=False, eps=1e-6)

        self.embedding = nn.Sequential(
            nn.PixelUnshuffle(patch_size),
            nn.Conv2d(c_in * (patch_size**2), c_hidden[0], kernel_size=1),
            WuerstchenLayerNorm(c_hidden[0], elementwise_affine=False, eps=1e-6),
        )

        def get_block(block_type, c_hidden, nhead, c_skip=0, dropout=0, self_attn=True):
            if block_type == "C":
                return ResBlockStageB(c_hidden, c_skip, kernel_size=kernel_size, dropout=dropout)
            elif block_type == "A":
                return AttnBlock(c_hidden, c_cond, nhead, self_attn=self_attn, dropout=dropout)
            elif block_type == "T":
                return TimestepBlock(c_hidden, c_r, conds=t_conds)
            else:
                raise Exception(f"Block type {block_type} not supported")

        # BLOCKS
        # -- down blocks
        self.down_blocks = nn.ModuleList()
        self.down_downscalers = nn.ModuleList()
        self.down_repeat_mappers = nn.ModuleList()
        for i in range(len(c_hidden)):
            if i > 0:
                self.down_downscalers.append(
                    nn.Sequential(
                        WuerstchenLayerNorm(c_hidden[i - 1], elementwise_affine=False, eps=1e-6),
                        UpDownBlock2d(c_hidden[i - 1], c_hidden[i], mode="down", enabled=switch_level[i - 1])
                        if switch_level is not None
                        else nn.Conv2d(c_hidden[i - 1], c_hidden[i], kernel_size=2, stride=2),
                    )
                )
            else:
                self.down_downscalers.append(nn.Identity())
            down_block = nn.ModuleList()
            for _ in range(blocks[0][i]):
                for block_type in level_config[i]:
                    block = get_block(block_type, c_hidden[i], nhead[i], dropout=dropout[i], self_attn=self_attn[i])
                    down_block.append(block)
            self.down_blocks.append(down_block)
            if block_repeat is not None:
                block_repeat_mappers = nn.ModuleList()
                for _ in range(block_repeat[0][i] - 1):
                    block_repeat_mappers.append(nn.Conv2d(c_hidden[i], c_hidden[i], kernel_size=1))
                self.down_repeat_mappers.append(block_repeat_mappers)

        # -- up blocks
        self.up_blocks = nn.ModuleList()
        self.up_upscalers = nn.ModuleList()
        self.up_repeat_mappers = nn.ModuleList()
        for i in reversed(range(len(c_hidden))):
            if i > 0:
                self.up_upscalers.append(
                    nn.Sequential(
                        WuerstchenLayerNorm(c_hidden[i], elementwise_affine=False, eps=1e-6),
                        UpDownBlock2d(c_hidden[i], c_hidden[i - 1], mode="up", enabled=switch_level[i - 1])
                        if switch_level is not None
                        else nn.ConvTranspose2d(c_hidden[i], c_hidden[i - 1], kernel_size=2, stride=2),
                    )
                )
            else:
                self.up_upscalers.append(nn.Identity())
            up_block = nn.ModuleList()
            for j in range(blocks[1][::-1][i]):
                for k, block_type in enumerate(level_config[i]):
                    c_skip = c_hidden[i] if i < len(c_hidden) - 1 and j == k == 0 else 0
                    block = get_block(
                        block_type, c_hidden[i], nhead[i], c_skip=c_skip, dropout=dropout[i], self_attn=self_attn[i]
                    )
                    up_block.append(block)
            self.up_blocks.append(up_block)
            if block_repeat is not None:
                block_repeat_mappers = nn.ModuleList()
                for _ in range(block_repeat[1][::-1][i] - 1):
                    block_repeat_mappers.append(nn.Conv2d(c_hidden[i], c_hidden[i], kernel_size=1))
                self.up_repeat_mappers.append(block_repeat_mappers)

        # OUTPUT
        self.clf = nn.Sequential(
            WuerstchenLayerNorm(c_hidden[0], elementwise_affine=False, eps=1e-6),
            nn.Conv2d(c_hidden[0], c_out * (patch_size**2), kernel_size=1),
            nn.PixelShuffle(patch_size),
        )

        # --- WEIGHT INIT ---
        # self.apply(self._init_weights)  # General init

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

    def gen_r_embedding(self, r, max_positions=10000):
        r = r * max_positions
        half_dim = self.c_r // 2
        emb = math.log(max_positions) / (half_dim - 1)
        emb = torch.arange(half_dim, device=r.device).float().mul(-emb).exp()
        emb = r[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=1)
        if self.c_r % 2 == 1:  # zero pad
            emb = nn.functional.pad(emb, (0, 1), mode="constant")
        return emb.to(dtype=r.dtype)

    def gen_c_embeddings(self, clip_txt_pooled, clip_txt=None, clip_img=None):
        if len(clip_txt_pooled.shape) == 2:
            clip_txt_pool = clip_txt_pooled.unsqueeze(1)
        clip_txt_pool = self.clip_txt_pooled_mapper(clip_txt_pooled).view(
            clip_txt_pooled.size(0), clip_txt_pooled.size(1) * self.c_clip_seq, -1
        )
        if clip_txt is not None and clip_img is not None:
            clip_txt = self.clip_txt_mapper(clip_txt)
            if len(clip_img.shape) == 2:
                clip_img = clip_img.unsqueeze(1)
            clip_img = self.clip_img_mapper(clip_img).view(clip_img.size(0), clip_img.size(1) * self.c_clip_seq, -1)
            clip = torch.cat([clip_txt, clip_txt_pool, clip_img], dim=1)
        else:
            clip = clip_txt_pool
        return self.clip_norm(clip)

    def _down_encode(self, x, r_embed, clip):
        level_outputs = []
        block_group = zip(self.down_blocks, self.down_downscalers, self.down_repeat_mappers)
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

    def forward(self, x, r, clip_text, clip_text_pooled, clip_img, **kwargs):
        # Process the conditioning embeddings
        r_embed = self.gen_r_embedding(r)
        for c in self.t_conds:
            t_cond = kwargs.get(c, torch.zeros_like(r))
            r_embed = torch.cat([r_embed, self.gen_r_embedding(t_cond)], dim=1)
        clip = self.gen_c_embeddings(clip_text_pooled=clip_text_pooled, clip_txt=clip_text, clip_img=clip_img)

        # Model Blocks
        x = self.embedding(x)
        level_outputs = self._down_encode(x, r_embed, clip)
        x = self._up_decode(level_outputs, r_embed, clip)
        return self.clf(x)
