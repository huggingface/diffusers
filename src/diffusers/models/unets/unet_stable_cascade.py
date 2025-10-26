# Copyright 2025 The HuggingFace Team. All rights reserved.
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
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from ...configuration_utils import ConfigMixin, register_to_config
from ...loaders import FromOriginalModelMixin
from ...utils import BaseOutput
from ..attention_processor import Attention
from ..modeling_utils import ModelMixin


# Copied from diffusers.pipelines.wuerstchen.modeling_wuerstchen_common.WuerstchenLayerNorm with WuerstchenLayerNorm -> SDCascadeLayerNorm
class SDCascadeLayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = super().forward(x)
        return x.permute(0, 3, 1, 2)


class SDCascadeTimestepBlock(nn.Module):
    def __init__(self, c, c_timestep, conds=[]):
        super().__init__()

        self.mapper = nn.Linear(c_timestep, c * 2)
        self.conds = conds
        for cname in conds:
            setattr(self, f"mapper_{cname}", nn.Linear(c_timestep, c * 2))

    def forward(self, x, t):
        t = t.chunk(len(self.conds) + 1, dim=1)
        a, b = self.mapper(t[0])[:, :, None, None].chunk(2, dim=1)
        for i, c in enumerate(self.conds):
            ac, bc = getattr(self, f"mapper_{c}")(t[i + 1])[:, :, None, None].chunk(2, dim=1)
            a, b = a + ac, b + bc
        return x * (1 + a) + b


class SDCascadeResBlock(nn.Module):
    def __init__(self, c, c_skip=0, kernel_size=3, dropout=0.0):
        super().__init__()
        self.depthwise = nn.Conv2d(c, c, kernel_size=kernel_size, padding=kernel_size // 2, groups=c)
        self.norm = SDCascadeLayerNorm(c, elementwise_affine=False, eps=1e-6)
        self.channelwise = nn.Sequential(
            nn.Linear(c + c_skip, c * 4),
            nn.GELU(),
            GlobalResponseNorm(c * 4),
            nn.Dropout(dropout),
            nn.Linear(c * 4, c),
        )

    def forward(self, x, x_skip=None):
        x_res = x
        x = self.norm(self.depthwise(x))
        if x_skip is not None:
            x = torch.cat([x, x_skip], dim=1)
        x = self.channelwise(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return x + x_res


# from https://github.com/facebookresearch/ConvNeXt-V2/blob/3608f67cc1dae164790c5d0aead7bf2d73d9719b/models/utils.py#L105
class GlobalResponseNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        agg_norm = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        stand_div_norm = agg_norm / (agg_norm.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * stand_div_norm) + self.beta + x


class SDCascadeAttnBlock(nn.Module):
    def __init__(self, c, c_cond, nhead, self_attn=True, dropout=0.0):
        super().__init__()

        self.self_attn = self_attn
        self.norm = SDCascadeLayerNorm(c, elementwise_affine=False, eps=1e-6)
        self.attention = Attention(query_dim=c, heads=nhead, dim_head=c // nhead, dropout=dropout, bias=True)
        self.kv_mapper = nn.Sequential(nn.SiLU(), nn.Linear(c_cond, c))

    def forward(self, x, kv):
        kv = self.kv_mapper(kv)
        norm_x = self.norm(x)
        if self.self_attn:
            batch_size, channel, _, _ = x.shape
            kv = torch.cat([norm_x.view(batch_size, channel, -1).transpose(1, 2), kv], dim=1)
        x = x + self.attention(norm_x, encoder_hidden_states=kv)
        return x


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
    sample: torch.Tensor = None


class StableCascadeUNet(ModelMixin, ConfigMixin, FromOriginalModelMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        in_channels: int = 16,
        out_channels: int = 16,
        timestep_ratio_embedding_dim: int = 64,
        patch_size: int = 1,
        conditioning_dim: int = 2048,
        block_out_channels: Tuple[int, ...] = (2048, 2048),
        num_attention_heads: Tuple[int, ...] = (32, 32),
        down_num_layers_per_block: Tuple[int, ...] = (8, 24),
        up_num_layers_per_block: Tuple[int, ...] = (24, 8),
        down_blocks_repeat_mappers: Optional[Tuple[int]] = (
            1,
            1,
        ),
        up_blocks_repeat_mappers: Optional[Tuple[int]] = (1, 1),
        block_types_per_layer: Tuple[Tuple[str]] = (
            ("SDCascadeResBlock", "SDCascadeTimestepBlock", "SDCascadeAttnBlock"),
            ("SDCascadeResBlock", "SDCascadeTimestepBlock", "SDCascadeAttnBlock"),
        ),
        clip_text_in_channels: Optional[int] = None,
        clip_text_pooled_in_channels=1280,
        clip_image_in_channels: Optional[int] = None,
        clip_seq=4,
        effnet_in_channels: Optional[int] = None,
        pixel_mapper_in_channels: Optional[int] = None,
        kernel_size=3,
        dropout: Union[float, Tuple[float]] = (0.1, 0.1),
        self_attn: Union[bool, Tuple[bool]] = True,
        timestep_conditioning_type: Tuple[str, ...] = ("sca", "crp"),
        switch_level: Optional[Tuple[bool]] = None,
    ):
        """

        Parameters:
            in_channels (`int`, defaults to 16):
                Number of channels in the input sample.
            out_channels (`int`, defaults to 16):
                Number of channels in the output sample.
            timestep_ratio_embedding_dim (`int`, defaults to 64):
                Dimension of the projected time embedding.
            patch_size (`int`, defaults to 1):
                Patch size to use for pixel unshuffling layer
            conditioning_dim (`int`, defaults to 2048):
                Dimension of the image and text conditional embedding.
            block_out_channels (Tuple[int], defaults to (2048, 2048)):
                Tuple of output channels for each block.
            num_attention_heads (Tuple[int], defaults to (32, 32)):
                Number of attention heads in each attention block. Set to -1 to if block types in a layer do not have
                attention.
            down_num_layers_per_block (Tuple[int], defaults to [8, 24]):
                Number of layers in each down block.
            up_num_layers_per_block (Tuple[int], defaults to [24, 8]):
                Number of layers in each up block.
            down_blocks_repeat_mappers (Tuple[int], optional, defaults to [1, 1]):
                Number of 1x1 Convolutional layers to repeat in each down block.
            up_blocks_repeat_mappers (Tuple[int], optional, defaults to [1, 1]):
                Number of 1x1 Convolutional layers to repeat in each up block.
            block_types_per_layer (Tuple[Tuple[str]], optional,
                defaults to (
                    ("SDCascadeResBlock", "SDCascadeTimestepBlock", "SDCascadeAttnBlock"), ("SDCascadeResBlock",
                    "SDCascadeTimestepBlock", "SDCascadeAttnBlock")
                ): Block types used in each layer of the up/down blocks.
            clip_text_in_channels (`int`, *optional*, defaults to `None`):
                Number of input channels for CLIP based text conditioning.
            clip_text_pooled_in_channels (`int`, *optional*, defaults to 1280):
                Number of input channels for pooled CLIP text embeddings.
            clip_image_in_channels (`int`, *optional*):
                Number of input channels for CLIP based image conditioning.
            clip_seq (`int`, *optional*, defaults to 4):
            effnet_in_channels (`int`, *optional*, defaults to `None`):
                Number of input channels for effnet conditioning.
            pixel_mapper_in_channels (`int`, defaults to `None`):
                Number of input channels for pixel mapper conditioning.
            kernel_size (`int`, *optional*, defaults to 3):
                Kernel size to use in the block convolutional layers.
            dropout (Tuple[float], *optional*, defaults to (0.1, 0.1)):
                Dropout to use per block.
            self_attn (Union[bool, Tuple[bool]]):
                Tuple of booleans that determine whether to use self attention in a block or not.
            timestep_conditioning_type (Tuple[str], defaults to ("sca", "crp")):
                Timestep conditioning type.
            switch_level (Optional[Tuple[bool]], *optional*, defaults to `None`):
                Tuple that indicates whether upsampling or downsampling should be applied in a block
        """

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

        if isinstance(dropout, float):
            dropout = (dropout,) * len(block_out_channels)
        if isinstance(self_attn, bool):
            self_attn = (self_attn,) * len(block_out_channels)

        # CONDITIONING
        if effnet_in_channels is not None:
            self.effnet_mapper = nn.Sequential(
                nn.Conv2d(effnet_in_channels, block_out_channels[0] * 4, kernel_size=1),
                nn.GELU(),
                nn.Conv2d(block_out_channels[0] * 4, block_out_channels[0], kernel_size=1),
                SDCascadeLayerNorm(block_out_channels[0], elementwise_affine=False, eps=1e-6),
            )
        if pixel_mapper_in_channels is not None:
            self.pixels_mapper = nn.Sequential(
                nn.Conv2d(pixel_mapper_in_channels, block_out_channels[0] * 4, kernel_size=1),
                nn.GELU(),
                nn.Conv2d(block_out_channels[0] * 4, block_out_channels[0], kernel_size=1),
                SDCascadeLayerNorm(block_out_channels[0], elementwise_affine=False, eps=1e-6),
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
            SDCascadeLayerNorm(block_out_channels[0], elementwise_affine=False, eps=1e-6),
        )

        def get_block(block_type, in_channels, nhead, c_skip=0, dropout=0, self_attn=True):
            if block_type == "SDCascadeResBlock":
                return SDCascadeResBlock(in_channels, c_skip, kernel_size=kernel_size, dropout=dropout)
            elif block_type == "SDCascadeAttnBlock":
                return SDCascadeAttnBlock(in_channels, conditioning_dim, nhead, self_attn=self_attn, dropout=dropout)
            elif block_type == "SDCascadeTimestepBlock":
                return SDCascadeTimestepBlock(
                    in_channels, timestep_ratio_embedding_dim, conds=timestep_conditioning_type
                )
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
                        SDCascadeLayerNorm(block_out_channels[i - 1], elementwise_affine=False, eps=1e-6),
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

            if down_blocks_repeat_mappers is not None:
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
                        SDCascadeLayerNorm(block_out_channels[i], elementwise_affine=False, eps=1e-6),
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

            if up_blocks_repeat_mappers is not None:
                block_repeat_mappers = nn.ModuleList()
                for _ in range(up_blocks_repeat_mappers[::-1][i] - 1):
                    block_repeat_mappers.append(nn.Conv2d(block_out_channels[i], block_out_channels[i], kernel_size=1))
                self.up_repeat_mappers.append(block_repeat_mappers)

        # OUTPUT
        self.clf = nn.Sequential(
            SDCascadeLayerNorm(block_out_channels[0], elementwise_affine=False, eps=1e-6),
            nn.Conv2d(block_out_channels[0], out_channels * (patch_size**2), kernel_size=1),
            nn.PixelShuffle(patch_size),
        )

        self.gradient_checkpointing = False

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
                if isinstance(block, SDCascadeResBlock):
                    block.channelwise[-1].weight.data *= np.sqrt(1 / sum(self.config.blocks[0]))
                elif isinstance(block, SDCascadeTimestepBlock):
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

        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for down_block, downscaler, repmap in block_group:
                x = downscaler(x)
                for i in range(len(repmap) + 1):
                    for block in down_block:
                        if isinstance(block, SDCascadeResBlock):
                            x = self._gradient_checkpointing_func(block, x)
                        elif isinstance(block, SDCascadeAttnBlock):
                            x = self._gradient_checkpointing_func(block, x, clip)
                        elif isinstance(block, SDCascadeTimestepBlock):
                            x = self._gradient_checkpointing_func(block, x, r_embed)
                        else:
                            x = self._gradient_checkpointing_func(block)
                    if i < len(repmap):
                        x = repmap[i](x)
                level_outputs.insert(0, x)
        else:
            for down_block, downscaler, repmap in block_group:
                x = downscaler(x)
                for i in range(len(repmap) + 1):
                    for block in down_block:
                        if isinstance(block, SDCascadeResBlock):
                            x = block(x)
                        elif isinstance(block, SDCascadeAttnBlock):
                            x = block(x, clip)
                        elif isinstance(block, SDCascadeTimestepBlock):
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

        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for i, (up_block, upscaler, repmap) in enumerate(block_group):
                for j in range(len(repmap) + 1):
                    for k, block in enumerate(up_block):
                        if isinstance(block, SDCascadeResBlock):
                            skip = level_outputs[i] if k == 0 and i > 0 else None
                            if skip is not None and (x.size(-1) != skip.size(-1) or x.size(-2) != skip.size(-2)):
                                orig_type = x.dtype
                                x = torch.nn.functional.interpolate(
                                    x.float(), skip.shape[-2:], mode="bilinear", align_corners=True
                                )
                                x = x.to(orig_type)
                            x = self._gradient_checkpointing_func(block, x, skip)
                        elif isinstance(block, SDCascadeAttnBlock):
                            x = self._gradient_checkpointing_func(block, x, clip)
                        elif isinstance(block, SDCascadeTimestepBlock):
                            x = self._gradient_checkpointing_func(block, x, r_embed)
                        else:
                            x = self._gradient_checkpointing_func(block, x)
                    if j < len(repmap):
                        x = repmap[j](x)
                x = upscaler(x)
        else:
            for i, (up_block, upscaler, repmap) in enumerate(block_group):
                for j in range(len(repmap) + 1):
                    for k, block in enumerate(up_block):
                        if isinstance(block, SDCascadeResBlock):
                            skip = level_outputs[i] if k == 0 and i > 0 else None
                            if skip is not None and (x.size(-1) != skip.size(-1) or x.size(-2) != skip.size(-2)):
                                orig_type = x.dtype
                                x = torch.nn.functional.interpolate(
                                    x.float(), skip.shape[-2:], mode="bilinear", align_corners=True
                                )
                                x = x.to(orig_type)
                            x = block(x, skip)
                        elif isinstance(block, SDCascadeAttnBlock):
                            x = block(x, clip)
                        elif isinstance(block, SDCascadeTimestepBlock):
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
