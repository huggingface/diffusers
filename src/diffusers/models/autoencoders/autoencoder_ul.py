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

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from ...configuration_utils import ConfigMixin, register_to_config
from ...utils.accelerate_utils import apply_forward_hook
from ..modeling_utils import ModelMixin
from ..unets.unet_2d_blocks import UNetMidBlock2D, get_down_block
from .vae import EncoderOutput


class ULEncoder(nn.Module):
    """
    ResNet-style encoder with per-stage residual depth.

    This is equivalent to the Diffusers VAE encoder structure, but supports `layers_per_block` as a tuple to match UL
    paper Section 5.1 exactly.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 4,
        down_block_types: tuple[str, ...] = (
            "DownEncoderBlock2D",
            "DownEncoderBlock2D",
            "DownEncoderBlock2D",
            "DownEncoderBlock2D",
        ),
        block_out_channels: tuple[int, ...] = (128, 256, 512, 512),
        layers_per_block: tuple[int, ...] = (2, 2, 2, 3),
        norm_num_groups: int = 32,
        act_fn: str = "silu",
        mid_block_add_attention: bool = True,
    ):
        super().__init__()
        if len(down_block_types) != len(block_out_channels):
            raise ValueError("`down_block_types` and `block_out_channels` must have the same length.")
        if len(layers_per_block) != len(block_out_channels):
            raise ValueError("`layers_per_block` must have the same length as `block_out_channels`.")

        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], kernel_size=3, stride=1, padding=1)
        self.down_blocks = nn.ModuleList([])

        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1
            down_block = get_down_block(
                down_block_type,
                num_layers=layers_per_block[i],
                in_channels=input_channel,
                out_channels=output_channel,
                add_downsample=not is_final_block,
                resnet_eps=1e-6,
                downsample_padding=0,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                attention_head_dim=output_channel,
                temb_channels=None,
            )
            self.down_blocks.append(down_block)

        self.mid_block = UNetMidBlock2D(
            in_channels=block_out_channels[-1],
            resnet_eps=1e-6,
            resnet_act_fn=act_fn,
            output_scale_factor=1,
            resnet_time_scale_shift="default",
            attention_head_dim=block_out_channels[-1],
            resnet_groups=norm_num_groups,
            temb_channels=None,
            add_attention=mid_block_add_attention,
        )
        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[-1], num_groups=norm_num_groups, eps=1e-6)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(block_out_channels[-1], out_channels, kernel_size=3, padding=1)
        self.gradient_checkpointing = False

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        sample = self.conv_in(sample)
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for down_block in self.down_blocks:
                sample = checkpoint(down_block, sample, use_reentrant=False)
            sample = checkpoint(self.mid_block, sample, use_reentrant=False)
        else:
            for down_block in self.down_blocks:
                sample = down_block(sample)
            sample = self.mid_block(sample)

        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)
        return sample


class AutoencoderULEncoder(ModelMixin, ConfigMixin):
    """
    Deterministic UL encoder model used in stage-1/stage-2 training.
    """

    _supports_gradient_checkpointing = True
    _no_split_modules = ["ResnetBlock2D"]

    @register_to_config
    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 4,
        down_block_types: tuple[str, ...] = (
            "DownEncoderBlock2D",
            "DownEncoderBlock2D",
            "DownEncoderBlock2D",
            "DownEncoderBlock2D",
        ),
        block_out_channels: tuple[int, ...] = (128, 256, 512, 512),
        layers_per_block: tuple[int, ...] = (2, 2, 2, 3),
        norm_num_groups: int = 32,
        act_fn: str = "silu",
        mid_block_add_attention: bool = True,
    ):
        super().__init__()
        self.encoder = ULEncoder(
            in_channels=in_channels,
            out_channels=latent_channels,
            down_block_types=down_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            norm_num_groups=norm_num_groups,
            act_fn=act_fn,
            mid_block_add_attention=mid_block_add_attention,
        )
        self.out_proj = nn.Conv2d(latent_channels, latent_channels, kernel_size=1)

    @apply_forward_hook
    def encode(self, x: torch.Tensor, return_dict: bool = True) -> EncoderOutput | tuple[torch.Tensor]:
        latents = self.out_proj(self.encoder(x))
        if not return_dict:
            return (latents,)
        return EncoderOutput(latent=latents)

    def forward(self, sample: torch.Tensor, return_dict: bool = True) -> EncoderOutput | tuple[torch.Tensor]:
        return self.encode(sample, return_dict=return_dict)

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value
