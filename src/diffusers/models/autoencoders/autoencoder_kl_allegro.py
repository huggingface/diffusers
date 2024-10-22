# Copyright 2024 The RhymesAI and The HuggingFace Team.
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

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from ...configuration_utils import ConfigMixin, register_to_config
from ..attention_processor import Attention, SpatialNorm
from ..autoencoders.vae import DecoderOutput, DiagonalGaussianDistribution
from ..downsampling import Downsample2D
from ..modeling_outputs import AutoencoderKLOutput
from ..modeling_utils import ModelMixin
from ..resnet import ResnetBlock2D
from ..upsampling import Upsample2D


class AllegroTemporalConvBlock(nn.Module):
    r"""
    Temporal convolutional layer that can be used for video (sequence of images) input. Code adapted from:
    https://github.com/modelscope/modelscope/blob/1509fdb973e5871f37148a4b5e5964cafd43e64d/modelscope/models/multi_modal/video_synthesis/unet_sd.py#L1016
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: Optional[int] = None,
        dropout: float = 0.0,
        up_sample: bool = False,
        down_sample: bool = False,
        stride: int = 1,
    ) -> None:
        super().__init__()

        out_dim = out_dim or in_dim
        pad_h = pad_w = int((stride - 1) * 0.5)
        pad_t = 0

        self.down_sample = down_sample
        self.up_sample = up_sample

        if down_sample:
            self.conv1 = nn.Sequential(
                nn.GroupNorm(32, in_dim),
                nn.SiLU(),
                nn.Conv3d(in_dim, out_dim, (2, stride, stride), stride=(2, 1, 1), padding=(0, pad_h, pad_w)),
            )
        elif up_sample:
            self.conv1 = nn.Sequential(
                nn.GroupNorm(32, in_dim),
                nn.SiLU(),
                nn.Conv3d(in_dim, out_dim * 2, (1, stride, stride), padding=(0, pad_h, pad_w)),
            )
        else:
            self.conv1 = nn.Sequential(
                nn.GroupNorm(32, in_dim),
                nn.SiLU(),
                nn.Conv3d(in_dim, out_dim, (3, stride, stride), padding=(pad_t, pad_h, pad_w)),
            )
        self.conv2 = nn.Sequential(
            nn.GroupNorm(32, out_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv3d(out_dim, in_dim, (3, stride, stride), padding=(pad_t, pad_h, pad_w)),
        )
        self.conv3 = nn.Sequential(
            nn.GroupNorm(32, out_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv3d(out_dim, in_dim, (3, stride, stride), padding=(pad_t, pad_h, pad_h)),
        )
        self.conv4 = nn.Sequential(
            nn.GroupNorm(32, out_dim),
            nn.SiLU(),
            nn.Conv3d(out_dim, in_dim, (3, stride, stride), padding=(pad_t, pad_h, pad_h)),
        )

        # zero out the last layer params, so the conv block is identity
        nn.init.zeros_(self.conv4[-1].weight)
        nn.init.zeros_(self.conv4[-1].bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        identity = hidden_states

        if self.down_sample:
            identity = identity[:, :, ::2]
        elif self.up_sample:
            hidden_states_new = torch.cat((hidden_states, hidden_states), dim=2)
            hidden_states_new[:, :, 0::2] = hidden_states
            hidden_states_new[:, :, 1::2] = hidden_states
            identity = hidden_states_new
            del hidden_states_new

        if self.down_sample or self.up_sample:
            hidden_states = self.conv1(hidden_states)
        else:
            hidden_states = torch.cat((hidden_states[:, :, 0:1], hidden_states), dim=2)
            hidden_states = torch.cat((hidden_states, hidden_states[:, :, -1:]), dim=2)
            hidden_states = self.conv1(hidden_states)

        if self.up_sample:
            hidden_states = hidden_states.unflatten(1, (2, -1)).permute(0, 2, 3, 1, 4, 5).flatten(2, 3)

        hidden_states = torch.cat((hidden_states[:, :, 0:1], hidden_states), dim=2)
        hidden_states = torch.cat((hidden_states, hidden_states[:, :, -1:]), dim=2)
        hidden_states = self.conv2(hidden_states)
        hidden_states = torch.cat((hidden_states[:, :, 0:1], hidden_states), dim=2)
        hidden_states = torch.cat((hidden_states, hidden_states[:, :, -1:]), dim=2)
        hidden_states = self.conv3(hidden_states)
        hidden_states = torch.cat((hidden_states[:, :, 0:1], hidden_states), dim=2)
        hidden_states = torch.cat((hidden_states, hidden_states[:, :, -1:]), dim=2)
        hidden_states = self.conv4(hidden_states)

        hidden_states = identity + hidden_states

        return hidden_states


class AllegroDownBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor=1.0,
        add_downsample=True,
        add_temp_downsample=False,
        downsample_padding=1,
    ):
        super().__init__()

        resnets = []
        temp_convs = []

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=None,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
            temp_convs.append(
                AllegroTemporalConvBlock(
                    out_channels,
                    out_channels,
                    dropout=0.1,
                )
            )

        self.resnets = nn.ModuleList(resnets)
        self.temp_convs = nn.ModuleList(temp_convs)

        if add_temp_downsample:
            self.temp_convs_down = AllegroTemporalConvBlock(
                out_channels, out_channels, dropout=0.1, down_sample=True, stride=3
            )
        self.add_temp_downsample = add_temp_downsample

        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    Downsample2D(
                        out_channels, use_conv=True, out_channels=out_channels, padding=downsample_padding, name="op"
                    )
                ]
            )
        else:
            self.downsamplers = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size = hidden_states.shape[0]

        for resnet, temp_conv in zip(self.resnets, self.temp_convs):
            hidden_states = hidden_states.permute(0, 2, 1, 3, 4).flatten(0, 1)
            hidden_states = resnet(hidden_states, temb=None)
            hidden_states = hidden_states.unflatten(0, (batch_size, -1)).permute(0, 2, 1, 3, 4)
            hidden_states = temp_conv(hidden_states)

        if self.add_temp_downsample:
            hidden_states = self.temp_convs_down(hidden_states)

        if self.downsamplers is not None:
            hidden_states = hidden_states.permute(0, 2, 1, 3, 4).flatten(0, 1)
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)
            hidden_states = hidden_states.unflatten(0, (batch_size, -1)).permute(0, 2, 1, 3, 4)

        return hidden_states


class AllegroUpBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",  # default, spatial
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor=1.0,
        add_upsample=True,
        add_temp_upsample=False,
        temb_channels=None,
    ):
        super().__init__()
        self.add_upsample = add_upsample

        resnets = []
        temp_convs = []

        for i in range(num_layers):
            input_channels = in_channels if i == 0 else out_channels

            resnets.append(
                ResnetBlock2D(
                    in_channels=input_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
            temp_convs.append(
                AllegroTemporalConvBlock(
                    out_channels,
                    out_channels,
                    dropout=0.1,
                )
            )

        self.resnets = nn.ModuleList(resnets)
        self.temp_convs = nn.ModuleList(temp_convs)

        self.add_temp_upsample = add_temp_upsample
        if add_temp_upsample:
            self.temp_conv_up = AllegroTemporalConvBlock(
                out_channels, out_channels, dropout=0.1, up_sample=True, stride=3
            )

        if self.add_upsample:
            self.upsamplers = nn.ModuleList([Upsample2D(out_channels, use_conv=True, out_channels=out_channels)])
        else:
            self.upsamplers = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size = hidden_states.shape[0]

        for resnet, temp_conv in zip(self.resnets, self.temp_convs):
            hidden_states = hidden_states.permute(0, 2, 1, 3, 4).flatten(0, 1)
            hidden_states = resnet(hidden_states, temb=None)
            hidden_states = hidden_states.unflatten(0, (batch_size, -1)).permute(0, 2, 1, 3, 4)
            hidden_states = temp_conv(hidden_states)

        if self.add_temp_upsample:
            hidden_states = self.temp_conv_up(hidden_states)

        if self.upsamplers is not None:
            hidden_states = hidden_states.permute(0, 2, 1, 3, 4).flatten(0, 1)
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)
            hidden_states = hidden_states.unflatten(0, (batch_size, -1)).permute(0, 2, 1, 3, 4)

        return hidden_states


class UNetMidBlock3DConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",  # default, spatial
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        add_attention: bool = True,
        attention_head_dim=1,
        output_scale_factor=1.0,
    ):
        super().__init__()
        resnet_groups = resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)
        self.add_attention = add_attention

        # there is always at least one resnet
        resnets = [
            ResnetBlock2D(
                in_channels=in_channels,
                out_channels=in_channels,
                temb_channels=temb_channels,
                eps=resnet_eps,
                groups=resnet_groups,
                dropout=dropout,
                time_embedding_norm=resnet_time_scale_shift,
                non_linearity=resnet_act_fn,
                output_scale_factor=output_scale_factor,
                pre_norm=resnet_pre_norm,
            )
        ]
        temp_convs = [
            AllegroTemporalConvBlock(
                in_channels,
                in_channels,
                dropout=0.1,
            )
        ]
        attentions = []

        if attention_head_dim is None:
            attention_head_dim = in_channels

        for _ in range(num_layers):
            if self.add_attention:
                attentions.append(
                    Attention(
                        in_channels,
                        heads=in_channels // attention_head_dim,
                        dim_head=attention_head_dim,
                        rescale_output_factor=output_scale_factor,
                        eps=resnet_eps,
                        norm_num_groups=resnet_groups if resnet_time_scale_shift == "default" else None,
                        spatial_norm_dim=temb_channels if resnet_time_scale_shift == "spatial" else None,
                        residual_connection=True,
                        bias=True,
                        upcast_softmax=True,
                        _from_deprecated_attn_block=True,
                    )
                )
            else:
                attentions.append(None)

            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )

            temp_convs.append(
                AllegroTemporalConvBlock(
                    in_channels,
                    in_channels,
                    dropout=0.1,
                )
            )

        self.resnets = nn.ModuleList(resnets)
        self.temp_convs = nn.ModuleList(temp_convs)
        self.attentions = nn.ModuleList(attentions)

    def forward(self, hidden_states: torch.Tensor):
        batch_size = hidden_states.shape[0]

        hidden_states = hidden_states.permute(0, 2, 1, 3, 4).flatten(0, 1)
        hidden_states = self.resnets[0](hidden_states, temb=None)
        hidden_states = hidden_states.unflatten(0, (batch_size, -1)).permute(0, 2, 1, 3, 4)
        hidden_states = self.temp_convs[0](hidden_states)
        hidden_states = hidden_states.permute(0, 2, 1, 3, 4).flatten(0, 1)

        for attn, resnet, temp_conv in zip(self.attentions, self.resnets[1:], self.temp_convs[1:]):
            hidden_states = attn(hidden_states)
            hidden_states = resnet(hidden_states, temb=None)
            hidden_states = hidden_states.unflatten(0, (batch_size, -1)).permute(0, 2, 1, 3, 4)
            hidden_states = temp_conv(hidden_states)

        return hidden_states


class AllegroEncoder3D(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: Tuple[str, ...] = (
            "AllegroDownBlock3D",
            "AllegroDownBlock3D",
            "AllegroDownBlock3D",
            "AllegroDownBlock3D",
        ),
        blocks_temp_li=[False, False, False, False],
        block_out_channels: Tuple[int, ...] = (128, 256, 512, 512),
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        act_fn: str = "silu",
        double_z: bool = True,
    ):
        super().__init__()

        self.layers_per_block = layers_per_block
        self.blocks_temp_li = blocks_temp_li

        self.conv_in = nn.Conv2d(
            in_channels,
            block_out_channels[0],
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.temp_conv_in = nn.Conv3d(
            in_channels=block_out_channels[0],
            out_channels=block_out_channels[0],
            kernel_size=(3, 1, 1),
            padding=(1, 0, 0),
        )

        self.down_blocks = nn.ModuleList([])

        # down
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            if down_block_type == "AllegroDownBlock3D":
                down_block = AllegroDownBlock3D(
                    num_layers=self.layers_per_block,
                    in_channels=input_channel,
                    out_channels=output_channel,
                    add_downsample=not is_final_block,
                    add_temp_downsample=blocks_temp_li[i],
                    resnet_eps=1e-6,
                    downsample_padding=0,
                    resnet_act_fn=act_fn,
                    resnet_groups=norm_num_groups,
                )
            else:
                raise ValueError("Invalid `down_block_type` encountered. Must be `AllegroDownBlock3D`")

            self.down_blocks.append(down_block)

        # mid
        self.mid_block = UNetMidBlock3DConv(
            in_channels=block_out_channels[-1],
            resnet_eps=1e-6,
            resnet_act_fn=act_fn,
            output_scale_factor=1,
            resnet_time_scale_shift="default",
            attention_head_dim=block_out_channels[-1],
            resnet_groups=norm_num_groups,
            temb_channels=None,
        )

        # out
        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[-1], num_groups=norm_num_groups, eps=1e-6)
        self.conv_act = nn.SiLU()

        conv_out_channels = 2 * out_channels if double_z else out_channels

        self.temp_conv_out = nn.Conv3d(block_out_channels[-1], block_out_channels[-1], (3, 1, 1), padding=(1, 0, 0))

        self.conv_out = nn.Conv2d(block_out_channels[-1], conv_out_channels, 3, padding=1)

        self.gradient_checkpointing = False

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        batch_size = sample.shape[0]

        sample = sample.permute(0, 2, 1, 3, 4).flatten(0, 1)
        sample = self.conv_in(sample)

        sample = sample.unflatten(0, (batch_size, -1)).permute(0, 2, 1, 3, 4)

        residual = sample
        sample = self.temp_conv_in(sample)
        sample = sample + residual

        # Down blocks
        for down_block in self.down_blocks:
            sample = down_block(sample)

        # Mid block
        sample = self.mid_block(sample)

        # Post process
        sample = sample.permute(0, 2, 1, 3, 4).flatten(0, 1)
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = sample.unflatten(0, (batch_size, -1)).permute(0, 2, 1, 3, 4)

        residual = sample
        sample = self.temp_conv_out(sample)
        sample = sample + residual
        sample = sample.permute(0, 2, 1, 3, 4).flatten(0, 1)

        sample = self.conv_out(sample)
        sample = sample.unflatten(0, (batch_size, -1)).permute(0, 2, 1, 3, 4)

        return sample


class AllegroDecoder3D(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 3,
        up_block_types: Tuple[str, ...] = (
            "AllegroUpBlock3D",
            "AllegroUpBlock3D",
            "AllegroUpBlock3D",
            "AllegroUpBlock3D",
        ),
        blocks_temp_li=[False, False, False, False],
        block_out_channels: Tuple[int, ...] = (128, 256, 512, 512),
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        act_fn: str = "silu",
        norm_type: str = "group",  # group, spatial
    ):
        super().__init__()

        self.layers_per_block = layers_per_block
        self.blocks_temp_li = blocks_temp_li

        self.conv_in = nn.Conv2d(
            in_channels,
            block_out_channels[-1],
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.temp_conv_in = nn.Conv3d(block_out_channels[-1], block_out_channels[-1], (3, 1, 1), padding=(1, 0, 0))

        self.mid_block = None
        self.up_blocks = nn.ModuleList([])

        temb_channels = in_channels if norm_type == "spatial" else None

        # mid
        self.mid_block = UNetMidBlock3DConv(
            in_channels=block_out_channels[-1],
            resnet_eps=1e-6,
            resnet_act_fn=act_fn,
            output_scale_factor=1,
            resnet_time_scale_shift="default" if norm_type == "group" else norm_type,
            attention_head_dim=block_out_channels[-1],
            resnet_groups=norm_num_groups,
            temb_channels=temb_channels,
        )

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]

            is_final_block = i == len(block_out_channels) - 1

            if up_block_type == "AllegroUpBlock3D":
                up_block = AllegroUpBlock3D(
                    num_layers=self.layers_per_block + 1,
                    in_channels=prev_output_channel,
                    out_channels=output_channel,
                    add_upsample=not is_final_block,
                    add_temp_upsample=blocks_temp_li[i],
                    resnet_eps=1e-6,
                    resnet_act_fn=act_fn,
                    resnet_groups=norm_num_groups,
                    temb_channels=temb_channels,
                    resnet_time_scale_shift=norm_type,
                )
            else:
                raise ValueError("Invalid `UP_block_type` encountered. Must be `AllegroUpBlock3D`")

            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # out
        if norm_type == "spatial":
            self.conv_norm_out = SpatialNorm(block_out_channels[0], temb_channels)
        else:
            self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=1e-6)

        self.conv_act = nn.SiLU()

        self.temp_conv_out = nn.Conv3d(block_out_channels[0], block_out_channels[0], (3, 1, 1), padding=(1, 0, 0))
        self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, 3, padding=1)

        # TODO(aryan): implement gradient checkpointing
        self.gradient_checkpointing = False

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        batch_size = sample.shape[0]

        sample = sample.permute(0, 2, 1, 3, 4).flatten(0, 1)
        sample = self.conv_in(sample)
        sample = sample.unflatten(0, (batch_size, -1)).permute(0, 2, 1, 3, 4)

        residual = sample
        sample = self.temp_conv_in(sample)
        sample = sample + residual

        upscale_dtype = next(iter(self.up_blocks.parameters())).dtype

        # Mid block
        sample = self.mid_block(sample)
        sample = sample.to(upscale_dtype)

        # Up blocks
        for up_block in self.up_blocks:
            sample = up_block(sample)

        # Post process
        sample = sample.permute(0, 2, 1, 3, 4).flatten(0, 1)
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = sample.unflatten(0, (batch_size, -1)).permute(0, 2, 1, 3, 4)

        residual = sample
        sample = self.temp_conv_out(sample)
        sample = sample + residual

        sample = sample.permute(0, 2, 1, 3, 4).flatten(0, 1)
        sample = self.conv_out(sample)
        sample = sample.unflatten(0, (batch_size, -1)).permute(0, 2, 1, 3, 4)

        return sample


class AutoencoderKLAllegro(ModelMixin, ConfigMixin):
    r"""
    A VAE model with KL loss for encoding videos into latents and decoding latent representations into videos. Used in
    [Allegro](https://github.com/rhymes-ai/Allegro).

    This model inherits from [`ModelMixin`]. Check the superclass documentation for it's generic methods implemented
    for all models (such as downloading or saving).

    Parameters:
        in_channels (int, defaults to `3`): Number of channels in the input image.
        out_channels (int, defaults to `3`): Number of channels in the output.
        down_block_types (`Tuple[str, ...]`, defaults to `("AllegroDownBlock3D", "AllegroDownBlock3D", "AllegroDownBlock3D", "AllegroDownBlock3D")`):
            Tuple of downsample block types.
        up_block_types (`Tuple[str]`, defaults to `("AllegroUpBlock3D", "AllegroUpBlock3D", "AllegroUpBlock3D", "AllegroUpBlock3D")`):
            Tuple of upsample block types.
        block_out_channels (`Tuple[int]`, defaults to `(128, 256, 512, 512)`):
            Tuple of block output channels.
        act_fn (`str`, defaults to `"silu"`):
            The activation function to use.
        sample_size (`int`, *optional*, defaults to `32`): Sample input size.
        scaling_factor (`float`, defaults to `0.13235`):
            The component-wise standard deviation of the trained latent space computed using the first batch of the
            training set. This is used to scale the latent space to have unit variance when training the diffusion
            model. The latents are scaled with the formula `z = z * scaling_factor` before being passed to the
            diffusion model. When decoding, the latents are scaled back to the original scale with the formula: `z = 1
            / scaling_factor * z`. For more details, refer to sections 4.3.2 and D.1 of the [High-Resolution Image
            Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) paper.
        force_upcast (`bool`, default to `True`):
            If enabled it will force the VAE to run in float32 for high image resolution pipelines, such as SD-XL. VAE
            can be fine-tuned / trained to a lower range without loosing too much precision in which case
            `force_upcast` can be set to `False` - see: https://huggingface.co/madebyollin/sdxl-vae-fp16-fix
        TODO(aryan): docs
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: Tuple[str, ...] = (
            "AllegroDownBlock3D",
            "AllegroDownBlock3D",
            "AllegroDownBlock3D",
            "AllegroDownBlock3D",
        ),
        up_block_types: Tuple[str, ...] = (
            "AllegroUpBlock3D",
            "AllegroUpBlock3D",
            "AllegroUpBlock3D",
            "AllegroUpBlock3D",
        ),
        block_out_channels: Tuple[int, ...] = (128, 256, 512, 512),
        latent_channels: int = 4,
        layers_per_block: int = 2,
        act_fn: str = "silu",
        norm_num_groups: int = 32,
        temporal_compression_ratio: float = 4,
        sample_size: int = 320,
        scaling_factor: float = 0.13235,
        force_upcast: bool = True,
        tile_overlap: tuple = (120, 80),
        chunk_len: int = 24,
        t_over: int = 8,
        blocks_tempdown_li=[True, True, False, False],
        blocks_tempup_li=[False, True, True, False],
    ) -> None:
        super().__init__()

        self.blocks_tempdown_li = blocks_tempdown_li
        self.blocks_tempup_li = blocks_tempup_li

        self.encoder = AllegroEncoder3D(
            in_channels=in_channels,
            out_channels=latent_channels,
            down_block_types=down_block_types,
            blocks_temp_li=blocks_tempdown_li,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
            double_z=True,
        )
        self.decoder = AllegroDecoder3D(
            in_channels=latent_channels,
            out_channels=out_channels,
            up_block_types=up_block_types,
            blocks_temp_li=blocks_tempup_li,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            norm_num_groups=norm_num_groups,
            act_fn=act_fn,
        )
        self.quant_conv = nn.Conv2d(2 * latent_channels, 2 * latent_channels, 1)
        self.post_quant_conv = nn.Conv2d(latent_channels, latent_channels, 1)

        self.use_slicing = False
        self.use_tiling = False

        # only relevant if vae tiling is enabled
        sample_size = sample_size[0] if isinstance(sample_size, (list, tuple)) else sample_size
        self.tile_overlap = tile_overlap
        self.vae_scale_factor = [4, 8, 8]
        self.sample_size = sample_size
        self.chunk_len = chunk_len
        self.t_over = t_over

        self.latent_chunk_len = self.chunk_len // 4
        self.latent_t_over = self.t_over // 4
        self.kernel = (self.chunk_len, self.sample_size, self.sample_size)  # (24, 256, 256)
        self.stride = (
            self.chunk_len - self.t_over,
            self.sample_size - self.tile_overlap[0],
            self.sample_size - self.tile_overlap[1],
        )  # (16, 112, 192)

    def encode(
        self, input_imgs: torch.Tensor, return_dict: bool = True, local_batch_size=1
    ) -> Union[AutoencoderKLOutput, Tuple[DiagonalGaussianDistribution]]:
        # TODO(aryan): rewrite to encode and tiled_encode
        KERNEL = self.kernel
        STRIDE = self.stride
        LOCAL_BS = local_batch_size
        OUT_C = 8

        B, C, N, H, W = input_imgs.shape

        out_n = math.floor((N - KERNEL[0]) / STRIDE[0]) + 1
        out_h = math.floor((H - KERNEL[1]) / STRIDE[1]) + 1
        out_w = math.floor((W - KERNEL[2]) / STRIDE[2]) + 1

        ## cut video into overlapped small cubes and batch forward
        num = 0

        out_latent = torch.zeros(
            (out_n * out_h * out_w, OUT_C, KERNEL[0] // 4, KERNEL[1] // 8, KERNEL[2] // 8),
            device=input_imgs.device,
            dtype=input_imgs.dtype,
        )
        vae_batch_input = torch.zeros(
            (LOCAL_BS, C, KERNEL[0], KERNEL[1], KERNEL[2]), device=input_imgs.device, dtype=input_imgs.dtype
        )

        for i in range(out_n):
            for j in range(out_h):
                for k in range(out_w):
                    n_start, n_end = i * STRIDE[0], i * STRIDE[0] + KERNEL[0]
                    h_start, h_end = j * STRIDE[1], j * STRIDE[1] + KERNEL[1]
                    w_start, w_end = k * STRIDE[2], k * STRIDE[2] + KERNEL[2]
                    video_cube = input_imgs[:, :, n_start:n_end, h_start:h_end, w_start:w_end]
                    vae_batch_input[num % LOCAL_BS] = video_cube

                    if num % LOCAL_BS == LOCAL_BS - 1 or num == out_n * out_h * out_w - 1:
                        latent = self.encoder(vae_batch_input)

                        if num == out_n * out_h * out_w - 1 and num % LOCAL_BS != LOCAL_BS - 1:
                            out_latent[num - num % LOCAL_BS :] = latent[: num % LOCAL_BS + 1]
                        else:
                            out_latent[num - LOCAL_BS + 1 : num + 1] = latent
                        vae_batch_input = torch.zeros(
                            (LOCAL_BS, C, KERNEL[0], KERNEL[1], KERNEL[2]),
                            device=input_imgs.device,
                            dtype=input_imgs.dtype,
                        )
                    num += 1

        ## flatten the batched out latent to videos and supress the overlapped parts
        B, C, N, H, W = input_imgs.shape

        out_video_cube = torch.zeros(
            (B, OUT_C, N // 4, H // 8, W // 8), device=input_imgs.device, dtype=input_imgs.dtype
        )
        OUT_KERNEL = KERNEL[0] // 4, KERNEL[1] // 8, KERNEL[2] // 8
        OUT_STRIDE = STRIDE[0] // 4, STRIDE[1] // 8, STRIDE[2] // 8
        OVERLAP = OUT_KERNEL[0] - OUT_STRIDE[0], OUT_KERNEL[1] - OUT_STRIDE[1], OUT_KERNEL[2] - OUT_STRIDE[2]

        for i in range(out_n):
            n_start, n_end = i * OUT_STRIDE[0], i * OUT_STRIDE[0] + OUT_KERNEL[0]
            for j in range(out_h):
                h_start, h_end = j * OUT_STRIDE[1], j * OUT_STRIDE[1] + OUT_KERNEL[1]
                for k in range(out_w):
                    w_start, w_end = k * OUT_STRIDE[2], k * OUT_STRIDE[2] + OUT_KERNEL[2]
                    latent_mean_blend = prepare_for_blend(
                        (i, out_n, OVERLAP[0]),
                        (j, out_h, OVERLAP[1]),
                        (k, out_w, OVERLAP[2]),
                        out_latent[i * out_h * out_w + j * out_w + k].unsqueeze(0),
                    )
                    out_video_cube[:, :, n_start:n_end, h_start:h_end, w_start:w_end] += latent_mean_blend

        # final conv
        out_video_cube = out_video_cube.permute(0, 2, 1, 3, 4).flatten(0, 1)
        out_video_cube = self.quant_conv(out_video_cube)
        out_video_cube = out_video_cube.unflatten(0, (B, -1)).permute(0, 2, 1, 3, 4)

        posterior = DiagonalGaussianDistribution(out_video_cube)

        if not return_dict:
            return (posterior,)

        return AutoencoderKLOutput(latent_dist=posterior)

    def decode(
        self, input_latents: torch.Tensor, return_dict: bool = True, local_batch_size=1
    ) -> Union[DecoderOutput, torch.Tensor]:
        # TODO(aryan): rewrite to decode and tiled_decode
        KERNEL = self.kernel
        STRIDE = self.stride

        LOCAL_BS = local_batch_size
        OUT_C = 3
        IN_KERNEL = KERNEL[0] // 4, KERNEL[1] // 8, KERNEL[2] // 8
        IN_STRIDE = STRIDE[0] // 4, STRIDE[1] // 8, STRIDE[2] // 8

        B, C, N, H, W = input_latents.shape

        ## post quant conv (a mapping)
        input_latents = input_latents.permute(0, 2, 1, 3, 4).flatten(0, 1)
        input_latents = self.post_quant_conv(input_latents)
        input_latents = input_latents.unflatten(0, (B, -1)).permute(0, 2, 1, 3, 4)

        ## out tensor shape
        out_n = math.floor((N - IN_KERNEL[0]) / IN_STRIDE[0]) + 1
        out_h = math.floor((H - IN_KERNEL[1]) / IN_STRIDE[1]) + 1
        out_w = math.floor((W - IN_KERNEL[2]) / IN_STRIDE[2]) + 1

        ## cut latent into overlapped small cubes and batch forward
        num = 0
        decoded_cube = torch.zeros(
            (out_n * out_h * out_w, OUT_C, KERNEL[0], KERNEL[1], KERNEL[2]),
            device=input_latents.device,
            dtype=input_latents.dtype,
        )
        vae_batch_input = torch.zeros(
            (LOCAL_BS, C, IN_KERNEL[0], IN_KERNEL[1], IN_KERNEL[2]),
            device=input_latents.device,
            dtype=input_latents.dtype,
        )
        for i in range(out_n):
            for j in range(out_h):
                for k in range(out_w):
                    n_start, n_end = i * IN_STRIDE[0], i * IN_STRIDE[0] + IN_KERNEL[0]
                    h_start, h_end = j * IN_STRIDE[1], j * IN_STRIDE[1] + IN_KERNEL[1]
                    w_start, w_end = k * IN_STRIDE[2], k * IN_STRIDE[2] + IN_KERNEL[2]
                    latent_cube = input_latents[:, :, n_start:n_end, h_start:h_end, w_start:w_end]
                    vae_batch_input[num % LOCAL_BS] = latent_cube
                    if num % LOCAL_BS == LOCAL_BS - 1 or num == out_n * out_h * out_w - 1:
                        latent = self.decoder(vae_batch_input)

                        if num == out_n * out_h * out_w - 1 and num % LOCAL_BS != LOCAL_BS - 1:
                            decoded_cube[num - num % LOCAL_BS :] = latent[: num % LOCAL_BS + 1]
                        else:
                            decoded_cube[num - LOCAL_BS + 1 : num + 1] = latent
                        vae_batch_input = torch.zeros(
                            (LOCAL_BS, C, IN_KERNEL[0], IN_KERNEL[1], IN_KERNEL[2]),
                            device=input_latents.device,
                            dtype=input_latents.dtype,
                        )
                    num += 1
        B, C, N, H, W = input_latents.shape

        out_video = torch.zeros(
            (B, OUT_C, N * 4, H * 8, W * 8), device=input_latents.device, dtype=input_latents.dtype
        )
        OVERLAP = KERNEL[0] - STRIDE[0], KERNEL[1] - STRIDE[1], KERNEL[2] - STRIDE[2]
        for i in range(out_n):
            n_start, n_end = i * STRIDE[0], i * STRIDE[0] + KERNEL[0]
            for j in range(out_h):
                h_start, h_end = j * STRIDE[1], j * STRIDE[1] + KERNEL[1]
                for k in range(out_w):
                    w_start, w_end = k * STRIDE[2], k * STRIDE[2] + KERNEL[2]
                    out_video_blend = prepare_for_blend(
                        (i, out_n, OVERLAP[0]),
                        (j, out_h, OVERLAP[1]),
                        (k, out_w, OVERLAP[2]),
                        decoded_cube[i * out_h * out_w + j * out_w + k].unsqueeze(0),
                    )
                    out_video[:, :, n_start:n_end, h_start:h_end, w_start:w_end] += out_video_blend

        out_video = out_video.permute(0, 2, 1, 3, 4).contiguous()

        decoded = out_video
        if not return_dict:
            return (decoded,)

        return DecoderOutput(sample=decoded)

    def forward(
        self,
        sample: torch.Tensor,
        sample_posterior: bool = False,
        return_dict: bool = True,
        generator: Optional[torch.Generator] = None,
        encoder_local_batch_size: int = 2,
        decoder_local_batch_size: int = 2,
    ) -> Union[DecoderOutput, torch.Tensor]:
        r"""
        Args:
            sample (`torch.Tensor`): Input sample.
            sample_posterior (`bool`, *optional*, defaults to `False`):
                Whether to sample from the posterior.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`DecoderOutput`] instead of a plain tuple.
            generator (`torch.Generator`, *optional*):
                PyTorch random number generator.
            encoder_local_batch_size (`int`, *optional*, defaults to 2):
                Local batch size for the encoder's batch inference.
            decoder_local_batch_size (`int`, *optional*, defaults to 2):
                Local batch size for the decoder's batch inference.
        """
        x = sample
        posterior = self.encode(x, local_batch_size=encoder_local_batch_size).latent_dist
        if sample_posterior:
            z = posterior.sample(generator=generator)
        else:
            z = posterior.mode()
        dec = self.decode(z, local_batch_size=decoder_local_batch_size).sample

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)


def prepare_for_blend(n_param, h_param, w_param, x):
    n, n_max, overlap_n = n_param
    h, h_max, overlap_h = h_param
    w, w_max, overlap_w = w_param
    if overlap_n > 0:
        if n > 0:  # the head overlap part decays from 0 to 1
            x[:, :, 0:overlap_n, :, :] = x[:, :, 0:overlap_n, :, :] * (
                torch.arange(0, overlap_n).float().to(x.device) / overlap_n
            ).reshape(overlap_n, 1, 1)
        if n < n_max - 1:  # the tail overlap part decays from 1 to 0
            x[:, :, -overlap_n:, :, :] = x[:, :, -overlap_n:, :, :] * (
                1 - torch.arange(0, overlap_n).float().to(x.device) / overlap_n
            ).reshape(overlap_n, 1, 1)
    if h > 0:
        x[:, :, :, 0:overlap_h, :] = x[:, :, :, 0:overlap_h, :] * (
            torch.arange(0, overlap_h).float().to(x.device) / overlap_h
        ).reshape(overlap_h, 1)
    if h < h_max - 1:
        x[:, :, :, -overlap_h:, :] = x[:, :, :, -overlap_h:, :] * (
            1 - torch.arange(0, overlap_h).float().to(x.device) / overlap_h
        ).reshape(overlap_h, 1)
    if w > 0:
        x[:, :, :, :, 0:overlap_w] = x[:, :, :, :, 0:overlap_w] * (
            torch.arange(0, overlap_w).float().to(x.device) / overlap_w
        )
    if w < w_max - 1:
        x[:, :, :, :, -overlap_w:] = x[:, :, :, :, -overlap_w:] * (
            1 - torch.arange(0, overlap_w).float().to(x.device) / overlap_w
        )
    return x
