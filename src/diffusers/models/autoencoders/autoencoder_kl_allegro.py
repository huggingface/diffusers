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
from ...utils.accelerate_utils import apply_forward_hook
from ..attention_processor import Attention, SpatialNorm
from ..autoencoders.vae import DecoderOutput, DiagonalGaussianDistribution
from ..downsampling import Downsample2D
from ..modeling_outputs import AutoencoderKLOutput
from ..modeling_utils import ModelMixin
from ..resnet import ResnetBlock2D
from ..upsampling import Upsample2D


class AllegroTemporalConvLayer(nn.Module):
    r"""
    Temporal convolutional layer that can be used for video (sequence of images) input. Code adapted from:
    https://github.com/modelscope/modelscope/blob/1509fdb973e5871f37148a4b5e5964cafd43e64d/modelscope/models/multi_modal/video_synthesis/unet_sd.py#L1016
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: Optional[int] = None,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
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
                nn.GroupNorm(norm_num_groups, in_dim),
                nn.SiLU(),
                nn.Conv3d(in_dim, out_dim, (2, stride, stride), stride=(2, 1, 1), padding=(0, pad_h, pad_w)),
            )
        elif up_sample:
            self.conv1 = nn.Sequential(
                nn.GroupNorm(norm_num_groups, in_dim),
                nn.SiLU(),
                nn.Conv3d(in_dim, out_dim * 2, (1, stride, stride), padding=(0, pad_h, pad_w)),
            )
        else:
            self.conv1 = nn.Sequential(
                nn.GroupNorm(norm_num_groups, in_dim),
                nn.SiLU(),
                nn.Conv3d(in_dim, out_dim, (3, stride, stride), padding=(pad_t, pad_h, pad_w)),
            )
        self.conv2 = nn.Sequential(
            nn.GroupNorm(norm_num_groups, out_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv3d(out_dim, in_dim, (3, stride, stride), padding=(pad_t, pad_h, pad_w)),
        )
        self.conv3 = nn.Sequential(
            nn.GroupNorm(norm_num_groups, out_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv3d(out_dim, in_dim, (3, stride, stride), padding=(pad_t, pad_h, pad_h)),
        )
        self.conv4 = nn.Sequential(
            nn.GroupNorm(norm_num_groups, out_dim),
            nn.SiLU(),
            nn.Conv3d(out_dim, in_dim, (3, stride, stride), padding=(pad_t, pad_h, pad_h)),
        )

    @staticmethod
    def _pad_temporal_dim(hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = torch.cat((hidden_states[:, :, 0:1], hidden_states), dim=2)
        hidden_states = torch.cat((hidden_states, hidden_states[:, :, -1:]), dim=2)
        return hidden_states

    def forward(self, hidden_states: torch.Tensor, batch_size: int) -> torch.Tensor:
        hidden_states = hidden_states.unflatten(0, (batch_size, -1)).permute(0, 2, 1, 3, 4)

        if self.down_sample:
            identity = hidden_states[:, :, ::2]
        elif self.up_sample:
            identity = hidden_states.repeat_interleave(2, dim=2)
        else:
            identity = hidden_states

        if self.down_sample or self.up_sample:
            hidden_states = self.conv1(hidden_states)
        else:
            hidden_states = self._pad_temporal_dim(hidden_states)
            hidden_states = self.conv1(hidden_states)

        if self.up_sample:
            hidden_states = hidden_states.unflatten(1, (2, -1)).permute(0, 2, 3, 1, 4, 5).flatten(2, 3)

        hidden_states = self._pad_temporal_dim(hidden_states)
        hidden_states = self.conv2(hidden_states)

        hidden_states = self._pad_temporal_dim(hidden_states)
        hidden_states = self.conv3(hidden_states)

        hidden_states = self._pad_temporal_dim(hidden_states)
        hidden_states = self.conv4(hidden_states)

        hidden_states = identity + hidden_states
        hidden_states = hidden_states.permute(0, 2, 1, 3, 4).flatten(0, 1)

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
        output_scale_factor: float = 1.0,
        spatial_downsample: bool = True,
        temporal_downsample: bool = False,
        downsample_padding: int = 1,
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
                AllegroTemporalConvLayer(
                    out_channels,
                    out_channels,
                    dropout=0.1,
                    norm_num_groups=resnet_groups,
                )
            )

        self.resnets = nn.ModuleList(resnets)
        self.temp_convs = nn.ModuleList(temp_convs)

        if temporal_downsample:
            self.temp_convs_down = AllegroTemporalConvLayer(
                out_channels, out_channels, dropout=0.1, norm_num_groups=resnet_groups, down_sample=True, stride=3
            )
        self.add_temp_downsample = temporal_downsample

        if spatial_downsample:
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

        hidden_states = hidden_states.permute(0, 2, 1, 3, 4).flatten(0, 1)

        for resnet, temp_conv in zip(self.resnets, self.temp_convs):
            hidden_states = resnet(hidden_states, temb=None)
            hidden_states = temp_conv(hidden_states, batch_size=batch_size)

        if self.add_temp_downsample:
            hidden_states = self.temp_convs_down(hidden_states, batch_size=batch_size)

        if self.downsamplers is not None:
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
        output_scale_factor: float = 1.0,
        spatial_upsample: bool = True,
        temporal_upsample: bool = False,
        temb_channels: Optional[int] = None,
    ):
        super().__init__()

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
                AllegroTemporalConvLayer(
                    out_channels,
                    out_channels,
                    dropout=0.1,
                    norm_num_groups=resnet_groups,
                )
            )

        self.resnets = nn.ModuleList(resnets)
        self.temp_convs = nn.ModuleList(temp_convs)

        self.add_temp_upsample = temporal_upsample
        if temporal_upsample:
            self.temp_conv_up = AllegroTemporalConvLayer(
                out_channels, out_channels, dropout=0.1, norm_num_groups=resnet_groups, up_sample=True, stride=3
            )

        if spatial_upsample:
            self.upsamplers = nn.ModuleList([Upsample2D(out_channels, use_conv=True, out_channels=out_channels)])
        else:
            self.upsamplers = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size = hidden_states.shape[0]

        hidden_states = hidden_states.permute(0, 2, 1, 3, 4).flatten(0, 1)

        for resnet, temp_conv in zip(self.resnets, self.temp_convs):
            hidden_states = resnet(hidden_states, temb=None)
            hidden_states = temp_conv(hidden_states, batch_size=batch_size)

        if self.add_temp_upsample:
            hidden_states = self.temp_conv_up(hidden_states, batch_size=batch_size)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)

        hidden_states = hidden_states.unflatten(0, (batch_size, -1)).permute(0, 2, 1, 3, 4)
        return hidden_states


class AllegroMidBlock3DConv(nn.Module):
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
        attention_head_dim: int = 1,
        output_scale_factor: float = 1.0,
    ):
        super().__init__()

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
            AllegroTemporalConvLayer(
                in_channels,
                in_channels,
                dropout=0.1,
                norm_num_groups=resnet_groups,
            )
        ]
        attentions = []

        if attention_head_dim is None:
            attention_head_dim = in_channels

        for _ in range(num_layers):
            if add_attention:
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
                AllegroTemporalConvLayer(
                    in_channels,
                    in_channels,
                    dropout=0.1,
                    norm_num_groups=resnet_groups,
                )
            )

        self.resnets = nn.ModuleList(resnets)
        self.temp_convs = nn.ModuleList(temp_convs)
        self.attentions = nn.ModuleList(attentions)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size = hidden_states.shape[0]

        hidden_states = hidden_states.permute(0, 2, 1, 3, 4).flatten(0, 1)
        hidden_states = self.resnets[0](hidden_states, temb=None)

        hidden_states = self.temp_convs[0](hidden_states, batch_size=batch_size)

        for attn, resnet, temp_conv in zip(self.attentions, self.resnets[1:], self.temp_convs[1:]):
            hidden_states = attn(hidden_states)
            hidden_states = resnet(hidden_states, temb=None)
            hidden_states = temp_conv(hidden_states, batch_size=batch_size)

        hidden_states = hidden_states.unflatten(0, (batch_size, -1)).permute(0, 2, 1, 3, 4)
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
        block_out_channels: Tuple[int, ...] = (128, 256, 512, 512),
        temporal_downsample_blocks: Tuple[bool, ...] = [True, True, False, False],
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        act_fn: str = "silu",
        double_z: bool = True,
    ):
        super().__init__()

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
                    num_layers=layers_per_block,
                    in_channels=input_channel,
                    out_channels=output_channel,
                    spatial_downsample=not is_final_block,
                    temporal_downsample=temporal_downsample_blocks[i],
                    resnet_eps=1e-6,
                    downsample_padding=0,
                    resnet_act_fn=act_fn,
                    resnet_groups=norm_num_groups,
                )
            else:
                raise ValueError("Invalid `down_block_type` encountered. Must be `AllegroDownBlock3D`")

            self.down_blocks.append(down_block)

        # mid
        self.mid_block = AllegroMidBlock3DConv(
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

        if torch.is_grad_enabled() and self.gradient_checkpointing:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward

            # Down blocks
            for down_block in self.down_blocks:
                sample = torch.utils.checkpoint.checkpoint(create_custom_forward(down_block), sample)

            # Mid block
            sample = torch.utils.checkpoint.checkpoint(create_custom_forward(self.mid_block), sample)
        else:
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
        temporal_upsample_blocks: Tuple[bool, ...] = [False, True, True, False],
        block_out_channels: Tuple[int, ...] = (128, 256, 512, 512),
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        act_fn: str = "silu",
        norm_type: str = "group",  # group, spatial
    ):
        super().__init__()

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
        self.mid_block = AllegroMidBlock3DConv(
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
                    num_layers=layers_per_block + 1,
                    in_channels=prev_output_channel,
                    out_channels=output_channel,
                    spatial_upsample=not is_final_block,
                    temporal_upsample=temporal_upsample_blocks[i],
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

        if torch.is_grad_enabled() and self.gradient_checkpointing:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward

            # Mid block
            sample = torch.utils.checkpoint.checkpoint(create_custom_forward(self.mid_block), sample)

            # Up blocks
            for up_block in self.up_blocks:
                sample = torch.utils.checkpoint.checkpoint(create_custom_forward(up_block), sample)

        else:
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
        in_channels (int, defaults to `3`):
            Number of channels in the input image.
        out_channels (int, defaults to `3`):
            Number of channels in the output.
        down_block_types (`Tuple[str, ...]`, defaults to `("AllegroDownBlock3D", "AllegroDownBlock3D", "AllegroDownBlock3D", "AllegroDownBlock3D")`):
            Tuple of strings denoting which types of down blocks to use.
        up_block_types (`Tuple[str, ...]`, defaults to `("AllegroUpBlock3D", "AllegroUpBlock3D", "AllegroUpBlock3D", "AllegroUpBlock3D")`):
            Tuple of strings denoting which types of up blocks to use.
        block_out_channels (`Tuple[int, ...]`, defaults to `(128, 256, 512, 512)`):
            Tuple of integers denoting number of output channels in each block.
        temporal_downsample_blocks (`Tuple[bool, ...]`, defaults to `(True, True, False, False)`):
            Tuple of booleans denoting which blocks to enable temporal downsampling in.
        latent_channels (`int`, defaults to `4`):
            Number of channels in latents.
        layers_per_block (`int`, defaults to `2`):
            Number of resnet or attention or temporal convolution layers per down/up block.
        act_fn (`str`, defaults to `"silu"`):
            The activation function to use.
        norm_num_groups (`int`, defaults to `32`):
            Number of groups to use in normalization layers.
        temporal_compression_ratio (`int`, defaults to `4`):
            Ratio by which temporal dimension of samples are compressed.
        sample_size (`int`, defaults to `320`):
            Default latent size.
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
        temporal_downsample_blocks: Tuple[bool, ...] = (True, True, False, False),
        temporal_upsample_blocks: Tuple[bool, ...] = (False, True, True, False),
        latent_channels: int = 4,
        layers_per_block: int = 2,
        act_fn: str = "silu",
        norm_num_groups: int = 32,
        temporal_compression_ratio: float = 4,
        sample_size: int = 320,
        scaling_factor: float = 0.13,
        force_upcast: bool = True,
    ) -> None:
        super().__init__()

        self.encoder = AllegroEncoder3D(
            in_channels=in_channels,
            out_channels=latent_channels,
            down_block_types=down_block_types,
            temporal_downsample_blocks=temporal_downsample_blocks,
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
            temporal_upsample_blocks=temporal_upsample_blocks,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            norm_num_groups=norm_num_groups,
            act_fn=act_fn,
        )
        self.quant_conv = nn.Conv2d(2 * latent_channels, 2 * latent_channels, 1)
        self.post_quant_conv = nn.Conv2d(latent_channels, latent_channels, 1)

        # TODO(aryan): For the 1.0.0 refactor, `temporal_compression_ratio` can be inferred directly and we don't need
        # to use a specific parameter here or in other VAEs.

        self.use_slicing = False
        self.use_tiling = False

        self.spatial_compression_ratio = 2 ** (len(block_out_channels) - 1)
        self.tile_overlap_t = 8
        self.tile_overlap_h = 120
        self.tile_overlap_w = 80
        sample_frames = 24

        self.kernel = (sample_frames, sample_size, sample_size)
        self.stride = (
            sample_frames - self.tile_overlap_t,
            sample_size - self.tile_overlap_h,
            sample_size - self.tile_overlap_w,
        )

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (AllegroEncoder3D, AllegroDecoder3D)):
            module.gradient_checkpointing = value

    def enable_tiling(self) -> None:
        r"""
        Enable tiled VAE decoding. When this option is enabled, the VAE will split the input tensor into tiles to
        compute decoding and encoding in several steps. This is useful for saving a large amount of memory and to allow
        processing larger images.
        """
        self.use_tiling = True

    def disable_tiling(self) -> None:
        r"""
        Disable tiled VAE decoding. If `enable_tiling` was previously enabled, this method will go back to computing
        decoding in one step.
        """
        self.use_tiling = False

    def enable_slicing(self) -> None:
        r"""
        Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
        compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.use_slicing = True

    def disable_slicing(self) -> None:
        r"""
        Disable sliced VAE decoding. If `enable_slicing` was previously enabled, this method will go back to computing
        decoding in one step.
        """
        self.use_slicing = False

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        # TODO(aryan)
        # if self.use_tiling and (width > self.tile_sample_min_width or height > self.tile_sample_min_height):
        if self.use_tiling:
            return self.tiled_encode(x)

        raise NotImplementedError("Encoding without tiling has not been implemented yet.")

    @apply_forward_hook
    def encode(
        self, x: torch.Tensor, return_dict: bool = True
    ) -> Union[AutoencoderKLOutput, Tuple[DiagonalGaussianDistribution]]:
        r"""
        Encode a batch of videos into latents.

        Args:
            x (`torch.Tensor`):
                Input batch of videos.
            return_dict (`bool`, defaults to `True`):
                Whether to return a [`~models.autoencoder_kl.AutoencoderKLOutput`] instead of a plain tuple.

        Returns:
                The latent representations of the encoded videos. If `return_dict` is True, a
                [`~models.autoencoder_kl.AutoencoderKLOutput`] is returned, otherwise a plain `tuple` is returned.
        """
        if self.use_slicing and x.shape[0] > 1:
            encoded_slices = [self._encode(x_slice) for x_slice in x.split(1)]
            h = torch.cat(encoded_slices)
        else:
            h = self._encode(x)

        posterior = DiagonalGaussianDistribution(h)

        if not return_dict:
            return (posterior,)
        return AutoencoderKLOutput(latent_dist=posterior)

    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        # TODO(aryan): refactor tiling implementation
        # if self.use_tiling and (width > self.tile_latent_min_width or height > self.tile_latent_min_height):
        if self.use_tiling:
            return self.tiled_decode(z)

        raise NotImplementedError("Decoding without tiling has not been implemented yet.")

    @apply_forward_hook
    def decode(self, z: torch.Tensor, return_dict: bool = True) -> Union[DecoderOutput, torch.Tensor]:
        """
        Decode a batch of videos.

        Args:
            z (`torch.Tensor`):
                Input batch of latent vectors.
            return_dict (`bool`, defaults to `True`):
                Whether to return a [`~models.vae.DecoderOutput`] instead of a plain tuple.

        Returns:
            [`~models.vae.DecoderOutput`] or `tuple`:
                If return_dict is True, a [`~models.vae.DecoderOutput`] is returned, otherwise a plain `tuple` is
                returned.
        """
        if self.use_slicing and z.shape[0] > 1:
            decoded_slices = [self._decode(z_slice) for z_slice in z.split(1)]
            decoded = torch.cat(decoded_slices)
        else:
            decoded = self._decode(z)

        if not return_dict:
            return (decoded,)
        return DecoderOutput(sample=decoded)

    def tiled_encode(self, x: torch.Tensor) -> torch.Tensor:
        local_batch_size = 1
        rs = self.spatial_compression_ratio
        rt = self.config.temporal_compression_ratio

        batch_size, num_channels, num_frames, height, width = x.shape

        output_num_frames = math.floor((num_frames - self.kernel[0]) / self.stride[0]) + 1
        output_height = math.floor((height - self.kernel[1]) / self.stride[1]) + 1
        output_width = math.floor((width - self.kernel[2]) / self.stride[2]) + 1

        count = 0
        output_latent = x.new_zeros(
            (
                output_num_frames * output_height * output_width,
                2 * self.config.latent_channels,
                self.kernel[0] // rt,
                self.kernel[1] // rs,
                self.kernel[2] // rs,
            )
        )
        vae_batch_input = x.new_zeros((local_batch_size, num_channels, self.kernel[0], self.kernel[1], self.kernel[2]))

        for i in range(output_num_frames):
            for j in range(output_height):
                for k in range(output_width):
                    n_start, n_end = i * self.stride[0], i * self.stride[0] + self.kernel[0]
                    h_start, h_end = j * self.stride[1], j * self.stride[1] + self.kernel[1]
                    w_start, w_end = k * self.stride[2], k * self.stride[2] + self.kernel[2]

                    video_cube = x[:, :, n_start:n_end, h_start:h_end, w_start:w_end]
                    vae_batch_input[count % local_batch_size] = video_cube

                    if (
                        count % local_batch_size == local_batch_size - 1
                        or count == output_num_frames * output_height * output_width - 1
                    ):
                        latent = self.encoder(vae_batch_input)

                        if (
                            count == output_num_frames * output_height * output_width - 1
                            and count % local_batch_size != local_batch_size - 1
                        ):
                            output_latent[count - count % local_batch_size :] = latent[: count % local_batch_size + 1]
                        else:
                            output_latent[count - local_batch_size + 1 : count + 1] = latent

                        vae_batch_input = x.new_zeros(
                            (local_batch_size, num_channels, self.kernel[0], self.kernel[1], self.kernel[2])
                        )

                    count += 1

        latent = x.new_zeros(
            (batch_size, 2 * self.config.latent_channels, num_frames // rt, height // rs, width // rs)
        )
        output_kernel = self.kernel[0] // rt, self.kernel[1] // rs, self.kernel[2] // rs
        output_stride = self.stride[0] // rt, self.stride[1] // rs, self.stride[2] // rs
        output_overlap = (
            output_kernel[0] - output_stride[0],
            output_kernel[1] - output_stride[1],
            output_kernel[2] - output_stride[2],
        )

        for i in range(output_num_frames):
            n_start, n_end = i * output_stride[0], i * output_stride[0] + output_kernel[0]
            for j in range(output_height):
                h_start, h_end = j * output_stride[1], j * output_stride[1] + output_kernel[1]
                for k in range(output_width):
                    w_start, w_end = k * output_stride[2], k * output_stride[2] + output_kernel[2]
                    latent_mean = _prepare_for_blend(
                        (i, output_num_frames, output_overlap[0]),
                        (j, output_height, output_overlap[1]),
                        (k, output_width, output_overlap[2]),
                        output_latent[i * output_height * output_width + j * output_width + k].unsqueeze(0),
                    )
                    latent[:, :, n_start:n_end, h_start:h_end, w_start:w_end] += latent_mean

        latent = latent.permute(0, 2, 1, 3, 4).flatten(0, 1)
        latent = self.quant_conv(latent)
        latent = latent.unflatten(0, (batch_size, -1)).permute(0, 2, 1, 3, 4)
        return latent

    def tiled_decode(self, z: torch.Tensor) -> torch.Tensor:
        local_batch_size = 1
        rs = self.spatial_compression_ratio
        rt = self.config.temporal_compression_ratio

        latent_kernel = self.kernel[0] // rt, self.kernel[1] // rs, self.kernel[2] // rs
        latent_stride = self.stride[0] // rt, self.stride[1] // rs, self.stride[2] // rs

        batch_size, num_channels, num_frames, height, width = z.shape

        ## post quant conv (a mapping)
        z = z.permute(0, 2, 1, 3, 4).flatten(0, 1)
        z = self.post_quant_conv(z)
        z = z.unflatten(0, (batch_size, -1)).permute(0, 2, 1, 3, 4)

        output_num_frames = math.floor((num_frames - latent_kernel[0]) / latent_stride[0]) + 1
        output_height = math.floor((height - latent_kernel[1]) / latent_stride[1]) + 1
        output_width = math.floor((width - latent_kernel[2]) / latent_stride[2]) + 1

        count = 0
        decoded_videos = z.new_zeros(
            (
                output_num_frames * output_height * output_width,
                self.config.out_channels,
                self.kernel[0],
                self.kernel[1],
                self.kernel[2],
            )
        )
        vae_batch_input = z.new_zeros(
            (local_batch_size, num_channels, latent_kernel[0], latent_kernel[1], latent_kernel[2])
        )

        for i in range(output_num_frames):
            for j in range(output_height):
                for k in range(output_width):
                    n_start, n_end = i * latent_stride[0], i * latent_stride[0] + latent_kernel[0]
                    h_start, h_end = j * latent_stride[1], j * latent_stride[1] + latent_kernel[1]
                    w_start, w_end = k * latent_stride[2], k * latent_stride[2] + latent_kernel[2]

                    current_latent = z[:, :, n_start:n_end, h_start:h_end, w_start:w_end]
                    vae_batch_input[count % local_batch_size] = current_latent

                    if (
                        count % local_batch_size == local_batch_size - 1
                        or count == output_num_frames * output_height * output_width - 1
                    ):
                        current_video = self.decoder(vae_batch_input)

                        if (
                            count == output_num_frames * output_height * output_width - 1
                            and count % local_batch_size != local_batch_size - 1
                        ):
                            decoded_videos[count - count % local_batch_size :] = current_video[
                                : count % local_batch_size + 1
                            ]
                        else:
                            decoded_videos[count - local_batch_size + 1 : count + 1] = current_video

                        vae_batch_input = z.new_zeros(
                            (local_batch_size, num_channels, latent_kernel[0], latent_kernel[1], latent_kernel[2])
                        )

                    count += 1

        video = z.new_zeros((batch_size, self.config.out_channels, num_frames * rt, height * rs, width * rs))
        video_overlap = (
            self.kernel[0] - self.stride[0],
            self.kernel[1] - self.stride[1],
            self.kernel[2] - self.stride[2],
        )

        for i in range(output_num_frames):
            n_start, n_end = i * self.stride[0], i * self.stride[0] + self.kernel[0]
            for j in range(output_height):
                h_start, h_end = j * self.stride[1], j * self.stride[1] + self.kernel[1]
                for k in range(output_width):
                    w_start, w_end = k * self.stride[2], k * self.stride[2] + self.kernel[2]
                    out_video_blend = _prepare_for_blend(
                        (i, output_num_frames, video_overlap[0]),
                        (j, output_height, video_overlap[1]),
                        (k, output_width, video_overlap[2]),
                        decoded_videos[i * output_height * output_width + j * output_width + k].unsqueeze(0),
                    )
                    video[:, :, n_start:n_end, h_start:h_end, w_start:w_end] += out_video_blend

        video = video.permute(0, 2, 1, 3, 4).contiguous()
        return video

    def forward(
        self,
        sample: torch.Tensor,
        sample_posterior: bool = False,
        return_dict: bool = True,
        generator: Optional[torch.Generator] = None,
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
        """
        x = sample
        posterior = self.encode(x).latent_dist
        if sample_posterior:
            z = posterior.sample(generator=generator)
        else:
            z = posterior.mode()
        dec = self.decode(z).sample

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)


def _prepare_for_blend(n_param, h_param, w_param, x):
    # TODO(aryan): refactor
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
