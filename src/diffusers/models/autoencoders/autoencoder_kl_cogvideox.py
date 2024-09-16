# Copyright 2024 The CogVideoX team, Tsinghua University & ZhipuAI and The HuggingFace Team.
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

from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...configuration_utils import ConfigMixin, register_to_config
from ...loaders.single_file_model import FromOriginalModelMixin
from ...utils import logging
from ...utils.accelerate_utils import apply_forward_hook
from ..activations import get_activation
from ..downsampling import CogVideoXDownsample3D
from ..modeling_outputs import AutoencoderKLOutput
from ..modeling_utils import ModelMixin
from ..upsampling import CogVideoXUpsample3D
from .vae import DecoderOutput, DiagonalGaussianDistribution


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class CogVideoXSafeConv3d(nn.Conv3d):
    r"""
    A 3D convolution layer that splits the input tensor into smaller parts to avoid OOM in CogVideoX Model.
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        memory_count = torch.prod(torch.tensor(input.shape)).item() * 2 / 1024**3

        # Set to 2GB, suitable for CuDNN
        if memory_count > 2:
            kernel_size = self.kernel_size[0]
            part_num = int(memory_count / 2) + 1
            input_chunks = torch.chunk(input, part_num, dim=2)

            if kernel_size > 1:
                input_chunks = [input_chunks[0]] + [
                    torch.cat((input_chunks[i - 1][:, :, -kernel_size + 1 :], input_chunks[i]), dim=2)
                    for i in range(1, len(input_chunks))
                ]

            output_chunks = []
            for input_chunk in input_chunks:
                output_chunks.append(super().forward(input_chunk))
            output = torch.cat(output_chunks, dim=2)
            return output
        else:
            return super().forward(input)


class CogVideoXCausalConv3d(nn.Module):
    r"""A 3D causal convolution layer that pads the input tensor to ensure causality in CogVideoX Model.

    Args:
        in_channels (`int`): Number of channels in the input tensor.
        out_channels (`int`): Number of output channels produced by the convolution.
        kernel_size (`int` or `Tuple[int, int, int]`): Kernel size of the convolutional kernel.
        stride (`int`, defaults to `1`): Stride of the convolution.
        dilation (`int`, defaults to `1`): Dilation rate of the convolution.
        pad_mode (`str`, defaults to `"constant"`): Padding mode.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: int = 1,
        dilation: int = 1,
        pad_mode: str = "constant",
    ):
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3

        time_kernel_size, height_kernel_size, width_kernel_size = kernel_size

        self.pad_mode = pad_mode
        time_pad = dilation * (time_kernel_size - 1) + (1 - stride)
        height_pad = height_kernel_size // 2
        width_pad = width_kernel_size // 2

        self.height_pad = height_pad
        self.width_pad = width_pad
        self.time_pad = time_pad
        self.time_causal_padding = (width_pad, width_pad, height_pad, height_pad, time_pad, 0)

        self.temporal_dim = 2
        self.time_kernel_size = time_kernel_size

        stride = (stride, 1, 1)
        dilation = (dilation, 1, 1)
        self.conv = CogVideoXSafeConv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
        )

        self.conv_cache = None

    def fake_context_parallel_forward(self, inputs: torch.Tensor) -> torch.Tensor:
        kernel_size = self.time_kernel_size
        if kernel_size > 1:
            cached_inputs = (
                [self.conv_cache] if self.conv_cache is not None else [inputs[:, :, :1]] * (kernel_size - 1)
            )
            inputs = torch.cat(cached_inputs + [inputs], dim=2)
        return inputs

    def _clear_fake_context_parallel_cache(self):
        del self.conv_cache
        self.conv_cache = None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = self.fake_context_parallel_forward(inputs)

        self._clear_fake_context_parallel_cache()
        # Note: we could move these to the cpu for a lower maximum memory usage but its only a few
        # hundred megabytes and so let's not do it for now
        self.conv_cache = inputs[:, :, -self.time_kernel_size + 1 :].clone()

        padding_2d = (self.width_pad, self.width_pad, self.height_pad, self.height_pad)
        inputs = F.pad(inputs, padding_2d, mode="constant", value=0)

        output = self.conv(inputs)
        return output


class CogVideoXSpatialNorm3D(nn.Module):
    r"""
    Spatially conditioned normalization as defined in https://arxiv.org/abs/2209.09002. This implementation is specific
    to 3D-video like data.

    CogVideoXSafeConv3d is used instead of nn.Conv3d to avoid OOM in CogVideoX Model.

    Args:
        f_channels (`int`):
            The number of channels for input to group normalization layer, and output of the spatial norm layer.
        zq_channels (`int`):
            The number of channels for the quantized vector as described in the paper.
        groups (`int`):
            Number of groups to separate the channels into for group normalization.
    """

    def __init__(
        self,
        f_channels: int,
        zq_channels: int,
        groups: int = 32,
    ):
        super().__init__()
        self.norm_layer = nn.GroupNorm(num_channels=f_channels, num_groups=groups, eps=1e-6, affine=True)
        self.conv_y = CogVideoXCausalConv3d(zq_channels, f_channels, kernel_size=1, stride=1)
        self.conv_b = CogVideoXCausalConv3d(zq_channels, f_channels, kernel_size=1, stride=1)

    def forward(self, f: torch.Tensor, zq: torch.Tensor) -> torch.Tensor:
        if f.shape[2] > 1 and f.shape[2] % 2 == 1:
            f_first, f_rest = f[:, :, :1], f[:, :, 1:]
            f_first_size, f_rest_size = f_first.shape[-3:], f_rest.shape[-3:]
            z_first, z_rest = zq[:, :, :1], zq[:, :, 1:]
            z_first = F.interpolate(z_first, size=f_first_size)
            z_rest = F.interpolate(z_rest, size=f_rest_size)
            zq = torch.cat([z_first, z_rest], dim=2)
        else:
            zq = F.interpolate(zq, size=f.shape[-3:])

        norm_f = self.norm_layer(f)
        new_f = norm_f * self.conv_y(zq) + self.conv_b(zq)
        return new_f


class CogVideoXResnetBlock3D(nn.Module):
    r"""
    A 3D ResNet block used in the CogVideoX model.

    Args:
        in_channels (`int`):
            Number of input channels.
        out_channels (`int`, *optional*):
            Number of output channels. If None, defaults to `in_channels`.
        dropout (`float`, defaults to `0.0`):
            Dropout rate.
        temb_channels (`int`, defaults to `512`):
            Number of time embedding channels.
        groups (`int`, defaults to `32`):
            Number of groups to separate the channels into for group normalization.
        eps (`float`, defaults to `1e-6`):
            Epsilon value for normalization layers.
        non_linearity (`str`, defaults to `"swish"`):
            Activation function to use.
        conv_shortcut (bool, defaults to `False`):
            Whether or not to use a convolution shortcut.
        spatial_norm_dim (`int`, *optional*):
            The dimension to use for spatial norm if it is to be used instead of group norm.
        pad_mode (str, defaults to `"first"`):
            Padding mode.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        dropout: float = 0.0,
        temb_channels: int = 512,
        groups: int = 32,
        eps: float = 1e-6,
        non_linearity: str = "swish",
        conv_shortcut: bool = False,
        spatial_norm_dim: Optional[int] = None,
        pad_mode: str = "first",
    ):
        super().__init__()

        out_channels = out_channels or in_channels

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nonlinearity = get_activation(non_linearity)
        self.use_conv_shortcut = conv_shortcut

        if spatial_norm_dim is None:
            self.norm1 = nn.GroupNorm(num_channels=in_channels, num_groups=groups, eps=eps)
            self.norm2 = nn.GroupNorm(num_channels=out_channels, num_groups=groups, eps=eps)
        else:
            self.norm1 = CogVideoXSpatialNorm3D(
                f_channels=in_channels,
                zq_channels=spatial_norm_dim,
                groups=groups,
            )
            self.norm2 = CogVideoXSpatialNorm3D(
                f_channels=out_channels,
                zq_channels=spatial_norm_dim,
                groups=groups,
            )

        self.conv1 = CogVideoXCausalConv3d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, pad_mode=pad_mode
        )

        if temb_channels > 0:
            self.temb_proj = nn.Linear(in_features=temb_channels, out_features=out_channels)

        self.dropout = nn.Dropout(dropout)
        self.conv2 = CogVideoXCausalConv3d(
            in_channels=out_channels, out_channels=out_channels, kernel_size=3, pad_mode=pad_mode
        )

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = CogVideoXCausalConv3d(
                    in_channels=in_channels, out_channels=out_channels, kernel_size=3, pad_mode=pad_mode
                )
            else:
                self.conv_shortcut = CogVideoXSafeConv3d(
                    in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0
                )

    def forward(
        self,
        inputs: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
        zq: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = inputs

        if zq is not None:
            hidden_states = self.norm1(hidden_states, zq)
        else:
            hidden_states = self.norm1(hidden_states)

        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv1(hidden_states)

        if temb is not None:
            hidden_states = hidden_states + self.temb_proj(self.nonlinearity(temb))[:, :, None, None, None]

        if zq is not None:
            hidden_states = self.norm2(hidden_states, zq)
        else:
            hidden_states = self.norm2(hidden_states)

        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.in_channels != self.out_channels:
            inputs = self.conv_shortcut(inputs)

        hidden_states = hidden_states + inputs
        return hidden_states


class CogVideoXDownBlock3D(nn.Module):
    r"""
    A downsampling block used in the CogVideoX model.

    Args:
        in_channels (`int`):
            Number of input channels.
        out_channels (`int`, *optional*):
            Number of output channels. If None, defaults to `in_channels`.
        temb_channels (`int`, defaults to `512`):
            Number of time embedding channels.
        num_layers (`int`, defaults to `1`):
            Number of resnet layers.
        dropout (`float`, defaults to `0.0`):
            Dropout rate.
        resnet_eps (`float`, defaults to `1e-6`):
            Epsilon value for normalization layers.
        resnet_act_fn (`str`, defaults to `"swish"`):
            Activation function to use.
        resnet_groups (`int`, defaults to `32`):
            Number of groups to separate the channels into for group normalization.
        add_downsample (`bool`, defaults to `True`):
            Whether or not to use a downsampling layer. If not used, output dimension would be same as input dimension.
        compress_time (`bool`, defaults to `False`):
            Whether or not to downsample across temporal dimension.
        pad_mode (str, defaults to `"first"`):
            Padding mode.
    """

    _supports_gradient_checkpointing = True

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        add_downsample: bool = True,
        downsample_padding: int = 0,
        compress_time: bool = False,
        pad_mode: str = "first",
    ):
        super().__init__()

        resnets = []
        for i in range(num_layers):
            in_channel = in_channels if i == 0 else out_channels
            resnets.append(
                CogVideoXResnetBlock3D(
                    in_channels=in_channel,
                    out_channels=out_channels,
                    dropout=dropout,
                    temb_channels=temb_channels,
                    groups=resnet_groups,
                    eps=resnet_eps,
                    non_linearity=resnet_act_fn,
                    pad_mode=pad_mode,
                )
            )

        self.resnets = nn.ModuleList(resnets)
        self.downsamplers = None

        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    CogVideoXDownsample3D(
                        out_channels, out_channels, padding=downsample_padding, compress_time=compress_time
                    )
                ]
            )

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
        zq: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for resnet in self.resnets:
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def create_forward(*inputs):
                        return module(*inputs)

                    return create_forward

                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(resnet), hidden_states, temb, zq
                )
            else:
                hidden_states = resnet(hidden_states, temb, zq)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

        return hidden_states


class CogVideoXMidBlock3D(nn.Module):
    r"""
    A middle block used in the CogVideoX model.

    Args:
        in_channels (`int`):
            Number of input channels.
        temb_channels (`int`, defaults to `512`):
            Number of time embedding channels.
        dropout (`float`, defaults to `0.0`):
            Dropout rate.
        num_layers (`int`, defaults to `1`):
            Number of resnet layers.
        resnet_eps (`float`, defaults to `1e-6`):
            Epsilon value for normalization layers.
        resnet_act_fn (`str`, defaults to `"swish"`):
            Activation function to use.
        resnet_groups (`int`, defaults to `32`):
            Number of groups to separate the channels into for group normalization.
        spatial_norm_dim (`int`, *optional*):
            The dimension to use for spatial norm if it is to be used instead of group norm.
        pad_mode (str, defaults to `"first"`):
            Padding mode.
    """

    _supports_gradient_checkpointing = True

    def __init__(
        self,
        in_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        spatial_norm_dim: Optional[int] = None,
        pad_mode: str = "first",
    ):
        super().__init__()

        resnets = []
        for _ in range(num_layers):
            resnets.append(
                CogVideoXResnetBlock3D(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    dropout=dropout,
                    temb_channels=temb_channels,
                    groups=resnet_groups,
                    eps=resnet_eps,
                    spatial_norm_dim=spatial_norm_dim,
                    non_linearity=resnet_act_fn,
                    pad_mode=pad_mode,
                )
            )
        self.resnets = nn.ModuleList(resnets)

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
        zq: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for resnet in self.resnets:
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def create_forward(*inputs):
                        return module(*inputs)

                    return create_forward

                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(resnet), hidden_states, temb, zq
                )
            else:
                hidden_states = resnet(hidden_states, temb, zq)

        return hidden_states


class CogVideoXUpBlock3D(nn.Module):
    r"""
    An upsampling block used in the CogVideoX model.

    Args:
        in_channels (`int`):
            Number of input channels.
        out_channels (`int`, *optional*):
            Number of output channels. If None, defaults to `in_channels`.
        temb_channels (`int`, defaults to `512`):
            Number of time embedding channels.
        dropout (`float`, defaults to `0.0`):
            Dropout rate.
        num_layers (`int`, defaults to `1`):
            Number of resnet layers.
        resnet_eps (`float`, defaults to `1e-6`):
            Epsilon value for normalization layers.
        resnet_act_fn (`str`, defaults to `"swish"`):
            Activation function to use.
        resnet_groups (`int`, defaults to `32`):
            Number of groups to separate the channels into for group normalization.
        spatial_norm_dim (`int`, defaults to `16`):
            The dimension to use for spatial norm if it is to be used instead of group norm.
        add_upsample (`bool`, defaults to `True`):
            Whether or not to use a upsampling layer. If not used, output dimension would be same as input dimension.
        compress_time (`bool`, defaults to `False`):
            Whether or not to downsample across temporal dimension.
        pad_mode (str, defaults to `"first"`):
            Padding mode.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        spatial_norm_dim: int = 16,
        add_upsample: bool = True,
        upsample_padding: int = 1,
        compress_time: bool = False,
        pad_mode: str = "first",
    ):
        super().__init__()

        resnets = []
        for i in range(num_layers):
            in_channel = in_channels if i == 0 else out_channels
            resnets.append(
                CogVideoXResnetBlock3D(
                    in_channels=in_channel,
                    out_channels=out_channels,
                    dropout=dropout,
                    temb_channels=temb_channels,
                    groups=resnet_groups,
                    eps=resnet_eps,
                    non_linearity=resnet_act_fn,
                    spatial_norm_dim=spatial_norm_dim,
                    pad_mode=pad_mode,
                )
            )

        self.resnets = nn.ModuleList(resnets)
        self.upsamplers = None

        if add_upsample:
            self.upsamplers = nn.ModuleList(
                [
                    CogVideoXUpsample3D(
                        out_channels, out_channels, padding=upsample_padding, compress_time=compress_time
                    )
                ]
            )

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
        zq: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        r"""Forward method of the `CogVideoXUpBlock3D` class."""
        for resnet in self.resnets:
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def create_forward(*inputs):
                        return module(*inputs)

                    return create_forward

                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(resnet), hidden_states, temb, zq
                )
            else:
                hidden_states = resnet(hidden_states, temb, zq)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)

        return hidden_states


class CogVideoXEncoder3D(nn.Module):
    r"""
    The `CogVideoXEncoder3D` layer of a variational autoencoder that encodes its input into a latent representation.

    Args:
        in_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        out_channels (`int`, *optional*, defaults to 3):
            The number of output channels.
        down_block_types (`Tuple[str, ...]`, *optional*, defaults to `("DownEncoderBlock2D",)`):
            The types of down blocks to use. See `~diffusers.models.unet_2d_blocks.get_down_block` for available
            options.
        block_out_channels (`Tuple[int, ...]`, *optional*, defaults to `(64,)`):
            The number of output channels for each block.
        act_fn (`str`, *optional*, defaults to `"silu"`):
            The activation function to use. See `~diffusers.models.activations.get_activation` for available options.
        layers_per_block (`int`, *optional*, defaults to 2):
            The number of layers per block.
        norm_num_groups (`int`, *optional*, defaults to 32):
            The number of groups for normalization.
    """

    _supports_gradient_checkpointing = True

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 16,
        down_block_types: Tuple[str, ...] = (
            "CogVideoXDownBlock3D",
            "CogVideoXDownBlock3D",
            "CogVideoXDownBlock3D",
            "CogVideoXDownBlock3D",
        ),
        block_out_channels: Tuple[int, ...] = (128, 256, 256, 512),
        layers_per_block: int = 3,
        act_fn: str = "silu",
        norm_eps: float = 1e-6,
        norm_num_groups: int = 32,
        dropout: float = 0.0,
        pad_mode: str = "first",
        temporal_compression_ratio: float = 4,
    ):
        super().__init__()

        # log2 of temporal_compress_times
        temporal_compress_level = int(np.log2(temporal_compression_ratio))

        self.conv_in = CogVideoXCausalConv3d(in_channels, block_out_channels[0], kernel_size=3, pad_mode=pad_mode)
        self.down_blocks = nn.ModuleList([])

        # down blocks
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1
            compress_time = i < temporal_compress_level

            if down_block_type == "CogVideoXDownBlock3D":
                down_block = CogVideoXDownBlock3D(
                    in_channels=input_channel,
                    out_channels=output_channel,
                    temb_channels=0,
                    dropout=dropout,
                    num_layers=layers_per_block,
                    resnet_eps=norm_eps,
                    resnet_act_fn=act_fn,
                    resnet_groups=norm_num_groups,
                    add_downsample=not is_final_block,
                    compress_time=compress_time,
                )
            else:
                raise ValueError("Invalid `down_block_type` encountered. Must be `CogVideoXDownBlock3D`")

            self.down_blocks.append(down_block)

        # mid block
        self.mid_block = CogVideoXMidBlock3D(
            in_channels=block_out_channels[-1],
            temb_channels=0,
            dropout=dropout,
            num_layers=2,
            resnet_eps=norm_eps,
            resnet_act_fn=act_fn,
            resnet_groups=norm_num_groups,
            pad_mode=pad_mode,
        )

        self.norm_out = nn.GroupNorm(norm_num_groups, block_out_channels[-1], eps=1e-6)
        self.conv_act = nn.SiLU()
        self.conv_out = CogVideoXCausalConv3d(
            block_out_channels[-1], 2 * out_channels, kernel_size=3, pad_mode=pad_mode
        )

        self.gradient_checkpointing = False

    def forward(self, sample: torch.Tensor, temb: Optional[torch.Tensor] = None) -> torch.Tensor:
        r"""The forward method of the `CogVideoXEncoder3D` class."""
        hidden_states = self.conv_in(sample)

        if self.training and self.gradient_checkpointing:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward

            # 1. Down
            for down_block in self.down_blocks:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(down_block), hidden_states, temb, None
                )

            # 2. Mid
            hidden_states = torch.utils.checkpoint.checkpoint(
                create_custom_forward(self.mid_block), hidden_states, temb, None
            )
        else:
            # 1. Down
            for down_block in self.down_blocks:
                hidden_states = down_block(hidden_states, temb, None)

            # 2. Mid
            hidden_states = self.mid_block(hidden_states, temb, None)

        # 3. Post-process
        hidden_states = self.norm_out(hidden_states)
        hidden_states = self.conv_act(hidden_states)
        hidden_states = self.conv_out(hidden_states)
        return hidden_states


class CogVideoXDecoder3D(nn.Module):
    r"""
    The `CogVideoXDecoder3D` layer of a variational autoencoder that decodes its latent representation into an output
    sample.

    Args:
        in_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        out_channels (`int`, *optional*, defaults to 3):
            The number of output channels.
        up_block_types (`Tuple[str, ...]`, *optional*, defaults to `("UpDecoderBlock2D",)`):
            The types of up blocks to use. See `~diffusers.models.unet_2d_blocks.get_up_block` for available options.
        block_out_channels (`Tuple[int, ...]`, *optional*, defaults to `(64,)`):
            The number of output channels for each block.
        act_fn (`str`, *optional*, defaults to `"silu"`):
            The activation function to use. See `~diffusers.models.activations.get_activation` for available options.
        layers_per_block (`int`, *optional*, defaults to 2):
            The number of layers per block.
        norm_num_groups (`int`, *optional*, defaults to 32):
            The number of groups for normalization.
    """

    _supports_gradient_checkpointing = True

    def __init__(
        self,
        in_channels: int = 16,
        out_channels: int = 3,
        up_block_types: Tuple[str, ...] = (
            "CogVideoXUpBlock3D",
            "CogVideoXUpBlock3D",
            "CogVideoXUpBlock3D",
            "CogVideoXUpBlock3D",
        ),
        block_out_channels: Tuple[int, ...] = (128, 256, 256, 512),
        layers_per_block: int = 3,
        act_fn: str = "silu",
        norm_eps: float = 1e-6,
        norm_num_groups: int = 32,
        dropout: float = 0.0,
        pad_mode: str = "first",
        temporal_compression_ratio: float = 4,
    ):
        super().__init__()

        reversed_block_out_channels = list(reversed(block_out_channels))

        self.conv_in = CogVideoXCausalConv3d(
            in_channels, reversed_block_out_channels[0], kernel_size=3, pad_mode=pad_mode
        )

        # mid block
        self.mid_block = CogVideoXMidBlock3D(
            in_channels=reversed_block_out_channels[0],
            temb_channels=0,
            num_layers=2,
            resnet_eps=norm_eps,
            resnet_act_fn=act_fn,
            resnet_groups=norm_num_groups,
            spatial_norm_dim=in_channels,
            pad_mode=pad_mode,
        )

        # up blocks
        self.up_blocks = nn.ModuleList([])

        output_channel = reversed_block_out_channels[0]
        temporal_compress_level = int(np.log2(temporal_compression_ratio))

        for i, up_block_type in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1
            compress_time = i < temporal_compress_level

            if up_block_type == "CogVideoXUpBlock3D":
                up_block = CogVideoXUpBlock3D(
                    in_channels=prev_output_channel,
                    out_channels=output_channel,
                    temb_channels=0,
                    dropout=dropout,
                    num_layers=layers_per_block + 1,
                    resnet_eps=norm_eps,
                    resnet_act_fn=act_fn,
                    resnet_groups=norm_num_groups,
                    spatial_norm_dim=in_channels,
                    add_upsample=not is_final_block,
                    compress_time=compress_time,
                    pad_mode=pad_mode,
                )
                prev_output_channel = output_channel
            else:
                raise ValueError("Invalid `up_block_type` encountered. Must be `CogVideoXUpBlock3D`")

            self.up_blocks.append(up_block)

        self.norm_out = CogVideoXSpatialNorm3D(reversed_block_out_channels[-1], in_channels, groups=norm_num_groups)
        self.conv_act = nn.SiLU()
        self.conv_out = CogVideoXCausalConv3d(
            reversed_block_out_channels[-1], out_channels, kernel_size=3, pad_mode=pad_mode
        )

        self.gradient_checkpointing = False

    def forward(self, sample: torch.Tensor, temb: Optional[torch.Tensor] = None) -> torch.Tensor:
        r"""The forward method of the `CogVideoXDecoder3D` class."""
        hidden_states = self.conv_in(sample)

        if self.training and self.gradient_checkpointing:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward

            # 1. Mid
            hidden_states = torch.utils.checkpoint.checkpoint(
                create_custom_forward(self.mid_block), hidden_states, temb, sample
            )

            # 2. Up
            for up_block in self.up_blocks:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(up_block), hidden_states, temb, sample
                )
        else:
            # 1. Mid
            hidden_states = self.mid_block(hidden_states, temb, sample)

            # 2. Up
            for up_block in self.up_blocks:
                hidden_states = up_block(hidden_states, temb, sample)

        # 3. Post-process
        hidden_states = self.norm_out(hidden_states, sample)
        hidden_states = self.conv_act(hidden_states)
        hidden_states = self.conv_out(hidden_states)
        return hidden_states


class AutoencoderKLCogVideoX(ModelMixin, ConfigMixin, FromOriginalModelMixin):
    r"""
    A VAE model with KL loss for encoding images into latents and decoding latent representations into images. Used in
    [CogVideoX](https://github.com/THUDM/CogVideo).

    This model inherits from [`ModelMixin`]. Check the superclass documentation for it's generic methods implemented
    for all models (such as downloading or saving).

    Parameters:
        in_channels (int, *optional*, defaults to 3): Number of channels in the input image.
        out_channels (int,  *optional*, defaults to 3): Number of channels in the output.
        down_block_types (`Tuple[str]`, *optional*, defaults to `("DownEncoderBlock2D",)`):
            Tuple of downsample block types.
        up_block_types (`Tuple[str]`, *optional*, defaults to `("UpDecoderBlock2D",)`):
            Tuple of upsample block types.
        block_out_channels (`Tuple[int]`, *optional*, defaults to `(64,)`):
            Tuple of block output channels.
        act_fn (`str`, *optional*, defaults to `"silu"`): The activation function to use.
        sample_size (`int`, *optional*, defaults to `32`): Sample input size.
        scaling_factor (`float`, *optional*, defaults to `1.15258426`):
            The component-wise standard deviation of the trained latent space computed using the first batch of the
            training set. This is used to scale the latent space to have unit variance when training the diffusion
            model. The latents are scaled with the formula `z = z * scaling_factor` before being passed to the
            diffusion model. When decoding, the latents are scaled back to the original scale with the formula: `z = 1
            / scaling_factor * z`. For more details, refer to sections 4.3.2 and D.1 of the [High-Resolution Image
            Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) paper.
        force_upcast (`bool`, *optional*, default to `True`):
            If enabled it will force the VAE to run in float32 for high image resolution pipelines, such as SD-XL. VAE
            can be fine-tuned / trained to a lower range without loosing too much precision in which case
            `force_upcast` can be set to `False` - see: https://huggingface.co/madebyollin/sdxl-vae-fp16-fix
    """

    _supports_gradient_checkpointing = True
    _no_split_modules = ["CogVideoXResnetBlock3D"]

    @register_to_config
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: Tuple[str] = (
            "CogVideoXDownBlock3D",
            "CogVideoXDownBlock3D",
            "CogVideoXDownBlock3D",
            "CogVideoXDownBlock3D",
        ),
        up_block_types: Tuple[str] = (
            "CogVideoXUpBlock3D",
            "CogVideoXUpBlock3D",
            "CogVideoXUpBlock3D",
            "CogVideoXUpBlock3D",
        ),
        block_out_channels: Tuple[int] = (128, 256, 256, 512),
        latent_channels: int = 16,
        layers_per_block: int = 3,
        act_fn: str = "silu",
        norm_eps: float = 1e-6,
        norm_num_groups: int = 32,
        temporal_compression_ratio: float = 4,
        sample_height: int = 480,
        sample_width: int = 720,
        scaling_factor: float = 1.15258426,
        shift_factor: Optional[float] = None,
        latents_mean: Optional[Tuple[float]] = None,
        latents_std: Optional[Tuple[float]] = None,
        force_upcast: float = True,
        use_quant_conv: bool = False,
        use_post_quant_conv: bool = False,
    ):
        super().__init__()

        self.encoder = CogVideoXEncoder3D(
            in_channels=in_channels,
            out_channels=latent_channels,
            down_block_types=down_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            norm_eps=norm_eps,
            norm_num_groups=norm_num_groups,
            temporal_compression_ratio=temporal_compression_ratio,
        )
        self.decoder = CogVideoXDecoder3D(
            in_channels=latent_channels,
            out_channels=out_channels,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            norm_eps=norm_eps,
            norm_num_groups=norm_num_groups,
            temporal_compression_ratio=temporal_compression_ratio,
        )
        self.quant_conv = CogVideoXSafeConv3d(2 * out_channels, 2 * out_channels, 1) if use_quant_conv else None
        self.post_quant_conv = CogVideoXSafeConv3d(out_channels, out_channels, 1) if use_post_quant_conv else None

        self.use_slicing = False
        self.use_tiling = False

        # Can be increased to decode more latent frames at once, but comes at a reasonable memory cost and it is not
        # recommended because the temporal parts of the VAE, here, are tricky to understand.
        # If you decode X latent frames together, the number of output frames is:
        #     (X + (2 conv cache) + (2 time upscale_1) + (4 time upscale_2) - (2 causal conv downscale)) => X + 6 frames
        #
        # Example with num_latent_frames_batch_size = 2:
        #     - 12 latent frames: (0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11) are processed together
        #         => (12 // 2 frame slices) * ((2 num_latent_frames_batch_size) + (2 conv cache) + (2 time upscale_1) + (4 time upscale_2) - (2 causal conv downscale))
        #         => 6 * 8 = 48 frames
        #     - 13 latent frames: (0, 1, 2) (special case), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12) are processed together
        #         => (1 frame slice) * ((3 num_latent_frames_batch_size) + (2 conv cache) + (2 time upscale_1) + (4 time upscale_2) - (2 causal conv downscale)) +
        #            ((13 - 3) // 2) * ((2 num_latent_frames_batch_size) + (2 conv cache) + (2 time upscale_1) + (4 time upscale_2) - (2 causal conv downscale))
        #         => 1 * 9 + 5 * 8 = 49 frames
        # It has been implemented this way so as to not have "magic values" in the code base that would be hard to explain. Note that
        # setting it to anything other than 2 would give poor results because the VAE hasn't been trained to be adaptive with different
        # number of temporal frames.
        self.num_latent_frames_batch_size = 2
        self.num_sample_frames_batch_size = 8

        # We make the minimum height and width of sample for tiling half that of the generally supported
        self.tile_sample_min_height = sample_height // 2
        self.tile_sample_min_width = sample_width // 2
        self.tile_latent_min_height = int(
            self.tile_sample_min_height / (2 ** (len(self.config.block_out_channels) - 1))
        )
        self.tile_latent_min_width = int(self.tile_sample_min_width / (2 ** (len(self.config.block_out_channels) - 1)))

        # These are experimental overlap factors that were chosen based on experimentation and seem to work best for
        # 720x480 (WxH) resolution. The above resolution is the strongly recommended generation resolution in CogVideoX
        # and so the tiling implementation has only been tested on those specific resolutions.
        self.tile_overlap_factor_height = 1 / 6
        self.tile_overlap_factor_width = 1 / 5

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (CogVideoXEncoder3D, CogVideoXDecoder3D)):
            module.gradient_checkpointing = value

    def _clear_fake_context_parallel_cache(self):
        for name, module in self.named_modules():
            if isinstance(module, CogVideoXCausalConv3d):
                logger.debug(f"Clearing fake Context Parallel cache for layer: {name}")
                module._clear_fake_context_parallel_cache()

    def enable_tiling(
        self,
        tile_sample_min_height: Optional[int] = None,
        tile_sample_min_width: Optional[int] = None,
        tile_overlap_factor_height: Optional[float] = None,
        tile_overlap_factor_width: Optional[float] = None,
    ) -> None:
        r"""
        Enable tiled VAE decoding. When this option is enabled, the VAE will split the input tensor into tiles to
        compute decoding and encoding in several steps. This is useful for saving a large amount of memory and to allow
        processing larger images.

        Args:
            tile_sample_min_height (`int`, *optional*):
                The minimum height required for a sample to be separated into tiles across the height dimension.
            tile_sample_min_width (`int`, *optional*):
                The minimum width required for a sample to be separated into tiles across the width dimension.
            tile_overlap_factor_height (`int`, *optional*):
                The minimum amount of overlap between two consecutive vertical tiles. This is to ensure that there are
                no tiling artifacts produced across the height dimension. Must be between 0 and 1. Setting a higher
                value might cause more tiles to be processed leading to slow down of the decoding process.
            tile_overlap_factor_width (`int`, *optional*):
                The minimum amount of overlap between two consecutive horizontal tiles. This is to ensure that there
                are no tiling artifacts produced across the width dimension. Must be between 0 and 1. Setting a higher
                value might cause more tiles to be processed leading to slow down of the decoding process.
        """
        self.use_tiling = True
        self.tile_sample_min_height = tile_sample_min_height or self.tile_sample_min_height
        self.tile_sample_min_width = tile_sample_min_width or self.tile_sample_min_width
        self.tile_latent_min_height = int(
            self.tile_sample_min_height / (2 ** (len(self.config.block_out_channels) - 1))
        )
        self.tile_latent_min_width = int(self.tile_sample_min_width / (2 ** (len(self.config.block_out_channels) - 1)))
        self.tile_overlap_factor_height = tile_overlap_factor_height or self.tile_overlap_factor_height
        self.tile_overlap_factor_width = tile_overlap_factor_width or self.tile_overlap_factor_width

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
        batch_size, num_channels, num_frames, height, width = x.shape

        if self.use_tiling and (width > self.tile_sample_min_width or height > self.tile_sample_min_height):
            return self.tiled_encode(x)

        frame_batch_size = self.num_sample_frames_batch_size
        # Note: We expect the number of frames to be either `1` or `frame_batch_size * k` or `frame_batch_size * k + 1` for some k.
        num_batches = num_frames // frame_batch_size if num_frames > 1 else 1
        enc = []
        for i in range(num_batches):
            remaining_frames = num_frames % frame_batch_size
            start_frame = frame_batch_size * i + (0 if i == 0 else remaining_frames)
            end_frame = frame_batch_size * (i + 1) + remaining_frames
            x_intermediate = x[:, :, start_frame:end_frame]
            x_intermediate = self.encoder(x_intermediate)
            if self.quant_conv is not None:
                x_intermediate = self.quant_conv(x_intermediate)
            enc.append(x_intermediate)

        self._clear_fake_context_parallel_cache()
        enc = torch.cat(enc, dim=2)

        return enc

    @apply_forward_hook
    def encode(
        self, x: torch.Tensor, return_dict: bool = True
    ) -> Union[AutoencoderKLOutput, Tuple[DiagonalGaussianDistribution]]:
        """
        Encode a batch of images into latents.

        Args:
            x (`torch.Tensor`): Input batch of images.
            return_dict (`bool`, *optional*, defaults to `True`):
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

    def _decode(self, z: torch.Tensor, return_dict: bool = True) -> Union[DecoderOutput, torch.Tensor]:
        batch_size, num_channels, num_frames, height, width = z.shape

        if self.use_tiling and (width > self.tile_latent_min_width or height > self.tile_latent_min_height):
            return self.tiled_decode(z, return_dict=return_dict)

        frame_batch_size = self.num_latent_frames_batch_size
        num_batches = num_frames // frame_batch_size
        dec = []
        for i in range(num_batches):
            remaining_frames = num_frames % frame_batch_size
            start_frame = frame_batch_size * i + (0 if i == 0 else remaining_frames)
            end_frame = frame_batch_size * (i + 1) + remaining_frames
            z_intermediate = z[:, :, start_frame:end_frame]
            if self.post_quant_conv is not None:
                z_intermediate = self.post_quant_conv(z_intermediate)
            z_intermediate = self.decoder(z_intermediate)
            dec.append(z_intermediate)

        self._clear_fake_context_parallel_cache()
        dec = torch.cat(dec, dim=2)

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)

    @apply_forward_hook
    def decode(self, z: torch.Tensor, return_dict: bool = True) -> Union[DecoderOutput, torch.Tensor]:
        """
        Decode a batch of images.

        Args:
            z (`torch.Tensor`): Input batch of latent vectors.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`~models.vae.DecoderOutput`] instead of a plain tuple.

        Returns:
            [`~models.vae.DecoderOutput`] or `tuple`:
                If return_dict is True, a [`~models.vae.DecoderOutput`] is returned, otherwise a plain `tuple` is
                returned.
        """
        if self.use_slicing and z.shape[0] > 1:
            decoded_slices = [self._decode(z_slice).sample for z_slice in z.split(1)]
            decoded = torch.cat(decoded_slices)
        else:
            decoded = self._decode(z).sample

        if not return_dict:
            return (decoded,)
        return DecoderOutput(sample=decoded)

    def blend_v(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        blend_extent = min(a.shape[3], b.shape[3], blend_extent)
        for y in range(blend_extent):
            b[:, :, :, y, :] = a[:, :, :, -blend_extent + y, :] * (1 - y / blend_extent) + b[:, :, :, y, :] * (
                y / blend_extent
            )
        return b

    def blend_h(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        blend_extent = min(a.shape[4], b.shape[4], blend_extent)
        for x in range(blend_extent):
            b[:, :, :, :, x] = a[:, :, :, :, -blend_extent + x] * (1 - x / blend_extent) + b[:, :, :, :, x] * (
                x / blend_extent
            )
        return b

    def tiled_encode(self, x: torch.Tensor) -> torch.Tensor:
        r"""Encode a batch of images using a tiled encoder.

        When this option is enabled, the VAE will split the input tensor into tiles to compute encoding in several
        steps. This is useful to keep memory use constant regardless of image size. The end result of tiled encoding is
        different from non-tiled encoding because each tile uses a different encoder. To avoid tiling artifacts, the
        tiles overlap and are blended together to form a smooth output. You may still see tile-sized changes in the
        output, but they should be much less noticeable.

        Args:
            x (`torch.Tensor`): Input batch of videos.

        Returns:
            `torch.Tensor`:
                The latent representation of the encoded videos.
        """
        # For a rough memory estimate, take a look at the `tiled_decode` method.
        batch_size, num_channels, num_frames, height, width = x.shape

        overlap_height = int(self.tile_sample_min_height * (1 - self.tile_overlap_factor_height))
        overlap_width = int(self.tile_sample_min_width * (1 - self.tile_overlap_factor_width))
        blend_extent_height = int(self.tile_latent_min_height * self.tile_overlap_factor_height)
        blend_extent_width = int(self.tile_latent_min_width * self.tile_overlap_factor_width)
        row_limit_height = self.tile_latent_min_height - blend_extent_height
        row_limit_width = self.tile_latent_min_width - blend_extent_width
        frame_batch_size = self.num_sample_frames_batch_size

        # Split x into overlapping tiles and encode them separately.
        # The tiles have an overlap to avoid seams between tiles.
        rows = []
        for i in range(0, height, overlap_height):
            row = []
            for j in range(0, width, overlap_width):
                # Note: We expect the number of frames to be either `1` or `frame_batch_size * k` or `frame_batch_size * k + 1` for some k.
                num_batches = num_frames // frame_batch_size if num_frames > 1 else 1
                time = []
                for k in range(num_batches):
                    remaining_frames = num_frames % frame_batch_size
                    start_frame = frame_batch_size * k + (0 if k == 0 else remaining_frames)
                    end_frame = frame_batch_size * (k + 1) + remaining_frames
                    tile = x[
                        :,
                        :,
                        start_frame:end_frame,
                        i : i + self.tile_sample_min_height,
                        j : j + self.tile_sample_min_width,
                    ]
                    tile = self.encoder(tile)
                    if self.quant_conv is not None:
                        tile = self.quant_conv(tile)
                    time.append(tile)
                self._clear_fake_context_parallel_cache()
                row.append(torch.cat(time, dim=2))
            rows.append(row)

        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                # blend the above tile and the left tile
                # to the current tile and add the current tile to the result row
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_extent_height)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_extent_width)
                result_row.append(tile[:, :, :, :row_limit_height, :row_limit_width])
            result_rows.append(torch.cat(result_row, dim=4))

        enc = torch.cat(result_rows, dim=3)
        return enc

    def tiled_decode(self, z: torch.Tensor, return_dict: bool = True) -> Union[DecoderOutput, torch.Tensor]:
        r"""
        Decode a batch of images using a tiled decoder.

        Args:
            z (`torch.Tensor`): Input batch of latent vectors.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.vae.DecoderOutput`] instead of a plain tuple.

        Returns:
            [`~models.vae.DecoderOutput`] or `tuple`:
                If return_dict is True, a [`~models.vae.DecoderOutput`] is returned, otherwise a plain `tuple` is
                returned.
        """
        # Rough memory assessment:
        #   - In CogVideoX-2B, there are a total of 24 CausalConv3d layers.
        #   - The biggest intermediate dimensions are: [1, 128, 9, 480, 720].
        #   - Assume fp16 (2 bytes per value).
        # Memory required: 1 * 128 * 9 * 480 * 720 * 24 * 2 / 1024**3 = 17.8 GB
        #
        # Memory assessment when using tiling:
        #   - Assume everything as above but now HxW is 240x360 by tiling in half
        # Memory required: 1 * 128 * 9 * 240 * 360 * 24 * 2 / 1024**3 = 4.5 GB

        batch_size, num_channels, num_frames, height, width = z.shape

        overlap_height = int(self.tile_latent_min_height * (1 - self.tile_overlap_factor_height))
        overlap_width = int(self.tile_latent_min_width * (1 - self.tile_overlap_factor_width))
        blend_extent_height = int(self.tile_sample_min_height * self.tile_overlap_factor_height)
        blend_extent_width = int(self.tile_sample_min_width * self.tile_overlap_factor_width)
        row_limit_height = self.tile_sample_min_height - blend_extent_height
        row_limit_width = self.tile_sample_min_width - blend_extent_width
        frame_batch_size = self.num_latent_frames_batch_size

        # Split z into overlapping tiles and decode them separately.
        # The tiles have an overlap to avoid seams between tiles.
        rows = []
        for i in range(0, height, overlap_height):
            row = []
            for j in range(0, width, overlap_width):
                num_batches = num_frames // frame_batch_size
                time = []
                for k in range(num_batches):
                    remaining_frames = num_frames % frame_batch_size
                    start_frame = frame_batch_size * k + (0 if k == 0 else remaining_frames)
                    end_frame = frame_batch_size * (k + 1) + remaining_frames
                    tile = z[
                        :,
                        :,
                        start_frame:end_frame,
                        i : i + self.tile_latent_min_height,
                        j : j + self.tile_latent_min_width,
                    ]
                    if self.post_quant_conv is not None:
                        tile = self.post_quant_conv(tile)
                    tile = self.decoder(tile)
                    time.append(tile)
                self._clear_fake_context_parallel_cache()
                row.append(torch.cat(time, dim=2))
            rows.append(row)

        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                # blend the above tile and the left tile
                # to the current tile and add the current tile to the result row
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_extent_height)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_extent_width)
                result_row.append(tile[:, :, :, :row_limit_height, :row_limit_width])
            result_rows.append(torch.cat(result_row, dim=4))

        dec = torch.cat(result_rows, dim=3)

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)

    def forward(
        self,
        sample: torch.Tensor,
        sample_posterior: bool = False,
        return_dict: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> Union[torch.Tensor, torch.Tensor]:
        x = sample
        posterior = self.encode(x).latent_dist
        if sample_posterior:
            z = posterior.sample(generator=generator)
        else:
            z = posterior.mode()
        dec = self.decode(z)
        if not return_dict:
            return (dec,)
        return dec
