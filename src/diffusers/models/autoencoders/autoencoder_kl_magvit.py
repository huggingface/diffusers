# Copyright 2025 The EasyAnimate team and The HuggingFace Team.
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
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from diffusers.utils import is_torch_version

from ...configuration_utils import ConfigMixin, register_to_config
from ...loaders.single_file_model import FromOriginalModelMixin
from ...utils import logging
from ...utils.accelerate_utils import apply_forward_hook
from ..activations import get_activation
from ..attention import Attention
from ..downsampling import EasyAnimateDownsampler3D
from ..modeling_outputs import AutoencoderKLOutput
from ..modeling_utils import ModelMixin
from ..upsampling import EasyAnimateUpsampler3D
from .vae import DecoderOutput, DiagonalGaussianDistribution

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def create_custom_forward(module, return_dict=None):
    def custom_forward(*inputs):
        if return_dict is not None:
            return module(*inputs, return_dict=return_dict)
        else:
            return module(*inputs)

    return custom_forward


def str_eval(item):
    if type(item) == str:
        return eval(item)
    else:
        return item


class CausalConv3d(nn.Conv3d):
    """
    A 3D causal convolutional layer that applies convolution across time (temporal dimension)
    while preserving causality, meaning the output at time t only depends on inputs up to time t.
    
    Parameters:
    - in_channels (int): Number of channels in the input tensor.
    - out_channels (int): Number of channels in the output tensor.
    - kernel_size (int | tuple[int, int, int]): Size of the convolutional kernel. Defaults to 3.
    - stride (int | tuple[int, int, int]): Stride of the convolution. Defaults to 1.
    - padding (int | tuple[int, int, int]): Padding added to all three sides of the input. Defaults to 1.
    - dilation (int | tuple[int, int, int]): Spacing between kernel elements. Defaults to 1.
    - **kwargs: Additional keyword arguments passed to the nn.Conv3d constructor.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3, # : int | tuple[int, int, int], 
        stride=1, # : int | tuple[int, int, int] = 1,
        padding=1, # : int | tuple[int, int, int],  # TODO: change it to 0.
        dilation=1, # :  int | tuple[int, int, int] = 1,
        **kwargs,
    ):
        # Ensure kernel_size, stride, and dilation are tuples of length 3
        kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size,) * 3
        assert len(kernel_size) == 3, f"Kernel size must be a 3-tuple, got {kernel_size} instead."

        stride = stride if isinstance(stride, tuple) else (stride,) * 3
        assert len(stride) == 3, f"Stride must be a 3-tuple, got {stride} instead."

        dilation = dilation if isinstance(dilation, tuple) else (dilation,) * 3
        assert len(dilation) == 3, f"Dilation must be a 3-tuple, got {dilation} instead."

        # Unpack kernel size, stride, and dilation for temporal, height, and width dimensions
        t_ks, h_ks, w_ks = kernel_size
        self.t_stride, h_stride, w_stride = stride
        t_dilation, h_dilation, w_dilation = dilation

        # Calculate padding for temporal dimension to maintain causality
        t_pad = (t_ks - 1) * t_dilation
        # TODO: align with SD
        # Calculate padding for height and width dimensions based on the padding parameter
        if padding is None:
            h_pad = math.ceil(((h_ks - 1) * h_dilation + (1 - h_stride)) / 2)
            w_pad = math.ceil(((w_ks - 1) * w_dilation + (1 - w_stride)) / 2)
        elif isinstance(padding, int):
            h_pad = w_pad = padding
        else:
            assert NotImplementedError

        # Store temporal padding and initialize flags and previous features cache
        self.temporal_padding = t_pad
        self.temporal_padding_origin = math.ceil(((t_ks - 1) * w_dilation + (1 - w_stride)) / 2)
        self.padding_flag = 0
        self.prev_features = None

        # Initialize the parent class with modified padding
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            padding=(0, h_pad, w_pad),
            **kwargs,
        )

    def _clear_conv_cache(self):
        """
        Clear the cache storing previous features to free memory.
        """
        del self.prev_features
        self.prev_features = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform forward pass of the causal convolution.

        Parameters:
        - x (torch.Tensor): Input tensor of shape (batch_size, channels, time, height, width).

        Returns:
        - torch.Tensor: Output tensor after applying causal convolution.
        """
        # Ensure input tensor is of the correct type
        dtype = x.dtype
        # Apply different padding strategies based on the padding_flag
        if self.padding_flag == 1:
            # Pad the input tensor in the temporal dimension to maintain causality
            x = F.pad(
                x,
                pad=(0, 0, 0, 0, self.temporal_padding, 0),
                mode="replicate",     # TODO: check if this is necessary
            )
            x = x.to(dtype=dtype)

            # Clear cache before processing and store previous features for causality
            self._clear_conv_cache()
            self.prev_features = x[:, :, -self.temporal_padding:].clone()

            # Process the input tensor in chunks along the temporal dimension
            b, c, f, h, w = x.size()
            outputs = []
            i = 0
            while i + self.temporal_padding + 1 <= f:
                out = super().forward(x[:, :, i:i + self.temporal_padding + 1])
                i += self.t_stride
                outputs.append(out)
            return torch.concat(outputs, 2)
        elif self.padding_flag == 2:
            # Concatenate previous features with the input tensor for continuous temporal processing
            if self.t_stride == 2:
                x = torch.concat(
                    [self.prev_features[:, :, -(self.temporal_padding - 1):], x], dim = 2
                )
            else:
                x = torch.concat(
                    [self.prev_features, x], dim = 2
                )
            x = x.to(dtype=dtype)

            # Clear cache and update previous features
            self._clear_conv_cache()
            self.prev_features = x[:, :, -self.temporal_padding:].clone()

            # Process the concatenated tensor in chunks along the temporal dimension
            b, c, f, h, w = x.size()
            outputs = []
            i = 0
            while i + self.temporal_padding + 1 <= f:
                out = super().forward(x[:, :, i:i + self.temporal_padding + 1])
                i += self.t_stride
                outputs.append(out)
            return torch.concat(outputs, 2)
        else:
            # Apply symmetric padding to the temporal dimension for the initial pass
            x = F.pad(
                x,
                pad=(0, 0, 0, 0, self.temporal_padding_origin, self.temporal_padding_origin),
            )
            x = x.to(dtype=dtype)
            return super().forward(x)


class ResidualBlock3D(nn.Module):
    """
    A 3D residual block for deep learning models, incorporating group normalization, 
    non-linear activation functions, and causal convolution. This block is a fundamental 
    component for building deeper 3D convolutional neural networks.
    
    Parameters:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels in the output tensor.
        non_linearity (str): Activation function to use, default is "silu".
        norm_num_groups (int): Number of groups for group normalization, default is 32.
        norm_eps (float): Epsilon value for group normalization, default is 1e-6.
        dropout (float): Dropout rate for regularization, default is 0.0.
        output_scale_factor (float): Scaling factor for the output of the block, default is 1.0.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        non_linearity: str = "silu",
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        dropout: float = 0.0,
        output_scale_factor: float = 1.0,
    ):
        super().__init__()

        self.output_scale_factor = output_scale_factor

        # Group normalization for input tensor
        self.norm1 = nn.GroupNorm(
            num_groups=norm_num_groups,
            num_channels=in_channels,
            eps=norm_eps,
            affine=True,
        )

        # Activation function
        self.nonlinearity = get_activation(non_linearity)

        # First causal convolution layer
        self.conv1 = CausalConv3d(in_channels, out_channels, kernel_size=3)

        # Group normalization for the output of the first convolution
        self.norm2 = nn.GroupNorm(
            num_groups=norm_num_groups,
            num_channels=out_channels,
            eps=norm_eps,
            affine=True,
        )

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

        # Second causal convolution layer
        self.conv2 = CausalConv3d(out_channels, out_channels, kernel_size=3)

        # Shortcut connection for residual learning
        if in_channels != out_channels:
            self.shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

        self.set_3dgroupnorm = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the residual block.
        
        Parameters:
            x (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: Output tensor after applying the residual block.
        """
        shortcut = self.shortcut(x)

        # Apply group normalization and activation function
        if self.set_3dgroupnorm:
            batch_size = x.shape[0]
            x = rearrange(x, "b c t h w -> (b t) c h w")
            x = self.norm1(x)
            x = rearrange(x, "(b t) c h w -> b c t h w", b=batch_size)
        else:
            x = self.norm1(x)
        x = self.nonlinearity(x)

        # First convolution
        x = self.conv1(x)

        # Apply group normalization and activation function again
        if self.set_3dgroupnorm:
            batch_size = x.shape[0]
            x = rearrange(x, "b c t h w -> (b t) c h w")
            x = self.norm2(x)
            x = rearrange(x, "(b t) c h w -> b c t h w", b=batch_size)
        else:
            x = self.norm2(x)
        x = self.nonlinearity(x)

        # Apply dropout and second convolution
        x = self.dropout(x)
        x = self.conv2(x)
        return (x + shortcut) / self.output_scale_factor


class SpatialDownBlock3D(nn.Module):
    """
    A spatial downblock for 3D inputs, combining multiple residual blocks and optional 
    downsampling to reduce spatial dimensions while increasing channel depth.
    
    Parameters:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels in the output tensor.
        num_layers (int): Number of residual layers in the block, default is 1.
        act_fn (str): Activation function to use, default is "silu".
        norm_num_groups (int): Number of groups for group normalization, default is 32.
        norm_eps (float): Epsilon value for group normalization, default is 1e-6.
        dropout (float): Dropout rate for regularization, default is 0.0.
        output_scale_factor (float): Scaling factor for the output of the block, default is 1.0.
        add_downsample (bool): Flag to add downsampling operation, default is True.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 1,
        act_fn: str = "silu",
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        dropout: float = 0.0,
        output_scale_factor: float = 1.0,
        add_downsample: bool = True,
    ):
        super().__init__()

        self.convs = nn.ModuleList([])
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            self.convs.append(
                ResidualBlock3D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    non_linearity=act_fn,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    dropout=dropout,
                    output_scale_factor=output_scale_factor,
                )
            )

        if add_downsample:
            self.downsampler = EasyAnimateDownsampler3D(
                out_channels, out_channels, 
                kernel_size=3, stride=(1, 2, 2), 
            )
            self.spatial_downsample_factor = 2
        else:
            self.downsampler = None
            self.spatial_downsample_factor = 1

        self.temporal_downsample_factor = 1

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """
        Forward pass of the spatial downblock.
        
        Parameters:
            x (torch.FloatTensor): Input tensor.
            
        Returns:
            torch.FloatTensor: Output tensor after applying the spatial downblock.
        """
        for conv in self.convs:
            x = conv(x)

        if self.downsampler is not None:
            x = self.downsampler(x)

        return x


class SpatialTemporalDownBlock3D(nn.Module):
    """
    A 3D down-block that performs spatial-temporal convolution and downsampling.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        num_layers (int): Number of residual layers. Defaults to 1.
        act_fn (str): Activation function to use. Defaults to "silu".
        norm_num_groups (int): Number of groups for group normalization. Defaults to 32.
        norm_eps (float): Epsilon for group normalization. Defaults to 1e-6.
        dropout (float): Dropout rate. Defaults to 0.0.
        output_scale_factor (float): Output scale factor. Defaults to 1.0.
        add_downsample (bool): Whether to add downsampling operation. Defaults to True.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 1,
        act_fn: str = "silu",
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        dropout: float = 0.0,
        output_scale_factor: float = 1.0,
        add_downsample: bool = True,
    ):
        super().__init__()

        self.convs = nn.ModuleList([])
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            self.convs.append(
                ResidualBlock3D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    non_linearity=act_fn,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    dropout=dropout,
                    output_scale_factor=output_scale_factor,
                )
            )

        if add_downsample:
            self.downsampler = EasyAnimateDownsampler3D(
                out_channels, out_channels, 
                kernel_size=3, stride=(2, 2, 2), 
            )
            self.spatial_downsample_factor = 2
            self.temporal_downsample_factor = 2
        else:
            self.downsampler = None
            self.spatial_downsample_factor = 1
            self.temporal_downsample_factor = 1

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        for conv in self.convs:
            x = conv(x)

        if self.downsampler is not None:
            x = self.downsampler(x)

        return x


class MidBlock3D(nn.Module):
    """
    A 3D UNet mid-block with multiple residual blocks and optional attention blocks.

    Args:
        in_channels (int): Number of input channels.
        num_layers (int): Number of residual blocks. Defaults to 1.
        act_fn (str): Activation function for the resnet blocks. Defaults to "silu".
        norm_num_groups (int): Number of groups for group normalization. Defaults to 32.
        norm_eps (float): Epsilon for group normalization. Defaults to 1e-6.
        dropout (float): Dropout rate. Defaults to 0.0.
        output_scale_factor (float): Output scale factor. Defaults to 1.0.

    Returns:
        torch.FloatTensor: Output of the last residual block, with shape (batch_size, in_channels, temporal_length, height, width).
    """

    def __init__(
        self,
        in_channels: int,
        num_layers: int = 1,
        act_fn: str = "silu",
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        dropout: float = 0.0,
        output_scale_factor: float = 1.0,
    ):
        super().__init__()

        norm_num_groups = norm_num_groups if norm_num_groups is not None else min(in_channels // 4, 32)

        self.convs = nn.ModuleList([
            ResidualBlock3D(
                in_channels=in_channels,
                out_channels=in_channels,
                non_linearity=act_fn,
                norm_num_groups=norm_num_groups,
                norm_eps=norm_eps,
                dropout=dropout,
                output_scale_factor=output_scale_factor,
            )
        ])

        for _ in range(num_layers - 1):
            self.convs.append(
                ResidualBlock3D(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    non_linearity=act_fn,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    dropout=dropout,
                    output_scale_factor=output_scale_factor,
                )
            )

    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        hidden_states = self.convs[0](hidden_states)

        for resnet in self.convs[1:]:
            hidden_states = resnet(hidden_states)

        return hidden_states


class SpatialUpBlock3D(nn.Module):
    """
    A 3D up-block that performs spatial convolution and upsampling without temporal upsampling.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        num_layers (int): Number of residual layers. Defaults to 1.
        act_fn (str): Activation function to use. Defaults to "silu".
        norm_num_groups (int): Number of groups for group normalization. Defaults to 32.
        norm_eps (float): Epsilon for group normalization. Defaults to 1e-6.
        dropout (float): Dropout rate. Defaults to 0.0.
        output_scale_factor (float): Output scale factor. Defaults to 1.0.
        add_upsample (bool): Whether to add upsampling operation. Defaults to True.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 1,
        act_fn: str = "silu",
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        dropout: float = 0.0,
        output_scale_factor: float = 1.0,
        add_upsample: bool = True,
    ):
        super().__init__()

        if add_upsample:
            self.upsampler = EasyAnimateUpsampler3D(in_channels, in_channels, temporal_upsample=False)
        else:
            self.upsampler = None

        self.convs = nn.ModuleList([])
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            self.convs.append(
                ResidualBlock3D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    non_linearity=act_fn,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    dropout=dropout,
                    output_scale_factor=output_scale_factor,
                )
            )

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        for conv in self.convs:
            x = conv(x)

        if self.upsampler is not None:
            x = self.upsampler(x)

        return x


class SpatialTemporalUpBlock3D(nn.Module):
    """
    A 3D up-block that performs spatial-temporal convolution and upsampling.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        num_layers (int): Number of residual layers. Defaults to 1.
        act_fn (str): Activation function to use. Defaults to "silu".
        norm_num_groups (int): Number of groups for group normalization. Defaults to 32.
        norm_eps (float): Epsilon for group normalization. Defaults to 1e-6.
        dropout (float): Dropout rate. Defaults to 0.0.
        output_scale_factor (float): Output scale factor. Defaults to 1.0.
        add_upsample (bool): Whether to add upsampling operation. Defaults to True.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 1,
        act_fn: str = "silu",
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        dropout: float = 0.0,
        output_scale_factor: float = 1.0,
        add_upsample: bool = True,
    ):
        super().__init__()

        self.convs = nn.ModuleList([])
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            self.convs.append(
                ResidualBlock3D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    non_linearity=act_fn,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    dropout=dropout,
                    output_scale_factor=output_scale_factor,
                )
            )

        if add_upsample:
            self.upsampler = EasyAnimateUpsampler3D(in_channels, in_channels, temporal_upsample=True)
        else:
            self.upsampler = None

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        for conv in self.convs:
            x = conv(x)

        if self.upsampler is not None:
            x = self.upsampler(x)

        return x

def get_mid_block(
    mid_block_type: str,
    in_channels: int,
    num_layers: int,
    act_fn: str,
    norm_num_groups: int = 32,
    norm_eps: float = 1e-6,
    dropout: float = 0.0,
    output_scale_factor: float = 1.0,
) -> nn.Module:
    if mid_block_type == "MidBlock3D":
        return MidBlock3D(
            in_channels=in_channels,
            num_layers=num_layers,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            dropout=dropout,
            output_scale_factor=output_scale_factor,
        )
    else:
        raise ValueError(f"Unknown mid block type: {mid_block_type}")


def get_down_block(
    down_block_type: str,
    in_channels: int,
    out_channels: int,
    num_layers: int,
    act_fn: str,
    norm_num_groups: int = 32,
    norm_eps: float = 1e-6,
    dropout: float = 0.0,
    output_scale_factor: float = 1.0,
    add_downsample: bool = True,
) -> nn.Module:
    if down_block_type == "SpatialDownBlock3D":
        return SpatialDownBlock3D(
            in_channels=in_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            dropout=dropout,
            output_scale_factor=output_scale_factor,
            add_downsample=add_downsample,
        )
    elif down_block_type == "SpatialTemporalDownBlock3D":
        return SpatialTemporalDownBlock3D(
            in_channels=in_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            dropout=dropout,
            output_scale_factor=output_scale_factor,
            add_downsample=add_downsample,
        )
    else:
        raise ValueError(f"Unknown down block type: {down_block_type}")


def get_up_block(
    up_block_type: str,
    in_channels: int,
    out_channels: int,
    num_layers: int,
    act_fn: str,
    norm_num_groups: int = 32,
    norm_eps: float = 1e-6,
    dropout: float = 0.0,
    output_scale_factor: float = 1.0,
    add_upsample: bool = True,
) -> nn.Module:
    if up_block_type == "SpatialUpBlock3D":
        return SpatialUpBlock3D(
            in_channels=in_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            dropout=dropout,
            output_scale_factor=output_scale_factor,
            add_upsample=add_upsample,
        )
    elif up_block_type == "SpatialTemporalUpBlock3D":
        return SpatialTemporalUpBlock3D(
            in_channels=in_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            dropout=dropout,
            output_scale_factor=output_scale_factor,
            add_upsample=add_upsample,
        )
    else:
        raise ValueError(f"Unknown up block type: {up_block_type}")


class Encoder(nn.Module):
    r"""
    The `Encoder` layer of a variational autoencoder that encodes its input into a latent representation.

    Args:
        in_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        out_channels (`int`, *optional*, defaults to 8):
            The number of output channels.
        down_block_types (`Tuple[str, ...]`, *optional*, defaults to `("SpatialDownBlock3D",)`):
            The types of down blocks to use. 
        block_out_channels (`Tuple[int, ...]`, *optional*, defaults to `(64,)`):
            The number of output channels for each block.
        mid_block_type (`str`, *optional*, defaults to `"MidBlock3D"`):
            The type of mid block to use. 
        layers_per_block (`int`, *optional*, defaults to 2):
            The number of layers per block.
        norm_num_groups (`int`, *optional*, defaults to 32):
            The number of groups for normalization.
        act_fn (`str`, *optional*, defaults to `"silu"`):
            The activation function to use. See `~diffusers.models.activations.get_activation` for available options.
        double_z (`bool`, *optional*, defaults to `True`):
            Whether to double the number of output channels for the last block.
        spatial_group_norm (`bool`, *optional*, defaults to `False`):
            Whether to use spatial group norm in the down blocks.
        mini_batch_encoder (`int`, *optional*, defaults to 9):
            The number of frames to encode in the loop.
    """

    _supports_gradient_checkpointing = True

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 8,
        down_block_types = ("SpatialDownBlock3D",),
        ch = 128,
        ch_mult = [1,2,4,4,],
        block_out_channels = [128, 256, 512, 512],
        mid_block_type: str = "MidBlock3D",
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        act_fn: str = "silu",
        double_z: bool = True,
        spatial_group_norm: bool = False,
        mini_batch_encoder: int = 9,
        verbose = False,
    ):
        super().__init__()
        # Initialize the input convolution layer
        if block_out_channels is None:
            block_out_channels = [ch * i for i in ch_mult]
        assert len(down_block_types) == len(block_out_channels), (
            "Number of down block types must match number of block output channels."
        )
        self.conv_in = CausalConv3d(
            in_channels,
            block_out_channels[0],
            kernel_size=3,
        )

        # Initialize the downsampling blocks
        self.down_blocks = nn.ModuleList([])
        output_channels = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channels = output_channels
            output_channels = block_out_channels[i]
            is_final_block = (i == len(block_out_channels) - 1)
            down_block = get_down_block(
                down_block_type,
                in_channels=input_channels,
                out_channels=output_channels,
                num_layers=layers_per_block,
                act_fn=act_fn,
                norm_num_groups=norm_num_groups,
                norm_eps=1e-6,
                add_downsample=not is_final_block,
            )
            self.down_blocks.append(down_block)

        # Initialize the middle block
        self.mid_block = get_mid_block(
            mid_block_type,
            in_channels=block_out_channels[-1],
            num_layers=layers_per_block,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
            norm_eps=1e-6,
        )

        # Initialize the output normalization and activation layers
        self.conv_norm_out = nn.GroupNorm(
            num_channels=block_out_channels[-1],
            num_groups=norm_num_groups,
            eps=1e-6,
        )
        self.conv_act = get_activation(act_fn)

        # Initialize the output convolution layer
        conv_out_channels = 2 * out_channels if double_z else out_channels
        self.conv_out = CausalConv3d(block_out_channels[-1], conv_out_channels, kernel_size=3)

        # Initialize additional attributes
        self.mini_batch_encoder = mini_batch_encoder
        self.spatial_group_norm = spatial_group_norm
        self.verbose = verbose

        self.gradient_checkpointing = False

    def set_padding_one_frame(self):
        """
        Recursively sets the padding mode for all submodules in the model to one frame.
        This method only affects modules with a 'padding_flag' attribute.
        """

        def _set_padding_one_frame(name, module):
            """
            Helper function to recursively set the padding mode for a given module and its submodules to one frame.
            
            Args:
                name (str): Name of the current module.
                module (nn.Module): Current module to set the padding mode for.
            """
            if hasattr(module, 'padding_flag'):
                if self.verbose:
                    print('Set pad mode for module[%s] type=%s' % (name, str(type(module))))
                module.padding_flag = 1
            for sub_name, sub_mod in module.named_children():
                _set_padding_one_frame(sub_name, sub_mod)

        for name, module in self.named_children():
            _set_padding_one_frame(name, module)

    def set_padding_more_frame(self):
        """
        Recursively sets the padding mode for all submodules in the model to more than one frame.
        This method only affects modules with a 'padding_flag' attribute.
        """

        def _set_padding_more_frame(name, module):
            """
            Helper function to recursively set the padding mode for a given module and its submodules to more than one frame.
            
            Args:
                name (str): Name of the current module.
                module (nn.Module): Current module to set the padding mode for.
            """
            if hasattr(module, 'padding_flag'):
                if self.verbose:
                    print('Set pad mode for module[%s] type=%s' % (name, str(type(module))))
                module.padding_flag = 2
            for sub_name, sub_mod in module.named_children():
                _set_padding_more_frame(sub_name, sub_mod)

        for name, module in self.named_children():
            _set_padding_more_frame(name, module)

    def set_3dgroupnorm_for_submodule(self):
        """
        Recursively enables 3D group normalization for all submodules in the model.
        This method only affects modules with a 'set_3dgroupnorm' attribute.
        """

        def _set_3dgroupnorm_for_submodule(name, module):
            """
            Helper function to recursively enable 3D group normalization for a given module and its submodules.
            
            Args:
                name (str): Name of the current module.
                module (nn.Module): Current module to enable 3D group normalization for.
            """
            if hasattr(module, 'set_3dgroupnorm'):
                if self.verbose:
                    print('Set pad mode for module[%s] type=%s' % (name, str(type(module))))
                module.set_3dgroupnorm = True
            for sub_name, sub_mod in module.named_children():
                _set_3dgroupnorm_for_submodule(sub_name, sub_mod)

        for name, module in self.named_children():
            _set_3dgroupnorm_for_submodule(name, module)

    def single_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass for a single input tensor.
        This method applies checkpointing for gradient computation during training to save memory.

        Args:
            x (torch.Tensor): Input tensor with shape (B, C, T, H, W).

        Returns:
            torch.Tensor: Output tensor after passing through the model.
        """
        # x: (B, C, T, H, W)
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
            x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(self.conv_in),
                    x,
                    **ckpt_kwargs,
                )
        else:
            x = self.conv_in(x)
        for down_block in self.down_blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(down_block),
                    x,
                    **ckpt_kwargs,
                )
            else:
                x = down_block(x)

        x = self.mid_block(x)

        if self.spatial_group_norm:
            batch_size = x.shape[0]
            x = rearrange(x, "b c t h w -> (b t) c h w")
            x = self.conv_norm_out(x)
            x = rearrange(x, "(b t) c h w -> b c t h w", b=batch_size)
        else:
            x = self.conv_norm_out(x)
        x = self.conv_act(x)
        x = self.conv_out(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward propagation process for the input tensor x.
        
        If spatial group normalization is enabled, apply 3D group normalization to all submodules.
        Adjust the padding mode based on the input tensor, process the first frame and subsequent frames in separate batches,
        and finally concatenate the processed results along the frame dimension.
        
        Parameters:
        - x (torch.Tensor): The input tensor, containing a batch of video frames.
        
        Returns:
        - torch.Tensor: The processed output tensor.
        """
        # Check if spatial group normalization is enabled, if so, set 3D group normalization for all submodules
        if self.spatial_group_norm:
            self.set_3dgroupnorm_for_submodule()

        # Set the padding mode for processing the first frame
        self.set_padding_one_frame()
        # Process the first frame and save the result
        first_frames = self.single_forward(x[:, :, 0:1, :, :])
        # Set the padding mode for processing subsequent frames
        self.set_padding_more_frame()
        # Initialize a list to store the processed frame results, with the first frame's result already added
        new_pixel_values = [first_frames]
        # Process the remaining frames in batches, excluding the first frame
        for i in range(1, x.shape[2], self.mini_batch_encoder):
            # Process the next batch of frames and add the result to the list
            next_frames = self.single_forward(x[:, :, i: i + self.mini_batch_encoder, :, :])
            new_pixel_values.append(next_frames)
        # Concatenate all processed frame results along the frame dimension
        new_pixel_values = torch.cat(new_pixel_values, dim=2)
        # Return the final concatenated tensor
        return new_pixel_values


class Decoder(nn.Module):
    r"""
    The `Decoder` layer of a variational autoencoder that decodes its latent representation into an output sample.

    Args:
        in_channels (`int`, *optional*, defaults to 8):
            The number of input channels.
        out_channels (`int`, *optional*, defaults to 3):
            The number of output channels.
        up_block_types (`Tuple[str, ...]`, *optional*, defaults to `("SpatialUpBlock3D",)`):
            The types of up blocks to use. 
        block_out_channels (`Tuple[int, ...]`, *optional*, defaults to `(64,)`):
            The number of output channels for each block.
        mid_block_type (`str`, *optional*, defaults to `"MidBlock3D"`):
            The type of mid block to use. 
        layers_per_block (`int`, *optional*, defaults to 2):
            The number of layers per block.
        norm_num_groups (`int`, *optional*, defaults to 32):
            The number of groups for normalization.
        act_fn (`str`, *optional*, defaults to `"silu"`):
            The activation function to use. See `~diffusers.models.activations.get_activation` for available options.
        spatial_group_norm (`bool`, *optional*, defaults to `False`):
            Whether to use spatial group norm in the up blocks.
        mini_batch_decoder (`int`, *optional*, defaults to 3):
            The number of frames to decode in the loop.
    """

    _supports_gradient_checkpointing = True

    def __init__(
        self,
        in_channels: int = 8,
        out_channels: int = 3,
        up_block_types  = ("SpatialUpBlock3D",),
        ch = 128,
        ch_mult = [1,2,4,4,],
        block_out_channels = [128, 256, 512, 512],
        mid_block_type: str = "MidBlock3D",
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        act_fn: str = "silu",
        spatial_group_norm: bool = False,
        mini_batch_decoder: int = 3, 
        verbose = False,
    ):
        super().__init__()
        # Initialize the block output channels based on ch and ch_mult if not provided
        if block_out_channels is None:
            block_out_channels = [ch * i for i in ch_mult]
        # Ensure the number of up block types matches the number of block output channels
        assert len(up_block_types) == len(block_out_channels), (
            "Number of up block types must match number of block output channels."
        )

        # Input convolution layer
        self.conv_in = CausalConv3d(
            in_channels,
            block_out_channels[-1],
            kernel_size=3,
        )

        # Middle block with attention mechanism
        self.mid_block = get_mid_block(
            mid_block_type,
            in_channels=block_out_channels[-1],
            num_layers=layers_per_block,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
            norm_eps=1e-6,
        )

        # Initialize up blocks for decoding
        self.up_blocks = nn.ModuleList([])
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channels = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            input_channels = output_channels
            output_channels = reversed_block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            # Create and append up block to up_blocks
            up_block = get_up_block(
                up_block_type,
                in_channels=input_channels,
                out_channels=output_channels,
                num_layers=layers_per_block + 1,
                act_fn=act_fn,
                norm_num_groups=norm_num_groups,
                norm_eps=1e-6,
                add_upsample=not is_final_block,
            )
            self.up_blocks.append(up_block)

        # Output normalization and activation
        self.conv_norm_out = nn.GroupNorm(
            num_channels=block_out_channels[0],
            num_groups=norm_num_groups,
            eps=1e-6,
        )
        self.conv_act = get_activation(act_fn)

        # Output convolution layer
        self.conv_out = CausalConv3d(block_out_channels[0], out_channels, kernel_size=3)
        
        # Initialize additional attributes
        self.mini_batch_decoder = mini_batch_decoder
        self.spatial_group_norm = spatial_group_norm
        self.verbose = verbose

        self.gradient_checkpointing = False


    def set_padding_one_frame(self):
        """
        Recursively sets the padding mode for all submodules in the model to one frame.
        This method only affects modules with a 'padding_flag' attribute.
        """

        def _set_padding_one_frame(name, module):
            """
            Helper function to recursively set the padding mode for a given module and its submodules to one frame.
            
            Args:
                name (str): Name of the current module.
                module (nn.Module): Current module to set the padding mode for.
            """
            if hasattr(module, 'padding_flag'):
                if self.verbose:
                    print('Set pad mode for module[%s] type=%s' % (name, str(type(module))))
                module.padding_flag = 1
            for sub_name, sub_mod in module.named_children():
                _set_padding_one_frame(sub_name, sub_mod)

        for name, module in self.named_children():
            _set_padding_one_frame(name, module)

    def set_padding_more_frame(self):
        """
        Recursively sets the padding mode for all submodules in the model to more than one frame.
        This method only affects modules with a 'padding_flag' attribute.
        """

        def _set_padding_more_frame(name, module):
            """
            Helper function to recursively set the padding mode for a given module and its submodules to more than one frame.
            
            Args:
                name (str): Name of the current module.
                module (nn.Module): Current module to set the padding mode for.
            """
            if hasattr(module, 'padding_flag'):
                if self.verbose:
                    print('Set pad mode for module[%s] type=%s' % (name, str(type(module))))
                module.padding_flag = 2
            for sub_name, sub_mod in module.named_children():
                _set_padding_more_frame(sub_name, sub_mod)

        for name, module in self.named_children():
            _set_padding_more_frame(name, module)

    def set_3dgroupnorm_for_submodule(self):
        """
        Recursively enables 3D group normalization for all submodules in the model.
        This method only affects modules with a 'set_3dgroupnorm' attribute.
        """

        def _set_3dgroupnorm_for_submodule(name, module):
            if hasattr(module, 'set_3dgroupnorm'):
                if self.verbose:
                    print('Set groupnorm mode for module[%s] type=%s' % (name, str(type(module))))
                module.set_3dgroupnorm = True
            for sub_name, sub_mod in module.named_children():
                _set_3dgroupnorm_for_submodule(sub_name, sub_mod)

        for name, module in self.named_children():
            _set_3dgroupnorm_for_submodule(name, module)
            
    def single_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass for a single input tensor.
        This method applies checkpointing for gradient computation during training to save memory.

        Args:
            x (torch.Tensor): Input tensor with shape (B, C, T, H, W).

        Returns:
            torch.Tensor: Output tensor after passing through the model.
        """

        # x: (B, C, T, H, W)
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
            x = torch.utils.checkpoint.checkpoint(
                create_custom_forward(self.conv_in),
                x,
                **ckpt_kwargs,
            )
            x = torch.utils.checkpoint.checkpoint(
                create_custom_forward(self.mid_block),
                x,
                **ckpt_kwargs,
            )
        else:
            x = self.conv_in(x)
            x = self.mid_block(x)
                
        for up_block in self.up_blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(up_block),
                    x,
                    **ckpt_kwargs,
                )
            else:
                x = up_block(x)

        if self.spatial_group_norm:
            batch_size = x.shape[0]
            x = rearrange(x, "b c t h w -> (b t) c h w")
            x = self.conv_norm_out(x)
            x = rearrange(x, "(b t) c h w -> b c t h w", b=batch_size)
        else:
            x = self.conv_norm_out(x)

        x = self.conv_act(x)
        x = self.conv_out(x)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward propagation process for the input tensor x.
        
        If spatial group normalization is enabled, apply 3D group normalization to all submodules.
        Adjust the padding mode based on the input tensor, process the first frame and subsequent frames in separate loops,
        and finally concatenate all processed frames along the channel dimension.
        
        Parameters:
        - x (torch.Tensor): The input tensor, containing a batch of video frames.
        
        Returns:
        - torch.Tensor: The processed output tensor.
        """
        # Check if spatial group normalization is enabled, if so, set 3D group normalization for all submodules
        if self.spatial_group_norm:
            self.set_3dgroupnorm_for_submodule()

        # Set the padding mode for processing the first frame
        self.set_padding_one_frame()
        # Process the first frame and save the result
        first_frames = self.single_forward(x[:, :, 0:1, :, :])
        # Set the padding mode for processing subsequent frames
        self.set_padding_more_frame()
        # Initialize the list to store the processed frames, starting with the first frame
        new_pixel_values = [first_frames]
        # Process the remaining frames, with the number of frames processed at a time determined by mini_batch_decoder
        for i in range(1, x.shape[2], self.mini_batch_decoder):
            next_frames = self.single_forward(x[:, :, i: i + self.mini_batch_decoder, :, :])
            new_pixel_values.append(next_frames)
        # Concatenate all processed frames along the channel dimension
        new_pixel_values = torch.cat(new_pixel_values, dim=2)
        # Return the processed output tensor
        return new_pixel_values


class AutoencoderKLMagvit(ModelMixin, ConfigMixin, FromOriginalModelMixin):
    r"""
    A VAE model with KL loss for encoding images into latents and decoding latent representations into images.

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
        latent_channels (`int`, *optional*, defaults to 4): Number of channels in the latent space.
        sample_size (`int`, *optional*, defaults to `32`): Sample input size.
        scaling_factor (`float`, *optional*, defaults to 0.18215):
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

    @register_to_config
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        ch =  128,
        ch_mult = [ 1,2,4,4 ],
        block_out_channels = [128, 256, 512, 512],
        down_block_types: tuple = None,
        up_block_types: tuple = None,
        mid_block_type: str = "MidBlock3D",
        layers_per_block: int = 2,
        act_fn: str = "silu",
        latent_channels: int = 4,
        norm_num_groups: int = 32,
        scaling_factor: float = 0.1825,
        force_upcast: float = True,
        use_tiling=False,
        mini_batch_encoder=9,
        mini_batch_decoder=3,
        spatial_group_norm=False,
        tile_sample_min_size=384,
        tile_overlap_factor=0.25,
    ):
        super().__init__()
        down_block_types = str_eval(down_block_types)
        up_block_types = str_eval(up_block_types)
        # Initialize the encoder
        self.encoder = Encoder(
            in_channels=in_channels,
            out_channels=latent_channels,
            down_block_types=down_block_types,
            ch=ch,
            ch_mult=ch_mult,
            block_out_channels=block_out_channels,
            mid_block_type=mid_block_type,
            layers_per_block=layers_per_block,
            norm_num_groups=norm_num_groups,
            act_fn=act_fn,
            double_z=True,
            mini_batch_encoder=mini_batch_encoder,
            spatial_group_norm=spatial_group_norm,
        )

        # Initialize the decoder
        self.decoder = Decoder(
            in_channels=latent_channels,
            out_channels=out_channels,
            up_block_types=up_block_types,
            ch=ch,
            ch_mult=ch_mult,
            block_out_channels=block_out_channels,
            mid_block_type=mid_block_type,
            layers_per_block=layers_per_block,
            norm_num_groups=norm_num_groups,
            act_fn=act_fn,
            mini_batch_decoder=mini_batch_decoder,
            spatial_group_norm=spatial_group_norm,
        )

        # Initialize convolution layers for quantization and post-quantization
        self.quant_conv = nn.Conv3d(2 * latent_channels, 2 * latent_channels, kernel_size=1)
        self.post_quant_conv = nn.Conv3d(latent_channels, latent_channels, kernel_size=1)

        # Assign mini-batch sizes for encoder and decoder
        self.mini_batch_encoder = mini_batch_encoder
        self.mini_batch_decoder = mini_batch_decoder
        # Initialize tiling and slicing flags
        self.use_slicing = False
        self.use_tiling = use_tiling
        # Set parameters for tiling if used
        self.tile_sample_min_size = tile_sample_min_size
        self.tile_overlap_factor = tile_overlap_factor
        self.tile_latent_min_size = int(self.tile_sample_min_size / (2 ** (len(ch_mult) - 1)))
        # Assign the scaling factor for latent space
        self.scaling_factor = scaling_factor

    def _set_gradient_checkpointing(self, module, value=False):
        # Enable or disable gradient checkpointing for encoder and decoder
        if isinstance(module, (Encoder, Decoder)):
            module.gradient_checkpointing = value

    def _clear_conv_cache(self):
        # Clear cache for convolutional layers if needed
        for name, module in self.named_modules():
            if isinstance(module, CausalConv3d):
                module._clear_conv_cache()

    @apply_forward_hook
    def _encode(
        self, x: torch.FloatTensor, return_dict: bool = True
    ) -> Union[AutoencoderKLOutput, Tuple[DiagonalGaussianDistribution]]:
        """
        Encode a batch of images into latents.

        Args:
            x (`torch.FloatTensor`): Input batch of images.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`~models.autoencoder_kl.AutoencoderKLOutput`] instead of a plain tuple.

        Returns:
                The latent representations of the encoded images. If `return_dict` is True, a
                [`~models.autoencoder_kl.AutoencoderKLOutput`] is returned, otherwise a plain `tuple` is returned.
        """
        if self.use_tiling and (x.shape[-1] > self.tile_sample_min_size or x.shape[-2] > self.tile_sample_min_size):
            x = self.tiled_encode(x, return_dict=return_dict)
            return x

        h = self.encoder(x)
        moments = self.quant_conv(h)
        return moments

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
        self._clear_conv_cache()

        if not return_dict:
            return (posterior,)
        return AutoencoderKLOutput(latent_dist=posterior)

    def _decode(self, z: torch.FloatTensor, return_dict: bool = True) -> Union[DecoderOutput, torch.FloatTensor]:
        if self.use_tiling and (z.shape[-1] > self.tile_latent_min_size or z.shape[-2] > self.tile_latent_min_size):
            return self.tiled_decode(z, return_dict=return_dict)

        z = self.post_quant_conv(z)
        dec = self.decoder(z)

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

        self._clear_conv_cache()
        if not return_dict:
            return (decoded,)
        return DecoderOutput(sample=decoded)

    def blend_v(
        self, a: torch.Tensor, b: torch.Tensor, blend_extent: int
    ) -> torch.Tensor:
        blend_extent = min(a.shape[3], b.shape[3], blend_extent)
        for y in range(blend_extent):
            b[:, :, :, y, :] = a[:, :, :, -blend_extent + y, :] * (
                1 - y / blend_extent
            ) + b[:, :, :, y, :] * (y / blend_extent)
        return b

    def blend_h(
        self, a: torch.Tensor, b: torch.Tensor, blend_extent: int
    ) -> torch.Tensor:
        blend_extent = min(a.shape[4], b.shape[4], blend_extent)
        for x in range(blend_extent):
            b[:, :, :, :, x] = a[:, :, :, :, -blend_extent + x] * (
                1 - x / blend_extent
            ) + b[:, :, :, :, x] * (x / blend_extent)
        return b

    def tiled_encode(self, x: torch.FloatTensor, return_dict: bool = True) -> AutoencoderKLOutput:
        overlap_size = int(self.tile_sample_min_size * (1 - self.tile_overlap_factor))
        blend_extent = int(self.tile_latent_min_size * self.tile_overlap_factor)
        row_limit = self.tile_latent_min_size - blend_extent

        # Split the image into 512x512 tiles and encode them separately.
        rows = []
        for i in range(0, x.shape[3], overlap_size):
            row = []
            for j in range(0, x.shape[4], overlap_size):
                tile = x[
                    :,
                    :,
                    :,
                    i : i + self.tile_sample_min_size,
                    j : j + self.tile_sample_min_size,
                ]
                tile = self.encoder(tile)
                tile = self.quant_conv(tile)
                row.append(tile)
            rows.append(row)
        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                # blend the above tile and the left tile
                # to the current tile and add the current tile to the result row
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_extent)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_extent)
                result_row.append(tile[:, :, :, :row_limit, :row_limit])
            result_rows.append(torch.cat(result_row, dim=4))

        moments = torch.cat(result_rows, dim=3)
        posterior = DiagonalGaussianDistribution(moments)

        if not return_dict:
            return (posterior,)

        return AutoencoderKLOutput(latent_dist=posterior)

    def tiled_decode(self, z: torch.FloatTensor, return_dict: bool = True) -> Union[DecoderOutput, torch.FloatTensor]:
        overlap_size = int(self.tile_latent_min_size * (1 - self.tile_overlap_factor))
        blend_extent = int(self.tile_sample_min_size * self.tile_overlap_factor)
        row_limit = self.tile_sample_min_size - blend_extent

        # Split z into overlapping 64x64 tiles and decode them separately.
        # The tiles have an overlap to avoid seams between tiles.
        rows = []
        for i in range(0, z.shape[3], overlap_size):
            row = []
            for j in range(0, z.shape[4], overlap_size):
                tile = z[
                    :,
                    :,
                    :,
                    i : i + self.tile_latent_min_size,
                    j : j + self.tile_latent_min_size,
                ]
                tile = self.post_quant_conv(tile)
                decoded = self.decoder(tile)
                row.append(decoded)
            rows.append(row)
        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                # blend the above tile and the left tile
                # to the current tile and add the current tile to the result row
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_extent)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_extent)
                result_row.append(tile[:, :, :, :row_limit, :row_limit])
            result_rows.append(torch.cat(result_row, dim=4))

        dec = torch.cat(result_rows, dim=3)
    
        # Handle the lower right corner tile separately
        lower_right_original = z[
            :,
            :,
            :,
            -self.tile_latent_min_size:,
            -self.tile_latent_min_size:
        ]
        quantized_lower_right = self.decoder(self.post_quant_conv(lower_right_original))

        # Combine
        H, W = quantized_lower_right.size(-2), quantized_lower_right.size(-1)
        x_weights = torch.linspace(0, 1, W).unsqueeze(0).repeat(H, 1)
        y_weights = torch.linspace(0, 1, H).unsqueeze(1).repeat(1, W)
        weights = torch.min(x_weights, y_weights)

        if len(dec.size()) == 4:
            weights = weights.unsqueeze(0).unsqueeze(0)
        elif len(dec.size()) == 5:
            weights = weights.unsqueeze(0).unsqueeze(0).unsqueeze(0)

        weights = weights.to(dec.device)
        quantized_area = dec[:, :, :, -H:, -W:]
        combined = weights * quantized_lower_right + (1 - weights) * quantized_area

        dec[:, :, :, -H:, -W:] = combined

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)

    def forward(
        self,
        sample: torch.FloatTensor,
        sample_posterior: bool = False,
        return_dict: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> Union[DecoderOutput, torch.FloatTensor]:
        r"""
        Args:
            sample (`torch.FloatTensor`): Input sample.
            sample_posterior (`bool`, *optional*, defaults to `False`):
                Whether to sample from the posterior.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`DecoderOutput`] instead of a plain tuple.
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
