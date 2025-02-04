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

from diffusers.utils import is_torch_version

from ...configuration_utils import ConfigMixin, register_to_config
from ...loaders.single_file_model import FromOriginalModelMixin
from ...utils import logging
from ...utils.accelerate_utils import apply_forward_hook
from ..activations import get_activation
from ..attention import Attention
from ..modeling_outputs import AutoencoderKLOutput
from ..modeling_utils import ModelMixin
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


class EasyAnimateCausalConv3d(nn.Conv3d):
    """
    A 3D causal convolutional layer that applies convolution across time (temporal dimension)
    while preserving causality, meaning the output at time t only depends on inputs up to time t.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode='zeros',
        device=None,
        dtype=None
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

        self.prev_features = None

        # Initialize the parent class with modified padding
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            padding=(0, h_pad, w_pad),
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype
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
        if self.prev_features is None:
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
        else:
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


class EasyAnimateResidualBlock3D(nn.Module):
    """
    A 3D residual block for deep learning models, incorporating group normalization, 
    non-linear activation functions, and causal convolution. This block is a fundamental 
    component for building deeper 3D convolutional neural networks.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        non_linearity: str = "silu",
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        spatial_group_norm: bool = True,
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
        self.conv1 = EasyAnimateCausalConv3d(in_channels, out_channels, kernel_size=3)

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
        self.conv2 = EasyAnimateCausalConv3d(out_channels, out_channels, kernel_size=3)

        # Shortcut connection for residual learning
        if in_channels != out_channels:
            self.shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

        self.spatial_group_norm = spatial_group_norm

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
        if self.spatial_group_norm:
            batch_size, channels, time, height, width = x.shape
            # Reshape x to merge batch and time dimensions
            x = x.permute(0, 2, 1, 3, 4)  # From (b, t, c, h, w) to (b, c, t, h, w)
            x = x.view(batch_size * time, channels, height, width)
            # Apply normalization
            x = self.norm1(x)
            # Reshape x back to original dimensions
            x = x.view(batch_size, time, channels, height, width)
            # Permute dimensions to match the original order
            x = x.permute(0, 2, 1, 3, 4)  # From (b, t, c, h, w) to (b, c, t, h, w)
        else:
            x = self.norm1(x)
        x = self.nonlinearity(x)

        # First convolution
        x = self.conv1(x)

        # Apply group normalization and activation function again
        if self.spatial_group_norm:
            batch_size, channels, time, height, width = x.shape
            # Reshape x to merge batch and time dimensions
            x = x.permute(0, 2, 1, 3, 4)  # From (b, t, c, h, w) to (b, c, t, h, w)
            x = x.view(batch_size * time, channels, height, width)
            # Apply normalization
            x = self.norm2(x)
            # Reshape x back to original dimensions
            x = x.view(batch_size, time, channels, height, width)
            # Permute dimensions to match the original order
            x = x.permute(0, 2, 1, 3, 4)  # From (b, t, c, h, w) to (b, c, t, h, w)
        else:
            x = self.norm2(x)
        x = self.nonlinearity(x)

        # Apply dropout and second convolution
        x = self.dropout(x)
        x = self.conv2(x)
        return (x + shortcut) / self.output_scale_factor


class EasyAnimateDownsampler3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: tuple = (2, 2, 2),
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.conv = EasyAnimateCausalConv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (0, 1, 0, 1))
        return self.conv(x)


class EasyAnimateUpsampler3D(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int,
        kernel_size: int = 3,
        temporal_upsample: bool = False,
        spatial_group_norm: bool = True,
    ):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels

        self.temporal_upsample = temporal_upsample
        self.spatial_group_norm = spatial_group_norm
        
        self.conv = EasyAnimateCausalConv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size
        )
        self.prev_features = None

    def _clear_conv_cache(self):
        """
        Clear the cache storing previous features to free memory.
        """
        del self.prev_features
        self.prev_features = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=(1, 2, 2), mode="nearest")
        x = self.conv(x)

        if self.temporal_upsample:
            if self.prev_features is None:
                self.prev_features = x
            else:
                x = F.interpolate(
                    x, 
                    scale_factor=(2, 1, 1), mode="trilinear" if not self.spatial_group_norm else "nearest"
                )
        return x


class EasyAnimateDownBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 1,
        act_fn: str = "silu",
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        spatial_group_norm: bool = True,
        dropout: float = 0.0,
        output_scale_factor: float = 1.0,
        add_downsample: bool = True,
        add_temporal_downsample: bool = True,
    ):
        super().__init__()

        self.convs = nn.ModuleList([])
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            self.convs.append(
                EasyAnimateResidualBlock3D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    non_linearity=act_fn,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    spatial_group_norm=spatial_group_norm,
                    dropout=dropout,
                    output_scale_factor=output_scale_factor,
                )
            )

        if add_downsample and add_temporal_downsample:
            self.downsampler = EasyAnimateDownsampler3D(
                out_channels, out_channels, 
                kernel_size=3, stride=(2, 2, 2), 
            )
            self.spatial_downsample_factor = 2
            self.temporal_downsample_factor = 2
        elif add_downsample and not add_temporal_downsample:
            self.downsampler = EasyAnimateDownsampler3D(
                out_channels, out_channels, 
                kernel_size=3, stride=(1, 2, 2), 
            )
            self.spatial_downsample_factor = 2
            self.temporal_downsample_factor = 1
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


class EasyAnimateUpBlock3D(nn.Module):
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
        spatial_group_norm: bool = False,
        dropout: float = 0.0,
        output_scale_factor: float = 1.0,
        add_upsample: bool = True,
        add_temporal_upsample: bool = True,
    ):
        super().__init__()

        self.convs = nn.ModuleList([])
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            self.convs.append(
                EasyAnimateResidualBlock3D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    non_linearity=act_fn,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    spatial_group_norm=spatial_group_norm,
                    dropout=dropout,
                    output_scale_factor=output_scale_factor,
                )
            )

        if add_upsample:
            self.upsampler = EasyAnimateUpsampler3D(
                in_channels, 
                in_channels, 
                temporal_upsample=add_temporal_upsample, 
                spatial_group_norm=spatial_group_norm
            )
        else:
            self.upsampler = None

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        for conv in self.convs:
            x = conv(x)

        if self.upsampler is not None:
            x = self.upsampler(x)

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
        spatial_group_norm: bool = True,
        dropout: float = 0.0,
        output_scale_factor: float = 1.0,
    ):
        super().__init__()

        norm_num_groups = norm_num_groups if norm_num_groups is not None else min(in_channels // 4, 32)

        self.convs = nn.ModuleList([
            EasyAnimateResidualBlock3D(
                in_channels=in_channels,
                out_channels=in_channels,
                non_linearity=act_fn,
                norm_num_groups=norm_num_groups,
                norm_eps=norm_eps,
                spatial_group_norm=spatial_group_norm,
                dropout=dropout,
                output_scale_factor=output_scale_factor,
            )
        ])

        for _ in range(num_layers - 1):
            self.convs.append(
                EasyAnimateResidualBlock3D(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    non_linearity=act_fn,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    spatial_group_norm=spatial_group_norm,
                    dropout=dropout,
                    output_scale_factor=output_scale_factor,
                )
            )

    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        hidden_states = self.convs[0](hidden_states)

        for resnet in self.convs[1:]:
            hidden_states = resnet(hidden_states)

        return hidden_states


class EasyAnimateEncoder(nn.Module):
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
        self.conv_in = EasyAnimateCausalConv3d(
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
            if down_block_type == "SpatialDownBlock3D":
                down_block = EasyAnimateDownBlock3D(
                    in_channels=input_channels,
                    out_channels=output_channels,
                    num_layers=layers_per_block,
                    act_fn=act_fn,
                    norm_num_groups=norm_num_groups,
                    norm_eps=1e-6,
                    spatial_group_norm=spatial_group_norm,
                    add_downsample=not is_final_block,
                    add_temporal_downsample=False,
                )
            elif down_block_type == "SpatialTemporalDownBlock3D":
                down_block = EasyAnimateDownBlock3D(
                    in_channels=input_channels,
                    out_channels=output_channels,
                    num_layers=layers_per_block,
                    act_fn=act_fn,
                    norm_num_groups=norm_num_groups,
                    norm_eps=1e-6,
                    spatial_group_norm=spatial_group_norm,
                    add_downsample=not is_final_block,
                    add_temporal_downsample=True,
                )
            else:
                raise ValueError(f"Unknown up block type: {down_block_type}")
            self.down_blocks.append(down_block)

        # Initialize the middle block
        self.mid_block = MidBlock3D(
            in_channels=block_out_channels[-1],
            num_layers=layers_per_block,
            act_fn=act_fn,
            spatial_group_norm=spatial_group_norm,
            norm_num_groups=norm_num_groups,
            norm_eps=1e-6,
            dropout=0,
            output_scale_factor=1,
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
        self.conv_out = EasyAnimateCausalConv3d(block_out_channels[-1], conv_out_channels, kernel_size=3)

        # Initialize additional attributes
        self.mini_batch_encoder = mini_batch_encoder
        self.spatial_group_norm = spatial_group_norm
        self.verbose = verbose

        self.gradient_checkpointing = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
            batch_size, channels, time, height, width = x.shape
            # Reshape x to merge batch and time dimensions
            x = x.permute(0, 2, 1, 3, 4)
            x = x.view(batch_size * time, channels, height, width)
            # Apply normalization
            x = self.conv_norm_out(x)
            # Reshape x back to original dimensions
            x = x.view(batch_size, time, channels, height, width)
            x = x.permute(0, 2, 1, 3, 4)
        else:
            x = self.conv_norm_out(x)
        x = self.conv_act(x)
        x = self.conv_out(x)
        return x


class EasyAnimateDecoder(nn.Module):
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
        self.conv_in = EasyAnimateCausalConv3d(
            in_channels,
            block_out_channels[-1],
            kernel_size=3,
        )

        # Middle block with attention mechanism
        self.mid_block = MidBlock3D(
            in_channels=block_out_channels[-1],
            num_layers=layers_per_block,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
            norm_eps=1e-6,
            dropout=0,
            output_scale_factor=1,
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
            if up_block_type == "SpatialUpBlock3D":
                up_block = EasyAnimateUpBlock3D(
                    in_channels=input_channels,
                    out_channels=output_channels,
                    num_layers=layers_per_block + 1,
                    act_fn=act_fn,
                    norm_num_groups=norm_num_groups,
                    norm_eps=1e-6,
                    spatial_group_norm=spatial_group_norm,
                    add_upsample=not is_final_block,
                    add_temporal_upsample=False,
                )
            elif up_block_type == "SpatialTemporalUpBlock3D":
                up_block = EasyAnimateUpBlock3D(
                    in_channels=input_channels,
                    out_channels=output_channels,
                    num_layers=layers_per_block + 1,
                    act_fn=act_fn,
                    norm_num_groups=norm_num_groups,
                    norm_eps=1e-6,
                    spatial_group_norm=spatial_group_norm,
                    add_upsample=not is_final_block,
                    add_temporal_upsample=True
                )
            else:
                raise ValueError(f"Unknown up block type: {up_block_type}")
            self.up_blocks.append(up_block)

        # Output normalization and activation
        self.conv_norm_out = nn.GroupNorm(
            num_channels=block_out_channels[0],
            num_groups=norm_num_groups,
            eps=1e-6,
        )
        self.conv_act = get_activation(act_fn)

        # Output convolution layer
        self.conv_out = EasyAnimateCausalConv3d(block_out_channels[0], out_channels, kernel_size=3)
        
        # Initialize additional attributes
        self.mini_batch_decoder = mini_batch_decoder
        self.spatial_group_norm = spatial_group_norm
        self.verbose = verbose

        self.gradient_checkpointing = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
            batch_size, channels, time, height, width = x.shape
            # Reshape x to merge batch and time dimensions
            x = x.permute(0, 2, 1, 3, 4)
            x = x.view(batch_size * time, channels, height, width)
            # Apply normalization
            x = self.conv_norm_out(x)
            # Reshape x back to original dimensions
            x = x.view(batch_size, time, channels, height, width)
            x = x.permute(0, 2, 1, 3, 4)
        else:
            x = self.conv_norm_out(x)

        x = self.conv_act(x)
        x = self.conv_out(x)
        return x


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
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        ch =  128,
        ch_mult = [1, 2, 4, 4],
        block_out_channels = [128, 256, 512, 512],
        down_block_types: tuple = [
            "SpatialDownBlock3D", 
            "EasyAnimateDownBlock3D", 
            "EasyAnimateDownBlock3D",
            "EasyAnimateDownBlock3D"
        ],
        up_block_types: tuple = [
            "SpatialUpBlock3D", 
            "EasyAnimateUpBlock3D", 
            "EasyAnimateUpBlock3D",
            "EasyAnimateUpBlock3D"
        ],
        layers_per_block: int = 2,
        act_fn: str = "silu",
        latent_channels: int = 16,
        norm_num_groups: int = 32,
        scaling_factor: float = 0.7125,
        spatial_group_norm=True,
        mini_batch_encoder=4,
        mini_batch_decoder=1,
        tile_sample_min_size=384,
        tile_overlap_factor=0.25,
    ):
        super().__init__()
        down_block_types = str_eval(down_block_types)
        up_block_types = str_eval(up_block_types)
        # Initialize the encoder
        self.encoder = EasyAnimateEncoder(
            in_channels=in_channels,
            out_channels=latent_channels,
            down_block_types=down_block_types,
            ch=ch,
            ch_mult=ch_mult,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            norm_num_groups=norm_num_groups,
            act_fn=act_fn,
            double_z=True,
            mini_batch_encoder=mini_batch_encoder,
            spatial_group_norm=spatial_group_norm,
        )

        # Initialize the decoder
        self.decoder = EasyAnimateDecoder(
            in_channels=latent_channels,
            out_channels=out_channels,
            up_block_types=up_block_types,
            ch=ch,
            ch_mult=ch_mult,
            block_out_channels=block_out_channels,
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
        self.use_tiling = False
        # Set parameters for tiling if used
        self.tile_sample_min_size = tile_sample_min_size
        self.tile_overlap_factor = tile_overlap_factor
        self.tile_latent_min_size = int(self.tile_sample_min_size / (2 ** (len(ch_mult) - 1)))
        # Assign the scaling factor for latent space
        self.scaling_factor = scaling_factor

    def _set_gradient_checkpointing(self, module, value=False):
        # Enable or disable gradient checkpointing for encoder and decoder
        if isinstance(module, (EasyAnimateEncoder, EasyAnimateDecoder)):
            module.gradient_checkpointing = value

    def _clear_conv_cache(self):
        # Clear cache for convolutional layers if needed
        for name, module in self.named_modules():
            if isinstance(module, EasyAnimateCausalConv3d):
                module._clear_conv_cache()
            if isinstance(module, EasyAnimateUpsampler3D):
                module._clear_conv_cache()

    def enable_tiling(
        self,
    ) -> None:
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

        first_frames = self.encoder(x[:, :, 0:1, :, :])
        h = [first_frames]
        for i in range(1, x.shape[2], self.mini_batch_encoder):
            next_frames = self.encoder(x[:, :, i: i + self.mini_batch_encoder, :, :])
            h.append(next_frames)
        h = torch.cat(h, dim=2)
        moments = self.quant_conv(h)

        self._clear_conv_cache()
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

        if not return_dict:
            return (posterior,)
        return AutoencoderKLOutput(latent_dist=posterior)

    def _decode(self, z: torch.FloatTensor, return_dict: bool = True) -> Union[DecoderOutput, torch.FloatTensor]:
        if self.use_tiling and (z.shape[-1] > self.tile_latent_min_size or z.shape[-2] > self.tile_latent_min_size):
            return self.tiled_decode(z, return_dict=return_dict)

        z = self.post_quant_conv(z)

        # Process the first frame and save the result
        first_frames = self.decoder(z[:, :, 0:1, :, :])
        # Initialize the list to store the processed frames, starting with the first frame
        dec = [first_frames]
        # Process the remaining frames, with the number of frames processed at a time determined by mini_batch_decoder
        for i in range(1, z.shape[2], self.mini_batch_decoder):
            next_frames = self.decoder(z[:, :, i: i + self.mini_batch_decoder, :, :])
            dec.append(next_frames)
        # Concatenate all processed frames along the channel dimension
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

                first_frames = self.encoder(tile[:, :, 0:1, :, :])
                tile_h = [first_frames]
                for frame_index in range(1, tile.shape[2], self.mini_batch_encoder):
                    next_frames = self.encoder(tile[:, :, frame_index: frame_index + self.mini_batch_encoder, :, :])
                    tile_h.append(next_frames)
                tile = torch.cat(tile_h, dim=2)
                tile = self.quant_conv(tile)
                self._clear_conv_cache()
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
        return moments

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

                # Process the first frame and save the result
                first_frames = self.decoder(tile[:, :, 0:1, :, :])
                # Initialize the list to store the processed frames, starting with the first frame
                tile_dec = [first_frames]
                # Process the remaining frames, with the number of frames processed at a time determined by mini_batch_decoder
                for frame_index in range(1, tile.shape[2], self.mini_batch_decoder):
                    next_frames = self.decoder(tile[:, :, frame_index: frame_index + self.mini_batch_decoder, :, :])
                    tile_dec.append(next_frames)
                # Concatenate all processed frames along the channel dimension
                decoded = torch.cat(tile_dec, dim=2)
                self._clear_conv_cache()
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
