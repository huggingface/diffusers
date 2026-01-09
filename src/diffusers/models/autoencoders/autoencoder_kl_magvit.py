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
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...configuration_utils import ConfigMixin, register_to_config
from ...utils import logging
from ...utils.accelerate_utils import apply_forward_hook
from ..activations import get_activation
from ..modeling_outputs import AutoencoderKLOutput
from ..modeling_utils import ModelMixin
from .vae import AutoencoderMixin, DecoderOutput, DiagonalGaussianDistribution


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class EasyAnimateCausalConv3d(nn.Conv3d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, ...]] = 3,
        stride: Union[int, Tuple[int, ...]] = 1,
        padding: Union[int, Tuple[int, ...]] = 1,
        dilation: Union[int, Tuple[int, ...]] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
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
        )

    def _clear_conv_cache(self):
        del self.prev_features
        self.prev_features = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Ensure input tensor is of the correct type
        dtype = hidden_states.dtype
        if self.prev_features is None:
            # Pad the input tensor in the temporal dimension to maintain causality
            hidden_states = F.pad(
                hidden_states,
                pad=(0, 0, 0, 0, self.temporal_padding, 0),
                mode="replicate",  # TODO: check if this is necessary
            )
            hidden_states = hidden_states.to(dtype=dtype)

            # Clear cache before processing and store previous features for causality
            self._clear_conv_cache()
            self.prev_features = hidden_states[:, :, -self.temporal_padding :].clone()

            # Process the input tensor in chunks along the temporal dimension
            num_frames = hidden_states.size(2)
            outputs = []
            i = 0
            while i + self.temporal_padding + 1 <= num_frames:
                out = super().forward(hidden_states[:, :, i : i + self.temporal_padding + 1])
                i += self.t_stride
                outputs.append(out)
            return torch.concat(outputs, 2)
        else:
            # Concatenate previous features with the input tensor for continuous temporal processing
            if self.t_stride == 2:
                hidden_states = torch.concat(
                    [self.prev_features[:, :, -(self.temporal_padding - 1) :], hidden_states], dim=2
                )
            else:
                hidden_states = torch.concat([self.prev_features, hidden_states], dim=2)
            hidden_states = hidden_states.to(dtype=dtype)

            # Clear cache and update previous features
            self._clear_conv_cache()
            self.prev_features = hidden_states[:, :, -self.temporal_padding :].clone()

            # Process the concatenated tensor in chunks along the temporal dimension
            num_frames = hidden_states.size(2)
            outputs = []
            i = 0
            while i + self.temporal_padding + 1 <= num_frames:
                out = super().forward(hidden_states[:, :, i : i + self.temporal_padding + 1])
                i += self.t_stride
                outputs.append(out)
            return torch.concat(outputs, 2)


class EasyAnimateResidualBlock3D(nn.Module):
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
        self.nonlinearity = get_activation(non_linearity)
        self.conv1 = EasyAnimateCausalConv3d(in_channels, out_channels, kernel_size=3)

        self.norm2 = nn.GroupNorm(num_groups=norm_num_groups, num_channels=out_channels, eps=norm_eps, affine=True)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = EasyAnimateCausalConv3d(out_channels, out_channels, kernel_size=3)

        if in_channels != out_channels:
            self.shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

        self.spatial_group_norm = spatial_group_norm

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        shortcut = self.shortcut(hidden_states)

        if self.spatial_group_norm:
            batch_size = hidden_states.size(0)
            hidden_states = hidden_states.permute(0, 2, 1, 3, 4).flatten(0, 1)  # [B, C, T, H, W] -> [B * T, C, H, W]
            hidden_states = self.norm1(hidden_states)
            hidden_states = hidden_states.unflatten(0, (batch_size, -1)).permute(
                0, 2, 1, 3, 4
            )  # [B * T, C, H, W] -> [B, C, T, H, W]
        else:
            hidden_states = self.norm1(hidden_states)

        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv1(hidden_states)

        if self.spatial_group_norm:
            batch_size = hidden_states.size(0)
            hidden_states = hidden_states.permute(0, 2, 1, 3, 4).flatten(0, 1)  # [B, C, T, H, W] -> [B * T, C, H, W]
            hidden_states = self.norm2(hidden_states)
            hidden_states = hidden_states.unflatten(0, (batch_size, -1)).permute(
                0, 2, 1, 3, 4
            )  # [B * T, C, H, W] -> [B, C, T, H, W]
        else:
            hidden_states = self.norm2(hidden_states)

        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        return (hidden_states + shortcut) / self.output_scale_factor


class EasyAnimateDownsampler3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: tuple = (2, 2, 2)):
        super().__init__()

        self.conv = EasyAnimateCausalConv3d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=0
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = F.pad(hidden_states, (0, 1, 0, 1))
        hidden_states = self.conv(hidden_states)
        return hidden_states


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
        out_channels = out_channels or in_channels

        self.temporal_upsample = temporal_upsample
        self.spatial_group_norm = spatial_group_norm

        self.conv = EasyAnimateCausalConv3d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size
        )
        self.prev_features = None

    def _clear_conv_cache(self):
        del self.prev_features
        self.prev_features = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = F.interpolate(hidden_states, scale_factor=(1, 2, 2), mode="nearest")
        hidden_states = self.conv(hidden_states)

        if self.temporal_upsample:
            if self.prev_features is None:
                self.prev_features = hidden_states
            else:
                hidden_states = F.interpolate(
                    hidden_states,
                    scale_factor=(2, 1, 1),
                    mode="trilinear" if not self.spatial_group_norm else "nearest",
                )
        return hidden_states


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
            self.downsampler = EasyAnimateDownsampler3D(out_channels, out_channels, kernel_size=3, stride=(2, 2, 2))
            self.spatial_downsample_factor = 2
            self.temporal_downsample_factor = 2
        elif add_downsample and not add_temporal_downsample:
            self.downsampler = EasyAnimateDownsampler3D(out_channels, out_channels, kernel_size=3, stride=(1, 2, 2))
            self.spatial_downsample_factor = 2
            self.temporal_downsample_factor = 1
        else:
            self.downsampler = None
            self.spatial_downsample_factor = 1
            self.temporal_downsample_factor = 1

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for conv in self.convs:
            hidden_states = conv(hidden_states)
        if self.downsampler is not None:
            hidden_states = self.downsampler(hidden_states)
        return hidden_states


class EasyAnimateUpBlock3d(nn.Module):
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
                spatial_group_norm=spatial_group_norm,
            )
        else:
            self.upsampler = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for conv in self.convs:
            hidden_states = conv(hidden_states)
        if self.upsampler is not None:
            hidden_states = self.upsampler(hidden_states)
        return hidden_states


class EasyAnimateMidBlock3d(nn.Module):
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

        self.convs = nn.ModuleList(
            [
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
            ]
        )

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

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.convs[0](hidden_states)
        for resnet in self.convs[1:]:
            hidden_states = resnet(hidden_states)
        return hidden_states


class EasyAnimateEncoder(nn.Module):
    r"""
    Causal encoder for 3D video-like data used in [EasyAnimate](https://huggingface.co/papers/2405.18991).
    """

    _supports_gradient_checkpointing = True

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 8,
        down_block_types: Tuple[str, ...] = (
            "SpatialDownBlock3D",
            "SpatialTemporalDownBlock3D",
            "SpatialTemporalDownBlock3D",
            "SpatialTemporalDownBlock3D",
        ),
        block_out_channels: Tuple[int, ...] = [128, 256, 512, 512],
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        act_fn: str = "silu",
        double_z: bool = True,
        spatial_group_norm: bool = False,
    ):
        super().__init__()

        # 1. Input convolution
        self.conv_in = EasyAnimateCausalConv3d(in_channels, block_out_channels[0], kernel_size=3)

        # 2. Down blocks
        self.down_blocks = nn.ModuleList([])
        output_channels = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channels = output_channels
            output_channels = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1
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

        # 3. Middle block
        self.mid_block = EasyAnimateMidBlock3d(
            in_channels=block_out_channels[-1],
            num_layers=layers_per_block,
            act_fn=act_fn,
            spatial_group_norm=spatial_group_norm,
            norm_num_groups=norm_num_groups,
            norm_eps=1e-6,
            dropout=0,
            output_scale_factor=1,
        )

        # 4. Output normalization & convolution
        self.spatial_group_norm = spatial_group_norm
        self.conv_norm_out = nn.GroupNorm(
            num_channels=block_out_channels[-1],
            num_groups=norm_num_groups,
            eps=1e-6,
        )
        self.conv_act = get_activation(act_fn)

        # Initialize the output convolution layer
        conv_out_channels = 2 * out_channels if double_z else out_channels
        self.conv_out = EasyAnimateCausalConv3d(block_out_channels[-1], conv_out_channels, kernel_size=3)

        self.gradient_checkpointing = False

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states: (B, C, T, H, W)
        hidden_states = self.conv_in(hidden_states)

        for down_block in self.down_blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(down_block, hidden_states)
            else:
                hidden_states = down_block(hidden_states)

        hidden_states = self.mid_block(hidden_states)

        if self.spatial_group_norm:
            batch_size = hidden_states.size(0)
            hidden_states = hidden_states.permute(0, 2, 1, 3, 4).flatten(0, 1)
            hidden_states = self.conv_norm_out(hidden_states)
            hidden_states = hidden_states.unflatten(0, (batch_size, -1)).permute(0, 2, 1, 3, 4)
        else:
            hidden_states = self.conv_norm_out(hidden_states)

        hidden_states = self.conv_act(hidden_states)
        hidden_states = self.conv_out(hidden_states)
        return hidden_states


class EasyAnimateDecoder(nn.Module):
    r"""
    Causal decoder for 3D video-like data used in [EasyAnimate](https://huggingface.co/papers/2405.18991).
    """

    _supports_gradient_checkpointing = True

    def __init__(
        self,
        in_channels: int = 8,
        out_channels: int = 3,
        up_block_types: Tuple[str, ...] = (
            "SpatialUpBlock3D",
            "SpatialTemporalUpBlock3D",
            "SpatialTemporalUpBlock3D",
            "SpatialTemporalUpBlock3D",
        ),
        block_out_channels: Tuple[int, ...] = [128, 256, 512, 512],
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        act_fn: str = "silu",
        spatial_group_norm: bool = False,
    ):
        super().__init__()

        # 1. Input convolution
        self.conv_in = EasyAnimateCausalConv3d(in_channels, block_out_channels[-1], kernel_size=3)

        # 2. Middle block
        self.mid_block = EasyAnimateMidBlock3d(
            in_channels=block_out_channels[-1],
            num_layers=layers_per_block,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
            norm_eps=1e-6,
            dropout=0,
            output_scale_factor=1,
        )

        # 3. Up blocks
        self.up_blocks = nn.ModuleList([])
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channels = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            input_channels = output_channels
            output_channels = reversed_block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            # Create and append up block to up_blocks
            if up_block_type == "SpatialUpBlock3D":
                up_block = EasyAnimateUpBlock3d(
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
                up_block = EasyAnimateUpBlock3d(
                    in_channels=input_channels,
                    out_channels=output_channels,
                    num_layers=layers_per_block + 1,
                    act_fn=act_fn,
                    norm_num_groups=norm_num_groups,
                    norm_eps=1e-6,
                    spatial_group_norm=spatial_group_norm,
                    add_upsample=not is_final_block,
                    add_temporal_upsample=True,
                )
            else:
                raise ValueError(f"Unknown up block type: {up_block_type}")
            self.up_blocks.append(up_block)

        # Output normalization and activation
        self.spatial_group_norm = spatial_group_norm
        self.conv_norm_out = nn.GroupNorm(
            num_channels=block_out_channels[0],
            num_groups=norm_num_groups,
            eps=1e-6,
        )
        self.conv_act = get_activation(act_fn)

        # Output convolution layer
        self.conv_out = EasyAnimateCausalConv3d(block_out_channels[0], out_channels, kernel_size=3)

        self.gradient_checkpointing = False

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states: (B, C, T, H, W)
        hidden_states = self.conv_in(hidden_states)

        if torch.is_grad_enabled() and self.gradient_checkpointing:
            hidden_states = self._gradient_checkpointing_func(self.mid_block, hidden_states)
        else:
            hidden_states = self.mid_block(hidden_states)

        for up_block in self.up_blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(up_block, hidden_states)
            else:
                hidden_states = up_block(hidden_states)

        if self.spatial_group_norm:
            batch_size = hidden_states.size(0)
            hidden_states = hidden_states.permute(0, 2, 1, 3, 4).flatten(0, 1)  # [B, C, T, H, W] -> [B * T, C, H, W]
            hidden_states = self.conv_norm_out(hidden_states)
            hidden_states = hidden_states.unflatten(0, (batch_size, -1)).permute(
                0, 2, 1, 3, 4
            )  # [B * T, C, H, W] -> [B, C, T, H, W]
        else:
            hidden_states = self.conv_norm_out(hidden_states)

        hidden_states = self.conv_act(hidden_states)
        hidden_states = self.conv_out(hidden_states)
        return hidden_states


class AutoencoderKLMagvit(ModelMixin, AutoencoderMixin, ConfigMixin):
    r"""
    A VAE model with KL loss for encoding images into latents and decoding latent representations into images. This
    model is used in [EasyAnimate](https://huggingface.co/papers/2405.18991).

    This model inherits from [`ModelMixin`]. Check the superclass documentation for it's generic methods implemented
    for all models (such as downloading or saving).
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 16,
        out_channels: int = 3,
        block_out_channels: Tuple[int, ...] = [128, 256, 512, 512],
        down_block_types: Tuple[str, ...] = [
            "SpatialDownBlock3D",
            "SpatialTemporalDownBlock3D",
            "SpatialTemporalDownBlock3D",
            "SpatialTemporalDownBlock3D",
        ],
        up_block_types: Tuple[str, ...] = [
            "SpatialUpBlock3D",
            "SpatialTemporalUpBlock3D",
            "SpatialTemporalUpBlock3D",
            "SpatialTemporalUpBlock3D",
        ],
        layers_per_block: int = 2,
        act_fn: str = "silu",
        norm_num_groups: int = 32,
        scaling_factor: float = 0.7125,
        spatial_group_norm: bool = True,
    ):
        super().__init__()

        # Initialize the encoder
        self.encoder = EasyAnimateEncoder(
            in_channels=in_channels,
            out_channels=latent_channels,
            down_block_types=down_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            norm_num_groups=norm_num_groups,
            act_fn=act_fn,
            double_z=True,
            spatial_group_norm=spatial_group_norm,
        )

        # Initialize the decoder
        self.decoder = EasyAnimateDecoder(
            in_channels=latent_channels,
            out_channels=out_channels,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            norm_num_groups=norm_num_groups,
            act_fn=act_fn,
            spatial_group_norm=spatial_group_norm,
        )

        # Initialize convolution layers for quantization and post-quantization
        self.quant_conv = nn.Conv3d(2 * latent_channels, 2 * latent_channels, kernel_size=1)
        self.post_quant_conv = nn.Conv3d(latent_channels, latent_channels, kernel_size=1)

        self.spatial_compression_ratio = 2 ** (len(block_out_channels) - 1)
        self.temporal_compression_ratio = 2 ** (len(block_out_channels) - 2)

        # When decoding a batch of video latents at a time, one can save memory by slicing across the batch dimension
        # to perform decoding of a single video latent at a time.
        self.use_slicing = False

        # When decoding spatially large video latents, the memory requirement is very high. By breaking the video latent
        # frames spatially into smaller tiles and performing multiple forward passes for decoding, and then blending the
        # intermediate tiles together, the memory requirement can be lowered.
        self.use_tiling = False

        # When decoding temporally long video latents, the memory requirement is very high. By decoding latent frames
        # at a fixed frame batch size (based on `self.num_latent_frames_batch_size`), the memory requirement can be lowered.
        self.use_framewise_encoding = False
        self.use_framewise_decoding = False

        # Assign mini-batch sizes for encoder and decoder
        self.num_sample_frames_batch_size = 4
        self.num_latent_frames_batch_size = 1

        # The minimal tile height and width for spatial tiling to be used
        self.tile_sample_min_height = 512
        self.tile_sample_min_width = 512
        self.tile_sample_min_num_frames = 4

        # The minimal distance between two spatial tiles
        self.tile_sample_stride_height = 448
        self.tile_sample_stride_width = 448
        self.tile_sample_stride_num_frames = 8

    def _clear_conv_cache(self):
        # Clear cache for convolutional layers if needed
        for name, module in self.named_modules():
            if isinstance(module, EasyAnimateCausalConv3d):
                module._clear_conv_cache()
            if isinstance(module, EasyAnimateUpsampler3D):
                module._clear_conv_cache()

    def enable_tiling(
        self,
        tile_sample_min_height: Optional[int] = None,
        tile_sample_min_width: Optional[int] = None,
        tile_sample_min_num_frames: Optional[int] = None,
        tile_sample_stride_height: Optional[float] = None,
        tile_sample_stride_width: Optional[float] = None,
        tile_sample_stride_num_frames: Optional[float] = None,
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
            tile_sample_stride_height (`int`, *optional*):
                The minimum amount of overlap between two consecutive vertical tiles. This is to ensure that there are
                no tiling artifacts produced across the height dimension.
            tile_sample_stride_width (`int`, *optional*):
                The stride between two consecutive horizontal tiles. This is to ensure that there are no tiling
                artifacts produced across the width dimension.
        """
        self.use_tiling = True
        self.use_framewise_decoding = True
        self.use_framewise_encoding = True
        self.tile_sample_min_height = tile_sample_min_height or self.tile_sample_min_height
        self.tile_sample_min_width = tile_sample_min_width or self.tile_sample_min_width
        self.tile_sample_min_num_frames = tile_sample_min_num_frames or self.tile_sample_min_num_frames
        self.tile_sample_stride_height = tile_sample_stride_height or self.tile_sample_stride_height
        self.tile_sample_stride_width = tile_sample_stride_width or self.tile_sample_stride_width
        self.tile_sample_stride_num_frames = tile_sample_stride_num_frames or self.tile_sample_stride_num_frames

    @apply_forward_hook
    def _encode(
        self, x: torch.Tensor, return_dict: bool = True
    ) -> Union[AutoencoderKLOutput, Tuple[DiagonalGaussianDistribution]]:
        """
        Encode a batch of images into latents.

        Args:
            x (`torch.Tensor`): Input batch of images.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`~models.autoencoder_kl.AutoencoderKLOutput`] instead of a plain tuple.

        Returns:
                The latent representations of the encoded images. If `return_dict` is True, a
                [`~models.autoencoder_kl.AutoencoderKLOutput`] is returned, otherwise a plain `tuple` is returned.
        """
        if self.use_tiling and (x.shape[-1] > self.tile_sample_min_height or x.shape[-2] > self.tile_sample_min_width):
            return self.tiled_encode(x, return_dict=return_dict)

        first_frames = self.encoder(x[:, :, :1, :, :])
        h = [first_frames]
        for i in range(1, x.shape[2], self.num_sample_frames_batch_size):
            next_frames = self.encoder(x[:, :, i : i + self.num_sample_frames_batch_size, :, :])
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

    def _decode(self, z: torch.Tensor, return_dict: bool = True) -> Union[DecoderOutput, torch.Tensor]:
        batch_size, num_channels, num_frames, height, width = z.shape
        tile_latent_min_height = self.tile_sample_min_height // self.spatial_compression_ratio
        tile_latent_min_width = self.tile_sample_min_width // self.spatial_compression_ratio

        if self.use_tiling and (z.shape[-1] > tile_latent_min_height or z.shape[-2] > tile_latent_min_width):
            return self.tiled_decode(z, return_dict=return_dict)

        z = self.post_quant_conv(z)

        # Process the first frame and save the result
        first_frames = self.decoder(z[:, :, :1, :, :])
        # Initialize the list to store the processed frames, starting with the first frame
        dec = [first_frames]
        # Process the remaining frames, with the number of frames processed at a time determined by mini_batch_decoder
        for i in range(1, z.shape[2], self.num_latent_frames_batch_size):
            next_frames = self.decoder(z[:, :, i : i + self.num_latent_frames_batch_size, :, :])
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

    def tiled_encode(self, x: torch.Tensor, return_dict: bool = True) -> AutoencoderKLOutput:
        batch_size, num_channels, num_frames, height, width = x.shape
        latent_height = height // self.spatial_compression_ratio
        latent_width = width // self.spatial_compression_ratio

        tile_latent_min_height = self.tile_sample_min_height // self.spatial_compression_ratio
        tile_latent_min_width = self.tile_sample_min_width // self.spatial_compression_ratio
        tile_latent_stride_height = self.tile_sample_stride_height // self.spatial_compression_ratio
        tile_latent_stride_width = self.tile_sample_stride_width // self.spatial_compression_ratio

        blend_height = tile_latent_min_height - tile_latent_stride_height
        blend_width = tile_latent_min_width - tile_latent_stride_width

        # Split the image into 512x512 tiles and encode them separately.
        rows = []
        for i in range(0, height, self.tile_sample_stride_height):
            row = []
            for j in range(0, width, self.tile_sample_stride_width):
                tile = x[
                    :,
                    :,
                    :,
                    i : i + self.tile_sample_min_height,
                    j : j + self.tile_sample_min_width,
                ]

                first_frames = self.encoder(tile[:, :, 0:1, :, :])
                tile_h = [first_frames]
                for k in range(1, num_frames, self.num_sample_frames_batch_size):
                    next_frames = self.encoder(tile[:, :, k : k + self.num_sample_frames_batch_size, :, :])
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
                    tile = self.blend_v(rows[i - 1][j], tile, blend_height)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_width)
                result_row.append(tile[:, :, :, :latent_height, :latent_width])
            result_rows.append(torch.cat(result_row, dim=4))

        moments = torch.cat(result_rows, dim=3)[:, :, :, :latent_height, :latent_width]
        return moments

    def tiled_decode(self, z: torch.Tensor, return_dict: bool = True) -> Union[DecoderOutput, torch.Tensor]:
        batch_size, num_channels, num_frames, height, width = z.shape
        sample_height = height * self.spatial_compression_ratio
        sample_width = width * self.spatial_compression_ratio

        tile_latent_min_height = self.tile_sample_min_height // self.spatial_compression_ratio
        tile_latent_min_width = self.tile_sample_min_width // self.spatial_compression_ratio
        tile_latent_stride_height = self.tile_sample_stride_height // self.spatial_compression_ratio
        tile_latent_stride_width = self.tile_sample_stride_width // self.spatial_compression_ratio

        blend_height = self.tile_sample_min_height - self.tile_sample_stride_height
        blend_width = self.tile_sample_min_width - self.tile_sample_stride_width

        # Split z into overlapping 64x64 tiles and decode them separately.
        # The tiles have an overlap to avoid seams between tiles.
        rows = []
        for i in range(0, height, tile_latent_stride_height):
            row = []
            for j in range(0, width, tile_latent_stride_width):
                tile = z[
                    :,
                    :,
                    :,
                    i : i + tile_latent_min_height,
                    j : j + tile_latent_min_width,
                ]
                tile = self.post_quant_conv(tile)

                # Process the first frame and save the result
                first_frames = self.decoder(tile[:, :, :1, :, :])
                # Initialize the list to store the processed frames, starting with the first frame
                tile_dec = [first_frames]
                # Process the remaining frames, with the number of frames processed at a time determined by mini_batch_decoder
                for k in range(1, num_frames, self.num_latent_frames_batch_size):
                    next_frames = self.decoder(tile[:, :, k : k + self.num_latent_frames_batch_size, :, :])
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
                    tile = self.blend_v(rows[i - 1][j], tile, blend_height)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_width)
                result_row.append(tile[:, :, :, : self.tile_sample_stride_height, : self.tile_sample_stride_width])
            result_rows.append(torch.cat(result_row, dim=4))

        dec = torch.cat(result_rows, dim=3)[:, :, :, :sample_height, :sample_width]

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)

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
