# Copyright 2024 The Mochi team and The HuggingFace Team.
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

from typing import Dict, Optional, Tuple, Union

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


import torch
import torch.nn as nn
import torch.nn.functional as F


# YiYi to-do: replace this with nn.Conv3d
class Conv1x1(nn.Linear):
    """*1x1 Conv implemented with a linear layer."""

    def __init__(self, in_features: int, out_features: int, *args, **kwargs):
        super().__init__(in_features, out_features, *args, **kwargs)

    def forward(self, x: torch.Tensor):
        """Forward pass.

        Args:
            x: Input tensor. Shape: [B, C, *] or [B, *, C].

        Returns:
            x: Output tensor. Shape: [B, C', *] or [B, *, C'].
        """
        x = x.movedim(1, -1)
        x = super().forward(x)
        x = x.movedim(-1, 1)
        return x
    

class MochiChunkedCausalConv3d(nn.Module):
    r"""A 3D causal convolution layer that pads the input tensor to ensure causality in CogVideoX Model.

    Args:
        in_channels (`int`): Number of channels in the input tensor.
        out_channels (`int`): Number of output channels produced by the convolution.
        kernel_size (`int` or `Tuple[int, int, int]`): Kernel size of the convolutional kernel.
        stride (`int` or `Tuple[int, int, int]`, defaults to `1`): Stride of the convolution.
        pad_mode (`str`, defaults to `"constant"`): Padding mode.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: Union[int, Tuple[int, int, int]],
        padding_mode: str = "replicate",
    ):
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3
        if isinstance(stride, int):
            stride = (stride,) * 3

        time_kernel_size, height_kernel_size, width_kernel_size = kernel_size

        self.padding_mode = padding_mode
        height_pad = (height_kernel_size - 1) // 2
        width_pad = (width_kernel_size - 1) // 2

        self.conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=(1, 1, 1),
            padding=(0, height_pad, width_pad),
            padding_mode=padding_mode,
        )



    def forward(self, hidden_states : torch.Tensor) -> torch.Tensor:
        time_kernel_size = self.conv.kernel_size[0]
        context_size = time_kernel_size - 1
        time_casual_padding = (0, 0, 0, 0, context_size, 0)
        hidden_states = F.pad(hidden_states, time_casual_padding, mode=self.padding_mode)
        
        # Memory-efficient chunked operation
        memory_count = torch.prod(torch.tensor(hidden_states.shape)).item() * 2 / 1024**3
        # YiYI Notes: testing only!! please remove
        memory_count = 3
        if memory_count > 2:
            part_num = int(memory_count / 2) + 1
            num_frames = hidden_states.shape[2]
            frames_idx = torch.arange(context_size, num_frames)
            frames_chunks_idx = torch.chunk(frames_idx, part_num, dim=0)

            output_chunks = []
            for frames_chunk_idx in frames_chunks_idx:
                frames_s = frames_chunk_idx[0] - context_size
                frames_e = frames_chunk_idx[-1] + 1
                frames_chunk = hidden_states[:, :, frames_s:frames_e, :, :]
                output_chunk = self.conv(frames_chunk)
                output_chunks.append(output_chunk)  # Append each output chunk to the list

            # Concatenate all output chunks along the temporal dimension
            hidden_states = torch.cat(output_chunks, dim=2) 

            return hidden_states
        else:
            return self.conv(hidden_states)


class MochiChunkedGroupNorm3D(nn.Module):
    r"""
    Group normalization applied per-frame.

    Args:

    """

    def __init__(
        self,
        num_channels: int,
        num_groups: int = 32,
        affine: bool = True,
        chunk_size: int = 8,
    ):
        super().__init__()
        self.norm_layer = nn.GroupNorm(num_channels=num_channels, num_groups=num_groups, affine=affine)
        self.chunk_size = chunk_size

    def forward(
        self, x: torch.Tensor = None
    ) -> torch.Tensor:

        batch_size, channels, num_frames, height, width = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, channels, height, width)
        
        num_chunks = (batch_size * num_frames + self.chunk_size - 1) // self.chunk_size
        
        output = torch.cat(
            [self.norm_layer(chunk) for chunk in x.split(self.chunk_size, dim=0)],
            dim=0
            )
        output = output.view(batch_size, num_frames, channels, height, width).permute(0, 2, 1, 3, 4)
        
        return output


class MochiResnetBlock3D(nn.Module):
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
        non_linearity: str = "swish",
    ):
        super().__init__()

        out_channels = out_channels or in_channels

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nonlinearity = get_activation(non_linearity)

        self.norm1 = MochiChunkedGroupNorm3D(num_channels=in_channels)
        self.conv1 = MochiChunkedCausalConv3d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1
        )
        self.norm2 = MochiChunkedGroupNorm3D(num_channels=out_channels)
        self.conv2 = MochiChunkedCausalConv3d(
            in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1
        )


    def forward(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:

        hidden_states = inputs
       
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv1(hidden_states)

        hidden_states = self.norm2(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv2(hidden_states)

        hidden_states = hidden_states + inputs
        return hidden_states


class MochiUpBlock3D(nn.Module):
    r"""
    An upsampling block used in the Mochi model.

    Args:
        in_channels (`int`):
            Number of input channels.
        out_channels (`int`, *optional*):
            Number of output channels. If None, defaults to `in_channels`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 1,
        temporal_expansion: int = 2,
        spatial_expansion: int = 2,
    ):
        super().__init__()
        self.temporal_expansion = temporal_expansion
        self.spatial_expansion = spatial_expansion

        resnets = []
        for i in range(num_layers):
            resnets.append(
                MochiResnetBlock3D(
                    in_channels=in_channels,
                )
            )

        self.resnets = nn.ModuleList(resnets)
        self.proj = Conv1x1(
            in_channels,
            out_channels * temporal_expansion * (spatial_expansion**2),
        )

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        r"""Forward method of the `MochiUpBlock3D` class."""

        for i, resnet in enumerate(self.resnets):

            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def create_forward(*inputs):
                        return module(*inputs)

                    return create_forward

                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(resnet),
                    hidden_states,
                )
            else:
                hidden_states = resnet(hidden_states)

        hidden_states = self.proj(hidden_states)

        # Calculate new shape
        B, C, T, H, W = hidden_states.shape
        st = self.temporal_expansion
        sh = self.spatial_expansion
        sw = self.spatial_expansion
        new_C = C // (st * sh * sw)

        # Reshape and permute
        hidden_states = hidden_states.view(B, new_C, st, sh, sw, T, H, W)
        hidden_states = hidden_states.permute(0, 1, 5, 2, 6, 3, 7, 4)
        hidden_states = hidden_states.contiguous().view(B, new_C, T * st, H * sh, W * sw)

        if self.temporal_expansion > 1:
            print(f"x: {hidden_states.shape}")
            # Drop the first self.temporal_expansion - 1 frames.
            hidden_states = hidden_states[:, :, self.temporal_expansion - 1 :]
            print(f"x: {hidden_states.shape}")

        return hidden_states


class MochiMidBlock3D(nn.Module):
    r"""
    A middle block used in the Mochi model.

    Args:
        in_channels (`int`):
            Number of input channels.
    """

    _supports_gradient_checkpointing = True

    def __init__(
        self,
        in_channels: int, # 768
        num_layers: int = 3,
    ):
        super().__init__()

        resnets = []
        for _ in range(num_layers):
            resnets.append(
                MochiResnetBlock3D(in_channels=in_channels)
            )
        self.resnets = nn.ModuleList(resnets)

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        r"""Forward method of the `MochiMidBlock3D` class."""

        for i, resnet in enumerate(self.resnets):
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def create_forward(*inputs):
                        return module(*inputs)

                    return create_forward

                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(resnet), hidden_states
                )
            else:
                hidden_states = resnet(hidden_states)

        return hidden_states


class MochiDecoder3D(nn.Module):
    _supports_gradient_checkpointing = True

    def __init__(
        self,
        in_channels: int, # 12
        out_channels: int, # 3
        block_out_channels: Tuple[int, ...] = (128, 256, 512, 768),
        layers_per_block: Tuple[int, ...] = (3, 3, 4, 6, 3),
        temporal_expansions: Tuple[int, ...] = (1, 2, 3),
        spatial_expansions: Tuple[int, ...] = (2, 2, 2),
        non_linearity: str = "swish",
    ):
        super().__init__()

        self.nonlinearity = get_activation(non_linearity)

        self.conv_in = nn.Conv3d(in_channels, block_out_channels[-1], kernel_size=(1, 1, 1))
        self.block_in = MochiMidBlock3D(
            in_channels=block_out_channels[-1],
            num_layers=layers_per_block[-1],
        )
        self.up_blocks = nn.ModuleList([])
        for i in range(len(block_out_channels) - 1):
            up_block = MochiUpBlock3D(
                in_channels=block_out_channels[-i - 1],
                out_channels=block_out_channels[-i - 2],
                num_layers=layers_per_block[-i - 2],
                temporal_expansion=temporal_expansions[-i - 1],
                spatial_expansion=spatial_expansions[-i - 1],
            )
            self.up_blocks.append(up_block)
        self.block_out = MochiMidBlock3D(
            in_channels=block_out_channels[0],
            num_layers=layers_per_block[0],
        )
        self.conv_out = Conv1x1(block_out_channels[0], out_channels)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        r"""Forward method of the `MochiDecoder3D` class."""

        print(f"hidden_states: {hidden_states.shape}, {hidden_states[0,:3,0,:2,:2]}")
        hidden_states = self.conv_in(hidden_states)
        print(f"hidden_states (after conv_in): {hidden_states.shape}, {hidden_states[0,:3,0,:2,:2]}")


        # 1. Mid
        hidden_states = self.block_in(hidden_states)
        print(f"hidden_states (after block_in): {hidden_states.shape}, {hidden_states[0,:3,0,:2,:2]}")
        # 2. Up
        for i, up_block in enumerate(self.up_blocks):
            hidden_states = up_block(hidden_states)
            print(f"hidden_states (after up_block {i}): {hidden_states.shape}, {hidden_states[0,:3,0,:2,:2]}")
        # 3. Post-process
        hidden_states = self.block_out(hidden_states)
        print(f"hidden_states (after block_out): {hidden_states.shape}, {hidden_states[0,:3,0,:2,:2]}")
        
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv_out(hidden_states)
        print(f"hidden_states (after conv_out): {hidden_states.shape}, {hidden_states[0,:3,0,:2,:2]}")

        return hidden_states
        