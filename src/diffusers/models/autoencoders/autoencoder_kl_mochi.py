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


class MochiCausalConv3d(nn.Module):
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
        self.time_kernel_size = time_kernel_size



    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        context_size = self.time_kernel_size - 1
        time_casual_padding = (0, 0, 0, 0, context_size, 0)

        inputs = F.pad(inputs, time_casual_padding, mode=self.padding_mode)
        
        # Memory-efficient chunked operation
        memory_count = torch.prod(torch.tensor(inputs.shape)).item() * 2 / 1024**3
        if memory_count > 2:
            part_num = int(memory_count / 2) + 1
            k = self.time_kernel_size
            input_idx = torch.arange(context_size, inputs.size(2))
            input_chunks_idx = torch.split(input_idx, input_idx.size(0) // part_num)

            # Compute output size
            B, _, T_in, H_in, W_in = inputs.shape
            output_size = (
                B,
                self.conv.out_channels,
                T_in - k + 1,
                H_in // self.conv.stride[1],
                W_in // self.conv.stride[2],
            )
            output = torch.empty(output_size, dtype=inputs.dtype, device=inputs.device)
            for input_chunk_idx in input_chunks_idx:
                input_s = input_chunk_idx[0] - k + 1
                input_e = input_chunk_idx[-1] + 1
                input_chunk = inputs[:, :, input_s:input_e, :, :]
                output_chunk = self.conv(input_chunk)

                output_s = input_s
                output_e = output_s + output_chunk.size(2)
                output[:, :, output_s:output_e, :, :] = output_chunk

            return output
        else:
            return self.conv(inputs)


class MochiGroupNorm3D(nn.Module):
    r"""
    Group normalization applied per-frame.

    Args:

    """

    def __init__(
        self,
        chunk_size: int = 8,
    ):
        super().__init__()
        self.norm_layer = nn.GroupNorm()
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


