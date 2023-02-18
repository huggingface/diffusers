# Copyright 2023 The HuggingFace Team. All rights reserved.
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

r"""
ControlNet Block Implementation

Paper: Adding Conditional Control to Text-to-Image Diffusion Models. https://arxiv.org/abs/2302.05543

Reference implementation: https://github.com/lllyasviel/ControlNet
"""

from typing import List, Tuple

import torch
import torch.nn as nn


def set_zero_parameters(module):
    for p in module.parameters():
        p.detach().zero_()
    return module


# ControlNet: Zero Convolution
def zero_conv(channels):
    return set_zero_parameters(nn.Conv2d(channels, channels, 1, padding=0))


class ControlNetInputHintBlock(nn.Module):
    def __init__(self, hint_channels: int = 3, channels: int = 320):
        super().__init__()
        #  Layer configurations are from reference implementation.
        self.input_hint_block = nn.Sequential(
            nn.Conv2d(hint_channels, 16, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 32, 3, padding=1, stride=2),
            nn.SiLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 96, 3, padding=1, stride=2),
            nn.SiLU(),
            nn.Conv2d(96, 96, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(96, 256, 3, padding=1, stride=2),
            nn.SiLU(),
            set_zero_parameters(nn.Conv2d(256, channels, 3, padding=1)),
        )

    def forward(self, hint: torch.Tensor):
        return self.input_hint_block(hint)


class ControlNetZeroConvBlock(nn.Module):
    def __init__(
        self,
        block_out_channels: Tuple[int] = (320, 640, 1280, 1280),
        down_block_types: Tuple[str] = (
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        ),
        layers_per_block: int = 2,
    ):
        super().__init__()
        self.input_zero_conv = zero_conv(block_out_channels[0])
        zero_convs = []
        for i, down_block_type in enumerate(down_block_types):
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1
            for _ in range(layers_per_block):
                zero_convs.append(zero_conv(output_channel))
            if not is_final_block:
                zero_convs.append(zero_conv(output_channel))
        self.zero_convs = nn.ModuleList(zero_convs)
        self.mid_zero_conv = zero_conv(block_out_channels[-1])

    def forward(
        self,
        down_block_res_samples: List[torch.Tensor],
        mid_block_sample: torch.Tensor,
    ) -> List[torch.Tensor]:
        outputs = []
        outputs.append(self.input_zero_conv(down_block_res_samples[0]))
        for res_sample, zero_conv in zip(down_block_res_samples[1:], self.zero_convs):
            outputs.append(zero_conv(res_sample))
        outputs.append(self.mid_zero_conv(mid_block_sample))
        return outputs
