# Copyright 2022 The HuggingFace Team. All rights reserved.
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
from torch import nn

from .attention import AttentionBlock
from .resnet import ResnetBlock2D


class UNetMidBlock(nn.Module):
    def __init__(
        self, in_channels: int, temb_channels: int, dropout: float, overwrite_qkv=False, overwrite_unet=False
    ):
        super().__init__()

        self.resnet_1 = ResnetBlock2D(
            in_channels=in_channels, out_channels=in_channels, temb_channels=temb_channels, dropout=dropout
        )
        self.attn = AttentionBlock(in_channels, overwrite_qkv=overwrite_qkv)
        self.resnet_2 = ResnetBlock2D(
            in_channels=in_channels, out_channels=in_channels, temb_channels=temb_channels, dropout=dropout
        )

        self.is_overwritten = False
        self.overwrite_unet = overwrite_unet
        if self.overwrite_unet:
            block_in = in_channels
            self.temb_ch = temb_channels
            self.block_1 = ResnetBlock2D(
                in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout
            )
            self.attn_1 = AttentionBlock(block_in, overwrite_qkv=True)
            self.block_2 = ResnetBlock2D(
                in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout
            )

    def forward(self, hidden_states, temb=None):
        if self.is_overwrite_unet:
            self.resnet_1 = self.block_1
            self.attn = self.attn_1
            self.resnet_2 = self.block_2

        hidden_states = self.resnet_1(hidden_states, temb)
        hidden_states = self.attn(hidden_states)
        hidden_states = self.resnet_2(hidden_states, temb)
        return hidden_states
