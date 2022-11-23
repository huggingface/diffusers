#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

"""
Implementations are translated from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention.py.
"""

from typing import Optional

from aitemplate.compiler.ops import reshape
from aitemplate.frontend import nn, Tensor


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other. Originally ported from here, but adapted
    to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    Uses three q, k, v linear layers to compute attention.
    Parameters:
        batch_size (:obj:`int`): The number of examples per batch.
        height (:obj:`int`): Height of each image example.
        width (:obj:`int`): Width of each image example.
        channels (:obj:`int`): The number of channels in the input and output.
        num_head_channels (:obj:`int`, *optional*):
            The number of channels in each head. If None, then `num_heads` = 1.
        num_groups (:obj:`int`, *optional*, defaults to 32): The number of groups to use for group norm.
        eps (:obj:`float`, *optional*, defaults to 1e-5): The epsilon value to use for group norm.
    """

    def __init__(
        self,
        batch_size: int,
        height: int,
        width: int,
        channels: int,
        num_head_channels: Optional[int] = None,
        num_groups: int = 32,
        rescale_output_factor: float = 1.0,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.channels = channels
        self.num_heads = (
            channels // num_head_channels if num_head_channels is not None else 1
        )
        self.num_head_size = num_head_channels
        self.group_norm = nn.GroupNorm(num_groups, channels, eps)
        self.attention = nn.MultiheadAttention(
            channels,
            batch_size,
            height * width,
            self.num_heads,
            qkv_bias=True,
            has_residual=True,
            use_mem_eff=True,
        )
        self.rescale_output_factor = rescale_output_factor

    def forward(self, hidden_states) -> Tensor:
        """
        input hidden_states shape: [batch, height, width, channel]
        output shape: [batch, height, width, channel]
        """
        residual = hidden_states

        # norm
        hidden_states = self.group_norm(hidden_states)

        hidden_states = reshape()(
            hidden_states, [self.batch_size, self.height * self.width, self.channels]
        )

        batch, hw, channel = hidden_states.shape()
        if (
            batch.value() != self.batch_size
            or hw.value() != self.width * self.height
            or channel.value() != self.channels
        ):
            raise RuntimeError(
                "nchw params do not match! "
                f"Expected: {self.batch_size}, {self.channels}, {self.height} * {self.width}, "
                f"actual: {batch}, {channel}, {hw}."
            )

        res = self.attention(hidden_states, residual) * (1 / self.rescale_output_factor)
        res = reshape()(res, [self.batch_size, self.height, self.width, self.channels])

        return res
