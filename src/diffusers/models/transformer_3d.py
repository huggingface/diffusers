# Based on the TuneAVideo Transformer3DModel from Showlab: https://arxiv.org/abs/2212.11565
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

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from ..configuration_utils import ConfigMixin, register_to_config
from ..utils import BaseOutput
from .attention import Attention, FeedForward
from .modeling_utils import ModelMixin


@dataclass
class Transformer3DModelOutput(BaseOutput):
    """
    Args:
        sample (`torch.FloatTensor` of shape `(batch_size, num_channels, num_frames, height, width)`.
    """

    sample: torch.FloatTensor


class Transformer3DModel(ModelMixin, ConfigMixin):
    """
    Transformer model for a video-like data.

    When input is continuous: First, project the input (aka embedding) and reshape to b, h * w, c. Then apply the
    sparse 3d transformer action. Finally, reshape to video again.

    Parameters:
        num_attention_heads (`int`, *optional*, defaults to 16):
            The number of heads to use for multi-head attention.
        attention_head_dim (`int`, *optional*, defaults to 88):
            The number of channels in each head.
        in_channels (`int`, *optional*):
            Pass if the input is continuous. The number of channels in the input and output.
        num_layers (`int`, *optional*, defaults to 1):
            The number of layers of Transformer blocks to use.
        dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability to use.
        norm_num_groups: (`int`, *optional*, defaults to 32):
            The number of norm groups for the group norm.
        cross_attention_dim (`int`, *optional*):
            The number of encoder_hidden_states dimensions to use.
        activation_fn (`str`, *optional*, defaults to `"geglu"`):
            Activation function to be used in feed-forward.

    """

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        in_channels: Optional[int] = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        cross_attention_dim: int = 1280,
        activation_fn: str = "geglu",
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        inner_dim = num_attention_heads * attention_head_dim

        # Define input layers
        self.in_channels = in_channels

        self.norm = torch.nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=1e-6, affine=True)
        self.proj_in = nn.Conv2d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)

        # Define transformers blocks
        self.transformer_blocks = nn.ModuleList(
            [
                BasicSparse3DTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    cross_attention_dim=cross_attention_dim,
                    activation_fn=activation_fn,
                )
                for d in range(num_layers)
            ]
        )

        # 4. Define output layers
        self.proj_out = nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        timestep=None,
        return_dict: bool = True,
    ):
        """
        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch size, channel, num_frames, height, width)`):
                Input hidden_states
            encoder_hidden_states ( `torch.LongTensor` of shape `(batch size, encoder_hidden_states dim)`, *optional*):
                Conditional embeddings for cross attention layer. If not given, cross-attention defaults to
                self-attention.
            timestep ( `torch.long`, *optional*):
                Optional timestep to be applied as an embedding in AdaLayerNorm's. Used to indicate denoising step.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformer_3d.Transformer3DModelOutput`] instead of a plain
                tuple.

        Returns:
            [`~models.transformer_3d.Transformer3DModelOutput`] or `tuple`:
            [`~models.transformer_3d.Transformer3DModelOutput`] if `return_dict` is True, otherwise a `tuple`. When
            returning a tuple, the first element is the sample tensor.
        """
        # Input
        if not hidden_states.dim() == 5:
            raise ValueError(f"Expected hidden_states to have ndim=5, but got ndim={hidden_states.dim()}.")
        video_length = hidden_states.shape[2]
        hidden_states = hidden_states.movedim((0, 1, 2, 3, 4), (0, 2, 1, 3, 4))
        hidden_states = hidden_states.flatten(0, 1)
        encoder_hidden_states = encoder_hidden_states.repeat_interleave(repeats=video_length, dim=0)

        batch, channel, height, weight = hidden_states.shape
        residual = hidden_states

        hidden_states = self.norm(hidden_states)
        hidden_states = self.proj_in(hidden_states)
        inner_dim = hidden_states.shape[1]
        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * weight, inner_dim)

        # Blocks
        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                video_length=video_length,
            )

        # Output
        hidden_states = hidden_states.reshape(batch, height, weight, inner_dim).permute(0, 3, 1, 2).contiguous()
        hidden_states = self.proj_out(hidden_states)

        output = hidden_states + residual

        # output = rearrange(output, "(b f) c h w -> b c f h w", f=video_length)
        output = output.reshape([-1, video_length, *output.shape[1:]])
        output = output.movedim((0, 1, 2, 3, 4), (0, 2, 1, 3, 4))
        if not return_dict:
            return (output,)

        return Transformer3DModelOutput(sample=output)


class BasicSparse3DTransformerBlock(nn.Module):
    r"""
    A modified basic Transformer block designed for use with Text to Video models. Currently only used by Tune A Video
    pipeline with attn1 processor set to the TuneAVideoAttnProcessor.

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        cross_attention_dim (`int`, *optional*): The size of the encoder_hidden_states vector for cross attention.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        cross_attention_dim: int = 1280,
        activation_fn: str = "geglu",
    ):
        super().__init__()

        # Temporal-Attention.
        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=False,
            cross_attention_dim=None,
            upcast_attention=False,
        )
        self.norm1 = nn.LayerNorm(dim)

        # Cross-Attn
        self.attn2 = Attention(
            query_dim=dim,
            cross_attention_dim=cross_attention_dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=False,
            upcast_attention=False,
        )

        self.norm2 = nn.LayerNorm(dim)

        # Feed-forward
        self.ff = FeedForward(dim, dropout=dropout, activation_fn=activation_fn)
        self.norm3 = nn.LayerNorm(dim)

        # Temp-Attn
        self.attn_temp = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=False,
            upcast_attention=False,
        )
        nn.init.zeros_(self.attn_temp.to_out[0].weight.data)
        self.norm_temp = nn.LayerNorm(dim)

    def forward(
        self, hidden_states, encoder_hidden_states=None, timestep=None, attention_mask=None, video_length=None
    ):
        # SparseCausal-Attention
        norm_hidden_states = self.norm1(hidden_states)

        hidden_states = (
            self.attn1(norm_hidden_states, attention_mask=attention_mask, video_length=video_length) + hidden_states
        )

        norm_hidden_states = self.norm2(hidden_states)
        hidden_states = (
            self.attn2(norm_hidden_states, encoder_hidden_states=encoder_hidden_states, attention_mask=attention_mask)
            + hidden_states
        )

        # Feed-forward
        hidden_states = self.ff(self.norm3(hidden_states)) + hidden_states

        # Temporal-Attention
        d = hidden_states.shape[1]
        # (b f) d c -> b f d c -> b d f c -> (b d) f c
        hidden_states = hidden_states.reshape([-1, video_length, *hidden_states.shape[1:]])
        hidden_states = hidden_states.movedim((0, 1, 2, 3), (0, 2, 1, 3))
        hidden_states = hidden_states.flatten(0, 1)
        norm_hidden_states = self.norm_temp(hidden_states)
        hidden_states = self.attn_temp(norm_hidden_states) + hidden_states
        # (b d) f c -> b d f c ->  b f d c -> (b f) d c
        hidden_states = hidden_states.reshape([-1, d, *hidden_states.shape[1:]])
        hidden_states = hidden_states.movedim((0, 1, 2, 3), (0, 2, 1, 3))
        hidden_states = hidden_states.flatten(0, 1)

        return hidden_states
