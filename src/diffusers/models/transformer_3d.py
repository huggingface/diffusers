# TODO: Check if showlab/TuneAVideo needs to be credited here.
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
from .attention import BasicSparseTransformerBlock
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
        attention_bias (`bool`, *optional*, defaults to False):
            Configure if the TransformerBlocks' attention should contain a bias parameter.
        activation_fn (`str`, *optional*, defaults to `"geglu"`):
            Activation function to be used in feed-forward.
        num_embeds_ada_norm ( `int`, *optional*): Pass if at least one of the norm_layers is `AdaLayerNorm`.
            The number of diffusion steps used during training. Note that this is fixed at training time as it is used
            to learn a number of embeddings that are added to the hidden states. During inference, you can denoise for
            up to but not more than steps than `num_embeds_ada_norm`.
        use_linear_projection: ( `bool`, *optional*, defaults to False):
            Pass True if linear projection is to be applied on the input hidden_states. If False, uses Conv2D instead.
        only_cross_attention: ( `bool`, *optional*, defaults to False):
            Input to the attention processor.
        upcast_attention: ( `bool`, *optional*, defaults to False),
            Input to the attention processor.

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
        cross_attention_dim: Optional[int] = None,
        attention_bias: bool = False,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        use_linear_projection: bool = False,
        only_cross_attention: bool = False,
        upcast_attention: bool = False,
    ):
        super().__init__()
        self.use_linear_projection = use_linear_projection
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        inner_dim = num_attention_heads * attention_head_dim

        # Define input layers
        self.in_channels = in_channels

        self.norm = torch.nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=1e-6, affine=True)
        if use_linear_projection:
            self.proj_in = nn.Linear(in_channels, inner_dim)
        else:
            self.proj_in = nn.Conv2d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)

        # Define transformers blocks
        self.transformer_blocks = nn.ModuleList(
            [
                BasicSparseTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    cross_attention_dim=cross_attention_dim,
                    activation_fn=activation_fn,
                    num_embeds_ada_norm=num_embeds_ada_norm,
                    attention_bias=attention_bias,
                    only_cross_attention=only_cross_attention,
                    upcast_attention=upcast_attention,
                )
                for d in range(num_layers)
            ]
        )

        # 4. Define output layers
        if use_linear_projection:
            self.proj_out = nn.Linear(in_channels, inner_dim)
        else:
            self.proj_out = nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        timestep=None,
        class_labels=None,
        cross_attention_kwargs=None,
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
            class_labels ( `torch.Tensor`, *optional*):
                Ignored right now. Added for interoperability with Transformer2DModel.
            cross_attention_kwargs (`dict`, *optional*):
                Ignored right now. Added for interoperability with Transformer2DModel.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformer_3d.Transformer3DModelOutput`] instead of a plain
                tuple.

        Returns:
            [`~models.transformer_3d.Transformer3DModelOutput`] or `tuple`:
            [`~models.transformer_3d.Transformer3DModelOutput`] if `return_dict` is True, otherwise a `tuple`. When
            returning a tuple, the first element is the sample tensor.
        """
        # TODO: Deal with class_labels, cross_attention_kwargs
        # Input
        assert hidden_states.dim() == 5, f"Expected hidden_states to have ndim=5, but got ndim={hidden_states.dim()}."
        video_length = hidden_states.shape[2]
        # hidden_states = rearrange(hidden_states, "b c f h w -> (b f) c h w")
        hidden_states = hidden_states.movedim((0, 1, 2, 3, 4), (0, 2, 1, 3, 4))
        hidden_states = hidden_states.flatten(0, 1)
        # encoder_hidden_states = repeat(encoder_hidden_states, "b n c -> (b f) n c", f=video_length)
        encoder_hidden_states = encoder_hidden_states.repeat_interleave(repeats=video_length, dim=0)

        batch, channel, height, weight = hidden_states.shape
        residual = hidden_states

        hidden_states = self.norm(hidden_states)
        if not self.use_linear_projection:
            hidden_states = self.proj_in(hidden_states)
            inner_dim = hidden_states.shape[1]
            hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * weight, inner_dim)
        else:
            inner_dim = hidden_states.shape[1]
            hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * weight, inner_dim)
            hidden_states = self.proj_in(hidden_states)

        # Blocks
        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                video_length=video_length,
            )

        # Output
        if not self.use_linear_projection:
            hidden_states = hidden_states.reshape(batch, height, weight, inner_dim).permute(0, 3, 1, 2).contiguous()
            hidden_states = self.proj_out(hidden_states)
        else:
            hidden_states = self.proj_out(hidden_states)
            hidden_states = hidden_states.reshape(batch, height, weight, inner_dim).permute(0, 3, 1, 2).contiguous()

        output = hidden_states + residual

        # output = rearrange(output, "(b f) c h w -> b c f h w", f=video_length)
        output = output.reshape([-1, video_length, *output.shape[1:]])
        output = output.movedim((0, 1, 2, 3, 4), (0, 2, 1, 3, 4))
        if not return_dict:
            return (output,)

        return Transformer3DModelOutput(sample=output)
