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

from .attention import AttentionBlock, LinearAttention, SpatialTransformer
from .resnet import ResnetBlock2D


class UNetMidBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_blocks: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        attention_layer_type: str = "self",
        attn_num_heads=1,
        attn_num_head_channels=None,
        attn_encoder_channels=None,
        attn_dim_head=None,
        attn_depth=None,
        output_scale_factor=1.0,
        overwrite_qkv=False,
        overwrite_unet=False,
    ):
        super().__init__()

        # there is always at least one resnet
        resnets = [
            ResnetBlock2D(
                in_channels=in_channels,
                out_channels=in_channels,
                temb_channels=temb_channels,
                groups=resnet_groups,
                dropout=dropout,
                time_embedding_norm=resnet_time_scale_shift,
                non_linearity=resnet_act_fn,
                output_scale_factor=output_scale_factor,
                pre_norm=resnet_pre_norm,
            )
        ]
        attentions = []

        for _ in range(num_blocks):
            if attention_layer_type == "self":
                attentions.append(
                    AttentionBlock(
                        in_channels,
                        num_heads=attn_num_heads,
                        num_head_channels=attn_num_head_channels,
                        encoder_channels=attn_encoder_channels,
                        overwrite_qkv=overwrite_qkv,
                        rescale_output_factor=output_scale_factor,
                    )
                )
            elif attention_layer_type == "spatial":
                attentions.append(
                    SpatialTransformer(
                        in_channels,
                        attn_num_heads,
                        attn_num_head_channels,
                        depth=attn_depth,
                        context_dim=attn_encoder_channels,
                    )
                )
            elif attention_layer_type == "linear":
                attentions.append(LinearAttention(in_channels))

            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    temb_channels=temb_channels,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

    def forward(self, hidden_states, temb=None, encoder_states=None, mask=1.0):
        hidden_states = self.resnets[0](hidden_states, temb, mask=mask)

        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            hidden_states = attn(hidden_states, encoder_states)
            hidden_states = resnet(hidden_states, temb, mask=mask)

        return hidden_states


# class UNetResAttnDownBlock(nn.Module):
#    def __init__(
#        self,
#        in_channels: int,
#        out_channels: int,
#        temb_channels: int,
#        dropout: float = 0.0,
#        resnet_eps: float = 1e-6,
#        resnet_time_scale_shift: str = "default",
#        resnet_act_fn: str = "swish",
#        resnet_groups: int = 32,
#        resnet_pre_norm: bool = True,
#        attention_layer_type: str = "self",
#        attn_num_heads=1,
#        attn_num_head_channels=None,
#        attn_encoder_channels=None,
#        attn_dim_head=None,
#        attn_depth=None,
#        output_scale_factor=1.0,
#        overwrite_qkv=False,
#        overwrite_unet=False,
#    ):
#
#        self.resents =
