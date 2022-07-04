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


class UNet2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        num_mid_resnets: int = 2,
        attention_layer_type: str = "self",
        attn_num_heads=1,
        attn_num_head_channels=None,
        attn_encoder_channels=None,
        attn_dim_head=None,
        attn_depth=None,
        num_mid_attentions: int = 1,
        output_scale_factor=1.0,
        overwrite_qkv=False,
        overwrite_unet=False,
    ):
        super().__init__()

        self.num_mid_resnets = num_mid_resnets
        mid_resnets = [
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
            ) for _ in range(self.num_mid_resnets)
        ]
        self.mid_resnets = nn.ModuleList(mid_resnets)

        self.num_mid_attentions = num_mid_attentions
        mid_attentions = []
        for _ in range(self.num_mid_attentions):
            if attention_layer_type == "self":
                mid_attentions.append(AttentionBlock(
                    in_channels,
                    num_heads=attn_num_heads,
                    num_head_channels=attn_num_head_channels,
                    encoder_channels=attn_encoder_channels,
                    overwrite_qkv=overwrite_qkv,
                    rescale_output_factor=output_scale_factor,
                ))
            elif attention_layer_type == "spatial":
                mid_attentions.append(SpatialTransformer(
                    attn_num_heads,
                    attn_num_head_channels,
                    depth=attn_depth,
                    context_dim=attn_encoder_channels,
                ))
            elif attention_layer_type == "linear":
                mid_attentions.append(LinearAttention(in_channels))
        self.mid_attentions = nn.ModuleList(mid_attentions)

        # TODO(Patrick) - delete all of the following code
        self.is_overwritten = False
        self.overwrite_unet = overwrite_unet
        if self.overwrite_unet:
            block_in = in_channels
            self.temb_ch = temb_channels
            self.block_1 = ResnetBlock2D(
                in_channels=block_in,
                out_channels=block_in,
                temb_channels=self.temb_ch,
                dropout=dropout,
                eps=resnet_eps,
            )
            self.attn_1 = AttentionBlock(
                block_in,
                num_heads=attn_num_heads,
                num_head_channels=attn_num_head_channels,
                encoder_channels=attn_encoder_channels,
                overwrite_qkv=True,
            )
            self.block_2 = ResnetBlock2D(
                in_channels=block_in,
                out_channels=block_in,
                temb_channels=self.temb_ch,
                dropout=dropout,
                eps=resnet_eps,
            )

    def forward(self, hidden_states, temb=None, encoder_states=None, mask=1.0):
        if not self.is_overwritten and self.overwrite_unet:
            self.mid_resnets[0] = self.block_1
            self.mid_resnets[1] = self.block_2
            self.mid_attentions[0] = self.attn_1
            self.is_overwritten = True

        for i in range(len(self.mid_resnets)):
            hidden_states = self.mid_resnets[i](hidden_states, temb, mask=mask)

            if i % 2 == 0:
                hidden_states = self.mid_attentions[i](hidden_states, encoder_states)

        return hidden_states
