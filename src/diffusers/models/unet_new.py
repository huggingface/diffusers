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
from .resnet import ResnetBlock2D, Downsample2D


class UNetMidBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
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

        for _ in range(num_layers):
            if attention_layer_type == "self":
                attentions.append(
                    AttentionBlock(
                        in_channels,
                        num_heads=attn_num_heads,
                        num_head_channels=attn_num_head_channels,
                        encoder_channels=attn_encoder_channels,
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


class UNetResAttnDownBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
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
        add_downsample=True,
    ):
        super().__init__()
        resnets = []
        attentions = []

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append([
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            ])
            attentions.append(
                AttentionBlock(
                    in_channels,
                    num_heads=attn_num_heads,
                    num_head_channels=attn_num_head_channels,
                    encoder_channels=attn_encoder_channels,
                    rescale_output_factor=output_scale_factor,
                )
            )
#            elif attention_layer_type == "spatial":
#                attentions.append(
#                    SpatialTransformer(
#                        in_channels,
#                        attn_num_heads,
#                        attn_num_head_channels,
#                        depth=attn_depth,
#                        context_dim=attn_encoder_channels,
#                    )
#                )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        if add_downsample:
            self.downsamplers = nn.ModuleList([Downsample2D(in_channels, use_conv=True, out_channels=out_channels, padding=1, name="op")])
        else:
            self.downsamplers = None

    def forward(self, hidden_states, temb=None):
        output_states = ()

        for resnet, attn in zip(self.resnets, self.attentions):
            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(hidden_states)
            output_states += (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

            output_states += (hidden_states,)

        return hidden_states, output_states


class UNetResDownBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
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
        add_downsample=True,
    ):
        super().__init__()
        resnets = []

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )

        self.resnets = nn.ModuleList(resnets)

        if add_downsample:
            self.downsamplers = nn.ModuleList([Downsample2D(in_channels, use_conv=True, out_channels=out_channels, padding=1, name="op")])
        else:
            self.downsamplers = None

    def forward(self, hidden_states, temb=None):
        output_states = ()

        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb)
            output_states += (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

            output_states += (hidden_states,)

        return hidden_states, output_states
