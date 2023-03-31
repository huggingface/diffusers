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

import torch
from torch import nn

from .attention import Transformer3DModel
from .resnet import Downsample3D, ResnetBlock3D, Upsample3D, Downsample2D, ResnetBlock2D, TemporalConvLayer, Upsample2D
from .transformer_2d import Transformer2DModel
from .transformer_temporal import TransformerTemporalModel


def get_down_block(
    down_block_type,
    num_layers,
    in_channels,
    out_channels,
    temb_channels,
    add_downsample,
    resnet_eps,
    resnet_act_fn,
    attn_num_head_channels,
    resnet_groups=None,
    cross_attention_dim=None,
    downsample_padding=None,
    dual_cross_attention=False,
    use_linear_projection=True,
    only_cross_attention=False,
    upcast_attention=False,
    resnet_time_scale_shift="default",
    use_temporal_transformer=True, #True for Text2VideoSD. False for TuneAVideo
    use_temporal_conv=True, #True for Text2VideoSD. False for TuneAVideo
    sub_blocks_type='2d' #2d for Text2VideoSD. #3d for TuneAVideo
):
    if down_block_type == "DownBlock3D":
        return DownBlock3D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            add_downsample=add_downsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            downsample_padding=downsample_padding,
            resnet_time_scale_shift=resnet_time_scale_shift,
            use_temporal_conv=True, #True for Text2VideoSD. False for TuneAVideo
            sub_blocks_type=sub_blocks_type #2d for Text2VideoSD. #3d for TuneAVideo
        )
    elif down_block_type == "CrossAttnDownBlock3D":
        if cross_attention_dim is None:
            raise ValueError("cross_attention_dim must be specified for CrossAttnDownBlock3D")
        return CrossAttnDownBlock3D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            add_downsample=add_downsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            downsample_padding=downsample_padding,
            cross_attention_dim=cross_attention_dim,
            attn_num_head_channels=attn_num_head_channels,
            dual_cross_attention=dual_cross_attention,
            use_linear_projection=use_linear_projection,
            only_cross_attention=only_cross_attention,
            upcast_attention=upcast_attention,
            resnet_time_scale_shift=resnet_time_scale_shift,
            use_temporal_transformer=True, #True for Text2VideoSD. False for TuneAVideo
            use_temporal_conv=True, #True for Text2VideoSD. False for TuneAVideo
            sub_blocks_type=sub_blocks_type #2d for Text2VideoSD. #3d for TuneAVideo
        )
    raise ValueError(f"{down_block_type} does not exist.")


def get_up_block(
    up_block_type,
    num_layers,
    in_channels,
    out_channels,
    prev_output_channel,
    temb_channels,
    add_upsample,
    resnet_eps,
    resnet_act_fn,
    attn_num_head_channels,
    resnet_groups=None,
    cross_attention_dim=None,
    dual_cross_attention=False,
    use_linear_projection=True,
    only_cross_attention=False,
    upcast_attention=False,
    resnet_time_scale_shift="default",
    use_temporal_transformer=True, #True for Text2VideoSD. False for TuneAVideo
    use_temporal_conv=True, #True for Text2VideoSD. False for TuneAVideo
    sub_blocks_type=sub_blocks_type #2d for Text2VideoSD. #3d for TuneAVideo
):
    if up_block_type == "UpBlock3D":
        return UpBlock3D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            prev_output_channel=prev_output_channel,
            temb_channels=temb_channels,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            resnet_time_scale_shift=resnet_time_scale_shift,
            use_temporal_transformer=True, #True for Text2VideoSD. False for TuneAVideo
            use_temporal_conv=True, #True for Text2VideoSD. False for TuneAVideo
            sub_blocks_type=sub_blocks_type #2d for Text2VideoSD. #3d for TuneAVideo
        )
    elif up_block_type == "CrossAttnUpBlock3D":
        if cross_attention_dim is None:
            raise ValueError("cross_attention_dim must be specified for CrossAttnUpBlock3D")
        return CrossAttnUpBlock3D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            prev_output_channel=prev_output_channel,
            temb_channels=temb_channels,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            cross_attention_dim=cross_attention_dim,
            attn_num_head_channels=attn_num_head_channels,
            dual_cross_attention=dual_cross_attention,
            use_linear_projection=use_linear_projection,
            only_cross_attention=only_cross_attention,
            upcast_attention=upcast_attention,
            resnet_time_scale_shift=resnet_time_scale_shift,
            use_temporal_transformer=True, #True for Text2VideoSD. False for TuneAVideo
            use_temporal_conv=True, #True for Text2VideoSD. False for TuneAVideo
            sub_blocks_type=sub_blocks_type #2d for Text2VideoSD. #3d for TuneAVideo
        )
    raise ValueError(f"{up_block_type} does not exist.")


class UNetMidBlock3DCrossAttn(nn.Module):
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
        attn_num_head_channels=1,
        output_scale_factor=1.0,
        cross_attention_dim=1280,
        dual_cross_attention=False,
        use_linear_projection=True,
        upcast_attention=False,
        use_temporal_transformer=True, #False for TuneAVideo
        use_temporal_conv=True, #False for TuneAVideo
        sub_blocks_type='2d' #2d for Text2VideoSD. #3d for TuneAVideo
    ):
        super().__init__()

        self.has_cross_attention = True
        self.attn_num_head_channels = attn_num_head_channels
        resnet_groups = resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)
        
        resnet_class = None
        transformer_class = None
        if sub_blocks_type == '2d':
            resnet_class = ResnetBlock2D
            transformer_class = Transformer2DModel
        elif sub_blocks_type == '3d':
            resnet_class = ResnetBlock3D
            transformer_class = Transformer3DModel

        # there is always at least one resnet
        resnets = [
            resnet_class(
                in_channels=in_channels,
                out_channels=in_channels,
                temb_channels=temb_channels,
                eps=resnet_eps,
                groups=resnet_groups,
                dropout=dropout,
                time_embedding_norm=resnet_time_scale_shift,
                non_linearity=resnet_act_fn,
                output_scale_factor=output_scale_factor,
                pre_norm=resnet_pre_norm,
            )
        ]
        
        temp_convs = []

        if use_temporal_conv:
            temp_convs = [
                TemporalConvLayer(
                    in_channels,
                    in_channels,
                    dropout=0.1,
                )
            ]
        attentions = []
        temp_attentions = []
        
        #TODO: Verify if this is a typo. Because TuneAVideo implementation is opposite
        if sub_blocks_type == '2d':
            num_attention_heads = in_channels // attn_num_head_channels
            attention_head_dim = attn_num_head_channels
        elif sub_blocks_type == '3d':
            num_attention_heads = attn_num_head_channels
            attention_head_dim = in_channels // attn_num_head_channels

        for _ in range(num_layers):
            if dual_cross_attention:
                raise NotImplementedError
            attentions.append(
                transformer_class(
                    num_attention_heads,
                    attention_head_dim,
                    in_channels=in_channels,
                    num_layers=1,
                    cross_attention_dim=cross_attention_dim,
                    norm_num_groups=resnet_groups,
                    use_linear_projection=use_linear_projection,
                    upcast_attention=upcast_attention,
                )
            )
            if use_temporal_transformer:
                temp_attentions.append(
                    TransformerTemporalModel(
                        in_channels // attn_num_head_channels,
                        attn_num_head_channels,
                        in_channels=in_channels,
                        num_layers=1,
                        cross_attention_dim=cross_attention_dim,
                        norm_num_groups=resnet_groups,
                    )
                )
            resnets.append(
                resnet_class(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )

        self.resnets = nn.ModuleList(resnets)
        self.temp_convs = nn.ModuleList(temp_convs)
        self.attentions = nn.ModuleList(attentions)
        self.temp_attentions = nn.ModuleList(temp_attentions)

    def forward(
        self,
        hidden_states,
        temb=None,
        encoder_hidden_states=None,
        attention_mask=None,
        num_frames=1,
        cross_attention_kwargs=None,
    ):  
        hidden_states = self.resnets[0](hidden_states, temb)
        if self.temp_convs:
            hidden_states = self.temp_convs[0](hidden_states, num_frames=num_frames)
        #TODO: Uncouple temp_conv and temp_attention
        if self.temp_convs and self.temp_attentions:
            for attn, temp_attn, resnet, temp_conv in zip(
                self.attentions, self.temp_attentions, self.resnets[1:], self.temp_convs[1:]
            ):
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                ).sample
                hidden_states = temp_attn(hidden_states, num_frames=num_frames).sample
                hidden_states = resnet(hidden_states, temb)
                hidden_states = temp_conv(hidden_states, num_frames=num_frames)
        
        elif not self.temp_convs and not self.temp_attentions:
            for attn, resnet in zip(self.attentions, self.resnets[1:]):
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs
                    ).sample
                hidden_states = resnet(hidden_states, temb)
        
        else:
            NotImplementedError("If using temp_conv, make sure temp_attentions are also setup")

        return hidden_states


class CrossAttnDownBlock3D(nn.Module):
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
        attn_num_head_channels=1,
        cross_attention_dim=1280,
        output_scale_factor=1.0,
        downsample_padding=1,
        add_downsample=True,
        dual_cross_attention=False,
        use_linear_projection=False,
        only_cross_attention=False,
        upcast_attention=False,
        use_temporal_transformer=True, #True for Text2VideoSD. False for TuneAVideo
        use_temporal_conv=True, #True for Text2VideoSD. False for TuneAVideo
        sub_blocks_type='2d', #2d for Text2VideoSD, 3d for TuneAVideo
    ):
        super().__init__()
        resnets = []
        attentions = []
        temp_attentions = []
        temp_convs = []

        self.has_cross_attention = True
        self.attn_num_head_channels = attn_num_head_channels

        if sub_blocks_type == '2d':
            resnet_class = ResnetBlock2D
            transformer_class = Transformer2DModel
            downsampler_class = Downsample2D
        elif sub_blocks_type == '3d':
            resnet_class = ResnetBlock3D
            transformer_class = Transformer3DModel
            downsampler_class = Downsample3D
        else:
            raise NotImplementedError(f'Unexpected sub_blocks_type {sub_blocks_type}. Only `2d` or `3d` is expected')

        #TODO: Verify if this is a typo. Because TuneAVideo implementation is opposite
        if sub_blocks_type == '2d':
            num_attention_heads = in_channels // attn_num_head_channels
            attention_head_dim = attn_num_head_channels
        elif sub_blocks_type == '3d':
            num_attention_heads = attn_num_head_channels
            attention_head_dim = in_channels // attn_num_head_channels

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                resnet_class(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
            
            if dual_cross_attention:
                raise NotImplementedError
            
            if use_temporal_conv:
                temp_convs.append(
                    TemporalConvLayer(
                        out_channels,
                        out_channels,
                        dropout=0.1,
                    )
                )
            
            attentions.append(
                transformer_class(
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    in_channels=out_channels,
                    num_layers=1,
                    cross_attention_dim=cross_attention_dim,
                    norm_num_groups=resnet_groups,
                    use_linear_projection=use_linear_projection,
                    only_cross_attention=only_cross_attention,
                    upcast_attention=upcast_attention,
                )
            )
            if use_temporal_transformer:
                temp_attentions.append(
                    TransformerTemporalModel(
                        out_channels // attn_num_head_channels,
                        attn_num_head_channels,
                        in_channels=out_channels,
                        num_layers=1,
                        cross_attention_dim=cross_attention_dim,
                        norm_num_groups=resnet_groups,
                    )
                )
        self.resnets = nn.ModuleList(resnets)
        self.temp_convs = nn.ModuleList(temp_convs)
        self.attentions = nn.ModuleList(attentions)
        self.temp_attentions = nn.ModuleList(temp_attentions)

        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    downsampler_class(
                        out_channels, use_conv=True, out_channels=out_channels, padding=downsample_padding, name="op"
                    )
                ]
            )
        else:
            self.downsamplers = None

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        temb=None,
        encoder_hidden_states=None,
        attention_mask=None,
        num_frames=1,
        cross_attention_kwargs=None,
    ):
        # TODO(Patrick, William) - attention mask is not used
        output_states = ()
        
        if self.temp_conv and self.temp_attn:
            for resnet, temp_conv, attn, temp_attn in zip(
                self.resnets, self.temp_convs, self.attentions, self.temp_attentions
            ):
                hidden_states = resnet(hidden_states, temb)
                hidden_states = temp_conv(hidden_states, num_frames=num_frames)
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                ).sample
                hidden_states = temp_attn(hidden_states, num_frames=num_frames).sample

                output_states += (hidden_states,)
        
        elif not self.temp_conv and not self.temp_attn:
            for resnet, attn in zip(self.resnets, self.attentions):
            # if self.training and self.gradient_checkpointing:

            #     def create_custom_forward(module, return_dict=None):
            #         def custom_forward(*inputs):
            #             if return_dict is not None:
            #                 return module(*inputs, return_dict=return_dict)
            #             else:
            #                 return module(*inputs)

            #         return custom_forward

            #     hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(resnet), hidden_states, temb)
            #     hidden_states = torch.utils.checkpoint.checkpoint(
            #         create_custom_forward(attn, return_dict=False),
            #         hidden_states,
            #         encoder_hidden_states,
            #     )[0]
            # else:
                hidden_states = resnet(hidden_states, temb)
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs
                ).sample

                output_states += (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

            output_states += (hidden_states,)

        return hidden_states, output_states


class DownBlock3D(nn.Module):
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
        output_scale_factor=1.0,
        add_downsample=True,
        downsample_padding=1,
        use_temporal_conv:bool = True, #Use false for TuneAVideo
        sub_blocks_type: str = '2d' #Use 2d for ResNet2D, Transformer2D blocks in DownBlock3D. This is in TuneAVideo
    ):
        super().__init__()
        resnets = []
        temp_convs = []

        if sub_blocks_type == '2d':
            resnet_class = ResnetBlock2D
            downsampler_class = Downsample2D
        elif sub_blocks_type == '3d':
            resnet_class = ResnetBlock3D
            downsampler_class = Downsample3D
        else:
            raise NotImplementedError

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                resnet_class(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
            if use_temporal_conv:
                temp_convs.append(
                    TemporalConvLayer(
                        out_channels,
                        out_channels,
                        dropout=0.1,
                    )
                )

        self.resnets = nn.ModuleList(resnets)
        self.temp_convs = nn.ModuleList(temp_convs)

        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    downsampler_class(
                        out_channels, use_conv=True, out_channels=out_channels, padding=downsample_padding, name="op"
                    )
                ]
            )
        else:
            self.downsamplers = None

        self.gradient_checkpointing = False

    def forward(self, hidden_states, temb=None, num_frames=1):
        output_states = ()
        
        if self.temp_conv:
            for resnet, temp_conv in zip(self.resnets, self.temp_convs):
                hidden_states = resnet(hidden_states, temb)
                hidden_states = temp_conv(hidden_states, num_frames=num_frames)

                output_states += (hidden_states,)
        else:
            for resnet in self.resnets:
                # if self.training and self.gradient_checkpointing:

                #     def create_custom_forward(module):
                #         def custom_forward(*inputs):
                #             return module(*inputs)

                #         return custom_forward

                #     hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(resnet), hidden_states, temb)
                # else:
                hidden_states = resnet(hidden_states, temb)
                output_states += (hidden_states,)


        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

            output_states += (hidden_states,)

        return hidden_states, output_states


class CrossAttnUpBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        prev_output_channel: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        attn_num_head_channels=1,
        cross_attention_dim=1280,
        output_scale_factor=1.0,
        add_upsample=True,
        dual_cross_attention=False,
        use_linear_projection=False,
        only_cross_attention=False,
        upcast_attention=False,
        use_temporal_transformer: bool = True, #Use false for TuneAVideo
        use_temporal_conv:bool = True, #Use false for TuneAVideo
        sub_blocks_type: str = '2d' #Use 2d for ResNet2D, Transformer2D blocks in DownBlock3D. This is in TuneAVideo
    ):
        super().__init__()
        resnets = []
        temp_convs = []
        attentions = []
        temp_attentions = []

        self.has_cross_attention = True
        self.attn_num_head_channels = attn_num_head_channels

        if sub_blocks_type == '2d':
            resnet_class = ResnetBlock2D
            transformer_class = Transformer2DModel
            upsampler_class = Upsample2D
        elif sub_blocks_type == '3d':
            resnet_class = ResnetBlock3D
            transformer_class = Transformer3DModel
            upsampler_class = Upsample3D
        else:
            raise NotImplementedError
            
        #TODO: Verify if this is a typo. Because TuneAVideo implementation is opposite
        if sub_blocks_type == '2d':
            num_attention_heads = out_channels // attn_num_head_channels
            attention_head_dim = attn_num_head_channels
        elif sub_blocks_type == '3d':
            num_attention_heads = attn_num_head_channels
            attention_head_dim = out_channels // attn_num_head_channels,
                    

        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                resnet_class(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )

            if dual_cross_attention:
                raise NotImplementedError
            
            if use_temporal_conv:
                temp_convs.append(
                    TemporalConvLayer(
                        out_channels,
                        out_channels,
                        dropout=0.1,
                    )
                )
            attentions.append(
                transformer_class(
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    in_channels=out_channels,
                    num_layers=1,
                    cross_attention_dim=cross_attention_dim,
                    norm_num_groups=resnet_groups,
                    use_linear_projection=use_linear_projection,
                    only_cross_attention=only_cross_attention,
                    upcast_attention=upcast_attention,
                )
            )

            if use_temporal_transformer:
                temp_attentions.append(
                    TransformerTemporalModel(
                        out_channels // attn_num_head_channels,
                        attn_num_head_channels,
                        in_channels=out_channels,
                        num_layers=1,
                        cross_attention_dim=cross_attention_dim,
                        norm_num_groups=resnet_groups,
                    )
                )
        self.resnets = nn.ModuleList(resnets)
        self.temp_convs = nn.ModuleList(temp_convs)
        self.attentions = nn.ModuleList(attentions)
        self.temp_attentions = nn.ModuleList(temp_attentions)

        if add_upsample:
            self.upsamplers = nn.ModuleList([upsampler_class(out_channels, use_conv=True, out_channels=out_channels)])
        else:
            self.upsamplers = None

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        res_hidden_states_tuple,
        temb=None,
        encoder_hidden_states=None,
        upsample_size=None,
        attention_mask=None,
        num_frames=1,
        cross_attention_kwargs=None,
    ):  
        # TODO(Patrick, William) - attention mask is not used

        if self.temp_conv and self.temp_attn:
            for resnet, temp_conv, attn, temp_attn in zip(
                self.resnets, self.temp_convs, self.attentions, self.temp_attentions
            ):
                # pop res hidden states
                res_hidden_states = res_hidden_states_tuple[-1]
                res_hidden_states_tuple = res_hidden_states_tuple[:-1]
                hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
                
                hidden_states = resnet(hidden_states, temb)
                hidden_states = temp_conv(hidden_states, num_frames=num_frames)
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                ).sample
                hidden_states = temp_attn(hidden_states, num_frames=num_frames).sample
        
        elif not self.temp_conv and not self.temp_attn:
            for resnet, attn in zip(
                self.resnets, self.attentions
            ):  
                # pop res hidden states
                res_hidden_states = res_hidden_states_tuple[-1]
                res_hidden_states_tuple = res_hidden_states_tuple[:-1]
                hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
                
                hidden_states = resnet(hidden_states, temb)
                hidden_states = attn(hidden_states, encoder_hidden_states=encoder_hidden_states,cross_attention_kwargs=cross_attention_kwargs).sample

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size)

        return hidden_states


class UpBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        prev_output_channel: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor=1.0,
        add_upsample=True,
        use_temporal_conv:bool = True, #Use false for TuneAVideo
        sub_blocks_type: str = '2d' #Use 2d for ResNet2D, Transformer2D blocks in DownBlock3D. This is in TuneAVideo
    ):
        super().__init__()
        resnets = []
        temp_convs = []
        
        if sub_blocks_type == '2d':
            resnet_class = ResnetBlock2D
            upsampler_class = Upsample2D
        elif sub_blocks_type == '3d':
            resnet_class = ResnetBlock3D
            upsampler_class = Upsample3D


        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                resnet_class(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
            if use_temporal_conv:
                temp_convs.append(
                    TemporalConvLayer(
                        out_channels,
                        out_channels,
                        dropout=0.1,
                    )
                )

        self.resnets = nn.ModuleList(resnets)
        self.temp_convs = nn.ModuleList(temp_convs)

        if add_upsample:
            self.upsamplers = nn.ModuleList([upsampler_class(out_channels, use_conv=True, out_channels=out_channels)])
        else:
            self.upsamplers = None

        self.gradient_checkpointing = False

    def forward(self, hidden_states, res_hidden_states_tuple, temb=None, upsample_size=None, num_frames=1):
        if self.temp_convs:
            for resnet, temp_conv in zip(self.resnets, self.temp_convs):
                # pop res hidden states
                res_hidden_states = res_hidden_states_tuple[-1]
                res_hidden_states_tuple = res_hidden_states_tuple[:-1]
                hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
                
                hidden_states = resnet(hidden_states, temb)
                hidden_states = temp_conv(hidden_states, num_frames=num_frames)
        else:
            for resnet in self.resnets:
                # pop res hidden states
                res_hidden_states = res_hidden_states_tuple[-1]
                res_hidden_states_tuple = res_hidden_states_tuple[:-1]
                hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

                hidden_states = resnet(hidden_states, temb)
            # if self.training and self.gradient_checkpointing:

            #     def create_custom_forward(module):
            #         def custom_forward(*inputs):
            #             return module(*inputs)

            #         return custom_forward

            #     hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(resnet), hidden_states, temb)
            # else:
                # hidden_states = resnet(hidden_states, temb)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size)

        return hidden_states
