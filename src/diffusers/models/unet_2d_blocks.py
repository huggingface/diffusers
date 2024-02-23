# Copyright 2024 The HuggingFace Team. All rights reserved.
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

from typing import Optional

from ..utils import deprecate
from .unets.unet_2d_blocks import (
    AttnDownBlock2D,
    AttnDownEncoderBlock2D,
    AttnSkipDownBlock2D,
    AttnSkipUpBlock2D,
    AttnUpBlock2D,
    AttnUpDecoderBlock2D,
    AutoencoderTinyBlock,
    CrossAttnDownBlock2D,
    CrossAttnUpBlock2D,
    DownBlock2D,
    KAttentionBlock,
    KCrossAttnDownBlock2D,
    KCrossAttnUpBlock2D,
    KDownBlock2D,
    KUpBlock2D,
    ResnetDownsampleBlock2D,
    ResnetUpsampleBlock2D,
    SimpleCrossAttnDownBlock2D,
    SimpleCrossAttnUpBlock2D,
    SkipDownBlock2D,
    SkipUpBlock2D,
    UNetMidBlock2D,
    UNetMidBlock2DCrossAttn,
    UNetMidBlock2DSimpleCrossAttn,
    UpBlock2D,
    UpDecoderBlock2D,
)


def get_down_block(
    down_block_type: str,
    num_layers: int,
    in_channels: int,
    out_channels: int,
    temb_channels: int,
    add_downsample: bool,
    resnet_eps: float,
    resnet_act_fn: str,
    transformer_layers_per_block: int = 1,
    num_attention_heads: Optional[int] = None,
    resnet_groups: Optional[int] = None,
    cross_attention_dim: Optional[int] = None,
    downsample_padding: Optional[int] = None,
    dual_cross_attention: bool = False,
    use_linear_projection: bool = False,
    only_cross_attention: bool = False,
    upcast_attention: bool = False,
    resnet_time_scale_shift: str = "default",
    attention_type: str = "default",
    resnet_skip_time_act: bool = False,
    resnet_out_scale_factor: float = 1.0,
    cross_attention_norm: Optional[str] = None,
    attention_head_dim: Optional[int] = None,
    downsample_type: Optional[str] = None,
    dropout: float = 0.0,
):
    deprecation_message = "Importing `get_down_block` from `diffusers.models.unet_2d_blocks` is deprecated and this will be removed in a future version. Please use `from diffusers.models.unets.unet_2d_blocks import get_down_block`, instead."
    deprecate("get_down_block", "0.29", deprecation_message)

    from .unets.unet_2d_blocks import get_down_block

    return get_down_block(
        down_block_type=down_block_type,
        num_layers=num_layers,
        in_channels=in_channels,
        out_channels=out_channels,
        temb_channels=temb_channels,
        add_downsample=add_downsample,
        resnet_eps=resnet_eps,
        resnet_act_fn=resnet_act_fn,
        transformer_layers_per_block=transformer_layers_per_block,
        num_attention_heads=num_attention_heads,
        resnet_groups=resnet_groups,
        cross_attention_dim=cross_attention_dim,
        downsample_padding=downsample_padding,
        dual_cross_attention=dual_cross_attention,
        use_linear_projection=use_linear_projection,
        only_cross_attention=only_cross_attention,
        upcast_attention=upcast_attention,
        resnet_time_scale_shift=resnet_time_scale_shift,
        attention_type=attention_type,
        resnet_skip_time_act=resnet_skip_time_act,
        resnet_out_scale_factor=resnet_out_scale_factor,
        cross_attention_norm=cross_attention_norm,
        attention_head_dim=attention_head_dim,
        downsample_type=downsample_type,
        dropout=dropout,
    )


def get_mid_block(
    mid_block_type: str,
    temb_channels: int,
    in_channels: int,
    resnet_eps: float,
    resnet_act_fn: str,
    resnet_groups: int,
    output_scale_factor: float = 1.0,
    transformer_layers_per_block: int = 1,
    num_attention_heads: Optional[int] = None,
    cross_attention_dim: Optional[int] = None,
    dual_cross_attention: bool = False,
    use_linear_projection: bool = False,
    mid_block_only_cross_attention: bool = False,
    upcast_attention: bool = False,
    resnet_time_scale_shift: str = "default",
    attention_type: str = "default",
    resnet_skip_time_act: bool = False,
    cross_attention_norm: Optional[str] = None,
    attention_head_dim: Optional[int] = 1,
    dropout: float = 0.0,
):
    if mid_block_type == "UNetMidBlock2DCrossAttn":
        return UNetMidBlock2DCrossAttn(
            transformer_layers_per_block=transformer_layers_per_block,
            in_channels=in_channels,
            temb_channels=temb_channels,
            dropout=dropout,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            output_scale_factor=output_scale_factor,
            resnet_time_scale_shift=resnet_time_scale_shift,
            cross_attention_dim=cross_attention_dim,
            num_attention_heads=num_attention_heads,
            resnet_groups=resnet_groups,
            dual_cross_attention=dual_cross_attention,
            use_linear_projection=use_linear_projection,
            upcast_attention=upcast_attention,
            attention_type=attention_type,
        )
    elif mid_block_type == "UNetMidBlock2DSimpleCrossAttn":
        return UNetMidBlock2DSimpleCrossAttn(
            in_channels=in_channels,
            temb_channels=temb_channels,
            dropout=dropout,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            output_scale_factor=output_scale_factor,
            cross_attention_dim=cross_attention_dim,
            attention_head_dim=attention_head_dim,
            resnet_groups=resnet_groups,
            resnet_time_scale_shift=resnet_time_scale_shift,
            skip_time_act=resnet_skip_time_act,
            only_cross_attention=mid_block_only_cross_attention,
            cross_attention_norm=cross_attention_norm,
        )
    elif mid_block_type == "UNetMidBlock2D":
        return UNetMidBlock2D(
            in_channels=in_channels,
            temb_channels=temb_channels,
            dropout=dropout,
            num_layers=0,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            output_scale_factor=output_scale_factor,
            resnet_groups=resnet_groups,
            resnet_time_scale_shift=resnet_time_scale_shift,
            add_attention=False,
        )
    elif mid_block_type is None:
        return None
    else:
        raise ValueError(f"unknown mid_block_type : {mid_block_type}")


def get_up_block(
    up_block_type: str,
    num_layers: int,
    in_channels: int,
    out_channels: int,
    prev_output_channel: int,
    temb_channels: int,
    add_upsample: bool,
    resnet_eps: float,
    resnet_act_fn: str,
    resolution_idx: Optional[int] = None,
    transformer_layers_per_block: int = 1,
    num_attention_heads: Optional[int] = None,
    resnet_groups: Optional[int] = None,
    cross_attention_dim: Optional[int] = None,
    dual_cross_attention: bool = False,
    use_linear_projection: bool = False,
    only_cross_attention: bool = False,
    upcast_attention: bool = False,
    resnet_time_scale_shift: str = "default",
    attention_type: str = "default",
    resnet_skip_time_act: bool = False,
    resnet_out_scale_factor: float = 1.0,
    cross_attention_norm: Optional[str] = None,
    attention_head_dim: Optional[int] = None,
    upsample_type: Optional[str] = None,
    dropout: float = 0.0,
):
    deprecation_message = "Importing `get_up_block` from `diffusers.models.unet_2d_blocks` is deprecated and this will be removed in a future version. Please use `from diffusers.models.unets.unet_2d_blocks import get_up_block`, instead."
    deprecate("get_up_block", "0.29", deprecation_message)

    from .unets.unet_2d_blocks import get_up_block

    return get_up_block(
        up_block_type=up_block_type,
        num_layers=num_layers,
        in_channels=in_channels,
        out_channels=out_channels,
        prev_output_channel=prev_output_channel,
        temb_channels=temb_channels,
        add_upsample=add_upsample,
        resnet_eps=resnet_eps,
        resnet_act_fn=resnet_act_fn,
        resolution_idx=resolution_idx,
        transformer_layers_per_block=transformer_layers_per_block,
        num_attention_heads=num_attention_heads,
        resnet_groups=resnet_groups,
        cross_attention_dim=cross_attention_dim,
        dual_cross_attention=dual_cross_attention,
        use_linear_projection=use_linear_projection,
        only_cross_attention=only_cross_attention,
        upcast_attention=upcast_attention,
        resnet_time_scale_shift=resnet_time_scale_shift,
        attention_type=attention_type,
        resnet_skip_time_act=resnet_skip_time_act,
        resnet_out_scale_factor=resnet_out_scale_factor,
        cross_attention_norm=cross_attention_norm,
        attention_head_dim=attention_head_dim,
        upsample_type=upsample_type,
        dropout=dropout,
    )


class AutoencoderTinyBlock(AutoencoderTinyBlock):
    deprecation_message = "Importing `AutoencoderTinyBlock` from `diffusers.models.unet_2d_blocks` is deprecated and this will be removed in a future version. Please use `from diffusers.models.unets.unet_2d_blocks import AutoencoderTinyBlock`, instead."
    deprecate("AutoencoderTinyBlock", "0.29", deprecation_message)


class UNetMidBlock2D(UNetMidBlock2D):
    deprecation_message = "Importing `UNetMidBlock2D` from `diffusers.models.unet_2d_blocks` is deprecated and this will be removed in a future version. Please use `from diffusers.models.unets.unet_2d_blocks import UNetMidBlock2D`, instead."
    deprecate("UNetMidBlock2D", "0.29", deprecation_message)


class UNetMidBlock2DCrossAttn(UNetMidBlock2DCrossAttn):
    deprecation_message = "Importing `UNetMidBlock2DCrossAttn` from `diffusers.models.unet_2d_blocks` is deprecated and this will be removed in a future version. Please use `from diffusers.models.unets.unet_2d_blocks import UNetMidBlock2DCrossAttn`, instead."
    deprecate("UNetMidBlock2DCrossAttn", "0.29", deprecation_message)


class UNetMidBlock2DSimpleCrossAttn(UNetMidBlock2DSimpleCrossAttn):
    deprecation_message = "Importing `UNetMidBlock2DSimpleCrossAttn` from `diffusers.models.unet_2d_blocks` is deprecated and this will be removed in a future version. Please use `from diffusers.models.unets.unet_2d_blocks import UNetMidBlock2DSimpleCrossAttn`, instead."
    deprecate("UNetMidBlock2DSimpleCrossAttn", "0.29", deprecation_message)


class AttnDownBlock2D(AttnDownBlock2D):
    deprecation_message = "Importing `AttnDownBlock2D` from `diffusers.models.unet_2d_blocks` is deprecated and this will be removed in a future version. Please use `from diffusers.models.unets.unet_2d_blocks import AttnDownBlock2D`, instead."
    deprecate("AttnDownBlock2D", "0.29", deprecation_message)


class CrossAttnDownBlock2D(CrossAttnDownBlock2D):
    deprecation_message = "Importing `AttnDownBlock2D` from `diffusers.models.unet_2d_blocks` is deprecated and this will be removed in a future version. Please use `from diffusers.models.unets.unet_2d_blocks import CrossAttnDownBlock2D`, instead."
    deprecate("CrossAttnDownBlock2D", "0.29", deprecation_message)


class DownBlock2D(DownBlock2D):
    deprecation_message = "Importing `DownBlock2D` from `diffusers.models.unet_2d_blocks` is deprecated and this will be removed in a future version. Please use `from diffusers.models.unets.unet_2d_blocks import DownBlock2D`, instead."
    deprecate("DownBlock2D", "0.29", deprecation_message)


class AttnDownEncoderBlock2D(AttnDownEncoderBlock2D):
    deprecation_message = "Importing `AttnDownEncoderBlock2D` from `diffusers.models.unet_2d_blocks` is deprecated and this will be removed in a future version. Please use `from diffusers.models.unets.unet_2d_blocks import AttnDownEncoderBlock2D`, instead."
    deprecate("AttnDownEncoderBlock2D", "0.29", deprecation_message)


class AttnSkipDownBlock2D(AttnSkipDownBlock2D):
    deprecation_message = "Importing `AttnSkipDownBlock2D` from `diffusers.models.unet_2d_blocks` is deprecated and this will be removed in a future version. Please use `from diffusers.models.unets.unet_2d_blocks import AttnSkipDownBlock2D`, instead."
    deprecate("AttnSkipDownBlock2D", "0.29", deprecation_message)


class SkipDownBlock2D(SkipDownBlock2D):
    deprecation_message = "Importing `SkipDownBlock2D` from `diffusers.models.unet_2d_blocks` is deprecated and this will be removed in a future version. Please use `from diffusers.models.unets.unet_2d_blocks import SkipDownBlock2D`, instead."
    deprecate("SkipDownBlock2D", "0.29", deprecation_message)


class ResnetDownsampleBlock2D(ResnetDownsampleBlock2D):
    deprecation_message = "Importing `ResnetDownsampleBlock2D` from `diffusers.models.unet_2d_blocks` is deprecated and this will be removed in a future version. Please use `from diffusers.models.unets.unet_2d_blocks import ResnetDownsampleBlock2D`, instead."
    deprecate("ResnetDownsampleBlock2D", "0.29", deprecation_message)


class SimpleCrossAttnDownBlock2D(SimpleCrossAttnDownBlock2D):
    deprecation_message = "Importing `SimpleCrossAttnDownBlock2D` from `diffusers.models.unet_2d_blocks` is deprecated and this will be removed in a future version. Please use `from diffusers.models.unets.unet_2d_blocks import SimpleCrossAttnDownBlock2D`, instead."
    deprecate("SimpleCrossAttnDownBlock2D", "0.29", deprecation_message)


class KDownBlock2D(KDownBlock2D):
    deprecation_message = "Importing `KDownBlock2D` from `diffusers.models.unet_2d_blocks` is deprecated and this will be removed in a future version. Please use `from diffusers.models.unets.unet_2d_blocks import KDownBlock2D`, instead."
    deprecate("KDownBlock2D", "0.29", deprecation_message)


class KCrossAttnDownBlock2D(KCrossAttnDownBlock2D):
    deprecation_message = "Importing `KCrossAttnDownBlock2D` from `diffusers.models.unet_2d_blocks` is deprecated and this will be removed in a future version. Please use `from diffusers.models.unets.unet_2d_blocks import KCrossAttnDownBlock2D`, instead."
    deprecate("KCrossAttnDownBlock2D", "0.29", deprecation_message)


class AttnUpBlock2D(AttnUpBlock2D):
    deprecation_message = "Importing `AttnUpBlock2D` from `diffusers.models.unet_2d_blocks` is deprecated and this will be removed in a future version. Please use `from diffusers.models.unets.unet_2d_blocks import AttnUpBlock2D`, instead."
    deprecate("AttnUpBlock2D", "0.29", deprecation_message)


class CrossAttnUpBlock2D(CrossAttnUpBlock2D):
    deprecation_message = "Importing `CrossAttnUpBlock2D` from `diffusers.models.unet_2d_blocks` is deprecated and this will be removed in a future version. Please use `from diffusers.models.unets.unet_2d_blocks import CrossAttnUpBlock2D`, instead."
    deprecate("CrossAttnUpBlock2D", "0.29", deprecation_message)


class UpBlock2D(UpBlock2D):
    deprecation_message = "Importing `UpBlock2D` from `diffusers.models.unet_2d_blocks` is deprecated and this will be removed in a future version. Please use `from diffusers.models.unets.unet_2d_blocks import UpBlock2D`, instead."
    deprecate("UpBlock2D", "0.29", deprecation_message)


class UpDecoderBlock2D(UpDecoderBlock2D):
    deprecation_message = "Importing `UpDecoderBlock2D` from `diffusers.models.unet_2d_blocks` is deprecated and this will be removed in a future version. Please use `from diffusers.models.unets.unet_2d_blocks import UpDecoderBlock2D`, instead."
    deprecate("UpDecoderBlock2D", "0.29", deprecation_message)


class AttnUpDecoderBlock2D(AttnUpDecoderBlock2D):
    deprecation_message = "Importing `AttnUpDecoderBlock2D` from `diffusers.models.unet_2d_blocks` is deprecated and this will be removed in a future version. Please use `from diffusers.models.unets.unet_2d_blocks import AttnUpDecoderBlock2D`, instead."
    deprecate("AttnUpDecoderBlock2D", "0.29", deprecation_message)


class AttnSkipUpBlock2D(AttnSkipUpBlock2D):
    deprecation_message = "Importing `AttnSkipUpBlock2D` from `diffusers.models.unet_2d_blocks` is deprecated and this will be removed in a future version. Please use `from diffusers.models.unets.unet_2d_blocks import AttnSkipUpBlock2D`, instead."
    deprecate("AttnSkipUpBlock2D", "0.29", deprecation_message)


class SkipUpBlock2D(SkipUpBlock2D):
    deprecation_message = "Importing `SkipUpBlock2D` from `diffusers.models.unet_2d_blocks` is deprecated and this will be removed in a future version. Please use `from diffusers.models.unets.unet_2d_blocks import SkipUpBlock2D`, instead."
    deprecate("SkipUpBlock2D", "0.29", deprecation_message)


class ResnetUpsampleBlock2D(ResnetUpsampleBlock2D):
    deprecation_message = "Importing `ResnetUpsampleBlock2D` from `diffusers.models.unet_2d_blocks` is deprecated and this will be removed in a future version. Please use `from diffusers.models.unets.unet_2d_blocks import ResnetUpsampleBlock2D`, instead."
    deprecate("ResnetUpsampleBlock2D", "0.29", deprecation_message)


class SimpleCrossAttnUpBlock2D(SimpleCrossAttnUpBlock2D):
    deprecation_message = "Importing `SimpleCrossAttnUpBlock2D` from `diffusers.models.unet_2d_blocks` is deprecated and this will be removed in a future version. Please use `from diffusers.models.unets.unet_2d_blocks import SimpleCrossAttnUpBlock2D`, instead."
    deprecate("SimpleCrossAttnUpBlock2D", "0.29", deprecation_message)


class KUpBlock2D(KUpBlock2D):
    deprecation_message = "Importing `KUpBlock2D` from `diffusers.models.unet_2d_blocks` is deprecated and this will be removed in a future version. Please use `from diffusers.models.unets.unet_2d_blocks import KUpBlock2D`, instead."
    deprecate("KUpBlock2D", "0.29", deprecation_message)


class KCrossAttnUpBlock2D(KCrossAttnUpBlock2D):
    deprecation_message = "Importing `KCrossAttnUpBlock2D` from `diffusers.models.unet_2d_blocks` is deprecated and this will be removed in a future version. Please use `from diffusers.models.unets.unet_2d_blocks import KCrossAttnUpBlock2D`, instead."
    deprecate("KCrossAttnUpBlock2D", "0.29", deprecation_message)


# can potentially later be renamed to `No-feed-forward` attention
class KAttentionBlock(KAttentionBlock):
    deprecation_message = "Importing `KAttentionBlock` from `diffusers.models.unet_2d_blocks` is deprecated and this will be removed in a future version. Please use `from diffusers.models.unets.unet_2d_blocks import KAttentionBlock`, instead."
    deprecate("KAttentionBlock", "0.29", deprecation_message)
