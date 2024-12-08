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
from typing import Optional, Tuple, Union

from ..utils import deprecate
from .controlnets.controlnet import (  # noqa
    ControlNetConditioningEmbedding,
    ControlNetModel,
    ControlNetOutput,
    zero_module,
)


class ControlNetOutput(ControlNetOutput):
    def __init__(self, *args, **kwargs):
        deprecation_message = "Importing `ControlNetOutput` from `diffusers.models.controlnet` is deprecated and this will be removed in a future version. Please use `from diffusers.models.controlnets.controlnet import ControlNetOutput`, instead."
        deprecate("diffusers.models.controlnet.ControlNetOutput", "0.34", deprecation_message)
        super().__init__(*args, **kwargs)


class ControlNetModel(ControlNetModel):
    def __init__(
        self,
        in_channels: int = 4,
        conditioning_channels: int = 3,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        down_block_types: Tuple[str, ...] = (
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        ),
        mid_block_type: Optional[str] = "UNetMidBlock2DCrossAttn",
        only_cross_attention: Union[bool, Tuple[bool]] = False,
        block_out_channels: Tuple[int, ...] = (320, 640, 1280, 1280),
        layers_per_block: int = 2,
        downsample_padding: int = 1,
        mid_block_scale_factor: float = 1,
        act_fn: str = "silu",
        norm_num_groups: Optional[int] = 32,
        norm_eps: float = 1e-5,
        cross_attention_dim: int = 1280,
        transformer_layers_per_block: Union[int, Tuple[int, ...]] = 1,
        encoder_hid_dim: Optional[int] = None,
        encoder_hid_dim_type: Optional[str] = None,
        attention_head_dim: Union[int, Tuple[int, ...]] = 8,
        num_attention_heads: Optional[Union[int, Tuple[int, ...]]] = None,
        use_linear_projection: bool = False,
        class_embed_type: Optional[str] = None,
        addition_embed_type: Optional[str] = None,
        addition_time_embed_dim: Optional[int] = None,
        num_class_embeds: Optional[int] = None,
        upcast_attention: bool = False,
        resnet_time_scale_shift: str = "default",
        projection_class_embeddings_input_dim: Optional[int] = None,
        controlnet_conditioning_channel_order: str = "rgb",
        conditioning_embedding_out_channels: Optional[Tuple[int, ...]] = (16, 32, 96, 256),
        global_pool_conditions: bool = False,
        addition_embed_type_num_heads: int = 64,
    ):
        deprecation_message = "Importing `ControlNetModel` from `diffusers.models.controlnet` is deprecated and this will be removed in a future version. Please use `from diffusers.models.controlnets.controlnet import ControlNetModel`, instead."
        deprecate("diffusers.models.controlnet.ControlNetModel", "0.34", deprecation_message)
        super().__init__(
            in_channels=in_channels,
            conditioning_channels=conditioning_channels,
            flip_sin_to_cos=flip_sin_to_cos,
            freq_shift=freq_shift,
            down_block_types=down_block_types,
            mid_block_type=mid_block_type,
            only_cross_attention=only_cross_attention,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            downsample_padding=downsample_padding,
            mid_block_scale_factor=mid_block_scale_factor,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            cross_attention_dim=cross_attention_dim,
            transformer_layers_per_block=transformer_layers_per_block,
            encoder_hid_dim=encoder_hid_dim,
            encoder_hid_dim_type=encoder_hid_dim_type,
            attention_head_dim=attention_head_dim,
            num_attention_heads=num_attention_heads,
            use_linear_projection=use_linear_projection,
            class_embed_type=class_embed_type,
            addition_embed_type=addition_embed_type,
            addition_time_embed_dim=addition_time_embed_dim,
            num_class_embeds=num_class_embeds,
            upcast_attention=upcast_attention,
            resnet_time_scale_shift=resnet_time_scale_shift,
            projection_class_embeddings_input_dim=projection_class_embeddings_input_dim,
            controlnet_conditioning_channel_order=controlnet_conditioning_channel_order,
            conditioning_embedding_out_channels=conditioning_embedding_out_channels,
            global_pool_conditions=global_pool_conditions,
            addition_embed_type_num_heads=addition_embed_type_num_heads,
        )


class ControlNetConditioningEmbedding(ControlNetConditioningEmbedding):
    def __init__(self, *args, **kwargs):
        deprecation_message = "Importing `ControlNetConditioningEmbedding` from `diffusers.models.controlnet` is deprecated and this will be removed in a future version. Please use `from diffusers.models.controlnets.controlnet import ControlNetConditioningEmbedding`, instead."
        deprecate("diffusers.models.controlnet.ControlNetConditioningEmbedding", "0.34", deprecation_message)
        super().__init__(*args, **kwargs)
