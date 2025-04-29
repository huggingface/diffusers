# Copyright 2025 The HuggingFace Team. All rights reserved.
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

from ..utils import deprecate, logging
from .controlnets.controlnet_sparsectrl import (  # noqa
    SparseControlNetConditioningEmbedding,
    SparseControlNetModel,
    SparseControlNetOutput,
    zero_module,
)


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class SparseControlNetOutput(SparseControlNetOutput):
    def __init__(self, *args, **kwargs):
        deprecation_message = "Importing `SparseControlNetOutput` from `diffusers.models.controlnet_sparsectrl` is deprecated and this will be removed in a future version. Please use `from diffusers.models.controlnets.controlnet_sparsectrl import SparseControlNetOutput`, instead."
        deprecate("diffusers.models.controlnet_sparsectrl.SparseControlNetOutput", "0.34", deprecation_message)
        super().__init__(*args, **kwargs)


class SparseControlNetConditioningEmbedding(SparseControlNetConditioningEmbedding):
    def __init__(self, *args, **kwargs):
        deprecation_message = "Importing `SparseControlNetConditioningEmbedding` from `diffusers.models.controlnet_sparsectrl` is deprecated and this will be removed in a future version. Please use `from diffusers.models.controlnets.controlnet_sparsectrl import SparseControlNetConditioningEmbedding`, instead."
        deprecate(
            "diffusers.models.controlnet_sparsectrl.SparseControlNetConditioningEmbedding", "0.34", deprecation_message
        )
        super().__init__(*args, **kwargs)


class SparseControlNetModel(SparseControlNetModel):
    def __init__(
        self,
        in_channels: int = 4,
        conditioning_channels: int = 4,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        down_block_types: Tuple[str, ...] = (
            "CrossAttnDownBlockMotion",
            "CrossAttnDownBlockMotion",
            "CrossAttnDownBlockMotion",
            "DownBlockMotion",
        ),
        only_cross_attention: Union[bool, Tuple[bool]] = False,
        block_out_channels: Tuple[int, ...] = (320, 640, 1280, 1280),
        layers_per_block: int = 2,
        downsample_padding: int = 1,
        mid_block_scale_factor: float = 1,
        act_fn: str = "silu",
        norm_num_groups: Optional[int] = 32,
        norm_eps: float = 1e-5,
        cross_attention_dim: int = 768,
        transformer_layers_per_block: Union[int, Tuple[int, ...]] = 1,
        transformer_layers_per_mid_block: Optional[Union[int, Tuple[int]]] = None,
        temporal_transformer_layers_per_block: Union[int, Tuple[int, ...]] = 1,
        attention_head_dim: Union[int, Tuple[int, ...]] = 8,
        num_attention_heads: Optional[Union[int, Tuple[int, ...]]] = None,
        use_linear_projection: bool = False,
        upcast_attention: bool = False,
        resnet_time_scale_shift: str = "default",
        conditioning_embedding_out_channels: Optional[Tuple[int, ...]] = (16, 32, 96, 256),
        global_pool_conditions: bool = False,
        controlnet_conditioning_channel_order: str = "rgb",
        motion_max_seq_length: int = 32,
        motion_num_attention_heads: int = 8,
        concat_conditioning_mask: bool = True,
        use_simplified_condition_embedding: bool = True,
    ):
        deprecation_message = "Importing `SparseControlNetModel` from `diffusers.models.controlnet_sparsectrl` is deprecated and this will be removed in a future version. Please use `from diffusers.models.controlnets.controlnet_sparsectrl import SparseControlNetModel`, instead."
        deprecate("diffusers.models.controlnet_sparsectrl.SparseControlNetModel", "0.34", deprecation_message)
        super().__init__(
            in_channels=in_channels,
            conditioning_channels=conditioning_channels,
            flip_sin_to_cos=flip_sin_to_cos,
            freq_shift=freq_shift,
            down_block_types=down_block_types,
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
            transformer_layers_per_mid_block=transformer_layers_per_mid_block,
            temporal_transformer_layers_per_block=temporal_transformer_layers_per_block,
            attention_head_dim=attention_head_dim,
            num_attention_heads=num_attention_heads,
            use_linear_projection=use_linear_projection,
            upcast_attention=upcast_attention,
            resnet_time_scale_shift=resnet_time_scale_shift,
            conditioning_embedding_out_channels=conditioning_embedding_out_channels,
            global_pool_conditions=global_pool_conditions,
            controlnet_conditioning_channel_order=controlnet_conditioning_channel_order,
            motion_max_seq_length=motion_max_seq_length,
            motion_num_attention_heads=motion_num_attention_heads,
            concat_conditioning_mask=concat_conditioning_mask,
            use_simplified_condition_embedding=use_simplified_condition_embedding,
        )
