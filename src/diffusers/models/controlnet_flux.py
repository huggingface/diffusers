# Copyright 2024 Black Forest Labs, The HuggingFace Team and The InstantX Team. All rights reserved.
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


from typing import List

from ..utils import deprecate, logging
from .controlnets.controlnet_flux import FluxControlNetModel, FluxControlNetOutput, FluxMultiControlNetModel


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class FluxControlNetOutput(FluxControlNetOutput):
    def __init__(self, *args, **kwargs):
        deprecation_message = "Importing `FluxControlNetOutput` from `diffusers.models.controlnet_flux` is deprecated and this will be removed in a future version. Please use `from diffusers.models.controlnets.controlnet_flux import FluxControlNetOutput`, instead."
        deprecate("diffusers.models.controlnet_flux.FluxControlNetOutput", "0.34", deprecation_message)
        super().__init__(*args, **kwargs)


class FluxControlNetModel(FluxControlNetModel):
    def __init__(
        self,
        patch_size: int = 1,
        in_channels: int = 64,
        num_layers: int = 19,
        num_single_layers: int = 38,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 4096,
        pooled_projection_dim: int = 768,
        guidance_embeds: bool = False,
        axes_dims_rope: List[int] = [16, 56, 56],
        num_mode: int = None,
        conditioning_embedding_channels: int = None,
    ):
        deprecation_message = "Importing `FluxControlNetModel` from `diffusers.models.controlnet_flux` is deprecated and this will be removed in a future version. Please use `from diffusers.models.controlnets.controlnet_flux import FluxControlNetModel`, instead."
        deprecate("diffusers.models.controlnet_flux.FluxControlNetModel", "0.34", deprecation_message)
        super().__init__(
            patch_size=patch_size,
            in_channels=in_channels,
            num_layers=num_layers,
            num_single_layers=num_single_layers,
            attention_head_dim=attention_head_dim,
            num_attention_heads=num_attention_heads,
            joint_attention_dim=joint_attention_dim,
            pooled_projection_dim=pooled_projection_dim,
            guidance_embeds=guidance_embeds,
            axes_dims_rope=axes_dims_rope,
            num_mode=num_mode,
            conditioning_embedding_channels=conditioning_embedding_channels,
        )


class FluxMultiControlNetModel(FluxMultiControlNetModel):
    def __init__(self, *args, **kwargs):
        deprecation_message = "Importing `FluxMultiControlNetModel` from `diffusers.models.controlnet_flux` is deprecated and this will be removed in a future version. Please use `from diffusers.models.controlnets.controlnet_flux import FluxMultiControlNetModel`, instead."
        deprecate("diffusers.models.controlnet_flux.FluxMultiControlNetModel", "0.34", deprecation_message)
        super().__init__(*args, **kwargs)
