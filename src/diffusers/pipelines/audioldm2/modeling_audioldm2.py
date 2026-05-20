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

from ...models.condition_embedders.projection_audioldm2 import (
    AudioLDM2ProjectionModel as _AudioLDM2ProjectionModel,
)
from ...models.condition_embedders.projection_audioldm2 import (
    AudioLDM2ProjectionModelOutput,  # noqa: F401  re-exported for back-compat
    add_special_tokens,  # noqa: F401  re-exported for back-compat
)
from ...models.unets.unet_2d_condition_audioldm2 import (
    AudioLDM2UNet2DConditionModel as _AudioLDM2UNet2DConditionModel,
)
from ...models.unets.unet_2d_condition_audioldm2 import (
    CrossAttnDownBlock2D,  # noqa: F401  re-exported for back-compat
    CrossAttnUpBlock2D,  # noqa: F401
    UNetMidBlock2DCrossAttn,  # noqa: F401
    get_down_block,  # noqa: F401
    get_up_block,  # noqa: F401
)
from ...utils import deprecate


class AudioLDM2ProjectionModel(_AudioLDM2ProjectionModel):
    def __init__(self, *args, **kwargs):
        deprecate(
            "AudioLDM2ProjectionModel",
            "1.0.0",
            "Importing `AudioLDM2ProjectionModel` from `diffusers.pipelines.audioldm2.modeling_audioldm2` is "
            "deprecated. Import it from `diffusers.models.condition_embedders` instead "
            "(or `from diffusers import AudioLDM2ProjectionModel`).",
        )
        super().__init__(*args, **kwargs)


class AudioLDM2UNet2DConditionModel(_AudioLDM2UNet2DConditionModel):
    def __init__(self, *args, **kwargs):
        deprecate(
            "AudioLDM2UNet2DConditionModel",
            "1.0.0",
            "Importing `AudioLDM2UNet2DConditionModel` from `diffusers.pipelines.audioldm2.modeling_audioldm2` is "
            "deprecated. Import it from `diffusers.models.unets` instead "
            "(or `from diffusers import AudioLDM2UNet2DConditionModel`).",
        )
        super().__init__(*args, **kwargs)
