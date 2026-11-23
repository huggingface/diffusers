# Copyright 2026 The HuggingFace Team. All rights reserved.
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

from ...models.condition_embedders.condition_embedder_audioldm2 import (
    AudioLDM2ProjectionModel as _AudioLDM2ProjectionModel,
)
from ...models.unets.unet_2d_condition_audioldm2 import AudioLDM2UNet2DConditionModel as _AudioLDM2UNet2DConditionModel
from ...utils import deprecate


# Deprecated alias kept for backward compatibility. The warning is raised from `__new__` rather than `__init__` so
# that the parent's `__init__` signature, which `ConfigMixin` inspects when loading a config, stays intact.
class AudioLDM2ProjectionModel(_AudioLDM2ProjectionModel):
    def __new__(cls, *args, **kwargs):
        deprecate(
            "AudioLDM2ProjectionModel",
            "1.0.0",
            "Importing `AudioLDM2ProjectionModel` from `diffusers.pipelines.audioldm2.modeling_audioldm2` is deprecated. "
            "Import it with `from diffusers import AudioLDM2ProjectionModel` instead.",
        )
        return super().__new__(cls)


class AudioLDM2UNet2DConditionModel(_AudioLDM2UNet2DConditionModel):
    def __new__(cls, *args, **kwargs):
        deprecate(
            "AudioLDM2UNet2DConditionModel",
            "1.0.0",
            "Importing `AudioLDM2UNet2DConditionModel` from `diffusers.pipelines.audioldm2.modeling_audioldm2` is deprecated. "
            "Import it with `from diffusers import AudioLDM2UNet2DConditionModel` instead.",
        )
        return super().__new__(cls)
