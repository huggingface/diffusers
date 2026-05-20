# Copyright 2025 The GLIGEN Authors and HuggingFace Team. All rights reserved.
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

from ...models.condition_embedders.projection_clip_image import CLIPImageProjection as _CLIPImageProjection
from ...utils import deprecate


class CLIPImageProjection(_CLIPImageProjection):
    def __init__(self, *args, **kwargs):
        deprecate(
            "CLIPImageProjection",
            "1.0.0",
            "Importing `CLIPImageProjection` from `diffusers.pipelines.stable_diffusion.clip_image_project_model` is "
            "deprecated. Import it from `diffusers.models.condition_embedders` instead "
            "(or `from diffusers import CLIPImageProjection`).",
        )
        super().__init__(*args, **kwargs)
