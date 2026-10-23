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

from ...models.condition_embedders.condition_embedder_clip_image import CLIPImageProjection as _CLIPImageProjection
from ...utils import deprecate


# Deprecated alias kept for backward compatibility. The warning is raised from `__new__` rather than `__init__` so
# that the parent's `__init__` signature, which `ConfigMixin` inspects when loading a config, stays intact.
class CLIPImageProjection(_CLIPImageProjection):
    def __new__(cls, *args, **kwargs):
        deprecate(
            "CLIPImageProjection",
            "1.0.0",
            "Importing `CLIPImageProjection` from `diffusers.pipelines.stable_diffusion.clip_image_project_model` is deprecated. "
            "Import it with `from diffusers import CLIPImageProjection` instead.",
        )
        return super().__new__(cls)
