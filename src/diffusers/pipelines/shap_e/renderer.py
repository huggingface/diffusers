# Copyright 2025 Open AI and The HuggingFace Team. All rights reserved.
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

from ...models.others.renderer_shap_e import (  # noqa: F401
    BoundingBoxVolume,
    ImportanceRaySampler,
    MLPNeRFModelOutput,
    StratifiedRaySampler,
    VoidNeRFModel,
)
from ...models.others.renderer_shap_e import MLPNeRSTFModel as _MLPNeRSTFModel
from ...models.others.renderer_shap_e import ShapEParamsProjModel as _ShapEParamsProjModel
from ...models.others.renderer_shap_e import ShapERenderer as _ShapERenderer
from ...utils import deprecate


# Deprecated aliases kept for backward compatibility. The warning is raised from `__new__` rather than `__init__` so
# that the parent's `__init__` signature, which `ConfigMixin` inspects when loading a config, stays intact.
class ShapERenderer(_ShapERenderer):
    def __new__(cls, *args, **kwargs):
        deprecate(
            "ShapERenderer",
            "1.0.0",
            "Importing `ShapERenderer` from `diffusers.pipelines.shap_e.renderer` is deprecated. "
            "Import it with `from diffusers import ShapERenderer` instead.",
        )
        return super().__new__(cls)


class MLPNeRSTFModel(_MLPNeRSTFModel):
    def __new__(cls, *args, **kwargs):
        deprecate(
            "MLPNeRSTFModel",
            "1.0.0",
            "Importing `MLPNeRSTFModel` from `diffusers.pipelines.shap_e.renderer` is deprecated. "
            "Import it from `diffusers.models.others.renderer_shap_e` instead.",
        )
        return super().__new__(cls)


class ShapEParamsProjModel(_ShapEParamsProjModel):
    def __new__(cls, *args, **kwargs):
        deprecate(
            "ShapEParamsProjModel",
            "1.0.0",
            "Importing `ShapEParamsProjModel` from `diffusers.pipelines.shap_e.renderer` is deprecated. "
            "Import it from `diffusers.models.others.renderer_shap_e` instead.",
        )
        return super().__new__(cls)
