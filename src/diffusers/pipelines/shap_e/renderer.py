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

from ...models.others.renderer_shap_e import (
    BoundingBoxVolume,  # noqa: F401  re-exported for back-compat
    ImportanceRaySampler,  # noqa: F401
    MLPNeRFModelOutput,  # noqa: F401
    MLPNeRSTFModel,  # noqa: F401
    ShapEParamsProjModel,  # noqa: F401
    StratifiedRaySampler,  # noqa: F401
    VoidNeRFModel,  # noqa: F401
)
from ...models.others.renderer_shap_e import (
    ShapERenderer as _ShapERenderer,
)
from ...utils import deprecate


# The deprecation warning is emitted from ``__new__`` rather than ``__init__`` so the shim does not
# override the parent's ``__init__`` signature — ``ConfigMixin.extract_init_dict`` reflects on
# ``inspect.signature(cls.__init__)`` to decide which saved config keys to forward at
# ``from_pretrained`` time, and an ``__init__(self, *args, **kwargs)`` override would erase them all.
class ShapERenderer(_ShapERenderer):
    def __new__(cls, *args, **kwargs):
        deprecate(
            "ShapERenderer",
            "1.0.0",
            "Importing `ShapERenderer` from `diffusers.pipelines.shap_e.renderer` is deprecated. "
            "Import it from `diffusers.models.others` instead.",
        )
        return super().__new__(cls)
