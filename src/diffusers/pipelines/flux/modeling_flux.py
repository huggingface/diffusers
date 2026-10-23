# Copyright 2025 Black Forest Labs and The HuggingFace Team. All rights reserved.
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

from ...models.condition_embedders.condition_embedder_flux_redux import ReduxImageEncoder as _ReduxImageEncoder
from ...models.condition_embedders.condition_embedder_flux_redux import (
    ReduxImageEncoderOutput,  # noqa: F401
)
from ...utils import deprecate


# Deprecated alias kept for backward compatibility. The warning is raised from `__new__` rather than `__init__` so
# that the parent's `__init__` signature, which `ConfigMixin` inspects when loading a config, stays intact.
class ReduxImageEncoder(_ReduxImageEncoder):
    def __new__(cls, *args, **kwargs):
        deprecate(
            "ReduxImageEncoder",
            "1.0.0",
            "Importing `ReduxImageEncoder` from `diffusers.pipelines.flux.modeling_flux` is deprecated. "
            "Import it with `from diffusers import ReduxImageEncoder` instead.",
        )
        return super().__new__(cls)
