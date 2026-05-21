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

from ...models.others.image_normalizer_stable_unclip import (
    StableUnCLIPImageNormalizer as _StableUnCLIPImageNormalizer,
)
from ...utils import deprecate


# The deprecation warning is emitted from ``__new__`` rather than ``__init__`` so the shim does not
# override the parent's ``__init__`` signature — ``ConfigMixin.extract_init_dict`` reflects on
# ``inspect.signature(cls.__init__)`` to decide which saved config keys to forward at
# ``from_pretrained`` time, and an ``__init__(self, *args, **kwargs)`` override would erase them all.
class StableUnCLIPImageNormalizer(_StableUnCLIPImageNormalizer):
    def __new__(cls, *args, **kwargs):
        deprecate(
            "StableUnCLIPImageNormalizer",
            "1.0.0",
            "Importing `StableUnCLIPImageNormalizer` from `diffusers.pipelines.stable_diffusion.stable_unclip_image_normalizer` "
            "is deprecated. Import it from `diffusers.models.others` instead.",
        )
        return super().__new__(cls)
