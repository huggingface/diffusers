# Copyright 2025 Lightricks and The HuggingFace Team. All rights reserved.
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

from ...models.autoencoders.latent_upsampler_ltx2 import (
    RATIONAL_RESAMPLER_SCALE_MAPPING,  # noqa: F401  re-exported for back-compat
    PixelShuffleND,  # noqa: F401  re-exported for back-compat
    ResBlock,  # noqa: F401  re-exported for back-compat
)
from ...models.autoencoders.latent_upsampler_ltx2 import (
    LTX2LatentUpsamplerModel as _LTX2LatentUpsamplerModel,
)
from ...utils import deprecate


# The deprecation warning is emitted from ``__new__`` rather than ``__init__`` so the shim does not
# override the parent's ``__init__`` signature — ``ConfigMixin.extract_init_dict`` reflects on
# ``inspect.signature(cls.__init__)`` to decide which saved config keys to forward at
# ``from_pretrained`` time, and an ``__init__(self, *args, **kwargs)`` override would erase them all.
class LTX2LatentUpsamplerModel(_LTX2LatentUpsamplerModel):
    def __new__(cls, *args, **kwargs):
        deprecate(
            "LTX2LatentUpsamplerModel",
            "1.0.0",
            "Importing `LTX2LatentUpsamplerModel` from `diffusers.pipelines.ltx2.latent_upsampler` is "
            "deprecated. Import it from `diffusers.models.autoencoders` instead "
            "(or `from diffusers import LTX2LatentUpsamplerModel`).",
        )
        return super().__new__(cls)
