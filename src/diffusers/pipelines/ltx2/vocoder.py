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

from ...models.autoencoders.vocoder_ltx2 import (
    AntiAliasAct1d,  # noqa: F401  re-exported for back-compat
    CausalSTFT,  # noqa: F401
    DownSample1d,  # noqa: F401
    MelSTFT,  # noqa: F401
    ResBlock,  # noqa: F401
    SnakeBeta,  # noqa: F401
    UpSample1d,  # noqa: F401
    kaiser_sinc_filter1d,  # noqa: F401
)
from ...models.autoencoders.vocoder_ltx2 import (
    LTX2Vocoder as _LTX2Vocoder,
)
from ...models.autoencoders.vocoder_ltx2 import (
    LTX2VocoderWithBWE as _LTX2VocoderWithBWE,
)
from ...utils import deprecate


# The deprecation warning is emitted from ``__new__`` rather than ``__init__`` so the shim does not
# override the parent's ``__init__`` signature — ``ConfigMixin.extract_init_dict`` reflects on
# ``inspect.signature(cls.__init__)`` to decide which saved config keys to forward at
# ``from_pretrained`` time, and an ``__init__(self, *args, **kwargs)`` override would erase them all.
class LTX2Vocoder(_LTX2Vocoder):
    def __new__(cls, *args, **kwargs):
        deprecate(
            "LTX2Vocoder",
            "1.0.0",
            "Importing `LTX2Vocoder` from `diffusers.pipelines.ltx2.vocoder` is deprecated. "
            "Import it from `diffusers.models.autoencoders` instead "
            "(or `from diffusers import LTX2Vocoder`).",
        )
        return super().__new__(cls)


class LTX2VocoderWithBWE(_LTX2VocoderWithBWE):
    def __new__(cls, *args, **kwargs):
        deprecate(
            "LTX2VocoderWithBWE",
            "1.0.0",
            "Importing `LTX2VocoderWithBWE` from `diffusers.pipelines.ltx2.vocoder` is deprecated. "
            "Import it from `diffusers.models.autoencoders` instead "
            "(or `from diffusers import LTX2VocoderWithBWE`).",
        )
        return super().__new__(cls)
