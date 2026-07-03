# Copyright 2025 The ACE-Step Team and The HuggingFace Team. All rights reserved.
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

from ...models.autoencoders.audio_tokenizer_ace_step import (
    AceStepAttentionPooler,  # noqa: F401  re-exported for back-compat
    _AceStepResidualFSQ,  # noqa: F401  re-exported for back-compat
)
from ...models.autoencoders.audio_tokenizer_ace_step import (
    AceStepAudioTokenDetokenizer as _AceStepAudioTokenDetokenizer,
)
from ...models.autoencoders.audio_tokenizer_ace_step import (
    AceStepAudioTokenizer as _AceStepAudioTokenizer,
)
from ...models.condition_embedders.condition_encoder_ace_step import (
    AceStepConditionEncoder as _AceStepConditionEncoder,
)
from ...models.condition_embedders.condition_encoder_ace_step import (
    AceStepEncoderLayer,  # noqa: F401  re-exported for back-compat
    _pack_sequences,  # noqa: F401  re-exported for back-compat
)
from ...models.condition_embedders.condition_encoder_ace_step import (
    AceStepLyricEncoder as _AceStepLyricEncoder,
)
from ...models.condition_embedders.condition_encoder_ace_step import (
    AceStepTimbreEncoder as _AceStepTimbreEncoder,
)
from ...utils import deprecate


# The deprecation warning is emitted from ``__new__`` rather than ``__init__`` so the shim does not
# override the parent's ``__init__`` signature — ``ConfigMixin.extract_init_dict`` reflects on
# ``inspect.signature(cls.__init__)`` to decide which saved config keys to forward at
# ``from_pretrained`` time, and an ``__init__(self, *args, **kwargs)`` override would erase them all.
class AceStepAudioTokenizer(_AceStepAudioTokenizer):
    def __new__(cls, *args, **kwargs):
        deprecate(
            "AceStepAudioTokenizer",
            "1.0.0",
            "Importing `AceStepAudioTokenizer` from `diffusers.pipelines.ace_step.modeling_ace_step` is deprecated. "
            "Import it from `diffusers.models.autoencoders` instead "
            "(or `from diffusers import AceStepAudioTokenizer`).",
        )
        return super().__new__(cls)


class AceStepAudioTokenDetokenizer(_AceStepAudioTokenDetokenizer):
    def __new__(cls, *args, **kwargs):
        deprecate(
            "AceStepAudioTokenDetokenizer",
            "1.0.0",
            "Importing `AceStepAudioTokenDetokenizer` from `diffusers.pipelines.ace_step.modeling_ace_step` is deprecated. "
            "Import it from `diffusers.models.autoencoders` instead "
            "(or `from diffusers import AceStepAudioTokenDetokenizer`).",
        )
        return super().__new__(cls)


class AceStepConditionEncoder(_AceStepConditionEncoder):
    def __new__(cls, *args, **kwargs):
        deprecate(
            "AceStepConditionEncoder",
            "1.0.0",
            "Importing `AceStepConditionEncoder` from `diffusers.pipelines.ace_step.modeling_ace_step` is deprecated. "
            "Import it from `diffusers.models.condition_embedders` instead "
            "(or `from diffusers import AceStepConditionEncoder`).",
        )
        return super().__new__(cls)


class AceStepLyricEncoder(_AceStepLyricEncoder):
    def __new__(cls, *args, **kwargs):
        deprecate(
            "AceStepLyricEncoder",
            "1.0.0",
            "Importing `AceStepLyricEncoder` from `diffusers.pipelines.ace_step.modeling_ace_step` is deprecated. "
            "Import it from `diffusers.models.condition_embedders.condition_encoder_ace_step` instead.",
        )
        return super().__new__(cls)


class AceStepTimbreEncoder(_AceStepTimbreEncoder):
    def __new__(cls, *args, **kwargs):
        deprecate(
            "AceStepTimbreEncoder",
            "1.0.0",
            "Importing `AceStepTimbreEncoder` from `diffusers.pipelines.ace_step.modeling_ace_step` is deprecated. "
            "Import it from `diffusers.models.condition_embedders.condition_encoder_ace_step` instead.",
        )
        return super().__new__(cls)
