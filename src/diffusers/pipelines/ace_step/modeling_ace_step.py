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
    AceStepAudioTokenDetokenizer as _AceStepAudioTokenDetokenizer,
)
from ...models.autoencoders.audio_tokenizer_ace_step import AceStepAudioTokenizer as _AceStepAudioTokenizer
from ...models.condition_embedders.condition_embedder_ace_step import (
    AceStepConditionEncoder as _AceStepConditionEncoder,
)
from ...models.condition_embedders.condition_embedder_ace_step import AceStepLyricEncoder as _AceStepLyricEncoder
from ...models.condition_embedders.condition_embedder_ace_step import AceStepTimbreEncoder as _AceStepTimbreEncoder
from ...utils import deprecate


# Deprecated aliases kept for backward compatibility. The warning is raised from `__new__` rather than `__init__` so
# that the parent's `__init__` signature, which `ConfigMixin` inspects when loading a config, stays intact.
class AceStepLyricEncoder(_AceStepLyricEncoder):
    def __new__(cls, *args, **kwargs):
        deprecate(
            "AceStepLyricEncoder",
            "1.0.0",
            "Importing `AceStepLyricEncoder` from `diffusers.pipelines.ace_step.modeling_ace_step` is deprecated. "
            "Import it from `diffusers.models.condition_embedders.condition_embedder_ace_step` instead.",
        )
        return super().__new__(cls)


class AceStepTimbreEncoder(_AceStepTimbreEncoder):
    def __new__(cls, *args, **kwargs):
        deprecate(
            "AceStepTimbreEncoder",
            "1.0.0",
            "Importing `AceStepTimbreEncoder` from `diffusers.pipelines.ace_step.modeling_ace_step` is deprecated. "
            "Import it from `diffusers.models.condition_embedders.condition_embedder_ace_step` instead.",
        )
        return super().__new__(cls)


class AceStepConditionEncoder(_AceStepConditionEncoder):
    def __new__(cls, *args, **kwargs):
        deprecate(
            "AceStepConditionEncoder",
            "1.0.0",
            "Importing `AceStepConditionEncoder` from `diffusers.pipelines.ace_step.modeling_ace_step` is deprecated. "
            "Import it with `from diffusers import AceStepConditionEncoder` instead.",
        )
        return super().__new__(cls)


class AceStepAudioTokenizer(_AceStepAudioTokenizer):
    def __new__(cls, *args, **kwargs):
        deprecate(
            "AceStepAudioTokenizer",
            "1.0.0",
            "Importing `AceStepAudioTokenizer` from `diffusers.pipelines.ace_step.modeling_ace_step` is deprecated. "
            "Import it with `from diffusers import AceStepAudioTokenizer` instead.",
        )
        return super().__new__(cls)


class AceStepAudioTokenDetokenizer(_AceStepAudioTokenDetokenizer):
    def __new__(cls, *args, **kwargs):
        deprecate(
            "AceStepAudioTokenDetokenizer",
            "1.0.0",
            "Importing `AceStepAudioTokenDetokenizer` from `diffusers.pipelines.ace_step.modeling_ace_step` is deprecated. "
            "Import it with `from diffusers import AceStepAudioTokenDetokenizer` instead.",
        )
        return super().__new__(cls)
