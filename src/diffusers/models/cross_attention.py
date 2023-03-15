# Copyright 2023 The HuggingFace Team. All rights reserved.
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
from attention_processor import (  # noqa: F401
    Attention,
    AttnAddedKVProcessor,
    LoRAAttnProcessor,
    LoRAXFormersAttnProcessor,
    SlicedAttnAddedKVProcessor,
    SlicedAttnProcessor,
    XFormersAttnProcessor,
    LoRALinearLayer,
    AttnProcessor2_0,
)
from attention_processor import (  # noqa: F401
    AttnProcessor as AttnProcessorRename,
    AttnProcessors as AttnProcessor,
)

from ..utils import deprecate


deprecate(
    "cross_attention",
    "0.18.0",
    "Importing from cross_attention is deprecated. Please import from attention_processor instead.",
    standard_warn=False,
)


class CrossAttention(Attention):
    def __init__(self, args, **kwargs):
        deprecation_message = f"{self.__class__.__name__} is deprecated and will be removed in `0.18.0`. Please use `from diffusers.models.attention_processor import {''.join(self.__class__.__name__.split('Cross'))} instead."
        deprecate("cross_attention", "0.18.0", deprecation_message, standard_warn=False)


class CrossAttnProcessor(AttnProcessorRename):
    def __init__(self, args, **kwargs):
        deprecation_message = f"{self.__class__.__name__} is deprecated and will be removed in `0.18.0`. Please use `from diffusers.models.attention_processor import {''.join(self.__class__.__name__.split('Cross'))} instead."
        deprecate("cross_attention", "0.18.0", deprecation_message, standard_warn=False)


class LoRACrossAttnProcessor(LoRAAttnProcessor):
    def __init__(self, args, **kwargs):
        deprecation_message = f"{self.__class__.__name__} is deprecated and will be removed in `0.18.0`. Please use `from diffusers.models.attention_processor import {''.join(self.__class__.__name__.split('Cross'))} instead."
        deprecate("cross_attention", "0.18.0", deprecation_message, standard_warn=False)


class CrossAttnAddedKVProcessor(AttnAddedKVProcessor):
    def __init__(self, args, **kwargs):
        deprecation_message = f"{self.__class__.__name__} is deprecated and will be removed in `0.18.0`. Please use `from diffusers.models.attention_processor import {''.join(self.__class__.__name__.split('Cross'))} instead."
        deprecate("cross_attention", "0.18.0", deprecation_message, standard_warn=False)


class XFormersCrossAttnProcessor(XFormersAttnProcessor):
    def __init__(self, args, **kwargs):
        deprecation_message = f"{self.__class__.__name__} is deprecated and will be removed in `0.18.0`. Please use `from diffusers.models.attention_processor import {''.join(self.__class__.__name__.split('Cross'))} instead."
        deprecate("cross_attention", "0.18.0", deprecation_message, standard_warn=False)


class LoRAXFormersCrossAttnProcessor(LoRAXFormersAttnProcessor):
    def __init__(self, args, **kwargs):
        deprecation_message = f"{self.__class__.__name__} is deprecated and will be removed in `0.18.0`. Please use `from diffusers.models.attention_processor import {''.join(self.__class__.__name__.split('Cross'))} instead."
        deprecate("cross_attention", "0.18.0", deprecation_message, standard_warn=False)


class SlicedCrossAttnProcessor(SlicedAttnProcessor):
    def __init__(self, args, **kwargs):
        deprecation_message = f"{self.__class__.__name__} is deprecated and will be removed in `0.18.0`. Please use `from diffusers.models.attention_processor import {''.join(self.__class__.__name__.split('Cross'))} instead."
        deprecate("cross_attention", "0.18.0", deprecation_message, standard_warn=False)


class SlicedCrossAttnAddedKVProcessor(SlicedAttnAddedKVProcessor):
    def __init__(self, args, **kwargs):
        deprecation_message = f"{self.__class__.__name__} is deprecated and will be removed in `0.18.0`. Please use `from diffusers.models.attention_processor import {''.join(self.__class__.__name__.split('Cross'))} instead."
        deprecate("cross_attention", "0.18.0", deprecation_message, standard_warn=False)
