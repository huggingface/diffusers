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
from typing import Dict

import torch

from .loaders.lora import LoraLoaderMixin, StableDiffusionXLLoraLoaderMixin
from .loaders.single_file import FromOriginalControlnetMixin, FromOriginalVAEMixin, FromSingleFileMixin
from .loaders.textual_inversion import TextualInversionLoaderMixin
from .loaders.unet import UNet2DConditionLoadersMixin
from .loaders.utils import AttnProcsLayers, PatchedLoraProjection
from .utils import deprecate


def text_encoder_attn_modules(text_encoder):
    deprecation_message = "Importing `text_encoder_attn_modules` from `diffusers.loaders` is depcrecated and will be removed in v1.0.0. Import it as `from diffusers.loaders.utils import text_encoder_attn_modules`, instead."
    deprecate("text_encoder_attn_modules", "1.0.0", deprecation_message, standard_warn=False)
    from .loaders.utils import text_encoder_attn_modules

    return text_encoder_attn_modules(text_encoder)


def text_encoder_mlp_modules(text_encoder):
    deprecation_message = "Importing `text_encoder_mlp_modules` from `diffusers.loaders` is depcrecated and will be removed in v1.0.0. Import it as `from diffusers.loaders.utils import text_encoder_mlp_modules`, instead."
    deprecate("text_encoder_mlp_modules", "1.0.0", deprecation_message, standard_warn=False)
    from .loaders.utils import text_encoder_mlp_modules

    return text_encoder_mlp_modules(text_encoder)


def text_encoder_lora_state_dict(text_encoder):
    deprecation_message = "Importing `text_encoder_lora_state_dict` from `diffusers.loaders` is depcrecated and will be removed in v1.0.0. Import it as `from diffusers.loaders.utils import text_encoder_lora_state_dict`, instead."
    deprecate("text_encoder_lora_state_dict", "1.0.0", deprecation_message, standard_warn=False)
    from .loaders.utils import text_encoder_lora_state_dict

    return text_encoder_lora_state_dict(text_encoder)


class PatchedLoraProjection(PatchedLoraProjection):
    def __init__(self, regular_linear_layer, lora_scale=1, network_alpha=None, rank=4, dtype=None):
        deprecation_message = "Importing `PatchedLoraProjection` from `diffusers.loaders` is depcrecated and will be removed in v1.0.0. Import it as `from diffusers.loaders.utils import PatchedLoraProjection`, instead."
        deprecate("PatchedLoraProjection", "1.0.0", deprecation_message, standard_warn=False)


class AttnProcsLayers(AttnProcsLayers):
    def __init__(self, state_dict: Dict[str, torch.Tensor]):
        deprecation_message = "Importing `AttnProcsLayers` from `diffusers.loaders` is depcrecated and will be removed in v1.0.0. Import it as `from diffusers.loaders.utils import AttnProcsLayers`, instead."
        deprecate("AttnProcsLayers", "1.0.0", deprecation_message, standard_warn=False)


class UNet2DConditionLoadersMixin(UNet2DConditionLoadersMixin):
    deprecation_message = "Importing `UNet2DConditionLoadersMixin` from `diffusers.loaders` is depcrecated and will be removed in v1.0.0. Import it as `from diffusers import UNet2DConditionLoadersMixin`, instead."
    deprecate("UNet2DConditionLoadersMixin", "1.0.0", deprecation_message, standard_warn=False)


class TextualInversionLoaderMixin(TextualInversionLoaderMixin):
    deprecation_message = "Importing `TextualInversionLoaderMixin` from `diffusers.loaders` is depcrecated and will be removed in v1.0.0. Import it as `from diffusers import TextualInversionLoaderMixin`, instead."
    deprecate("TextualInversionLoaderMixin", "1.0.0", deprecation_message, standard_warn=False)


class LoraLoaderMixin(LoraLoaderMixin):
    deprecation_message = "Importing `LoraLoaderMixin` from `diffusers.loaders` is depcrecated and will be removed in v1.0.0. Import it as `from diffusers import LoraLoaderMixin`, instead."
    deprecate("LoraLoaderMixin", "1.0.0", deprecation_message, standard_warn=False)


class StableDiffusionXLLoraLoaderMixin(StableDiffusionXLLoraLoaderMixin):
    deprecation_message = "Importing `StableDiffusionXLLoraLoaderMixin` from `diffusers.loaders` is depcrecated and will be removed in v1.0.0. Import it as `from diffusers import StableDiffusionXLLoraLoaderMixin`, instead."
    deprecate("StableDiffusionXLLoraLoaderMixin", "1.0.0", deprecation_message, standard_warn=False)


class FromSingleFileMixin(FromSingleFileMixin):
    deprecation_message = "Importing `FromSingleFileMixin` from `diffusers.loaders` is depcrecated and will be removed in v1.0.0. Import it as `from diffusers import FromSingleFileMixin`, instead."
    deprecate("FromSingleFileMixin", "1.0.0", deprecation_message, standard_warn=False)


class FromOriginalVAEMixin(FromOriginalVAEMixin):
    deprecation_message = "Importing `FromOriginalVAEMixin` from `diffusers.loaders` is depcrecated and will be removed in v1.0.0. Import it as `from diffusers import FromOriginalVAEMixin`, instead."
    deprecate("FromOriginalVAEMixin", "1.0.0", deprecation_message, standard_warn=False)


class FromOriginalControlnetMixin(FromOriginalControlnetMixin):
    deprecation_message = "Importing `FromOriginalControlnetMixin` from `diffusers.loaders` is depcrecated and will be removed in v1.0.0. Import it as `from diffusers import FromOriginalControlnetMixin`, instead."
    deprecate("FromOriginalControlnetMixin", "1.0.0", deprecation_message, standard_warn=False)
