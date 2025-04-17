# Copyright 2024 The HuggingFace Team. All rights reserved.
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


from ..utils import deprecate
from .lora.lora_pipeline import (
    TEXT_ENCODER_NAME,  # noqa: F401
    TRANSFORMER_NAME,  # noqa: F401
    UNET_NAME,  # noqa: F401
    AmusedLoraLoaderMixin,
    AuraFlowLoraLoaderMixin,
    CogVideoXLoraLoaderMixin,
    CogView4LoraLoaderMixin,
    FluxLoraLoaderMixin,
    HunyuanVideoLoraLoaderMixin,
    LTXVideoLoraLoaderMixin,
    Lumina2LoraLoaderMixin,
    Mochi1LoraLoaderMixin,
    SanaLoraLoaderMixin,
    SD3LoraLoaderMixin,
    StableDiffusionLoraLoaderMixin,
    StableDiffusionXLLoraLoaderMixin,
    WanLoraLoaderMixin,
)


class StableDiffusionLoraLoaderMixin(StableDiffusionLoraLoaderMixin):
    def __init__(self, *args, **kwargs):
        deprecation_message = "Importing `StableDiffusionLoraLoaderMixin` from diffusers.loaders.lora_pipeline has been deprecated. Please use `from diffusers.loaders.lora.lora_pipeline import StableDiffusionLoraLoaderMixin` instead."
        deprecate("diffusers.loaders.lora_pipeline.StableDiffusionLoraLoaderMixin", "0.36", deprecation_message)
        super().__init__(*args, **kwargs)


class StableDiffusionXLLoraLoaderMixin(StableDiffusionXLLoraLoaderMixin):
    def __init__(self, *args, **kwargs):
        deprecation_message = "Importing `StableDiffusionXLLoraLoaderMixin` from diffusers.loaders.lora_pipeline has been deprecated. Please use `from diffusers.loaders.lora.lora_pipeline import StableDiffusionXLLoraLoaderMixin` instead."
        deprecate("diffusers.loaders.lora_pipeline.StableDiffusionXLLoraLoaderMixin", "0.36", deprecation_message)
        super().__init__(*args, **kwargs)


class SD3LoraLoaderMixin(SD3LoraLoaderMixin):
    def __init__(self, *args, **kwargs):
        deprecation_message = "Importing `SD3LoraLoaderMixin` from diffusers.loaders.lora_pipeline has been deprecated. Please use `from diffusers.loaders.lora.lora_pipeline import SD3LoraLoaderMixin` instead."
        deprecate("diffusers.loaders.lora_pipeline.SD3LoraLoaderMixin", "0.36", deprecation_message)
        super().__init__(*args, **kwargs)


class AuraFlowLoraLoaderMixin(AuraFlowLoraLoaderMixin):
    def __init__(self, *args, **kwargs):
        deprecation_message = "Importing `AuraFlowLoraLoaderMixin` from diffusers.loaders.lora_pipeline has been deprecated. Please use `from diffusers.loaders.lora.lora_pipeline import AuraFlowLoraLoaderMixin` instead."
        deprecate("diffusers.loaders.lora_pipeline.AuraFlowLoraLoaderMixin", "0.36", deprecation_message)
        super().__init__(*args, **kwargs)


class FluxLoraLoaderMixin(FluxLoraLoaderMixin):
    def __init__(self, *args, **kwargs):
        deprecation_message = "Importing `FluxLoraLoaderMixin` from diffusers.loaders.lora_pipeline has been deprecated. Please use `from diffusers.loaders.lora.lora_pipeline import FluxLoraLoaderMixin` instead."
        deprecate("diffusers.loaders.lora_pipeline.FluxLoraLoaderMixin", "0.36", deprecation_message)
        super().__init__(*args, **kwargs)


class AmusedLoraLoaderMixin(AmusedLoraLoaderMixin):
    def __init__(self, *args, **kwargs):
        deprecation_message = "Importing `AmusedLoraLoaderMixin` from diffusers.loaders.lora_pipeline has been deprecated. Please use `from diffusers.loaders.lora.lora_pipeline import AmusedLoraLoaderMixin` instead."
        deprecate("diffusers.loaders.lora_pipeline.AmusedLoraLoaderMixin", "0.36", deprecation_message)
        super().__init__(*args, **kwargs)


class CogVideoXLoraLoaderMixin(CogVideoXLoraLoaderMixin):
    def __init__(self, *args, **kwargs):
        deprecation_message = "Importing `CogVideoXLoraLoaderMixin` from diffusers.loaders.lora_pipeline has been deprecated. Please use `from diffusers.loaders.lora.lora_pipeline import CogVideoXLoraLoaderMixin` instead."
        deprecate("diffusers.loaders.lora_pipeline.CogVideoXLoraLoaderMixin", "0.36", deprecation_message)
        super().__init__(*args, **kwargs)


class Mochi1LoraLoaderMixin(Mochi1LoraLoaderMixin):
    def __init__(self, *args, **kwargs):
        deprecation_message = "Importing `Mochi1LoraLoaderMixin` from diffusers.loaders.lora_pipeline has been deprecated. Please use `from diffusers.loaders.lora.lora_pipeline import Mochi1LoraLoaderMixin` instead."
        deprecate("diffusers.loaders.lora_pipeline.Mochi1LoraLoaderMixin", "0.36", deprecation_message)
        super().__init__(*args, **kwargs)


class LTXVideoLoraLoaderMixin(LTXVideoLoraLoaderMixin):
    def __init__(self, *args, **kwargs):
        deprecation_message = "Importing `LTXVideoLoraLoaderMixin` from diffusers.loaders.lora_pipeline has been deprecated. Please use `from diffusers.loaders.lora.lora_pipeline import LTXVideoLoraLoaderMixin` instead."
        deprecate("diffusers.loaders.lora_pipeline.LTXVideoLoraLoaderMixin", "0.36", deprecation_message)
        super().__init__(*args, **kwargs)


class SanaLoraLoaderMixin(SanaLoraLoaderMixin):
    def __init__(self, *args, **kwargs):
        deprecation_message = "Importing `SanaLoraLoaderMixin` from diffusers.loaders.lora_pipeline has been deprecated. Please use `from diffusers.loaders.lora.lora_pipeline import SanaLoraLoaderMixin` instead."
        deprecate("diffusers.loaders.lora_pipeline.SanaLoraLoaderMixin", "0.36", deprecation_message)
        super().__init__(*args, **kwargs)


class HunyuanVideoLoraLoaderMixin(HunyuanVideoLoraLoaderMixin):
    def __init__(self, *args, **kwargs):
        deprecation_message = "Importing `HunyuanVideoLoraLoaderMixin` from diffusers.loaders.lora_pipeline has been deprecated. Please use `from diffusers.loaders.lora.lora_pipeline import HunyuanVideoLoraLoaderMixin` instead."
        deprecate("diffusers.loaders.lora_pipeline.HunyuanVideoLoraLoaderMixin", "0.36", deprecation_message)
        super().__init__(*args, **kwargs)


class Lumina2LoraLoaderMixin(Lumina2LoraLoaderMixin):
    def __init__(self, *args, **kwargs):
        deprecation_message = "Importing `Lumina2LoraLoaderMixin` from diffusers.loaders.lora_pipeline has been deprecated. Please use `from diffusers.loaders.lora.lora_pipeline import Lumina2LoraLoaderMixin` instead."
        deprecate("diffusers.loaders.lora_pipeline.Lumina2LoraLoaderMixin", "0.36", deprecation_message)
        super().__init__(*args, **kwargs)


class WanLoraLoaderMixin(WanLoraLoaderMixin):
    def __init__(self, *args, **kwargs):
        deprecation_message = "Importing `WanLoraLoaderMixin` from diffusers.loaders.lora_pipeline has been deprecated. Please use `from diffusers.loaders.lora.lora_pipeline import WanLoraLoaderMixin` instead."
        deprecate("diffusers.loaders.lora_pipeline.WanLoraLoaderMixin", "0.36", deprecation_message)
        super().__init__(*args, **kwargs)


class CogView4LoraLoaderMixin(CogView4LoraLoaderMixin):
    def __init__(self, *args, **kwargs):
        deprecation_message = "Importing `CogView4LoraLoaderMixin` from diffusers.loaders.lora_pipeline has been deprecated. Please use `from diffusers.loaders.lora.lora_pipeline import CogView4LoraLoaderMixin` instead."
        deprecate("diffusers.loaders.lora_pipeline.CogView4LoraLoaderMixin", "0.36", deprecation_message)
        super().__init__(*args, **kwargs)


class LoraLoaderMixin(StableDiffusionLoraLoaderMixin):
    def __init__(self, *args, **kwargs):
        deprecation_message = "LoraLoaderMixin is deprecated and this will be removed in a future version. Please use `StableDiffusionLoraLoaderMixin`, instead."
        deprecate("LoraLoaderMixin", "1.0.0", deprecation_message)
        super().__init__(*args, **kwargs)
