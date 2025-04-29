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
from .lora.lora_base import LORA_WEIGHT_NAME, LORA_WEIGHT_NAME_SAFE, LoraBaseMixin  # noqa: F401


def fuse_text_encoder_lora(text_encoder, lora_scale=1.0, safe_fusing=False, adapter_names=None):
    from .lora.lora_base import fuse_text_encoder_lora

    deprecation_message = "Importing `fuse_text_encoder_lora()` from diffusers.loaders.lora_base has been deprecated. Please use `from diffusers.loaders.lora.lora_base import fuse_text_encoder_lora` instead."
    deprecate("diffusers.loaders.lora_base.fuse_text_encoder_lora", "0.36", deprecation_message)

    return fuse_text_encoder_lora(
        text_encoder, lora_scale=lora_scale, safe_fusing=safe_fusing, adapter_names=adapter_names
    )


def unfuse_text_encoder_lora(text_encoder):
    from .lora.lora_base import unfuse_text_encoder_lora

    deprecation_message = "Importing `unfuse_text_encoder_lora()` from diffusers.loaders.lora_base has been deprecated. Please use `from diffusers.loaders.lora.lora_base import unfuse_text_encoder_lora` instead."
    deprecate("diffusers.loaders.lora_base.unfuse_text_encoder_lora", "0.36", deprecation_message)

    return unfuse_text_encoder_lora(text_encoder)


def set_adapters_for_text_encoder(
    adapter_names,
    text_encoder=None,
    text_encoder_weights=None,
):
    from .lora.lora_base import set_adapters_for_text_encoder

    deprecation_message = "Importing `set_adapters_for_text_encoder()` from diffusers.loaders.lora_base has been deprecated. Please use `from diffusers.loaders.lora.lora_base import set_adapters_for_text_encoder` instead."
    deprecate("diffusers.loaders.lora_base.set_adapters_for_text_encoder", "0.36", deprecation_message)

    return set_adapters_for_text_encoder(
        adapter_names=adapter_names, text_encoder=text_encoder, text_encoder_weights=text_encoder_weights
    )


def disable_lora_for_text_encoder(text_encoder=None):
    from .lora.lora_base import disable_lora_for_text_encoder

    deprecation_message = "Importing `disable_lora_for_text_encoder()` from diffusers.loaders.lora_base has been deprecated. Please use `from diffusers.loaders.lora.lora_base import disable_lora_for_text_encoder` instead."
    deprecate("diffusers.loaders.lora_base.disable_lora_for_text_encoder", "0.36", deprecation_message)

    return disable_lora_for_text_encoder(text_encoder=text_encoder)


def enable_lora_for_text_encoder(text_encoder=None):
    from .lora.lora_base import enable_lora_for_text_encoder

    deprecation_message = "Importing `enable_lora_for_text_encoder()` from diffusers.loaders.lora_base has been deprecated. Please use `from diffusers.loaders.lora.lora_base import enable_lora_for_text_encoder` instead."
    deprecate("diffusers.loaders.lora_base.enable_lora_for_text_encoder", "0.36", deprecation_message)

    return enable_lora_for_text_encoder(text_encoder=text_encoder)


class LoraBaseMixin(LoraBaseMixin):
    def __init__(self, *args, **kwargs):
        deprecation_message = "Importing `LoraBaseMixin` from diffusers.loaders.lora_base has been deprecated. Please use `from diffusers.loaders.lora.lora_base import LoraBaseMixin` instead."
        deprecate("diffusers.loaders.lora_base.LoraBaseMixin", "0.36", deprecation_message)
        super().__init__(*args, **kwargs)
