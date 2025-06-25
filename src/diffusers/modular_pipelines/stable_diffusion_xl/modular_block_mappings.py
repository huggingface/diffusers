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

from ..modular_pipeline_utils import InsertableOrderedDict
from .before_denoise import (
    StableDiffusionXLAutoBeforeDenoiseStep,
    StableDiffusionXLControlNetInputStep,
    StableDiffusionXLControlNetUnionInputStep,
    StableDiffusionXLImg2ImgPrepareAdditionalConditioningStep,
    StableDiffusionXLImg2ImgPrepareLatentsStep,
    StableDiffusionXLImg2ImgSetTimestepsStep,
    StableDiffusionXLInpaintPrepareLatentsStep,
    StableDiffusionXLInputStep,
    StableDiffusionXLPrepareAdditionalConditioningStep,
    StableDiffusionXLPrepareLatentsStep,
    StableDiffusionXLSetTimestepsStep,
)
from .decoders import StableDiffusionXLAutoDecodeStep, StableDiffusionXLDecodeStep, StableDiffusionXLInpaintDecodeStep

# Import all the necessary block classes
from .denoise import (
    StableDiffusionXLAutoDenoiseStep,
    StableDiffusionXLControlNetDenoiseStep,
    StableDiffusionXLDenoiseLoop,
    StableDiffusionXLInpaintDenoiseLoop,
)
from .encoders import (
    StableDiffusionXLAutoIPAdapterStep,
    StableDiffusionXLAutoVaeEncoderStep,
    StableDiffusionXLInpaintVaeEncoderStep,
    StableDiffusionXLIPAdapterStep,
    StableDiffusionXLTextEncoderStep,
    StableDiffusionXLVaeEncoderStep,
)


# YiYi notes: comment out for now, work on this later
# block mapping
TEXT2IMAGE_BLOCKS = InsertableOrderedDict([
    ("text_encoder", StableDiffusionXLTextEncoderStep),
    ("input", StableDiffusionXLInputStep),
    ("set_timesteps", StableDiffusionXLSetTimestepsStep),
    ("prepare_latents", StableDiffusionXLPrepareLatentsStep),
    ("prepare_add_cond", StableDiffusionXLPrepareAdditionalConditioningStep),
    ("denoise", StableDiffusionXLDenoiseLoop),
    ("decode", StableDiffusionXLDecodeStep)
])

IMAGE2IMAGE_BLOCKS = InsertableOrderedDict([
    ("text_encoder", StableDiffusionXLTextEncoderStep),
    ("image_encoder", StableDiffusionXLVaeEncoderStep),
    ("input", StableDiffusionXLInputStep),
    ("set_timesteps", StableDiffusionXLImg2ImgSetTimestepsStep),
    ("prepare_latents", StableDiffusionXLImg2ImgPrepareLatentsStep),
    ("prepare_add_cond", StableDiffusionXLImg2ImgPrepareAdditionalConditioningStep),
    ("denoise", StableDiffusionXLDenoiseLoop),
    ("decode", StableDiffusionXLDecodeStep)
])

INPAINT_BLOCKS = InsertableOrderedDict([
    ("text_encoder", StableDiffusionXLTextEncoderStep),
    ("image_encoder", StableDiffusionXLInpaintVaeEncoderStep),
    ("input", StableDiffusionXLInputStep),
    ("set_timesteps", StableDiffusionXLImg2ImgSetTimestepsStep),
    ("prepare_latents", StableDiffusionXLInpaintPrepareLatentsStep),
    ("prepare_add_cond", StableDiffusionXLImg2ImgPrepareAdditionalConditioningStep),
    ("denoise", StableDiffusionXLInpaintDenoiseLoop),
    ("decode", StableDiffusionXLInpaintDecodeStep)
])

CONTROLNET_BLOCKS = InsertableOrderedDict([
    ("controlnet_input", StableDiffusionXLControlNetInputStep),
    ("denoise", StableDiffusionXLControlNetDenoiseStep),
])

CONTROLNET_UNION_BLOCKS = InsertableOrderedDict([
    ("controlnet_input", StableDiffusionXLControlNetUnionInputStep),
    ("denoise", StableDiffusionXLControlNetDenoiseStep),
])

IP_ADAPTER_BLOCKS = InsertableOrderedDict([
    ("ip_adapter", StableDiffusionXLIPAdapterStep),
])

AUTO_BLOCKS = InsertableOrderedDict([
    ("text_encoder", StableDiffusionXLTextEncoderStep),
    ("ip_adapter", StableDiffusionXLAutoIPAdapterStep),
    ("image_encoder", StableDiffusionXLAutoVaeEncoderStep),
    ("before_denoise", StableDiffusionXLAutoBeforeDenoiseStep),
    ("denoise", StableDiffusionXLAutoDenoiseStep),
    ("decode", StableDiffusionXLAutoDecodeStep)
])


SDXL_SUPPORTED_BLOCKS = {
    "text2img": TEXT2IMAGE_BLOCKS,
    "img2img": IMAGE2IMAGE_BLOCKS,
    "inpaint": INPAINT_BLOCKS,
    "controlnet": CONTROLNET_BLOCKS,
    "controlnet_union": CONTROLNET_UNION_BLOCKS,
    "ip_adapter": IP_ADAPTER_BLOCKS,
    "auto": AUTO_BLOCKS
}



