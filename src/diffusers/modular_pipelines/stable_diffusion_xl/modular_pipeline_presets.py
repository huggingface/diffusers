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

from typing import Any, List, Optional, Tuple, Union, Dict
from ...utils import logging
from ..modular_pipeline import SequentialPipelineBlocks

from .denoise import StableDiffusionXLAutoDenoiseStep
from .before_denoise import StableDiffusionXLAutoBeforeDenoiseStep
from .after_denoise import StableDiffusionXLAutoDecodeStep
from .encoders import StableDiffusionXLTextEncoderStep, StableDiffusionXLAutoIPAdapterStep, StableDiffusionXLAutoVaeEncoderStep

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class StableDiffusionXLAutoPipeline(SequentialPipelineBlocks):
    block_classes = [StableDiffusionXLTextEncoderStep, StableDiffusionXLAutoIPAdapterStep, StableDiffusionXLAutoVaeEncoderStep, StableDiffusionXLAutoBeforeDenoiseStep, StableDiffusionXLAutoDenoiseStep, StableDiffusionXLAutoDecodeStep]
    block_names = ["text_encoder", "ip_adapter", "image_encoder", "before_denoise", "denoise", "after_denoise"]

    @property
    def description(self):
        return "Auto Modular pipeline for text-to-image, image-to-image, inpainting, and controlnet tasks using Stable Diffusion XL.\n" + \
               "- for image-to-image generation, you need to provide either `image` or `image_latents`\n" + \
               "- for inpainting, you need to provide `mask_image` and `image`, optionally you can provide `padding_mask_crop` \n" + \
               "- to run the controlnet workflow, you need to provide `control_image`\n" + \
               "- to run the controlnet_union workflow, you need to provide `control_image` and `control_mode`\n" + \
               "- to run the ip_adapter workflow, you need to provide `ip_adapter_image`\n" + \
               "- for text-to-image generation, all you need to provide is `prompt`"



# YiYi notes: comment out for now, work on this later
# # block mapping 
# TEXT2IMAGE_BLOCKS = OrderedDict([
#     ("text_encoder", StableDiffusionXLTextEncoderStep),
#     ("ip_adapter", StableDiffusionXLAutoIPAdapterStep),
#     ("input", StableDiffusionXLInputStep),
#     ("set_timesteps", StableDiffusionXLSetTimestepsStep),
#     ("prepare_latents", StableDiffusionXLPrepareLatentsStep),
#     ("prepare_add_cond", StableDiffusionXLPrepareAdditionalConditioningStep),
#     ("denoise", StableDiffusionXLDenoiseStep),
#     ("decode", StableDiffusionXLDecodeStep)
# ])

# IMAGE2IMAGE_BLOCKS = OrderedDict([
#     ("text_encoder", StableDiffusionXLTextEncoderStep),
#     ("ip_adapter", StableDiffusionXLAutoIPAdapterStep),
#     ("image_encoder", StableDiffusionXLVaeEncoderStep),
#     ("input", StableDiffusionXLInputStep),
#     ("set_timesteps", StableDiffusionXLImg2ImgSetTimestepsStep),
#     ("prepare_latents", StableDiffusionXLImg2ImgPrepareLatentsStep),
#     ("prepare_add_cond", StableDiffusionXLImg2ImgPrepareAdditionalConditioningStep),
#     ("denoise", StableDiffusionXLDenoiseStep),
#     ("decode", StableDiffusionXLDecodeStep)
# ])

# INPAINT_BLOCKS = OrderedDict([
#     ("text_encoder", StableDiffusionXLTextEncoderStep),
#     ("ip_adapter", StableDiffusionXLAutoIPAdapterStep),
#     ("image_encoder", StableDiffusionXLInpaintVaeEncoderStep),
#     ("input", StableDiffusionXLInputStep),
#     ("set_timesteps", StableDiffusionXLImg2ImgSetTimestepsStep),
#     ("prepare_latents", StableDiffusionXLInpaintPrepareLatentsStep),
#     ("prepare_add_cond", StableDiffusionXLImg2ImgPrepareAdditionalConditioningStep),
#     ("denoise", StableDiffusionXLDenoiseStep),
#     ("decode", StableDiffusionXLInpaintDecodeStep)
# ])

# CONTROLNET_BLOCKS = OrderedDict([
#     ("controlnet_input", StableDiffusionXLControlNetInputStep),
#     ("denoise", StableDiffusionXLControlNetDenoiseStep),
# ])

# CONTROLNET_UNION_BLOCKS = OrderedDict([
#     ("controlnet_input", StableDiffusionXLControlNetUnionInputStep),
#     ("denoise", StableDiffusionXLControlNetDenoiseStep),
# ])

# IP_ADAPTER_BLOCKS = OrderedDict([
#     ("ip_adapter", StableDiffusionXLIPAdapterStep),
# ])

# AUTO_BLOCKS = OrderedDict([
#     ("text_encoder", StableDiffusionXLTextEncoderStep),
#     ("ip_adapter", StableDiffusionXLAutoIPAdapterStep),
#     ("image_encoder", StableDiffusionXLAutoVaeEncoderStep),
#     ("before_denoise", StableDiffusionXLAutoBeforeDenoiseStep),
#     ("denoise", StableDiffusionXLAutoDenoiseStep),
#     ("decode", StableDiffusionXLAutoDecodeStep)
# ])

# AUTO_CORE_BLOCKS = OrderedDict([
#     ("before_denoise", StableDiffusionXLAutoBeforeDenoiseStep),
#     ("denoise", StableDiffusionXLAutoDenoiseStep),
# ])


# SDXL_SUPPORTED_BLOCKS = {
#     "text2img": TEXT2IMAGE_BLOCKS,
#     "img2img": IMAGE2IMAGE_BLOCKS,
#     "inpaint": INPAINT_BLOCKS,
#     "controlnet": CONTROLNET_BLOCKS,
#     "controlnet_union": CONTROLNET_UNION_BLOCKS,
#     "ip_adapter": IP_ADAPTER_BLOCKS,
#     "auto": AUTO_BLOCKS
# }


