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

from ...utils import logging
from ..modular_pipeline import SequentialPipelineBlocks
from .before_denoise import StableDiffusionXLAutoBeforeDenoiseStep
from .decoders import StableDiffusionXLAutoDecodeStep
from .denoise import StableDiffusionXLAutoDenoiseStep
from .encoders import (
    StableDiffusionXLAutoIPAdapterStep,
    StableDiffusionXLAutoVaeEncoderStep,
    StableDiffusionXLTextEncoderStep,
)


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class StableDiffusionXLAutoPipeline(SequentialPipelineBlocks):
    block_classes = [
        StableDiffusionXLTextEncoderStep,
        StableDiffusionXLAutoIPAdapterStep,
        StableDiffusionXLAutoVaeEncoderStep,
        StableDiffusionXLAutoBeforeDenoiseStep,
        StableDiffusionXLAutoDenoiseStep,
        StableDiffusionXLAutoDecodeStep,
    ]
    block_names = ["text_encoder", "ip_adapter", "image_encoder", "before_denoise", "denoise", "decoder"]

    @property
    def description(self):
        return (
            "Auto Modular pipeline for text-to-image, image-to-image, inpainting, and controlnet tasks using Stable Diffusion XL.\n"
            + "- for image-to-image generation, you need to provide either `image` or `image_latents`\n"
            + "- for inpainting, you need to provide `mask_image` and `image`, optionally you can provide `padding_mask_crop` \n"
            + "- to run the controlnet workflow, you need to provide `control_image`\n"
            + "- to run the controlnet_union workflow, you need to provide `control_image` and `control_mode`\n"
            + "- to run the ip_adapter workflow, you need to provide `ip_adapter_image`\n"
            + "- for text-to-image generation, all you need to provide is `prompt`"
        )
