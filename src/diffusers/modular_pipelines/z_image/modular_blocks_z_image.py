# Copyright 2025 Alibaba Z-Image Team and The HuggingFace Team. All rights reserved.
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
from ..modular_pipeline import AutoPipelineBlocks, SequentialPipelineBlocks
from ..modular_pipeline_utils import InsertableDict
from .before_denoise import (
    ZImageAdditionalInputsStep,
    ZImagePrepareLatentsStep,
    ZImagePrepareLatentswithImageStep,
    ZImageSetTimestepsStep,
    ZImageSetTimestepsWithStrengthStep,
    ZImageTextInputStep,
)
from .decoders import ZImageVaeDecoderStep
from .denoise import (
    ZImageDenoiseStep,
)
from .encoders import (
    ZImageTextEncoderStep,
    ZImageVaeImageEncoderStep,
)


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# ====================
# 1. DENOISE
# ====================

# text2image: inputs(text) -> set_timesteps -> prepare_latents -> denoise
# auto_docstring
class ZImageCoreDenoiseStep(SequentialPipelineBlocks):
    block_classes = [
        ZImageTextInputStep,
        ZImagePrepareLatentsStep,
        ZImageSetTimestepsStep,
        ZImageDenoiseStep,
    ]
    block_names = ["input", "prepare_latents", "set_timesteps", "denoise"]

    @property
    def description(self):
        return (
            "denoise block that takes encoded conditions and runs the denoising process.\n"
            + "This is a sequential pipeline blocks:\n"
            + " - `ZImageTextInputStep` is used to adjust the batch size of the model inputs\n"
            + " - `ZImagePrepareLatentsStep` is used to prepare the latents\n"
            + " - `ZImageSetTimestepsStep` is used to set the timesteps\n"
            + " - `ZImageDenoiseStep` is used to denoise the latents\n"
        )


# image2image: inputs(text + image_latents) -> prepare_latents -> set_timesteps -> set_timesteps_with_strength -> prepare_latents_with_image -> denoise 
# auto_docstring
class ZImageImage2ImageCoreDenoiseStep(SequentialPipelineBlocks):
    block_classes = [
        ZImageTextInputStep,
        ZImageAdditionalInputsStep(image_latent_inputs=["image_latents"]),
        ZImagePrepareLatentsStep,
        ZImageSetTimestepsStep,
        ZImageSetTimestepsWithStrengthStep,
        ZImagePrepareLatentswithImageStep,
        ZImageDenoiseStep,
    ]
    block_names = [
        "input",
        "additional_inputs",
        "prepare_latents",
        "set_timesteps",
        "set_timesteps_with_strength",
        "prepare_latents_with_image",
        "denoise",
    ]

    @property
    def description(self):
        return (
            "denoise block that takes encoded text and image latent conditions and runs the denoising process.\n"
            + "This is a sequential pipeline blocks:\n"
            + " - `ZImageTextInputStep` is used to adjust the batch size of the model inputs\n"
            + " - `ZImageAdditionalInputsStep` is used to adjust the batch size of the latent conditions\n"
            + " - `ZImagePrepareLatentsStep` is used to prepare the latents\n"
            + " - `ZImageSetTimestepsStep` is used to set the timesteps\n"
            + " - `ZImageSetTimestepsWithStrengthStep` is used to set the timesteps with strength\n"
            + " - `ZImagePrepareLatentswithImageStep` is used to prepare the latents with image\n"
            + " - `ZImageDenoiseStep` is used to denoise the latents\n"
        )


# auto_docstring
class ZImageAutoDenoiseStep(AutoPipelineBlocks):
    block_classes = [
        ZImageImage2ImageCoreDenoiseStep,
        ZImageCoreDenoiseStep,
    ]
    block_names = ["image2image", "text2image"]
    block_trigger_inputs = ["image_latents", None]

    @property
    def description(self) -> str:
        return (
            "Denoise step that iteratively denoise the latents. "
            "This is a auto pipeline block that works for text2image and image2image tasks."
            " - `ZImageCoreDenoiseStep` (text2image) for text2image tasks."
            " - `ZImageImage2ImageCoreDenoiseStep` (image2image) for image2image tasks."
            + " - if `image_latents` is provided, `ZImageImage2ImageCoreDenoiseStep` will be used.\n"
            + " - if `image_latents` is not provided, `ZImageCoreDenoiseStep` will be used.\n"
        )


# auto_docstring
class ZImageAutoVaeImageEncoderStep(AutoPipelineBlocks):
    block_classes = [ZImageVaeImageEncoderStep]
    block_names = ["vae_encoder"]
    block_trigger_inputs = ["image"]

    @property
    def description(self) -> str:
        return "Vae Image Encoder step that encode the image to generate the image latents"
        +"This is an auto pipeline block that works for image2image tasks."
        +" - `ZImageVaeImageEncoderStep` is used when `image` is provided."
        +" - if `image` is not provided, step will be skipped."


# auto_docstring
class ZImageAutoBlocks(SequentialPipelineBlocks):
    block_classes = [
        ZImageTextEncoderStep,
        ZImageAutoVaeImageEncoderStep,
        ZImageAutoDenoiseStep,
        ZImageVaeDecoderStep,
    ]
    block_names = ["text_encoder", "vae_encoder", "denoise", "decode"]
    _workflow_map = {
        "text2image": {"prompt": True},
        "image2image": {"image": True, "prompt": True},
    }

    @property
    def description(self) -> str:
        return "Auto Modular pipeline for text-to-image and image-to-image using ZImage."

