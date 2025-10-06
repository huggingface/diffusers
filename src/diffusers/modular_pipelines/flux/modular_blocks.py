# Copyright 2025 The HuggingFace Team. All rights reserved.
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
    FluxImg2ImgPrepareLatentsStep,
    FluxImg2ImgSetTimestepsStep,
    FluxInputStep,
    FluxPrepareLatentsStep,
    FluxSetTimestepsStep,
)
from .decoders import FluxDecodeStep
from .denoise import FluxDenoiseStep
from .encoders import FluxTextEncoderStep, FluxVaeEncoderStep


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# vae encoder (run before before_denoise)
class FluxAutoVaeEncoderStep(AutoPipelineBlocks):
    block_classes = [FluxVaeEncoderStep]
    block_names = ["img2img"]
    block_trigger_inputs = ["image"]

    @property
    def description(self):
        return (
            "Vae encoder step that encode the image inputs into their latent representations.\n"
            + "This is an auto pipeline block that works for img2img tasks.\n"
            + " - `FluxVaeEncoderStep` (img2img) is used when only `image` is provided."
            + " - if `image` is provided, step will be skipped."
        )


# before_denoise: text2img, img2img
class FluxBeforeDenoiseStep(SequentialPipelineBlocks):
    block_classes = [
        FluxInputStep,
        FluxPrepareLatentsStep,
        FluxSetTimestepsStep,
    ]
    block_names = ["input", "prepare_latents", "set_timesteps"]

    @property
    def description(self):
        return (
            "Before denoise step that prepare the inputs for the denoise step.\n"
            + "This is a sequential pipeline blocks:\n"
            + " - `FluxInputStep` is used to adjust the batch size of the model inputs\n"
            + " - `FluxPrepareLatentsStep` is used to prepare the latents\n"
            + " - `FluxSetTimestepsStep` is used to set the timesteps\n"
        )


# before_denoise: img2img
class FluxImg2ImgBeforeDenoiseStep(SequentialPipelineBlocks):
    block_classes = [FluxInputStep, FluxImg2ImgSetTimestepsStep, FluxImg2ImgPrepareLatentsStep]
    block_names = ["input", "set_timesteps", "prepare_latents"]

    @property
    def description(self):
        return (
            "Before denoise step that prepare the inputs for the denoise step for img2img task.\n"
            + "This is a sequential pipeline blocks:\n"
            + " - `FluxInputStep` is used to adjust the batch size of the model inputs\n"
            + " - `FluxImg2ImgSetTimestepsStep` is used to set the timesteps\n"
            + " - `FluxImg2ImgPrepareLatentsStep` is used to prepare the latents\n"
        )


# before_denoise: all task (text2img, img2img)
class FluxAutoBeforeDenoiseStep(AutoPipelineBlocks):
    block_classes = [FluxImg2ImgBeforeDenoiseStep, FluxBeforeDenoiseStep]
    block_names = ["img2img", "text2image"]
    block_trigger_inputs = ["image_latents", None]

    @property
    def description(self):
        return (
            "Before denoise step that prepare the inputs for the denoise step.\n"
            + "This is an auto pipeline block that works for text2image.\n"
            + " - `FluxBeforeDenoiseStep` (text2image) is used.\n"
            + " - `FluxImg2ImgBeforeDenoiseStep` (img2img) is used when only `image_latents` is provided.\n"
        )


# denoise: text2image
class FluxAutoDenoiseStep(AutoPipelineBlocks):
    block_classes = [FluxDenoiseStep]
    block_names = ["denoise"]
    block_trigger_inputs = [None]

    @property
    def description(self) -> str:
        return (
            "Denoise step that iteratively denoise the latents. "
            "This is a auto pipeline block that works for text2image and img2img tasks."
            " - `FluxDenoiseStep` (denoise) for text2image and img2img tasks."
        )


# decode: all task (text2img, img2img, inpainting)
class FluxAutoDecodeStep(AutoPipelineBlocks):
    block_classes = [FluxDecodeStep]
    block_names = ["non-inpaint"]
    block_trigger_inputs = [None]

    @property
    def description(self):
        return "Decode step that decode the denoised latents into image outputs.\n - `FluxDecodeStep`"


class FluxCoreDenoiseStep(SequentialPipelineBlocks):
    block_classes = [FluxInputStep, FluxAutoBeforeDenoiseStep, FluxAutoDenoiseStep]
    block_names = ["input", "before_denoise", "denoise"]

    @property
    def description(self):
        return (
            "Core step that performs the denoising process. \n"
            + " - `FluxInputStep` (input) standardizes the inputs for the denoising step.\n"
            + " - `FluxAutoBeforeDenoiseStep` (before_denoise) prepares the inputs for the denoising step.\n"
            + " - `FluxAutoDenoiseStep` (denoise) iteratively denoises the latents.\n"
            + "This step support text-to-image and image-to-image tasks for Flux:\n"
            + " - for image-to-image generation, you need to provide `image_latents`\n"
            + " - for text-to-image generation, all you need to provide is prompt embeddings"
        )


# text2image
class FluxAutoBlocks(SequentialPipelineBlocks):
    block_classes = [
        FluxTextEncoderStep,
        FluxAutoVaeEncoderStep,
        FluxCoreDenoiseStep,
        FluxAutoDecodeStep,
    ]
    block_names = ["text_encoder", "image_encoder", "denoise", "decode"]

    @property
    def description(self):
        return (
            "Auto Modular pipeline for text-to-image and image-to-image using Flux.\n"
            + "- for text-to-image generation, all you need to provide is `prompt`\n"
            + "- for image-to-image generation, you need to provide either `image` or `image_latents`"
        )


TEXT2IMAGE_BLOCKS = InsertableDict(
    [
        ("text_encoder", FluxTextEncoderStep),
        ("input", FluxInputStep),
        ("prepare_latents", FluxPrepareLatentsStep),
        ("set_timesteps", FluxSetTimestepsStep),
        ("denoise", FluxDenoiseStep),
        ("decode", FluxDecodeStep),
    ]
)

IMAGE2IMAGE_BLOCKS = InsertableDict(
    [
        ("text_encoder", FluxTextEncoderStep),
        ("image_encoder", FluxVaeEncoderStep),
        ("input", FluxInputStep),
        ("set_timesteps", FluxImg2ImgSetTimestepsStep),
        ("prepare_latents", FluxImg2ImgPrepareLatentsStep),
        ("denoise", FluxDenoiseStep),
        ("decode", FluxDecodeStep),
    ]
)

AUTO_BLOCKS = InsertableDict(
    [
        ("text_encoder", FluxTextEncoderStep),
        ("image_encoder", FluxAutoVaeEncoderStep),
        ("denoise", FluxCoreDenoiseStep),
        ("decode", FluxAutoDecodeStep),
    ]
)


ALL_BLOCKS = {"text2image": TEXT2IMAGE_BLOCKS, "img2img": IMAGE2IMAGE_BLOCKS, "auto": AUTO_BLOCKS}
