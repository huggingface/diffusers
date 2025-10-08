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
    FluxPrepareLatentsStep,
    FluxRoPEInputsStep,
    FluxSetTimestepsStep,
)
from .decoders import FluxDecodeStep
from .denoise import FluxDenoiseStep
from .encoders import FluxProcessImagesInputStep, FluxTextEncoderStep, FluxVaeEncoderDynamicStep
from .inputs import FluxInputsDynamicStep, FluxTextInputStep


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# vae encoder (run before before_denoise)
FluxImg2ImgVaeEncoderBlocks = InsertableDict(
    [
        ("preprocess", FluxProcessImagesInputStep()),
        ("encode", FluxVaeEncoderDynamicStep()),
    ]
)


class FluxImg2ImgVaeEncoderStep(SequentialPipelineBlocks):
    model_name = "flux"

    block_classes = FluxImg2ImgVaeEncoderBlocks.values()
    block_names = FluxImg2ImgVaeEncoderBlocks.keys()

    @property
    def description(self) -> str:
        return "Vae encoder step that preprocess andencode the image inputs into their latent representations."


class FluxAutoVaeEncoderStep(AutoPipelineBlocks):
    block_classes = [FluxImg2ImgVaeEncoderStep]
    block_names = ["img2img"]
    block_trigger_inputs = ["image"]

    @property
    def description(self):
        return (
            "Vae encoder step that encode the image inputs into their latent representations.\n"
            + "This is an auto pipeline block that works for img2img tasks.\n"
            + " - `FluxImg2ImgVaeEncoderStep` (img2img) is used when only `image` is provided."
            + " - if `image` is not provided, step will be skipped."
        )


# before_denoise: text2img
FluxBeforeDenoiseBlocks = InsertableDict(
    [
        ("prepare_latents", FluxPrepareLatentsStep()),
        ("set_timesteps", FluxSetTimestepsStep()),
        ("prepare_rope_inputs", FluxRoPEInputsStep()),
    ]
)


class FluxBeforeDenoiseStep(SequentialPipelineBlocks):
    block_classes = FluxBeforeDenoiseBlocks.values()
    block_names = FluxBeforeDenoiseBlocks.keys()

    @property
    def description(self):
        return "Before denoise step that prepares the inputs for the denoise step in text-to-image generation."


# before_denoise: img2img
FluxImg2ImgBeforeDenoiseBlocks = InsertableDict(
    [
        ("prepare_latents", FluxPrepareLatentsStep()),
        ("set_timesteps", FluxImg2ImgSetTimestepsStep()),
        ("prepare_img2img_latents", FluxImg2ImgPrepareLatentsStep()),
        ("prepare_rope_inputs", FluxRoPEInputsStep()),
    ]
)


class FluxImg2ImgBeforeDenoiseStep(SequentialPipelineBlocks):
    block_classes = FluxImg2ImgBeforeDenoiseBlocks.values()
    block_names = FluxImg2ImgBeforeDenoiseBlocks.keys()

    @property
    def description(self):
        return "Before denoise step that prepare the inputs for the denoise step for img2img task."


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


# decode: all task (text2img, img2img)
class FluxAutoDecodeStep(AutoPipelineBlocks):
    block_classes = [FluxDecodeStep]
    block_names = ["non-inpaint"]
    block_trigger_inputs = [None]

    @property
    def description(self):
        return "Decode step that decode the denoised latents into image outputs.\n - `FluxDecodeStep`"


# inputs: text2image/img2img
FluxImg2ImgBlocks = InsertableDict(
    [("text_inputs", FluxTextInputStep()), ("additional_inputs", FluxInputsDynamicStep())]
)


class FluxImg2ImgInputStep(SequentialPipelineBlocks):
    model_name = "flux"
    block_classes = FluxImg2ImgBlocks.values()
    block_names = FluxImg2ImgBlocks.keys()

    @property
    def description(self):
        return "Input step that prepares the inputs for the img2img denoising step. It:\n"
        " - make sure the text embeddings have consistent batch size as well as the additional inputs (`image_latents`).\n"
        " - update height/width based `image_latents`, patchify `image_latents`."


class FluxImageAutoInputStep(AutoPipelineBlocks):
    block_classes = [FluxImg2ImgInputStep, FluxTextInputStep]
    block_names = ["img2img", "text2image"]
    block_trigger_inputs = ["image_latents", None]

    @property
    def description(self):
        return (
            "Input step that standardize the inputs for the denoising step, e.g. make sure inputs have consistent batch size, and patchified. \n"
            " This is an auto pipeline block that works for text2image/img2img tasks.\n"
            + " - `FluxImg2ImgInputStep` (img2img) is used when `image_latents` is provided.\n"
            + " - `FluxTextInputStep` (text2image) is used when `image_latents` are not provided.\n"
        )


class FluxCoreDenoiseStep(SequentialPipelineBlocks):
    model_name = "flux"
    block_classes = [FluxImageAutoInputStep, FluxAutoBeforeDenoiseStep, FluxAutoDenoiseStep]
    block_names = ["input", "before_denoise", "denoise"]

    @property
    def description(self):
        return (
            "Core step that performs the denoising process. \n"
            + " - `FluxImageAutoInputStep` (input) standardizes the inputs for the denoising step.\n"
            + " - `FluxAutoBeforeDenoiseStep` (before_denoise) prepares the inputs for the denoising step.\n"
            + " - `FluxAutoDenoiseStep` (denoise) iteratively denoises the latents.\n"
            + "This step supports text-to-image and image-to-image tasks for Flux:\n"
            + " - for image-to-image generation, you need to provide `image_latents`\n"
            + " - for text-to-image generation, all you need to provide is prompt embeddings."
        )


# Auto blocks (text2image and img2img)
AUTO_BLOCKS = InsertableDict(
    [
        ("text_encoder", FluxTextEncoderStep()),
        ("image_encoder", FluxAutoVaeEncoderStep()),
        ("denoise", FluxCoreDenoiseStep()),
        ("decode", FluxDecodeStep()),
    ]
)


class FluxAutoBlocks(SequentialPipelineBlocks):
    model_name = "flux"

    block_classes = AUTO_BLOCKS.values()
    block_names = AUTO_BLOCKS.keys()

    @property
    def description(self):
        return (
            "Auto Modular pipeline for text-to-image and image-to-image using Flux.\n"
            + "- for text-to-image generation, all you need to provide is `prompt`\n"
            + "- for image-to-image generation, you need to provide either `image` or `image_latents`"
        )


TEXT2IMAGE_BLOCKS = InsertableDict(
    [
        ("text_encoder", FluxTextEncoderStep()),
        ("input", FluxTextInputStep()),
        ("prepare_latents", FluxPrepareLatentsStep()),
        ("set_timesteps", FluxSetTimestepsStep()),
        ("prepare_rope_inputs", FluxRoPEInputsStep()),
        ("denoise", FluxDenoiseStep()),
        ("decode", FluxDecodeStep()),
    ]
)

IMAGE2IMAGE_BLOCKS = InsertableDict(
    [
        ("text_encoder", FluxTextEncoderStep()),
        ("vae_encoder", FluxVaeEncoderDynamicStep()),
        ("input", FluxImg2ImgInputStep()),
        ("prepare_latents", FluxPrepareLatentsStep()),
        ("set_timesteps", FluxImg2ImgSetTimestepsStep()),
        ("prepare_img2img_latents", FluxImg2ImgPrepareLatentsStep()),
        ("prepare_rope_inputs", FluxRoPEInputsStep()),
        ("denoise", FluxDenoiseStep()),
        ("decode", FluxDecodeStep()),
    ]
)

ALL_BLOCKS = {"text2image": TEXT2IMAGE_BLOCKS, "img2img": IMAGE2IMAGE_BLOCKS, "auto": AUTO_BLOCKS}
