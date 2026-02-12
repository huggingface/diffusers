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
    FluxKontextRoPEInputsStep,
    FluxPrepareLatentsStep,
    FluxRoPEInputsStep,
    FluxSetTimestepsStep,
)
from .decoders import FluxDecodeStep
from .denoise import FluxDenoiseStep, FluxKontextDenoiseStep
from .encoders import (
    FluxKontextProcessImagesInputStep,
    FluxProcessImagesInputStep,
    FluxTextEncoderStep,
    FluxVaeEncoderStep,
)
from .inputs import (
    FluxInputsDynamicStep,
    FluxKontextInputsDynamicStep,
    FluxKontextSetResolutionStep,
    FluxTextInputStep,
)


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# Flux Kontext vae encoder (run before before_denoise)

FluxKontextVaeEncoderBlocks = InsertableDict(
    [("preprocess", FluxKontextProcessImagesInputStep()), ("encode", FluxVaeEncoderStep(sample_mode="argmax"))]
)


class FluxKontextVaeEncoderStep(SequentialPipelineBlocks):
    model_name = "flux-kontext"

    block_classes = FluxKontextVaeEncoderBlocks.values()
    block_names = FluxKontextVaeEncoderBlocks.keys()

    @property
    def description(self) -> str:
        return "Vae encoder step that preprocess andencode the image inputs into their latent representations."


class FluxKontextAutoVaeEncoderStep(AutoPipelineBlocks):
    block_classes = [FluxKontextVaeEncoderStep]
    block_names = ["img2img"]
    block_trigger_inputs = ["image"]

    @property
    def description(self):
        return (
            "Vae encoder step that encode the image inputs into their latent representations.\n"
            + "This is an auto pipeline block that works for img2img tasks.\n"
            + " - `FluxKontextVaeEncoderStep` (img2img) is used when only `image` is provided."
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


# before_denoise: FluxKontext

FluxKontextBeforeDenoiseBlocks = InsertableDict(
    [
        ("prepare_latents", FluxPrepareLatentsStep()),
        ("set_timesteps", FluxSetTimestepsStep()),
        ("prepare_rope_inputs", FluxKontextRoPEInputsStep()),
    ]
)


class FluxKontextBeforeDenoiseStep(SequentialPipelineBlocks):
    block_classes = FluxKontextBeforeDenoiseBlocks.values()
    block_names = FluxKontextBeforeDenoiseBlocks.keys()

    @property
    def description(self):
        return (
            "Before denoise step that prepare the inputs for the denoise step\n"
            "for img2img/text2img task for Flux Kontext."
        )


class FluxKontextAutoBeforeDenoiseStep(AutoPipelineBlocks):
    block_classes = [FluxKontextBeforeDenoiseStep, FluxBeforeDenoiseStep]
    block_names = ["img2img", "text2image"]
    block_trigger_inputs = ["image_latents", None]

    @property
    def description(self):
        return (
            "Before denoise step that prepare the inputs for the denoise step.\n"
            + "This is an auto pipeline block that works for text2image.\n"
            + " - `FluxBeforeDenoiseStep` (text2image) is used.\n"
            + " - `FluxKontextBeforeDenoiseStep` (img2img) is used when only `image_latents` is provided.\n"
        )

# inputs: Flux Kontext

FluxKontextBlocks = InsertableDict(
    [
        ("set_resolution", FluxKontextSetResolutionStep()),
        ("text_inputs", FluxTextInputStep()),
        ("additional_inputs", FluxKontextInputsDynamicStep()),
    ]
)


class FluxKontextInputStep(SequentialPipelineBlocks):
    model_name = "flux-kontext"
    block_classes = FluxKontextBlocks.values()
    block_names = FluxKontextBlocks.keys()

    @property
    def description(self):
        return (
            "Input step that prepares the inputs for the both text2img and img2img denoising step. It:\n"
            " - make sure the text embeddings have consistent batch size as well as the additional inputs (`image_latents`).\n"
            " - update height/width based `image_latents`, patchify `image_latents`."
        )


class FluxKontextAutoInputStep(AutoPipelineBlocks):
    model_name = "flux-kontext"
    block_classes = [FluxKontextInputStep, FluxTextInputStep]
    block_names = ["img2img", "text2img"]
    block_trigger_inputs = ["image_latents", None]

    @property
    def description(self):
        return (
            "Input step that standardize the inputs for the denoising step, e.g. make sure inputs have consistent batch size, and patchified. \n"
            " This is an auto pipeline block that works for text2image/img2img tasks.\n"
            + " - `FluxKontextInputStep` (img2img) is used when `image_latents` is provided.\n"
            + " - `FluxKontextInputStep` is also capable of handling text2image task when `image_latent` isn't present."
        )


# auto_docstring
class FluxKontextCoreDenoiseStep(SequentialPipelineBlocks):
    model_name = "flux-kontext"
    block_classes = [FluxKontextAutoInputStep, FluxKontextAutoBeforeDenoiseStep, FluxKontextDenoiseStep]
    block_names = ["input", "before_denoise", "denoise"]

    @property
    def description(self):
        return (
            "Core step that performs the denoising process. \n"
            + " - `FluxKontextAutoInputStep` (input) standardizes the inputs for the denoising step.\n"
            + " - `FluxKontextAutoBeforeDenoiseStep` (before_denoise) prepares the inputs for the denoising step.\n"
            + " - `FluxKontextDenoiseStep` (denoise) iteratively denoises the latents.\n"
            + "This step supports text-to-image and image-to-image tasks for Flux-Kontext:\n"
            + " - for image-to-image generation, you need to provide `image_latents`\n"
            + " - for text-to-image generation, all you need to provide is prompt embeddings."
        )


AUTO_BLOCKS_KONTEXT = InsertableDict(
    [
        ("text_encoder", FluxTextEncoderStep()),
        ("vae_encoder", FluxKontextAutoVaeEncoderStep()),
        ("denoise", FluxKontextCoreDenoiseStep()),
        ("decode", FluxDecodeStep()),
    ]
)



class FluxKontextAutoBlocks(SequentialPipelineBlocks):
    model_name = "flux-kontext"

    block_classes = AUTO_BLOCKS_KONTEXT.values()
    block_names = AUTO_BLOCKS_KONTEXT.keys()
    _workflow_map = {
        "img2img": {"image": True, "prompt": True},
        "text2image": {"prompt": True},
    }

    @property
    def description(self):
        return (
            "Modular pipeline for image-to-image using Flux Kontext."
        )



# FLUX_KONTEXT_BLOCKS = InsertableDict(
#     [
#         ("text_encoder", FluxTextEncoderStep()),
#         ("vae_encoder", FluxVaeEncoderStep(sample_mode="argmax")),
#         ("input", FluxKontextInputStep()),
#         ("prepare_latents", FluxPrepareLatentsStep()),
#         ("set_timesteps", FluxSetTimestepsStep()),
#         ("prepare_rope_inputs", FluxKontextRoPEInputsStep()),
#         ("denoise", FluxKontextDenoiseStep()),
#         ("decode", FluxDecodeStep()),
#     ]
# )

