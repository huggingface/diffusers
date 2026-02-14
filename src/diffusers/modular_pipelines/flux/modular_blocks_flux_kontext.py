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
    FluxKontextRoPEInputsStep,
    FluxPrepareLatentsStep,
    FluxRoPEInputsStep,
    FluxSetTimestepsStep,
)
from .decoders import FluxDecodeStep
from .denoise import FluxKontextDenoiseStep
from .encoders import (
    FluxKontextProcessImagesInputStep,
    FluxTextEncoderStep,
    FluxVaeEncoderStep,
)
from .inputs import (
    FluxKontextAdditionalInputsStep,
    FluxKontextSetResolutionStep,
    FluxTextInputStep,
)


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# Flux Kontext vae encoder (run before before_denoise)
# auto_docstring
class FluxKontextVaeEncoderStep(SequentialPipelineBlocks):
    model_name = "flux-kontext"

    block_classes = [FluxKontextProcessImagesInputStep(), FluxVaeEncoderStep(sample_mode="argmax")]
    block_names = ["preprocess", "encode"]

    @property
    def description(self) -> str:
        return "Vae encoder step that preprocess andencode the image inputs into their latent representations."


# auto_docstring
class FluxKontextAutoVaeEncoderStep(AutoPipelineBlocks):
    model_name = "flux-kontext"

    block_classes = [FluxKontextVaeEncoderStep]
    block_names = ["image_conditioned"]
    block_trigger_inputs = ["image"]

    @property
    def description(self):
        return (
            "Vae encoder step that encode the image inputs into their latent representations.\n"
            + "This is an auto pipeline block that works for image-conditioned tasks.\n"
            + " - `FluxKontextVaeEncoderStep` (image_conditioned) is used when only `image` is provided."
            + " - if `image` is not provided, step will be skipped."
        )


# before_denoise: text2img
# auto_docstring
class FluxKontextBeforeDenoiseStep(SequentialPipelineBlocks):
    model_name = "flux-kontext"

    block_classes = [FluxPrepareLatentsStep(), FluxSetTimestepsStep(), FluxRoPEInputsStep()]
    block_names = ["prepare_latents", "set_timesteps", "prepare_rope_inputs"]

    @property
    def description(self):
        return "Before denoise step that prepares the inputs for the denoise step for Flux Kontext\n"
        "for text-to-image tasks."


# before_denoise: image-conditioned
# auto_docstring
class FluxKontextImageConditionedBeforeDenoiseStep(SequentialPipelineBlocks):
    model_name = "flux-kontext"

    block_classes = [FluxPrepareLatentsStep(), FluxSetTimestepsStep(), FluxKontextRoPEInputsStep()]
    block_names = ["prepare_latents", "set_timesteps", "prepare_rope_inputs"]

    @property
    def description(self):
        return (
            "Before denoise step that prepare the inputs for the denoise step for Flux Kontext\n"
            "for image-conditioned tasks."
        )


# auto_docstring
class FluxKontextAutoBeforeDenoiseStep(AutoPipelineBlocks):
    model_name = "flux-kontext"

    block_classes = [FluxKontextImageConditionedBeforeDenoiseStep, FluxKontextBeforeDenoiseStep]
    block_names = ["image_conditioned", "text2image"]
    block_trigger_inputs = ["image_latents", None]

    @property
    def description(self):
        return (
            "Before denoise step that prepare the inputs for the denoise step.\n"
            + "This is an auto pipeline block that works for text2image.\n"
            + " - `FluxKontextBeforeDenoiseStep` (text2image) is used.\n"
            + " - `FluxKontextImageConditionedBeforeDenoiseStep` (image_conditioned) is used when only `image_latents` is provided.\n"
        )


# inputs: Flux Kontext
# auto_docstring
class FluxKontextInputStep(SequentialPipelineBlocks):
    model_name = "flux-kontext"
    block_classes = [FluxKontextSetResolutionStep(), FluxTextInputStep(), FluxKontextAdditionalInputsStep()]
    block_names = ["set_resolution", "text_inputs", "additional_inputs"]

    @property
    def description(self):
        return (
            "Input step that prepares the inputs for the both text2img and img2img denoising step. It:\n"
            " - make sure the text embeddings have consistent batch size as well as the additional inputs (`image_latents`).\n"
            " - update height/width based `image_latents`, patchify `image_latents`."
        )


# auto_docstring
class FluxKontextAutoInputStep(AutoPipelineBlocks):
    model_name = "flux-kontext"
    block_classes = [FluxKontextInputStep, FluxTextInputStep]
    block_names = ["image_conditioned", "text2image"]
    block_trigger_inputs = ["image_latents", None]

    @property
    def description(self):
        return (
            "Input step that standardize the inputs for the denoising step, e.g. make sure inputs have consistent batch size, and patchified. \n"
            " This is an auto pipeline block that works for text2image/img2img tasks.\n"
            + " - `FluxKontextInputStep` (image_conditioned) is used when `image_latents` is provided.\n"
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
            "Core step that performs the denoising process for Flux Kontext.\n"
            + "This step supports text-to-image and image-conditioned tasks for Flux Kontext:\n"
            + " - for image-conditioned generation, you need to provide `image_latents`\n"
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


# auto_docstring
class FluxKontextAutoBlocks(SequentialPipelineBlocks):
    model_name = "flux-kontext"

    block_classes = AUTO_BLOCKS_KONTEXT.values()
    block_names = AUTO_BLOCKS_KONTEXT.keys()
    _workflow_map = {
        "image_conditioned": {"image": True, "prompt": True},
        "text2image": {"prompt": True},
    }

    @property
    def description(self):
        return "Modular pipeline for image-to-image using Flux Kontext."
