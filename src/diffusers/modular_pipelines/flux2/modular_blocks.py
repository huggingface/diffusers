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
    Flux2PrepareImageLatentsStep,
    Flux2PrepareLatentsStep,
    Flux2RoPEInputsStep,
    Flux2SetTimestepsStep,
)
from .decoders import Flux2DecodeStep
from .denoise import Flux2DenoiseStep
from .encoders import (
    Flux2ProcessImagesInputStep,
    Flux2RemoteTextEncoderStep,
    Flux2TextEncoderStep,
    Flux2VaeEncoderStep,
)
from .inputs import (
    Flux2ImageInputStep,
    Flux2TextInputStep,
)


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


Flux2VaeEncoderBlocks = InsertableDict(
    [
        ("preprocess", Flux2ProcessImagesInputStep()),
        ("encode", Flux2VaeEncoderStep()),
        ("prepare_image_latents", Flux2PrepareImageLatentsStep()),
    ]
)


class Flux2VaeEncoderSequentialStep(SequentialPipelineBlocks):
    model_name = "flux2"

    block_classes = Flux2VaeEncoderBlocks.values()
    block_names = Flux2VaeEncoderBlocks.keys()

    @property
    def description(self) -> str:
        return "VAE encoder step that preprocesses, encodes, and prepares image latents for Flux2 conditioning."


class Flux2AutoVaeEncoderStep(AutoPipelineBlocks):
    block_classes = [Flux2VaeEncoderSequentialStep]
    block_names = ["img_conditioning"]
    block_trigger_inputs = ["image"]

    @property
    def description(self):
        return (
            "VAE encoder step that encodes the image inputs into their latent representations.\n"
            "This is an auto pipeline block that works for image conditioning tasks.\n"
            " - `Flux2VaeEncoderSequentialStep` is used when `image` is provided.\n"
            " - If `image` is not provided, step will be skipped."
        )


class Flux2AutoTextEncoderStep(AutoPipelineBlocks):
    block_classes = [Flux2RemoteTextEncoderStep, Flux2TextEncoderStep]
    block_names = ["remote", "local"]
    block_trigger_inputs = ["remote_text_encoder", None]

    @property
    def description(self):
        return (
            "Text encoder step that generates text embeddings to guide the image generation.\n"
            "This is an auto pipeline block that selects between local and remote text encoding.\n"
            " - `Flux2RemoteTextEncoderStep` is used when `remote_text_encoder=True`.\n"
            " - `Flux2TextEncoderStep` is used otherwise (default)."
        )


Flux2BeforeDenoiseBlocks = InsertableDict(
    [
        ("prepare_latents", Flux2PrepareLatentsStep()),
        ("set_timesteps", Flux2SetTimestepsStep()),
        ("prepare_rope_inputs", Flux2RoPEInputsStep()),
    ]
)


class Flux2BeforeDenoiseStep(SequentialPipelineBlocks):
    model_name = "flux2"

    block_classes = Flux2BeforeDenoiseBlocks.values()
    block_names = Flux2BeforeDenoiseBlocks.keys()

    @property
    def description(self):
        return "Before denoise step that prepares the inputs for the denoise step in Flux2 generation."


class Flux2AutoBeforeDenoiseStep(AutoPipelineBlocks):
    model_name = "flux2"
    block_classes = [Flux2BeforeDenoiseStep]
    block_names = ["before_denoise"]
    block_trigger_inputs = [None]

    @property
    def description(self):
        return (
            "Before denoise step that prepares the inputs for the denoise step.\n"
            "This is an auto pipeline block for Flux2.\n"
            " - `Flux2BeforeDenoiseStep` is used for both text-to-image and image-conditioned generation."
        )


class Flux2AutoDenoiseStep(AutoPipelineBlocks):
    block_classes = [Flux2DenoiseStep]
    block_names = ["denoise"]
    block_trigger_inputs = [None]

    @property
    def description(self) -> str:
        return (
            "Denoise step that iteratively denoises the latents. "
            "This is an auto pipeline block that works for Flux2 text-to-image and image-conditioned tasks."
            " - `Flux2DenoiseStep` (denoise) for text-to-image and image-conditioned tasks."
        )


class Flux2AutoDecodeStep(AutoPipelineBlocks):
    block_classes = [Flux2DecodeStep]
    block_names = ["decode"]
    block_trigger_inputs = [None]

    @property
    def description(self):
        return "Decode step that decodes the denoised latents into image outputs.\n - `Flux2DecodeStep`"


Flux2InputBlocks = InsertableDict(
    [
        ("text_inputs", Flux2TextInputStep()),
        ("image_inputs", Flux2ImageInputStep()),
    ]
)


class Flux2InputSequentialStep(SequentialPipelineBlocks):
    model_name = "flux2"
    block_classes = Flux2InputBlocks.values()
    block_names = Flux2InputBlocks.keys()

    @property
    def description(self):
        return (
            "Input step that prepares the inputs for the Flux2 denoising step. It:\n"
            " - Makes sure the text embeddings have consistent batch size.\n"
            " - Processes image latents if provided."
        )


class Flux2AutoInputStep(AutoPipelineBlocks):
    block_classes = [Flux2InputSequentialStep, Flux2TextInputStep]
    block_names = ["img_conditioning", "text2image"]
    block_trigger_inputs = ["image_latents", None]

    @property
    def description(self):
        return (
            "Input step that standardizes the inputs for the denoising step.\n"
            "This is an auto pipeline block that works for text-to-image/image-conditioned tasks.\n"
            " - `Flux2InputSequentialStep` is used when `image_latents` is provided.\n"
            " - `Flux2TextInputStep` is used when `image_latents` is not provided.\n"
        )


class Flux2CoreDenoiseStep(SequentialPipelineBlocks):
    model_name = "flux2"
    block_classes = [Flux2AutoInputStep, Flux2AutoBeforeDenoiseStep, Flux2AutoDenoiseStep]
    block_names = ["input", "before_denoise", "denoise"]

    @property
    def description(self):
        return (
            "Core step that performs the denoising process for Flux2. \n"
            " - `Flux2AutoInputStep` (input) standardizes the inputs for the denoising step.\n"
            " - `Flux2AutoBeforeDenoiseStep` (before_denoise) prepares the inputs for the denoising step.\n"
            " - `Flux2AutoDenoiseStep` (denoise) iteratively denoises the latents.\n"
            "This step supports text-to-image and image-conditioned tasks for Flux2:\n"
            " - For image-conditioned generation, you need to provide `packed_image_latents`.\n"
            " - For text-to-image generation, all you need to provide is prompt embeddings."
        )


AUTO_BLOCKS = InsertableDict(
    [
        ("text_encoder", Flux2AutoTextEncoderStep()),
        ("image_encoder", Flux2AutoVaeEncoderStep()),
        ("denoise", Flux2CoreDenoiseStep()),
        ("decode", Flux2DecodeStep()),
    ]
)


class Flux2AutoBlocks(SequentialPipelineBlocks):
    model_name = "flux2"

    block_classes = AUTO_BLOCKS.values()
    block_names = AUTO_BLOCKS.keys()

    @property
    def description(self):
        return (
            "Auto Modular pipeline for text-to-image and image-conditioned generation using Flux2.\n"
            "- For text-to-image generation, all you need to provide is `prompt`.\n"
            "- For image-conditioned generation, you need to provide `image` (list of PIL images)."
        )


TEXT2IMAGE_BLOCKS = InsertableDict(
    [
        ("text_encoder", Flux2TextEncoderStep()),
        ("input", Flux2TextInputStep()),
        ("prepare_latents", Flux2PrepareLatentsStep()),
        ("set_timesteps", Flux2SetTimestepsStep()),
        ("prepare_rope_inputs", Flux2RoPEInputsStep()),
        ("denoise", Flux2DenoiseStep()),
        ("decode", Flux2DecodeStep()),
    ]
)

IMAGE_CONDITIONED_BLOCKS = InsertableDict(
    [
        ("text_encoder", Flux2TextEncoderStep()),
        ("preprocess_images", Flux2ProcessImagesInputStep()),
        ("vae_encoder", Flux2VaeEncoderStep()),
        ("prepare_image_latents", Flux2PrepareImageLatentsStep()),
        ("input", Flux2InputSequentialStep()),
        ("prepare_latents", Flux2PrepareLatentsStep()),
        ("set_timesteps", Flux2SetTimestepsStep()),
        ("prepare_rope_inputs", Flux2RoPEInputsStep()),
        ("denoise", Flux2DenoiseStep()),
        ("decode", Flux2DecodeStep()),
    ]
)

ALL_BLOCKS = {
    "text2image": TEXT2IMAGE_BLOCKS,
    "image_conditioned": IMAGE_CONDITIONED_BLOCKS,
    "auto": AUTO_BLOCKS,
}
