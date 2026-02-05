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

from typing import List

import PIL.Image
import torch

from ...utils import logging
from ..modular_pipeline import AutoPipelineBlocks, SequentialPipelineBlocks
from ..modular_pipeline_utils import InsertableDict, OutputParam
from .before_denoise import (
    Flux2KleinBaseRoPEInputsStep,
    Flux2PrepareImageLatentsStep,
    Flux2PrepareLatentsStep,
    Flux2RoPEInputsStep,
    Flux2SetTimestepsStep,
)
from .decoders import Flux2DecodeStep, Flux2UnpackLatentsStep
from .denoise import Flux2KleinBaseDenoiseStep, Flux2KleinDenoiseStep
from .encoders import (
    Flux2KleinBaseTextEncoderStep,
    Flux2KleinTextEncoderStep,
    Flux2VaeEncoderStep,
)
from .inputs import (
    Flux2KleinBaseTextInputStep,
    Flux2ProcessImagesInputStep,
    Flux2TextInputStep,
)


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

################
# VAE encoder
################

Flux2KleinVaeEncoderBlocks = InsertableDict(
    [
        ("preprocess", Flux2ProcessImagesInputStep()),
        ("encode", Flux2VaeEncoderStep()),
    ]
)


class Flux2KleinVaeEncoderSequentialStep(SequentialPipelineBlocks):
    model_name = "flux2"

    block_classes = Flux2KleinVaeEncoderBlocks.values()
    block_names = Flux2KleinVaeEncoderBlocks.keys()

    @property
    def description(self) -> str:
        return "VAE encoder step that preprocesses and encodes the image inputs into their latent representations."


class Flux2KleinAutoVaeEncoderStep(AutoPipelineBlocks):
    block_classes = [Flux2KleinVaeEncoderSequentialStep]
    block_names = ["img_conditioning"]
    block_trigger_inputs = ["image"]

    @property
    def description(self):
        return (
            "VAE encoder step that encodes the image inputs into their latent representations.\n"
            "This is an auto pipeline block that works for image conditioning tasks.\n"
            " - `Flux2KleinVaeEncoderSequentialStep` is used when `image` is provided.\n"
            " - If `image` is not provided, step will be skipped."
        )


###
### Core denoise
###

Flux2KleinCoreDenoiseBlocks = InsertableDict(
    [
        ("input", Flux2TextInputStep()),
        ("prepare_image_latents", Flux2PrepareImageLatentsStep()),
        ("prepare_latents", Flux2PrepareLatentsStep()),
        ("set_timesteps", Flux2SetTimestepsStep()),
        ("prepare_rope_inputs", Flux2RoPEInputsStep()),
        ("denoise", Flux2KleinDenoiseStep()),
        ("after_denoise", Flux2UnpackLatentsStep()),
    ]
)


class Flux2KleinCoreDenoiseStep(SequentialPipelineBlocks):
    model_name = "flux2-klein"

    block_classes = Flux2KleinCoreDenoiseBlocks.values()
    block_names = Flux2KleinCoreDenoiseBlocks.keys()

    @property
    def description(self):
        return (
            "Core denoise step that performs the denoising process for Flux2-Klein (distilled model).\n"
            " - `Flux2KleinTextInputStep` (input) standardizes the text inputs (prompt_embeds) for the denoising step.\n"
            " - `Flux2PrepareImageLatentsStep` (prepare_image_latents) prepares the image latents  and image_latent_ids for the denoising step.\n"
            " - `Flux2PrepareLatentsStep` (prepare_latents) prepares the initial latents (latents) and latent_ids for the denoising step.\n"
            " - `Flux2SetTimestepsStep` (set_timesteps) sets the timesteps for the denoising step.\n"
            " - `Flux2RoPEInputsStep` (prepare_rope_inputs) prepares the RoPE inputs (txt_ids) for the denoising step.\n"
            " - `Flux2KleinDenoiseStep` (denoise) iteratively denoises the latents.\n"
            " - `Flux2UnpackLatentsStep` (after_denoise) unpacks the latents from the denoising step.\n"
        )

    @property
    def outputs(self):
        return [
            OutputParam(
                name="latents",
                type_hint=torch.Tensor,
                description="The latents from the denoising step.",
            )
        ]


Flux2KleinBaseCoreDenoiseBlocks = InsertableDict(
    [
        ("input", Flux2KleinBaseTextInputStep()),
        ("prepare_latents", Flux2PrepareLatentsStep()),
        ("prepare_image_latents", Flux2PrepareImageLatentsStep()),
        ("set_timesteps", Flux2SetTimestepsStep()),
        ("prepare_rope_inputs", Flux2KleinBaseRoPEInputsStep()),
        ("denoise", Flux2KleinBaseDenoiseStep()),
        ("after_denoise", Flux2UnpackLatentsStep()),
    ]
)


class Flux2KleinBaseCoreDenoiseStep(SequentialPipelineBlocks):
    model_name = "flux2-klein"
    block_classes = Flux2KleinBaseCoreDenoiseBlocks.values()
    block_names = Flux2KleinBaseCoreDenoiseBlocks.keys()

    @property
    def description(self):
        return "Core denoise step that performs the denoising process for Flux2-Klein (base model)."
        return (
            "Core denoise step that performs the denoising process for Flux2-Klein (base model).\n"
            " - `Flux2KleinBaseTextInputStep` (input) standardizes the text inputs (prompt_embeds + negative_prompt_embeds) for the denoising step.\n"
            " - `Flux2PrepareImageLatentsStep` (prepare_image_latents) prepares the image latents and image_latent_ids for the denoising step.\n"
            " - `Flux2PrepareLatentsStep` (prepare_latents) prepares the initial latents (latents) and latent_ids for the denoising step.\n"
            " - `Flux2SetTimestepsStep` (set_timesteps) sets the timesteps for the denoising step.\n"
            " - `Flux2KleinBaseRoPEInputsStep` (prepare_rope_inputs) prepares the RoPE inputs (txt_ids + negative_txt_ids) for the denoising step.\n"
            " - `Flux2KleinBaseDenoiseStep` (denoise) iteratively denoises the latents using Classifier-Free Guidance.\n"
            " - `Flux2UnpackLatentsStep` (after_denoise) unpacks the latents from the denoising step.\n"
        )

    @property
    def outputs(self):
        return [
            OutputParam(
                name="latents",
                type_hint=torch.Tensor,
                description="The latents from the denoising step.",
            )
        ]


###
### Auto blocks
###
class Flux2KleinAutoBlocks(SequentialPipelineBlocks):
    model_name = "flux2-klein"
    block_classes = [
        Flux2KleinTextEncoderStep(),
        Flux2KleinAutoVaeEncoderStep(),
        Flux2KleinCoreDenoiseStep(),
        Flux2DecodeStep(),
    ]
    block_names = ["text_encoder", "vae_encoder", "denoise", "decode"]

    @property
    def description(self):
        return (
            "Auto blocks that perform the text-to-image and image-conditioned generation using Flux2-Klein.\n"
            + " - for image-conditioned generation, you need to provide `image` (list of PIL images).\n"
            + " - for text-to-image generation, all you need to provide is `prompt`.\n"
        )

    @property
    def outputs(self):
        return [
            OutputParam(
                name="images",
                type_hint=List[PIL.Image.Image],
                description="The images from the decoding step.",
            )
        ]


class Flux2KleinBaseAutoBlocks(SequentialPipelineBlocks):
    model_name = "flux2-klein"
    block_classes = [
        Flux2KleinBaseTextEncoderStep(),
        Flux2KleinAutoVaeEncoderStep(),
        Flux2KleinBaseCoreDenoiseStep(),
        Flux2DecodeStep(),
    ]
    block_names = ["text_encoder", "vae_encoder", "denoise", "decode"]

    @property
    def description(self):
        return (
            "Auto blocks that perform the text-to-image and image-conditioned generation using Flux2-Klein (base model).\n"
            + " - for image-conditioned generation, you need to provide `image` (list of PIL images).\n"
            + " - for text-to-image generation, all you need to provide is `prompt`.\n"
        )

    @property
    def outputs(self):
        return [
            OutputParam(
                name="images",
                type_hint=List[PIL.Image.Image],
                description="The images from the decoding step.",
            )
        ]
