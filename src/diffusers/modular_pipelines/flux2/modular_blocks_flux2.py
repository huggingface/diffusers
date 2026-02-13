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
    Flux2PrepareGuidanceStep,
    Flux2PrepareImageLatentsStep,
    Flux2PrepareLatentsStep,
    Flux2RoPEInputsStep,
    Flux2SetTimestepsStep,
)
from .decoders import Flux2DecodeStep, Flux2UnpackLatentsStep
from .denoise import Flux2DenoiseStep
from .encoders import (
    Flux2RemoteTextEncoderStep,
    Flux2TextEncoderStep,
    Flux2VaeEncoderStep,
)
from .inputs import (
    Flux2ProcessImagesInputStep,
    Flux2TextInputStep,
)


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


Flux2VaeEncoderBlocks = InsertableDict(
    [
        ("preprocess", Flux2ProcessImagesInputStep()),
        ("encode", Flux2VaeEncoderStep()),
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


Flux2CoreDenoiseBlocks = InsertableDict(
    [
        ("input", Flux2TextInputStep()),
        ("prepare_image_latents", Flux2PrepareImageLatentsStep()),
        ("prepare_latents", Flux2PrepareLatentsStep()),
        ("set_timesteps", Flux2SetTimestepsStep()),
        ("prepare_guidance", Flux2PrepareGuidanceStep()),
        ("prepare_rope_inputs", Flux2RoPEInputsStep()),
        ("denoise", Flux2DenoiseStep()),
        ("after_denoise", Flux2UnpackLatentsStep()),
    ]
)


class Flux2CoreDenoiseStep(SequentialPipelineBlocks):
    model_name = "flux2"

    block_classes = Flux2CoreDenoiseBlocks.values()
    block_names = Flux2CoreDenoiseBlocks.keys()

    @property
    def description(self):
        return (
            "Core denoise step that performs the denoising process for Flux2-dev.\n"
            " - `Flux2TextInputStep` (input) standardizes the text inputs (prompt_embeds) for the denoising step.\n"
            " - `Flux2PrepareImageLatentsStep` (prepare_image_latents) prepares the image latents and image_latent_ids for the denoising step.\n"
            " - `Flux2PrepareLatentsStep` (prepare_latents) prepares the initial latents (latents) and latent_ids for the denoising step.\n"
            " - `Flux2SetTimestepsStep` (set_timesteps) sets the timesteps for the denoising step.\n"
            " - `Flux2PrepareGuidanceStep` (prepare_guidance) prepares the guidance tensor for the denoising step.\n"
            " - `Flux2RoPEInputsStep` (prepare_rope_inputs) prepares the RoPE inputs (txt_ids) for the denoising step.\n"
            " - `Flux2DenoiseStep` (denoise) iteratively denoises the latents.\n"
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


AUTO_BLOCKS = InsertableDict(
    [
        ("text_encoder", Flux2TextEncoderStep()),
        ("vae_encoder", Flux2AutoVaeEncoderStep()),
        ("denoise", Flux2CoreDenoiseStep()),
        ("decode", Flux2DecodeStep()),
    ]
)


REMOTE_AUTO_BLOCKS = InsertableDict(
    [
        ("text_encoder", Flux2RemoteTextEncoderStep()),
        ("vae_encoder", Flux2AutoVaeEncoderStep()),
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

    @property
    def outputs(self):
        return [
            OutputParam(
                name="images",
                type_hint=List[PIL.Image.Image],
                description="The images from the decoding step.",
            )
        ]


TEXT2IMAGE_BLOCKS = InsertableDict(
    [
        ("text_encoder", Flux2TextEncoderStep()),
        ("text_input", Flux2TextInputStep()),
        ("prepare_latents", Flux2PrepareLatentsStep()),
        ("set_timesteps", Flux2SetTimestepsStep()),
        ("prepare_guidance", Flux2PrepareGuidanceStep()),
        ("prepare_rope_inputs", Flux2RoPEInputsStep()),
        ("denoise", Flux2DenoiseStep()),
        ("after_denoise", Flux2UnpackLatentsStep()),
        ("decode", Flux2DecodeStep()),
    ]
)

IMAGE_CONDITIONED_BLOCKS = InsertableDict(
    [
        ("text_encoder", Flux2TextEncoderStep()),
        ("text_input", Flux2TextInputStep()),
        ("preprocess_images", Flux2ProcessImagesInputStep()),
        ("vae_encoder", Flux2VaeEncoderStep()),
        ("prepare_image_latents", Flux2PrepareImageLatentsStep()),
        ("prepare_latents", Flux2PrepareLatentsStep()),
        ("set_timesteps", Flux2SetTimestepsStep()),
        ("prepare_guidance", Flux2PrepareGuidanceStep()),
        ("prepare_rope_inputs", Flux2RoPEInputsStep()),
        ("denoise", Flux2DenoiseStep()),
        ("after_denoise", Flux2UnpackLatentsStep()),
        ("decode", Flux2DecodeStep()),
    ]
)

ALL_BLOCKS = {
    "text2image": TEXT2IMAGE_BLOCKS,
    "image_conditioned": IMAGE_CONDITIONED_BLOCKS,
    "auto": AUTO_BLOCKS,
    "remote": REMOTE_AUTO_BLOCKS,
}
