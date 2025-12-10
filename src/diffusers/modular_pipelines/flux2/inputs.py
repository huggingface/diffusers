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

import torch

from ...configuration_utils import FrozenDict
from ...pipelines.flux2.image_processor import Flux2ImageProcessor
from ...utils import logging
from ..modular_pipeline import ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam
from .modular_pipeline import Flux2ModularPipeline


logger = logging.get_logger(__name__)


class Flux2TextInputStep(ModularPipelineBlocks):
    model_name = "flux2"

    @property
    def description(self) -> str:
        return (
            "This step:\n"
            "  1. Determines `batch_size` and `dtype` based on `prompt_embeds`\n"
            "  2. Ensures all text embeddings have consistent batch sizes (batch_size * num_images_per_prompt)"
        )

    @property
    def inputs(self) -> List[InputParam]:
        return [
            InputParam("num_images_per_prompt", default=1),
            InputParam(
                "prompt_embeds",
                required=True,
                kwargs_type="denoiser_input_fields",
                type_hint=torch.Tensor,
                description="Pre-generated text embeddings from Mistral3. Can be generated from text_encoder step.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> List[str]:
        return [
            OutputParam(
                "batch_size",
                type_hint=int,
                description="Number of prompts, the final batch size of model inputs should be batch_size * num_images_per_prompt",
            ),
            OutputParam(
                "dtype",
                type_hint=torch.dtype,
                description="Data type of model tensor inputs (determined by `prompt_embeds`)",
            ),
            OutputParam(
                "prompt_embeds",
                type_hint=torch.Tensor,
                kwargs_type="denoiser_input_fields",
                description="Text embeddings used to guide the image generation",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components: Flux2ModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        block_state.batch_size = block_state.prompt_embeds.shape[0]
        block_state.dtype = block_state.prompt_embeds.dtype

        _, seq_len, _ = block_state.prompt_embeds.shape
        block_state.prompt_embeds = block_state.prompt_embeds.repeat(1, block_state.num_images_per_prompt, 1)
        block_state.prompt_embeds = block_state.prompt_embeds.view(
            block_state.batch_size * block_state.num_images_per_prompt, seq_len, -1
        )

        self.set_block_state(state, block_state)
        return components, state


class Flux2ProcessImagesInputStep(ModularPipelineBlocks):
    model_name = "flux2"

    @property
    def description(self) -> str:
        return "Image preprocess step for Flux2. Validates and preprocesses reference images."

    @property
    def expected_components(self) -> List[ComponentSpec]:
        return [
            ComponentSpec(
                "image_processor",
                Flux2ImageProcessor,
                config=FrozenDict({"vae_scale_factor": 16, "vae_latent_channels": 32}),
                default_creation_method="from_config",
            ),
        ]

    @property
    def inputs(self) -> List[InputParam]:
        return [
            InputParam("image"),
            InputParam("height"),
            InputParam("width"),
        ]

    @property
    def intermediate_outputs(self) -> List[OutputParam]:
        return [OutputParam(name="condition_images", type_hint=List[torch.Tensor])]

    @torch.no_grad()
    def __call__(self, components: Flux2ModularPipeline, state: PipelineState):
        block_state = self.get_block_state(state)
        images = block_state.image

        if images is None:
            block_state.condition_images = None
            self.set_block_state(state, block_state)
            return components, state

        if not isinstance(images, list):
            images = [images]

        condition_images = []
        for img in images:
            components.image_processor.check_image_input(img)

            image_width, image_height = img.size
            if image_width * image_height > 1024 * 1024:
                img = components.image_processor._resize_to_target_area(img, 1024 * 1024)
                image_width, image_height = img.size

            multiple_of = components.vae_scale_factor * 2
            image_width = (image_width // multiple_of) * multiple_of
            image_height = (image_height // multiple_of) * multiple_of
            condition_img = components.image_processor.preprocess(
                img, height=image_height, width=image_width, resize_mode="crop"
            )
            condition_images.append(condition_img)

            if block_state.height is None:
                block_state.height = image_height
            if block_state.width is None:
                block_state.width = image_width

        block_state.condition_images = condition_images

        self.set_block_state(state, block_state)
        return components, state
