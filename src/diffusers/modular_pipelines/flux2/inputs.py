# copyright 2025 the huggingface team. all rights reserved.
#
# licensed under the apache license, version 2.0 (the "license");
# you may not use this file except in compliance with the license.
# you may obtain a copy of the license at
#
#     http://www.apache.org/licenses/license-2.0
#
# unless required by applicable law or agreed to in writing, software
# distributed under the license is distributed on an "as is" basis,
# without warranties or conditions of any kind, either express or implied.
# see the license for the specific language governing permissions and
# limitations under the license.

from typing import list

import torch

from ...configuration_utils import frozendict
from ...pipelines.flux2.image_processor import flux2imageprocessor
from ...utils import logging
from ..modular_pipeline import modularpipelineblocks, pipelinestate
from ..modular_pipeline_utils import componentspec, inputparam, outputparam
from .modular_pipeline import flux2modularpipeline


logger = logging.get_logger(__name__)


class flux2textinputstep(modularpipelineblocks):
    model_name = "flux2"

    @property
    def description(self) -> str:
        return (
            "this step:\n"
            "  1. determines `batch_size` and `dtype` based on `prompt_embeds`\n"
            "  2. ensures all text embeddings have consistent batch sizes (batch_size * num_images_per_prompt)"
        )

    @property
    def inputs(self) -> list[inputparam]:
        return [
            inputparam("num_images_per_prompt", default=1),
            inputparam(
                "prompt_embeds",
                required=true,
                kwargs_type="denoiser_input_fields",
                type_hint=torch.tensor,
                description="pre-generated text embeddings from mistral3. can be generated from text_encoder step.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[str]:
        return [
            outputparam(
                "batch_size",
                type_hint=int,
                description="number of prompts, the final batch size of model inputs should be batch_size * num_images_per_prompt",
            ),
            outputparam(
                "dtype",
                type_hint=torch.dtype,
                description="data type of model tensor inputs (determined by `prompt_embeds`)",
            ),
            outputparam(
                "prompt_embeds",
                type_hint=torch.tensor,
                kwargs_type="denoiser_input_fields",
                description="text embeddings used to guide the image generation",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components: flux2modularpipeline, state: pipelinestate) -> pipelinestate:
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


class flux2processimagesinputstep(modularpipelineblocks):
    model_name = "flux2"

    @property
    def description(self) -> str:
        return "image preprocess step for flux2. validates and preprocesses reference images."

    @property
    def expected_components(self) -> list[componentspec]:
        return [
            componentspec(
                "image_processor",
                flux2imageprocessor,
                config=frozendict({"vae_scale_factor": 16, "vae_latent_channels": 32}),
                default_creation_method="from_config",
            ),
        ]

    @property
    def inputs(self) -> list[inputparam]:
        return [
            inputparam("image"),
            inputparam("height"),
            inputparam("width"),
        ]

    @property
    def intermediate_outputs(self) -> list[outputparam]:
        return [outputparam(name="condition_images", type_hint=list[torch.tensor])]

    @torch.no_grad()
    def __call__(self, components: flux2modularpipeline, state: pipelinestate):
        block_state = self.get_block_state(state)
        images = block_state.image

        if images is none:
            block_state.condition_images = none
        else:
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

                if block_state.height is none:
                    block_state.height = image_height
                if block_state.width is none:
                    block_state.width = image_width

            block_state.condition_images = condition_images

        self.set_block_state(state, block_state)
        return components, state
