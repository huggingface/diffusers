# Copyright 2025 Qwen-Image Team and The HuggingFace Team. All rights reserved.
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

from ..modular_pipeline import ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import InputParam, OutputParam
from .modular_pipeline import QwenImageModularPipeline


class QwenImageInputsDynamicStep(ModularPipelineBlocks):
    model_name = "qwenimage"

    @property
    def description(self) -> str:
        # Default behavior section
        default_section = (
            "Input processing step that:\n"
            "  1. Determines `batch_size` and `dtype` based on `prompt_embeds`\n"
            "  2. Adjusts batch dimension for default text inputs: `prompt_embeds`, `prompt_embeds_mask`, `negative_prompt_embeds`, and `negative_prompt_embeds_mask`"
        )

        # Additional inputs info
        additional_inputs_info = ""
        if self._additional_input_names:
            additional_inputs_info = f", and additional inputs: {self._additional_input_names}"

        default_section += additional_inputs_info + "\n"
        default_section += "  3. Verifies `height` and `width` are divisible by vae_scale_factor * 2 if provided\n"

        # Image latent configuration
        image_latent_info = ""
        if self._image_latent_input_names:
            image_latent_info = (
                f"  4. Updates `height` and `width` based on {self._image_latent_input_names[0]} if not provided\n"
            )

        # Placement guidance
        placement_section = "\nThis block should be placed right after all the encoder steps (e.g., text_encoder, image_encoder, vae_encoder)."

        return default_section + image_latent_info + placement_section

    @property
    def inputs(self) -> List[InputParam]:
        additional_inputs = []
        for input_name in self._additional_input_names:
            additional_inputs.append(InputParam(name=input_name, required=True))

        for image_latent_input_name in self._image_latent_input_names:
            additional_inputs.append(InputParam(name=image_latent_input_name))

        return [
            InputParam(name="num_images_per_prompt", default=1),
            InputParam(name="prompt_embeds", required=True, kwargs_type="denoiser_input_fields"),
            InputParam(name="prompt_embeds_mask", required=True, kwargs_type="denoiser_input_fields"),
            InputParam(name="negative_prompt_embeds", kwargs_type="denoiser_input_fields"),
            InputParam(name="negative_prompt_embeds_mask", kwargs_type="denoiser_input_fields"),
            InputParam(name="height"),
            InputParam(name="width"),
        ] + additional_inputs

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
        ]

    @staticmethod
    def check_inputs(
        prompt_embeds,
        prompt_embeds_mask,
        negative_prompt_embeds,
        negative_prompt_embeds_mask,
        height,
        width,
        vae_scale_factor,
    ):
        if negative_prompt_embeds is not None and negative_prompt_embeds_mask is None:
            raise ValueError("`negative_prompt_embeds_mask` is required when `negative_prompt_embeds` is not None")

        if negative_prompt_embeds is None and negative_prompt_embeds_mask is not None:
            raise ValueError("cannot pass `negative_prompt_embeds_mask` without `negative_prompt_embeds`")

        if prompt_embeds_mask.shape[0] != prompt_embeds.shape[0]:
            raise ValueError("`prompt_embeds_mask` must have the same batch size as `prompt_embeds`")

        elif negative_prompt_embeds is not None and negative_prompt_embeds.shape[0] != prompt_embeds.shape[0]:
            raise ValueError("`negative_prompt_embeds` must have the same batch size as `prompt_embeds`")

        elif (
            negative_prompt_embeds_mask is not None and negative_prompt_embeds_mask.shape[0] != prompt_embeds.shape[0]
        ):
            raise ValueError("`negative_prompt_embeds_mask` must have the same batch size as `prompt_embeds`")

        if height is not None and height % (vae_scale_factor * 2) != 0:
            raise ValueError(f"Height must be divisible by {vae_scale_factor * 2} but is {height}")

        if width is not None and width % (vae_scale_factor * 2) != 0:
            raise ValueError(f"Width must be divisible by {vae_scale_factor * 2} but is {width}")

    def __init__(self, additional_input_names: List[str] = [], image_latent_input_names: List[str] = []):
        """Initialize a dynamic input processing block for standardizing conditional inputs.

        In modular pipelines, blocks are usually arranged in this sequence: encoders first (text_encoder,
        image_encoder, vae_encoder), then before_denoise steps (prepare latents, setup timesteps). This block should be
        placed after all encoder steps to process and standardize all conditional inputs that will be passed to
        subsequent blocks.

        This block handles common input processing tasks:
        - Adjusts batch dimensions for text inputs (prompt_embeds, prompt_embeds_mask, etc.) and any additional inputs
        - Determines dtype from text embeddings
        - Updates height/width to match image inputs (unless user explicitly provides different values)
        - Standardizes conditional inputs for the rest of the pipeline

        Args:
            additional_input_names (List[str], optional): Names of additional conditional input tensors to process.
                These tensors will have their batch dimensions adjusted to match the final batch size. Can be a single
                string or list of strings. Defaults to []. Examples: ["image_latents"], ["control_image_latents",
                "reference_image"]
            image_latent_input_names (List[str], optional): Names of image latent tensors to use for
                determining height/width if not explicitly provided. Can be a single string or list of strings.
                Defaults to []. Examples: ["image_latents"], ["control_image_latents"]

        Examples:
            # Basic usage - only processes default text embeddings QwenImageExpandInputsDynamicStep()

            # this will repeat the image latents to match the batch size of prompts
            QwenImageExpandInputsDynamicStep(additional_input_names=["image_latents"])

            # this will use the image latents to determine height/width
            QwenImageExpandInputsDynamicStep(image_latent_input_names=["image_latents"])

            # expand the batch dimension for multiple additional inputs and use image latents for height/width
            QwenImageExpandInputsDynamicStep(
                additional_input_names=["image_latents", "control_image_latents"],
                image_latent_input_names=["image_latents"]
            )
        """
        if not isinstance(additional_input_names, list):
            additional_input_names = [additional_input_names]
        if not isinstance(image_latent_input_names, list):
            image_latent_input_names = [image_latent_input_names]

        self._additional_input_names = additional_input_names
        self._image_latent_input_names = image_latent_input_names
        super().__init__()

    def __call__(self, components: QwenImageModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        self.check_inputs(
            prompt_embeds=block_state.prompt_embeds,
            prompt_embeds_mask=block_state.prompt_embeds_mask,
            negative_prompt_embeds=block_state.negative_prompt_embeds,
            negative_prompt_embeds_mask=block_state.negative_prompt_embeds_mask,
            height=block_state.height,
            width=block_state.width,
            vae_scale_factor=components.vae_scale_factor,
        )

        block_state.batch_size = block_state.prompt_embeds.shape[0]
        block_state.dtype = block_state.prompt_embeds.dtype

        _, seq_len, _ = block_state.prompt_embeds.shape

        block_state.prompt_embeds = block_state.prompt_embeds.repeat(1, block_state.num_images_per_prompt, 1)
        block_state.prompt_embeds = block_state.prompt_embeds.view(
            block_state.batch_size * block_state.num_images_per_prompt, seq_len, -1
        )

        block_state.prompt_embeds_mask = block_state.prompt_embeds_mask.repeat(1, block_state.num_images_per_prompt, 1)
        block_state.prompt_embeds_mask = block_state.prompt_embeds_mask.view(
            block_state.batch_size * block_state.num_images_per_prompt, seq_len
        )

        if block_state.negative_prompt_embeds is not None:
            _, seq_len, _ = block_state.negative_prompt_embeds.shape
            block_state.negative_prompt_embeds = block_state.negative_prompt_embeds.repeat(
                1, block_state.num_images_per_prompt, 1
            )
            block_state.negative_prompt_embeds = block_state.negative_prompt_embeds.view(
                block_state.batch_size * block_state.num_images_per_prompt, seq_len, -1
            )

            block_state.negative_prompt_embeds_mask = block_state.negative_prompt_embeds_mask.repeat(
                1, block_state.num_images_per_prompt, 1
            )
            block_state.negative_prompt_embeds_mask = block_state.negative_prompt_embeds_mask.view(
                block_state.batch_size * block_state.num_images_per_prompt, seq_len
            )

        # optionally, expand additional inputs to match the batch size of prompts
        if self._additional_input_names:
            final_batch_size = block_state.batch_size * block_state.num_images_per_prompt

            for input_name in self._additional_input_names:
                input_tensor = getattr(block_state, input_name)
                # make sure input tensor e.g. image_latents has batch size 1 or batch_size same as prompts
                if input_tensor.shape[0] != 1 and input_tensor.shape[0] != block_state.batch_size:
                    raise ValueError(
                        f"`{input_name}` must have have batch size 1 or {block_state.batch_size}, but got {input_tensor.shape[0]}"
                    )
                # expand the tensor to match the batch_size * num_images_per_prompt
                repeat_pattern = [final_batch_size // input_tensor.shape[0]] + [1] * (input_tensor.dim() - 1)
                input_tensor = input_tensor.repeat(*repeat_pattern)

                setattr(block_state, input_name, input_tensor)

        if self._image_latent_input_names:
            for image_latent_input_name in self._image_latent_input_names:
                image_latent_tensor = getattr(block_state, image_latent_input_name)
                if image_latent_tensor is None:
                    continue

                height_latents, width_latents = image_latent_tensor.shape[-2:]

                if block_state.height is None:
                    block_state.height = height_latents * components.vae_scale_factor
                if block_state.width is None:
                    block_state.width = width_latents * components.vae_scale_factor

        self.set_block_state(state, block_state)

        return components, state
