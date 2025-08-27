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

import inspect
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from ...configuration_utils import FrozenDict
from ...image_processor import VaeImageProcessor, InpaintProcessor
from ...models import QwenImageControlNetModel, QwenImageMultiControlNetModel
from ...pipelines.qwenimage.pipeline_qwenimage_edit import calculate_dimensions
from ...schedulers import FlowMatchEulerDiscreteScheduler
from ...utils.torch_utils import randn_tensor, unwrap_module
from ..modular_pipeline import ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam
from .modular_pipeline import QwenImageModularPipeline


class QwenImageEditResizeStep(ModularPipelineBlocks):
    model_name = "qwenimage"

    @property
    def description(self) -> str:
        return "Image Resize step that resize the image to the target area while maintaining the aspect ratio."

    @property
    def expected_components(self) -> List[ComponentSpec]:
        return [
            ComponentSpec(
                "image_resize_processor",
                VaeImageProcessor,
                config=FrozenDict({"vae_scale_factor": 16}),
                default_creation_method="from_config",
            ),
        ]

    @property
    def inputs(self) -> List[InputParam]:
        return [
            InputParam(name="image", required=True, type_hint=torch.Tensor, description="The image to resize"),
        ]

    @torch.no_grad()
    def __call__(self, components: QwenImageModularPipeline, state: PipelineState):
        block_state = self.get_block_state(state)

        images = block_state.image
        if not isinstance(images, list):
            images = [images]

        image_width, image_height = images[0].size
        calculated_width, calculated_height, _ = calculate_dimensions(1024 * 1024, image_width / image_height)

        resized_images = [
            components.image_processor.resize(image, height=calculated_height, width=calculated_width)
            for image in images
        ]

        block_state.image = resized_images
        self.set_block_state(state, block_state)
        return components, state


class QwenImageInpaintProcessImagesInputStep(ModularPipelineBlocks):
    model_name = "qwenimage"

    @property
    def description(self) -> str:
        return "Image Mask step that resize the image to the target area while maintaining the aspect ratio."


    @property
    def expected_components(self) -> List[ComponentSpec]:
        return [
            ComponentSpec(
                "image_mask_processor",
                InpaintProcessor,
                config=FrozenDict({"vae_scale_factor": 16}),
                default_creation_method="from_config",
            ),
        ]

    @property
    def inputs(self) -> List[InputParam]:
        return [
            InputParam("image", required=True),
            InputParam("mask_image", required=True),
            InputParam("height"),
            InputParam("width"),
            InputParam("padding_mask_crop"),
        ]
    
    @property
    def intermediate_outputs(self) -> List[OutputParam]:
        return [
            OutputParam(name="original_image", type_hint=torch.Tensor, description="The original image"),
            OutputParam(name="original_mask", type_hint=torch.Tensor, description="The original mask"),
            OutputParam(name="crop_coords", type_hint=List[Tuple[int, int]], description="The crop coordinates to use for the preprocess/postprocess of the image and mask",),
        ]
    
    @torch.no_grad()
    def __call__(self, components: QwenImageModularPipeline, state: PipelineState):
        block_state = self.get_block_state(state)

        block_state.height = block_state.height or components.default_height
        block_state.width = block_state.width or components.default_width

        block_state.image, block_state.mask_image, postprocessing_kwargs = components.image_mask_processor.preprocess(
            image=block_state.image,
            mask=block_state.mask_image,
            height=block_state.height,
            width=block_state.width,
            padding_mask_crop=block_state.padding_mask_crop,
        )

        if postprocessing_kwargs:
            block_state.original_image = postprocessing_kwargs["original_image"]
            block_state.original_mask = postprocessing_kwargs["original_mask"]
            block_state.crop_coords = postprocessing_kwargs["crops_coords"]
        
        self.set_block_state(state, block_state)
        return components, state


class QwenImageInputsDynamicStep(ModularPipelineBlocks):
    model_name = "qwenimage"

    @property
    def description(self) -> str:
        # Default behavior section
        default_section = (
            "Input processing step that performs the following by default:\n"
            "  1. Determines `batch_size` and `dtype` based on `prompt_embeds`\n"
            "  2. Adjusts batch dimension for default text inputs: `prompt_embeds`, `prompt_embeds_mask`, `negative_prompt_embeds`, and `negative_prompt_embeds_mask`\n"
            "  3. Verifies `height` and `width` are divisible by vae_scale_factor * 2 if provided\n\n"
        )
        
        # Dynamic configuration section
        dynamic_section = "This is a dynamic block that you can configure to:\n\n"
        
        # Additional inputs configuration
        additional_inputs_info = ""
        if self._additional_input_names:
            additional_inputs_info = f"* Adjust batch dimension for additional inputs by passing `additional_input_names` when initializing the block. Currently configured to process: {self._additional_input_names}\n"
        else:
            additional_inputs_info = "* Adjust batch dimension for additional inputs by passing `additional_input_names` when initializing the block\n"
        
        # Image latent configuration
        image_latent_info = ""
        if self._image_latent_input_names:
            image_latent_info = f"* Use {self._image_latent_input_names} to update `height` and `width` if not defined. Currently configured to use: {self._image_latent_input_names}\n"
        else:
            image_latent_info = "* Use image latents to update `height` and `width` if not defined by passing `image_latent_input_names` when initializing the block\n"
        
        # Placement guidance
        placement_section = "\nThis block should be placed right after all the encoder steps (e.g., text_encoder, image_encoder, vae_encoder)."
        
        return default_section + dynamic_section + additional_inputs_info + image_latent_info + placement_section


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
    def check_inputs(prompt_embeds, prompt_embeds_mask, negative_prompt_embeds, negative_prompt_embeds_mask, height, width, vae_scale_factor):
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
                    raise ValueError(f"`{input_name}` must have have batch size 1 or {block_state.batch_size}, but got {input_tensor.shape[0]}")
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


class QwenImageInpaintProcessImagesOutputStep(ModularPipelineBlocks):
    model_name = "qwenimage"

    @property
    def description(self) -> str:
        return "postprocess the generated image, optional apply the mask overally to the original image.."


    @property
    def expected_components(self) -> List[ComponentSpec]:
        return [
            ComponentSpec(
                "image_mask_processor",
                InpaintProcessor,
                config=FrozenDict({"vae_scale_factor": 16}),
                default_creation_method="from_config",
            ),
        ]

    @property
    def inputs(self) -> List[InputParam]:
        return [
            InputParam("images", required=True, description= "the generated image from decoders step"),
            InputParam("original_image"),
            InputParam("original_mask"),
            InputParam("crop_coords"),
        ]
    
    @torch.no_grad()
    def __call__(self, components: QwenImageModularPipeline, state: PipelineState):
        block_state = self.get_block_state(state)

        block_state.images = components.image_mask_processor.postprocess(
            image=block_state.images,
            original_image=block_state.original_image,
            original_mask=block_state.original_mask,
            crops_coords=block_state.crop_coords,
        )

        self.set_block_state(state, block_state)
        return components, state