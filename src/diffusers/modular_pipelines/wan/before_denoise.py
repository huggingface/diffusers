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

from typing import List, Optional, Union

import torch

from ...pipelines.pipeline_utils import retrieve_timesteps
from ...schedulers import UniPCMultistepScheduler
from ...utils import logging
from ...utils.torch_utils import randn_tensor
from ..modular_pipeline import ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam
from .modular_pipeline import WanModularPipeline


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# TODO(yiyi, aryan): We need another step before text encoder to set the `num_inference_steps` attribute for guider so that
# things like when to do guidance and how many conditions to be prepared can be determined. Currently, this is done by
# always assuming you want to do guidance in the Guiders. So, negative embeddings are prepared regardless of what the
# configuration of guider is.


class WanInputStep(ModularPipelineBlocks):
    model_name = "wan"

    @property
    def description(self) -> str:
        return (
            "Input processing step that:\n"
            "  1. Determines `batch_size` and `dtype` based on `prompt_embeds`\n"
            "  2. Adjusts input tensor shapes based on `batch_size` (number of prompts) and `num_videos_per_prompt`\n\n"
            "All input tensors are expected to have either batch_size=1 or match the batch_size\n"
            "of prompt_embeds. The tensors will be duplicated across the batch dimension to\n"
            "have a final batch_size of batch_size * num_videos_per_prompt."
        )

    @property
    def inputs(self) -> List[InputParam]:
        return [
            InputParam("num_videos_per_prompt", default=1),
        ]

    @property
    def intermediate_inputs(self) -> List[str]:
        return [
            InputParam(
                "prompt_embeds",
                required=True,
                type_hint=torch.Tensor,
                description="Pre-generated text embeddings. Can be generated from text_encoder step.",
            ),
            InputParam(
                "negative_prompt_embeds",
                type_hint=torch.Tensor,
                description="Pre-generated negative text embeddings. Can be generated from text_encoder step.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> List[str]:
        return [
            OutputParam(
                "batch_size",
                type_hint=int,
                description="Number of prompts, the final batch size of model inputs should be batch_size * num_videos_per_prompt",
            ),
            OutputParam(
                "dtype",
                type_hint=torch.dtype,
                description="Data type of model tensor inputs (determined by `prompt_embeds`)",
            ),
            OutputParam(
                "prompt_embeds",
                type_hint=torch.Tensor,
                kwargs_type="denoiser_input_fields",  # already in intermedites state but declare here again for denoiser_input_fields
                description="text embeddings used to guide the image generation",
            ),
            OutputParam(
                "negative_prompt_embeds",
                type_hint=torch.Tensor,
                kwargs_type="denoiser_input_fields",  # already in intermedites state but declare here again for denoiser_input_fields
                description="negative text embeddings used to guide the image generation",
            ),
        ]

    def check_inputs(self, components, block_state):
        if block_state.prompt_embeds is not None and block_state.negative_prompt_embeds is not None:
            if block_state.prompt_embeds.shape != block_state.negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {block_state.prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {block_state.negative_prompt_embeds.shape}."
                )

    @torch.no_grad()
    def __call__(self, components: WanModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        self.check_inputs(components, block_state)

        block_state.batch_size = block_state.prompt_embeds.shape[0]
        block_state.dtype = block_state.prompt_embeds.dtype

        _, seq_len, _ = block_state.prompt_embeds.shape
        block_state.prompt_embeds = block_state.prompt_embeds.repeat(1, block_state.num_videos_per_prompt, 1)
        block_state.prompt_embeds = block_state.prompt_embeds.view(
            block_state.batch_size * block_state.num_videos_per_prompt, seq_len, -1
        )

        if block_state.negative_prompt_embeds is not None:
            _, seq_len, _ = block_state.negative_prompt_embeds.shape
            block_state.negative_prompt_embeds = block_state.negative_prompt_embeds.repeat(
                1, block_state.num_videos_per_prompt, 1
            )
            block_state.negative_prompt_embeds = block_state.negative_prompt_embeds.view(
                block_state.batch_size * block_state.num_videos_per_prompt, seq_len, -1
            )

        self.set_block_state(state, block_state)

        return components, state


class WanSetTimestepsStep(ModularPipelineBlocks):
    model_name = "wan"

    @property
    def expected_components(self) -> List[ComponentSpec]:
        return [
            ComponentSpec("scheduler", UniPCMultistepScheduler),
        ]

    @property
    def description(self) -> str:
        return "Step that sets the scheduler's timesteps for inference"

    @property
    def inputs(self) -> List[InputParam]:
        return [
            InputParam("num_inference_steps", default=50),
            InputParam("timesteps"),
            InputParam("sigmas"),
        ]

    @property
    def intermediate_outputs(self) -> List[OutputParam]:
        return [
            OutputParam("timesteps", type_hint=torch.Tensor, description="The timesteps to use for inference"),
            OutputParam(
                "num_inference_steps",
                type_hint=int,
                description="The number of denoising steps to perform at inference time",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components: WanModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        block_state.device = components._execution_device

        block_state.timesteps, block_state.num_inference_steps = retrieve_timesteps(
            components.scheduler,
            block_state.num_inference_steps,
            block_state.device,
            block_state.timesteps,
            block_state.sigmas,
        )

        self.set_block_state(state, block_state)
        return components, state


class WanPrepareLatentsStep(ModularPipelineBlocks):
    model_name = "wan"

    @property
    def expected_components(self) -> List[ComponentSpec]:
        return []

    @property
    def description(self) -> str:
        return "Prepare latents step that prepares the latents for the text-to-video generation process"

    @property
    def inputs(self) -> List[InputParam]:
        return [
            InputParam("height", type_hint=int),
            InputParam("width", type_hint=int),
            InputParam("num_frames", type_hint=int),
            InputParam("latents", type_hint=Optional[torch.Tensor]),
            InputParam("num_videos_per_prompt", type_hint=int, default=1),
        ]

    @property
    def intermediate_inputs(self) -> List[InputParam]:
        return [
            InputParam("generator"),
            InputParam(
                "batch_size",
                required=True,
                type_hint=int,
                description="Number of prompts, the final batch size of model inputs should be `batch_size * num_videos_per_prompt`. Can be generated in input step.",
            ),
            InputParam("dtype", type_hint=torch.dtype, description="The dtype of the model inputs"),
        ]

    @property
    def intermediate_outputs(self) -> List[OutputParam]:
        return [
            OutputParam(
                "latents", type_hint=torch.Tensor, description="The initial latents to use for the denoising process"
            )
        ]

    @staticmethod
    def check_inputs(components, block_state):
        if (block_state.height is not None and block_state.height % components.vae_scale_factor_spatial != 0) or (
            block_state.width is not None and block_state.width % components.vae_scale_factor_spatial != 0
        ):
            raise ValueError(
                f"`height` and `width` have to be divisible by {components.vae_scale_factor_spatial} but are {block_state.height} and {block_state.width}."
            )
        if block_state.num_frames is not None and (
            block_state.num_frames < 1 or (block_state.num_frames - 1) % components.vae_scale_factor_temporal != 0
        ):
            raise ValueError(
                f"`num_frames` has to be greater than 0, and (num_frames - 1) must be divisible by {components.vae_scale_factor_temporal}, but got {block_state.num_frames}."
            )

    @staticmethod
    # Copied from diffusers.pipelines.wan.pipeline_wan.WanPipeline.prepare_latents with self->comp
    def prepare_latents(
        comp,
        batch_size: int,
        num_channels_latents: int = 16,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        num_latent_frames = (num_frames - 1) // comp.vae_scale_factor_temporal + 1
        shape = (
            batch_size,
            num_channels_latents,
            num_latent_frames,
            int(height) // comp.vae_scale_factor_spatial,
            int(width) // comp.vae_scale_factor_spatial,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        return latents

    @torch.no_grad()
    def __call__(self, components: WanModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        block_state.height = block_state.height or components.default_height
        block_state.width = block_state.width or components.default_width
        block_state.num_frames = block_state.num_frames or components.default_num_frames
        block_state.device = components._execution_device
        block_state.dtype = torch.float32  # Wan latents should be torch.float32 for best quality
        block_state.num_channels_latents = components.num_channels_latents

        self.check_inputs(components, block_state)

        block_state.latents = self.prepare_latents(
            components,
            block_state.batch_size * block_state.num_videos_per_prompt,
            block_state.num_channels_latents,
            block_state.height,
            block_state.width,
            block_state.num_frames,
            block_state.dtype,
            block_state.device,
            block_state.generator,
            block_state.latents,
        )

        self.set_block_state(state, block_state)

        return components, state
