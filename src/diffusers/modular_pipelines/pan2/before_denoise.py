# Copyright 2026 The HuggingFace Team. All rights reserved.
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

import numpy as np
import torch

from ...models import PAN2Transformer3DModel
from ...schedulers import FlowMatchEulerDiscreteScheduler
from ...utils import logging
from ...utils.torch_utils import randn_tensor
from ..modular_pipeline import ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam
from .modular_pipeline import PAN2ModularPipeline


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class PAN2TextInputStep(ModularPipelineBlocks):
    model_name = "pan2"

    @property
    def description(self) -> str:
        return (
            "Input step that determines `batch_size` from the prompt embeddings and repeats the text conditioning "
            "`num_videos_per_prompt` times."
        )

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template(
                "num_images_per_prompt",
                name="num_videos_per_prompt",
                description="The number of videos to generate per prompt.",
            ),
            InputParam.template("prompt_embeds", description="Text embeddings used to guide the video generation."),
            InputParam.template(
                "negative_prompt_embeds",
                description="Negative text embeddings used to guide the video generation.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "batch_size",
                type_hint=int,
                description="Number of prompts; the model batch size is `batch_size * num_videos_per_prompt`.",
            ),
            OutputParam.template("prompt_embeds"),
            OutputParam.template("negative_prompt_embeds"),
        ]

    @torch.no_grad()
    def __call__(
        self, components: PAN2ModularPipeline, state: PipelineState
    ) -> tuple[PAN2ModularPipeline, PipelineState]:
        block_state = self.get_block_state(state)
        num_videos_per_prompt = block_state.num_videos_per_prompt

        block_state.batch_size = block_state.prompt_embeds.shape[0]
        block_state.prompt_embeds = block_state.prompt_embeds.repeat_interleave(num_videos_per_prompt, dim=0)
        if block_state.negative_prompt_embeds is not None:
            block_state.negative_prompt_embeds = block_state.negative_prompt_embeds.repeat_interleave(
                num_videos_per_prompt, dim=0
            )

        self.set_block_state(state, block_state)
        return components, state


class PAN2Image2VideoInputStep(ModularPipelineBlocks):
    model_name = "pan2"

    @property
    def description(self) -> str:
        return "Input step that derives `height` and `width` from the first-frame latents when they are not provided."

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                "image_latents",
                required=True,
                type_hint=torch.Tensor,
                description="The normalized first-frame latents. Can be generated from the vae_encoder step.",
            ),
            InputParam.template(
                "height",
                description="The height in pixels of the generated video, by default that of `image_latents`.",
            ),
            InputParam.template(
                "width", description="The width in pixels of the generated video, by default that of `image_latents`."
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("height", type_hint=int, description="The height in pixels of the generated video."),
            OutputParam("width", type_hint=int, description="The width in pixels of the generated video."),
        ]

    @torch.no_grad()
    def __call__(
        self, components: PAN2ModularPipeline, state: PipelineState
    ) -> tuple[PAN2ModularPipeline, PipelineState]:
        block_state = self.get_block_state(state)

        latent_height, latent_width = block_state.image_latents.shape[-2:]
        height = latent_height * components.vae_scale_factor_spatial
        width = latent_width * components.vae_scale_factor_spatial
        if (block_state.height or height, block_state.width or width) != (height, width):
            raise ValueError(
                f"`height` and `width` ({block_state.height}, {block_state.width}) do not match the size of "
                f"`image_latents` ({height}, {width}); leave them unset to take the size of `image_latents`."
            )
        block_state.height, block_state.width = height, width

        self.set_block_state(state, block_state)
        return components, state


class PAN2SetTimestepsStep(ModularPipelineBlocks):
    model_name = "pan2"

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("scheduler", FlowMatchEulerDiscreteScheduler)]

    @property
    def description(self) -> str:
        return "Step that sets the scheduler's timesteps for inference."

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("num_inference_steps", default=50),
            InputParam.template("sigmas"),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("timesteps", type_hint=torch.Tensor, description="The timesteps to denoise over."),
            OutputParam("num_inference_steps", type_hint=int, description="The number of denoising steps."),
        ]

    @torch.no_grad()
    def __call__(
        self, components: PAN2ModularPipeline, state: PipelineState
    ) -> tuple[PAN2ModularPipeline, PipelineState]:
        block_state = self.get_block_state(state)
        device = components._execution_device

        # PAN2 samples on uniformly spaced sigmas from 1 down to (excluding) 0, then shifted by the scheduler.
        sigmas = block_state.sigmas
        if sigmas is None:
            sigmas = np.linspace(1.0, 0.0, block_state.num_inference_steps + 1)[:-1]
        components.scheduler.set_timesteps(sigmas=sigmas, device=device)
        block_state.timesteps = components.scheduler.timesteps
        block_state.num_inference_steps = len(block_state.timesteps)

        self.set_block_state(state, block_state)
        return components, state


class PAN2PrepareLatentsStep(ModularPipelineBlocks):
    model_name = "pan2"

    @property
    def description(self) -> str:
        return "Prepare latents step that prepares the initial noise latents for video generation."

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("transformer", PAN2Transformer3DModel)]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("height", description="The height in pixels of the generated video, 704 by default."),
            InputParam.template("width", description="The width in pixels of the generated video, 1248 by default."),
            InputParam(
                "num_frames",
                type_hint=int,
                default=313,
                description="Number of video frames to generate, of the form `4 * n + 1`.",
            ),
            InputParam.template(
                "latents", description="Pre-generated noisy latents to use as inputs for video generation."
            ),
            InputParam.template(
                "num_images_per_prompt",
                name="num_videos_per_prompt",
                description="The number of videos to generate per prompt.",
            ),
            InputParam.template("generator"),
            InputParam.template("batch_size", required=True, default=None),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "latents", type_hint=torch.Tensor, description="The initial float32 noise latents to denoise."
            ),
        ]

    @staticmethod
    def check_inputs(components, height, width, num_frames):
        multiple = components.vae_scale_factor_spatial * components.transformer.config.patch_size[1]
        if height % multiple != 0 or width % multiple != 0:
            raise ValueError(f"`height` and `width` have to be divisible by {multiple} but are {height} and {width}.")
        if (num_frames - 1) % components.vae_scale_factor_temporal != 0:
            raise ValueError(
                f"`num_frames - 1` has to be divisible by {components.vae_scale_factor_temporal}, got {num_frames}."
            )

    @torch.no_grad()
    def __call__(
        self, components: PAN2ModularPipeline, state: PipelineState
    ) -> tuple[PAN2ModularPipeline, PipelineState]:
        block_state = self.get_block_state(state)
        block_state.height = block_state.height or components.default_height
        block_state.width = block_state.width or components.default_width
        self.check_inputs(components, block_state.height, block_state.width, block_state.num_frames)
        device = components._execution_device
        # The latents stay in float32 across the denoising loop and are cast to the transformer dtype per step.
        dtype = torch.float32

        batch_size = block_state.batch_size * block_state.num_videos_per_prompt
        shape = (
            batch_size,
            components.num_channels_latents,
            (block_state.num_frames - 1) // components.vae_scale_factor_temporal + 1,
            block_state.height // components.vae_scale_factor_spatial,
            block_state.width // components.vae_scale_factor_spatial,
        )

        if block_state.latents is not None:
            if block_state.latents.shape != shape:
                raise ValueError(f"`latents` should have shape {shape}, but got {tuple(block_state.latents.shape)}.")
            block_state.latents = block_state.latents.to(device=device, dtype=dtype)
        else:
            if isinstance(block_state.generator, list) and len(block_state.generator) != batch_size:
                raise ValueError(
                    f"You have passed a list of generators of length {len(block_state.generator)}, but requested an "
                    f"effective batch size of {batch_size}. Make sure the batch size matches the length of the "
                    "generators."
                )
            block_state.latents = randn_tensor(shape, generator=block_state.generator, device=device, dtype=dtype)

        self.set_block_state(state, block_state)
        return components, state


class PAN2PrepareConditionLatentsStep(ModularPipelineBlocks):
    model_name = "pan2"

    @property
    def description(self) -> str:
        return "Prepare the conditioning latents for text-to-video: empty conditioning latents and mask."

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("latents", required=True, type_hint=torch.Tensor, description="The initial noise latents."),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "condition_latents",
                type_hint=torch.Tensor,
                description=(
                    "The conditioning latents and the conditioning mask stacked along channels, concatenated to the "
                    "noisy latents at every denoising step."
                ),
            ),
        ]

    @torch.no_grad()
    def __call__(
        self, components: PAN2ModularPipeline, state: PipelineState
    ) -> tuple[PAN2ModularPipeline, PipelineState]:
        block_state = self.get_block_state(state)
        batch_size, num_channels, num_frames, height, width = block_state.latents.shape
        block_state.condition_latents = block_state.latents.new_zeros(
            batch_size, num_channels + 1, num_frames, height, width
        )

        self.set_block_state(state, block_state)
        return components, state


class PAN2Image2VideoPrepareConditionLatentsStep(ModularPipelineBlocks):
    model_name = "pan2"

    @property
    def description(self) -> str:
        return (
            "Prepare the conditioning latents for image-to-video: the first-frame latents in the first latent frame, "
            "and a mask marking that frame as given."
        )

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("latents", required=True, type_hint=torch.Tensor, description="The initial noise latents."),
            InputParam(
                "image_latents",
                required=True,
                type_hint=torch.Tensor,
                description="The normalized first-frame latents. Can be generated from the vae_encoder step.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "condition_latents",
                type_hint=torch.Tensor,
                description=(
                    "The conditioning latents and the conditioning mask stacked along channels, concatenated to the "
                    "noisy latents at every denoising step."
                ),
            ),
        ]

    @torch.no_grad()
    def __call__(
        self, components: PAN2ModularPipeline, state: PipelineState
    ) -> tuple[PAN2ModularPipeline, PipelineState]:
        block_state = self.get_block_state(state)
        latents = block_state.latents
        batch_size, num_channels, num_frames, height, width = latents.shape

        image_latents = block_state.image_latents.to(device=latents.device, dtype=latents.dtype)
        if image_latents.shape[0] != batch_size:
            if batch_size % image_latents.shape[0] != 0:
                raise ValueError(
                    f"The batch size of `image_latents` ({image_latents.shape[0]}) has to divide the batch size of "
                    f"`latents` ({batch_size})."
                )
            image_latents = image_latents.repeat_interleave(batch_size // image_latents.shape[0], dim=0)

        condition_latents = latents.new_zeros(batch_size, num_channels + 1, num_frames, height, width)
        condition_latents[:, :num_channels, :1] = image_latents
        condition_latents[:, num_channels:, :1] = 1.0
        block_state.condition_latents = condition_latents

        self.set_block_state(state, block_state)
        return components, state
