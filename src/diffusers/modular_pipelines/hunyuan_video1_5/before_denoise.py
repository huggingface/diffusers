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

import inspect

import numpy as np
import torch

from ...models import HunyuanVideo15Transformer3DModel
from ...schedulers import FlowMatchEulerDiscreteScheduler
from ...utils import logging
from ...utils.torch_utils import randn_tensor
from ..modular_pipeline import ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam
from .modular_pipeline import HunyuanVideo15ModularPipeline


logger = logging.get_logger(__name__)


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps=None,
    device=None,
    timesteps=None,
    sigmas=None,
    **kwargs,
):
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed.")
    if timesteps is not None:
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom sigmas."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class HunyuanVideo15TextInputStep(ModularPipelineBlocks):
    model_name = "hunyuan-video-1.5"

    @property
    def description(self) -> str:
        return "Input processing step that determines batch_size and dtype"

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("transformer", HunyuanVideo15Transformer3DModel),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("num_videos_per_prompt", default=1),
            InputParam("prompt_embeds", required=True, type_hint=torch.Tensor),
            InputParam("batch_size", type_hint=int),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("batch_size", type_hint=int),
            OutputParam("dtype", type_hint=torch.dtype),
        ]

    @torch.no_grad()
    def __call__(self, components: HunyuanVideo15ModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        block_state.batch_size = getattr(block_state, "batch_size", None) or block_state.prompt_embeds.shape[0]
        block_state.dtype = components.transformer.dtype
        self.set_block_state(state, block_state)
        return components, state


class HunyuanVideo15SetTimestepsStep(ModularPipelineBlocks):
    model_name = "hunyuan-video-1.5"

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("scheduler", FlowMatchEulerDiscreteScheduler)]

    @property
    def description(self) -> str:
        return "Step that sets the scheduler's timesteps for inference"

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("num_inference_steps", default=50),
            InputParam("sigmas"),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("timesteps", type_hint=torch.Tensor),
            OutputParam("num_inference_steps", type_hint=int),
        ]

    # Copied from pipeline_hunyuan_video1_5.py line 702-704
    @torch.no_grad()
    def __call__(self, components: HunyuanVideo15ModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        device = components._execution_device

        sigmas = block_state.sigmas
        if sigmas is None:
            sigmas = np.linspace(1.0, 0.0, block_state.num_inference_steps + 1)[:-1]

        block_state.timesteps, block_state.num_inference_steps = retrieve_timesteps(
            components.scheduler, block_state.num_inference_steps, device, sigmas=sigmas
        )

        self.set_block_state(state, block_state)
        return components, state


class HunyuanVideo15PrepareLatentsStep(ModularPipelineBlocks):
    model_name = "hunyuan-video-1.5"

    @property
    def description(self) -> str:
        return "Prepare latents, conditioning latents, mask, and image_embeds for T2V"

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("height", type_hint=int),
            InputParam("width", type_hint=int),
            InputParam("num_frames", type_hint=int, default=121),
            InputParam("latents", type_hint=torch.Tensor | None),
            InputParam("num_videos_per_prompt", type_hint=int, default=1),
            InputParam("generator"),
            InputParam("batch_size", required=True, type_hint=int),
            InputParam("dtype", type_hint=torch.dtype),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("latents", type_hint=torch.Tensor),
            OutputParam("cond_latents_concat", type_hint=torch.Tensor),
            OutputParam("mask_concat", type_hint=torch.Tensor),
            OutputParam("image_embeds", type_hint=torch.Tensor),
        ]

    # Copied from pipeline_hunyuan_video1_5.py lines 652-655, 477-524, 706-725 with self->components
    @torch.no_grad()
    def __call__(self, components: HunyuanVideo15ModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        device = components._execution_device
        dtype = block_state.dtype

        height = block_state.height
        width = block_state.width
        if height is None and width is None:
            height, width = components.video_processor.calculate_default_height_width(
                components.default_aspect_ratio[1], components.default_aspect_ratio[0], components.target_size
            )

        batch_size = block_state.batch_size * block_state.num_videos_per_prompt
        num_frames = block_state.num_frames

        # Copied from HunyuanVideo15Pipeline.prepare_latents with self->components
        latents = block_state.latents
        if latents is not None:
            latents = latents.to(device=device, dtype=dtype)
        else:
            shape = (
                batch_size,
                components.num_channels_latents,
                (num_frames - 1) // components.vae_scale_factor_temporal + 1,
                int(height) // components.vae_scale_factor_spatial,
                int(width) // components.vae_scale_factor_spatial,
            )
            if isinstance(block_state.generator, list) and len(block_state.generator) != batch_size:
                raise ValueError(
                    f"You have passed a list of generators of length {len(block_state.generator)}, but requested an effective batch"
                    f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                )
            latents = randn_tensor(shape, generator=block_state.generator, device=device, dtype=dtype)

        block_state.latents = latents

        # Copied from HunyuanVideo15Pipeline.prepare_cond_latents_and_mask with self->components
        b, c, f, h, w = latents.shape
        block_state.cond_latents_concat = torch.zeros(b, c, f, h, w, dtype=dtype, device=device)
        block_state.mask_concat = torch.zeros(b, 1, f, h, w, dtype=dtype, device=device)

        # T2V: zero image_embeds
        block_state.image_embeds = torch.zeros(
            block_state.batch_size,
            components.vision_num_semantic_tokens,
            components.vision_states_dim,
            dtype=dtype,
            device=device,
        )

        self.set_block_state(state, block_state)
        return components, state
