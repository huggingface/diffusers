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

from ...configuration_utils import FrozenDict
from ...models import HunyuanVideo15Transformer3DModel
from ...pipelines.hunyuan_video1_5.image_processor import HunyuanVideo15ImageProcessor
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
    num_inference_steps: int | None = None,
    device: str | torch.device | None = None,
    timesteps: list[int] | None = None,
    sigmas: list[float] | None = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`list[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`list[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
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
        return "Input processing step that determines batch_size"

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("prompt_embeds"),
            InputParam.template("batch_size", default=None),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("batch_size", type_hint=int),
        ]

    @torch.no_grad()
    def __call__(self, components: HunyuanVideo15ModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        block_state.batch_size = getattr(block_state, "batch_size", None) or block_state.prompt_embeds.shape[0]
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
            InputParam.template("num_inference_steps"),
            InputParam.template("sigmas"),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("timesteps", type_hint=torch.Tensor),
            OutputParam("num_inference_steps", type_hint=int),
        ]

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
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("transformer", HunyuanVideo15Transformer3DModel),
            ComponentSpec(
                "video_processor",
                HunyuanVideo15ImageProcessor,
                config=FrozenDict({"vae_scale_factor": 16}),
                default_creation_method="from_config",
            ),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("height"),
            InputParam.template("width"),
            InputParam("num_frames", type_hint=int, default=121, description="Number of video frames to generate."),
            InputParam.template("latents"),
            InputParam.template("num_images_per_prompt", name="num_videos_per_prompt"),
            InputParam.template("generator"),
            InputParam.template("batch_size", required=True, default=None),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("latents", type_hint=torch.Tensor, description="Pure noise latents"),
            OutputParam("cond_latents_concat", type_hint=torch.Tensor),
            OutputParam("mask_concat", type_hint=torch.Tensor),
            OutputParam("image_embeds", type_hint=torch.Tensor),
        ]

    @torch.no_grad()
    def __call__(self, components: HunyuanVideo15ModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        device = components._execution_device
        dtype = components.transformer.dtype

        height = block_state.height
        width = block_state.width
        if height is None and width is None:
            height, width = components.video_processor.calculate_default_height_width(
                components.default_aspect_ratio[1], components.default_aspect_ratio[0], components.target_size
            )

        batch_size = block_state.batch_size * block_state.num_videos_per_prompt
        num_frames = block_state.num_frames

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

        b, c, f, h, w = latents.shape
        block_state.cond_latents_concat = torch.zeros(b, c, f, h, w, dtype=dtype, device=device)
        block_state.mask_concat = torch.zeros(b, 1, f, h, w, dtype=dtype, device=device)

        block_state.image_embeds = torch.zeros(
            block_state.batch_size,
            components.vision_num_semantic_tokens,
            components.vision_states_dim,
            dtype=dtype,
            device=device,
        )

        self.set_block_state(state, block_state)
        return components, state


class HunyuanVideo15Image2VideoPrepareLatentsStep(ModularPipelineBlocks):
    model_name = "hunyuan-video-1.5"

    @property
    def description(self) -> str:
        return (
            "Prepare I2V conditioning from image_latents and image_embeds. "
            "Expects pure noise `latents` from HunyuanVideo15PrepareLatentsStep. "
            "Builds cond_latents_concat and mask_concat for the denoiser."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("transformer", HunyuanVideo15Transformer3DModel)]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                "image_latents",
                type_hint=torch.Tensor,
                required=True,
                description="Pre-encoded image latents from the VAE encoder step, used as conditioning for I2V.",
            ),
            InputParam(
                "image_embeds",
                type_hint=torch.Tensor,
                required=True,
                description="Siglip image embeddings from the image encoder step, used as extra conditioning for I2V.",
            ),
            InputParam.template("latents", required=True),
            InputParam.template("num_images_per_prompt", name="num_videos_per_prompt"),
            InputParam.template("batch_size", required=True, default=None),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("cond_latents_concat", type_hint=torch.Tensor),
            OutputParam("mask_concat", type_hint=torch.Tensor),
            OutputParam("image_embeds", type_hint=torch.Tensor),
        ]

    @torch.no_grad()
    def __call__(self, components: HunyuanVideo15ModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        device = components._execution_device
        dtype = components.transformer.dtype

        batch_size = block_state.batch_size * block_state.num_videos_per_prompt

        b, c, f, h, w = block_state.latents.shape

        latent_condition = block_state.image_latents.to(device=device, dtype=dtype)
        latent_condition = latent_condition.repeat(batch_size, 1, f, 1, 1)
        latent_condition[:, :, 1:, :, :] = 0
        block_state.cond_latents_concat = latent_condition

        latent_mask = torch.zeros(b, 1, f, h, w, dtype=dtype, device=device)
        latent_mask[:, :, 0, :, :] = 1.0
        block_state.mask_concat = latent_mask

        image_embeds = block_state.image_embeds.to(device=device, dtype=dtype)
        if image_embeds.shape[0] == 1 and batch_size > 1:
            image_embeds = image_embeds.repeat(batch_size, 1, 1)
        block_state.image_embeds = image_embeds

        self.set_block_state(state, block_state)
        return components, state
