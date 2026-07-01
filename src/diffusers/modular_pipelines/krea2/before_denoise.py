# Copyright 2026 Krea AI and The HuggingFace Team. All rights reserved.
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

from ...schedulers import FlowMatchEulerDiscreteScheduler
from ...utils import logging
from ...utils.torch_utils import randn_tensor
from ..modular_pipeline import ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import ComponentSpec, ConfigSpec, InputParam, OutputParam
from .modular_pipeline import Krea2ModularPipeline


logger = logging.get_logger(__name__)


# Copied from diffusers.pipelines.flux.pipeline_flux.calculate_shift
def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: int | None = None,
    device: str | torch.device | None = None,
    timesteps: list[int] | None = None,
    sigmas: list[float] | None = None,
    **kwargs,
):
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


class Krea2PrepareLatentsStep(ModularPipelineBlocks):
    """
    Prepare packed Krea 2 latent noise for text-to-image generation.
    """

    model_name = "krea2"

    @property
    def expected_configs(self) -> list[ConfigSpec]:
        return [ConfigSpec("patch_size", default=2, description="Patch size used to pack Krea 2 latents.")]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("latents"),
            InputParam.template("height"),
            InputParam.template("width"),
            InputParam.template("num_images_per_prompt"),
            InputParam.template("generator"),
            InputParam.template("batch_size"),
            InputParam.template("dtype"),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(name="height", type_hint=int, description="Height rounded to the Krea 2 latent multiple."),
            OutputParam(name="width", type_hint=int, description="Width rounded to the Krea 2 latent multiple."),
            OutputParam.template("latents"),
        ]

    @torch.no_grad()
    def __call__(self, components: Krea2ModularPipeline, state: PipelineState):
        block_state = self.get_block_state(state)
        device = components._execution_device
        batch_size = block_state.batch_size * block_state.num_images_per_prompt

        block_state.height = block_state.height or components.default_height
        block_state.width = block_state.width or components.default_width

        multiple = components.vae_scale_factor * components.patch_size
        if block_state.height % multiple != 0 or block_state.width % multiple != 0:
            rounded_height = ((block_state.height + multiple - 1) // multiple) * multiple
            rounded_width = ((block_state.width + multiple - 1) // multiple) * multiple
            logger.warning(
                f"`height` and `width` must be multiples of {multiple}; rounding up from"
                f" {block_state.height}x{block_state.width} to {rounded_height}x{rounded_width}."
            )
            block_state.height, block_state.width = rounded_height, rounded_width

        if block_state.latents is not None:
            block_state.latents = block_state.latents.to(device=device, dtype=block_state.dtype)
            self.set_block_state(state, block_state)
            return components, state

        latent_height = block_state.height // components.vae_scale_factor
        latent_width = block_state.width // components.vae_scale_factor
        shape = (batch_size, components.num_channels_latents, latent_height, latent_width)

        if isinstance(block_state.generator, list) and len(block_state.generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(block_state.generator)}, but requested an"
                f" effective batch size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        block_state.latents = randn_tensor(
            shape, generator=block_state.generator, device=device, dtype=block_state.dtype
        )
        block_state.latents = components.pack_latents(block_state.latents)

        self.set_block_state(state, block_state)
        return components, state


class Krea2SetTimestepsStep(ModularPipelineBlocks):
    """
    Set the Krea 2 scheduler timesteps, including the resolution-aware dynamic shift.
    """

    model_name = "krea2"

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("scheduler", FlowMatchEulerDiscreteScheduler)]

    @property
    def expected_configs(self) -> list[ConfigSpec]:
        return [
            ConfigSpec("is_distilled", default=False, description="Whether to use Krea 2 distilled timestep shift.")
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("num_inference_steps", default=28),
            InputParam.template("sigmas"),
            InputParam("latents", required=True, type_hint=torch.Tensor),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(name="timesteps", type_hint=torch.Tensor, description="Timesteps for denoising."),
            OutputParam(name="num_inference_steps", type_hint=int, description="Resolved number of denoising steps."),
        ]

    def __call__(self, components: Krea2ModularPipeline, state: PipelineState):
        block_state = self.get_block_state(state)
        device = components._execution_device
        sigmas = (
            np.linspace(1.0, 1 / block_state.num_inference_steps, block_state.num_inference_steps)
            if block_state.sigmas is None
            else block_state.sigmas
        )
        if components.config.get("is_distilled", False):
            mu = 1.15
        else:
            mu = calculate_shift(
                block_state.latents.shape[1],
                components.scheduler.config.get("base_image_seq_len", 256),
                components.scheduler.config.get("max_image_seq_len", 6400),
                components.scheduler.config.get("base_shift", 0.5),
                components.scheduler.config.get("max_shift", 1.15),
            )

        block_state.timesteps, block_state.num_inference_steps = retrieve_timesteps(
            components.scheduler,
            block_state.num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )
        components.scheduler.set_begin_index(0)

        self.set_block_state(state, block_state)
        return components, state


class Krea2PositionIdsStep(ModularPipelineBlocks):
    """
    Build Krea 2 text and image rotary position ids for the combined sequence.
    """

    model_name = "krea2"

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("prompt_embeds"),
            InputParam.template("height", required=True),
            InputParam.template("width", required=True),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [OutputParam(name="position_ids", type_hint=torch.Tensor, description="Krea 2 RoPE position ids.")]

    @staticmethod
    def prepare_position_ids(text_seq_len: int, grid_height: int, grid_width: int, device: torch.device):
        text_ids = torch.zeros(text_seq_len, 3, device=device)
        image_ids = torch.zeros(grid_height, grid_width, 3, device=device)
        image_ids[..., 1] = torch.arange(grid_height, device=device)[:, None]
        image_ids[..., 2] = torch.arange(grid_width, device=device)[None, :]
        image_ids = image_ids.reshape(grid_height * grid_width, 3)
        return torch.cat([text_ids, image_ids], dim=0)

    def __call__(self, components: Krea2ModularPipeline, state: PipelineState):
        block_state = self.get_block_state(state)
        grid_height = block_state.height // (components.vae_scale_factor * components.patch_size)
        grid_width = block_state.width // (components.vae_scale_factor * components.patch_size)
        block_state.position_ids = self.prepare_position_ids(
            block_state.prompt_embeds.shape[1],
            grid_height,
            grid_width,
            components._execution_device,
        )

        self.set_block_state(state, block_state)
        return components, state
