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
from typing import List, Optional, Union

import numpy as np
import torch

from ...models import Flux2Transformer2DModel
from ...schedulers import FlowMatchEulerDiscreteScheduler
from ...utils import logging
from ...utils.torch_utils import randn_tensor
from ..modular_pipeline import ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam
from .modular_pipeline import Flux2ModularPipeline


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
    """Compute empirical mu for Flux2 timestep scheduling."""
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666

    if image_seq_len > 4300:
        mu = a2 * image_seq_len + b2
        return float(mu)

    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1

    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    mu = a * num_steps + b

    return float(mu)


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
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
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
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


class Flux2SetTimestepsStep(ModularPipelineBlocks):
    model_name = "flux2"

    @property
    def expected_components(self) -> List[ComponentSpec]:
        return [
            ComponentSpec("scheduler", FlowMatchEulerDiscreteScheduler),
            ComponentSpec("transformer", Flux2Transformer2DModel),
        ]

    @property
    def description(self) -> str:
        return "Step that sets the scheduler's timesteps for Flux2 inference using empirical mu calculation"

    @property
    def inputs(self) -> List[InputParam]:
        return [
            InputParam("num_inference_steps", default=50),
            InputParam("timesteps"),
            InputParam("sigmas"),
            InputParam("guidance_scale", default=4.0),
            InputParam("latents", type_hint=torch.Tensor),
            InputParam("num_images_per_prompt", default=1),
            InputParam("height", type_hint=int),
            InputParam("width", type_hint=int),
            InputParam(
                "batch_size",
                required=True,
                type_hint=int,
                description="Number of prompts, the final batch size of model inputs should be `batch_size * num_images_per_prompt`.",
            ),
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
            OutputParam("guidance", type_hint=torch.Tensor, description="Guidance scale tensor"),
        ]

    @torch.no_grad()
    def __call__(self, components: Flux2ModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        block_state.device = components._execution_device

        scheduler = components.scheduler

        height = block_state.height or components.default_height
        width = block_state.width or components.default_width
        vae_scale_factor = components.vae_scale_factor

        latent_height = 2 * (int(height) // (vae_scale_factor * 2))
        latent_width = 2 * (int(width) // (vae_scale_factor * 2))
        image_seq_len = (latent_height // 2) * (latent_width // 2)

        num_inference_steps = block_state.num_inference_steps
        sigmas = block_state.sigmas
        timesteps = block_state.timesteps

        if timesteps is None and sigmas is None:
            sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        if hasattr(scheduler.config, "use_flow_sigmas") and scheduler.config.use_flow_sigmas:
            sigmas = None

        mu = compute_empirical_mu(image_seq_len=image_seq_len, num_steps=num_inference_steps)

        timesteps, num_inference_steps = retrieve_timesteps(
            scheduler,
            num_inference_steps,
            block_state.device,
            timesteps=timesteps,
            sigmas=sigmas,
            mu=mu,
        )
        block_state.timesteps = timesteps
        block_state.num_inference_steps = num_inference_steps

        batch_size = block_state.batch_size * block_state.num_images_per_prompt
        guidance = torch.full([1], block_state.guidance_scale, device=block_state.device, dtype=torch.float32)
        guidance = guidance.expand(batch_size)
        block_state.guidance = guidance

        components.scheduler.set_begin_index(0)

        self.set_block_state(state, block_state)
        return components, state


class Flux2PrepareLatentsStep(ModularPipelineBlocks):
    model_name = "flux2"

    @property
    def expected_components(self) -> List[ComponentSpec]:
        return []

    @property
    def description(self) -> str:
        return "Prepare latents step that prepares the initial noise latents for Flux2 text-to-image generation"

    @property
    def inputs(self) -> List[InputParam]:
        return [
            InputParam("height", type_hint=int),
            InputParam("width", type_hint=int),
            InputParam("latents", type_hint=Optional[torch.Tensor]),
            InputParam("num_images_per_prompt", type_hint=int, default=1),
            InputParam("generator"),
            InputParam(
                "batch_size",
                required=True,
                type_hint=int,
                description="Number of prompts, the final batch size of model inputs should be `batch_size * num_images_per_prompt`.",
            ),
            InputParam("dtype", type_hint=torch.dtype, description="The dtype of the model inputs"),
        ]

    @property
    def intermediate_outputs(self) -> List[OutputParam]:
        return [
            OutputParam(
                "latents", type_hint=torch.Tensor, description="The initial latents to use for the denoising process"
            ),
            OutputParam("latent_ids", type_hint=torch.Tensor, description="Position IDs for the latents (for RoPE)"),
        ]

    @staticmethod
    def check_inputs(components, block_state):
        vae_scale_factor = components.vae_scale_factor
        if (block_state.height is not None and block_state.height % (vae_scale_factor * 2) != 0) or (
            block_state.width is not None and block_state.width % (vae_scale_factor * 2) != 0
        ):
            logger.warning(
                f"`height` and `width` have to be divisible by {vae_scale_factor * 2} but are {block_state.height} and {block_state.width}."
            )

    @staticmethod
    def _prepare_latent_ids(latents: torch.Tensor):
        """
        Generates 4D position coordinates (T, H, W, L) for latent tensors.

        Args:
            latents: Latent tensor of shape (B, C, H, W)

        Returns:
            Position IDs tensor of shape (B, H*W, 4)
        """
        batch_size, _, height, width = latents.shape

        t = torch.arange(1)
        h = torch.arange(height)
        w = torch.arange(width)
        l = torch.arange(1)

        latent_ids = torch.cartesian_prod(t, h, w, l)
        latent_ids = latent_ids.unsqueeze(0).expand(batch_size, -1, -1)

        return latent_ids

    @staticmethod
    def _pack_latents(latents):
        """Pack latents: (batch_size, num_channels, height, width) -> (batch_size, height * width, num_channels)"""
        batch_size, num_channels, height, width = latents.shape
        latents = latents.reshape(batch_size, num_channels, height * width).permute(0, 2, 1)
        return latents

    @staticmethod
    def prepare_latents(
        comp,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        height = 2 * (int(height) // (comp.vae_scale_factor * 2))
        width = 2 * (int(width) // (comp.vae_scale_factor * 2))

        shape = (batch_size, num_channels_latents * 4, height // 2, width // 2)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device=device, dtype=dtype)

        return latents

    @torch.no_grad()
    def __call__(self, components: Flux2ModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        block_state.height = block_state.height or components.default_height
        block_state.width = block_state.width or components.default_width
        block_state.device = components._execution_device
        block_state.num_channels_latents = components.num_channels_latents

        self.check_inputs(components, block_state)
        batch_size = block_state.batch_size * block_state.num_images_per_prompt

        latents = self.prepare_latents(
            components,
            batch_size,
            block_state.num_channels_latents,
            block_state.height,
            block_state.width,
            block_state.dtype,
            block_state.device,
            block_state.generator,
            block_state.latents,
        )

        latent_ids = self._prepare_latent_ids(latents)
        latent_ids = latent_ids.to(block_state.device)

        latents = self._pack_latents(latents)

        block_state.latents = latents
        block_state.latent_ids = latent_ids

        self.set_block_state(state, block_state)
        return components, state


class Flux2RoPEInputsStep(ModularPipelineBlocks):
    model_name = "flux2"

    @property
    def description(self) -> str:
        return "Step that prepares the 4D RoPE position IDs for Flux2 denoising. Should be placed after text encoder and latent preparation steps."

    @property
    def inputs(self) -> List[InputParam]:
        return [
            InputParam(name="prompt_embeds", required=True),
            InputParam(name="latent_ids"),
        ]

    @property
    def intermediate_outputs(self) -> List[OutputParam]:
        return [
            OutputParam(
                name="txt_ids",
                kwargs_type="denoiser_input_fields",
                type_hint=torch.Tensor,
                description="4D position IDs (T, H, W, L) for text tokens, used for RoPE calculation.",
            ),
            OutputParam(
                name="latent_ids",
                kwargs_type="denoiser_input_fields",
                type_hint=torch.Tensor,
                description="4D position IDs (T, H, W, L) for image latents, used for RoPE calculation.",
            ),
        ]

    @staticmethod
    def _prepare_text_ids(x: torch.Tensor, t_coord: Optional[torch.Tensor] = None):
        """Prepare 4D position IDs for text tokens."""
        B, L, _ = x.shape
        out_ids = []

        for i in range(B):
            t = torch.arange(1) if t_coord is None else t_coord[i]
            h = torch.arange(1)
            w = torch.arange(1)
            seq_l = torch.arange(L)

            coords = torch.cartesian_prod(t, h, w, seq_l)
            out_ids.append(coords)

        return torch.stack(out_ids)

    def __call__(self, components: Flux2ModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        prompt_embeds = block_state.prompt_embeds
        device = prompt_embeds.device

        block_state.txt_ids = self._prepare_text_ids(prompt_embeds)
        block_state.txt_ids = block_state.txt_ids.to(device)

        self.set_block_state(state, block_state)
        return components, state


class Flux2PrepareImageLatentsStep(ModularPipelineBlocks):
    model_name = "flux2"

    @property
    def description(self) -> str:
        return "Step that prepares image latents and their position IDs for Flux2 image conditioning."

    @property
    def inputs(self) -> List[InputParam]:
        return [
            InputParam("image_latents", type_hint=List[torch.Tensor]),
            InputParam("batch_size", required=True, type_hint=int),
            InputParam("num_images_per_prompt", default=1, type_hint=int),
        ]

    @property
    def intermediate_outputs(self) -> List[OutputParam]:
        return [
            OutputParam(
                "image_latents",
                type_hint=torch.Tensor,
                description="Packed image latents for conditioning",
            ),
            OutputParam(
                "image_latent_ids",
                type_hint=torch.Tensor,
                description="Position IDs for image latents",
            ),
        ]

    @staticmethod
    def _prepare_image_ids(image_latents: List[torch.Tensor], scale: int = 10):
        """
        Generates 4D time-space coordinates (T, H, W, L) for a sequence of image latents.

        Args:
            image_latents: A list of image latent feature tensors of shape (1, C, H, W).
            scale: Factor used to define the time separation between latents.

        Returns:
            Combined coordinate tensor of shape (1, N_total, 4)
        """
        if not isinstance(image_latents, list):
            raise ValueError(f"Expected `image_latents` to be a list, got {type(image_latents)}.")

        t_coords = [scale + scale * t for t in torch.arange(0, len(image_latents))]
        t_coords = [t.view(-1) for t in t_coords]

        image_latent_ids = []
        for x, t in zip(image_latents, t_coords):
            x = x.squeeze(0)
            _, height, width = x.shape

            x_ids = torch.cartesian_prod(t, torch.arange(height), torch.arange(width), torch.arange(1))
            image_latent_ids.append(x_ids)

        image_latent_ids = torch.cat(image_latent_ids, dim=0)
        image_latent_ids = image_latent_ids.unsqueeze(0)

        return image_latent_ids

    @staticmethod
    def _pack_latents(latents):
        """Pack latents: (batch_size, num_channels, height, width) -> (batch_size, height * width, num_channels)"""
        batch_size, num_channels, height, width = latents.shape
        latents = latents.reshape(batch_size, num_channels, height * width).permute(0, 2, 1)
        return latents

    @torch.no_grad()
    def __call__(self, components: Flux2ModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        image_latents = block_state.image_latents

        if image_latents is None:
            block_state.image_latents = None
            block_state.image_latent_ids = None
            self.set_block_state(state, block_state)

            return components, state

        device = components._execution_device
        batch_size = block_state.batch_size * block_state.num_images_per_prompt

        image_latent_ids = self._prepare_image_ids(image_latents)

        packed_latents = []
        for latent in image_latents:
            packed = self._pack_latents(latent)
            packed = packed.squeeze(0)
            packed_latents.append(packed)

        image_latents = torch.cat(packed_latents, dim=0)
        image_latents = image_latents.unsqueeze(0)

        image_latents = image_latents.repeat(batch_size, 1, 1)
        image_latent_ids = image_latent_ids.repeat(batch_size, 1, 1)
        image_latent_ids = image_latent_ids.to(device)

        block_state.image_latents = image_latents
        block_state.image_latent_ids = image_latent_ids

        self.set_block_state(state, block_state)
        return components, state
