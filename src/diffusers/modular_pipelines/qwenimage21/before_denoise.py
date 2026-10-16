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

import inspect

import numpy as np
import torch

from ...schedulers import FlowMatchEulerDiscreteScheduler
from ...utils import logging
from ...utils.torch_utils import randn_tensor
from ..modular_pipeline import ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import ComponentSpec, ConfigSpec, InputParam, OutputParam
from .inputs import pack_latents
from .modular_pipeline import QwenImage21ModularPipeline


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# Copied from diffusers.pipelines.qwenimage.pipeline_qwenimage.calculate_shift
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


# auto_docstring
class QwenImage21PrepareLatentsStep(ModularPipelineBlocks):
    """
    Prepare the initial random noise for the generation process. `height` and `width` default to `output_resolution`
    and are rounded down to a multiple of 32, as the transformer groups the 16x compressed latents in 2x2 blocks.

      Inputs:
          latents (`Tensor`, *optional*):
              Pre-generated noisy latents for image generation.
          height (`int`, *optional*):
              The height in pixels of the generated image.
          width (`int`, *optional*):
              The width in pixels of the generated image.
          output_resolution (`int`, *optional*, defaults to 1024):
              Target side length used to derive the output size and to resize condition images.
          num_images_per_prompt (`int`, *optional*, defaults to 1):
              The number of images to generate per prompt.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.
          batch_size (`int`, *optional*, defaults to 1):
              Number of prompts, the final batch size of model inputs should be batch_size * num_images_per_prompt. Can
              be generated in input step.
          dtype (`dtype`, *optional*, defaults to torch.float32):
              The dtype of the model inputs, can be generated in input step.

      Outputs:
          height (`int`):
              if not set, updated to the default value
          width (`int`):
              if not set, updated to the default value
          latents (`Tensor`):
              The initial latents to use for the denoising process, packed to (B, H * W, C)
    """

    model_name = "qwenimage21"

    @property
    def description(self) -> str:
        return (
            "Prepare the initial random noise for the generation process. `height` and `width` default to "
            "`output_resolution` and are rounded down to a multiple of 32, as the transformer groups the 16x "
            "compressed latents in 2x2 blocks."
        )

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("latents"),
            InputParam.template("height"),
            InputParam.template("width"),
            InputParam(
                name="output_resolution",
                type_hint=int,
                default=1024,
                description="Target side length used to derive the output size and to resize condition images.",
            ),
            InputParam.template("num_images_per_prompt"),
            InputParam.template("generator"),
            InputParam.template("batch_size"),
            InputParam.template("dtype"),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(name="height", type_hint=int, description="if not set, updated to the default value"),
            OutputParam(name="width", type_hint=int, description="if not set, updated to the default value"),
            OutputParam(
                name="latents",
                type_hint=torch.Tensor,
                description="The initial latents to use for the denoising process, packed to (B, H * W, C)",
            ),
        ]

    @torch.no_grad()
    def __call__(
        self, components: QwenImage21ModularPipeline, state: PipelineState
    ) -> tuple[QwenImage21ModularPipeline, PipelineState]:
        block_state = self.get_block_state(state)

        device = components._execution_device
        batch_size = block_state.batch_size * block_state.num_images_per_prompt

        multiple_of = components.vae_scale_factor * 2
        height = block_state.height or block_state.output_resolution
        width = block_state.width or block_state.output_resolution
        block_state.height = height // multiple_of * multiple_of
        block_state.width = width // multiple_of * multiple_of

        latent_height = block_state.height // components.vae_scale_factor
        latent_width = block_state.width // components.vae_scale_factor
        shape = (batch_size, components.num_channels_latents, 1, latent_height, latent_width)

        if isinstance(block_state.generator, list) and len(block_state.generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(block_state.generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if block_state.latents is None:
            latents = randn_tensor(shape, generator=block_state.generator, device=device, dtype=block_state.dtype)
            block_state.latents = pack_latents(latents)
        else:
            block_state.latents = block_state.latents.to(device=device, dtype=block_state.dtype)

        self.set_block_state(state, block_state)
        return components, state


# auto_docstring
class QwenImage21SetTimestepsStep(ModularPipelineBlocks):
    """
    Step that sets the scheduler's timesteps. The sampling grid comes from `sigmas` when passed, otherwise from
    `num_inference_steps`. Should be run after the prepare latents step.

      Components:
          scheduler (`FlowMatchEulerDiscreteScheduler`)

      Inputs:
          num_inference_steps (`int`, *optional*, defaults to 40):
              The number of denoising steps.
          sigmas (`list`, *optional*):
              Custom sigmas for the denoising process.
          latents (`Tensor`):
              The initial random noised latents for the denoising process. Can be generated in prepare latents step.

      Outputs:
          timesteps (`Tensor`):
              The timesteps to use for the denoising process
          num_inference_steps (`int`):
              The number of denoising steps, updated when the grid comes from sigmas
    """

    model_name = "qwenimage21"

    @property
    def description(self) -> str:
        return (
            "Step that sets the scheduler's timesteps. The sampling grid comes from `sigmas` when passed, otherwise "
            "from `num_inference_steps`. Should be run after the prepare latents step."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("scheduler", FlowMatchEulerDiscreteScheduler),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("num_inference_steps", default=40),
            InputParam.template("sigmas"),
            InputParam(
                name="latents",
                required=True,
                type_hint=torch.Tensor,
                description="The initial random noised latents for the denoising process. Can be generated in prepare latents step.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                name="timesteps", type_hint=torch.Tensor, description="The timesteps to use for the denoising process"
            ),
            OutputParam(
                name="num_inference_steps",
                type_hint=int,
                description="The number of denoising steps, updated when the grid comes from sigmas",
            ),
        ]

    def __call__(
        self, components: QwenImage21ModularPipeline, state: PipelineState
    ) -> tuple[QwenImage21ModularPipeline, PipelineState]:
        block_state = self.get_block_state(state)

        device = components._execution_device

        sigmas = block_state.sigmas
        if sigmas is None:
            sigmas = np.linspace(1.0, 1 / block_state.num_inference_steps, block_state.num_inference_steps)

        mu = calculate_shift(
            image_seq_len=block_state.latents.shape[1],
            base_seq_len=components.scheduler.config.get("base_image_seq_len", 256),
            max_seq_len=components.scheduler.config.get("max_image_seq_len", 4096),
            base_shift=components.scheduler.config.get("base_shift", 0.5),
            max_shift=components.scheduler.config.get("max_shift", 1.15),
        )
        block_state.timesteps, block_state.num_inference_steps = retrieve_timesteps(
            scheduler=components.scheduler,
            num_inference_steps=block_state.num_inference_steps,
            device=device,
            sigmas=sigmas,
            mu=mu,
        )
        components.scheduler.set_begin_index(0)

        self.set_block_state(state, block_state)
        return components, state


# auto_docstring
class QwenImage21RoPEInputsStep(ModularPipelineBlocks):
    """
    Step that prepares the layout of the transformer's joint sequence for text-to-image generation: the latent grid of
    the target image. Should be placed after the prepare latents step.

      Inputs:
          batch_size (`int`, *optional*, defaults to 1):
              Number of prompts, the final batch size of model inputs should be batch_size * num_images_per_prompt. Can
              be generated in input step.
          height (`int`, *optional*):
              The height in pixels of the generated image.
          width (`int`, *optional*):
              The width in pixels of the generated image.

      Outputs:
          img_shapes (`list`):
              Per-sample (frame, height, width) of the target image in latent tokens
    """

    model_name = "qwenimage21"

    @property
    def description(self) -> str:
        return (
            "Step that prepares the layout of the transformer's joint sequence for text-to-image generation: the "
            "latent grid of the target image. Should be placed after the prepare latents step."
        )

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("batch_size"),
            InputParam.template("height"),
            InputParam.template("width"),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                name="img_shapes",
                kwargs_type="denoiser_input_fields",
                type_hint=list[list[tuple[int, int, int]]],
                description="Per-sample (frame, height, width) of the target image in latent tokens",
            ),
        ]

    def __call__(
        self, components: QwenImage21ModularPipeline, state: PipelineState
    ) -> tuple[QwenImage21ModularPipeline, PipelineState]:
        block_state = self.get_block_state(state)

        vae_scale_factor = components.vae_scale_factor
        block_state.img_shapes = [
            [(1, block_state.height // vae_scale_factor, block_state.width // vae_scale_factor)]
        ] * block_state.batch_size

        self.set_block_state(state, block_state)
        return components, state


# auto_docstring
class QwenImage21ImageConditionedRoPEInputsStep(ModularPipelineBlocks):
    """
    Step that prepares the layout of the transformer's joint sequence for image-conditioned generation: the latent grid
    of every condition image followed by the target image. Should be placed after the prepare latents step.

      Inputs:
          batch_size (`int`, *optional*, defaults to 1):
              Number of prompts, the final batch size of model inputs should be batch_size * num_images_per_prompt. Can
              be generated in input step.
          height (`int`, *optional*):
              The height in pixels of the generated image.
          width (`int`, *optional*):
              The width in pixels of the generated image.
          image_height (`list`):
              The pixel heights of the condition images. Can be generated in the additional inputs step.
          image_width (`list`):
              The pixel widths of the condition images. Can be generated in the additional inputs step.

      Outputs:
          img_shapes (`list`):
              Per-sample (frame, height, width) of each image in latent tokens, condition images first and the target
              image last
    """

    model_name = "qwenimage21"

    @property
    def description(self) -> str:
        return (
            "Step that prepares the layout of the transformer's joint sequence for image-conditioned generation: the "
            "latent grid of every condition image followed by the target image. Should be placed after the prepare "
            "latents step."
        )

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("batch_size"),
            InputParam.template("height"),
            InputParam.template("width"),
            InputParam(
                name="image_height",
                required=True,
                type_hint=list[int],
                description="The pixel heights of the condition images. Can be generated in the additional inputs step.",
            ),
            InputParam(
                name="image_width",
                required=True,
                type_hint=list[int],
                description="The pixel widths of the condition images. Can be generated in the additional inputs step.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                name="img_shapes",
                kwargs_type="denoiser_input_fields",
                type_hint=list[list[tuple[int, int, int]]],
                description="Per-sample (frame, height, width) of each image in latent tokens, condition images first and the target image last",
            ),
        ]

    def __call__(
        self, components: QwenImage21ModularPipeline, state: PipelineState
    ) -> tuple[QwenImage21ModularPipeline, PipelineState]:
        block_state = self.get_block_state(state)

        vae_scale_factor = components.vae_scale_factor
        block_state.img_shapes = [
            [
                *[
                    (1, image_height // vae_scale_factor, image_width // vae_scale_factor)
                    for image_height, image_width in zip(block_state.image_height, block_state.image_width)
                ],
                (1, block_state.height // vae_scale_factor, block_state.width // vae_scale_factor),
            ]
        ] * block_state.batch_size

        self.set_block_state(state, block_state)
        return components, state


# auto_docstring
class QwenImage21TurboSetTimestepsStep(ModularPipelineBlocks):
    """
    Step that sets the scheduler's timesteps for the Turbo checkpoint. The sampling grid comes from `sigmas` when
    passed, otherwise from the checkpoint's `sample_sigmas` config, otherwise from `num_inference_steps`. Should be run
    after the prepare latents step.

      Components:
          scheduler (`FlowMatchEulerDiscreteScheduler`)

      Configs:
          sample_sigmas (default: None): Sampling grid the checkpoint ships with, used when `sigmas` is not passed.

      Inputs:
          num_inference_steps (`int`, *optional*, defaults to 8):
              The number of denoising steps.
          sigmas (`list`, *optional*):
              Custom sigmas for the denoising process.
          latents (`Tensor`):
              The initial random noised latents for the denoising process. Can be generated in prepare latents step.

      Outputs:
          timesteps (`Tensor`):
              The timesteps to use for the denoising process
          num_inference_steps (`int`):
              The number of denoising steps, updated when the grid comes from sigmas
    """

    model_name = "qwenimage21-turbo"

    @property
    def description(self) -> str:
        return (
            "Step that sets the scheduler's timesteps for the Turbo checkpoint. The sampling grid comes from `sigmas` "
            "when passed, otherwise from the checkpoint's `sample_sigmas` config, otherwise from "
            "`num_inference_steps`. Should be run after the prepare latents step."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("scheduler", FlowMatchEulerDiscreteScheduler),
        ]

    @property
    def expected_configs(self) -> list[ConfigSpec]:
        return [
            ConfigSpec(
                name="sample_sigmas",
                default=None,
                description="Sampling grid the checkpoint ships with, used when `sigmas` is not passed.",
            ),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("num_inference_steps", default=8),
            InputParam.template("sigmas"),
            InputParam(
                name="latents",
                required=True,
                type_hint=torch.Tensor,
                description="The initial random noised latents for the denoising process. Can be generated in prepare latents step.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                name="timesteps", type_hint=torch.Tensor, description="The timesteps to use for the denoising process"
            ),
            OutputParam(
                name="num_inference_steps",
                type_hint=int,
                description="The number of denoising steps, updated when the grid comes from sigmas",
            ),
        ]

    def __call__(
        self, components: QwenImage21ModularPipeline, state: PipelineState
    ) -> tuple[QwenImage21ModularPipeline, PipelineState]:
        block_state = self.get_block_state(state)

        device = components._execution_device

        sigmas = block_state.sigmas
        if sigmas is None:
            sigmas = components.config.sample_sigmas
        if sigmas is None:
            sigmas = np.linspace(1.0, 1 / block_state.num_inference_steps, block_state.num_inference_steps)

        mu = calculate_shift(
            image_seq_len=block_state.latents.shape[1],
            base_seq_len=components.scheduler.config.get("base_image_seq_len", 256),
            max_seq_len=components.scheduler.config.get("max_image_seq_len", 4096),
            base_shift=components.scheduler.config.get("base_shift", 0.5),
            max_shift=components.scheduler.config.get("max_shift", 1.15),
        )
        block_state.timesteps, block_state.num_inference_steps = retrieve_timesteps(
            scheduler=components.scheduler,
            num_inference_steps=block_state.num_inference_steps,
            device=device,
            sigmas=sigmas,
            mu=mu,
        )
        components.scheduler.set_begin_index(0)

        self.set_block_state(state, block_state)
        return components, state
