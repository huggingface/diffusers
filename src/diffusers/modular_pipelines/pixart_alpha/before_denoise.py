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

import torch

from ...models import PixArtTransformer2DModel
from ...schedulers import DPMSolverMultistepScheduler
from ...utils import logging
from ...utils.torch_utils import randn_tensor
from ..modular_pipeline import ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam
from .modular_pipeline import PixArtAlphaModularPipeline


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


# Copied from diffusers.modular_pipelines.qwenimage.inputs.repeat_tensor_to_batch_size
def repeat_tensor_to_batch_size(
    input_name: str,
    input_tensor: torch.Tensor,
    batch_size: int,
    num_images_per_prompt: int = 1,
) -> torch.Tensor:
    """Repeat tensor elements to match the final batch size.

    This function expands a tensor's batch dimension to match the final batch size (batch_size * num_images_per_prompt)
    by repeating each element along dimension 0.

    The input tensor must have batch size 1 or batch_size. The function will:
    - If batch size is 1: repeat each element (batch_size * num_images_per_prompt) times
    - If batch size equals batch_size: repeat each element num_images_per_prompt times

    Args:
        input_name (str): Name of the input tensor (used for error messages)
        input_tensor (torch.Tensor): The tensor to repeat. Must have batch size 1 or batch_size.
        batch_size (int): The base batch size (number of prompts)
        num_images_per_prompt (int, optional): Number of images to generate per prompt. Defaults to 1.

    Returns:
        torch.Tensor: The repeated tensor with final batch size (batch_size * num_images_per_prompt)

    Raises:
        ValueError: If input_tensor is not a torch.Tensor or has invalid batch size

    Examples:
        tensor = torch.tensor([[1, 2, 3]]) # shape: [1, 3] repeated = repeat_tensor_to_batch_size("image", tensor,
        batch_size=2, num_images_per_prompt=2) repeated # tensor([[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]]) - shape:
        [4, 3]

        tensor = torch.tensor([[1, 2, 3], [4, 5, 6]]) # shape: [2, 3] repeated = repeat_tensor_to_batch_size("image",
        tensor, batch_size=2, num_images_per_prompt=2) repeated # tensor([[1, 2, 3], [1, 2, 3], [4, 5, 6], [4, 5, 6]])
        - shape: [4, 3]
    """
    # make sure input is a tensor
    if not isinstance(input_tensor, torch.Tensor):
        raise ValueError(f"`{input_name}` must be a tensor")

    # make sure input tensor e.g. image_latents has batch size 1 or batch_size same as prompts
    if input_tensor.shape[0] == 1:
        repeat_by = batch_size * num_images_per_prompt
    elif input_tensor.shape[0] == batch_size:
        repeat_by = num_images_per_prompt
    else:
        raise ValueError(
            f"`{input_name}` must have have batch size 1 or {batch_size}, but got {input_tensor.shape[0]}"
        )

    # expand the tensor to match the batch_size * num_images_per_prompt
    input_tensor = input_tensor.repeat_interleave(repeat_by, dim=0)

    return input_tensor


# text input step (per-prompt expansion)


# auto_docstring
class PixArtAlphaTextInputStep(ModularPipelineBlocks):
    """
    Input step that expands the text embeddings to the final batch size (batch_size * num_images_per_prompt).

      Inputs:
          num_images_per_prompt (`int`, *optional*, defaults to 1):
              The number of images to generate per prompt.
          prompt_embeds (`Tensor`):
              text embeddings used to guide the image generation. Can be generated from text_encoder step.
          prompt_embeds_mask (`Tensor`):
              mask for the text embeddings. Can be generated from text_encoder step.
          negative_prompt_embeds (`Tensor`, *optional*):
              negative text embeddings used to guide the image generation. Can be generated from text_encoder step.
          negative_prompt_embeds_mask (`Tensor`, *optional*):
              mask for the negative text embeddings. Can be generated from text_encoder step.

      Outputs:
          batch_size (`int`):
              The number of prompts.
          dtype (`dtype`):
              The dtype of the text embeddings, used for the latents.
          prompt_embeds (`Tensor`):
              The prompt embeddings.
          prompt_embeds_mask (`Tensor`):
              The encoder attention mask.
          negative_prompt_embeds (`Tensor`):
              The negative prompt embeddings.
          negative_prompt_embeds_mask (`Tensor`):
              The negative prompt embeddings mask.
    """

    model_name = "pixart-alpha"

    @property
    def description(self) -> str:
        return (
            "Input step that expands the text embeddings to the final batch size (batch_size * num_images_per_prompt)."
        )

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("num_images_per_prompt"),
            InputParam.template("prompt_embeds"),
            InputParam.template("prompt_embeds_mask"),
            InputParam.template("negative_prompt_embeds"),
            InputParam.template("negative_prompt_embeds_mask"),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("batch_size", type_hint=int, description="The number of prompts."),
            OutputParam(
                "dtype",
                type_hint=torch.dtype,
                description="The dtype of the text embeddings, used for the latents.",
            ),
            OutputParam.template("prompt_embeds"),
            OutputParam.template("prompt_embeds_mask"),
            OutputParam.template("negative_prompt_embeds"),
            OutputParam.template("negative_prompt_embeds_mask"),
        ]

    @torch.no_grad()
    def __call__(self, components: PixArtAlphaModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        block_state.batch_size = block_state.prompt_embeds.shape[0]
        block_state.dtype = block_state.prompt_embeds.dtype
        batch_size = block_state.batch_size
        num_images_per_prompt = block_state.num_images_per_prompt

        block_state.prompt_embeds = repeat_tensor_to_batch_size(
            "prompt_embeds", block_state.prompt_embeds, batch_size, num_images_per_prompt
        )
        block_state.prompt_embeds_mask = repeat_tensor_to_batch_size(
            "prompt_embeds_mask", block_state.prompt_embeds_mask, batch_size, num_images_per_prompt
        )

        if block_state.negative_prompt_embeds is not None:
            block_state.negative_prompt_embeds = repeat_tensor_to_batch_size(
                "negative_prompt_embeds", block_state.negative_prompt_embeds, batch_size, num_images_per_prompt
            )
            block_state.negative_prompt_embeds_mask = repeat_tensor_to_batch_size(
                "negative_prompt_embeds_mask",
                block_state.negative_prompt_embeds_mask,
                batch_size,
                num_images_per_prompt,
            )

        self.set_block_state(state, block_state)
        return components, state


# set timesteps step


# auto_docstring
class PixArtAlphaSetTimestepsStep(ModularPipelineBlocks):
    """
    Step that sets the scheduler's timesteps for the denoising process.

      Components:
          scheduler (`DPMSolverMultistepScheduler`)

      Inputs:
          num_inference_steps (`int`, *optional*, defaults to 50):
              The number of denoising steps.
          timesteps (`Tensor`, *optional*):
              Timesteps for the denoising process.
          sigmas (`list`, *optional*):
              Custom sigmas for the denoising process.

      Outputs:
          timesteps (`Tensor`):
              The timestep schedule for the denoising loop.
          num_inference_steps (`int`):
              The number of denoising steps.
    """

    model_name = "pixart-alpha"

    @property
    def description(self) -> str:
        return "Step that sets the scheduler's timesteps for the denoising process."

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("scheduler", DPMSolverMultistepScheduler),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("num_inference_steps"),
            InputParam.template("timesteps"),
            InputParam.template("sigmas"),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "timesteps", type_hint=torch.Tensor, description="The timestep schedule for the denoising loop."
            ),
            OutputParam("num_inference_steps", type_hint=int, description="The number of denoising steps."),
        ]

    @torch.no_grad()
    def __call__(self, components: PixArtAlphaModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        block_state.timesteps, block_state.num_inference_steps = retrieve_timesteps(
            components.scheduler,
            block_state.num_inference_steps,
            components._execution_device,
            block_state.timesteps,
            block_state.sigmas,
        )
        components.scheduler.set_begin_index(0)

        self.set_block_state(state, block_state)
        return components, state


# prepare latents step


# auto_docstring
class PixArtAlphaPrepareLatentsStep(ModularPipelineBlocks):
    """
    Step that prepares the initial random noise latents for the denoising process.

      Components:
          scheduler (`DPMSolverMultistepScheduler`)

      Inputs:
          height (`int`, *optional*):
              The height in pixels of the generated image.
          width (`int`, *optional*):
              The width in pixels of the generated image.
          num_images_per_prompt (`int`, *optional*, defaults to 1):
              The number of images to generate per prompt.
          batch_size (`int`):
              Number of prompts, the final batch size of model inputs should be batch_size * num_images_per_prompt. Can
              be generated in input step.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.
          latents (`Tensor`, *optional*):
              Pre-generated noisy latents for image generation.
          dtype (`dtype`, *optional*, defaults to torch.float32):
              The dtype of the model inputs, can be generated in input step.

      Outputs:
          latents (`Tensor`):
              The initial noisy latents for the denoising loop.
    """

    model_name = "pixart-alpha"

    @property
    def description(self) -> str:
        return "Step that prepares the initial random noise latents for the denoising process."

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("scheduler", DPMSolverMultistepScheduler),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("height"),
            InputParam.template("width"),
            InputParam.template("num_images_per_prompt"),
            InputParam.template("batch_size", required=True),
            InputParam.template("generator"),
            InputParam.template("latents"),
            InputParam.template("dtype"),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "latents",
                type_hint=torch.Tensor,
                description="The initial noisy latents for the denoising loop.",
            ),
        ]

    @staticmethod
    def check_inputs(height, width, vae_scale_factor):
        if height is not None and height % vae_scale_factor != 0:
            raise ValueError(f"`height` must be divisible by {vae_scale_factor} but is {height}.")
        if width is not None and width % vae_scale_factor != 0:
            raise ValueError(f"`width` must be divisible by {vae_scale_factor} but is {width}.")

    @torch.no_grad()
    def __call__(self, components: PixArtAlphaModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        self.check_inputs(block_state.height, block_state.width, components.vae_scale_factor)

        device = components._execution_device
        height = block_state.height or components.default_height
        width = block_state.width or components.default_width
        batch_size = block_state.batch_size * block_state.num_images_per_prompt

        shape = (
            batch_size,
            components.num_channels_latents,
            int(height) // components.vae_scale_factor,
            int(width) // components.vae_scale_factor,
        )

        if block_state.latents is None:
            block_state.latents = randn_tensor(
                shape, generator=block_state.generator, device=device, dtype=block_state.dtype
            )
        else:
            block_state.latents = block_state.latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        block_state.latents = block_state.latents * components.scheduler.init_noise_sigma

        self.set_block_state(state, block_state)
        return components, state


# prepare micro-conditions step (resolution / aspect ratio)


# auto_docstring
class PixArtAlphaPrepareMicroConditionsStep(ModularPipelineBlocks):
    """
    Step that prepares the `resolution` and `aspect_ratio` micro-conditions consumed by the transformer.

      Components:
          transformer (`PixArtTransformer2DModel`)

      Inputs:
          height (`int`, *optional*):
              The height in pixels of the generated image.
          width (`int`, *optional*):
              The width in pixels of the generated image.
          batch_size (`int`):
              Number of prompts, the final batch size of model inputs should be batch_size * num_images_per_prompt. Can
              be generated in input step.
          num_images_per_prompt (`int`, *optional*, defaults to 1):
              The number of images to generate per prompt.
          dtype (`dtype`, *optional*, defaults to torch.float32):
              The dtype of the model inputs, can be generated in input step.

      Outputs:
          resolution (`Tensor`):
              The resolution micro-condition, or None when unused by the checkpoint.
          aspect_ratio (`Tensor`):
              The aspect-ratio micro-condition, or None when unused by the checkpoint.
    """

    model_name = "pixart-alpha"

    @property
    def description(self) -> str:
        return "Step that prepares the `resolution` and `aspect_ratio` micro-conditions consumed by the transformer."

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("transformer", PixArtTransformer2DModel),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("height"),
            InputParam.template("width"),
            InputParam.template("batch_size", required=True),
            InputParam.template("num_images_per_prompt"),
            InputParam.template("dtype"),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "resolution",
                type_hint=torch.Tensor,
                description="The resolution micro-condition, or None when unused by the checkpoint.",
            ),
            OutputParam(
                "aspect_ratio",
                type_hint=torch.Tensor,
                description="The aspect-ratio micro-condition, or None when unused by the checkpoint.",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components: PixArtAlphaModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        block_state.resolution = None
        block_state.aspect_ratio = None

        if components.transformer.config.sample_size == 128:
            device = components._execution_device
            height = block_state.height or components.default_height
            width = block_state.width or components.default_width
            batch_size = block_state.batch_size * block_state.num_images_per_prompt

            resolution = torch.tensor([height, width]).repeat(batch_size, 1)
            aspect_ratio = torch.tensor([float(height / width)]).repeat(batch_size, 1)
            block_state.resolution = resolution.to(dtype=block_state.dtype, device=device)
            block_state.aspect_ratio = aspect_ratio.to(dtype=block_state.dtype, device=device)

        self.set_block_state(state, block_state)
        return components, state
