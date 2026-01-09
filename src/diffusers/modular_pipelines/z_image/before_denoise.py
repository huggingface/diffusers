# Copyright 2025 Alibaba Z-Image Team and The HuggingFace Team. All rights reserved.
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

import torch

from ...models import ZImageTransformer2DModel
from ...schedulers import FlowMatchEulerDiscreteScheduler
from ...utils import logging
from ...utils.torch_utils import randn_tensor
from ..modular_pipeline import ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam
from .modular_pipeline import ZImageModularPipeline


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# TODO(yiyi, aryan): We need another step before text encoder to set the `num_inference_steps` attribute for guider so that
# things like when to do guidance and how many conditions to be prepared can be determined. Currently, this is done by
# always assuming you want to do guidance in the Guiders. So, negative embeddings are prepared regardless of what the
# configuration of guider is.


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


def calculate_dimension_from_latents(latents: torch.Tensor, vae_scale_factor_spatial: int) -> Tuple[int, int]:
    """Calculate image dimensions from latent tensor dimensions.

    This function converts latent spatial dimensions to image spatial dimensions by multiplying the latent height/width
    by the VAE scale factor.

    Args:
        latents (torch.Tensor): The latent tensor. Must have 4 dimensions.
            Expected shapes: [batch, channels, height, width]
        vae_scale_factor (int): The scale factor used by the VAE to compress image spatial dimension.
            By default, it is 16
    Returns:
        Tuple[int, int]: The calculated image dimensions as (height, width)
    """
    latent_height, latent_width = latents.shape[2:]
    height = latent_height * vae_scale_factor_spatial // 2
    width = latent_width * vae_scale_factor_spatial // 2

    return height, width


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


class ZImageTextInputStep(ModularPipelineBlocks):
    model_name = "z-image"

    @property
    def description(self) -> str:
        return (
            "Input processing step that:\n"
            "  1. Determines `batch_size` and `dtype` based on `prompt_embeds`\n"
            "  2. Adjusts input tensor shapes based on `batch_size` (number of prompts) and `num_images_per_prompt`\n\n"
            "All input tensors are expected to have either batch_size=1 or match the batch_size\n"
            "of prompt_embeds. The tensors will be duplicated across the batch dimension to\n"
            "have a final batch_size of batch_size * num_images_per_prompt."
        )

    @property
    def expected_components(self) -> List[ComponentSpec]:
        return [
            ComponentSpec("transformer", ZImageTransformer2DModel),
        ]

    @property
    def inputs(self) -> List[InputParam]:
        return [
            InputParam("num_images_per_prompt", default=1),
            InputParam(
                "prompt_embeds",
                required=True,
                type_hint=List[torch.Tensor],
                description="Pre-generated text embeddings. Can be generated from text_encoder step.",
            ),
            InputParam(
                "negative_prompt_embeds",
                type_hint=List[torch.Tensor],
                description="Pre-generated negative text embeddings. Can be generated from text_encoder step.",
            ),
        ]

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
                description="Data type of model tensor inputs (determined by `transformer.dtype`)",
            ),
        ]

    def check_inputs(self, components, block_state):
        if block_state.prompt_embeds is not None and block_state.negative_prompt_embeds is not None:
            if not isinstance(block_state.prompt_embeds, list):
                raise ValueError(
                    f"`prompt_embeds` must be a list when passed directly, but got {type(block_state.prompt_embeds)}."
                )
            if not isinstance(block_state.negative_prompt_embeds, list):
                raise ValueError(
                    f"`negative_prompt_embeds` must be a list when passed directly, but got {type(block_state.negative_prompt_embeds)}."
                )
            if len(block_state.prompt_embeds) != len(block_state.negative_prompt_embeds):
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same length when passed directly, but"
                    f" got: `prompt_embeds` {len(block_state.prompt_embeds)} != `negative_prompt_embeds`"
                    f" {len(block_state.negative_prompt_embeds)}."
                )

    @torch.no_grad()
    def __call__(self, components: ZImageModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        self.check_inputs(components, block_state)

        block_state.batch_size = len(block_state.prompt_embeds)
        block_state.dtype = block_state.prompt_embeds[0].dtype

        if block_state.num_images_per_prompt > 1:
            prompt_embeds = [pe for pe in block_state.prompt_embeds for _ in range(block_state.num_images_per_prompt)]
            block_state.prompt_embeds = prompt_embeds

            if block_state.negative_prompt_embeds is not None:
                negative_prompt_embeds = [
                    npe for npe in block_state.negative_prompt_embeds for _ in range(block_state.num_images_per_prompt)
                ]
                block_state.negative_prompt_embeds = negative_prompt_embeds

        self.set_block_state(state, block_state)

        return components, state


class ZImageAdditionalInputsStep(ModularPipelineBlocks):
    model_name = "z-image"

    def __init__(
        self,
        image_latent_inputs: List[str] = ["image_latents"],
        additional_batch_inputs: List[str] = [],
    ):
        """Initialize a configurable step that standardizes the inputs for the denoising step. It:\n"

        This step handles multiple common tasks to prepare inputs for the denoising step:
        1. For encoded image latents, use it update height/width if None, and expands batch size
        2. For additional_batch_inputs: Only expands batch dimensions to match final batch size

        This is a dynamic block that allows you to configure which inputs to process.

        Args:
            image_latent_inputs (List[str], optional): Names of image latent tensors to process.
                In additional to adjust batch size of these inputs, they will be used to determine height/width. Can be
                a single string or list of strings. Defaults to ["image_latents"].
            additional_batch_inputs (List[str], optional):
                Names of additional conditional input tensors to expand batch size. These tensors will only have their
                batch dimensions adjusted to match the final batch size. Can be a single string or list of strings.
                Defaults to [].

        Examples:
            # Configure to process image_latents (default behavior) ZImageAdditionalInputsStep()

            # Configure to process multiple image latent inputs
            ZImageAdditionalInputsStep(image_latent_inputs=["image_latents", "control_image_latents"])

            # Configure to process image latents and additional batch inputs ZImageAdditionalInputsStep(
                image_latent_inputs=["image_latents"], additional_batch_inputs=["image_embeds"]
            )
        """
        if not isinstance(image_latent_inputs, list):
            image_latent_inputs = [image_latent_inputs]
        if not isinstance(additional_batch_inputs, list):
            additional_batch_inputs = [additional_batch_inputs]

        self._image_latent_inputs = image_latent_inputs
        self._additional_batch_inputs = additional_batch_inputs
        super().__init__()

    @property
    def description(self) -> str:
        # Functionality section
        summary_section = (
            "Input processing step that:\n"
            "  1. For image latent inputs: Updates height/width if None, and expands batch size\n"
            "  2. For additional batch inputs: Expands batch dimensions to match final batch size"
        )

        # Inputs info
        inputs_info = ""
        if self._image_latent_inputs or self._additional_batch_inputs:
            inputs_info = "\n\nConfigured inputs:"
            if self._image_latent_inputs:
                inputs_info += f"\n  - Image latent inputs: {self._image_latent_inputs}"
            if self._additional_batch_inputs:
                inputs_info += f"\n  - Additional batch inputs: {self._additional_batch_inputs}"

        # Placement guidance
        placement_section = "\n\nThis block should be placed after the encoder steps and the text input step."

        return summary_section + inputs_info + placement_section

    @property
    def inputs(self) -> List[InputParam]:
        inputs = [
            InputParam(name="num_images_per_prompt", default=1),
            InputParam(name="batch_size", required=True),
            InputParam(name="height"),
            InputParam(name="width"),
        ]

        # Add image latent inputs
        for image_latent_input_name in self._image_latent_inputs:
            inputs.append(InputParam(name=image_latent_input_name))

        # Add additional batch inputs
        for input_name in self._additional_batch_inputs:
            inputs.append(InputParam(name=input_name))

        return inputs

    def __call__(self, components: ZImageModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        # Process image latent inputs (height/width calculation, patchify, and batch expansion)
        for image_latent_input_name in self._image_latent_inputs:
            image_latent_tensor = getattr(block_state, image_latent_input_name)
            if image_latent_tensor is None:
                continue

            # 1. Calculate num_frames, height/width from latents
            height, width = calculate_dimension_from_latents(image_latent_tensor, components.vae_scale_factor_spatial)
            block_state.height = block_state.height or height
            block_state.width = block_state.width or width

        # Process additional batch inputs (only batch expansion)
        for input_name in self._additional_batch_inputs:
            input_tensor = getattr(block_state, input_name)
            if input_tensor is None:
                continue

            # Only expand batch size
            input_tensor = repeat_tensor_to_batch_size(
                input_name=input_name,
                input_tensor=input_tensor,
                num_images_per_prompt=block_state.num_images_per_prompt,
                batch_size=block_state.batch_size,
            )

            setattr(block_state, input_name, input_tensor)

        self.set_block_state(state, block_state)
        return components, state


class ZImagePrepareLatentsStep(ModularPipelineBlocks):
    model_name = "z-image"

    @property
    def description(self) -> str:
        return "Prepare latents step that prepares the latents for the text-to-video generation process"

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
                description="Number of prompts, the final batch size of model inputs should be `batch_size * num_images_per_prompt`. Can be generated in input step.",
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

    def check_inputs(self, components, block_state):
        if (block_state.height is not None and block_state.height % components.vae_scale_factor_spatial != 0) or (
            block_state.width is not None and block_state.width % components.vae_scale_factor_spatial != 0
        ):
            raise ValueError(
                f"`height` and `width` have to be divisible by {components.vae_scale_factor_spatial} but are {block_state.height} and {block_state.width}."
            )

    @staticmethod
    # Copied from diffusers.pipelines.z_image.pipeline_z_image.ZImagePipeline.prepare_latents with self->comp
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

        shape = (batch_size, num_channels_latents, height, width)

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
            latents = latents.to(device)
        return latents

    @torch.no_grad()
    def __call__(self, components: ZImageModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        self.check_inputs(components, block_state)

        device = components._execution_device
        dtype = torch.float32

        block_state.height = block_state.height or components.default_height
        block_state.width = block_state.width or components.default_width

        block_state.latents = self.prepare_latents(
            components,
            batch_size=block_state.batch_size * block_state.num_images_per_prompt,
            num_channels_latents=components.num_channels_latents,
            height=block_state.height,
            width=block_state.width,
            dtype=dtype,
            device=device,
            generator=block_state.generator,
            latents=block_state.latents,
        )

        self.set_block_state(state, block_state)

        return components, state


class ZImageSetTimestepsStep(ModularPipelineBlocks):
    model_name = "z-image"

    @property
    def expected_components(self) -> List[ComponentSpec]:
        return [
            ComponentSpec("scheduler", FlowMatchEulerDiscreteScheduler),
        ]

    @property
    def description(self) -> str:
        return "Step that sets the scheduler's timesteps for inference. Need to run after prepare latents step."

    @property
    def inputs(self) -> List[InputParam]:
        return [
            InputParam("latents", required=True),
            InputParam("num_inference_steps", default=9),
            InputParam("sigmas"),
        ]

    @property
    def intermediate_outputs(self) -> List[OutputParam]:
        return [
            OutputParam(
                "timesteps", type_hint=torch.Tensor, description="The timesteps to use for the denoising process"
            ),
        ]

    @torch.no_grad()
    def __call__(self, components: ZImageModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        device = components._execution_device

        latent_height, latent_width = block_state.latents.shape[2], block_state.latents.shape[3]
        image_seq_len = (latent_height // 2) * (latent_width // 2)  # sequence length  after patchify

        mu = calculate_shift(
            image_seq_len,
            base_seq_len=components.scheduler.config.get("base_image_seq_len", 256),
            max_seq_len=components.scheduler.config.get("max_image_seq_len", 4096),
            base_shift=components.scheduler.config.get("base_shift", 0.5),
            max_shift=components.scheduler.config.get("max_shift", 1.15),
        )
        components.scheduler.sigma_min = 0.0

        block_state.timesteps, block_state.num_inference_steps = retrieve_timesteps(
            components.scheduler,
            block_state.num_inference_steps,
            device,
            sigmas=block_state.sigmas,
            mu=mu,
        )

        self.set_block_state(state, block_state)
        return components, state


class ZImageSetTimestepsWithStrengthStep(ModularPipelineBlocks):
    model_name = "z-image"

    @property
    def expected_components(self) -> List[ComponentSpec]:
        return [
            ComponentSpec("scheduler", FlowMatchEulerDiscreteScheduler),
        ]

    @property
    def description(self) -> str:
        return "Step that sets the scheduler's timesteps for inference with strength. Need to run after set timesteps step."

    @property
    def inputs(self) -> List[InputParam]:
        return [
            InputParam("timesteps", required=True),
            InputParam("num_inference_steps", required=True),
            InputParam("strength", default=0.6),
        ]

    def check_inputs(self, components, block_state):
        if block_state.strength < 0.0 or block_state.strength > 1.0:
            raise ValueError(f"Strength must be between 0.0 and 1.0, but got {block_state.strength}")

    @torch.no_grad()
    def __call__(self, components: ZImageModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        self.check_inputs(components, block_state)

        init_timestep = min(block_state.num_inference_steps * block_state.strength, block_state.num_inference_steps)

        t_start = int(max(block_state.num_inference_steps - init_timestep, 0))
        timesteps = components.scheduler.timesteps[t_start * components.scheduler.order :]
        if hasattr(components.scheduler, "set_begin_index"):
            components.scheduler.set_begin_index(t_start * components.scheduler.order)

        block_state.timesteps = timesteps
        block_state.num_inference_steps = block_state.num_inference_steps - t_start

        self.set_block_state(state, block_state)
        return components, state


class ZImagePrepareLatentswithImageStep(ModularPipelineBlocks):
    model_name = "z-image"

    @property
    def description(self) -> str:
        return "step that prepares the latents with image condition, need to run after set timesteps and prepare latents step."

    @property
    def inputs(self) -> List[InputParam]:
        return [
            InputParam("latents", required=True),
            InputParam("image_latents", required=True),
            InputParam("timesteps", required=True),
        ]

    def __call__(self, components: ZImageModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        latent_timestep = block_state.timesteps[:1].repeat(block_state.latents.shape[0])
        block_state.latents = components.scheduler.scale_noise(
            block_state.image_latents, latent_timestep, block_state.latents
        )

        self.set_block_state(state, block_state)
        return components, state
