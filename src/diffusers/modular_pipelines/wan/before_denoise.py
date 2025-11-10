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
from typing import List, Optional, Tuple, Union

import torch

from ...models import WanTransformer3DModel
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


def repeat_tensor_to_batch_size(
    input_name: str,
    input_tensor: torch.Tensor,
    batch_size: int,
    num_videos_per_prompt: int = 1,
) -> torch.Tensor:
    """Repeat tensor elements to match the final batch size.

    This function expands a tensor's batch dimension to match the final batch size (batch_size * num_videos_per_prompt)
    by repeating each element along dimension 0.

    The input tensor must have batch size 1 or batch_size. The function will:
    - If batch size is 1: repeat each element (batch_size * num_videos_per_prompt) times
    - If batch size equals batch_size: repeat each element num_videos_per_prompt times

    Args:
        input_name (str): Name of the input tensor (used for error messages)
        input_tensor (torch.Tensor): The tensor to repeat. Must have batch size 1 or batch_size.
        batch_size (int): The base batch size (number of prompts)
        num_videos_per_prompt (int, optional): Number of videos to generate per prompt. Defaults to 1.

    Returns:
        torch.Tensor: The repeated tensor with final batch size (batch_size * num_videos_per_prompt)

    Raises:
        ValueError: If input_tensor is not a torch.Tensor or has invalid batch size

    Examples:
        tensor = torch.tensor([[1, 2, 3]]) # shape: [1, 3] repeated = repeat_tensor_to_batch_size("image", tensor,
        batch_size=2, num_videos_per_prompt=2) repeated # tensor([[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]]) - shape:
        [4, 3]

        tensor = torch.tensor([[1, 2, 3], [4, 5, 6]]) # shape: [2, 3] repeated = repeat_tensor_to_batch_size("image",
        tensor, batch_size=2, num_videos_per_prompt=2) repeated # tensor([[1, 2, 3], [1, 2, 3], [4, 5, 6], [4, 5, 6]])
        - shape: [4, 3]
    """
    # make sure input is a tensor
    if not isinstance(input_tensor, torch.Tensor):
        raise ValueError(f"`{input_name}` must be a tensor")

    # make sure input tensor e.g. image_latents has batch size 1 or batch_size same as prompts
    if input_tensor.shape[0] == 1:
        repeat_by = batch_size * num_videos_per_prompt
    elif input_tensor.shape[0] == batch_size:
        repeat_by = num_videos_per_prompt
    else:
        raise ValueError(
            f"`{input_name}` must have have batch size 1 or {batch_size}, but got {input_tensor.shape[0]}"
        )

    # expand the tensor to match the batch_size * num_videos_per_prompt
    input_tensor = input_tensor.repeat_interleave(repeat_by, dim=0)

    return input_tensor


def calculate_dimension_from_latents(
    latents: torch.Tensor, vae_scale_factor_temporal: int, vae_scale_factor_spatial: int
) -> Tuple[int, int]:
    """Calculate image dimensions from latent tensor dimensions.

    This function converts latent temporal and spatial dimensions to image temporal and spatial dimensions by
    multiplying the latent num_frames/height/width by the VAE scale factor.

    Args:
        latents (torch.Tensor): The latent tensor. Must have 4 or 5 dimensions.
            Expected shapes: [batch, channels, height, width] or [batch, channels, frames, height, width]
        vae_scale_factor_temporal (int): The scale factor used by the VAE to compress temporal dimension.
            Typically 4 for most VAEs (video is 4x larger than latents in temporal dimension)
        vae_scale_factor_spatial (int): The scale factor used by the VAE to compress spatial dimension.
            Typically 8 for most VAEs (image is 8x larger than latents in each dimension)

    Returns:
        Tuple[int, int]: The calculated image dimensions as (height, width)

    Raises:
        ValueError: If latents tensor doesn't have 4 or 5 dimensions

    """
    if latents.ndim != 5:
        raise ValueError(f"latents must have 5 dimensions, but got {latents.ndim}")

    _, _, num_latent_frames, latent_height, latent_width = latents.shape

    num_frames = (num_latent_frames - 1) * vae_scale_factor_temporal + 1
    height = latent_height * vae_scale_factor_spatial
    width = latent_width * vae_scale_factor_spatial

    return num_frames, height, width


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


class WanTextInputStep(ModularPipelineBlocks):
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
    def expected_components(self) -> List[ComponentSpec]:
        return [
            ComponentSpec("transformer", WanTransformer3DModel),
        ]

    @property
    def inputs(self) -> List[InputParam]:
        return [
            InputParam("num_videos_per_prompt", default=1),
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
                description="Data type of model tensor inputs (determined by `transformer.dtype`)",
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


class WanAdditionalInputsStep(ModularPipelineBlocks):
    model_name = "wan"

    def __init__(
        self,
        image_latent_inputs: List[str] = ["first_frame_latents"],
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
                a single string or list of strings. Defaults to ["first_frame_latents"].
            additional_batch_inputs (List[str], optional):
                Names of additional conditional input tensors to expand batch size. These tensors will only have their
                batch dimensions adjusted to match the final batch size. Can be a single string or list of strings.
                Defaults to [].

        Examples:
            # Configure to process first_frame_latents (default behavior) WanAdditionalInputsStep()

            # Configure to process multiple image latent inputs
            WanAdditionalInputsStep(image_latent_inputs=["first_frame_latents", "last_frame_latents"])

            # Configure to process image latents and additional batch inputs WanAdditionalInputsStep(
                image_latent_inputs=["first_frame_latents"], additional_batch_inputs=["image_embeds"]
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
            InputParam(name="num_videos_per_prompt", default=1),
            InputParam(name="batch_size", required=True),
            InputParam(name="height"),
            InputParam(name="width"),
            InputParam(name="num_frames"),
        ]

        # Add image latent inputs
        for image_latent_input_name in self._image_latent_inputs:
            inputs.append(InputParam(name=image_latent_input_name))

        # Add additional batch inputs
        for input_name in self._additional_batch_inputs:
            inputs.append(InputParam(name=input_name))

        return inputs

    def __call__(self, components: WanModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        # Process image latent inputs (height/width calculation, patchify, and batch expansion)
        for image_latent_input_name in self._image_latent_inputs:
            image_latent_tensor = getattr(block_state, image_latent_input_name)
            if image_latent_tensor is None:
                continue

            # 1. Calculate num_frames, height/width from latents
            num_frames, height, width = calculate_dimension_from_latents(
                image_latent_tensor, components.vae_scale_factor_temporal, components.vae_scale_factor_spatial
            )
            block_state.num_frames = block_state.num_frames or num_frames
            block_state.height = block_state.height or height
            block_state.width = block_state.width or width

            # 3. Expand batch size
            image_latent_tensor = repeat_tensor_to_batch_size(
                input_name=image_latent_input_name,
                input_tensor=image_latent_tensor,
                num_videos_per_prompt=block_state.num_videos_per_prompt,
                batch_size=block_state.batch_size,
            )

            setattr(block_state, image_latent_input_name, image_latent_tensor)

        # Process additional batch inputs (only batch expansion)
        for input_name in self._additional_batch_inputs:
            input_tensor = getattr(block_state, input_name)
            if input_tensor is None:
                continue

            # Only expand batch size
            input_tensor = repeat_tensor_to_batch_size(
                input_name=input_name,
                input_tensor=input_tensor,
                num_videos_per_prompt=block_state.num_videos_per_prompt,
                batch_size=block_state.batch_size,
            )

            setattr(block_state, input_name, input_tensor)

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

    @torch.no_grad()
    def __call__(self, components: WanModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        device = components._execution_device

        block_state.timesteps, block_state.num_inference_steps = retrieve_timesteps(
            components.scheduler,
            block_state.num_inference_steps,
            device,
            block_state.timesteps,
            block_state.sigmas,
        )

        self.set_block_state(state, block_state)
        return components, state


class WanPrepareLatentsStep(ModularPipelineBlocks):
    model_name = "wan"

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
        self.check_inputs(components, block_state)

        device = components._execution_device
        dtype = torch.float32  # Wan latents should be torch.float32 for best quality

        block_state.height = block_state.height or components.default_height
        block_state.width = block_state.width or components.default_width
        block_state.num_frames = block_state.num_frames or components.default_num_frames

        block_state.latents = self.prepare_latents(
            components,
            batch_size=block_state.batch_size * block_state.num_videos_per_prompt,
            num_channels_latents=components.num_channels_latents,
            height=block_state.height,
            width=block_state.width,
            num_frames=block_state.num_frames,
            dtype=dtype,
            device=device,
            generator=block_state.generator,
            latents=block_state.latents,
        )

        self.set_block_state(state, block_state)

        return components, state


class WanPrepareFirstFrameLatentsStep(ModularPipelineBlocks):
    model_name = "wan"

    @property
    def description(self) -> str:
        return "step that prepares the masked first frame latents and add it to the latent condition"

    @property
    def inputs(self) -> List[InputParam]:
        return [
            InputParam("first_frame_latents", type_hint=Optional[torch.Tensor]),
            InputParam("num_frames", type_hint=int),
        ]

    def __call__(self, components: WanModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        batch_size, _, _, latent_height, latent_width = block_state.first_frame_latents.shape

        mask_lat_size = torch.ones(batch_size, 1, block_state.num_frames, latent_height, latent_width)
        mask_lat_size[:, :, list(range(1, block_state.num_frames))] = 0

        first_frame_mask = mask_lat_size[:, :, 0:1]
        first_frame_mask = torch.repeat_interleave(
            first_frame_mask, dim=2, repeats=components.vae_scale_factor_temporal
        )
        mask_lat_size = torch.concat([first_frame_mask, mask_lat_size[:, :, 1:, :]], dim=2)
        mask_lat_size = mask_lat_size.view(
            batch_size, -1, components.vae_scale_factor_temporal, latent_height, latent_width
        )
        mask_lat_size = mask_lat_size.transpose(1, 2)
        mask_lat_size = mask_lat_size.to(block_state.first_frame_latents.device)
        block_state.first_frame_latents = torch.concat([mask_lat_size, block_state.first_frame_latents], dim=1)

        self.set_block_state(state, block_state)
        return components, state


class WanPrepareFirstLastFrameLatentsStep(ModularPipelineBlocks):
    model_name = "wan"

    @property
    def description(self) -> str:
        return "step that prepares the masked latents with first and last frames and add it to the latent condition"

    @property
    def inputs(self) -> List[InputParam]:
        return [
            InputParam("first_last_frame_latents", type_hint=Optional[torch.Tensor]),
            InputParam("num_frames", type_hint=int),
        ]

    def __call__(self, components: WanModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        batch_size, _, _, latent_height, latent_width = block_state.first_last_frame_latents.shape

        mask_lat_size = torch.ones(batch_size, 1, block_state.num_frames, latent_height, latent_width)
        mask_lat_size[:, :, list(range(1, block_state.num_frames - 1))] = 0

        first_frame_mask = mask_lat_size[:, :, 0:1]
        first_frame_mask = torch.repeat_interleave(
            first_frame_mask, dim=2, repeats=components.vae_scale_factor_temporal
        )
        mask_lat_size = torch.concat([first_frame_mask, mask_lat_size[:, :, 1:, :]], dim=2)
        mask_lat_size = mask_lat_size.view(
            batch_size, -1, components.vae_scale_factor_temporal, latent_height, latent_width
        )
        mask_lat_size = mask_lat_size.transpose(1, 2)
        mask_lat_size = mask_lat_size.to(block_state.first_last_frame_latents.device)
        block_state.first_last_frame_latents = torch.concat(
            [mask_lat_size, block_state.first_last_frame_latents], dim=1
        )

        self.set_block_state(state, block_state)
        return components, state
