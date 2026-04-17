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

import numpy as np
import torch

from ...models import HeliosTransformer3DModel
from ...schedulers import HeliosScheduler
from ...utils import logging
from ...utils.torch_utils import randn_tensor
from ..modular_pipeline import ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam
from .modular_pipeline import HeliosModularPipeline


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


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


class HeliosTextInputStep(ModularPipelineBlocks):
    model_name = "helios"

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
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                "num_videos_per_prompt",
                default=1,
                type_hint=int,
                description="Number of videos to generate per prompt.",
            ),
            InputParam.template("prompt_embeds"),
            InputParam.template("negative_prompt_embeds"),
        ]

    @property
    def intermediate_outputs(self) -> list[str]:
        return [
            OutputParam(
                "batch_size",
                type_hint=int,
                description="Number of prompts, the final batch size of model inputs should be batch_size * num_videos_per_prompt",
            ),
            OutputParam(
                "dtype",
                type_hint=torch.dtype,
                description="Data type of model tensor inputs (determined by `prompt_embeds.dtype`)",
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
    def __call__(self, components: HeliosModularPipeline, state: PipelineState) -> PipelineState:
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


# Copied from diffusers.modular_pipelines.wan.before_denoise.repeat_tensor_to_batch_size
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


# Copied from diffusers.modular_pipelines.wan.before_denoise.calculate_dimension_from_latents
def calculate_dimension_from_latents(
    latents: torch.Tensor, vae_scale_factor_temporal: int, vae_scale_factor_spatial: int
) -> tuple[int, int]:
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
        tuple[int, int]: The calculated image dimensions as (height, width)

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


class HeliosAdditionalInputsStep(ModularPipelineBlocks):
    """Configurable step that standardizes inputs for the denoising step.

    This step handles:
    1. For encoded image latents: Computes height/width from latents and expands batch size
    2. For additional_batch_inputs: Expands batch dimensions to match final batch size
    """

    model_name = "helios"

    def __init__(
        self,
        image_latent_inputs: list[InputParam] | None = None,
        additional_batch_inputs: list[InputParam] | None = None,
    ):
        if image_latent_inputs is None:
            image_latent_inputs = [InputParam.template("image_latents")]
        if additional_batch_inputs is None:
            additional_batch_inputs = []

        if not isinstance(image_latent_inputs, list):
            raise ValueError(f"image_latent_inputs must be a list, but got {type(image_latent_inputs)}")
        else:
            for input_param in image_latent_inputs:
                if not isinstance(input_param, InputParam):
                    raise ValueError(f"image_latent_inputs must be a list of InputParam, but got {type(input_param)}")

        if not isinstance(additional_batch_inputs, list):
            raise ValueError(f"additional_batch_inputs must be a list, but got {type(additional_batch_inputs)}")
        else:
            for input_param in additional_batch_inputs:
                if not isinstance(input_param, InputParam):
                    raise ValueError(
                        f"additional_batch_inputs must be a list of InputParam, but got {type(input_param)}"
                    )

        self._image_latent_inputs = image_latent_inputs
        self._additional_batch_inputs = additional_batch_inputs
        super().__init__()

    @property
    def description(self) -> str:
        summary_section = (
            "Input processing step that:\n"
            "  1. For image latent inputs: Computes height/width from latents and expands batch size\n"
            "  2. For additional batch inputs: Expands batch dimensions to match final batch size"
        )

        inputs_info = ""
        if self._image_latent_inputs or self._additional_batch_inputs:
            inputs_info = "\n\nConfigured inputs:"
            if self._image_latent_inputs:
                inputs_info += f"\n  - Image latent inputs: {[p.name for p in self._image_latent_inputs]}"
            if self._additional_batch_inputs:
                inputs_info += f"\n  - Additional batch inputs: {[p.name for p in self._additional_batch_inputs]}"

        placement_section = "\n\nThis block should be placed after the encoder steps and the text input step."

        return summary_section + inputs_info + placement_section

    @property
    def inputs(self) -> list[InputParam]:
        inputs = [
            InputParam(name="num_videos_per_prompt", default=1),
            InputParam(name="batch_size", required=True),
        ]
        inputs += self._image_latent_inputs + self._additional_batch_inputs

        return inputs

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        outputs = [
            OutputParam("height", type_hint=int),
            OutputParam("width", type_hint=int),
        ]

        for input_param in self._image_latent_inputs:
            outputs.append(OutputParam(input_param.name, type_hint=torch.Tensor))

        for input_param in self._additional_batch_inputs:
            outputs.append(OutputParam(input_param.name, type_hint=torch.Tensor))

        return outputs

    def __call__(self, components: HeliosModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        for input_param in self._image_latent_inputs:
            image_latent_tensor = getattr(block_state, input_param.name)
            if image_latent_tensor is None:
                continue

            # Calculate height/width from latents
            _, height, width = calculate_dimension_from_latents(
                image_latent_tensor, components.vae_scale_factor_temporal, components.vae_scale_factor_spatial
            )
            block_state.height = height
            block_state.width = width

            # Expand batch size
            image_latent_tensor = repeat_tensor_to_batch_size(
                input_name=input_param.name,
                input_tensor=image_latent_tensor,
                num_videos_per_prompt=block_state.num_videos_per_prompt,
                batch_size=block_state.batch_size,
            )

            setattr(block_state, input_param.name, image_latent_tensor)

        for input_param in self._additional_batch_inputs:
            input_tensor = getattr(block_state, input_param.name)
            if input_tensor is None:
                continue

            input_tensor = repeat_tensor_to_batch_size(
                input_name=input_param.name,
                input_tensor=input_tensor,
                num_videos_per_prompt=block_state.num_videos_per_prompt,
                batch_size=block_state.batch_size,
            )

            setattr(block_state, input_param.name, input_tensor)

        self.set_block_state(state, block_state)
        return components, state


class HeliosAddNoiseToImageLatentsStep(ModularPipelineBlocks):
    """Adds noise to image_latents and fake_image_latents for I2V conditioning.

    Applies single-sigma noise to image_latents (using image_noise_sigma range) and single-sigma noise to
    fake_image_latents (using video_noise_sigma range).
    """

    model_name = "helios"

    @property
    def description(self) -> str:
        return (
            "Adds noise to image_latents and fake_image_latents for I2V conditioning. "
            "Uses random sigma from configured ranges for each."
        )

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("image_latents"),
            InputParam(
                "fake_image_latents",
                required=True,
                type_hint=torch.Tensor,
                description="Fake image latents used as history seed for I2V generation.",
            ),
            InputParam(
                "image_noise_sigma_min",
                default=0.111,
                type_hint=float,
                description="Minimum sigma for image latent noise.",
            ),
            InputParam(
                "image_noise_sigma_max",
                default=0.135,
                type_hint=float,
                description="Maximum sigma for image latent noise.",
            ),
            InputParam(
                "video_noise_sigma_min",
                default=0.111,
                type_hint=float,
                description="Minimum sigma for video/fake-image latent noise.",
            ),
            InputParam(
                "video_noise_sigma_max",
                default=0.135,
                type_hint=float,
                description="Maximum sigma for video/fake-image latent noise.",
            ),
            InputParam.template("generator"),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam.template("image_latents"),
            OutputParam("fake_image_latents", type_hint=torch.Tensor, description="Noisy fake image latents"),
        ]

    @torch.no_grad()
    def __call__(self, components: HeliosModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        device = components._execution_device
        image_latents = block_state.image_latents
        fake_image_latents = block_state.fake_image_latents

        # Add noise to image_latents
        image_noise_sigma = (
            torch.rand(1, device=device, generator=block_state.generator)
            * (block_state.image_noise_sigma_max - block_state.image_noise_sigma_min)
            + block_state.image_noise_sigma_min
        )
        image_latents = (
            image_noise_sigma * randn_tensor(image_latents.shape, generator=block_state.generator, device=device)
            + (1 - image_noise_sigma) * image_latents
        )

        # Add noise to fake_image_latents
        fake_image_noise_sigma = (
            torch.rand(1, device=device, generator=block_state.generator)
            * (block_state.video_noise_sigma_max - block_state.video_noise_sigma_min)
            + block_state.video_noise_sigma_min
        )
        fake_image_latents = (
            fake_image_noise_sigma
            * randn_tensor(fake_image_latents.shape, generator=block_state.generator, device=device)
            + (1 - fake_image_noise_sigma) * fake_image_latents
        )

        block_state.image_latents = image_latents.to(device=device, dtype=torch.float32)
        block_state.fake_image_latents = fake_image_latents.to(device=device, dtype=torch.float32)

        self.set_block_state(state, block_state)
        return components, state


class HeliosAddNoiseToVideoLatentsStep(ModularPipelineBlocks):
    """Adds noise to image_latents and video_latents for V2V conditioning.

    Applies single-sigma noise to image_latents (using image_noise_sigma range) and per-frame noise to video_latents in
    chunks (using video_noise_sigma range).
    """

    model_name = "helios"

    @property
    def description(self) -> str:
        return (
            "Adds noise to image_latents and video_latents for V2V conditioning. "
            "Uses single-sigma noise for image_latents and per-frame noise for video chunks."
        )

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("image_latents"),
            InputParam(
                "video_latents",
                required=True,
                type_hint=torch.Tensor,
                description="Encoded video latents for V2V generation.",
            ),
            InputParam(
                "num_latent_frames_per_chunk",
                default=9,
                type_hint=int,
                description="Number of latent frames per temporal chunk.",
            ),
            InputParam(
                "image_noise_sigma_min",
                default=0.111,
                type_hint=float,
                description="Minimum sigma for image latent noise.",
            ),
            InputParam(
                "image_noise_sigma_max",
                default=0.135,
                type_hint=float,
                description="Maximum sigma for image latent noise.",
            ),
            InputParam(
                "video_noise_sigma_min",
                default=0.111,
                type_hint=float,
                description="Minimum sigma for video latent noise.",
            ),
            InputParam(
                "video_noise_sigma_max",
                default=0.135,
                type_hint=float,
                description="Maximum sigma for video latent noise.",
            ),
            InputParam.template("generator"),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam.template("image_latents"),
            OutputParam("video_latents", type_hint=torch.Tensor, description="Noisy video latents"),
        ]

    @torch.no_grad()
    def __call__(self, components: HeliosModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        device = components._execution_device
        image_latents = block_state.image_latents
        video_latents = block_state.video_latents
        num_latent_frames_per_chunk = block_state.num_latent_frames_per_chunk

        # Add noise to first frame (single sigma)
        image_noise_sigma = (
            torch.rand(1, device=device, generator=block_state.generator)
            * (block_state.image_noise_sigma_max - block_state.image_noise_sigma_min)
            + block_state.image_noise_sigma_min
        )
        image_latents = (
            image_noise_sigma * randn_tensor(image_latents.shape, generator=block_state.generator, device=device)
            + (1 - image_noise_sigma) * image_latents
        )

        # Add per-frame noise to video chunks
        noisy_latents_chunks = []
        num_latent_chunks = video_latents.shape[2] // num_latent_frames_per_chunk
        for i in range(num_latent_chunks):
            chunk_start = i * num_latent_frames_per_chunk
            chunk_end = chunk_start + num_latent_frames_per_chunk
            latent_chunk = video_latents[:, :, chunk_start:chunk_end, :, :]

            chunk_frames = latent_chunk.shape[2]
            frame_sigmas = (
                torch.rand(chunk_frames, device=device, generator=block_state.generator)
                * (block_state.video_noise_sigma_max - block_state.video_noise_sigma_min)
                + block_state.video_noise_sigma_min
            )
            frame_sigmas = frame_sigmas.view(1, 1, chunk_frames, 1, 1)

            noisy_chunk = (
                frame_sigmas * randn_tensor(latent_chunk.shape, generator=block_state.generator, device=device)
                + (1 - frame_sigmas) * latent_chunk
            )
            noisy_latents_chunks.append(noisy_chunk)
        video_latents = torch.cat(noisy_latents_chunks, dim=2)

        block_state.image_latents = image_latents.to(device=device, dtype=torch.float32)
        block_state.video_latents = video_latents.to(device=device, dtype=torch.float32)

        self.set_block_state(state, block_state)
        return components, state


class HeliosPrepareHistoryStep(ModularPipelineBlocks):
    """Prepares chunk/history indices and initializes history state for the chunk loop."""

    model_name = "helios"

    @property
    def description(self) -> str:
        return (
            "Prepares the chunk loop by computing latent dimensions, number of chunks, "
            "history indices, and initializing history state (history_latents, image_latents, latent_chunks)."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("transformer", HeliosTransformer3DModel),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("height", default=384),
            InputParam.template("width", default=640),
            InputParam(
                "num_frames", default=132, type_hint=int, description="Total number of video frames to generate."
            ),
            InputParam("batch_size", required=True, type_hint=int),
            InputParam(
                "num_latent_frames_per_chunk",
                default=9,
                type_hint=int,
                description="Number of latent frames per temporal chunk.",
            ),
            InputParam(
                "history_sizes",
                default=[16, 2, 1],
                type_hint=list,
                description="Sizes of long/mid/short history buffers for temporal context.",
            ),
            InputParam(
                "keep_first_frame",
                default=True,
                type_hint=bool,
                description="Whether to keep the first frame as a prefix in history.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("num_latent_chunk", type_hint=int, description="Number of temporal chunks"),
            OutputParam("latent_shape", type_hint=tuple, description="Shape of latent tensor per chunk"),
            OutputParam("history_sizes", type_hint=list, description="Adjusted history sizes (sorted, descending)"),
            OutputParam("indices_hidden_states", type_hint=torch.Tensor, kwargs_type="denoiser_input_fields"),
            OutputParam("indices_latents_history_short", type_hint=torch.Tensor, kwargs_type="denoiser_input_fields"),
            OutputParam("indices_latents_history_mid", type_hint=torch.Tensor, kwargs_type="denoiser_input_fields"),
            OutputParam("indices_latents_history_long", type_hint=torch.Tensor, kwargs_type="denoiser_input_fields"),
            OutputParam("history_latents", type_hint=torch.Tensor, description="Initialized zero history latents"),
        ]

    @torch.no_grad()
    def __call__(self, components: HeliosModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        batch_size = block_state.batch_size
        device = components._execution_device

        block_state.num_frames = max(block_state.num_frames, 1)
        history_sizes = sorted(block_state.history_sizes, reverse=True)

        num_channels_latents = components.num_channels_latents
        h_latent = block_state.height // components.vae_scale_factor_spatial
        w_latent = block_state.width // components.vae_scale_factor_spatial

        # Compute number of chunks
        block_state.window_num_frames = (
            block_state.num_latent_frames_per_chunk - 1
        ) * components.vae_scale_factor_temporal + 1
        block_state.num_latent_chunk = max(
            1, (block_state.num_frames + block_state.window_num_frames - 1) // block_state.window_num_frames
        )

        # Modify history_sizes for non-keep_first_frame (matching pipeline behavior)
        if not block_state.keep_first_frame:
            history_sizes = history_sizes.copy()
            history_sizes[-1] = history_sizes[-1] + 1

        # Compute indices ONCE (same structure for all chunks)
        if block_state.keep_first_frame:
            indices = torch.arange(0, sum([1, *history_sizes, block_state.num_latent_frames_per_chunk]))
            (
                indices_prefix,
                indices_latents_history_long,
                indices_latents_history_mid,
                indices_latents_history_1x,
                indices_hidden_states,
            ) = indices.split([1, *history_sizes, block_state.num_latent_frames_per_chunk], dim=0)
            indices_latents_history_short = torch.cat([indices_prefix, indices_latents_history_1x], dim=0)
        else:
            indices = torch.arange(0, sum([*history_sizes, block_state.num_latent_frames_per_chunk]))
            (
                indices_latents_history_long,
                indices_latents_history_mid,
                indices_latents_history_short,
                indices_hidden_states,
            ) = indices.split([*history_sizes, block_state.num_latent_frames_per_chunk], dim=0)

        # Latent shape per chunk
        block_state.latent_shape = (
            batch_size,
            num_channels_latents,
            block_state.num_latent_frames_per_chunk,
            h_latent,
            w_latent,
        )

        # Set outputs
        block_state.history_sizes = history_sizes
        block_state.indices_hidden_states = indices_hidden_states.unsqueeze(0)
        block_state.indices_latents_history_short = indices_latents_history_short.unsqueeze(0)
        block_state.indices_latents_history_mid = indices_latents_history_mid.unsqueeze(0)
        block_state.indices_latents_history_long = indices_latents_history_long.unsqueeze(0)
        block_state.history_latents = torch.zeros(
            batch_size,
            num_channels_latents,
            sum(history_sizes),
            h_latent,
            w_latent,
            device=device,
            dtype=torch.float32,
        )

        self.set_block_state(state, block_state)

        return components, state


class HeliosI2VSeedHistoryStep(ModularPipelineBlocks):
    """Seeds history_latents with fake_image_latents for I2V pipelines.

    This small additive step runs after HeliosPrepareHistoryStep and appends fake_image_latents to the initialized
    history_latents tensor.
    """

    model_name = "helios"

    @property
    def description(self) -> str:
        return "I2V history seeding: appends fake_image_latents to history_latents."

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("history_latents", required=True, type_hint=torch.Tensor),
            InputParam("fake_image_latents", required=True, type_hint=torch.Tensor),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "history_latents", type_hint=torch.Tensor, description="History latents seeded with fake_image_latents"
            ),
        ]

    @torch.no_grad()
    def __call__(self, components: HeliosModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        block_state.history_latents = torch.cat([block_state.history_latents, block_state.fake_image_latents], dim=2)

        self.set_block_state(state, block_state)
        return components, state


class HeliosV2VSeedHistoryStep(ModularPipelineBlocks):
    """Seeds history_latents with video_latents for V2V pipelines.

    This step runs after HeliosPrepareHistoryStep and replaces the tail of history_latents with video_latents. If the
    video has fewer frames than the history, the beginning of history is preserved.
    """

    model_name = "helios"

    @property
    def description(self) -> str:
        return "V2V history seeding: replaces the tail of history_latents with video_latents."

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("history_latents", required=True, type_hint=torch.Tensor),
            InputParam("video_latents", required=True, type_hint=torch.Tensor),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "history_latents", type_hint=torch.Tensor, description="History latents seeded with video_latents"
            ),
        ]

    @torch.no_grad()
    def __call__(self, components: HeliosModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        history_latents = block_state.history_latents
        video_latents = block_state.video_latents

        history_frames = history_latents.shape[2]
        video_frames = video_latents.shape[2]
        if video_frames < history_frames:
            keep_frames = history_frames - video_frames
            history_latents = torch.cat([history_latents[:, :, :keep_frames, :, :], video_latents], dim=2)
        else:
            history_latents = video_latents

        block_state.history_latents = history_latents

        self.set_block_state(state, block_state)
        return components, state


class HeliosSetTimestepsStep(ModularPipelineBlocks):
    """Computes scheduler parameters (mu, sigmas) for the chunk loop."""

    model_name = "helios"

    @property
    def description(self) -> str:
        return "Computes scheduler shift parameter (mu) and default sigmas for the Helios chunk loop."

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("transformer", HeliosTransformer3DModel),
            ComponentSpec("scheduler", HeliosScheduler),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("latent_shape", required=True, type_hint=tuple),
            InputParam.template("num_inference_steps"),
            InputParam.template("sigmas"),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("mu", type_hint=float, description="Scheduler shift parameter"),
            OutputParam("sigmas", type_hint=list, description="Sigma schedule for diffusion"),
        ]

    @torch.no_grad()
    def __call__(self, components: HeliosModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        patch_size = components.transformer.config.patch_size
        latent_shape = block_state.latent_shape
        image_seq_len = (latent_shape[-1] * latent_shape[-2] * latent_shape[-3]) // (
            patch_size[0] * patch_size[1] * patch_size[2]
        )

        if block_state.sigmas is None:
            block_state.sigmas = np.linspace(0.999, 0.0, block_state.num_inference_steps + 1)[:-1]

        block_state.mu = calculate_shift(
            image_seq_len,
            components.scheduler.config.get("base_image_seq_len", 256),
            components.scheduler.config.get("max_image_seq_len", 4096),
            components.scheduler.config.get("base_shift", 0.5),
            components.scheduler.config.get("max_shift", 1.15),
        )

        self.set_block_state(state, block_state)

        return components, state
