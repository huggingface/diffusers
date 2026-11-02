# Copyright 2026 Black Forest Labs and The HuggingFace Team. All rights reserved.
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
from ...schedulers import FlowMatchEulerDiscreteScheduler
from ...utils import logging
from ...utils.torch_utils import randn_tensor
from ..modular_pipeline import ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam
from .modular_pipeline import Flux3ModularPipeline


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


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


class Flux3SetTimestepsStep(ModularPipelineBlocks):
    model_name = "flux3"

    def __init__(self, shift: float = 6.93, num_inference_steps: int = 50):
        # the constant timestep shift ("alpha") of the FLUX.3 sampling recipe, configured on the scheduler, and the
        # default number of steps (50; 16 for the step-distilled checkpoints)
        self.shift = shift
        self.num_inference_steps = num_inference_steps
        super().__init__()

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec(
                "scheduler",
                FlowMatchEulerDiscreteScheduler,
                config=FrozenDict({"shift": self.shift}),
                default_creation_method="from_config",
            ),
        ]

    @property
    def description(self) -> str:
        return (
            "Step that sets the scheduler's timesteps for FLUX.3 inference (constant shift, configured on the "
            "scheduler)."
        )

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("num_inference_steps", default=self.num_inference_steps),
            InputParam.template("timesteps", note="overrides `num_inference_steps` when provided"),
            InputParam.template("sigmas", note="overrides `num_inference_steps` when provided"),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("timesteps", type_hint=torch.Tensor, description="The timesteps to use for inference"),
            OutputParam(
                "num_inference_steps",
                type_hint=int,
                description="The number of denoising steps to perform at inference time",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components: Flux3ModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        device = components._execution_device

        scheduler = components.scheduler

        num_inference_steps = block_state.num_inference_steps
        sigmas = block_state.sigmas
        timesteps = block_state.timesteps

        if timesteps is None and sigmas is None:
            sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)

        timesteps, num_inference_steps = retrieve_timesteps(
            scheduler,
            num_inference_steps,
            device,
            timesteps=timesteps,
            sigmas=sigmas,
        )
        block_state.timesteps = timesteps
        block_state.num_inference_steps = num_inference_steps

        components.scheduler.set_begin_index(0)

        self.set_block_state(state, block_state)
        return components, state


def _prepare_image_ids(height: int, width: int, image_rope_scale: float, l_coord: int = 0) -> torch.Tensor:
    """4D position ids `(t, h, w, l)` of shape `(height * width, 4)` for an image token grid: `t=0`, the H/W axes
    scaled by the FLUX.3 image RoPE scale, and a constant `l` (0 for the generated image, negative for references)."""
    t = torch.arange(1)
    h = torch.arange(height)
    w = torch.arange(width)
    l = torch.tensor([l_coord])

    ids = torch.cartesian_prod(t, h, w, l).to(dtype=torch.float32)
    ids[:, 1:3] = ids[:, 1:3] * image_rope_scale

    return ids


class Flux3PrepareLatentsStep(ModularPipelineBlocks):
    model_name = "flux3"

    def __init__(self, default_height: int = 1024, default_width: int = 1024, image_rope_scale: float = 0.5):
        # text-to-image size; image-to-image derives its size from the reference image upstream (its branch declares
        # no default, each branch of the auto blocks resolves its own)
        self.default_height = default_height
        self.default_width = default_width
        # the model was trained with the image H/W position ids scaled by this factor
        self.image_rope_scale = image_rope_scale
        super().__init__()

    @property
    def description(self) -> str:
        return "Prepare latents step that prepares the initial noise image latents as tokens, with their position ids."

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("height", default=self.default_height),
            InputParam.template("width", default=self.default_width),
            InputParam.template("latents"),
            InputParam.template("num_images_per_prompt"),
            InputParam.template("generator"),
            InputParam(
                "batch_size",
                required=True,
                type_hint=int,
                description="Number of prompts, the final batch size of model inputs should be `batch_size * num_images_per_prompt`.",
            ),
            InputParam("dtype", type_hint=torch.dtype, description="The dtype of the model inputs"),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "latents", type_hint=torch.Tensor, description="The initial latents to use for the denoising process"
            ),
            OutputParam(
                "img_ids",
                kwargs_type="denoiser_input_fields",
                type_hint=torch.Tensor,
                description="Position IDs (T, H, W, L) for the image latent tokens (for RoPE)",
            ),
        ]

    @staticmethod
    def check_inputs(height: int, width: int, image_scale_factor: int):
        if height % image_scale_factor != 0 or width % image_scale_factor != 0:
            raise ValueError(
                f"`height` and `width` have to be divisible by {image_scale_factor} but are {height} and {width}."
            )

    @torch.no_grad()
    def __call__(self, components: Flux3ModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        device = components._execution_device

        self.check_inputs(block_state.height, block_state.width, components.vae_scale_factor * 2)
        batch_size = block_state.batch_size * block_state.num_images_per_prompt

        # image tokens sit on a (height // 16, width // 16) grid with vae_latent_channels * 4 channels
        # (the VAE downsamples 8x and its latents are packed 2x2)
        latent_height = block_state.height // (components.vae_scale_factor * 2)
        latent_width = block_state.width // (components.vae_scale_factor * 2)

        latents = block_state.latents
        if latents is None:
            shape = (batch_size, components.num_channels_latents, latent_height, latent_width)
            if isinstance(block_state.generator, list) and len(block_state.generator) != batch_size:
                raise ValueError(
                    f"You have passed a list of generators of length {len(block_state.generator)}, but requested an"
                    f" effective batch size of {batch_size}. Make sure the batch size matches the length of the"
                    " generators."
                )
            # draw in float32 then cast: a reduced-precision draw consumes the RNG stream differently,
            # which would break seed-compatibility with the reference implementation
            latents = randn_tensor(shape, generator=block_state.generator, device=device, dtype=torch.float32)
            latents = latents.to(block_state.dtype)
        else:
            latents = latents.to(device=device, dtype=block_state.dtype)

        block_state.latents = latents.flatten(2).transpose(1, 2)  # pack to tokens: (B, seq_len, C)
        block_state.img_ids = _prepare_image_ids(latent_height, latent_width, self.image_rope_scale).to(device)

        self.set_block_state(state, block_state)
        return components, state


class Flux3PrepareImageCondLatentsStep(ModularPipelineBlocks):
    model_name = "flux3"

    def __init__(self, image_rope_scale: float = 0.5):
        self.image_rope_scale = image_rope_scale
        super().__init__()

    @property
    def description(self) -> str:
        return (
            "Step that packs the reference image latents into one token sequence with their position ids for FLUX.3 "
            "image-to-image generation. The references are told apart on the `l` position axis (-1, -2, ...)."
        )

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                "image_cond_latents",
                required=True,
                type_hint=list[torch.Tensor],
                description="Normalized packed latents of the reference images, one tensor of shape (B, C, height_i//16, width_i//16) per image",
            ),
            InputParam("dtype", type_hint=torch.dtype, description="The dtype of the model inputs"),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "image_cond_hidden_states",
                kwargs_type="denoiser_input_fields",
                type_hint=torch.Tensor,
                description="Reference image latent tokens, shape (batch_size, cond_seq_len, C)",
            ),
            OutputParam(
                "image_cond_ids",
                kwargs_type="denoiser_input_fields",
                type_hint=torch.Tensor,
                description="Position IDs (T, H, W, L) of the reference image tokens, shape (cond_seq_len, 4)",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components: Flux3ModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        device = components._execution_device

        cond_tokens, cond_ids = [], []
        for index, latents in enumerate(block_state.image_cond_latents):
            latent_height, latent_width = latents.shape[-2:]
            cond_tokens.append(latents.flatten(2).transpose(1, 2))  # pack to tokens: (B, seq_len, C)
            cond_ids.append(
                _prepare_image_ids(latent_height, latent_width, self.image_rope_scale, l_coord=-(1 + index))
            )

        block_state.image_cond_hidden_states = torch.cat(cond_tokens, dim=1).to(device=device, dtype=block_state.dtype)
        block_state.image_cond_ids = torch.cat(cond_ids, dim=0).to(device)

        self.set_block_state(state, block_state)
        return components, state


class Flux3TextRoPEInputsStep(ModularPipelineBlocks):
    model_name = "flux3"

    @property
    def description(self) -> str:
        return (
            "Step that prepares the 4D RoPE position IDs of the text tokens for FLUX.3 denoising. Place it after "
            "the text encoder step."
        )

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template(
                "prompt_embeds",
                description="Text embeddings used to guide the generation. Can be generated from text_encoder step.",
            ),
            InputParam.template(
                "negative_prompt_embeds",
                description="Negative text embeddings used to guide the generation. Can be generated from text_encoder step.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                name="txt_ids",
                kwargs_type="denoiser_input_fields",
                type_hint=torch.Tensor,
                description="4D position IDs (T, H, W, L) for text tokens, used for RoPE calculation.",
            ),
            OutputParam(
                name="negative_txt_ids",
                kwargs_type="denoiser_input_fields",
                type_hint=torch.Tensor,
                description="4D position IDs (T, H, W, L) for negative text tokens, used for RoPE calculation.",
            ),
        ]

    @staticmethod
    def _prepare_text_ids(x: torch.Tensor):
        """Prepare 4D position IDs (T, H, W, L) of shape (seq_len, 4) for text tokens; text advances along the L axis."""
        seq_len = x.shape[1]
        t = torch.arange(1)
        h = torch.arange(1)
        w = torch.arange(1)
        seq_l = torch.arange(seq_len)

        return torch.cartesian_prod(t, h, w, seq_l).to(dtype=torch.float32)

    def __call__(self, components: Flux3ModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        device = block_state.prompt_embeds.device
        block_state.txt_ids = self._prepare_text_ids(block_state.prompt_embeds).to(device)

        block_state.negative_txt_ids = None
        if block_state.negative_prompt_embeds is not None:
            block_state.negative_txt_ids = self._prepare_text_ids(block_state.negative_prompt_embeds).to(device)

        self.set_block_state(state, block_state)
        return components, state


class Flux3DistilledTextRoPEInputsStep(Flux3TextRoPEInputsStep):
    model_name = "flux3-distilled"

    @property
    def description(self) -> str:
        return (
            "Step that prepares the 4D RoPE position IDs of the text tokens for the step-distilled FLUX.3 model (no "
            "negative prompt). Place it after the text encoder step."
        )

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template(
                "prompt_embeds",
                description="Text embeddings used to guide the generation. Can be generated from text_encoder step.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                name="txt_ids",
                kwargs_type="denoiser_input_fields",
                type_hint=torch.Tensor,
                description="4D position IDs (T, H, W, L) for text tokens, used for RoPE calculation.",
            ),
        ]

    def __call__(self, components: Flux3ModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        block_state.txt_ids = self._prepare_text_ids(block_state.prompt_embeds).to(block_state.prompt_embeds.device)
        self.set_block_state(state, block_state)
        return components, state
