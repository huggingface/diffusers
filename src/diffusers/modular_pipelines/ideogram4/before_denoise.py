# Copyright 2026 Ideogram AI and The HuggingFace Team. All rights reserved.
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


import math

import torch

from ...models.transformers.transformer_ideogram4 import (
    IMAGE_POSITION_OFFSET,
    LLM_TOKEN_INDICATOR,
    OUTPUT_IMAGE_INDICATOR,
    SEQUENCE_PADDING_INDICATOR,
    Ideogram4Transformer2DModel,
)
from ...schedulers import FlowMatchEulerDiscreteScheduler
from ...utils import logging
from ...utils.torch_utils import randn_tensor
from ..modular_pipeline import ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam
from .modular_pipeline import Ideogram4ModularPipeline


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# Default per-step guidance schedule (length must equal `num_inference_steps`): 7.0 for the main steps,
# dropping to 3.0 for the final 3 "polish" steps.
DEFAULT_GUIDANCE_SCHEDULE = (7.0,) * 45 + (3.0,) * 3


# Copied from diffusers.pipelines.ideogram4.pipeline_ideogram4._logit_normal_sigmas
def _logit_normal_sigmas(
    num_inference_steps: int,
    mu: float,
    std: float = 1.0,
    logsnr_min: float = -15.0,
    logsnr_max: float = 18.0,
    device: torch.device | None = None,
) -> torch.Tensor:
    r"""
    Build a length-`num_inference_steps` sigma schedule using the Ideogram4 logit-normal flow-matching schedule.

    Sigmas are returned in `[0, 1]` in decreasing order (sigma close to 1 corresponds to pure noise, sigma close to 0
    to clean data), matching diffusers conventions.

    The Ideogram4 schedule applies `sigma(s) = 1 - logit_normal_cdf_inverse(1 - s)` to `s = linspace(0, 1, N + 1)` and
    keeps the first `N` entries; a terminal zero is appended downstream by the scheduler.
    """
    intervals = torch.linspace(0.0, 1.0, num_inference_steps + 1, dtype=torch.float64)
    # Apply the inverse CDF of a normal then push through the logistic to obtain a logit-normal CDF inverse.
    z = torch.special.ndtri(intervals)
    y = mu + std * z
    t = 1.0 - torch.special.expit(y)
    t_min = 1.0 / (1.0 + math.exp(0.5 * logsnr_max))
    t_max = 1.0 / (1.0 + math.exp(0.5 * logsnr_min))
    t = t.clamp(t_min, t_max)
    # Convert from model time (0 = noise, 1 = data) to diffusers sigma (1 = noise, 0 = data) and reverse.
    sigmas = (1.0 - t).flip(0)
    # Drop the trailing 0; FlowMatchEulerDiscreteScheduler.set_timesteps appends one back internally.
    sigmas = sigmas[:-1].to(dtype=torch.float32, device=device)
    return sigmas


# Copied from diffusers.pipelines.ideogram4.pipeline_ideogram4._resolution_aware_mu
def _resolution_aware_mu(
    height: int,
    width: int,
    base_mu: float,
    base_resolution: tuple[int, int] = (512, 512),
) -> float:
    """Shift the schedule mean as a function of image resolution."""
    num_pixels = height * width
    base_pixels = base_resolution[0] * base_resolution[1]
    return base_mu + 0.5 * math.log(num_pixels / base_pixels)


# Copied from diffusers.pipelines.ideogram4.pipeline_ideogram4._expand_tensor_to_effective_batch
def _expand_tensor_to_effective_batch(
    tensor: torch.Tensor,
    batch_size: int,
    num_per_prompt: int,
    tensor_name: str | None = None,
) -> torch.Tensor:
    """Replicate `tensor` along dim 0 from `batch_size` (or 1) to `batch_size * num_per_prompt`."""
    target_batch_size = batch_size * num_per_prompt

    if tensor.shape[0] == target_batch_size:
        return tensor

    if tensor.shape[0] == 1:
        repeat_by = target_batch_size
    elif tensor.shape[0] == batch_size:
        repeat_by = num_per_prompt
    else:
        tensor_name = f"`{tensor_name}`" if tensor_name is not None else "Tensor"
        raise ValueError(
            f"{tensor_name} batch size must be 1, `batch_size` ({batch_size}), or "
            f"`batch_size * num_*_per_prompt` ({target_batch_size}), but got {tensor.shape[0]}."
        )

    return torch.repeat_interleave(tensor, repeats=repeat_by, dim=0, output_size=tensor.shape[0] * repeat_by)


# auto_docstring
class Ideogram4TextInputsStep(ModularPipelineBlocks):
    """
    Input step that determines `batch_size`/`dtype` from the per-prompt `text_features` and replicates the text outputs
    to `batch_size * num_images_per_prompt`. Place after the text encoder.

      Inputs:
          num_images_per_prompt (`int`, *optional*, defaults to 1):
              The number of images to generate per prompt.
          text_features (`Tensor`):
              Per-prompt text features from the encoder.
          text_lengths (`list`):
              Per-prompt text-token counts from the encoder.

      Outputs:
          batch_size (`int`):
              Effective batch size (num prompts * num_images_per_prompt).
          dtype (`dtype`):
              The dtype of the text features.
          text_features (`Tensor`):
              Text features, batch-expanded.
          text_lengths (`list`):
              Text-token counts, batch-expanded.
    """

    model_name = "ideogram4"

    @property
    def description(self) -> str:
        return (
            "Input step that determines `batch_size`/`dtype` from the per-prompt `text_features` and replicates the "
            "text outputs to `batch_size * num_images_per_prompt`. Place after the text encoder."
        )

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("num_images_per_prompt", default=1),
            InputParam(
                name="text_features",
                required=True,
                type_hint=torch.Tensor,
                description="Per-prompt text features from the encoder.",
            ),
            InputParam(
                name="text_lengths",
                required=True,
                type_hint=list,
                description="Per-prompt text-token counts from the encoder.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                name="batch_size",
                type_hint=int,
                description="Effective batch size (num prompts * num_images_per_prompt).",
            ),
            OutputParam(name="dtype", type_hint=torch.dtype, description="The dtype of the text features."),
            OutputParam(name="text_features", type_hint=torch.Tensor, description="Text features, batch-expanded."),
            OutputParam(name="text_lengths", type_hint=list, description="Text-token counts, batch-expanded."),
        ]

    @torch.no_grad()
    def __call__(self, components: Ideogram4ModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        prompt_batch = block_state.text_features.shape[0]
        num_per_prompt = block_state.num_images_per_prompt

        block_state.dtype = block_state.text_features.dtype
        block_state.text_features = _expand_tensor_to_effective_batch(
            block_state.text_features, prompt_batch, num_per_prompt, "text_features"
        )
        block_state.text_lengths = [n for n in block_state.text_lengths for _ in range(num_per_prompt)]
        block_state.batch_size = prompt_batch * num_per_prompt

        self.set_block_state(state, block_state)
        return components, state


# auto_docstring
class Ideogram4PrepareLatentsStep(ModularPipelineBlocks):
    """
    Step that prepares the packed image latents (B, num_image_tokens, latent_dim) for the denoising loop.

      Components:
          transformer (`Ideogram4Transformer2DModel`)

      Inputs:
          latents (`Tensor`, *optional*):
              Pre-generated noisy latents for image generation.
          height (`int`):
              The height in pixels of the generated image.
          width (`int`):
              The width in pixels of the generated image.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.
          batch_size (`int`):
              Effective batch size.

      Outputs:
          latents (`Tensor`):
              The initial packed image latents (B, num_image_tokens, latent_dim).
          num_image_tokens (`int`):
              Number of image tokens (grid_h * grid_w).
    """

    model_name = "ideogram4"

    @property
    def description(self) -> str:
        return "Step that prepares the packed image latents (B, num_image_tokens, latent_dim) for the denoising loop."

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("transformer", Ideogram4Transformer2DModel)]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("latents"),
            InputParam.template("height", required=True),
            InputParam.template("width", required=True),
            InputParam.template("generator"),
            InputParam(name="batch_size", required=True, type_hint=int, description="Effective batch size."),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                name="latents",
                type_hint=torch.Tensor,
                description="The initial packed image latents (B, num_image_tokens, latent_dim).",
            ),
            OutputParam(
                name="num_image_tokens", type_hint=int, description="Number of image tokens (grid_h * grid_w)."
            ),
        ]

    @torch.no_grad()
    def __call__(self, components: Ideogram4ModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        device = components._execution_device
        patch = components.patch_size
        grid_h = block_state.height // (components.vae_scale_factor * patch)
        grid_w = block_state.width // (components.vae_scale_factor * patch)
        num_image_tokens = grid_h * grid_w
        latent_dim = components.transformer.config.in_channels

        shape = (block_state.batch_size, num_image_tokens, latent_dim)
        if block_state.latents is None:
            block_state.latents = randn_tensor(
                shape, generator=block_state.generator, device=device, dtype=torch.float32
            )
        else:
            block_state.latents = block_state.latents.to(device=device, dtype=torch.float32)

        block_state.num_image_tokens = num_image_tokens

        self.set_block_state(state, block_state)
        return components, state


# auto_docstring
class Ideogram4SetTimestepsStep(ModularPipelineBlocks):
    """
    Step that sets the resolution-aware logit-normal sigma schedule on the scheduler and resolves the per-step guidance
    weights.

      Components:
          scheduler (`FlowMatchEulerDiscreteScheduler`)

      Inputs:
          num_inference_steps (`int`, *optional*, defaults to 48):
              The number of denoising steps.
          height (`int`):
              The height in pixels of the generated image.
          width (`int`):
              The width in pixels of the generated image.
          mu (`float`, *optional*, defaults to 0.0):
              Base mean of the logit-normal schedule.
          std (`float`, *optional*, defaults to 1.5):
              Std of the logit-normal schedule.
          guidance_schedule (`list`, *optional*, defaults to (7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0,
          7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0,
          7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 3.0, 3.0, 3.0)):
              Per-step guidance scale schedule (length num_inference_steps).

      Outputs:
          timesteps (`Tensor`):
              The denoising timesteps.
          gw (`Tensor`):
              Per-step guidance weights (num_inference_steps,).
    """

    model_name = "ideogram4"

    @property
    def description(self) -> str:
        return (
            "Step that sets the resolution-aware logit-normal sigma schedule on the scheduler and resolves the "
            "per-step guidance weights."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("scheduler", FlowMatchEulerDiscreteScheduler)]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("num_inference_steps", default=48),
            InputParam.template("height", required=True),
            InputParam.template("width", required=True),
            InputParam(name="mu", default=0.0, type_hint=float, description="Base mean of the logit-normal schedule."),
            InputParam(name="std", default=1.5, type_hint=float, description="Std of the logit-normal schedule."),
            InputParam(
                name="guidance_schedule",
                default=DEFAULT_GUIDANCE_SCHEDULE,
                type_hint=list,
                description="Per-step guidance scale schedule (length num_inference_steps).",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(name="timesteps", type_hint=torch.Tensor, description="The denoising timesteps."),
            OutputParam(
                name="gw", type_hint=torch.Tensor, description="Per-step guidance weights (num_inference_steps,)."
            ),
        ]

    @torch.no_grad()
    def __call__(self, components: Ideogram4ModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        device = components._execution_device
        if len(block_state.guidance_schedule) != block_state.num_inference_steps:
            raise ValueError(
                f"`guidance_schedule` must have length `num_inference_steps` ({block_state.num_inference_steps}), "
                f"got {len(block_state.guidance_schedule)}."
            )

        schedule_mu = _resolution_aware_mu(height=block_state.height, width=block_state.width, base_mu=block_state.mu)
        sigmas = _logit_normal_sigmas(block_state.num_inference_steps, schedule_mu, std=block_state.std, device=device)
        components.scheduler.set_timesteps(sigmas=sigmas.tolist(), device=device)

        block_state.timesteps = components.scheduler.timesteps
        block_state.gw = torch.as_tensor(block_state.guidance_schedule, dtype=torch.float32, device=device)

        self.set_block_state(state, block_state)
        return components, state


# auto_docstring
class Ideogram4PrepareAdditionalInputsStep(ModularPipelineBlocks):
    """
    Step that prepares the additional denoiser inputs from the packed-sequence layout: the conditional
    encoder_hidden_states (text features packed with image padding) and the position_ids/segment_ids/indicator, plus
    the unconditional (image-only) counterparts. Place after prepare_latents.

      Inputs:
          height (`int`):
              The height in pixels of the generated image.
          width (`int`):
              The width in pixels of the generated image.
          text_features (`Tensor`):
              Batch-expanded text features.
          text_lengths (`list`):
              Batch-expanded text-token counts.
          batch_size (`int`):
              Effective batch size.

      Outputs:
          prompt_embeds (`Tensor`):
              Packed conditional encoder_hidden_states (B, total_seq, dim).
          position_ids (`Tensor`):
              Conditional 3-axis MRoPE position ids.
          segment_ids (`Tensor`):
              Conditional block-diagonal segment ids.
          indicator (`Tensor`):
              Conditional per-token text/image/pad role.
          negative_prompt_embeds (`Tensor`):
              Unconditional (zeroed) text features (B, num_image_tokens, dim).
          negative_position_ids (`Tensor`):
              Unconditional position ids (image region).
          negative_segment_ids (`Tensor`):
              Unconditional segment ids (image region).
          negative_indicator (`Tensor`):
              Unconditional indicator (image region).
    """

    model_name = "ideogram4"

    @property
    def description(self) -> str:
        return (
            "Step that prepares the additional denoiser inputs from the packed-sequence layout: the conditional "
            "encoder_hidden_states (text features packed with image padding) and the position_ids/segment_ids/"
            "indicator, plus the unconditional (image-only) counterparts. Place after prepare_latents."
        )

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("height", required=True),
            InputParam.template("width", required=True),
            InputParam(
                name="text_features",
                required=True,
                type_hint=torch.Tensor,
                description="Batch-expanded text features.",
            ),
            InputParam(
                name="text_lengths", required=True, type_hint=list, description="Batch-expanded text-token counts."
            ),
            InputParam(name="batch_size", required=True, type_hint=int, description="Effective batch size."),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                name="prompt_embeds",
                type_hint=torch.Tensor,
                description="Packed conditional encoder_hidden_states (B, total_seq, dim).",
            ),
            OutputParam(
                name="position_ids", type_hint=torch.Tensor, description="Conditional 3-axis MRoPE position ids."
            ),
            OutputParam(
                name="segment_ids", type_hint=torch.Tensor, description="Conditional block-diagonal segment ids."
            ),
            OutputParam(
                name="indicator", type_hint=torch.Tensor, description="Conditional per-token text/image/pad role."
            ),
            OutputParam(
                name="negative_prompt_embeds",
                type_hint=torch.Tensor,
                description="Unconditional (zeroed) text features (B, num_image_tokens, dim).",
            ),
            OutputParam(
                name="negative_position_ids",
                type_hint=torch.Tensor,
                description="Unconditional position ids (image region).",
            ),
            OutputParam(
                name="negative_segment_ids",
                type_hint=torch.Tensor,
                description="Unconditional segment ids (image region).",
            ),
            OutputParam(
                name="negative_indicator",
                type_hint=torch.Tensor,
                description="Unconditional indicator (image region).",
            ),
        ]

    @staticmethod
    # Copied from diffusers.pipelines.ideogram4.pipeline_ideogram4.Ideogram4Pipeline._prepare_ids
    def _prepare_ids(
        text_lengths: list[int],
        grid_h: int,
        grid_w: int,
        max_text_tokens: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build the packed `[left-pad][text][image]` layout from the per-prompt text lengths and the image grid.

        Returns `position_ids` (3-axis MRoPE), `segment_ids` (block-diagonal attention) and `indicator` (per-token
        text/image/pad role).
        """
        batch_size = len(text_lengths)
        num_image_tokens = grid_h * grid_w
        total_seq_len = max_text_tokens + num_image_tokens

        # Image position ids (t=0, h, w); offset keeps them disjoint from text positions.
        h_idx = torch.arange(grid_h).view(-1, 1).expand(grid_h, grid_w).reshape(-1)
        w_idx = torch.arange(grid_w).view(1, -1).expand(grid_h, grid_w).reshape(-1)
        t_idx = torch.zeros_like(h_idx)
        image_pos = torch.stack([t_idx, h_idx, w_idx], dim=1) + IMAGE_POSITION_OFFSET

        position_ids = torch.zeros(batch_size, total_seq_len, 3, dtype=torch.long)
        segment_ids = torch.full((batch_size, total_seq_len), SEQUENCE_PADDING_INDICATOR, dtype=torch.long)
        indicator = torch.zeros(batch_size, total_seq_len, dtype=torch.long)

        for b, num_text in enumerate(text_lengths):
            offset = max_text_tokens - num_text

            text_pos = torch.arange(num_text)
            text_pos_3d = torch.stack([text_pos, text_pos, text_pos], dim=1)
            position_ids[b, offset : offset + num_text] = text_pos_3d
            position_ids[b, offset + num_text :] = image_pos

            indicator[b, offset : offset + num_text] = LLM_TOKEN_INDICATOR
            indicator[b, offset + num_text :] = OUTPUT_IMAGE_INDICATOR

            segment_ids[b, offset : offset + num_text + num_image_tokens] = 1

        return position_ids.to(device), segment_ids.to(device), indicator.to(device)

    @torch.no_grad()
    def __call__(self, components: Ideogram4ModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        device = components._execution_device
        patch = components.patch_size
        grid_h = block_state.height // (components.vae_scale_factor * patch)
        grid_w = block_state.width // (components.vae_scale_factor * patch)
        num_image_tokens = grid_h * grid_w

        text_features = block_state.text_features
        max_text_tokens = text_features.shape[1]
        feature_dim = text_features.shape[-1]

        position_ids, segment_ids, indicator = self._prepare_ids(
            block_state.text_lengths, grid_h, grid_w, max_text_tokens, device
        )

        # Pack the text features into the full sequence; image positions carry no text features.
        image_feature_padding = torch.zeros(
            block_state.batch_size, num_image_tokens, feature_dim, dtype=text_features.dtype, device=device
        )
        block_state.prompt_embeds = torch.cat([text_features, image_feature_padding], dim=1)

        # Unconditional (image-only) branch, derived from the conditioning.
        block_state.negative_prompt_embeds = torch.zeros(
            block_state.batch_size, num_image_tokens, feature_dim, dtype=text_features.dtype, device=device
        )
        block_state.position_ids = position_ids
        block_state.segment_ids = segment_ids
        block_state.indicator = indicator
        block_state.negative_position_ids = position_ids[:, max_text_tokens:]
        block_state.negative_segment_ids = segment_ids[:, max_text_tokens:]
        block_state.negative_indicator = indicator[:, max_text_tokens:]

        self.set_block_state(state, block_state)
        return components, state
