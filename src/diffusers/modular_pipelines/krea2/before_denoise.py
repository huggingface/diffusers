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


import numpy as np
import torch

from ...models.transformers.transformer_krea2 import Krea2Transformer2DModel
from ...schedulers import FlowMatchEulerDiscreteScheduler
from ...utils import logging
from ...utils.torch_utils import randn_tensor
from ..modular_pipeline import ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import ComponentSpec, ConfigSpec, InputParam, OutputParam
from .modular_pipeline import Krea2ModularPipeline


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# Copied from diffusers.pipelines.krea2.pipeline_krea2.calculate_shift
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


# auto_docstring
class Krea2TextInputsStep(ModularPipelineBlocks):
    """
    Input step that determines `batch_size`/`dtype` from the per-prompt `prompt_embeds` and replicates the text
    conditioning (and the optional negative branch) to `batch_size * num_images_per_prompt`. Place after the text
    encoder.

      Inputs:
          num_images_per_prompt (`int`, *optional*, defaults to 1):
              The number of images to generate per prompt.
          prompt_embeds (`Tensor`):
              Per-prompt stacked text features (B, text_seq_len, num_text_layers, text_hidden_dim).
          prompt_embeds_mask (`Tensor`):
              Per-prompt boolean text mask (B, text_seq_len).
          negative_prompt_embeds (`Tensor`, *optional*):
              Per-prompt negative text features.
          negative_prompt_embeds_mask (`Tensor`, *optional*):
              Per-prompt negative text mask.

      Outputs:
          batch_size (`int`):
              Effective batch size (num prompts * num_images_per_prompt).
          dtype (`dtype`):
              The dtype of the text features.
          prompt_embeds (`Tensor`):
              Text features, batch-expanded.
          prompt_embeds_mask (`Tensor`):
              Text mask, batch-expanded.
          negative_prompt_embeds (`Tensor`):
              Negative text features, batch-expanded.
          negative_prompt_embeds_mask (`Tensor`):
              Negative text mask, batch-expanded.
    """

    model_name = "krea2"

    @property
    def description(self) -> str:
        return (
            "Input step that determines `batch_size`/`dtype` from the per-prompt `prompt_embeds` and replicates the "
            "text conditioning (and the optional negative branch) to `batch_size * num_images_per_prompt`. Place after "
            "the text encoder."
        )

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("num_images_per_prompt", default=1),
            InputParam(
                name="prompt_embeds",
                required=True,
                type_hint=torch.Tensor,
                description="Per-prompt stacked text features (B, text_seq_len, num_text_layers, text_hidden_dim).",
            ),
            InputParam(
                name="prompt_embeds_mask",
                required=True,
                type_hint=torch.Tensor,
                description="Per-prompt boolean text mask (B, text_seq_len).",
            ),
            InputParam(
                name="negative_prompt_embeds",
                type_hint=torch.Tensor,
                description="Per-prompt negative text features.",
            ),
            InputParam(
                name="negative_prompt_embeds_mask",
                type_hint=torch.Tensor,
                description="Per-prompt negative text mask.",
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
            OutputParam(name="prompt_embeds", type_hint=torch.Tensor, description="Text features, batch-expanded."),
            OutputParam(name="prompt_embeds_mask", type_hint=torch.Tensor, description="Text mask, batch-expanded."),
            OutputParam(
                name="negative_prompt_embeds",
                type_hint=torch.Tensor,
                description="Negative text features, batch-expanded.",
            ),
            OutputParam(
                name="negative_prompt_embeds_mask",
                type_hint=torch.Tensor,
                description="Negative text mask, batch-expanded.",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components: Krea2ModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        prompt_batch, seq_len, num_layers, dim = block_state.prompt_embeds.shape
        n = block_state.num_images_per_prompt

        block_state.dtype = block_state.prompt_embeds.dtype
        block_state.batch_size = prompt_batch * n

        block_state.prompt_embeds = block_state.prompt_embeds.repeat(1, n, 1, 1).view(
            prompt_batch * n, seq_len, num_layers, dim
        )
        block_state.prompt_embeds_mask = block_state.prompt_embeds_mask.repeat(1, n).view(prompt_batch * n, seq_len)

        if block_state.negative_prompt_embeds is not None:
            block_state.negative_prompt_embeds = block_state.negative_prompt_embeds.repeat(1, n, 1, 1).view(
                prompt_batch * n, seq_len, num_layers, dim
            )
            block_state.negative_prompt_embeds_mask = block_state.negative_prompt_embeds_mask.repeat(1, n).view(
                prompt_batch * n, seq_len
            )

        self.set_block_state(state, block_state)
        return components, state


# auto_docstring
class Krea2PrepareLatentsStep(ModularPipelineBlocks):
    """
    Step that samples the spatial image latents and patch-packs them into (B, image_seq_len, in_channels) for the
    denoising loop.

      Components:
          transformer (`Krea2Transformer2DModel`)

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
          dtype (`dtype`):
              The working dtype.

      Outputs:
          latents (`Tensor`):
              The initial packed image latents (B, image_seq_len, in_channels).
          image_seq_len (`int`):
              Number of image tokens (grid_h * grid_w).
    """

    model_name = "krea2"

    @property
    def description(self) -> str:
        return (
            "Step that samples the spatial image latents and patch-packs them into (B, image_seq_len, in_channels) "
            "for the denoising loop."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("transformer", Krea2Transformer2DModel)]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("latents"),
            InputParam.template("height", required=True),
            InputParam.template("width", required=True),
            InputParam.template("generator"),
            InputParam(name="batch_size", required=True, type_hint=int, description="Effective batch size."),
            InputParam(name="dtype", required=True, type_hint=torch.dtype, description="The working dtype."),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                name="latents",
                type_hint=torch.Tensor,
                description="The initial packed image latents (B, image_seq_len, in_channels).",
            ),
            OutputParam(name="image_seq_len", type_hint=int, description="Number of image tokens (grid_h * grid_w)."),
        ]

    @torch.no_grad()
    def __call__(self, components: Krea2ModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        device = components._execution_device
        p = components.patch_size
        num_channels_latents = components.transformer.config.in_channels // (p**2)
        latent_height = block_state.height // components.vae_scale_factor
        latent_width = block_state.width // components.vae_scale_factor

        if block_state.latents is not None:
            block_state.latents = block_state.latents.to(device=device, dtype=block_state.dtype)
        else:
            latents = randn_tensor(
                (block_state.batch_size, num_channels_latents, latent_height, latent_width),
                generator=block_state.generator,
                device=device,
                dtype=block_state.dtype,
            )
            latents = latents.view(
                block_state.batch_size, num_channels_latents, latent_height // p, p, latent_width // p, p
            )
            latents = latents.permute(0, 2, 4, 1, 3, 5)
            block_state.latents = latents.reshape(
                block_state.batch_size, (latent_height // p) * (latent_width // p), num_channels_latents * p * p
            )

        block_state.image_seq_len = block_state.latents.shape[1]

        self.set_block_state(state, block_state)
        return components, state


# auto_docstring
class Krea2SetTimestepsStep(ModularPipelineBlocks):
    """
    Step that sets the Krea 2 flow-matching schedule on the scheduler: a linear sigma schedule with a resolution-aware
    (or fixed, for distilled checkpoints) dynamic time shift `mu`.

      Components:
          scheduler (`FlowMatchEulerDiscreteScheduler`)

      Configs:
          is_distilled (default: False)

      Inputs:
          num_inference_steps (`int`, *optional*, defaults to 28):
              The number of denoising steps.
          sigmas (`list`, *optional*):
              Custom sigma schedule (defaults to a linear ramp).
          image_seq_len (`int`):
              Number of image tokens, used to compute the resolution-aware shift.

      Outputs:
          timesteps (`Tensor`):
              The denoising timesteps.
    """

    model_name = "krea2"

    @property
    def description(self) -> str:
        return (
            "Step that sets the Krea 2 flow-matching schedule on the scheduler: a linear sigma schedule with a "
            "resolution-aware (or fixed, for distilled checkpoints) dynamic time shift `mu`."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("scheduler", FlowMatchEulerDiscreteScheduler)]

    @property
    def expected_configs(self) -> list[ConfigSpec]:
        return [ConfigSpec(name="is_distilled", default=False)]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("num_inference_steps", default=28),
            InputParam(
                name="sigmas", type_hint=list, description="Custom sigma schedule (defaults to a linear ramp)."
            ),
            InputParam(
                name="image_seq_len",
                required=True,
                type_hint=int,
                description="Number of image tokens, used to compute the resolution-aware shift.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [OutputParam(name="timesteps", type_hint=torch.Tensor, description="The denoising timesteps.")]

    @torch.no_grad()
    def __call__(self, components: Krea2ModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        device = components._execution_device
        num_inference_steps = block_state.num_inference_steps

        sigmas = block_state.sigmas
        if sigmas is None:
            sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)

        if components.config.is_distilled:
            mu = 1.15
        else:
            config = components.scheduler.config
            mu = calculate_shift(
                block_state.image_seq_len,
                config.get("base_image_seq_len", 256),
                config.get("max_image_seq_len", 6400),
                config.get("base_shift", 0.5),
                config.get("max_shift", 1.15),
            )

        components.scheduler.set_timesteps(sigmas=sigmas, mu=mu, device=device)
        components.scheduler.set_begin_index(0)
        block_state.timesteps = components.scheduler.timesteps

        self.set_block_state(state, block_state)
        return components, state


# auto_docstring
class Krea2PreparePositionIdsStep(ModularPipelineBlocks):
    """
    Step that builds the shared rotary position ids for the combined [text | image] sequence: text at the origin, image
    tokens at their (0, h, w) latent-grid coordinates. Place after prepare_latents.

      Inputs:
          height (`int`):
              The height in pixels of the generated image.
          width (`int`):
              The width in pixels of the generated image.
          prompt_embeds (`Tensor`):
              Batch-expanded text features (only text_seq_len is used).

      Outputs:
          position_ids (`Tensor`):
              Shared rotary coordinates (text_seq_len + grid_h * grid_w, 3).
    """

    model_name = "krea2"

    @property
    def description(self) -> str:
        return (
            "Step that builds the shared rotary position ids for the combined [text | image] sequence: text at the "
            "origin, image tokens at their (0, h, w) latent-grid coordinates. Place after prepare_latents."
        )

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("height", required=True),
            InputParam.template("width", required=True),
            InputParam(
                name="prompt_embeds",
                required=True,
                type_hint=torch.Tensor,
                description="Batch-expanded text features (only text_seq_len is used).",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                name="position_ids",
                type_hint=torch.Tensor,
                description="Shared rotary coordinates (text_seq_len + grid_h * grid_w, 3).",
            )
        ]

    @staticmethod
    # Copied from diffusers.pipelines.krea2.pipeline_krea2.Krea2Pipeline.prepare_position_ids
    def prepare_position_ids(text_seq_len: int, grid_height: int, grid_width: int, device: torch.device):
        """Build the `(text_seq_len + grid_height * grid_width, 3)` rotary coordinates for the combined sequence:
        text tokens sit at the origin, image tokens carry their `(0, h, w)` latent-grid coordinates."""
        text_ids = torch.zeros(text_seq_len, 3, device=device)
        image_ids = torch.zeros(grid_height, grid_width, 3, device=device)
        image_ids[..., 1] = torch.arange(grid_height, device=device)[:, None]
        image_ids[..., 2] = torch.arange(grid_width, device=device)[None, :]
        image_ids = image_ids.reshape(grid_height * grid_width, 3)
        return torch.cat([text_ids, image_ids], dim=0)

    @torch.no_grad()
    def __call__(self, components: Krea2ModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        device = components._execution_device
        p = components.patch_size
        grid_h = block_state.height // (components.vae_scale_factor * p)
        grid_w = block_state.width // (components.vae_scale_factor * p)
        text_seq_len = block_state.prompt_embeds.shape[1]

        block_state.position_ids = self.prepare_position_ids(text_seq_len, grid_h, grid_w, device)

        self.set_block_state(state, block_state)
        return components, state
