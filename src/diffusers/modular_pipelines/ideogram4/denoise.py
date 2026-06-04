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


import torch

from ...models.transformers.transformer_ideogram4 import Ideogram4Transformer2DModel
from ...schedulers import FlowMatchEulerDiscreteScheduler
from ...utils import logging
from ..modular_pipeline import (
    BlockState,
    LoopSequentialPipelineBlocks,
    ModularPipelineBlocks,
    PipelineState,
)
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam
from .modular_pipeline import Ideogram4ModularPipeline


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class Ideogram4LoopBeforeDenoiser(ModularPipelineBlocks):
    model_name = "ideogram4"

    @property
    def description(self) -> str:
        return (
            "Within the denoising loop: build the conditional packed input `[text-padding][image latents]` and the "
            "model timestep. Compose into the `sub_blocks` of `Ideogram4DenoiseLoopWrapper`."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("scheduler", FlowMatchEulerDiscreteScheduler)]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(name="latents", required=True, type_hint=torch.Tensor, description="Packed image latents."),
            InputParam(
                name="position_ids", required=True, type_hint=torch.Tensor, description="Conditional position ids."
            ),
            InputParam(name="batch_size", required=True, type_hint=int, description="Effective batch size."),
        ]

    @torch.no_grad()
    def __call__(self, components: Ideogram4ModularPipeline, block_state: BlockState, i: int, t: torch.Tensor):
        # Conditional packed sequence is [text-padding][image latents]; text region length = total - image tokens.
        max_text_tokens = block_state.position_ids.shape[1] - block_state.latents.shape[1]
        text_z_padding = torch.zeros(
            block_state.latents.shape[0],
            max_text_tokens,
            block_state.latents.shape[-1],
            dtype=block_state.latents.dtype,
            device=block_state.latents.device,
        )
        block_state.pos_z = torch.cat([text_z_padding, block_state.latents], dim=1)
        block_state.max_text_tokens = max_text_tokens

        # Map sigma-domain timestep to model time t in [0, 1] (0 = noise, 1 = clean data).
        num_train_timesteps = components.scheduler.config.num_train_timesteps
        t_model = 1.0 - (t.float() / num_train_timesteps)
        block_state.t_model = t_model.expand(block_state.batch_size)
        return components, block_state


class Ideogram4LoopDenoiser(ModularPipelineBlocks):
    model_name = "ideogram4"

    @property
    def description(self) -> str:
        return (
            "Within the denoising loop: run the conditional `transformer` on the full packed sequence and the "
            "`unconditional_transformer` on the image-only sequence, then blend with the per-step guidance weight "
            "(asymmetric CFG, no guider). Compose into `Ideogram4DenoiseLoopWrapper`."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("transformer", Ideogram4Transformer2DModel),
            ComponentSpec("unconditional_transformer", Ideogram4Transformer2DModel),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                name="prompt_embeds",
                required=True,
                type_hint=torch.Tensor,
                description="Packed conditional encoder_hidden_states.",
            ),
            InputParam(
                name="position_ids",
                required=True,
                type_hint=torch.Tensor,
                description="Conditional 3-axis MRoPE position ids.",
            ),
            InputParam(
                name="segment_ids",
                required=True,
                type_hint=torch.Tensor,
                description="Conditional block-diagonal segment ids.",
            ),
            InputParam(
                name="indicator",
                required=True,
                type_hint=torch.Tensor,
                description="Conditional per-token text/image/pad role.",
            ),
            InputParam(
                name="negative_prompt_embeds",
                required=True,
                type_hint=torch.Tensor,
                description="Unconditional (zeroed) text features.",
            ),
            InputParam(
                name="negative_position_ids",
                required=True,
                type_hint=torch.Tensor,
                description="Unconditional position ids (image region).",
            ),
            InputParam(
                name="negative_segment_ids",
                required=True,
                type_hint=torch.Tensor,
                description="Unconditional segment ids (image region).",
            ),
            InputParam(
                name="negative_indicator",
                required=True,
                type_hint=torch.Tensor,
                description="Unconditional indicator (image region).",
            ),
            InputParam(name="gw", required=True, type_hint=torch.Tensor, description="Per-step guidance weights."),
            InputParam(name="latents", required=True, type_hint=torch.Tensor, description="Packed image latents."),
        ]

    @torch.no_grad()
    def __call__(self, components: Ideogram4ModularPipeline, block_state: BlockState, i: int, t: torch.Tensor):
        transformer = components.transformer
        unconditional_transformer = components.unconditional_transformer

        # Conditional pass operates on the full packed sequence; the velocity is the image-token region.
        pos_out = transformer(
            hidden_states=block_state.pos_z.to(transformer.dtype),
            timestep=block_state.t_model.to(transformer.dtype),
            encoder_hidden_states=block_state.prompt_embeds.to(transformer.dtype),
            position_ids=block_state.position_ids,
            segment_ids=block_state.segment_ids,
            indicator=block_state.indicator,
            return_dict=False,
        )[0]
        pos_v = pos_out[:, block_state.max_text_tokens :].to(torch.float32)

        # Unconditional pass uses the image-only positions with zeroed text features.
        neg_v = unconditional_transformer(
            hidden_states=block_state.latents.to(unconditional_transformer.dtype),
            timestep=block_state.t_model.to(unconditional_transformer.dtype),
            encoder_hidden_states=block_state.negative_prompt_embeds.to(unconditional_transformer.dtype),
            position_ids=block_state.negative_position_ids,
            segment_ids=block_state.negative_segment_ids,
            indicator=block_state.negative_indicator,
            return_dict=False,
        )[0].to(torch.float32)

        gw_i = block_state.gw[i]
        v = gw_i * pos_v + (1.0 - gw_i) * neg_v
        # The scheduler integrates `-v` (Ideogram predicts velocity v = x0 - noise).
        block_state.noise_pred = -v
        return components, block_state


class Ideogram4LoopAfterDenoiser(ModularPipelineBlocks):
    model_name = "ideogram4"

    @property
    def description(self) -> str:
        return "Within the denoising loop: scheduler step. Compose into `Ideogram4DenoiseLoopWrapper`."

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("scheduler", FlowMatchEulerDiscreteScheduler)]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [OutputParam(name="latents", type_hint=torch.Tensor, description="The denoised latents.")]

    @torch.no_grad()
    def __call__(self, components: Ideogram4ModularPipeline, block_state: BlockState, i: int, t: torch.Tensor):
        block_state.latents = components.scheduler.step(
            block_state.noise_pred, t, block_state.latents, return_dict=False
        )[0]
        return components, block_state


# auto_docstring
class Ideogram4DenoiseStep(LoopSequentialPipelineBlocks):
    """
    Denoising loop that iteratively denoises the packed image latents over `timesteps`, running both the conditional
    and unconditional transformers and blending with the per-step guidance schedule.

      Components:
          scheduler (`FlowMatchEulerDiscreteScheduler`) transformer (`Ideogram4Transformer2DModel`)
          unconditional_transformer (`Ideogram4Transformer2DModel`)

      Inputs:
          timesteps (`Tensor`):
              Denoising timesteps from set_timesteps.
          num_inference_steps (`int`, *optional*, defaults to 48):
              The number of denoising steps.
          latents (`Tensor`):
              Packed image latents.
          position_ids (`Tensor`):
              Conditional position ids.
          batch_size (`int`):
              Effective batch size.
          prompt_embeds (`Tensor`):
              Packed conditional encoder_hidden_states.
          position_ids (`Tensor`):
              Conditional 3-axis MRoPE position ids.
          segment_ids (`Tensor`):
              Conditional block-diagonal segment ids.
          indicator (`Tensor`):
              Conditional per-token text/image/pad role.
          negative_prompt_embeds (`Tensor`):
              Unconditional (zeroed) text features.
          negative_position_ids (`Tensor`):
              Unconditional position ids (image region).
          negative_segment_ids (`Tensor`):
              Unconditional segment ids (image region).
          negative_indicator (`Tensor`):
              Unconditional indicator (image region).
          gw (`Tensor`):
              Per-step guidance weights.

      Outputs:
          latents (`Tensor`):
              The denoised latents.
    """

    model_name = "ideogram4"
    block_classes = [Ideogram4LoopBeforeDenoiser, Ideogram4LoopDenoiser, Ideogram4LoopAfterDenoiser]
    block_names = ["before_denoiser", "denoiser", "after_denoiser"]

    @property
    def description(self) -> str:
        return (
            "Denoising loop that iteratively denoises the packed image latents over `timesteps`, running both the "
            "conditional and unconditional transformers and blending with the per-step guidance schedule."
        )

    @property
    def loop_expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("scheduler", FlowMatchEulerDiscreteScheduler)]

    @property
    def loop_inputs(self) -> list[InputParam]:
        return [
            InputParam(
                name="timesteps",
                required=True,
                type_hint=torch.Tensor,
                description="Denoising timesteps from set_timesteps.",
            ),
            InputParam.template("num_inference_steps", default=48),
        ]

    @torch.no_grad()
    def __call__(self, components: Ideogram4ModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        with self.progress_bar(total=block_state.num_inference_steps) as progress_bar:
            for i, t in enumerate(block_state.timesteps):
                components, block_state = self.loop_step(components, block_state, i=i, t=t)
                progress_bar.update()

        self.set_block_state(state, block_state)
        return components, state


# auto_docstring
class Ideogram4AfterDenoiseStep(ModularPipelineBlocks):
    """
    Step that runs after the denoising loop: unpatchifies the packed image latents (B, num_image_tokens, ae_channels *
    patch ** 2) into a (B, ae_channels, H, W) latent for the decoder.

      Inputs:
          height (`int`):
              The height in pixels of the generated image.
          width (`int`):
              The width in pixels of the generated image.
          latents (`Tensor`):
              The denoised packed image latents (B, num_image_tokens, latent_dim).

      Outputs:
          latents (`Tensor`):
              Unpatchified latents (B, ae_channels, H, W) ready for the VAE decoder.
    """

    model_name = "ideogram4"

    @property
    def description(self) -> str:
        return (
            "Step that runs after the denoising loop: unpatchifies the packed image latents "
            "(B, num_image_tokens, ae_channels * patch ** 2) into a (B, ae_channels, H, W) latent for the decoder."
        )

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("height", required=True),
            InputParam.template("width", required=True),
            InputParam(
                name="latents",
                required=True,
                type_hint=torch.Tensor,
                description="The denoised packed image latents (B, num_image_tokens, latent_dim).",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                name="latents",
                type_hint=torch.Tensor,
                description="Unpatchified latents (B, ae_channels, H, W) ready for the VAE decoder.",
            )
        ]

    @torch.no_grad()
    def __call__(self, components: Ideogram4ModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        z = block_state.latents
        patch = components.patch_size
        grid_h = block_state.height // (components.vae_scale_factor * patch)
        grid_w = block_state.width // (components.vae_scale_factor * patch)

        ae_channels = z.shape[-1] // (patch * patch)
        z = z.view(z.shape[0], grid_h, grid_w, patch, patch, ae_channels)
        z = z.permute(0, 5, 1, 3, 2, 4).contiguous()
        z = z.view(z.shape[0], ae_channels, grid_h * patch, grid_w * patch)

        block_state.latents = z

        self.set_block_state(state, block_state)
        return components, state
