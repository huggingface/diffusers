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


import torch

from ...models.transformers.transformer_krea2 import Krea2Transformer2DModel
from ...schedulers import FlowMatchEulerDiscreteScheduler
from ...utils import logging
from ..modular_pipeline import (
    BlockState,
    LoopSequentialPipelineBlocks,
    ModularPipelineBlocks,
    PipelineState,
)
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam
from .modular_pipeline import Krea2ModularPipeline


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class Krea2LoopBeforeDenoiser(ModularPipelineBlocks):
    model_name = "krea2"

    @property
    def description(self) -> str:
        return (
            "Within the denoising loop: normalize the scheduler timestep into the model's flow time and broadcast it "
            "across the batch. Compose into the `sub_blocks` of `Krea2DenoiseStep`."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("scheduler", FlowMatchEulerDiscreteScheduler)]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(name="latents", required=True, type_hint=torch.Tensor, description="Packed image latents."),
            InputParam(name="batch_size", required=True, type_hint=int, description="Effective batch size."),
        ]

    @torch.no_grad()
    def __call__(self, components: Krea2ModularPipeline, block_state: BlockState, i: int, t: torch.Tensor):
        # Krea 2 flow time is the sigma-domain timestep normalized to [0, 1] (1 = noise, 0 = clean data).
        num_train_timesteps = components.scheduler.config.num_train_timesteps
        block_state.timestep = (t / num_train_timesteps).expand(block_state.batch_size)
        return components, block_state


class Krea2LoopDenoiser(ModularPipelineBlocks):
    model_name = "krea2"

    @property
    def description(self) -> str:
        return (
            "Within the denoising loop: run the `transformer` on the conditional text features and, when "
            "`guidance_scale > 0`, again on the negative features, then combine with symmetric classifier-free "
            "guidance. Compose into `Krea2DenoiseStep`."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("transformer", Krea2Transformer2DModel)]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(name="latents", required=True, type_hint=torch.Tensor, description="Packed image latents."),
            InputParam(
                name="prompt_embeds",
                required=True,
                type_hint=torch.Tensor,
                description="Conditional stacked text features.",
            ),
            InputParam(
                name="prompt_embeds_mask", required=True, type_hint=torch.Tensor, description="Conditional text mask."
            ),
            InputParam(
                name="position_ids",
                required=True,
                type_hint=torch.Tensor,
                description="Shared rotary coordinates for the [text | image] sequence.",
            ),
            InputParam(
                name="negative_prompt_embeds", type_hint=torch.Tensor, description="Negative stacked text features."
            ),
            InputParam(name="negative_prompt_embeds_mask", type_hint=torch.Tensor, description="Negative text mask."),
            InputParam(
                name="guidance_scale",
                default=4.5,
                type_hint=float,
                description="Symmetric CFG scale; guidance is applied when > 0.",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components: Krea2ModularPipeline, block_state: BlockState, i: int, t: torch.Tensor):
        transformer = components.transformer

        latents = block_state.latents.to(transformer.dtype)
        timestep = block_state.timestep.to(transformer.dtype)

        noise_pred = transformer(
            hidden_states=latents,
            encoder_hidden_states=block_state.prompt_embeds.to(transformer.dtype),
            timestep=timestep,
            position_ids=block_state.position_ids,
            encoder_attention_mask=block_state.prompt_embeds_mask,
            attention_kwargs=block_state.attention_kwargs,
            return_dict=False,
        )[0]

        if block_state.guidance_scale > 0 and block_state.negative_prompt_embeds is not None:
            neg_noise_pred = transformer(
                hidden_states=latents,
                encoder_hidden_states=block_state.negative_prompt_embeds.to(transformer.dtype),
                timestep=timestep,
                position_ids=block_state.position_ids,
                encoder_attention_mask=block_state.negative_prompt_embeds_mask,
                attention_kwargs=block_state.attention_kwargs,
                return_dict=False,
            )[0]
            # Symmetric CFG: cond + scale * (cond - uncond), i.e. the usual CFG with scale `1 + guidance_scale`.
            noise_pred = noise_pred + block_state.guidance_scale * (noise_pred - neg_noise_pred)

        block_state.noise_pred = noise_pred
        return components, block_state


class Krea2LoopAfterDenoiser(ModularPipelineBlocks):
    model_name = "krea2"

    @property
    def description(self) -> str:
        return "Within the denoising loop: scheduler step. Compose into `Krea2DenoiseStep`."

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("scheduler", FlowMatchEulerDiscreteScheduler)]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [OutputParam(name="latents", type_hint=torch.Tensor, description="The denoised latents.")]

    @torch.no_grad()
    def __call__(self, components: Krea2ModularPipeline, block_state: BlockState, i: int, t: torch.Tensor):
        latents_dtype = block_state.latents.dtype
        block_state.latents = components.scheduler.step(
            block_state.noise_pred, t, block_state.latents, return_dict=False
        )[0]
        block_state.latents = block_state.latents.to(latents_dtype)
        return components, block_state


# auto_docstring
class Krea2DenoiseStep(LoopSequentialPipelineBlocks):
    """
    Denoising loop that iteratively denoises the packed image latents over `timesteps`, running the transformer on the
    conditional (and, when `guidance_scale > 0`, the negative) text features and combining them with symmetric
    classifier-free guidance.

      Components:
          scheduler (`FlowMatchEulerDiscreteScheduler`) transformer (`Krea2Transformer2DModel`)

      Inputs:
          timesteps (`Tensor`):
              Denoising timesteps from set_timesteps.
          num_inference_steps (`int`, *optional*, defaults to 28):
              The number of denoising steps.
          latents (`Tensor`):
              Packed image latents.
          batch_size (`int`):
              Effective batch size.
          prompt_embeds (`Tensor`):
              Conditional stacked text features.
          prompt_embeds_mask (`Tensor`):
              Conditional text mask.
          position_ids (`Tensor`):
              Shared rotary coordinates for the [text | image] sequence.
          negative_prompt_embeds (`Tensor`):
              Negative stacked text features.
          negative_prompt_embeds_mask (`Tensor`):
              Negative text mask.
          guidance_scale (`float`, *optional*, defaults to 4.5):
              Symmetric CFG scale; guidance is applied when > 0.
          attention_kwargs (`dict`, *optional*):
              Forwarded to the transformer's attention processor.

      Outputs:
          latents (`Tensor`):
              The denoised packed latents (B, image_seq_len, in_channels).
    """

    model_name = "krea2"
    block_classes = [Krea2LoopBeforeDenoiser, Krea2LoopDenoiser, Krea2LoopAfterDenoiser]
    block_names = ["before_denoiser", "denoiser", "after_denoiser"]

    @property
    def description(self) -> str:
        return (
            "Denoising loop that iteratively denoises the packed image latents over `timesteps`, running the "
            "transformer on the conditional (and, when `guidance_scale > 0`, the negative) text features and "
            "combining them with symmetric classifier-free guidance."
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
            InputParam.template("num_inference_steps", default=28),
            InputParam.template("attention_kwargs"),
        ]

    @torch.no_grad()
    def __call__(self, components: Krea2ModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        with self.progress_bar(total=block_state.num_inference_steps) as progress_bar:
            for i, t in enumerate(block_state.timesteps):
                components, block_state = self.loop_step(components, block_state, i=i, t=t)
                progress_bar.update()

        self.set_block_state(state, block_state)
        return components, state
