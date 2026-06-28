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

from ...configuration_utils import FrozenDict
from ...guiders import ClassifierFreeGuidance
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
        num_train_timesteps = components.scheduler.config.num_train_timesteps
        block_state.timestep = (t / num_train_timesteps).expand(block_state.batch_size)
        return components, block_state


class Krea2LoopDenoiser(ModularPipelineBlocks):
    model_name = "krea2"

    @property
    def description(self) -> str:
        return (
            "Within the denoising loop: run the `transformer` on the conditional (and, when the guider enables CFG, "
            "the negative) text features and combine them through the `guider`. Compose into `Krea2DenoiseStep`."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec(
                "guider",
                ClassifierFreeGuidance,
                # Krea 2 uses cond-anchored CFG (`cond + scale * (cond - uncond)`), which is the
                # `use_original_formulation` branch of ClassifierFreeGuidance; scale 0 disables it (distilled TDM).
                config=FrozenDict({"guidance_scale": 4.5, "use_original_formulation": True}),
                default_creation_method="from_config",
            ),
            ComponentSpec("transformer", Krea2Transformer2DModel),
        ]

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
        ]

    @torch.no_grad()
    def __call__(self, components: Krea2ModularPipeline, block_state: BlockState, i: int, t: torch.Tensor):
        transformer = components.transformer

        latents = block_state.latents.to(transformer.dtype)
        timestep = block_state.timestep.to(transformer.dtype)

        guider_inputs = {
            "encoder_hidden_states": (
                block_state.prompt_embeds.to(transformer.dtype),
                block_state.negative_prompt_embeds.to(transformer.dtype)
                if block_state.negative_prompt_embeds is not None
                else None,
            ),
            "encoder_attention_mask": (
                block_state.prompt_embeds_mask,
                block_state.negative_prompt_embeds_mask,
            ),
        }

        components.guider.set_state(step=i, num_inference_steps=block_state.num_inference_steps, timestep=t)
        guider_state = components.guider.prepare_inputs(guider_inputs)

        for guider_state_batch in guider_state:
            components.guider.prepare_models(components.transformer)
            cond_kwargs = {name: getattr(guider_state_batch, name) for name in guider_inputs}
            guider_state_batch.noise_pred = transformer(
                hidden_states=latents,
                timestep=timestep,
                position_ids=block_state.position_ids,
                attention_kwargs=block_state.attention_kwargs,
                return_dict=False,
                **cond_kwargs,
            )[0]
            components.guider.cleanup_models(components.transformer)

        block_state.noise_pred = components.guider(guider_state).pred
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
    conditional (and, when the guider enables CFG, the negative) text features and combining them through the `guider`.

      Components:
          scheduler (`FlowMatchEulerDiscreteScheduler`) guider (`ClassifierFreeGuidance`) transformer
          (`Krea2Transformer2DModel`)

      Inputs:
          timesteps (`Tensor`):
              Denoising timesteps from set_timesteps.
          num_inference_steps (`int`, *optional*, defaults to 28):
              The number of denoising steps.
          attention_kwargs (`dict`, *optional*):
              Additional kwargs for attention processors.
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
          negative_prompt_embeds (`Tensor`, *optional*):
              Negative stacked text features.
          negative_prompt_embeds_mask (`Tensor`, *optional*):
              Negative text mask.

      Outputs:
          latents (`Tensor`):
              The denoised latents.
    """

    model_name = "krea2"
    block_classes = [Krea2LoopBeforeDenoiser, Krea2LoopDenoiser, Krea2LoopAfterDenoiser]
    block_names = ["before_denoiser", "denoiser", "after_denoiser"]

    @property
    def description(self) -> str:
        return (
            "Denoising loop that iteratively denoises the packed image latents over `timesteps`, running the "
            "transformer on the conditional (and, when the guider enables CFG, the negative) text features and "
            "combining them through the `guider`."
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
