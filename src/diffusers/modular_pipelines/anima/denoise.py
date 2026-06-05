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

from typing import Any

import torch

from ...configuration_utils import FrozenDict
from ...guiders import ClassifierFreeGuidance
from ...models import CosmosTransformer3DModel
from ...schedulers import FlowMatchEulerDiscreteScheduler
from ..modular_pipeline import BlockState, LoopSequentialPipelineBlocks, ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import ComponentSpec, InputParam
from .modular_pipeline import AnimaModularPipeline


class AnimaLoopBeforeDenoiser(ModularPipelineBlocks):
    model_name = "anima"

    @property
    def description(self) -> str:
        return "Step within the denoising loop that prepares Anima latent and timestep inputs."

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("latents", required=True, type_hint=torch.Tensor, description="Current Anima latents."),
            InputParam("dtype", required=True, type_hint=torch.dtype, description="Dtype used by the Anima denoiser."),
        ]

    @torch.no_grad()
    def __call__(self, components: AnimaModularPipeline, block_state: BlockState, i: int, t: torch.Tensor):
        block_state.latent_model_input = block_state.latents.to(block_state.dtype)

        timestep = t.expand(block_state.latents.shape[0]).to(block_state.dtype)
        block_state.timestep = timestep / components.scheduler.config.num_train_timesteps
        return components, block_state


class AnimaLoopDenoiser(ModularPipelineBlocks):
    model_name = "anima"

    def __init__(
        self,
        guider_input_fields: dict[str, Any] | None = None,
    ):
        if guider_input_fields is None:
            guider_input_fields = {"encoder_hidden_states": ("prompt_embeds", "negative_prompt_embeds")}
        if not isinstance(guider_input_fields, dict):
            raise ValueError(f"`guider_input_fields` must be a dictionary but is {type(guider_input_fields)}")
        self._guider_input_fields = guider_input_fields
        super().__init__()

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec(
                "guider",
                ClassifierFreeGuidance,
                config=FrozenDict({"guidance_scale": 4.0}),
                default_creation_method="from_config",
            ),
            ComponentSpec("transformer", CosmosTransformer3DModel),
        ]

    @property
    def description(self) -> str:
        return "Step within the denoising loop that predicts Anima noise with guidance."

    @property
    def inputs(self) -> list[InputParam]:
        inputs = [
            InputParam(
                "num_inference_steps",
                required=True,
                type_hint=int,
                description="Number of denoising steps.",
            ),
            InputParam(
                "padding_mask",
                required=True,
                type_hint=torch.Tensor,
                description="Cosmos padding mask for image latents.",
            ),
            InputParam(
                kwargs_type="denoiser_input_fields",
                description="The conditional model inputs for the Anima denoiser.",
            ),
        ]

        guider_input_names = []
        uncond_guider_input_names = []
        for value in self._guider_input_fields.values():
            if isinstance(value, tuple):
                guider_input_names.append(value[0])
                uncond_guider_input_names.append(value[1])
            else:
                guider_input_names.append(value)

        for name in guider_input_names:
            inputs.append(InputParam(name=name, required=True))
        for name in uncond_guider_input_names:
            inputs.append(InputParam(name=name))
        return inputs

    @torch.no_grad()
    def __call__(
        self, components: AnimaModularPipeline, block_state: BlockState, i: int, t: torch.Tensor
    ) -> PipelineState:
        components.guider.set_state(step=i, num_inference_steps=block_state.num_inference_steps, timestep=t)
        guider_state = components.guider.prepare_inputs_from_block_state(block_state, self._guider_input_fields)

        for guider_state_batch in guider_state:
            components.guider.prepare_models(components.transformer)
            cond_kwargs = {
                key: getattr(guider_state_batch, key).to(block_state.dtype) for key in self._guider_input_fields.keys()
            }
            guider_state_batch.noise_pred = components.transformer(
                hidden_states=block_state.latent_model_input,
                timestep=block_state.timestep,
                padding_mask=block_state.padding_mask,
                return_dict=False,
                **cond_kwargs,
            )[0]
            components.guider.cleanup_models(components.transformer)

        block_state.noise_pred = components.guider(guider_state)[0]
        return components, block_state


class AnimaLoopAfterDenoiser(ModularPipelineBlocks):
    model_name = "anima"

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("scheduler", FlowMatchEulerDiscreteScheduler)]

    @property
    def description(self) -> str:
        return "Step within the denoising loop that updates Anima latents."

    @torch.no_grad()
    def __call__(self, components: AnimaModularPipeline, block_state: BlockState, i: int, t: torch.Tensor):
        latents_dtype = block_state.latents.dtype
        block_state.latents = components.scheduler.step(
            block_state.noise_pred, t, block_state.latents, return_dict=False
        )[0]
        if block_state.latents.dtype != latents_dtype and torch.backends.mps.is_available():
            block_state.latents = block_state.latents.to(latents_dtype)

        return components, block_state


class AnimaDenoiseLoopWrapper(LoopSequentialPipelineBlocks):
    model_name = "anima"

    @property
    def description(self) -> str:
        return "Pipeline block that iteratively denoises Anima latents over scheduler timesteps."

    @property
    def loop_expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("scheduler", FlowMatchEulerDiscreteScheduler)]

    @property
    def loop_inputs(self) -> list[InputParam]:
        return [
            InputParam("timesteps", required=True, type_hint=torch.Tensor, description="Timesteps to denoise over."),
            InputParam("num_inference_steps", required=True, type_hint=int, description="Number of denoising steps."),
        ]

    @torch.no_grad()
    def __call__(self, components: AnimaModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        num_warmup_steps = len(block_state.timesteps) - block_state.num_inference_steps * components.scheduler.order

        with self.progress_bar(total=block_state.num_inference_steps) as progress_bar:
            for i, t in enumerate(block_state.timesteps):
                components, block_state = self.loop_step(components, block_state, i=i, t=t)
                if i == len(block_state.timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % components.scheduler.order == 0
                ):
                    progress_bar.update()

        self.set_block_state(state, block_state)
        return components, state


class AnimaDenoiseStep(AnimaDenoiseLoopWrapper):
    block_classes = [
        AnimaLoopBeforeDenoiser,
        AnimaLoopDenoiser(guider_input_fields={"encoder_hidden_states": ("prompt_embeds", "negative_prompt_embeds")}),
        AnimaLoopAfterDenoiser,
    ]
    block_names = ["before_denoiser", "denoiser", "after_denoiser"]

    @property
    def description(self) -> str:
        return "Denoise step that iteratively denoises image latents for Anima."
