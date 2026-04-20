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


import torch

from ...configuration_utils import FrozenDict
from ...guiders import ClassifierFreeGuidance
from ...models.transformers import SD3Transformer2DModel
from ...schedulers import FlowMatchEulerDiscreteScheduler
from ...utils import logging
from ..modular_pipeline import (
    BlockState,
    LoopSequentialPipelineBlocks,
    ModularPipelineBlocks,
    PipelineState,
)
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam
from .modular_pipeline import StableDiffusion3ModularPipeline

logger = logging.get_logger(__name__)


class StableDiffusion3LoopDenoiser(ModularPipelineBlocks):
    model_name = "stable-diffusion-3"

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec(
                "guider",
                ClassifierFreeGuidance,
                config=FrozenDict({"guidance_scale": 7.0}),
                default_creation_method="from_config",
            ),
            ComponentSpec("transformer", SD3Transformer2DModel),
        ]

    @property
    def description(self) -> str:
        return "Step within the denoising loop that denoises the latents."

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                "joint_attention_kwargs",
                type_hint=dict,
                description="A kwargs dictionary passed along to the AttentionProcessor.",
            ),
            InputParam(
                "latents",
                required=True,
                type_hint=torch.Tensor,
                description="The initial latents to use for the denoising process.",
            ),
            InputParam(
                "prompt_embeds",
                required=True,
                type_hint=torch.Tensor,
                description="Text embeddings for guidance.",
            ),
            InputParam(
                "pooled_prompt_embeds",
                required=True,
                type_hint=torch.Tensor,
                description="Pooled text embeddings for guidance.",
            ),
            InputParam(
                "negative_prompt_embeds",
                type_hint=torch.Tensor,
                description="Negative text embeddings for guidance.",
            ),
            InputParam(
                "negative_pooled_prompt_embeds",
                type_hint=torch.Tensor,
                description="Negative pooled text embeddings for guidance.",
            ),
            InputParam(
                "num_inference_steps",
                type_hint=int,
                description="The number of denoising steps.",
            ),
        ]

    @torch.no_grad()
    def __call__(
        self,
        components: StableDiffusion3ModularPipeline,
        block_state: BlockState,
        i: int,
        t: torch.Tensor,
    ) -> PipelineState:
        do_cfg = getattr(block_state, "negative_prompt_embeds", None) is not None

        guider_inputs = {
            "hidden_states": (
                (block_state.latents, block_state.latents)
                if do_cfg
                else block_state.latents
            ),
            "encoder_hidden_states": (
                (
                    getattr(block_state, "prompt_embeds", None),
                    getattr(block_state, "negative_prompt_embeds", None),
                )
                if do_cfg
                else getattr(block_state, "prompt_embeds", None)
            ),
            "text_embeds": (
                (
                    getattr(block_state, "pooled_prompt_embeds", None),
                    getattr(block_state, "negative_pooled_prompt_embeds", None),
                )
                if do_cfg
                else getattr(block_state, "pooled_prompt_embeds", None)
            ),
        }

        components.guider.set_state(
            step=i, num_inference_steps=block_state.num_inference_steps, timestep=t
        )
        guider_state = components.guider.prepare_inputs(guider_inputs)

        for guider_state_batch in guider_state:
            components.guider.prepare_models(components.transformer)

            latent_model_input = guider_state_batch.hidden_states
            prompt_embeds = guider_state_batch.encoder_hidden_states
            pooled_projections = getattr(guider_state_batch, "text_embeds", None)

            timestep = t.expand(latent_model_input.shape[0])

            guider_state_batch.noise_pred = components.transformer(
                hidden_states=latent_model_input,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_projections,
                joint_attention_kwargs=getattr(
                    block_state, "joint_attention_kwargs", None
                ),
                return_dict=False,
            )[0]

            components.guider.cleanup_models(components.transformer)

        guider_output = components.guider(guider_state)
        block_state.noise_pred = guider_output.pred

        return components, block_state


class StableDiffusion3LoopAfterDenoiser(ModularPipelineBlocks):
    model_name = "stable-diffusion-3"

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("scheduler", FlowMatchEulerDiscreteScheduler)]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "latents",
                type_hint=torch.Tensor,
                description="The denoised latent tensors.",
            )
        ]

    @torch.no_grad()
    def __call__(
        self,
        components: StableDiffusion3ModularPipeline,
        block_state: BlockState,
        i: int,
        t: torch.Tensor,
    ):
        latents_dtype = block_state.latents.dtype
        block_state.latents = components.scheduler.step(
            block_state.noise_pred,
            t,
            block_state.latents,
            return_dict=False,
        )[0]

        if block_state.latents.dtype != latents_dtype:
            block_state.latents = block_state.latents.to(latents_dtype)

        return components, block_state


class StableDiffusion3DenoiseLoopWrapper(LoopSequentialPipelineBlocks):
    model_name = "stable-diffusion-3"

    @property
    def loop_expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("scheduler", FlowMatchEulerDiscreteScheduler),
            ComponentSpec("transformer", SD3Transformer2DModel),
        ]

    @property
    def loop_inputs(self) -> list[InputParam]:
        return [
            InputParam("timesteps", required=True, type_hint=torch.Tensor),
            InputParam("num_inference_steps", required=True, type_hint=int),
        ]

    @torch.no_grad()
    def __call__(
        self, components: StableDiffusion3ModularPipeline, state: PipelineState
    ) -> PipelineState:
        block_state = self.get_block_state(state)
        block_state.num_warmup_steps = max(
            len(block_state.timesteps)
            - block_state.num_inference_steps * components.scheduler.order,
            0,
        )

        with self.progress_bar(total=block_state.num_inference_steps) as progress_bar:
            for i, t in enumerate(block_state.timesteps):
                components, block_state = self.loop_step(
                    components, block_state, i=i, t=t
                )
                if i == len(block_state.timesteps) - 1 or (
                    (i + 1) > block_state.num_warmup_steps
                    and (i + 1) % components.scheduler.order == 0
                ):
                    progress_bar.update()

        self.set_block_state(state, block_state)
        return components, state


# auto_docstring
class StableDiffusion3DenoiseStep(StableDiffusion3DenoiseLoopWrapper):
    block_classes = [StableDiffusion3LoopDenoiser, StableDiffusion3LoopAfterDenoiser]
    block_names = ["denoiser", "after_denoiser"]
