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
from ...models import PixArtTransformer2DModel
from ...schedulers import DPMSolverMultistepScheduler
from ...utils import logging
from ..modular_pipeline import (
    BlockState,
    LoopSequentialPipelineBlocks,
    ModularPipelineBlocks,
    PipelineState,
)
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam
from .modular_pipeline import PixArtAlphaModularPipeline


logger = logging.get_logger(__name__)


class PixArtAlphaLoopDenoiser(ModularPipelineBlocks):
    model_name = "pixart-alpha"

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec(
                "guider",
                ClassifierFreeGuidance,
                config=FrozenDict({"guidance_scale": 4.5}),
                default_creation_method="from_config",
            ),
            ComponentSpec("transformer", PixArtTransformer2DModel),
            ComponentSpec("scheduler", DPMSolverMultistepScheduler),
        ]

    @property
    def description(self) -> str:
        return "Step within the denoising loop that predicts the noise with classifier-free guidance."

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                "latents",
                required=True,
                type_hint=torch.Tensor,
                description="The latents to denoise.",
            ),
            InputParam(
                "num_inference_steps",
                required=True,
                type_hint=int,
                description="The number of denoising steps.",
            ),
            InputParam(
                "prompt_embeds",
                required=True,
                type_hint=torch.Tensor,
                description="The prompt embeddings used to guide the denoising.",
            ),
            InputParam(
                "prompt_embeds_mask",
                required=True,
                type_hint=torch.Tensor,
                description="The attention mask for the prompt embeddings.",
            ),
            InputParam(
                "negative_prompt_embeds",
                type_hint=torch.Tensor,
                description="The negative prompt embeddings used for classifier-free guidance.",
            ),
            InputParam(
                "negative_prompt_embeds_mask",
                type_hint=torch.Tensor,
                description="The attention mask for the negative prompt embeddings.",
            ),
            InputParam(
                "resolution",
                type_hint=torch.Tensor,
                description="The resolution micro-condition, or None when unused by the checkpoint.",
            ),
            InputParam(
                "aspect_ratio",
                type_hint=torch.Tensor,
                description="The aspect-ratio micro-condition, or None when unused by the checkpoint.",
            ),
        ]

    @torch.no_grad()
    def __call__(
        self,
        components: PixArtAlphaModularPipeline,
        block_state: BlockState,
        i: int,
        t: torch.Tensor,
    ) -> PipelineState:
        do_cfg = block_state.negative_prompt_embeds is not None

        guider_inputs = {
            "hidden_states": (block_state.latents, block_state.latents) if do_cfg else block_state.latents,
            "encoder_hidden_states": (block_state.prompt_embeds, block_state.negative_prompt_embeds)
            if do_cfg
            else block_state.prompt_embeds,
            "encoder_attention_mask": (block_state.prompt_embeds_mask, block_state.negative_prompt_embeds_mask)
            if do_cfg
            else block_state.prompt_embeds_mask,
        }

        components.guider.set_state(step=i, num_inference_steps=block_state.num_inference_steps, timestep=t)
        guider_state = components.guider.prepare_inputs(guider_inputs)

        added_cond_kwargs = {"resolution": block_state.resolution, "aspect_ratio": block_state.aspect_ratio}

        for guider_state_batch in guider_state:
            components.guider.prepare_models(components.transformer)

            latent_model_input = components.scheduler.scale_model_input(guider_state_batch.hidden_states, t)
            timestep = t.expand(latent_model_input.shape[0])

            noise_pred = components.transformer(
                hidden_states=latent_model_input,
                encoder_hidden_states=guider_state_batch.encoder_hidden_states,
                encoder_attention_mask=guider_state_batch.encoder_attention_mask,
                timestep=timestep,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]

            # PixArt transformers with a learned sigma predict both the noise and the variance; keep only the noise.
            if components.transformer.config.out_channels // 2 == components.transformer.config.in_channels:
                noise_pred = noise_pred.chunk(2, dim=1)[0]

            guider_state_batch.noise_pred = noise_pred
            components.guider.cleanup_models(components.transformer)

        guider_output = components.guider(guider_state)
        block_state.noise_pred = guider_output.pred

        return components, block_state


class PixArtAlphaLoopAfterDenoiser(ModularPipelineBlocks):
    model_name = "pixart-alpha"

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("scheduler", DPMSolverMultistepScheduler)]

    @property
    def description(self) -> str:
        return "Step within the denoising loop that updates the latents with the scheduler."

    @property
    def inputs(self) -> list[InputParam]:
        return [InputParam.template("generator")]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "latents",
                type_hint=torch.Tensor,
                description="The denoised latents.",
            )
        ]

    @torch.no_grad()
    def __call__(
        self,
        components: PixArtAlphaModularPipeline,
        block_state: BlockState,
        i: int,
        t: torch.Tensor,
    ):
        block_state.latents = components.scheduler.step(
            block_state.noise_pred, t, block_state.latents, generator=block_state.generator, return_dict=False
        )[0]

        return components, block_state


class PixArtAlphaDenoiseLoopWrapper(LoopSequentialPipelineBlocks):
    model_name = "pixart-alpha"

    @property
    def description(self) -> str:
        return "Pipeline block that iteratively denoises the latents over the scheduler's timesteps."

    @property
    def loop_expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec(
                "guider",
                ClassifierFreeGuidance,
                config=FrozenDict({"guidance_scale": 4.5}),
                default_creation_method="from_config",
            ),
            ComponentSpec("transformer", PixArtTransformer2DModel),
            ComponentSpec("scheduler", DPMSolverMultistepScheduler),
        ]

    @property
    def loop_inputs(self) -> list[InputParam]:
        return [
            InputParam("timesteps", required=True, type_hint=torch.Tensor),
            InputParam("num_inference_steps", required=True, type_hint=int),
        ]

    @torch.no_grad()
    def __call__(self, components: PixArtAlphaModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        block_state.num_warmup_steps = max(
            len(block_state.timesteps) - block_state.num_inference_steps * components.scheduler.order,
            0,
        )

        with self.progress_bar(total=block_state.num_inference_steps) as progress_bar:
            for i, t in enumerate(block_state.timesteps):
                components, block_state = self.loop_step(components, block_state, i=i, t=t)
                if i == len(block_state.timesteps) - 1 or (
                    (i + 1) > block_state.num_warmup_steps and (i + 1) % components.scheduler.order == 0
                ):
                    progress_bar.update()

        self.set_block_state(state, block_state)
        return components, state


class PixArtAlphaDenoiseStep(PixArtAlphaDenoiseLoopWrapper):
    block_classes = [PixArtAlphaLoopDenoiser, PixArtAlphaLoopAfterDenoiser]
    block_names = ["denoiser", "after_denoiser"]
