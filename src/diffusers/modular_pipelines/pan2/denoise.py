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
from ...models import PAN2Transformer3DModel
from ...schedulers import FlowMatchEulerDiscreteScheduler
from ...utils import logging
from ..modular_pipeline import (
    BlockState,
    LoopSequentialPipelineBlocks,
    ModularPipelineBlocks,
    PipelineState,
)
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam
from .modular_pipeline import PAN2ModularPipeline


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class PAN2LoopBeforeDenoiser(ModularPipelineBlocks):
    model_name = "pan2"

    @property
    def description(self) -> str:
        return (
            "Step within the denoising loop that stacks the conditioning latents onto the noisy latents. "
            "This block should be used to compose the `sub_blocks` attribute of a `LoopSequentialPipelineBlocks` "
            "object (e.g. `PAN2DenoiseLoopWrapper`)"
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("transformer", PAN2Transformer3DModel)]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("latents", required=True, type_hint=torch.Tensor, description="The latents to denoise."),
            InputParam(
                "condition_latents",
                required=True,
                type_hint=torch.Tensor,
                description="The conditioning latents and mask. Can be generated from a prepare condition step.",
            ),
        ]

    @torch.no_grad()
    def __call__(
        self, components: PAN2ModularPipeline, block_state: BlockState, i: int, t: torch.Tensor
    ) -> tuple[PAN2ModularPipeline, BlockState]:
        dtype = components.transformer.dtype
        block_state.latent_model_input = torch.cat(
            [block_state.latents.to(dtype), block_state.condition_latents.to(dtype)], dim=1
        )
        return components, block_state


class PAN2LoopDenoiser(ModularPipelineBlocks):
    model_name = "pan2"

    @property
    def description(self) -> str:
        return (
            "Step within the denoising loop that predicts the velocity with guidance. "
            "This block should be used to compose the `sub_blocks` attribute of a `LoopSequentialPipelineBlocks` "
            "object (e.g. `PAN2DenoiseLoopWrapper`)"
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec(
                "guider",
                ClassifierFreeGuidance,
                config=FrozenDict({"guidance_scale": 3.0}),
                default_creation_method="from_config",
            ),
            ComponentSpec("transformer", PAN2Transformer3DModel),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("attention_kwargs"),
            InputParam(
                "num_inference_steps",
                required=True,
                type_hint=int,
                description="The number of denoising steps. Can be generated in set_timesteps step.",
            ),
            InputParam.template("prompt_embeds", description="Text embeddings used to guide the video generation."),
            InputParam.template(
                "negative_prompt_embeds",
                description="Negative text embeddings used to guide the video generation.",
            ),
        ]

    @torch.no_grad()
    def __call__(
        self, components: PAN2ModularPipeline, block_state: BlockState, i: int, t: torch.Tensor
    ) -> tuple[PAN2ModularPipeline, BlockState]:
        dtype = components.transformer.dtype
        latent_model_input = block_state.latent_model_input
        timestep = t.expand(latent_model_input.shape[0])

        guider_inputs = {
            "encoder_hidden_states": (block_state.prompt_embeds, block_state.negative_prompt_embeds),
        }

        components.guider.set_state(step=i, num_inference_steps=block_state.num_inference_steps, timestep=t)
        guider_state = components.guider.prepare_inputs(guider_inputs)

        for guider_state_batch in guider_state:
            components.guider.prepare_models(components.transformer)
            cond_kwargs = {input_name: getattr(guider_state_batch, input_name) for input_name in guider_inputs}
            cond_kwargs["encoder_hidden_states"] = cond_kwargs["encoder_hidden_states"].to(dtype)

            context_name = getattr(guider_state_batch, components.guider._identifier_key)
            with components.transformer.cache_context(context_name):
                guider_state_batch.noise_pred = components.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    attention_kwargs=block_state.attention_kwargs,
                    return_dict=False,
                    **cond_kwargs,
                )[0]
            components.guider.cleanup_models(components.transformer)

        block_state.noise_pred = components.guider(guider_state).pred
        return components, block_state


class PAN2LoopAfterDenoiser(ModularPipelineBlocks):
    model_name = "pan2"

    @property
    def description(self) -> str:
        return (
            "Step within the denoising loop that updates the latents with the scheduler. "
            "This block should be used to compose the `sub_blocks` attribute of a `LoopSequentialPipelineBlocks` "
            "object (e.g. `PAN2DenoiseLoopWrapper`)"
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("scheduler", FlowMatchEulerDiscreteScheduler)]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [OutputParam("latents", type_hint=torch.Tensor, description="The latents after this denoising step.")]

    @torch.no_grad()
    def __call__(
        self, components: PAN2ModularPipeline, block_state: BlockState, i: int, t: torch.Tensor
    ) -> tuple[PAN2ModularPipeline, BlockState]:
        # PAN2 keeps the latents in float32 across the loop: the scheduler returns the sample in the dtype of the
        # prediction, so it is upcast first.
        block_state.latents = components.scheduler.step(
            block_state.noise_pred.float(), t, block_state.latents, return_dict=False
        )[0]
        return components, block_state


class PAN2DenoiseLoopWrapper(LoopSequentialPipelineBlocks):
    model_name = "pan2"

    @property
    def description(self) -> str:
        return (
            "Pipeline block that iteratively denoises the latents over `timesteps`. "
            "The specific steps within each iteration can be customized with `sub_blocks` attribute"
        )

    @property
    def loop_expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("scheduler", FlowMatchEulerDiscreteScheduler)]

    @property
    def loop_inputs(self) -> list[InputParam]:
        return [
            InputParam(
                "timesteps",
                required=True,
                type_hint=torch.Tensor,
                description="The timesteps to denoise over. Can be generated in set_timesteps step.",
            ),
            InputParam(
                "num_inference_steps",
                required=True,
                type_hint=int,
                description="The number of denoising steps. Can be generated in set_timesteps step.",
            ),
        ]

    @torch.no_grad()
    def __call__(
        self, components: PAN2ModularPipeline, state: PipelineState
    ) -> tuple[PAN2ModularPipeline, PipelineState]:
        block_state = self.get_block_state(state)

        with self.progress_bar(total=block_state.num_inference_steps) as progress_bar:
            for i, t in enumerate(block_state.timesteps):
                components, block_state = self.loop_step(components, block_state, i=i, t=t)
                progress_bar.update()

        self.set_block_state(state, block_state)
        return components, state


class PAN2DenoiseStep(PAN2DenoiseLoopWrapper):
    block_classes = [PAN2LoopBeforeDenoiser, PAN2LoopDenoiser, PAN2LoopAfterDenoiser]
    block_names = ["before_denoiser", "denoiser", "after_denoiser"]

    @property
    def description(self) -> str:
        return (
            "Denoise step that iteratively denoises the latents.\n"
            "Its loop logic is defined in `PAN2DenoiseLoopWrapper.__call__` method.\n"
            "At each iteration, it runs blocks defined in `sub_blocks` sequentially:\n"
            " - `PAN2LoopBeforeDenoiser`\n"
            " - `PAN2LoopDenoiser`\n"
            " - `PAN2LoopAfterDenoiser`\n"
            "This block supports both text-to-video and image-to-video."
        )
