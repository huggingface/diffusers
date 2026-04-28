# Copyright 2025 Baidu ERNIE-Image Team and The HuggingFace Team. All rights reserved.
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
from ...models import ErnieImageTransformer2DModel
from ...schedulers import FlowMatchEulerDiscreteScheduler
from ...utils import logging
from ..modular_pipeline import (
    BlockState,
    LoopSequentialPipelineBlocks,
    ModularPipelineBlocks,
    PipelineState,
)
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam
from .modular_pipeline import ErnieImageModularPipeline


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class ErnieImageLoopBeforeDenoiser(ModularPipelineBlocks):
    model_name = "ernie-image"

    @property
    def description(self) -> str:
        return (
            "Step within the denoising loop that prepares the latent model input and timestep tensor. "
            "This block should be used to compose the `sub_blocks` attribute of a `LoopSequentialPipelineBlocks` "
            "object (e.g. `ErnieImageDenoiseLoopWrapper`)."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("transformer", ErnieImageTransformer2DModel)]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                "latents",
                required=True,
                type_hint=torch.Tensor,
                description="The latents to denoise.",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components: ErnieImageModularPipeline, block_state: BlockState, i: int, t: torch.Tensor):
        latents = block_state.latents
        block_state.latent_model_input = latents.to(components.transformer.dtype)
        block_state.timestep = t.expand(latents.shape[0]).to(components.transformer.dtype)
        return components, block_state


class ErnieImageLoopDenoiser(ModularPipelineBlocks):
    model_name = "ernie-image"

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("transformer", ErnieImageTransformer2DModel),
            ComponentSpec(
                "guider",
                ClassifierFreeGuidance,
                config=FrozenDict({"guidance_scale": 4.0}),
                default_creation_method="from_config",
            ),
        ]

    @property
    def description(self) -> str:
        return (
            "Step within the denoising loop that runs the ErnieImage transformer with classifier-free guidance via "
            "the configured guider."
        )

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                "text_bth",
                required=True,
                type_hint=torch.Tensor,
                description="Padded text hidden states fed into the transformer.",
            ),
            InputParam(
                "text_lens",
                required=True,
                type_hint=torch.Tensor,
                description="Per-prompt text lengths used by the transformer attention mask.",
            ),
            InputParam(
                "negative_text_bth",
                type_hint=torch.Tensor,
                description="Padded negative text hidden states for classifier-free guidance.",
            ),
            InputParam(
                "negative_text_lens",
                type_hint=torch.Tensor,
                description="Per-prompt negative text lengths for classifier-free guidance.",
            ),
            InputParam(
                "num_inference_steps",
                required=True,
                type_hint=int,
                description="Total number of denoising steps. Used by the guider for step-aware scheduling.",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components: ErnieImageModularPipeline, block_state: BlockState, i: int, t: torch.Tensor):
        guider_inputs = {
            "text_bth": (block_state.text_bth, block_state.negative_text_bth),
            "text_lens": (block_state.text_lens, block_state.negative_text_lens),
        }

        components.guider.set_state(step=i, num_inference_steps=block_state.num_inference_steps, timestep=t)
        guider_state = components.guider.prepare_inputs(guider_inputs)

        for guider_state_batch in guider_state:
            components.guider.prepare_models(components.transformer)
            cond_kwargs = {name: getattr(guider_state_batch, name) for name in guider_inputs.keys()}
            noise_pred = components.transformer(
                hidden_states=block_state.latent_model_input,
                timestep=block_state.timestep,
                return_dict=False,
                **cond_kwargs,
            )[0]
            guider_state_batch.noise_pred = noise_pred
            components.guider.cleanup_models(components.transformer)

        block_state.noise_pred = components.guider(guider_state)[0]
        return components, block_state


class ErnieImageLoopAfterDenoiser(ModularPipelineBlocks):
    model_name = "ernie-image"

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("scheduler", FlowMatchEulerDiscreteScheduler)]

    @property
    def description(self) -> str:
        return "Step within the denoising loop that updates the latents using the scheduler step."

    @torch.no_grad()
    def __call__(self, components: ErnieImageModularPipeline, block_state: BlockState, i: int, t: torch.Tensor):
        latents_dtype = block_state.latents.dtype
        block_state.latents = components.scheduler.step(
            block_state.noise_pred, t, block_state.latents, return_dict=False
        )[0]
        if block_state.latents.dtype != latents_dtype and torch.backends.mps.is_available():
            block_state.latents = block_state.latents.to(latents_dtype)
        return components, block_state


class ErnieImageDenoiseLoopWrapper(LoopSequentialPipelineBlocks):
    model_name = "ernie-image"

    @property
    def description(self) -> str:
        return (
            "Pipeline block that iteratively denoises the latents over `timesteps`. "
            "The specific steps within each iteration can be customized with `sub_blocks` attribute."
        )

    @property
    def loop_expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("scheduler", FlowMatchEulerDiscreteScheduler),
            ComponentSpec("transformer", ErnieImageTransformer2DModel),
        ]

    @property
    def loop_inputs(self) -> list[InputParam]:
        return [
            InputParam(
                "timesteps",
                required=True,
                type_hint=torch.Tensor,
                description="The timesteps to use for inference.",
            ),
            InputParam(
                "num_inference_steps",
                required=True,
                type_hint=int,
                description="The number of denoising steps.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [OutputParam("latents", type_hint=torch.Tensor, description="The denoised latents.")]

    @torch.no_grad()
    def __call__(self, components: ErnieImageModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        with self.progress_bar(total=block_state.num_inference_steps) as progress_bar:
            for i, t in enumerate(block_state.timesteps):
                components, block_state = self.loop_step(components, block_state, i=i, t=t)
                progress_bar.update()
        self.set_block_state(state, block_state)
        return components, state


class ErnieImageDenoiseStep(ErnieImageDenoiseLoopWrapper):
    block_classes = [
        ErnieImageLoopBeforeDenoiser,
        ErnieImageLoopDenoiser,
        ErnieImageLoopAfterDenoiser,
    ]
    block_names = ["before_denoiser", "denoiser", "after_denoiser"]

    @property
    def description(self) -> str:
        return (
            "Denoise step that iteratively denoises the latents. At each iteration it runs:\n"
            " - `ErnieImageLoopBeforeDenoiser`\n"
            " - `ErnieImageLoopDenoiser`\n"
            " - `ErnieImageLoopAfterDenoiser`"
        )
