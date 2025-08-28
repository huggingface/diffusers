# Copyright 2025 The HuggingFace Team. All rights reserved.
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

from typing import Any, List, Tuple

import torch

from ...configuration_utils import FrozenDict
from ...guiders import ClassifierFreeGuidance
from ...models import WanTransformer3DModel
from ...schedulers import UniPCMultistepScheduler
from ...utils import logging
from ..modular_pipeline import (
    BlockState,
    LoopSequentialPipelineBlocks,
    ModularPipelineBlocks,
    PipelineState,
)
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam
from .modular_pipeline import WanModularPipeline


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class WanLoopDenoiser(ModularPipelineBlocks):
    model_name = "wan"

    @property
    def expected_components(self) -> List[ComponentSpec]:
        return [
            ComponentSpec(
                "guider",
                ClassifierFreeGuidance,
                config=FrozenDict({"guidance_scale": 5.0}),
                default_creation_method="from_config",
            ),
            ComponentSpec("transformer", WanTransformer3DModel),
        ]

    @property
    def description(self) -> str:
        return (
            "Step within the denoising loop that denoise the latents with guidance. "
            "This block should be used to compose the `sub_blocks` attribute of a `LoopSequentialPipelineBlocks` "
            "object (e.g. `WanDenoiseLoopWrapper`)"
        )

    @property
    def inputs(self) -> List[Tuple[str, Any]]:
        return [
            InputParam("attention_kwargs"),
        ]

    @property
    def intermediate_inputs(self) -> List[str]:
        return [
            InputParam(
                "latents",
                required=True,
                type_hint=torch.Tensor,
                description="The initial latents to use for the denoising process. Can be generated in prepare_latent step.",
            ),
            InputParam(
                "num_inference_steps",
                required=True,
                type_hint=int,
                description="The number of inference steps to use for the denoising process. Can be generated in set_timesteps step.",
            ),
            InputParam(
                kwargs_type="guider_input_fields",
                description=(
                    "All conditional model inputs that need to be prepared with guider. "
                    "It should contain prompt_embeds/negative_prompt_embeds. "
                    "Please add `kwargs_type=guider_input_fields` to their parameter spec (`OutputParam`) when they are created and added to the pipeline state"
                ),
            ),
        ]

    @torch.no_grad()
    def __call__(
        self, components: WanModularPipeline, block_state: BlockState, i: int, t: torch.Tensor
    ) -> PipelineState:
        #  Map the keys we'll see on each `guider_state_batch` (e.g. guider_state_batch.prompt_embeds)
        #  to the corresponding (cond, uncond) fields on block_state. (e.g. block_state.prompt_embeds, block_state.negative_prompt_embeds)
        guider_input_fields = {
            "prompt_embeds": ("prompt_embeds", "negative_prompt_embeds"),
        }
        transformer_dtype = components.transformer.dtype

        components.guider.set_state(step=i, num_inference_steps=block_state.num_inference_steps, timestep=t)

        # Prepare mini‐batches according to guidance method and `guider_input_fields`
        # Each guider_state_batch will have .prompt_embeds, .time_ids, text_embeds, image_embeds.
        # e.g. for CFG, we prepare two batches: one for uncond, one for cond
        # for first batch, guider_state_batch.prompt_embeds correspond to block_state.prompt_embeds
        # for second batch, guider_state_batch.prompt_embeds correspond to block_state.negative_prompt_embeds
        guider_state = components.guider.prepare_inputs(block_state, guider_input_fields)

        # run the denoiser for each guidance batch
        for guider_state_batch in guider_state:
            components.guider.prepare_models(components.transformer)
            cond_kwargs = guider_state_batch.as_dict()
            cond_kwargs = {k: v for k, v in cond_kwargs.items() if k in guider_input_fields}
            prompt_embeds = cond_kwargs.pop("prompt_embeds")

            # Predict the noise residual
            # store the noise_pred in guider_state_batch so that we can apply guidance across all batches
            guider_state_batch.noise_pred = components.transformer(
                hidden_states=block_state.latents.to(transformer_dtype),
                timestep=t.flatten(),
                encoder_hidden_states=prompt_embeds,
                attention_kwargs=block_state.attention_kwargs,
                return_dict=False,
            )[0]
            components.guider.cleanup_models(components.transformer)

        # Perform guidance
        block_state.noise_pred = components.guider(guider_state)[0]

        return components, block_state


class WanLoopAfterDenoiser(ModularPipelineBlocks):
    model_name = "wan"

    @property
    def expected_components(self) -> List[ComponentSpec]:
        return [
            ComponentSpec("scheduler", UniPCMultistepScheduler),
        ]

    @property
    def description(self) -> str:
        return (
            "step within the denoising loop that update the latents. "
            "This block should be used to compose the `sub_blocks` attribute of a `LoopSequentialPipelineBlocks` "
            "object (e.g. `WanDenoiseLoopWrapper`)"
        )

    @property
    def inputs(self) -> List[Tuple[str, Any]]:
        return []

    @property
    def intermediate_inputs(self) -> List[str]:
        return [
            InputParam("generator"),
        ]

    @property
    def intermediate_outputs(self) -> List[OutputParam]:
        return [OutputParam("latents", type_hint=torch.Tensor, description="The denoised latents")]

    @torch.no_grad()
    def __call__(self, components: WanModularPipeline, block_state: BlockState, i: int, t: torch.Tensor):
        # Perform scheduler step using the predicted output
        latents_dtype = block_state.latents.dtype
        block_state.latents = components.scheduler.step(
            block_state.noise_pred.float(),
            t,
            block_state.latents.float(),
            return_dict=False,
        )[0]

        if block_state.latents.dtype != latents_dtype:
            block_state.latents = block_state.latents.to(latents_dtype)

        return components, block_state


class WanDenoiseLoopWrapper(LoopSequentialPipelineBlocks):
    model_name = "wan"

    @property
    def description(self) -> str:
        return (
            "Pipeline block that iteratively denoise the latents over `timesteps`. "
            "The specific steps with each iteration can be customized with `sub_blocks` attributes"
        )

    @property
    def loop_expected_components(self) -> List[ComponentSpec]:
        return [
            ComponentSpec(
                "guider",
                ClassifierFreeGuidance,
                config=FrozenDict({"guidance_scale": 5.0}),
                default_creation_method="from_config",
            ),
            ComponentSpec("scheduler", UniPCMultistepScheduler),
            ComponentSpec("transformer", WanTransformer3DModel),
        ]

    @property
    def loop_intermediate_inputs(self) -> List[InputParam]:
        return [
            InputParam(
                "timesteps",
                required=True,
                type_hint=torch.Tensor,
                description="The timesteps to use for the denoising process. Can be generated in set_timesteps step.",
            ),
            InputParam(
                "num_inference_steps",
                required=True,
                type_hint=int,
                description="The number of inference steps to use for the denoising process. Can be generated in set_timesteps step.",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components: WanModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        block_state.num_warmup_steps = max(
            len(block_state.timesteps) - block_state.num_inference_steps * components.scheduler.order, 0
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


class WanDenoiseStep(WanDenoiseLoopWrapper):
    block_classes = [
        WanLoopDenoiser,
        WanLoopAfterDenoiser,
    ]
    block_names = ["before_denoiser", "denoiser", "after_denoiser"]

    @property
    def description(self) -> str:
        return (
            "Denoise step that iteratively denoise the latents. \n"
            "Its loop logic is defined in `WanDenoiseLoopWrapper.__call__` method \n"
            "At each iteration, it runs blocks defined in `sub_blocks` sequencially:\n"
            " - `WanLoopDenoiser`\n"
            " - `WanLoopAfterDenoiser`\n"
            "This block supports both text2vid tasks."
        )
