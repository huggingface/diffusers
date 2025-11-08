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

from typing import Any, List, Tuple, Dict

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
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam, ConfigSpec
from .modular_pipeline import WanModularPipeline


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class WanLoopBeforeDenoiser(ModularPipelineBlocks):
    model_name = "wan"

    @property
    def description(self) -> str:
        return (
            "step within the denoising loop that prepares the latent input for the denoiser. "
            "This block should be used to compose the `sub_blocks` attribute of a `LoopSequentialPipelineBlocks` "
            "object (e.g. `WanDenoiseLoopWrapper`)"
        )

    @property
    def inputs(self) -> List[InputParam]:
        return [
            InputParam(
                "latents",
                required=True,
                type_hint=torch.Tensor,
                description="The initial latents to use for the denoising process. Can be generated in prepare_latent step.",
            ),
        ]
    

    @torch.no_grad()
    def __call__(self, components: WanModularPipeline, block_state: BlockState, i: int, t: torch.Tensor):
        block_state.latent_model_input = block_state.latents
        return components, block_state


class WanImage2VideoLoopBeforeDenoiser(ModularPipelineBlocks):
    model_name = "wan"

    @property
    def description(self) -> str:
        return (
            "step within the denoising loop that prepares the latent input for the denoiser. "
            "This block should be used to compose the `sub_blocks` attribute of a `LoopSequentialPipelineBlocks` "
            "object (e.g. `WanDenoiseLoopWrapper`)"
        )

    @property
    def inputs(self) -> List[InputParam]:
        return [
            InputParam(
                "latents",
                required=True,
                type_hint=torch.Tensor,
                description="The initial latents to use for the denoising process. Can be generated in prepare_latent step.",
            ),
            InputParam(
                "condition_latents",
                required=True,
                type_hint=torch.Tensor,
                description="The condition latents to use for the denoising process. Can be generated in prepare_condition_latents step.",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components: WanModularPipeline, block_state: BlockState, i: int, t: torch.Tensor):
        block_state.latent_model_input = torch.cat([block_state.latents, block_state.condition_latents], dim=1)
        return components, block_state

class WanLoopDenoiserDynamic(ModularPipelineBlocks):
    model_name = "wan"

    #  guider_input_fields maps the keys we'll see on each `guider_state_batch` (e.g. guider_state_batch.encoder_hidden_states)
     # to the corresponding (cond, uncond) fields on block_state. (e.g. block_state.prompt_embeds, block_state.negative_prompt_embeds)
    def __init__(self, guider_input_fields: Dict[str, Any] = {"encoder_hidden_states": ("prompt_embeds", "negative_prompt_embeds")}):
        if not isinstance(guider_input_fields, dict):
            raise ValueError(f"guider_input_fields must be a dictionary but is {type(guider_input_fields)}")
        self._guider_input_fields = guider_input_fields
        super().__init__()

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

        inputs = [
            InputParam("attention_kwargs"),
            InputParam(
                "num_inference_steps",
                required=True,
                type_hint=int,
                description="The number of inference steps to use for the denoising process. Can be generated in set_timesteps step.",
            ),
        ]
        guider_input_names = []
        for value in self._guider_input_fields.values():
            if isinstance(value, tuple):
                guider_input_names.extend(value)
            else:
                guider_input_names.append(value)

        for name in guider_input_names:
            inputs.append(InputParam(name=name, required=True, type_hint=torch.Tensor))
        return inputs


    @torch.no_grad()
    def __call__(
        self, components: WanModularPipeline, block_state: BlockState, i: int, t: torch.Tensor
    ) -> PipelineState:

        transformer_dtype = components.transformer.dtype

        components.guider.set_state(step=i, num_inference_steps=block_state.num_inference_steps, timestep=t)

        # The guider splits model inputs into separate batches for conditional/unconditional predictions.
        # For CFG with guider_inputs = {"encoder_hidden_states": (prompt_embeds, negative_prompt_embeds)}:
        # you will get a guider_state with two batches:
        #   guider_state = [
        #       {"encoder_hidden_states": prompt_embeds, "__guidance_identifier__": "pred_cond"},      # conditional batch
        #       {"encoder_hidden_states": negative_prompt_embeds, "__guidance_identifier__": "pred_uncond"},  # unconditional batch
        #   ]
        # Other guidance methods may return 1 batch (no guidance) or 3+ batches (e.g., PAG, APG).
        guider_state = components.guider.prepare_inputs_from_block_state(block_state, self._guider_input_fields)

        # run the denoiser for each guidance batch
        for guider_state_batch in guider_state:
            components.guider.prepare_models(components.transformer)
            cond_kwargs = guider_state_batch.as_dict()
            cond_kwargs = {k: v for k, v in cond_kwargs.items() if k in self._guider_input_fields.keys()}

            # Predict the noise residual
            # store the noise_pred in guider_state_batch so that we can apply guidance across all batches
            guider_state_batch.noise_pred = components.transformer(
                hidden_states=block_state.latent_model_input.to(transformer_dtype),
                timestep=t.expand(block_state.latent_model_input.shape[0]).to(block_state.latent_model_input.dtype),
                attention_kwargs=block_state.attention_kwargs,
                return_dict=False,
                **cond_kwargs,
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
            ComponentSpec("scheduler", UniPCMultistepScheduler),
        ]

    @property
    def loop_inputs(self) -> List[InputParam]:
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
        WanLoopBeforeDenoiser,
        WanLoopDenoiserDynamic(
            guider_input_fields={
                "encoder_hidden_states": ("prompt_embeds", "negative_prompt_embeds"),
            }
        ),
        WanLoopAfterDenoiser,
    ]
    block_names = ["before_denoiser", "denoiser", "after_denoiser"]

    @property
    def description(self) -> str:
        return (
            "Denoise step that iteratively denoise the latents. \n"
            "Its loop logic is defined in `WanDenoiseLoopWrapper.__call__` method \n"
            "At each iteration, it runs blocks defined in `sub_blocks` sequentially:\n"
            " - `WanLoopDenoiser`\n"
            " - `WanLoopAfterDenoiser`\n"
            "This block supports text-to-video tasks."
        )

class WanImage2VideoDenoiseStep(WanDenoiseLoopWrapper):
    block_classes = [
        WanImage2VideoLoopBeforeDenoiser,
        WanLoopDenoiserDynamic(
            guider_input_fields={
                "encoder_hidden_states": ("prompt_embeds", "negative_prompt_embeds"),
                "encoder_hidden_states_image": "image_embeds",
            }
        ),
        WanLoopAfterDenoiser,
    ]
    block_names = ["before_denoiser", "denoiser", "after_denoiser"]

    @property
    def description(self) -> str:
        return (
            "Denoise step that iteratively denoise the latents. \n"
            "Its loop logic is defined in `WanDenoiseLoopWrapper.__call__` method \n"
            "At each iteration, it runs blocks defined in `sub_blocks` sequentially:\n"
            " - `WanLoopDenoiser`\n"
            " - `WanLoopAfterDenoiser`\n"
            "This block supports image-to-video tasks."
        )
