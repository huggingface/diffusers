# Copyright 2026 The MiniMax and HuggingFace Teams. All rights reserved.
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

import inspect

import torch

from ...models import MiniMaxH3ImageTransformer2DModel
from ...schedulers import MiniMaxH3Scheduler
from ...utils import logging
from ..modular_pipeline import (
    BlockState,
    LoopSequentialPipelineBlocks,
    ModularPipelineBlocks,
    PipelineState,
)
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam
from .modular_pipeline import MiniMaxH3ImageModularPipeline


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class MiniMaxH3ImageLoopDenoiser(ModularPipelineBlocks):
    model_name = "minimax-h3-image"

    @property
    def description(self) -> str:
        return (
            "Runs the one transformer forward pass of a denoising iteration, which predicts the velocity of every "
            "image row of the packed sequence through the output head of that step. The checkpoint is "
            "guidance-distilled, so there is no unconditional pass and no guider."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("transformer", MiniMaxH3ImageTransformer2DModel)]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                name="latents",
                type_hint=torch.Tensor,
                required=True,
                description="The image rows of the packed sequence, reference rows first.",
            ),
            InputParam.template("prompt_embeds"),
            InputParam(
                name="row_timestep_plan",
                type_hint=list,
                required=True,
                description="One `(timestep, timestep_indices)` pair per step.",
            ),
            InputParam(
                kwargs_type="denoiser_input_fields",
                description=(
                    "The structural description of the packed sequence the transformer reads by name: `token_tags`, "
                    "`position_ids` and the two row-index tensors."
                ),
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "noise_pred",
                type_hint=torch.Tensor,
                description="Predicted velocity of the image rows of the sequence.",
            ),
        ]

    @torch.no_grad()
    def __call__(
        self, components: MiniMaxH3ImageModularPipeline, block_state: BlockState, i: int, t: torch.Tensor
    ) -> tuple[MiniMaxH3ImageModularPipeline, BlockState]:
        unique_timesteps, timestep_indices = block_state.row_timestep_plan[i]
        # Every field tagged `denoiser_input_fields` whose name is one of the transformer's parameters is passed to
        # it. In the shipped layouts, `position_ids`, `token_tags`, `image_indices` and `text_indices` are tagged
        # this way; they describe the rows of the packed sequence. A custom block can tag further transformer inputs
        # the same way.
        transformer_args = set(inspect.signature(components.transformer.forward).parameters)
        additional_denoiser_kwargs = {
            name: value for name, value in block_state.denoiser_input_fields.items() if name in transformer_args
        }
        # `latents` and `prompt_embeds` go in as `hidden_states` and `encoder_hidden_states`. Any other field passed
        # as `denoiser_input_fields` that the transformer does not take is not used, and we warn about it.
        unused = sorted(set(block_state.denoiser_input_fields) - transformer_args - {"latents", "prompt_embeds"})
        if unused and i == 0:
            logger.warning(f"`denoiser_input_fields` {unused} are not inputs of the transformer and are not used.")
        block_state.noise_pred = components.transformer(
            hidden_states=block_state.latents[None],
            encoder_hidden_states=block_state.prompt_embeds,
            timestep=unique_timesteps,
            timestep_indices=timestep_indices,
            pdd_step_index=i,
            return_dict=False,
            **additional_denoiser_kwargs,
        )[0]
        return components, block_state


class MiniMaxH3ImageLoopSchedulerStep(ModularPipelineBlocks):
    model_name = "minimax-h3-image"

    @property
    def description(self) -> str:
        return (
            "Steps the generated rows down the schedule. Only the generated rows are ever written, so the reference "
            "rows of an edit survive the whole loop."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("scheduler", MiniMaxH3Scheduler)]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                name="latents",
                type_hint=torch.Tensor,
                required=True,
                description="The image rows of the packed sequence, reference rows first.",
            ),
            InputParam(
                name="noise_pred",
                type_hint=torch.Tensor,
                required=True,
                description="Predicted velocity of the image rows.",
            ),
            InputParam(
                name="num_reference_rows",
                type_hint=int,
                default=0,
                description="How many leading image rows are reference rows.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "latents",
                type_hint=torch.Tensor,
                description="The image rows of the packed sequence after one step.",
            ),
        ]

    @torch.no_grad()
    def __call__(
        self, components: MiniMaxH3ImageModularPipeline, block_state: BlockState, i: int, t: torch.Tensor
    ) -> tuple[MiniMaxH3ImageModularPipeline, BlockState]:
        num_reference_rows = block_state.num_reference_rows
        block_state.latents[num_reference_rows:] = components.scheduler.step(
            block_state.noise_pred[0, num_reference_rows:].float(),
            t,
            block_state.latents[num_reference_rows:],
            return_dict=False,
        )[0]
        return components, block_state


class MiniMaxH3ImageDenoiseLoopWrapper(LoopSequentialPipelineBlocks):
    model_name = "minimax-h3-image"

    @property
    def description(self) -> str:
        return "Iteratively denoises the packed MiniMax-H3-Image sequence over its fixed schedule."

    @property
    def loop_expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("scheduler", MiniMaxH3Scheduler)]

    @property
    def loop_inputs(self) -> list[InputParam]:
        return [
            InputParam.template("timesteps", required=True, description="Timesteps of the schedule."),
        ]

    @torch.no_grad()
    def __call__(
        self, components: MiniMaxH3ImageModularPipeline, state: PipelineState
    ) -> tuple[MiniMaxH3ImageModularPipeline, PipelineState]:
        block_state = self.get_block_state(state)
        with self.progress_bar(total=len(block_state.timesteps)) as progress_bar:
            for i, t in enumerate(block_state.timesteps):
                components, block_state = self.loop_step(components, block_state, i=i, t=t)
                progress_bar.update()
        self.set_block_state(state, block_state)
        return components, state


class MiniMaxH3ImageDenoiseStep(MiniMaxH3ImageDenoiseLoopWrapper):
    block_classes = [MiniMaxH3ImageLoopDenoiser, MiniMaxH3ImageLoopSchedulerStep]
    block_names = ["denoiser", "update"]

    @property
    def description(self) -> str:
        return "Runs the MiniMax-H3-Image denoising loop, one forward pass per step."
