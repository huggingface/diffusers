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

import torch

from ...models import MiniMaxH3Transformer3DModel
from ...schedulers import MiniMaxH3Scheduler
from ...utils import logging
from ..modular_pipeline import (
    BlockState,
    LoopSequentialPipelineBlocks,
    ModularPipelineBlocks,
    PipelineState,
)
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam
from .modular_pipeline import MiniMaxH3ModularPipeline, MiniMaxH3Ref2VAModularPipeline


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def _denoiser_inputs() -> list[InputParam]:
    r"""Everything one MiniMax-H3 forward reads, beyond the transformer itself."""
    return [
        InputParam(
            name="latents",
            type_hint=torch.Tensor,
            required=True,
            description="The video rows of the packed sequence, conditioning rows first.",
        ),
        InputParam(
            name="audio_latents",
            type_hint=torch.Tensor,
            required=True,
            description="The channel-major audio rows of the packed sequence, reference rows first.",
        ),
        InputParam.template("prompt_embeds"),
        InputParam(
            name="row_timestep_plan",
            type_hint=list,
            required=True,
            description="One `(timestep, timestep_indices)` pair per step.",
        ),
        InputParam(
            name="token_tags", type_hint=torch.Tensor, required=True, description="The modality tag of every row."
        ),
        InputParam(
            name="position_ids",
            type_hint=torch.Tensor,
            required=True,
            description="The `(t, h, w)` rotary coordinate of every row.",
        ),
        InputParam(
            name="video_indices",
            type_hint=torch.Tensor,
            required=True,
            description="Sequence positions of the video rows.",
        ),
        InputParam(
            name="audio_indices",
            type_hint=torch.Tensor,
            required=True,
            description="Sequence positions of the audio rows.",
        ),
        InputParam(
            name="text_indices",
            type_hint=torch.Tensor,
            required=True,
            description="Sequence positions of the text rows.",
        ),
        InputParam.template("attention_kwargs"),
    ]


def _denoiser_outputs() -> list[OutputParam]:
    return [
        OutputParam(
            "noise_pred", type_hint=torch.Tensor, description="Predicted velocity of the video rows of the sequence."
        ),
        OutputParam(
            "audio_noise_pred",
            type_hint=torch.Tensor,
            description="Predicted velocity of the audio rows of the sequence.",
        ),
    ]


def _predict_velocity(transformer: MiniMaxH3Transformer3DModel, block_state: BlockState, i: int):
    r"""One MiniMax-H3 forward pass: every row of the packed sequence, at its own noise level, at once."""
    unique_timesteps, timestep_indices = block_state.row_timestep_plan[i]
    return transformer(
        hidden_states=block_state.latents[None],
        audio_hidden_states=block_state.audio_latents[None],
        encoder_hidden_states=block_state.prompt_embeds,
        timestep=unique_timesteps,
        timestep_indices=timestep_indices,
        token_tags=block_state.token_tags,
        position_ids=block_state.position_ids,
        video_indices=block_state.video_indices,
        audio_indices=block_state.audio_indices,
        text_indices=block_state.text_indices,
        attention_kwargs=block_state.attention_kwargs,
        return_dict=False,
    )


class MiniMaxH3LoopDenoiser(ModularPipelineBlocks):
    model_name = "minimax-h3"

    @property
    def description(self) -> str:
        return (
            "Runs the one MiniMax-H3 forward pass of a denoising iteration, which predicts the velocity of every row "
            "of the packed sequence at once. The checkpoint is guidance-distilled, so there is no unconditional pass "
            "and no guider."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("transformer", MiniMaxH3Transformer3DModel)]

    @property
    def inputs(self) -> list[InputParam]:
        return _denoiser_inputs()

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return _denoiser_outputs()

    @torch.no_grad()
    def __call__(self, components: MiniMaxH3ModularPipeline, block_state: BlockState, i: int, t: torch.Tensor):
        block_state.noise_pred, block_state.audio_noise_pred = _predict_velocity(
            components.transformer, block_state, i
        )
        return components, block_state


class MiniMaxH3Ref2VALoopDenoiser(ModularPipelineBlocks):
    model_name = "minimax-h3-ref2va"

    @property
    def description(self) -> str:
        return (
            "Runs the one MiniMax-H3 forward pass of a `ref2va` denoising iteration, against the `transformer_ref` "
            "partition of the checkpoint."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("transformer_ref", MiniMaxH3Transformer3DModel)]

    @property
    def inputs(self) -> list[InputParam]:
        return _denoiser_inputs()

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return _denoiser_outputs()

    @torch.no_grad()
    def __call__(self, components: MiniMaxH3Ref2VAModularPipeline, block_state: BlockState, i: int, t: torch.Tensor):
        block_state.noise_pred, block_state.audio_noise_pred = _predict_velocity(
            components.transformer_ref, block_state, i
        )
        return components, block_state


class MiniMaxH3LoopSchedulerStep(ModularPipelineBlocks):
    model_name = "minimax-h3"

    @property
    def description(self) -> str:
        return (
            "Steps the generated video and audio rows down their own schedule. The conditioning rows are re-imposed "
            "by construction: only the generated rows are ever written, so the anchors survive the whole loop."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("scheduler", MiniMaxH3Scheduler),
            ComponentSpec("audio_scheduler", MiniMaxH3Scheduler),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                name="latents",
                type_hint=torch.Tensor,
                required=True,
                description="The video rows of the packed sequence, conditioning rows first.",
            ),
            InputParam(
                name="audio_latents",
                type_hint=torch.Tensor,
                required=True,
                description="The channel-major audio rows of the packed sequence, reference rows first.",
            ),
            InputParam(
                name="noise_pred",
                type_hint=torch.Tensor,
                required=True,
                description="Predicted velocity of the video rows.",
            ),
            InputParam(
                name="audio_noise_pred",
                type_hint=torch.Tensor,
                required=True,
                description="Predicted velocity of the audio rows.",
            ),
            InputParam(
                name="audio_timesteps",
                type_hint=torch.Tensor,
                required=True,
                description="Timesteps of the audio schedule.",
            ),
            InputParam(
                name="num_condition_video_rows",
                type_hint=int,
                default=0,
                description="How many leading video rows are conditioning rows.",
            ),
            InputParam(
                name="num_condition_audio_rows",
                type_hint=int,
                default=0,
                description="How many leading audio rows are reference rows.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "latents",
                type_hint=torch.Tensor,
                description="The video rows of the packed sequence after one step.",
            ),
            OutputParam(
                "audio_latents",
                type_hint=torch.Tensor,
                description="The audio rows of the packed sequence after one step.",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components: MiniMaxH3ModularPipeline, block_state: BlockState, i: int, t: torch.Tensor):
        num_condition_video_rows = block_state.num_condition_video_rows
        num_condition_audio_rows = block_state.num_condition_audio_rows

        block_state.latents[num_condition_video_rows:] = components.scheduler.step(
            block_state.noise_pred[0, num_condition_video_rows:].float(),
            t,
            block_state.latents[num_condition_video_rows:],
            return_dict=False,
        )[0]
        block_state.audio_latents[num_condition_audio_rows:] = components.audio_scheduler.step(
            block_state.audio_noise_pred[0, num_condition_audio_rows:].float(),
            block_state.audio_timesteps[i],
            block_state.audio_latents[num_condition_audio_rows:],
            return_dict=False,
        )[0]
        return components, block_state


class MiniMaxH3DenoiseLoopWrapper(LoopSequentialPipelineBlocks):
    model_name = "minimax-h3"

    @property
    def description(self) -> str:
        return "Iteratively denoises the packed MiniMax-H3 sequence over the two schedules."

    @property
    def loop_expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("scheduler", MiniMaxH3Scheduler),
            ComponentSpec("audio_scheduler", MiniMaxH3Scheduler),
        ]

    @property
    def loop_inputs(self) -> list[InputParam]:
        return [
            InputParam.template("timesteps", required=True, description="Timesteps of the video schedule."),
        ]

    @torch.no_grad()
    def __call__(self, components: MiniMaxH3ModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        with self.progress_bar(total=len(block_state.timesteps)) as progress_bar:
            for i, t in enumerate(block_state.timesteps):
                components, block_state = self.loop_step(components, block_state, i=i, t=t)
                progress_bar.update()
        self.set_block_state(state, block_state)
        return components, state


class MiniMaxH3DenoiseStep(MiniMaxH3DenoiseLoopWrapper):
    block_classes = [MiniMaxH3LoopDenoiser, MiniMaxH3LoopSchedulerStep]
    block_names = ["denoiser", "update"]

    @property
    def description(self) -> str:
        return "Runs the `t2va` / `fl2va` MiniMax-H3 denoising loop, one forward pass per step."


class MiniMaxH3Ref2VADenoiseStep(MiniMaxH3DenoiseLoopWrapper):
    model_name = "minimax-h3-ref2va"
    block_classes = [MiniMaxH3Ref2VALoopDenoiser, MiniMaxH3LoopSchedulerStep]
    block_names = ["denoiser", "update"]

    @property
    def description(self) -> str:
        return "Runs the `ref2va` MiniMax-H3 denoising loop, one forward pass per step."
