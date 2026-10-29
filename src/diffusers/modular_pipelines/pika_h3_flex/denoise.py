# Copyright 2026 Mellis, Inc. (Pika), The MiniMax and HuggingFace Teams. All rights reserved.
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
from .modular_pipeline import PikaH3FlexModularPipeline


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# Copied from diffusers.modular_pipelines.minimax_h3.denoise.MiniMaxH3LoopDenoiser with \bMiniMaxH3->PikaH3Flex, minimax-h3->pika-h3-flex, PikaH3FlexTransformer3DModel->MiniMaxH3Transformer3DModel
class PikaH3FlexLoopDenoiser(ModularPipelineBlocks):
    model_name = "pika-h3-flex"

    def __init__(self, transformer_name: str = "transformer"):
        r"""
        Run the one MiniMax-H3 forward pass of a denoising iteration.

        Args:
            transformer_name (`str`, defaults to `"transformer"`):
                The component the forward runs against. One repository holds both checkpoint partitions —
                `transformer/` for `t2va`/`fl2va` and `transformer_ref/` for `ref2va` — under different component
                names, so which partition a loop drives is configuration, not a separate block.
        """
        self.transformer_name = transformer_name
        super().__init__()

    @property
    def description(self) -> str:
        return (
            "Runs the one MiniMax-H3 forward pass of a denoising iteration, which predicts the velocity of every row "
            "of the packed sequence at once. The checkpoint is guidance-distilled, so there is no unconditional pass "
            "and no guider."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec(self.transformer_name, MiniMaxH3Transformer3DModel)]

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
                    "`position_ids` and the three row-index tensors."
                ),
            ),
            InputParam.template("attention_kwargs"),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "noise_pred",
                type_hint=torch.Tensor,
                description="Predicted velocity of the video rows of the sequence.",
            ),
            OutputParam(
                "audio_noise_pred",
                type_hint=torch.Tensor,
                description="Predicted velocity of the audio rows of the sequence.",
            ),
        ]

    @torch.no_grad()
    def __call__(
        self, components: PikaH3FlexModularPipeline, block_state: BlockState, i: int, t: torch.Tensor
    ) -> tuple[PikaH3FlexModularPipeline, BlockState]:
        transformer = getattr(components, self.transformer_name)
        unique_timesteps, timestep_indices = block_state.row_timestep_plan[i]
        # The layout tags its outputs `denoiser_input_fields`, and their names are the transformer's own parameter
        # names, so the rows of the packed sequence are described to it without this block enumerating them.
        layout_kwargs = {
            name: value
            for name, value in block_state.denoiser_input_fields.items()
            if name in inspect.signature(transformer.forward).parameters
        }
        block_state.noise_pred, block_state.audio_noise_pred = transformer(
            hidden_states=block_state.latents[None],
            audio_hidden_states=block_state.audio_latents[None],
            encoder_hidden_states=block_state.prompt_embeds,
            timestep=unique_timesteps,
            timestep_indices=timestep_indices,
            attention_kwargs=block_state.attention_kwargs,
            return_dict=False,
            **layout_kwargs,
        )
        return components, block_state


# Copied from diffusers.modular_pipelines.minimax_h3.denoise.MiniMaxH3Ref2VALoopDenoiser with \bMiniMaxH3->PikaH3Flex, minimax-h3->pika-h3-flex
class PikaH3FlexRef2VALoopDenoiser(PikaH3FlexLoopDenoiser):
    model_name = "pika-h3-flex"

    def __init__(self):
        super().__init__(transformer_name="transformer_ref")


class PikaH3FlexLoopSchedulerStep(ModularPipelineBlocks):
    model_name = "pika-h3-flex"

    @property
    def description(self) -> str:
        return (
            "Re-noises the generated video and audio rows onto the next sigma of their own schedule: each scheduler "
            "turns the velocity into the clean prediction and, unless this was the last step, mixes it with fresh "
            "noise drawn from the request's generator — the video rows first, then the audio rows. The "
            "conditioning rows are re-imposed by construction: only the generated rows are ever written, so the "
            "anchors survive the whole loop."
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
            InputParam.template(
                "generator",
                description=(
                    "The generator of the request. Every step but the last draws the fresh noise of the next sigma "
                    "from it, the video rows' first and the audio rows' second."
                ),
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
    def __call__(
        self, components: PikaH3FlexModularPipeline, block_state: BlockState, i: int, t: torch.Tensor
    ) -> tuple[PikaH3FlexModularPipeline, BlockState]:
        num_condition_video_rows = block_state.num_condition_video_rows
        num_condition_audio_rows = block_state.num_condition_audio_rows

        block_state.latents[num_condition_video_rows:] = components.scheduler.step(
            block_state.noise_pred[0, num_condition_video_rows:].float(),
            t,
            block_state.latents[num_condition_video_rows:],
            generator=block_state.generator,
            return_dict=False,
        )[0]
        block_state.audio_latents[num_condition_audio_rows:] = components.audio_scheduler.step(
            block_state.audio_noise_pred[0, num_condition_audio_rows:].float(),
            block_state.audio_timesteps[i],
            block_state.audio_latents[num_condition_audio_rows:],
            generator=block_state.generator,
            return_dict=False,
        )[0]
        return components, block_state


# Copied from diffusers.modular_pipelines.minimax_h3.denoise.MiniMaxH3DenoiseLoopWrapper with \bMiniMaxH3->PikaH3Flex, minimax-h3->pika-h3-flex, PikaH3FlexScheduler->MiniMaxH3Scheduler
class PikaH3FlexDenoiseLoopWrapper(LoopSequentialPipelineBlocks):
    model_name = "pika-h3-flex"

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
    def __call__(
        self, components: PikaH3FlexModularPipeline, state: PipelineState
    ) -> tuple[PikaH3FlexModularPipeline, PipelineState]:
        block_state = self.get_block_state(state)
        with self.progress_bar(total=len(block_state.timesteps)) as progress_bar:
            for i, t in enumerate(block_state.timesteps):
                components, block_state = self.loop_step(components, block_state, i=i, t=t)
                progress_bar.update()
        self.set_block_state(state, block_state)
        return components, state


# Copied from diffusers.modular_pipelines.minimax_h3.denoise.MiniMaxH3DenoiseStep with \bMiniMaxH3->PikaH3Flex
class PikaH3FlexDenoiseStep(PikaH3FlexDenoiseLoopWrapper):
    block_classes = [PikaH3FlexLoopDenoiser, PikaH3FlexLoopSchedulerStep]
    block_names = ["denoiser", "update"]

    @property
    def description(self) -> str:
        return "Runs the `t2va` / `fl2va` MiniMax-H3 denoising loop, one forward pass per step."


# Copied from diffusers.modular_pipelines.minimax_h3.denoise.MiniMaxH3Ref2VADenoiseStep with \bMiniMaxH3->PikaH3Flex, minimax-h3->pika-h3-flex
class PikaH3FlexRef2VADenoiseStep(PikaH3FlexDenoiseLoopWrapper):
    model_name = "pika-h3-flex"
    block_classes = [PikaH3FlexRef2VALoopDenoiser, PikaH3FlexLoopSchedulerStep]
    block_names = ["denoiser", "update"]

    @property
    def description(self) -> str:
        return "Runs the `ref2va` MiniMax-H3 denoising loop, one forward pass per step."
