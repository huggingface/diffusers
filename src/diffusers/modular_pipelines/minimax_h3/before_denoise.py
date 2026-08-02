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

from ...schedulers import MiniMaxH3Scheduler
from ...utils import logging
from ...utils.torch_utils import randn_tensor
from ..modular_pipeline import ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam
from .modular_pipeline import MiniMaxH3ModularPipeline, MiniMaxH3Ref2VAModularPipeline
from .packing import (
    MINIMAX_H3_AUDIO_CHANNELS,
    MINIMAX_H3_KEYFRAME_NOISE_AUG,
    MiniMaxH3PackedSequence,
    build_packed_sequence,
    build_row_timesteps,
    patchify_video_latents,
)
from .packing_ref2va import MiniMaxH3PreparedReference, build_ref2va_packed_sequence


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def _layout_inputs() -> list[InputParam]:
    r"""What both packed layouts are built from, beyond the conditioning of the task itself."""
    return [
        InputParam(
            name="text_token_tags",
            type_hint=torch.Tensor,
            required=True,
            description="The per-row modality tag of every row of `prompt_embeds`.",
        ),
        InputParam(
            name="num_latent_frames", type_hint=int, required=True, description="Number of video latent frames."
        ),
        InputParam(name="latent_height", type_hint=int, required=True, description="Height of the video latents."),
        InputParam(name="latent_width", type_hint=int, required=True, description="Width of the video latents."),
        InputParam(
            name="num_audio_latents",
            type_hint=int,
            required=True,
            description="Number of audio latents per channel.",
        ),
    ]


def _layout_outputs() -> list[OutputParam]:
    r"""The row layout of the packed sequence, shared by the two tasks."""
    return [
        OutputParam(
            "layout",
            type_hint=MiniMaxH3PackedSequence,
            description="The structural description of the packed sequence.",
        ),
        OutputParam(
            "position_ids",
            type_hint=torch.Tensor,
            description="The `(t, h, w)` rotary coordinate of every row, in float64.",
        ),
        OutputParam("token_tags", type_hint=torch.Tensor, description="The modality tag of every row."),
        OutputParam(
            "video_indices",
            type_hint=torch.Tensor,
            description="Sequence positions of the video rows, conditioning rows first.",
        ),
        OutputParam(
            "audio_indices",
            type_hint=torch.Tensor,
            description="Sequence positions of the audio rows, reference rows first.",
        ),
        OutputParam("text_indices", type_hint=torch.Tensor, description="Sequence positions of the text rows."),
        OutputParam(
            "num_condition_video_rows",
            type_hint=int,
            description="How many leading video rows are conditioning rows rather than generated rows.",
        ),
        OutputParam(
            "num_condition_audio_rows",
            type_hint=int,
            description="How many leading audio rows are reference rows rather than generated rows.",
        ),
    ]


def _set_layout_state(block_state, layout: MiniMaxH3PackedSequence, device: torch.device) -> None:
    block_state.layout = layout
    block_state.position_ids = layout.position_ids.to(device)
    block_state.token_tags = layout.token_tags.to(device)
    block_state.video_indices = layout.video_indices.to(device)
    block_state.audio_indices = layout.audio_indices.to(device)
    block_state.text_indices = layout.text_indices.to(device)
    block_state.num_condition_video_rows = layout.num_condition_video_rows
    block_state.num_condition_audio_rows = layout.num_condition_audio_rows


class MiniMaxH3PrepareLayoutStep(ModularPipelineBlocks):
    model_name = "minimax-h3"

    @property
    def description(self) -> str:
        return (
            "Builds the packed layout of a `t2va` / `fl2va` request — `[text | keyframe conditions | target audio | "
            "target video]` — and its fp64 rotary grid. MiniMax-H3 runs full self-attention over this one sequence, "
            "so the layout is what every later block addresses rows through."
        )

    @property
    def inputs(self) -> list[InputParam]:
        return [
            *_layout_inputs(),
            InputParam(
                name="keyframe_anchors",
                type_hint=tuple,
                default=(),
                description="Which end of the video every keyframe is anchored to, in packed order.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return _layout_outputs()

    @torch.no_grad()
    def __call__(self, components: MiniMaxH3ModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        layout = build_packed_sequence(
            block_state.text_token_tags,
            block_state.num_latent_frames,
            block_state.latent_height,
            block_state.latent_width,
            block_state.num_audio_latents,
            components.patch_size,
            block_state.keyframe_anchors,
        )
        _set_layout_state(block_state, layout, components._execution_device)

        self.set_block_state(state, block_state)
        return components, state


class MiniMaxH3Ref2VAPrepareLayoutStep(ModularPipelineBlocks):
    model_name = "minimax-h3-ref2va"

    @property
    def description(self) -> str:
        return (
            "Builds the packed layout of a `ref2va` request — `[text | reference blocks | target audio | target "
            "video]` — and its fp64 rotary grid. The reference order advances the shared audio/video rotary clock, so "
            "it is part of the layout rather than a detail of the presentation."
        )

    @property
    def inputs(self) -> list[InputParam]:
        return [
            *_layout_inputs(),
            InputParam(
                name="prepared_references",
                type_hint=list[MiniMaxH3PreparedReference],
                required=True,
                description="The prepared references, in packed order, with their latent geometry filled in.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return _layout_outputs()

    @torch.no_grad()
    def __call__(self, components: MiniMaxH3Ref2VAModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        layout = build_ref2va_packed_sequence(
            block_state.text_token_tags,
            block_state.prepared_references,
            block_state.num_latent_frames,
            block_state.latent_height,
            block_state.latent_width,
            block_state.num_audio_latents,
            components.patch_size,
        )
        _set_layout_state(block_state, layout, components._execution_device)

        self.set_block_state(state, block_state)
        return components, state


class MiniMaxH3PrepareLatentsStep(ModularPipelineBlocks):
    model_name = "minimax-h3"

    @property
    def description(self) -> str:
        return (
            "Draws the initial noise of the generated rows and prepends the conditioning rows. MiniMax-H3 draws the "
            "video noise as a latent tensor and patchifies it afterwards, then the audio noise directly in row "
            "layout — both off the request's generator, after the conditioning noise of the encoder step."
        )

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                name="num_latent_frames", type_hint=int, required=True, description="Number of video latent frames."
            ),
            InputParam(name="latent_height", type_hint=int, required=True, description="Height of the video latents."),
            InputParam(name="latent_width", type_hint=int, required=True, description="Width of the video latents."),
            InputParam(
                name="num_audio_latents",
                type_hint=int,
                required=True,
                description="Number of audio latents per channel.",
            ),
            InputParam.template(
                "generator",
                description=(
                    "The generator of the request. The video noise is drawn from it first, then the audio noise."
                ),
            ),
            InputParam(
                name="latents",
                type_hint=torch.Tensor,
                description=(
                    "Pre-generated video noise of shape `(1, 24, num_latent_frames, latent_height, latent_width)`, "
                    "used instead of the draw."
                ),
            ),
            InputParam(
                name="audio_latents",
                type_hint=torch.Tensor,
                description="Pre-generated audio noise of shape `(2, 32, num_audio_latents)`.",
            ),
            InputParam(
                name="condition_latents",
                type_hint=torch.Tensor,
                description="The video conditioning rows to prepend, or None for a request that has none.",
            ),
            InputParam(
                name="audio_condition_latents",
                type_hint=torch.Tensor,
                description="The audio conditioning rows to prepend, or None for a request that has none.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "latents",
                type_hint=torch.Tensor,
                description="The video rows of the packed sequence, conditioning rows first.",
            ),
            OutputParam(
                "audio_latents",
                type_hint=torch.Tensor,
                description="The channel-major audio rows of the packed sequence, reference rows first.",
            ),
        ]

    @staticmethod
    def prepare_latents(
        components,
        num_latent_frames: int,
        latent_height: int,
        latent_width: int,
        num_audio_latents: int,
        device: torch.device,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        audio_latents: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""
        Draw the initial noise of both modalities and pack it into transformer rows.

        A request draws every stream from the one generator it is given, and the order is part of what that generator
        reproduces: the conditioning noise of the keyframes or references first (one draw per condition, in
        [`~modular_pipelines.minimax_h3.packing.keyframe_condition_noise`]), then the video noise here, as a latent tensor
        that is patchified afterwards, then the audio noise, directly in row layout. Passing `latents` or
        `audio_latents` skips its draw and shifts the ones after it.

        Args:
            num_latent_frames (`int`): Number of video latent frames.
            latent_height (`int`): Latent height.
            latent_width (`int`): Latent width.
            num_audio_latents (`int`): Number of audio latents per channel.
            device (`torch.device`): The device the rows are drawn on.
            generator (`torch.Generator`, *optional*): The generator of the request.
            latents (`torch.Tensor`, *optional*):
                Pre-generated video noise of shape `(1, latent_channels, num_latent_frames, latent_height,
                latent_width)`, used instead of the draw.
            audio_latents (`torch.Tensor`, *optional*):
                Pre-generated audio noise of shape `(2, audio_latent_channels, num_audio_latents)`.

        Returns:
            `tuple[torch.Tensor, torch.Tensor]`: the video rows and the channel-major audio rows.
        """
        if latents is None:
            latents = randn_tensor(
                (1, components.vae_latent_channels, num_latent_frames, latent_height, latent_width),
                generator=generator,
                device=device,
                dtype=torch.float32,
            )
        video_rows = patchify_video_latents(latents.to(torch.float32), components.patch_size)

        if audio_latents is None:
            audio_rows = randn_tensor(
                (num_audio_latents * MINIMAX_H3_AUDIO_CHANNELS, components.audio_latent_channels),
                generator=generator,
                device=device,
                dtype=torch.float32,
            )
        else:
            audio_rows = audio_latents.to(torch.float32).permute(0, 2, 1).reshape(-1, components.audio_latent_channels)
        return video_rows.to(device), audio_rows.to(device)

    @torch.no_grad()
    def __call__(self, components: MiniMaxH3ModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        latents, audio_latents = self.prepare_latents(
            components,
            block_state.num_latent_frames,
            block_state.latent_height,
            block_state.latent_width,
            block_state.num_audio_latents,
            components._execution_device,
            block_state.generator,
            block_state.latents,
            block_state.audio_latents,
        )
        if block_state.condition_latents is not None:
            latents = torch.cat([block_state.condition_latents, latents])
        if block_state.audio_condition_latents is not None:
            audio_latents = torch.cat([block_state.audio_condition_latents, audio_latents])
        block_state.latents, block_state.audio_latents = latents, audio_latents

        self.set_block_state(state, block_state)
        return components, state


class MiniMaxH3SetTimestepsStep(ModularPipelineBlocks):
    model_name = "minimax-h3"

    @property
    def description(self) -> str:
        return (
            "Initializes the two schedules — `shift = 12.0` for video, `shift = 3.0` for audio — and stages the "
            "row-to-timestep plan of every step. One forward serves every modality and every noise level at once: "
            "the generated rows step down their own schedule while the conditioning rows stay pinned at their "
            "noise-augmentation level, and that assignment is static per step."
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
            InputParam.template("num_inference_steps", required=True),
            InputParam(
                name="layout",
                type_hint=MiniMaxH3PackedSequence,
                required=True,
                description="The structural description of the packed sequence.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("timesteps", type_hint=torch.Tensor, description="Timesteps of the video schedule."),
            OutputParam("audio_timesteps", type_hint=torch.Tensor, description="Timesteps of the audio schedule."),
            OutputParam(
                "row_timestep_plan",
                type_hint=list,
                description=(
                    "One `(timestep, timestep_indices)` pair per step: the distinct timesteps of the sequence and the "
                    "index of every row into them."
                ),
            ),
        ]

    @torch.no_grad()
    def __call__(self, components: MiniMaxH3ModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        device = components._execution_device

        components.scheduler.set_timesteps(block_state.num_inference_steps, device=device)
        components.audio_scheduler.set_timesteps(block_state.num_inference_steps, device=device)
        block_state.timesteps = components.scheduler.timesteps
        block_state.audio_timesteps = components.audio_scheduler.timesteps

        block_state.row_timestep_plan = [
            tuple(
                tensor.to(device)
                for tensor in build_row_timesteps(
                    block_state.layout,
                    float(timestep),
                    float(audio_timestep),
                    max(float(timestep), MINIMAX_H3_KEYFRAME_NOISE_AUG),
                    1.0,
                )
            )
            for timestep, audio_timestep in zip(block_state.timesteps, block_state.audio_timesteps)
        ]

        self.set_block_state(state, block_state)
        return components, state
