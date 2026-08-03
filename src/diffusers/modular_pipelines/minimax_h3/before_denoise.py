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
    align_num_frames,
    audio_latent_num_frames,
    build_packed_sequence,
    build_row_timesteps,
    patchify_video_latents,
    resolve_canvas_size,
    video_latent_num_frames,
)
from .packing_ref2va import MiniMaxH3Reference, build_ref2va_packed_sequence


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class MiniMaxH3PrepareLayoutStep(ModularPipelineBlocks):
    model_name = "minimax-h3"

    @property
    def description(self) -> str:
        return (
            "Resolves the geometry of a `t2va` / `fl2va` request — the canvas, the `17 * n + 5` frame count the video "
            "VAE can decode and the latent shapes every later block keys off — and builds the packed layout from it: "
            "`[text | keyframe conditions | target audio | target video]` plus its fp64 rotary grid. MiniMax-H3 runs "
            "full self-attention over this one sequence, so the layout is what every later block addresses rows "
            "through."
        )

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                name="text_token_tags",
                type_hint=torch.Tensor,
                required=True,
                description="The per-row modality tag of every row of `prompt_embeds`.",
            ),
            InputParam.template("height", description="Height of the generated video in pixels, a multiple of 32."),
            InputParam.template("width", description="Width of the generated video in pixels, a multiple of 32."),
            InputParam(
                name="num_frames",
                type_hint=int,
                default=124,
                description=(
                    "Number of frames to generate, at the fixed 24 fps. Snapped up to the next `17 * n + 5` the video "
                    "VAE can decode; the resulting duration must stay between 5 and 15 seconds."
                ),
            ),
            InputParam(
                name="keyframe_anchors",
                type_hint=tuple,
                default=(),
                description="Which end of the video every keyframe is anchored to, in packed order.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("height", type_hint=int, description="Resolved height of the generated video in pixels."),
            OutputParam("width", type_hint=int, description="Resolved width of the generated video in pixels."),
            OutputParam("num_frames", type_hint=int, description="Resolved number of frames, of the form 17 * n + 5."),
            OutputParam("num_latent_frames", type_hint=int, description="Number of generated video latent frames."),
            OutputParam("latent_height", type_hint=int, description="Height of the generated video latents."),
            OutputParam("latent_width", type_hint=int, description="Width of the generated video latents."),
            OutputParam(
                "num_audio_latents", type_hint=int, description="Number of generated audio latents per channel."
            ),
            OutputParam(
                "position_ids",
                type_hint=torch.Tensor,
                kwargs_type="denoiser_input_fields",
                description="The `(t, h, w)` rotary coordinate of every row, in float64.",
            ),
            OutputParam(
                "token_tags",
                type_hint=torch.Tensor,
                kwargs_type="denoiser_input_fields",
                description="The modality tag of every row.",
            ),
            OutputParam(
                "video_indices",
                type_hint=torch.Tensor,
                kwargs_type="denoiser_input_fields",
                description="Sequence positions of the video rows, conditioning rows first.",
            ),
            OutputParam(
                "audio_indices",
                type_hint=torch.Tensor,
                kwargs_type="denoiser_input_fields",
                description="Sequence positions of the audio rows, reference rows first.",
            ),
            OutputParam(
                "text_indices",
                type_hint=torch.Tensor,
                kwargs_type="denoiser_input_fields",
                description="Sequence positions of the text rows.",
            ),
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

    @torch.no_grad()
    def __call__(self, components: MiniMaxH3ModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        device = components._execution_device

        # Without a keyframe to take the aspect ratio from, MiniMax-H3 generates on its own 16:9 canvas.
        if block_state.height is None:
            block_state.height, block_state.width = resolve_canvas_size(16, 9, components.canvas_multiple)
        if block_state.height % components.canvas_multiple or block_state.width % components.canvas_multiple:
            raise ValueError(
                f"`height` and `width` must be multiples of {components.canvas_multiple}, got "
                f"{block_state.height}x{block_state.width}."
            )

        frames_per_chunk = components.vae_frames_per_chunk
        latents_per_chunk = components.vae_latents_per_chunk
        aligned_num_frames = align_num_frames(block_state.num_frames, frames_per_chunk, latents_per_chunk)
        if aligned_num_frames != block_state.num_frames:
            logger.warning(
                f"`num_frames` has to be of the form 17 * n + 5 for the video VAE; rounding {block_state.num_frames} "
                f"up to {aligned_num_frames}."
            )
            block_state.num_frames = aligned_num_frames
        # The duration the request generates is the one of the *aligned* frame count, so that is what the ceiling has
        # to hold for: 346 frames would otherwise pass the check and then be rounded up to 362, i.e. 15.083 seconds.
        duration = block_state.num_frames / components.fps
        if not components.min_duration <= duration <= components.max_duration:
            raise ValueError(
                f"MiniMax-H3 generates between {components.min_duration} and {components.max_duration} seconds "
                f"at {components.fps} fps, so `num_frames`, rounded up to the next `17 * n + 5` the video VAE "
                f"can encode, must be between {int(components.min_duration * components.fps)} and "
                f"{int(components.max_duration * components.fps)}, got {block_state.num_frames}."
            )

        ratio = components.vae_spatial_compression_ratio
        block_state.num_latent_frames = video_latent_num_frames(
            block_state.num_frames, frames_per_chunk, latents_per_chunk
        )
        block_state.latent_height = block_state.height // ratio
        block_state.latent_width = block_state.width // ratio
        block_state.num_audio_latents = audio_latent_num_frames(block_state.num_frames)

        (
            position_ids,
            token_tags,
            video_indices,
            audio_indices,
            text_indices,
            block_state.num_condition_video_rows,
            block_state.num_condition_audio_rows,
        ) = build_packed_sequence(
            block_state.text_token_tags,
            block_state.num_latent_frames,
            block_state.latent_height,
            block_state.latent_width,
            block_state.num_audio_latents,
            components.patch_size,
            block_state.keyframe_anchors,
        )
        block_state.position_ids = position_ids.to(device)
        block_state.token_tags = token_tags.to(device)
        block_state.video_indices = video_indices.to(device)
        block_state.audio_indices = audio_indices.to(device)
        block_state.text_indices = text_indices.to(device)

        self.set_block_state(state, block_state)
        return components, state


class MiniMaxH3Ref2VAPrepareLayoutStep(ModularPipelineBlocks):
    model_name = "minimax-h3-ref2va"

    @property
    def description(self) -> str:
        return (
            "Resolves the latent shapes of a `ref2va` request and builds its packed layout — `[text | reference "
            "blocks | target audio | target video]` — plus its fp64 rotary grid. The reference order advances the "
            "shared audio/video rotary clock, so it is part of the layout rather than a detail of the presentation."
        )

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                name="text_token_tags",
                type_hint=torch.Tensor,
                required=True,
                description="The per-row modality tag of every row of `prompt_embeds`.",
            ),
            InputParam(
                name="prepared_references",
                type_hint=list[MiniMaxH3Reference],
                required=True,
                description="The references normalized by the setup step, in packed order.",
            ),
            InputParam(
                name="condition_latents",
                type_hint=list[torch.Tensor],
                required=True,
                description=(
                    "The encoded video conditioning latents, one per image and video reference in packed order. "
                    "Their shape is where every reference block's geometry comes from."
                ),
            ),
            InputParam(
                name="audio_condition_latents",
                type_hint=list[torch.Tensor],
                required=True,
                description="The encoded audio conditioning rows, one per audio-bearing reference in packed order.",
            ),
            InputParam.template("height", required=True, description="Height of the generated video in pixels."),
            InputParam.template("width", required=True, description="Width of the generated video in pixels."),
            InputParam(
                name="num_frames",
                type_hint=int,
                required=True,
                description="Resolved number of frames, of the form 17 * n + 5.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("height", type_hint=int, description="Resolved height of the generated video in pixels."),
            OutputParam("width", type_hint=int, description="Resolved width of the generated video in pixels."),
            OutputParam("num_frames", type_hint=int, description="Resolved number of frames, of the form 17 * n + 5."),
            OutputParam("num_latent_frames", type_hint=int, description="Number of generated video latent frames."),
            OutputParam("latent_height", type_hint=int, description="Height of the generated video latents."),
            OutputParam("latent_width", type_hint=int, description="Width of the generated video latents."),
            OutputParam(
                "num_audio_latents", type_hint=int, description="Number of generated audio latents per channel."
            ),
            OutputParam(
                "position_ids",
                type_hint=torch.Tensor,
                kwargs_type="denoiser_input_fields",
                description="The `(t, h, w)` rotary coordinate of every row, in float64.",
            ),
            OutputParam(
                "token_tags",
                type_hint=torch.Tensor,
                kwargs_type="denoiser_input_fields",
                description="The modality tag of every row.",
            ),
            OutputParam(
                "video_indices",
                type_hint=torch.Tensor,
                kwargs_type="denoiser_input_fields",
                description="Sequence positions of the video rows, conditioning rows first.",
            ),
            OutputParam(
                "audio_indices",
                type_hint=torch.Tensor,
                kwargs_type="denoiser_input_fields",
                description="Sequence positions of the audio rows, reference rows first.",
            ),
            OutputParam(
                "text_indices",
                type_hint=torch.Tensor,
                kwargs_type="denoiser_input_fields",
                description="Sequence positions of the text rows.",
            ),
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

    @torch.no_grad()
    def __call__(self, components: MiniMaxH3Ref2VAModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        device = components._execution_device

        # The canvas and the frame count are settled by the setup step: a `ref2va` soundtrack is truncated to the
        # generated duration as the references are prepared, so `num_frames` has to be final before that runs.
        ratio = components.vae_spatial_compression_ratio
        block_state.num_latent_frames = video_latent_num_frames(
            block_state.num_frames, components.vae_frames_per_chunk, components.vae_latents_per_chunk
        )
        block_state.latent_height = block_state.height // ratio
        block_state.latent_width = block_state.width // ratio
        block_state.num_audio_latents = audio_latent_num_frames(block_state.num_frames)

        (
            position_ids,
            token_tags,
            video_indices,
            audio_indices,
            text_indices,
            block_state.num_condition_video_rows,
            block_state.num_condition_audio_rows,
        ) = build_ref2va_packed_sequence(
            block_state.text_token_tags,
            block_state.prepared_references,
            block_state.condition_latents,
            block_state.audio_condition_latents,
            block_state.num_latent_frames,
            block_state.latent_height,
            block_state.latent_width,
            block_state.num_audio_latents,
            components.patch_size,
        )
        block_state.position_ids = position_ids.to(device)
        block_state.token_tags = token_tags.to(device)
        block_state.video_indices = video_indices.to(device)
        block_state.audio_indices = audio_indices.to(device)
        block_state.text_indices = text_indices.to(device)

        self.set_block_state(state, block_state)
        return components, state


class MiniMaxH3PrepareLatentsStep(ModularPipelineBlocks):
    model_name = "minimax-h3"

    @property
    def description(self) -> str:
        return (
            "Draws every noise stream of the request and packs the video rows. MiniMax-H3 draws one condition at a "
            "time first — noising the encoded anchors to its conditioning level — then the video noise as a latent "
            "tensor, then the audio noise directly in row layout, all off the request's generator, in that order."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("scheduler", MiniMaxH3Scheduler)]

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
                name="num_condition_video_rows",
                type_hint=int,
                default=0,
                description="How many conditioning rows the layout reserved, which the packed conditioning must match.",
            ),
            InputParam(
                name="condition_latents",
                type_hint=list[torch.Tensor],
                description=(
                    "The encoded video conditioning latents, one `(1, latent_channels, num_latent_frames, "
                    "latent_height, latent_width)` tensor per condition in packed order, or None for a request that "
                    "has none. Noised and packed here."
                ),
            ),
            InputParam(
                name="audio_condition_latents",
                type_hint=list[torch.Tensor],
                description=(
                    "The audio conditioning rows to prepend, one tensor per audio-bearing reference in packed "
                    "order. Empty for a request that has none, which is every `t2va` and `fl2va` one."
                ),
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

    @torch.no_grad()
    def __call__(self, components: MiniMaxH3ModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        device = components._execution_device
        patch_size = components.patch_size

        # A request draws every stream from the one generator it is given, and the order is part of what that
        # generator reproduces: one draw per condition first, then the video noise as a latent tensor, then the audio
        # noise directly in row layout. Passing `latents` or `audio_latents` skips its draw and shifts the ones after
        # it.
        condition_rows = None
        if block_state.condition_latents:
            # One draw per condition, in packed order. Each is packed on its own because `ref2va` references are
            # encoded at their own resolutions, so their latents do not share a shape.
            packed = []
            for condition in block_state.condition_latents:
                noise = randn_tensor(
                    condition.shape, generator=block_state.generator, device=device, dtype=torch.float32
                )
                # The anchors are not fully clean: the released model noises them to `t = 0.999` and holds them there
                # for every step. Mixing before the patchify is the same arithmetic, since patchify only permutes.
                noised = components.scheduler.scale_noise(condition.to(device), components.keyframe_noise_aug, noise)
                packed.append(patchify_video_latents(noised, patch_size))
            condition_rows = torch.cat(packed)
            # In a hand-assembled chain the canvas reaching the layout is user input, so it can disagree with the
            # keyframes that were actually encoded. Left alone the mismatch first surfaces as an `index_copy` shape
            # error inside the transformer, 50 layers deep.
            if condition_rows.shape[0] != block_state.num_condition_video_rows:
                raise ValueError(
                    f"The layout reserved {block_state.num_condition_video_rows} conditioning rows but the encoded "
                    f"conditioning latents pack into {condition_rows.shape[0]}. The canvas the layout was built from "
                    "and the one the conditioning was encoded at do not agree."
                )

        latents = block_state.latents
        if latents is None:
            latents = randn_tensor(
                (
                    1,
                    components.vae_latent_channels,
                    block_state.num_latent_frames,
                    block_state.latent_height,
                    block_state.latent_width,
                ),
                generator=block_state.generator,
                device=device,
                dtype=torch.float32,
            )
        video_rows = patchify_video_latents(latents.to(device, torch.float32), patch_size)

        if block_state.audio_latents is None:
            audio_rows = randn_tensor(
                (block_state.num_audio_latents * components.audio_channels, components.audio_latent_channels),
                generator=block_state.generator,
                device=device,
                dtype=torch.float32,
            )
        else:
            audio_rows = (
                block_state.audio_latents.to(device, torch.float32)
                .permute(0, 2, 1)
                .reshape(-1, components.audio_latent_channels)
            )

        if condition_rows is not None:
            video_rows = torch.cat([condition_rows, video_rows])
        if block_state.audio_condition_latents:
            audio_rows = torch.cat(
                [rows.to(device) for rows in block_state.audio_condition_latents] + [audio_rows]
            )
        block_state.latents, block_state.audio_latents = video_rows, audio_rows

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
                name="video_indices",
                type_hint=torch.Tensor,
                required=True,
                description="Sequence positions of the video rows, conditioning rows first.",
            ),
            InputParam(
                name="audio_indices",
                type_hint=torch.Tensor,
                required=True,
                description="Sequence positions of the audio rows, reference rows first.",
            ),
            InputParam(
                name="text_indices",
                type_hint=torch.Tensor,
                required=True,
                description="Sequence positions of the text rows.",
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
                    block_state.video_indices,
                    block_state.audio_indices,
                    block_state.num_condition_video_rows,
                    block_state.num_condition_audio_rows,
                    block_state.text_indices.numel(),
                    float(timestep),
                    float(audio_timestep),
                    max(float(timestep), components.keyframe_noise_aug),
                    1.0,
                )
            )
            for timestep, audio_timestep in zip(block_state.timesteps, block_state.audio_timesteps)
        ]

        self.set_block_state(state, block_state)
        return components, state
