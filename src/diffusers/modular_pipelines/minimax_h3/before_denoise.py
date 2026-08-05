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

import numpy as np
import torch

from ...schedulers import MiniMaxH3Scheduler
from ...utils import logging
from ...utils.torch_utils import randn_tensor
from ..modular_pipeline import ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import ComponentSpec, ConfigSpec, InputParam, OutputParam
from .modular_pipeline import (
    MINIMAX_H3_AUDIO_CHANNELS,
    MINIMAX_H3_AUDIO_TAG,
    MINIMAX_H3_VIDEO_TAG,
    MiniMaxH3ModularPipeline,
    align_num_frames,
    audio_latent_num_frames,
    resolve_canvas_size,
    video_latent_num_frames,
)
from .references import MiniMaxH3Reference


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# Rotary-time constants. One latent frame spans `5/3 * frames_per_latent` rotary units, where the pattern
# `(1, 4, 4, 4, 4)` mirrors the VAE's 17-pixel-frames-to-5-latent-frames grouping; the spatial axes are normalized
# by the square root of the latent area and scaled by 32.
_ROPE_FRAME_RESCALE = 5.0 / 3.0
_ROPE_FRAMES_PER_LATENT = (1, 4, 4, 4, 4)
_ROPE_SPATIAL_SCALE = 32


def patchify_video_latents(latents: torch.Tensor, patch_size: tuple[int, int, int]) -> torch.Tensor:
    r"""
    Pack video latents into transformer rows.

    Args:
        latents (`torch.Tensor` of shape `(batch_size, channels, num_frames, height, width)`):
            The latents to pack.
        patch_size (`tuple[int, int, int]`): The `(t, h, w)` patch.

    Returns:
        `torch.Tensor` of shape `(batch_size * num_patches, channels * prod(patch_size))`: The packed rows, ordered
        frame-major then row-major.
    """
    patch_t, patch_h, patch_w = patch_size
    batch_size, channels, num_frames, height, width = latents.shape
    if num_frames % patch_t or height % patch_h or width % patch_w:
        raise ValueError(f"Latents of shape {tuple(latents.shape)} are not divisible by the patch {patch_size}.")

    latents = latents.reshape(
        batch_size,
        channels,
        num_frames // patch_t,
        patch_t,
        height // patch_h,
        patch_h,
        width // patch_w,
        patch_w,
    )
    latents = latents.permute(0, 2, 4, 6, 1, 3, 5, 7)
    return latents.reshape(-1, channels * patch_t * patch_h * patch_w).contiguous()


def _spatial_position_grid(dim: int, patch: int, sqrt_area: float) -> torch.Tensor:
    r"""
    One aspect-normalized spatial rotary axis: `dim // patch` coordinates centred on the unit interval, scaled up by
    32. The right endpoint is excluded, so a square canvas spans `[0, 32)`.
    """
    ratio = dim / sqrt_area
    left = (1.0 - ratio) / 2.0
    # Built with numpy: `np.linspace(..., endpoint=False)` is `start + arange(num) * (stop - start) / num`, which is
    # not what `torch.linspace` computes, and the float64 grid has to be reproduced exactly.
    grid = np.linspace(left, left + ratio, dim // patch, endpoint=False) * _ROPE_SPATIAL_SCALE
    return torch.from_numpy(grid).to(torch.float64)


def _temporal_position_grid(num_latent_frames: int, origin: float) -> torch.Tensor:
    r"""The rotary time of every latent frame, starting at `origin`. Spacing is non-uniform: `5/3 * (1, 4, 4, 4, 4)`."""
    spans = torch.tensor(
        [
            _ROPE_FRAME_RESCALE * _ROPE_FRAMES_PER_LATENT[index % len(_ROPE_FRAMES_PER_LATENT)]
            for index in range(num_latent_frames)
        ],
        dtype=torch.float64,
    )
    return origin + torch.cat([torch.zeros(1, dtype=torch.float64), spans[:-1].cumsum(0)])


def _temporal_position_span_pairwise(num_latent_frames: int) -> float:
    r"""
    The rotary time spanned by `num_latent_frames` latent frames.

    Summed by numpy (pairwise summation) rather than sequentially: the reference computes the keyframe anchor this
    way and the two summation orders differ in the last ulp from 16 latent frames onwards.
    """
    spans = np.ones(num_latent_frames, dtype=np.float64) * _ROPE_FRAME_RESCALE
    for index in range(len(_ROPE_FRAMES_PER_LATENT)):
        spans[index :: len(_ROPE_FRAMES_PER_LATENT)] *= _ROPE_FRAMES_PER_LATENT[index]
    return float(spans.sum())


def _temporal_position_span_sequential(num_latent_frames: int) -> float:
    r"""
    The rotary time a video reference advances the clock by.

    Summed sequentially in float64, which is *not* how [`_temporal_position_span_pairwise`] sums
    the same series: that one reproduces a numpy pairwise sum, and the two orders differ in the last ulp from 16
    latent frames onwards. The reference implementation keeps both, one per call site, so the port has to as well.
    """
    return sum(
        _ROPE_FRAME_RESCALE * _ROPE_FRAMES_PER_LATENT[index % len(_ROPE_FRAMES_PER_LATENT)]
        for index in range(num_latent_frames)
    )


def _frame_position_grid(
    latent_height: int, latent_width: int, patch_h: int, patch_w: int
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""The `(h, w)` rotary coordinates of one latent frame, and the width axis they were built from."""
    sqrt_area = np.sqrt(latent_height * latent_width)
    height_grid = _spatial_position_grid(latent_height, patch_h, sqrt_area)
    width_grid = _spatial_position_grid(latent_width, patch_w, sqrt_area)
    grids = torch.meshgrid(height_grid, width_grid, indexing="ij")
    return torch.stack([grid.reshape(-1) for grid in grids], dim=-1), width_grid


def _fill_audio_positions(
    position_ids: torch.Tensor,
    rows: slice,
    num_audio_latents: int,
    rotary_time: float,
    width_grid: torch.Tensor,
) -> None:
    r"""
    Place one channel-major audio block.

    Audio rows carry no height coordinate and are pinned to the two extremes of the width grid of *their own* block —
    the target grid for a standalone audio reference, the video's grid for a soundtrack.
    """
    time = rotary_time + torch.arange(num_audio_latents, dtype=torch.float64)
    position_ids[rows, 0] = time.repeat(MINIMAX_H3_AUDIO_CHANNELS)
    position_ids[rows, 2] = torch.cat(
        [
            torch.full((num_audio_latents,), float(width_grid[0]), dtype=torch.float64),
            torch.full((num_audio_latents,), float(width_grid[-1]), dtype=torch.float64),
        ]
    )


class MiniMaxH3NoKeyframeAnchorsStep(ModularPipelineBlocks):
    model_name = "minimax-h3"

    @property
    def description(self) -> str:
        return (
            "Declares that a `t2va` request anchors no keyframes. The layout step is the same block for `t2va` and "
            "`fl2va` and reads the anchors the request resolved; a text-only one resolves none, and saying so here "
            "keeps `keyframe_anchors` off the `t2va` signature."
        )

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "keyframe_anchors",
                type_hint=tuple,
                description="Which end of the video every keyframe is anchored to — empty, since there are none.",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components: MiniMaxH3ModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        block_state.keyframe_anchors = ()

        self.set_block_state(state, block_state)
        return components, state


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
    def expected_configs(self) -> list[ConfigSpec]:
        # The canvas MiniMax-H3 was released for, which a request without keyframes generates on at 16:9.
        return [ConfigSpec("canvas_short_edge", 768), ConfigSpec("canvas_max_pixels", 768 * 1344)]

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

    @staticmethod
    def build_packed_sequence(
        text_token_tags: torch.Tensor,
        num_latent_frames: int,
        latent_height: int,
        latent_width: int,
        num_audio_latents: int,
        patch_size: tuple[int, int, int],
        keyframe_anchors: tuple[str, ...] = (),
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
        r"""
        Build the `[text | keyframe conditions | target audio | target video]` layout used by the `t2va` and `fl2va`
        tasks.

        Args:
            text_token_tags (`torch.Tensor` of shape `(num_text_tokens,)`):
                The modality tag of every text row. Text is tagged `1`, except for the rows of a keyframe's vision block,
                which MiniMax-H3 tags `0` (video).
            num_latent_frames (`int`): Number of target latent frames.
            latent_height (`int`): Target latent height.
            latent_width (`int`): Target latent width.
            num_audio_latents (`int`): Number of target audio latents per channel.
            patch_size (`tuple[int, int, int]`): The transformer's `(t, h, w)` patch.
            keyframe_anchors (`tuple[str, ...]`):
                One entry per keyframe conditioning block, in packed order: `"first"` anchors the block at the first
                latent frame, `"last"` at the last one.

        Returns:
            `tuple`: `position_ids`, `token_tags`, `video_indices`, `audio_indices`, `text_indices`, and the number of
            leading video and audio rows that are conditioning rather than generated.
        """
        _, patch_h, patch_w = patch_size
        rows_per_frame = (latent_height // patch_h) * (latent_width // patch_w)
        num_text_tokens = text_token_tags.shape[0]
        num_condition_rows = len(keyframe_anchors) * rows_per_frame
        num_audio_rows = num_audio_latents * MINIMAX_H3_AUDIO_CHANNELS
        num_video_rows = num_latent_frames * rows_per_frame
        sequence_length = num_text_tokens + num_condition_rows + num_audio_rows + num_video_rows

        condition_start = num_text_tokens
        audio_start = condition_start + num_condition_rows
        video_start = audio_start + num_audio_rows

        # 1. The (t, h, w) grid. Text rows sit on the time axis at their row index, and the media rows continue the time
        # axis from there, so text length shifts the whole media clock.
        position_ids = torch.zeros(sequence_length, 3, dtype=torch.float64)
        position_ids[:num_text_tokens, 0] = torch.arange(num_text_tokens, dtype=torch.float64)

        sqrt_area = np.sqrt(latent_height * latent_width)
        height_grid = _spatial_position_grid(latent_height, patch_h, sqrt_area)
        width_grid = _spatial_position_grid(latent_width, patch_w, sqrt_area)
        frame_grid = torch.stack(
            [grid.reshape(-1) for grid in torch.meshgrid(height_grid, width_grid, indexing="ij")], -1
        )

        for index, anchor in enumerate(keyframe_anchors):
            if anchor == "first":
                anchor_time = float(num_text_tokens)
            elif anchor == "last":
                anchor_time = (
                    float(num_text_tokens) + _temporal_position_span_pairwise(num_latent_frames) - _ROPE_FRAME_RESCALE
                )
            else:
                raise ValueError(f"A keyframe anchor must be 'first' or 'last', got {anchor!r}.")
            rows = slice(condition_start + index * rows_per_frame, condition_start + (index + 1) * rows_per_frame)
            position_ids[rows, 0] = anchor_time
            position_ids[rows, 1:] = frame_grid

        # Audio rows are channel-major and share the video's rotary clock: one unit per latent at 40 latents/s equals
        # 24 fps * 5/3. They carry no height coordinate and are pinned to the two extremes of the width grid.
        audio_time = float(num_text_tokens) + torch.arange(num_audio_latents, dtype=torch.float64)
        position_ids[audio_start:video_start, 0] = audio_time.repeat(MINIMAX_H3_AUDIO_CHANNELS)
        position_ids[audio_start:video_start, 2] = torch.cat(
            [
                torch.full((num_audio_latents,), float(width_grid[0]), dtype=torch.float64),
                torch.full((num_audio_rows - num_audio_latents,), float(width_grid[-1]), dtype=torch.float64),
            ]
        )

        video_position_ids = torch.empty(num_latent_frames, rows_per_frame, 3, dtype=torch.float64)
        video_position_ids[:, :, 0] = _temporal_position_grid(num_latent_frames, float(num_text_tokens))[:, None]
        video_position_ids[:, :, 1:] = frame_grid[None]
        position_ids[video_start:] = video_position_ids.reshape(-1, 3)

        # 2. Row indices and modality tags.
        video_indices = torch.cat(
            [torch.arange(condition_start, audio_start), torch.arange(video_start, sequence_length)]
        )
        audio_indices = torch.arange(audio_start, video_start)
        text_indices = torch.arange(num_text_tokens)

        token_tags = torch.empty(sequence_length, dtype=torch.long)
        token_tags[text_indices] = text_token_tags.to(torch.long)
        token_tags[audio_indices] = MINIMAX_H3_AUDIO_TAG
        token_tags[video_indices] = MINIMAX_H3_VIDEO_TAG

        return position_ids, token_tags, video_indices, audio_indices, text_indices, num_condition_rows, 0

    @torch.no_grad()
    def __call__(self, components: MiniMaxH3ModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        device = components._execution_device

        # Without a keyframe to take the aspect ratio from, MiniMax-H3 generates on its own 16:9 canvas.
        if block_state.height is None:
            block_state.height, block_state.width = resolve_canvas_size(
                16,
                9,
                components.canvas_multiple,
                components.config.canvas_short_edge,
                components.config.canvas_max_pixels,
            )
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
        ) = self.build_packed_sequence(
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
    model_name = "minimax-h3"

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
                name="normalized_references",
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

    @staticmethod
    def build_ref2va_packed_sequence(
        text_token_tags: torch.Tensor,
        references: list[MiniMaxH3Reference],
        condition_latents: list[torch.Tensor],
        audio_condition_latents: list[torch.Tensor],
        num_latent_frames: int,
        latent_height: int,
        latent_width: int,
        num_audio_latents: int,
        patch_size: tuple[int, int, int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
        r"""
        Build the `[text | reference blocks | target audio | target video]` layout of the `ref2va` task.

        Args:
            text_token_tags (`torch.Tensor` of shape `(num_text_tokens,)`):
                The modality tag of every text row. Text is tagged `1`, except for the rows of a reference's vision block,
                which MiniMax-H3 tags `0` (video).
            references (`list[MiniMaxH3Reference]`):
                The references, in packed order. Only their modality is read here; the geometry comes from the latents.
            condition_latents (`list[torch.Tensor]`):
                One `(1, channels, num_latent_frames, latent_height, latent_width)` tensor per image and video reference,
                in packed order, as [`MiniMaxH3Ref2VAReferenceEncoderStep`] produced them.
            audio_condition_latents (`list[torch.Tensor]`):
                One `(num_audio_latents * 2, audio_latent_channels)` tensor per audio-bearing reference, in packed order.
            num_latent_frames (`int`): Number of target latent frames.
            latent_height (`int`): Target latent height.
            latent_width (`int`): Target latent width.
            num_audio_latents (`int`): Number of target audio latents per channel.
            patch_size (`tuple[int, int, int]`): The transformer's `(t, h, w)` patch.

        Returns:
            `tuple`: `position_ids`, `token_tags`, `video_indices`, `audio_indices`, `text_indices`, and the number of
            leading video and audio rows that are references rather than generated.
        """
        _, patch_h, patch_w = patch_size
        num_text_tokens = text_token_tags.shape[0]
        num_target_video_rows = num_latent_frames * (latent_height // patch_h) * (latent_width // patch_w)
        num_target_audio_rows = num_audio_latents * MINIMAX_H3_AUDIO_CHANNELS

        # The geometry of every reference block is the shape of what the encoder produced for it, so the two can never
        # disagree. Both lists are in packed order but skip the references they do not apply to, so they are consumed as
        # iterators alongside the reference list rather than indexed by it.
        visual_geometry = iter(tuple(latents.shape[2:5]) for latents in condition_latents)
        audio_row_counts = iter(rows.shape[0] for rows in audio_condition_latents)
        num_reference_video_rows = sum(
            frames * (height // patch_h) * (width // patch_w)
            for frames, height, width in (tuple(latents.shape[2:5]) for latents in condition_latents)
        )
        num_reference_audio_rows = sum(rows.shape[0] for rows in audio_condition_latents)
        sequence_length = (
            num_text_tokens
            + num_reference_video_rows
            + num_reference_audio_rows
            + num_target_audio_rows
            + num_target_video_rows
        )

        position_ids = torch.zeros(sequence_length, 3, dtype=torch.float64)
        position_ids[:num_text_tokens, 0] = torch.arange(num_text_tokens, dtype=torch.float64)
        target_frame_grid, target_width_grid = _frame_position_grid(latent_height, latent_width, patch_h, patch_w)

        # Reference blocks, in request order. `rotary_time` is the shared audio/video clock: it starts where the text
        # rows end and every block pushes it forward by the time that block occupies.
        video_indices, audio_indices = [], []
        cursor = num_text_tokens
        rotary_time = float(num_text_tokens)
        for reference in references:
            if reference.kind == "image":
                num_latent_frames_, reference_height, reference_width = next(visual_geometry)
                num_video_rows = num_latent_frames_ * (reference_height // patch_h) * (reference_width // patch_w)
                rows = slice(cursor, cursor + num_video_rows)
                cursor = rows.stop
                video_indices.append(torch.arange(rows.start, rows.stop))
                frame_grid, _ = _frame_position_grid(reference_height, reference_width, patch_h, patch_w)
                position_ids[rows, 0] = rotary_time
                position_ids[rows, 1:] = frame_grid
                # An image is a single frame and takes a single integer rotary slot, not a latent frame's 5/3 units.
                rotary_time += 1.0
            elif reference.kind == "audio":
                num_audio_rows = next(audio_row_counts)
                reference_audio_latents = num_audio_rows // MINIMAX_H3_AUDIO_CHANNELS
                rows = slice(cursor, cursor + num_audio_rows)
                cursor = rows.stop
                audio_indices.append(torch.arange(rows.start, rows.stop))
                _fill_audio_positions(position_ids, rows, reference_audio_latents, rotary_time, target_width_grid)
                rotary_time += float(reference_audio_latents)
            elif reference.kind == "video":
                # A video reference's soundtrack rows are packed immediately before its video rows and share their
                # origin, so the two are rotary-aligned exactly as the generated audio and video are.
                num_audio_rows = next(audio_row_counts) if reference.has_audio else 0
                reference_audio_latents = num_audio_rows // MINIMAX_H3_AUDIO_CHANNELS
                num_latent_frames_, reference_height, reference_width = next(visual_geometry)
                num_video_rows = num_latent_frames_ * (reference_height // patch_h) * (reference_width // patch_w)
                audio_rows = slice(cursor, cursor + num_audio_rows)
                video_rows = slice(audio_rows.stop, audio_rows.stop + num_video_rows)
                cursor = video_rows.stop
                audio_indices.append(torch.arange(audio_rows.start, audio_rows.stop))
                video_indices.append(torch.arange(video_rows.start, video_rows.stop))

                frame_grid, width_grid = _frame_position_grid(reference_height, reference_width, patch_h, patch_w)
                _fill_audio_positions(position_ids, audio_rows, reference_audio_latents, rotary_time, width_grid)
                frame_time = _temporal_position_grid(num_latent_frames_, rotary_time)
                position_ids[video_rows, 0] = frame_time.repeat_interleave(frame_grid.shape[0])
                position_ids[video_rows, 1:] = frame_grid.repeat(num_latent_frames_, 1)
                rotary_time += max(
                    float(reference_audio_latents), _temporal_position_span_sequential(num_latent_frames_)
                )
            else:
                raise ValueError(f"A reference must be an 'image', a 'video' or an 'audio', got {reference.kind!r}.")

        # The generated rows. Target audio and target video share the origin the reference blocks left behind.
        audio_start = cursor
        video_start = audio_start + num_target_audio_rows
        _fill_audio_positions(
            position_ids, slice(audio_start, video_start), num_audio_latents, rotary_time, target_width_grid
        )
        frame_time = _temporal_position_grid(num_latent_frames, rotary_time)
        position_ids[video_start:, 0] = frame_time.repeat_interleave(target_frame_grid.shape[0])
        position_ids[video_start:, 1:] = target_frame_grid.repeat(num_latent_frames, 1)

        video_indices = torch.cat(video_indices + [torch.arange(video_start, sequence_length)])
        audio_indices = torch.cat(audio_indices + [torch.arange(audio_start, video_start)])
        text_indices = torch.arange(num_text_tokens)

        token_tags = torch.empty(sequence_length, dtype=torch.long)
        token_tags[text_indices] = text_token_tags.to(torch.long)
        token_tags[audio_indices] = MINIMAX_H3_AUDIO_TAG
        token_tags[video_indices] = MINIMAX_H3_VIDEO_TAG

        return (
            position_ids,
            token_tags,
            video_indices,
            audio_indices,
            text_indices,
            num_reference_video_rows,
            num_reference_audio_rows,
        )

    @torch.no_grad()
    def __call__(self, components: MiniMaxH3ModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        device = components._execution_device

        # The canvas and the frame count are settled by the setup step, but in a hand-assembled chain this block
        # sees user input directly — and a canvas off the multiple floor-divides into a broken decode.
        multiple = components.canvas_multiple
        if block_state.height % multiple or block_state.width % multiple:
            raise ValueError(
                f"`height` and `width` must be multiples of {multiple}, got {block_state.height}x{block_state.width}."
            )
        # A `ref2va` soundtrack is truncated to the generated duration as the references are prepared, so
        # `num_frames` has to be final before the setup step runs.
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
        ) = self.build_ref2va_packed_sequence(
            block_state.text_token_tags,
            block_state.normalized_references,
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
            "Draws the noise of the generated rows and packs them: the video noise as a latent tensor first, then "
            "the audio noise directly in row layout, both off the request's generator, in that order. Every "
            "workflow generates the same way, so this is the same block for all of them; a request that conditions "
            "on something noises it *before* this block — the draw order is part of what its generator reproduces — "
            "and puts it in front of these rows after."
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
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "latents",
                type_hint=torch.Tensor,
                description="The generated video rows of the packed sequence.",
            ),
            OutputParam(
                "audio_latents",
                type_hint=torch.Tensor,
                description="The generated audio rows of the packed sequence, channel-major.",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components: MiniMaxH3ModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        device = components._execution_device
        patch_size = components.patch_size

        # A request draws every stream from the one generator it is given, and the order is part of what that
        # generator reproduces: any conditioning noise first (drawn before this block), then the video noise as a
        # latent tensor, then the audio noise directly in row layout. Passing `latents` or `audio_latents` skips its
        # draw and shifts the ones after it.
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

        block_state.latents, block_state.audio_latents = video_rows, audio_rows

        self.set_block_state(state, block_state)
        return components, state


class MiniMaxH3PrepareConditionLatentsStep(ModularPipelineBlocks):
    model_name = "minimax-h3"

    @property
    def description(self) -> str:
        return (
            "Noises the encoded visual conditioning to MiniMax-H3's conditioning level and packs it into rows — the "
            "`fl2va` keyframes or the `ref2va` image and video references, the same recipe either way. It runs "
            "*before* the noise of the generated rows is drawn, because a request draws one condition at a time "
            "first and that order is part of what its generator reproduces."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("scheduler", MiniMaxH3Scheduler)]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template(
                "generator",
                description=(
                    "The generator of the request. The conditioning noise is drawn from it first, one draw per "
                    "condition, before the noise of the generated rows."
                ),
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
                required=True,
                description=(
                    "The encoded video conditioning latents, one `(1, latent_channels, num_latent_frames, "
                    "latent_height, latent_width)` tensor per condition in packed order."
                ),
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "condition_rows",
                type_hint=torch.Tensor,
                description="The noised conditioning, packed into the leading video rows of the sequence.",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components: MiniMaxH3ModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        device = components._execution_device
        patch_size = components.patch_size

        # One draw per condition, in packed order. Each is packed on its own because `ref2va` references are encoded
        # at their own resolutions, so their latents do not share a shape.
        packed = []
        for condition in block_state.condition_latents:
            noise = randn_tensor(condition.shape, generator=block_state.generator, device=device, dtype=torch.float32)
            # The anchors are not fully clean: the released model noises them to `t = 0.999` and holds them there
            # for every step. Mixing before the patchify is the same arithmetic, since patchify only permutes.
            noised = components.scheduler.scale_noise(condition.to(device), components.keyframe_noise_aug, noise)
            packed.append(patchify_video_latents(noised, patch_size))
        block_state.condition_rows = torch.cat(packed)
        # In a hand-assembled chain the canvas reaching the layout is user input, so it can disagree with the
        # conditioning that was actually encoded. Left alone the mismatch first surfaces as an `index_copy` shape
        # error inside the transformer, 50 layers deep.
        if block_state.condition_rows.shape[0] != block_state.num_condition_video_rows:
            raise ValueError(
                f"The layout reserved {block_state.num_condition_video_rows} conditioning rows but the encoded "
                f"conditioning latents pack into {block_state.condition_rows.shape[0]}. The canvas the layout was "
                "built from and the one the conditioning was encoded at do not agree."
            )

        self.set_block_state(state, block_state)
        return components, state


class MiniMaxH3FL2VAPrepareLatentsStep(ModularPipelineBlocks):
    model_name = "minimax-h3"

    @property
    def description(self) -> str:
        return (
            "Finishes the video rows of a `fl2va` request by putting its noised keyframe conditioning in front of "
            "the generated rows, which is where the layout reserved it. The denoising loop only ever writes the "
            "generated rows, so the conditioning rides through every step unchanged."
        )

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                name="condition_rows",
                type_hint=torch.Tensor,
                required=True,
                description="The noised keyframe conditioning, packed into rows.",
            ),
            InputParam(
                name="latents",
                type_hint=torch.Tensor,
                required=True,
                description="The generated video rows of the packed sequence.",
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
        ]

    @torch.no_grad()
    def __call__(self, components: MiniMaxH3ModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        block_state.latents = torch.cat([block_state.condition_rows, block_state.latents])

        self.set_block_state(state, block_state)
        return components, state


class MiniMaxH3Ref2VAPrepareLatentsStep(ModularPipelineBlocks):
    model_name = "minimax-h3"

    @property
    def description(self) -> str:
        return (
            "Finishes both streams of a `ref2va` request by putting its conditioning in front of the generated "
            "rows, which is where the layout reserved it: the noised image and video references on the video side, "
            "and the reference soundtracks on the audio side. The soundtracks are never noised — a reference "
            "soundtrack conditions at `t = 0` — and the denoising loop only ever writes the generated rows, so all "
            "of it rides through every step unchanged."
        )

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                name="condition_rows",
                type_hint=torch.Tensor,
                required=True,
                description="The noised conditioning of the image and video references, packed into rows.",
            ),
            InputParam(
                name="latents",
                type_hint=torch.Tensor,
                required=True,
                description="The generated video rows of the packed sequence.",
            ),
            InputParam(
                name="audio_condition_latents",
                type_hint=list[torch.Tensor],
                required=True,
                description=(
                    "The audio conditioning rows to prepend, one tensor per audio-bearing reference in packed "
                    "order. Empty for a request that has none."
                ),
            ),
            InputParam(
                name="num_condition_audio_rows",
                type_hint=int,
                default=0,
                description="How many reference audio rows the layout reserved, which the packed rows must match.",
            ),
            InputParam(
                name="audio_latents",
                type_hint=torch.Tensor,
                required=True,
                description="The generated audio rows of the packed sequence, channel-major.",
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

        block_state.latents = torch.cat([block_state.condition_rows, block_state.latents])

        if block_state.audio_condition_latents:
            num_reference_audio_rows = sum(rows.shape[0] for rows in block_state.audio_condition_latents)
            if num_reference_audio_rows != block_state.num_condition_audio_rows:
                raise ValueError(
                    f"The layout reserved {block_state.num_condition_audio_rows} reference audio rows but the encoded "
                    f"soundtracks pack into {num_reference_audio_rows}. The references the layout was built from and "
                    "the ones the audio conditioning was encoded from do not agree."
                )
            block_state.audio_latents = torch.cat(
                [rows.to(device) for rows in block_state.audio_condition_latents] + [block_state.audio_latents]
            )

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

    @staticmethod
    def build_row_timesteps(
        video_indices: torch.Tensor,
        audio_indices: torch.Tensor,
        num_condition_video_rows: int,
        num_condition_audio_rows: int,
        num_text_tokens: int,
        video_timestep: float,
        audio_timestep: float,
        condition_video_timestep: float,
        condition_audio_timestep: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""
        Assign a timestep to every row of the packed sequence and reduce it to the transformer's `(timestep,
        timestep_indices)` pair.

        One forward serves rows at different noise levels: the generated video and audio rows step down their own
        schedules while the conditioning rows stay pinned at their noise-augmentation level. Text rows never reach an
        output head and inherit the video timestep.

        Args:
            video_indices (`torch.Tensor`): Sequence positions of the video rows, conditioning rows first.
            audio_indices (`torch.Tensor`): Sequence positions of the audio rows, reference rows first.
            num_condition_video_rows (`int`): How many leading video rows are conditioning rows.
            num_condition_audio_rows (`int`): How many leading audio rows are reference rows.
            num_text_tokens (`int`): Number of text rows, which never reach an output head.
            video_timestep (`float`): Timestep of the generated video rows.
            audio_timestep (`float`): Timestep of the generated audio rows.
            condition_video_timestep (`float`): Timestep of the video conditioning rows.
            condition_audio_timestep (`float`): Timestep of the audio reference rows.

        Returns:
            `tuple[torch.Tensor, torch.Tensor]`: the distinct timesteps, sorted, and the index of every row into them.
        """
        sequence_length = int(video_indices.numel() + audio_indices.numel() + num_text_tokens)
        row_timesteps = torch.full((sequence_length,), video_timestep, dtype=torch.float32)
        row_timesteps[video_indices[:num_condition_video_rows]] = condition_video_timestep
        row_timesteps[audio_indices[num_condition_audio_rows:]] = audio_timestep
        row_timesteps[audio_indices[:num_condition_audio_rows]] = condition_audio_timestep
        return torch.unique(row_timesteps, sorted=True, return_inverse=True)

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
                for tensor in self.build_row_timesteps(
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
