# Copyright 2026 Pruna AI, The MiniMax and HuggingFace Teams. All rights reserved.
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
from ...utils.torch_utils import maybe_adjust_dtype_for_device, randn_tensor
from ..modular_pipeline import ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import ComponentSpec, ConfigSpec, InputParam, OutputParam
from .before_encoder import round_canvas
from .modular_pipeline import PH3ImageModularPipeline


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# Rotary-time constants. One latent frame spans `5/3 * frames_per_latent` rotary units, where the pattern
# `(1, 4, 4, 4, 4)` mirrors the VAE's 17-pixel-frames-to-5-latent-frames grouping; the spatial axes are normalized
# by the square root of the latent area and scaled by 32.
_ROPE_FRAME_RESCALE = 5.0 / 3.0
_ROPE_FRAMES_PER_LATENT = (1, 4, 4, 4, 4)
_ROPE_SPATIAL_SCALE = 32


# Copied from diffusers.modular_pipelines.minimax_h3.before_denoise.patchify_video_latents
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


# Copied from diffusers.modular_pipelines.minimax_h3.before_denoise._spatial_position_grid
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


# Copied from diffusers.modular_pipelines.minimax_h3.before_denoise._temporal_position_grid
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


# Copied from diffusers.modular_pipelines.minimax_h3.before_denoise._frame_position_grid
def _frame_position_grid(
    latent_height: int, latent_width: int, patch_h: int, patch_w: int
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""The `(h, w)` rotary coordinates of one latent frame, and the width axis they were built from."""
    sqrt_area = np.sqrt(latent_height * latent_width)
    height_grid = _spatial_position_grid(latent_height, patch_h, sqrt_area)
    width_grid = _spatial_position_grid(latent_width, patch_w, sqrt_area)
    grids = torch.meshgrid(height_grid, width_grid, indexing="ij")
    return torch.stack([grid.reshape(-1) for grid in grids], dim=-1), width_grid


# Copied from diffusers.modular_pipelines.minimax_h3.before_denoise._fill_audio_positions
def _fill_audio_positions(
    position_ids: torch.Tensor,
    rows: slice,
    num_audio_latents: int,
    rotary_time: float,
    width_grid: torch.Tensor,
    audio_channels: int,
) -> None:
    r"""
    Place one channel-major audio block.

    Audio rows carry no height coordinate and are pinned to the two extremes of the width grid of *their own* block —
    the target grid for a standalone audio reference, the video's grid for a soundtrack.
    """
    time = rotary_time + torch.arange(num_audio_latents, dtype=torch.float64)
    position_ids[rows, 0] = time.repeat(audio_channels)
    position_ids[rows, 2] = torch.cat(
        [
            torch.full((num_audio_latents,), float(width_grid[0]), dtype=torch.float64),
            torch.full((num_audio_latents,), float(width_grid[-1]), dtype=torch.float64),
        ]
    )


class PH3ImagePrepareLayoutStep(ModularPipelineBlocks):
    model_name = "p-h3-image"

    @property
    def description(self) -> str:
        return (
            "Builds the packed `[text | audio | image]` layout of a `text2image` request: MiniMax-H3's layout for "
            "one latent frame and two audio latents."
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
            InputParam.template(
                "height",
                default=2912,
                description=(
                    "Height of the generated image in pixels, rounded to a multiple of 32. The model works best at 8 "
                    "megapixels."
                ),
            ),
            InputParam.template(
                "width",
                default=2912,
                description=(
                    "Width of the generated image in pixels, rounded to a multiple of 32. The model works best at 8 "
                    "megapixels."
                ),
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("height", type_hint=int, description="Resolved height of the generated image in pixels."),
            OutputParam("width", type_hint=int, description="Resolved width of the generated image in pixels."),
            OutputParam(
                "num_latent_frames", type_hint=int, description="Number of generated latent frames, always 1."
            ),
            OutputParam("latent_height", type_hint=int, description="Height of the generated latents."),
            OutputParam("latent_width", type_hint=int, description="Width of the generated latents."),
            OutputParam(
                "num_audio_latents",
                type_hint=int,
                description="Number of audio latents per channel denoised alongside the image.",
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
                description="Sequence positions of the image rows, conditioning rows first.",
            ),
            OutputParam(
                "audio_indices",
                type_hint=torch.Tensor,
                kwargs_type="denoiser_input_fields",
                description="Sequence positions of the audio rows.",
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
                description="How many leading image rows are conditioning rows rather than generated rows.",
            ),
            OutputParam(
                "num_condition_audio_rows",
                type_hint=int,
                description="How many leading audio rows are reference rows rather than generated rows, always 0.",
            ),
        ]

    @staticmethod
    # Copied from diffusers.modular_pipelines.minimax_h3.before_denoise.MiniMaxH3PrepareLayoutStep.build_packed_sequence
    def build_packed_sequence(
        text_token_tags: torch.Tensor,
        num_latent_frames: int,
        latent_height: int,
        latent_width: int,
        num_audio_latents: int,
        patch_size: tuple[int, int, int],
        audio_channels: int,
        audio_tag: int,
        video_tag: int,
        keyframe_anchors: tuple[str, ...] = (),
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
        r"""
        Build the `[text | keyframe conditions | target audio | target video]` layout used by the `t2va` and `fl2va`
        tasks.

        Args:
            text_token_tags (`torch.Tensor` of shape `(num_text_tokens,)`):
                The modality tag of every text row. Text is tagged `1`, except for the rows of a keyframe's vision
                block, which MiniMax-H3 tags `0` (video).
            num_latent_frames (`int`): Number of target latent frames.
            latent_height (`int`): Target latent height.
            latent_width (`int`): Target latent width.
            num_audio_latents (`int`): Number of target audio latents per channel.
            patch_size (`tuple[int, int, int]`): The transformer's `(t, h, w)` patch.
            audio_channels (`int`): Channels the soundtrack is packed channel-major over, i.e.
                `components.audio_channels`.
            audio_tag (`int`), video_tag (`int`):
                The modality tags an audio and a video row carry, i.e. `components.audio_tag` and
                `components.video_tag`.
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
        num_audio_rows = num_audio_latents * audio_channels
        num_video_rows = num_latent_frames * rows_per_frame
        sequence_length = num_text_tokens + num_condition_rows + num_audio_rows + num_video_rows

        condition_start = num_text_tokens
        audio_start = condition_start + num_condition_rows
        video_start = audio_start + num_audio_rows

        # 1. The (t, h, w) grid. Text rows sit on the time axis at their row index, and the media rows continue the time
        # axis from there, so text length shifts the whole media clock.
        position_ids = torch.zeros(sequence_length, 3, dtype=torch.float64)
        position_ids[:num_text_tokens, 0] = torch.arange(num_text_tokens, dtype=torch.float64)

        frame_grid, width_grid = _frame_position_grid(latent_height, latent_width, patch_h, patch_w)

        for index, anchor in enumerate(keyframe_anchors):
            if anchor == "first":
                anchor_time = float(num_text_tokens)
            elif anchor == "last":
                # The rotary time the generated frames span, summed by numpy's pairwise summation because that is
                # how the reference computes this anchor. The soundtrack span in the `ref2va` layout sums the same
                # series sequentially, and the two orders differ in the last ulp from 16 latent frames onwards, so
                # the port keeps both — one per call site.
                spans = np.ones(num_latent_frames, dtype=np.float64) * _ROPE_FRAME_RESCALE
                for offset in range(len(_ROPE_FRAMES_PER_LATENT)):
                    spans[offset :: len(_ROPE_FRAMES_PER_LATENT)] *= _ROPE_FRAMES_PER_LATENT[offset]
                anchor_time = float(num_text_tokens) + float(spans.sum()) - _ROPE_FRAME_RESCALE
            else:
                raise ValueError(f"A keyframe anchor must be 'first' or 'last', got {anchor!r}.")
            rows = slice(condition_start + index * rows_per_frame, condition_start + (index + 1) * rows_per_frame)
            position_ids[rows, 0] = anchor_time
            position_ids[rows, 1:] = frame_grid

        # Audio rows are channel-major and share the video's rotary clock: one unit per latent at 40 latents/s equals
        # 24 fps * 5/3. They carry no height coordinate and are pinned to the two extremes of the width grid.
        audio_time = float(num_text_tokens) + torch.arange(num_audio_latents, dtype=torch.float64)
        position_ids[audio_start:video_start, 0] = audio_time.repeat(audio_channels)
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
        token_tags[audio_indices] = audio_tag
        token_tags[video_indices] = video_tag

        return position_ids, token_tags, video_indices, audio_indices, text_indices, num_condition_rows, 0

    @torch.no_grad()
    def __call__(
        self, components: PH3ImageModularPipeline, state: PipelineState
    ) -> tuple[PH3ImageModularPipeline, PipelineState]:
        block_state = self.get_block_state(state)
        device = components._execution_device

        block_state.height, block_state.width = round_canvas(
            block_state.height, block_state.width, components.canvas_multiple
        )
        ratio = components.vae_spatial_compression_ratio
        block_state.num_latent_frames = 1
        block_state.latent_height = block_state.height // ratio
        block_state.latent_width = block_state.width // ratio
        block_state.num_audio_latents = components.num_audio_latents

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
            components.audio_channels,
            components.audio_tag,
            components.video_tag,
        )
        # MPS, NPU and Neuron have no float64; the rope casts the grid to float32 anyway.
        position_ids_dtype = maybe_adjust_dtype_for_device(position_ids.dtype, device)
        block_state.position_ids = position_ids.to(device, position_ids_dtype)
        block_state.token_tags = token_tags.to(device)
        block_state.video_indices = video_indices.to(device)
        block_state.audio_indices = audio_indices.to(device)
        block_state.text_indices = text_indices.to(device)

        self.set_block_state(state, block_state)
        return components, state


class PH3ImageEditPrepareLayoutStep(ModularPipelineBlocks):
    model_name = "p-h3-image"

    @property
    def description(self) -> str:
        return (
            "Builds the packed `[text | references | audio | image]` layout of an edit: MiniMax-H3's layout for "
            "image references, one latent frame and two audio latents."
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
                name="condition_latents",
                type_hint=list[torch.Tensor],
                required=True,
                description=(
                    "The conditioning latents of the images of the edit. Their shapes give the geometry of the "
                    "reference blocks."
                ),
            ),
            InputParam.template("height", required=True, description="Height of the edited image in pixels."),
            InputParam.template("width", required=True, description="Width of the edited image in pixels."),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "num_latent_frames", type_hint=int, description="Number of generated latent frames, always 1."
            ),
            OutputParam("latent_height", type_hint=int, description="Height of the generated latents."),
            OutputParam("latent_width", type_hint=int, description="Width of the generated latents."),
            OutputParam(
                "num_audio_latents",
                type_hint=int,
                description="Number of audio latents per channel denoised alongside the image.",
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
                description="Sequence positions of the image rows, conditioning rows first.",
            ),
            OutputParam(
                "audio_indices",
                type_hint=torch.Tensor,
                kwargs_type="denoiser_input_fields",
                description="Sequence positions of the audio rows.",
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
                description="How many leading image rows are conditioning rows rather than generated rows.",
            ),
            OutputParam(
                "num_condition_audio_rows",
                type_hint=int,
                description="How many leading audio rows are reference rows rather than generated rows, always 0.",
            ),
        ]

    @staticmethod
    def build_edit_packed_sequence(
        text_token_tags: torch.Tensor,
        condition_latents: list[torch.Tensor],
        latent_height: int,
        latent_width: int,
        num_audio_latents: int,
        patch_size: tuple[int, int, int],
        audio_channels: int,
        audio_tag: int,
        video_tag: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
        r"""
        Build the `[text | images | target audio | target image]` layout of an edit: MiniMax-H3's `ref2va` layout with
        image references only.

        Args:
            text_token_tags (`torch.Tensor` of shape `(num_text_tokens,)`):
                The modality tag of every text row. Text is tagged `1`, except for the rows of an image's vision block,
                which are tagged `0` (video).
            condition_latents (`list[torch.Tensor]`):
                One `(1, channels, 1, latent_height, latent_width)` tensor per image of the edit, in packed order.
            latent_height (`int`): Target latent height.
            latent_width (`int`): Target latent width.
            num_audio_latents (`int`): Number of target audio latents per channel.
            patch_size (`tuple[int, int, int]`): The transformer's `(t, h, w)` patch.
            audio_channels (`int`): Channels the audio rows are packed channel-major over, i.e.
                `components.audio_channels`.
            audio_tag (`int`), video_tag (`int`):
                The modality tags an audio and an image row carry, i.e. `components.audio_tag` and
                `components.video_tag`.

        Returns:
            `tuple`: `position_ids`, `token_tags`, `video_indices`, `audio_indices`, `text_indices`, and the number of
            leading image and audio rows that are conditioning rather than generated.
        """
        _, patch_h, patch_w = patch_size
        num_text_tokens = text_token_tags.shape[0]
        num_condition_rows = sum(
            (latents.shape[3] // patch_h) * (latents.shape[4] // patch_w) for latents in condition_latents
        )
        num_audio_rows = num_audio_latents * audio_channels
        num_target_rows = (latent_height // patch_h) * (latent_width // patch_w)
        sequence_length = num_text_tokens + num_condition_rows + num_audio_rows + num_target_rows

        position_ids = torch.zeros(sequence_length, 3, dtype=torch.float64)
        position_ids[:num_text_tokens, 0] = torch.arange(num_text_tokens, dtype=torch.float64)

        # The images, in request order. Each takes a single integer slot on the rotary clock, which starts where the
        # text rows end.
        cursor = num_text_tokens
        rotary_time = float(num_text_tokens)
        for latents in condition_latents:
            frame_grid, _ = _frame_position_grid(latents.shape[3], latents.shape[4], patch_h, patch_w)
            rows = slice(cursor, cursor + frame_grid.shape[0])
            position_ids[rows, 0] = rotary_time
            position_ids[rows, 1:] = frame_grid
            cursor = rows.stop
            rotary_time += 1.0

        # The generated rows. Target audio and target image share the origin the images left behind.
        audio_start = cursor
        video_start = audio_start + num_audio_rows
        target_frame_grid, target_width_grid = _frame_position_grid(latent_height, latent_width, patch_h, patch_w)
        _fill_audio_positions(
            position_ids,
            slice(audio_start, video_start),
            num_audio_latents,
            rotary_time,
            target_width_grid,
            audio_channels,
        )
        position_ids[video_start:, 0] = rotary_time
        position_ids[video_start:, 1:] = target_frame_grid

        video_indices = torch.cat(
            [torch.arange(num_text_tokens, audio_start), torch.arange(video_start, sequence_length)]
        )
        audio_indices = torch.arange(audio_start, video_start)
        text_indices = torch.arange(num_text_tokens)

        token_tags = torch.empty(sequence_length, dtype=torch.long)
        token_tags[text_indices] = text_token_tags.to(torch.long)
        token_tags[audio_indices] = audio_tag
        token_tags[video_indices] = video_tag

        return position_ids, token_tags, video_indices, audio_indices, text_indices, num_condition_rows, 0

    @torch.no_grad()
    def __call__(
        self, components: PH3ImageModularPipeline, state: PipelineState
    ) -> tuple[PH3ImageModularPipeline, PipelineState]:
        block_state = self.get_block_state(state)
        device = components._execution_device

        multiple = components.canvas_multiple
        if block_state.height % multiple or block_state.width % multiple:
            raise ValueError(
                f"`height` and `width` must be multiples of {multiple}, got {block_state.height}x{block_state.width}."
            )
        ratio = components.vae_spatial_compression_ratio
        block_state.num_latent_frames = 1
        block_state.latent_height = block_state.height // ratio
        block_state.latent_width = block_state.width // ratio
        block_state.num_audio_latents = components.num_audio_latents

        (
            position_ids,
            token_tags,
            video_indices,
            audio_indices,
            text_indices,
            block_state.num_condition_video_rows,
            block_state.num_condition_audio_rows,
        ) = self.build_edit_packed_sequence(
            block_state.text_token_tags,
            block_state.condition_latents,
            block_state.latent_height,
            block_state.latent_width,
            block_state.num_audio_latents,
            components.patch_size,
            components.audio_channels,
            components.audio_tag,
            components.video_tag,
        )
        # MPS, NPU and Neuron have no float64; the rope casts the grid to float32 anyway.
        position_ids_dtype = maybe_adjust_dtype_for_device(position_ids.dtype, device)
        block_state.position_ids = position_ids.to(device, position_ids_dtype)
        block_state.token_tags = token_tags.to(device)
        block_state.video_indices = video_indices.to(device)
        block_state.audio_indices = audio_indices.to(device)
        block_state.text_indices = text_indices.to(device)

        self.set_block_state(state, block_state)
        return components, state


# Copied from diffusers.modular_pipelines.minimax_h3.before_denoise.MiniMaxH3PrepareConditionLatentsStep with \bMiniMaxH3->PH3Image, minimax-h3->p-h3-image, PH3ImageScheduler->MiniMaxH3Scheduler
class PH3ImagePrepareConditionLatentsStep(ModularPipelineBlocks):
    model_name = "p-h3-image"

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
    def __call__(
        self, components: PH3ImageModularPipeline, state: PipelineState
    ) -> tuple[PH3ImageModularPipeline, PipelineState]:
        block_state = self.get_block_state(state)
        device = components._execution_device
        patch_size = components.patch_size

        # One draw per condition, in packed order. Each is packed on its own because `ref2va` references are encoded
        # at their own resolutions, so their latents do not share a shape.
        packed = []
        for condition in block_state.condition_latents:
            noise = randn_tensor(condition.shape, generator=block_state.generator, device=device, dtype=torch.float32)
            # The anchors are not fully clean: the released model noises them to `t = 0.999` and holds them there
            # for every step. Mixing before the patchify is the same arithmetic, since patchify only permutes. The
            # level goes in as a tensor so `1 - t` is evaluated in float32, the way this port has always mixed it.
            noised = components.scheduler.scale_noise(
                condition.to(device), torch.tensor(components.keyframe_noise_aug), noise
            )
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


class PH3ImagePrepareLatentsStep(ModularPipelineBlocks):
    model_name = "p-h3-image"

    @property
    def description(self) -> str:
        return (
            "Draws the noise of the image rows, then of the audio rows MiniMax-H3's layout denoises alongside them, "
            "both off the request's generator, and packs them."
        )

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(name="latent_height", type_hint=int, required=True, description="Height of the image latents."),
            InputParam(name="latent_width", type_hint=int, required=True, description="Width of the image latents."),
            InputParam(
                name="num_audio_latents",
                type_hint=int,
                required=True,
                description="Number of audio latents per channel.",
            ),
            InputParam.template("generator"),
            InputParam(
                name="latents",
                type_hint=torch.Tensor,
                description=(
                    "Pre-generated image noise of shape `(1, latent_channels, 1, latent_height, latent_width)`, used "
                    "instead of the draw."
                ),
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "latents", type_hint=torch.Tensor, description="The generated image rows of the packed sequence."
            ),
            OutputParam(
                "audio_latents",
                type_hint=torch.Tensor,
                description="The generated audio rows of the packed sequence, channel-major.",
            ),
        ]

    @torch.no_grad()
    def __call__(
        self, components: PH3ImageModularPipeline, state: PipelineState
    ) -> tuple[PH3ImageModularPipeline, PipelineState]:
        block_state = self.get_block_state(state)
        device = components._execution_device

        latents = block_state.latents
        if latents is None:
            shape = (1, components.vae_latent_channels, 1, block_state.latent_height, block_state.latent_width)
            latents = randn_tensor(shape, generator=block_state.generator, device=device, dtype=torch.float32)
        block_state.latents = patchify_video_latents(latents.to(device, torch.float32), components.patch_size)
        block_state.audio_latents = randn_tensor(
            (block_state.num_audio_latents * components.audio_channels, components.audio_latent_channels),
            generator=block_state.generator,
            device=device,
            dtype=torch.float32,
        )

        self.set_block_state(state, block_state)
        return components, state


# Copied from diffusers.modular_pipelines.minimax_h3.before_denoise.MiniMaxH3FL2VAPrepareLatentsStep with MiniMaxH3FL2VA->PH3ImageEdit, \bMiniMaxH3->PH3Image, minimax-h3->p-h3-image
class PH3ImageEditPrepareLatentsStep(ModularPipelineBlocks):
    model_name = "p-h3-image"

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
    def __call__(
        self, components: PH3ImageModularPipeline, state: PipelineState
    ) -> tuple[PH3ImageModularPipeline, PipelineState]:
        block_state = self.get_block_state(state)

        block_state.latents = torch.cat([block_state.condition_rows, block_state.latents])

        self.set_block_state(state, block_state)
        return components, state


class PH3ImageSetTimestepsStep(ModularPipelineBlocks):
    model_name = "p-h3-image"

    @property
    def description(self) -> str:
        return (
            "Sets the image and audio schedulers to the checkpoint's sampling grid, which both modalities share, and "
            "plans the per-row timesteps of every step. The conditioning rows of an edit stay at their "
            "noise-augmentation level."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("scheduler", MiniMaxH3Scheduler),
            ComponentSpec("audio_scheduler", MiniMaxH3Scheduler),
        ]

    @property
    def expected_configs(self) -> list[ConfigSpec]:
        return [
            ConfigSpec(
                name="sample_sigmas",
                default=[1.0, 0.984753, 0.967905, 0.951688, 0.925372, 0.881923, 0.810083, 0.528718, 0.0],
                description="The checkpoint's sampling grid, used when `sigmas` is not passed.",
            ),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template(
                "sigmas",
                description=(
                    "A custom sampling grid, strictly decreasing and ending at 0. Defaults to the checkpoint's "
                    "`sample_sigmas`."
                ),
            ),
            InputParam(
                name="video_indices",
                type_hint=torch.Tensor,
                required=True,
                description="Sequence positions of the image rows, conditioning rows first.",
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
            InputParam(
                name="num_condition_video_rows",
                type_hint=int,
                default=0,
                description="How many leading image rows are conditioning rows.",
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
            OutputParam("timesteps", type_hint=torch.Tensor, description="Timesteps of the image schedule."),
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
    # Copied from diffusers.modular_pipelines.minimax_h3.before_denoise.MiniMaxH3SetTimestepsStep.build_row_timesteps
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
        # The layout step puts the indices on the execution device, but this plan is built on CPU.
        video_indices = video_indices.cpu()
        audio_indices = audio_indices.cpu()

        sequence_length = int(video_indices.numel() + audio_indices.numel() + num_text_tokens)
        row_timesteps = torch.full((sequence_length,), video_timestep, dtype=torch.float32)
        row_timesteps[video_indices[:num_condition_video_rows]] = condition_video_timestep
        row_timesteps[audio_indices[num_condition_audio_rows:]] = audio_timestep
        row_timesteps[audio_indices[:num_condition_audio_rows]] = condition_audio_timestep
        return torch.unique(row_timesteps, sorted=True, return_inverse=True)

    @torch.no_grad()
    def __call__(
        self, components: PH3ImageModularPipeline, state: PipelineState
    ) -> tuple[PH3ImageModularPipeline, PipelineState]:
        block_state = self.get_block_state(state)
        device = components._execution_device

        sigmas = block_state.sigmas if block_state.sigmas is not None else components.config.sample_sigmas
        components.scheduler.set_timesteps(sigmas=sigmas, device=device)
        components.audio_scheduler.set_timesteps(sigmas=sigmas, device=device)
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
