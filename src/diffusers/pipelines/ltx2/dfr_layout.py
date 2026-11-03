# Copyright 2025 Lightricks and The HuggingFace Team. All rights reserved.
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

"""Canvas layout for the DFR pipeline: keyframe segment grid, temporal tile plan, epilogue tiling, and the token plan a
tiled transformer call walks."""

import itertools
from collections.abc import Sequence
from typing import NamedTuple

import torch

from ...utils import logging


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# Candidate keyframe segment lengths, in *latent* frames -- 24 and 32 pixel frames on the LTX-2.5 VAE. The grid picks
# whichever pads the request least. A segment must be a whole number of latent frames because every keyframe sits on a
# latent border.
SEGMENT_LATENT_CANDIDATES = (3, 4)

# Latent-grid overlap between the spatial epilogue's height and width tiles. Its tiles agree on their overlaps at
# every denoising step, so the overlap only has to be wide enough to blend the seam away.
EPILOGUE_SPATIAL_OVERLAP = 12


class DimensionInterval(NamedTuple):
    """
    One tile's extent along a single axis, in that axis' own units.

    `start` / `end` are half-open. `left_ramp` / `right_ramp` are the lengths of the regions at each end that overlap
    the neighbouring tile. How a ramp is resolved is the caller's choice: the temporal tile plan cuts on keyframe seams
    and drops the ramp outright (both tiles reproduce a known frame there, so averaging only smears it), while the
    spatial epilogue blends its ramps with a trapezoidal mask.
    """

    start: int
    end: int
    left_ramp: int
    right_ramp: int


class LTX2DFRTemporalTile(NamedTuple):
    """
    One temporal-refine window: its latent interval plus the pixel-frame keyframes it carries.

    `pixel_start` / `pixel_end` are inclusive pixel-frame bounds of `interval`. `anchors` are the seam keyframes inside
    the window, carried in from the previous round; on every tile but the first this includes `pixel_start` itself.
    `slots` are the mid-segment positions this window invents.
    """

    interval: DimensionInterval
    pixel_start: int
    pixel_end: int
    anchors: tuple[int, ...]
    slots: tuple[int, ...]


class LTX2DFREpilogueTile(NamedTuple):
    """
    One spatial-epilogue tile: its extent on each latent axis plus the window its prediction is weighted by.

    `frames` / `heights` / `widths` are half-open slices into the latent grid. `blend_weight` is the separable `(F, H,
    W)` window over that extent; adjacent tiles' windows sum to exactly one, so accumulating `tile * blend_weight`
    reconstructs the canvas without a normalization pass.
    """

    frames: slice
    heights: slice
    widths: slice
    blend_weight: torch.Tensor


class LTX2DFRTokenPlan(NamedTuple):
    """
    One tile's slice of a packed token sequence, ready for a tiled transformer call.

    `keep` are the token indices the tile processes, base-grid tokens first. `weights` carries one blend weight per
    kept token. `coords` are those tokens' positions, rebased on the tile's own first base token.
    """

    keep: torch.Tensor
    weights: torch.Tensor
    coords: torch.Tensor


def choose_segment_length(content_frames: int, temporal_compression_ratio: int = 8) -> int:
    """
    Pick the keyframe segment length, in pixel frames, that pads `content_frames` least.

    Args:
        content_frames (`int`):
            `num_frames - 1`, the frame count the segment grid has to cover.
        temporal_compression_ratio (`int`, defaults to `8`):
            The VAE's temporal compression ratio. Candidates are `SEGMENT_LATENT_CANDIDATES` scaled by it, so the
            shipped LTX-2.5 VAE offers 24 and 32 pixel frames.

    Returns:
        `int`: the chosen segment length in pixel frames. Ties keep the larger segment.
    """
    candidates = [candidate * temporal_compression_ratio for candidate in SEGMENT_LATENT_CANDIDATES]
    return max(candidates, key=lambda candidate: (-((candidate - content_frames % candidate) % candidate), candidate))


def resolve_canvas(num_frames: int, temporal_compression_ratio: int = 8) -> tuple[int, int, list[int]]:
    """
    Pad `num_frames - 1` up to a multiple of the keyframe segment length.

    Args:
        num_frames (`int`):
            Requested pixel frame count. Must satisfy `(num_frames - 1) % temporal_compression_ratio == 0` and be at
            least `temporal_compression_ratio + 1`.
        temporal_compression_ratio (`int`, defaults to `8`):
            The VAE's temporal compression ratio.

    Returns:
        `tuple[int, int, list[int]]`: the padded frame count, the chosen segment length, and the keyframe positions
        `[S, 2S, ..., N' - 1]`. Frame 0 is excluded (under causal encoding its latent already covers a single pixel
        frame) and the terminal frame is included.
    """
    if (num_frames - 1) % temporal_compression_ratio != 0:
        raise ValueError(
            f"`num_frames` must satisfy (num_frames - 1) % {temporal_compression_ratio} == 0, got {num_frames}"
        )
    content = num_frames - 1
    if content < temporal_compression_ratio:
        raise ValueError(f"The DFR canvas needs at least {temporal_compression_ratio + 1} pixel frames")

    segment = choose_segment_length(content, temporal_compression_ratio)
    content_padded = content + (segment - content % segment) % segment
    positions = [segment * index for index in range(1, content_padded // segment + 1)]
    return content_padded + 1, segment, positions


def pixel_to_latent_index(pixel_frame: int, temporal_compression_ratio: int = 8) -> int:
    """Map a pixel frame sitting on a latent border to its latent index."""
    if pixel_frame < 0:
        raise ValueError(f"`pixel_frame` must be >= 0, got {pixel_frame}")
    if pixel_frame % temporal_compression_ratio != 0:
        raise ValueError(f"Pixel frame {pixel_frame} is not on the x{temporal_compression_ratio} latent border")
    return pixel_frame // temporal_compression_ratio


def split_canvas_at_seams(
    seams: Sequence[int], num_tiles: int, overlap: int, dim_size: int
) -> list[DimensionInterval]:
    """
    Split a canvas on keyframe boundary cells, tolerating a remainder segment count.

    Each tile but the first starts `overlap` cells before the boundary it resumes after. That run-up is context only:
    it lands in the interval's `left_ramp`, and the temporal rounds drop it, so the earlier tile keeps the boundary
    cell and this one contributes strictly after it.

    Args:
        seams (`Sequence[int]`):
            The `K + 1` segment edges in grid cells, starting at `0` and ending at `dim_size - 1`.
        num_tiles (`int`):
            Requested number of tiles, clamped to the segment count `K`.
        overlap (`int`):
            Run-up each non-first tile reaches back, in grid cells.
        dim_size (`int`):
            Length of the axis being split, in grid cells.
    """
    seams = tuple(int(seam) for seam in seams)
    if overlap < 0:
        raise ValueError(f"`overlap` must be >= 0, got {overlap}")
    if len(seams) < 2 or seams[0] != 0:
        raise ValueError(f"`seams` must start at 0 and hold at least one segment, got {list(seams)}")
    if any(later <= earlier for earlier, later in itertools.pairwise(seams)):
        raise ValueError(f"`seams` must be strictly increasing, got {list(seams)}")
    if seams[-1] != dim_size - 1:
        raise ValueError(f"`seams` must end at the last cell ({dim_size - 1}), got {seams[-1]}")

    if num_tiles < 1:
        raise ValueError(f"`num_tiles` must be >= 1, got {num_tiles}")
    # Deal the segments out so the leftovers land on the leading tiles. A tile cannot own zero segments, so more tiles
    # than segments collapses to one tile per segment.
    num_segments = len(seams) - 1
    num_tiles = min(num_tiles, num_segments)
    base, remainder = divmod(num_segments, num_tiles)
    counts = [base + (1 if index < remainder else 0) for index in range(num_tiles)]

    intervals = []
    cursor = 0
    for tile_index, count in enumerate(counts):
        resume = seams[cursor] + 1
        start = 0 if tile_index == 0 else max(0, resume - overlap)
        cursor += count
        intervals.append(
            DimensionInterval(
                start=start,
                end=seams[cursor] + 1,
                left_ramp=0 if tile_index == 0 else resume - start,
                right_ramp=0,
            )
        )
    return intervals


def temporal_tile_plan(
    seam_positions: Sequence[int],
    num_frames: int,
    num_tiles: int,
    temporal_compression_ratio: int = 8,
) -> list[LTX2DFRTemporalTile]:
    """
    Partition one temporal-refine round's canvas into keyframe-seam tiles.

    The overlap is one canvas segment plus the shared seam cell, so a tile's local latent 0 is the seam it inherits and
    its lead-in covers the image latent the keyframe-at-0 lock produces -- neither may be spliced into the mid-canvas
    stream, and both fall inside `interval.left_ramp`.

    Args:
        seam_positions (`Sequence[int]`):
            Keyframe pixel positions on this round's grid, strictly increasing, ending on `num_frames - 1`.
        num_frames (`int`):
            Pixel frame count of this round's canvas.
        num_tiles (`int`):
            Requested number of tiles, clamped to the segment count.
        temporal_compression_ratio (`int`, defaults to `8`):
            The VAE's temporal compression ratio.
    """
    seams = [0, *(pixel_to_latent_index(position, temporal_compression_ratio) for position in seam_positions)]
    latent_length = (num_frames - 1) // temporal_compression_ratio + 1
    overlap = seams[1] - seams[0] + 1
    tiles = []
    for interval in split_canvas_at_seams(seams, num_tiles, overlap, latent_length):
        pixel_start = interval.start * temporal_compression_ratio
        pixel_end = (interval.end - 1) * temporal_compression_ratio
        anchors = tuple(position for position in seam_positions if pixel_start <= position <= pixel_end)
        marks = [pixel_start, *(position for position in seam_positions if pixel_start < position <= pixel_end)]
        slots = tuple((left + right) // 2 for left, right in itertools.pairwise(marks))
        tiles.append(LTX2DFRTemporalTile(interval, pixel_start, pixel_end, anchors, slots))
    return tiles


def split_by_count(dim_size: int, num_tiles: int, overlap: int) -> list[DimensionInterval]:
    """
    Split an axis into `num_tiles` evenly sized tiles sharing `overlap` cells with each neighbour.

    Leading tiles absorb the remainder, so the tiles cover `[0, dim_size)` exactly and every adjacent pair shares
    precisely `overlap` cells -- which is what makes `trapezoidal_mask_1d` sum to one across the seam.
    """
    if num_tiles == 1:
        return [DimensionInterval(start=0, end=dim_size, left_ramp=0, right_ramp=0)]

    total = dim_size + overlap * (num_tiles - 1)
    tile_size, remainder = divmod(total, num_tiles)
    if tile_size <= overlap:
        raise ValueError(
            f"Tile size {tile_size} is not larger than the overlap {overlap} for dim_size={dim_size}, "
            f"num_tiles={num_tiles}"
        )
    stride = tile_size - overlap

    intervals = []
    for index in range(num_tiles):
        shift = min(index, remainder)
        grow = 1 if index < remainder else 0
        intervals.append(
            DimensionInterval(
                start=index * stride + shift,
                end=index * stride + tile_size + shift + grow,
                left_ramp=0 if index == 0 else overlap,
                right_ramp=0 if index == num_tiles - 1 else overlap,
            )
        )
    return intervals


def trapezoidal_mask_1d(length: int, left_ramp: int, right_ramp: int) -> torch.Tensor:
    """
    Build a `(length,)` blending weight that fades in over `left_ramp` cells and out over `right_ramp` cells.

    The ramps are the interior of a `linspace`, so two tiles sharing `k` cells contribute `i / (k + 1)` and `(k + 1 -
    i) / (k + 1)` there and their weights sum to exactly one.
    """
    mask = torch.ones(length)
    if left_ramp > 0:
        mask[:left_ramp] *= torch.linspace(0.0, 1.0, left_ramp + 2)[1:-1]
    if right_ramp > 0:
        mask[-right_ramp:] *= torch.linspace(1.0, 0.0, right_ramp + 2)[1:-1]
    return mask


def rectangular_mask_1d(length: int, left_ramp: int) -> torch.Tensor:
    """
    Build a `(length,)` blending weight that drops `left_ramp` cells outright and keeps the rest at full weight.

    This is the seam cut's counterpart to `trapezoidal_mask_1d`. Where a ramp is what two tiles need when neither knows
    the truth at the border, a seam is a cell both of them reproduce from the same keyframe, so the later tile discards
    its run-up instead of averaging it into the earlier tile's answer.
    """
    mask = torch.ones(length)
    mask[:left_ramp] = 0.0
    return mask


def epilogue_tiles(
    latent_shape: tuple[int, int, int], frame_tiles: int, frame_seams: Sequence[int]
) -> list[LTX2DFREpilogueTile]:
    """
    Lay the spatial detailing epilogue's `(frames, height, width)` tiling over a latent grid.

    Height and width are always split in two with `EPILOGUE_SPATIAL_OVERLAP` cells of trapezoidal overlap. The temporal
    axis is cut on `frame_seams` instead: those cells carry a keyframe the last refine round already settled, so both
    neighbours reproduce the same content there and the later tile drops its run-up rather than averaging it into the
    earlier tile's answer. The run-up itself is one segment plus the shared seam cell, the same handover the refine
    rounds use. Seams that are not interior to the grid leave the axis blended.

    An axis too short to hold its tile count falls back to a single tile, and an overlap it cannot hold is clamped,
    since evenly sized tiles need `overlap < tile_size`.

    Returns one `(frame_slice, height_slice, width_slice, blend_weight)` per tile. `blend_weight` is the separable
    window over the tile's `(F, H, W)` extent; adjacent windows sum to exactly one, so accumulating `tile *
    blend_weight` reconstructs the canvas without a normalization pass.
    """
    frame_size = latent_shape[0]
    frame_overlap = frame_seams[0] + 1 if frame_tiles > 1 and frame_seams else 0

    axis_configs = []
    for dim_size, num_tiles, overlap, axis in zip(
        latent_shape,
        (frame_tiles, 2, 2),
        (frame_overlap, EPILOGUE_SPATIAL_OVERLAP, EPILOGUE_SPATIAL_OVERLAP),
        ("frames", "height", "width"),
    ):
        if num_tiles > 1 and dim_size < num_tiles:
            logger.warning(
                f"Spatial epilogue: latent {axis} is {dim_size} cells, too short for {num_tiles} tiles; running "
                f"this axis untiled."
            )
            num_tiles, overlap = 1, 0
        elif num_tiles > 1 and overlap > dim_size - num_tiles:
            logger.warning(
                f"Spatial epilogue: latent {axis} is {dim_size} cells, so an overlap of {overlap} leaves no room "
                f"for {num_tiles} tiles; clamping the overlap to {dim_size - num_tiles}."
            )
            overlap = dim_size - num_tiles
        axis_configs.append((dim_size, num_tiles, overlap))

    _, frame_num_tiles, frame_overlap = axis_configs[0]
    interior_seams = sorted({cell for cell in frame_seams if 0 < cell < frame_size - 1})
    seam_cut = frame_num_tiles > 1 and bool(interior_seams)
    if seam_cut:
        frame_intervals = split_canvas_at_seams(
            [0, *interior_seams, frame_size - 1], frame_num_tiles, frame_overlap, frame_size
        )
    else:
        if frame_seams and frame_num_tiles > 1:
            logger.warning(
                f"Spatial epilogue: seams {list(frame_seams)} are not interior to {frame_size} latent frames; "
                f"blending the temporal tiles instead of cutting them."
            )
        frame_intervals = split_by_count(frame_size, frame_num_tiles, frame_overlap)

    tiles = []
    for frames, heights, widths in itertools.product(
        frame_intervals, *(split_by_count(*config) for config in axis_configs[1:])
    ):
        frame_mask = (
            rectangular_mask_1d(frames.end - frames.start, frames.left_ramp)
            if seam_cut
            else trapezoidal_mask_1d(frames.end - frames.start, frames.left_ramp, frames.right_ramp)
        )
        height_mask = trapezoidal_mask_1d(heights.end - heights.start, heights.left_ramp, heights.right_ramp)
        width_mask = trapezoidal_mask_1d(widths.end - widths.start, widths.left_ramp, widths.right_ramp)
        tiles.append(
            LTX2DFREpilogueTile(
                frames=slice(frames.start, frames.end),
                heights=slice(heights.start, heights.end),
                widths=slice(widths.start, widths.end),
                blend_weight=frame_mask[:, None, None] * height_mask[None, :, None] * width_mask[None, None, :],
            )
        )
    return tiles


def video_tile_plan(
    tiles: list[LTX2DFREpilogueTile],
    video_coords: torch.Tensor,
    latent_num_frames: int,
    latent_height: int,
    latent_width: int,
) -> list[LTX2DFRTokenPlan]:
    """
    Resolve `tiles` into the per-tile token plan a tiled transformer call needs.

    A tile takes the base-grid tokens inside its `(F, H, W)` window plus every appended conditioning token whose RoPE
    interval overlaps that window on all three axes -- token-level filtering, so a keyframe or reference token reaches
    exactly the tiles whose picture it describes.

    Positions are rebased on the tile's own first base token, matching what a standalone pass over that crop would have
    built. Base tokens carry the tile's separable blend window; a conditioning token kept by `n` tiles carries `1 / n`,
    so both groups sum to one across the tiling and the blended prediction needs no normalization pass.

    Returns one dict per tile: `keep` (token indices, base tokens first), `weights` (one per kept token) and `coords`
    (the kept tokens' rebased positions).
    """
    tokens_per_latent_frame = latent_height * latent_width
    num_base_tokens = latent_num_frames * tokens_per_latent_frame
    device = video_coords.device
    condition_coords = video_coords[:, :, num_base_tokens:, :]

    base_indices, tile_starts, kept_conditions = [], [], []
    for frames, heights, widths, _ in tiles:
        frame_range = torch.arange(frames.start, frames.stop, device=device)
        height_range = torch.arange(heights.start, heights.stop, device=device)
        width_range = torch.arange(widths.start, widths.stop, device=device)
        base = (
            frame_range[:, None, None] * tokens_per_latent_frame
            + height_range[None, :, None] * latent_width
            + width_range[None, None, :]
        ).reshape(-1)
        base_indices.append(base)

        # The window's extent in RoPE units, read off the tokens it owns rather than recomputed, so it stays in
        # step with however `prepare_video_coords` laid the canvas out.
        base_coords = video_coords[:, :, base, :]
        starts = base_coords[..., 0].amin(dim=2)
        ends = base_coords[..., 1].amax(dim=2)
        tile_starts.append(starts)
        overlaps = (condition_coords[..., 0] < ends[..., None]) & (condition_coords[..., 1] > starts[..., None])
        kept_conditions.append(overlaps.all(dim=1).any(dim=0))

    condition_keepers = torch.stack(kept_conditions).sum(dim=0)
    if not bool((condition_keepers > 0).all()):
        unclaimed = (condition_keepers == 0).nonzero(as_tuple=False).squeeze(1).tolist()
        raise ValueError(
            f"Conditioning tokens {unclaimed} fall outside every tile, so no tile would denoise them. Their RoPE "
            f"position lies off the canvas the tiling covers."
        )

    plan = []
    for tile, base, starts, keep_condition in zip(tiles, base_indices, tile_starts, kept_conditions):
        condition_indices = num_base_tokens + keep_condition.nonzero(as_tuple=False).squeeze(1)
        keep = torch.cat([base, condition_indices])
        weights = torch.cat(
            [
                tile.blend_weight.reshape(-1).to(device=device),
                1.0 / condition_keepers[keep_condition].to(device=device, dtype=tile.blend_weight.dtype),
            ]
        )
        plan.append(
            LTX2DFRTokenPlan(
                keep=keep,
                weights=weights,
                coords=video_coords[:, :, keep, :] - starts[..., None, None],
            )
        )
    return plan
