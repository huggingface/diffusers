# Copyright 2026 Lightricks and The HuggingFace Team. All rights reserved.
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
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...configuration_utils import ConfigMixin, register_to_config
from ...utils import is_kernels_available, logging
from ...utils.accelerate_utils import apply_forward_hook
from ...utils.constants import DIFFUSERS_DISABLE_REMOTE_CODE
from ...utils.torch_utils import maybe_adjust_dtype_for_device, randn_tensor
from ..attention import AttentionMixin, AttentionModuleMixin
from ..attention_dispatch import dispatch_attention_fn
from ..embeddings import PixArtAlphaCombinedTimestepSizeEmbeddings
from ..modeling_utils import ModelMixin
from .vae import DecoderOutput


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def _patchify(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Space-to-depth on H/W only: `(B, C, F, H, W)` -> `(B, C * patch_size**2, F, H // p, W // p)`.

    The channel packing order is `(channel, width_offset, height_offset)`, matching the reference implementation's `b c
    (f p) (h q) (w r) -> b (c p r q) f h w` with `p = 1`.
    """
    batch_size, num_channels, num_frames, height, width = x.shape
    x = x.reshape(
        batch_size, num_channels, num_frames, height // patch_size, patch_size, width // patch_size, patch_size
    )
    x = x.permute(0, 1, 6, 4, 2, 3, 5)
    return x.reshape(
        batch_size, num_channels * patch_size * patch_size, num_frames, height // patch_size, width // patch_size
    )


def _unpatchify(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Depth-to-space on H/W only, the exact inverse of [`_patchify`]."""
    batch_size, num_channels, num_frames, height, width = x.shape
    num_channels = num_channels // (patch_size * patch_size)
    x = x.reshape(batch_size, num_channels, patch_size, patch_size, num_frames, height, width)
    x = x.permute(0, 1, 4, 5, 3, 6, 2)
    return x.reshape(batch_size, num_channels, num_frames, height * patch_size, width * patch_size)


def _neighborhood_block_mask(
    num_frames: int, height: int, width: int, kernel_size: tuple[int, int, int], device: torch.device
):
    """Build a FlexAttention `BlockMask` for 3D neighborhood attention.

    Each query attends to a `kernel_size` window that is centered where possible and *shifted inward* at the grid
    boundaries so it always holds exactly `kernel_size` positions. That inward shift (rather than truncating the
    window) is what NATTEN's `na3d` does, which is why the flex path can stand in for it.
    """
    from torch.nn.attention.flex_attention import create_block_mask

    kernel_t, kernel_h, kernel_w = kernel_size
    kernel_t, kernel_h, kernel_w = min(kernel_t, num_frames), min(kernel_h, height), min(kernel_w, width)
    hw = height * width

    def mask_mod(batch_idx, head_idx, q_idx, kv_idx):
        q_t, q_rem = q_idx // hw, q_idx % hw
        q_h, q_w = q_rem // width, q_rem % width
        k_t, k_rem = kv_idx // hw, kv_idx % hw
        k_h, k_w = k_rem // width, k_rem % width

        start_t = torch.clamp(q_t - kernel_t // 2, 0, num_frames - kernel_t)
        start_h = torch.clamp(q_h - kernel_h // 2, 0, height - kernel_h)
        start_w = torch.clamp(q_w - kernel_w // 2, 0, width - kernel_w)
        window_t = (k_t >= start_t) & (k_t < start_t + kernel_t)
        window_h = (k_h >= start_h) & (k_h < start_h + kernel_h)
        window_w = (k_w >= start_w) & (k_w < start_w + kernel_w)
        return window_t & window_h & window_w

    seq_len = num_frames * hw
    return create_block_mask(mask_mod, B=None, H=None, Q_LEN=seq_len, KV_LEN=seq_len, device=device)


# --------------------------------------------------------------------------------------------------------------------
# Keyframe-aware decoding
#
# A keyframe decode carries a second stream through the decoder: a stack of single-frame latent *planes* `(B, P, H,
# W, C)` whose plane axis sits in the video's temporal slot. Every weight is shared between the two streams, and they
# only mix inside one joint neighborhood-attention softmax, where each video query also sees the same spatial window
# on its two nearest planes and each plane query also sees its two nearest video frames.
# --------------------------------------------------------------------------------------------------------------------

# Keyframe planes visible to one video query, and video frames visible to one plane query.
_KEYFRAME_CONTEXT_SLOTS = 2


def _keyframe_stage_times(pixel_frame_indices: torch.Tensor, remaining_time_stride: int) -> torch.Tensor:
    """Chunk-center position of each keyframe plane in the temporal units of one decoder stage.

    A stage whose remaining temporal upsampling is `r` has cells covering `r` pixel frames each, except cell 0 which
    covers only pixel frame 0 (the causal first frame). So `t(0) = 0` and `t(f) = (f + (r - 1) / 2) / r`, the center of
    the cell holding `f`. In the diffusion stage `r == 1`, making the times the raw pixel indices.
    """
    frames = pixel_frame_indices.to(torch.float32)
    times = (frames + (remaining_time_stride - 1) / 2) / remaining_time_stride
    return torch.where(frames == 0, torch.zeros_like(times), times)


def _keyframe_planes_for_tile(pixel_frame_indices: torch.Tensor, frame_lo: int, frame_hi: int) -> torch.Tensor:
    """`(P,)` bool: the planes a tile spanning pixel frames `[frame_lo, frame_hi]` (inclusive) has to carry.

    Every plane inside the span plus the nearest plane on each side outside it. The two outside planes are what keep a
    tiled decode consistent with a whole one: a frame near a tile edge ranks its planes by temporal distance, so
    dropping the closest plane beyond the edge would make it attend to a farther one instead.
    """
    indices = pixel_frame_indices.to(torch.int64)
    keep = (indices >= frame_lo) & (indices <= frame_hi)
    before = indices < frame_lo
    if bool(before.any()):
        keep[int(torch.where(before, indices, torch.full_like(indices, -1)).argmax())] = True
    after = indices > frame_hi
    if bool(after.any()):
        sentinel = int(indices.max()) + 1
        keep[int(torch.where(after, indices, torch.full_like(indices, sentinel)).argmin())] = True
    return keep


def _nearest_slots(query_times: torch.Tensor, candidate_times: torch.Tensor, num_slots: int) -> torch.Tensor:
    """`(Q, num_slots)` candidate indices ranked by `(|dt|, index)`, `-1` where there are fewer candidates."""
    distances = (query_times[:, None] - candidate_times[None, :]).abs().to(torch.float32)
    # A stable sort breaks distance ties by ascending candidate index.
    order = torch.argsort(distances, dim=-1, stable=True)
    take = min(num_slots, candidate_times.shape[0])
    chosen = order[:, :take]
    if take < num_slots:
        pad = torch.full((chosen.shape[0], num_slots - take), -1, dtype=chosen.dtype, device=chosen.device)
        chosen = torch.cat([chosen, pad], dim=1)
    return chosen


def _upsample_keyframe_planes(upsample: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    """Spatially upsample keyframe planes `(B, P, H, W, C)` with the video stream's upsampler.

    Each plane goes through as its own one-frame clip with the leading frame always dropped, so a temporal stride of 2
    expands it to two frames and takes it back to one: the plane count never changes, only `H` and `W` grow.
    """
    batch_size, num_planes = hidden_states.shape[:2]
    flat = hidden_states.reshape(batch_size * num_planes, 1, *hidden_states.shape[2:])
    upsampled = upsample(flat, drop_leading_frame=True)
    return upsampled.reshape(batch_size, num_planes, *upsampled.shape[2:])


# Joint neighborhood attention. Queries are grouped into `(bt, bh, bw)` bricks and many bricks share one
# `scaled_dot_product_attention` call as its batch dimension. All queries in a brick share one gathered key slab, the
# visible-key pattern is one mask shared by every brick, and per-key validity (outside the volume, an empty slot)
# rides in an extra key channel that adds `_JOINT_DEAD_KEY` to the score of a dead key.
_JOINT_DEAD_KEY = -1.0e4
# Keep the query/key head dim a multiple of this, so SDPA can keep a fused kernel once the bias channel is added.
_JOINT_HEAD_DIM_ALIGN = 8
_JOINT_BRICK_QUERIES = 64
_JOINT_BRICK_DEPTH = 4
# Transient memory budget of one staging pass and one key/value block.
_JOINT_WORKSPACE_BYTES = 256 * 1024**2
# Peak-to-staging multipliers for SDPA kernels that keep the scores on chip, and for those that materialize them.
_JOINT_STAGING_FACTOR_FUSED = 4.75
_JOINT_STAGING_FACTOR_MATERIALIZED = 22.1


def _joint_key_channels(head_dim: int) -> int:
    return -(-(head_dim + 1) // _JOINT_HEAD_DIM_ALIGN) * _JOINT_HEAD_DIM_ALIGN


def _joint_window(kernel: int) -> tuple[int, int]:
    """`(lo, hi)` halo of one axis: a centered window of `kernel` positions."""
    lo = kernel // 2
    return lo, kernel - lo - 1


class _JointGeometry:
    """Brick decomposition of one `(H, W)` grid, plus the padding and slab extents it implies."""

    def __init__(self, height: int, width: int, kernel: tuple[int, int, int], brick: tuple[int, int, int]):
        kernel_t, kernel_h, kernel_w = kernel
        lo_h, hi_h = _joint_window(kernel_h)
        lo_w, hi_w = _joint_window(kernel_w)
        self.height, self.width = height, width
        self.brick = brick
        self.kernel = kernel
        self.grid = (-(-height // brick[1]), -(-width // brick[2]))
        self.span_t = brick[0] + kernel_t - 1
        self.span = (brick[1] + kernel_h - 1, brick[2] + kernel_w - 1)
        self.pad_h = (lo_h, hi_h + self.grid[0] * brick[1] - height)
        self.pad_w = (lo_w, hi_w + self.grid[1] * brick[2] - width)
        self.pad_t = _joint_window(kernel_t)
        self.queries = brick[0] * brick[1] * brick[2]
        self.footprint = self.span[0] * self.span[1]
        self.padded_height = height + sum(self.pad_h)
        self.padded_width = width + sum(self.pad_w)

    def row_extent(self, rows: int) -> int:
        return (rows - 1) * self.brick[1] + self.span[0]


class _JointSchedule:
    """How the loops are cut so transient memory stays within `_JOINT_WORKSPACE_BYTES`."""

    def __init__(
        self,
        geometry: _JointGeometry,
        blocks: int,
        heads: int,
        head_dim: int,
        axis_bricks: int,
        element_size: int,
        factor: float,
    ):
        channels = _joint_key_channels(head_dim)
        keys = blocks * geometry.footprint
        pair_bytes = geometry.grid[1] * heads * keys * (channels + head_dim) * element_size
        pairs = max(1, int(_JOINT_WORKSPACE_BYTES / max(pair_bytes * factor, 1.0)))
        if pairs >= geometry.grid[0]:
            self.group_axis = min(axis_bricks, max(1, pairs // geometry.grid[0]))
            self.group_rows = geometry.grid[0]
        else:
            self.group_axis = 1
            self.group_rows = pairs
        staged = geometry.padded_height * geometry.padded_width * heads * (channels + head_dim) * element_size
        per_axis_brick = staged * geometry.brick[0]
        self.stage_axis = min(axis_bricks, max(self.group_axis, _JOINT_WORKSPACE_BYTES // max(per_axis_brick, 1)))


def _joint_banded(queries: int, span: int, kernel: int, device: torch.device) -> torch.Tensor:
    key = torch.arange(span, device=device)[None, :]
    query = torch.arange(queries, device=device)[:, None]
    return (key >= query) & (key < query + kernel)


def _joint_mask(geometry: _JointGeometry, num_slots: int, device: torch.device) -> torch.Tensor:
    """`(1, 1, Nq, Nk)` visibility shared by every brick: the video slab, then `num_slots` plane slabs.

    Plane keys carry no temporal condition, which is what lets a whole brick share one set of planes.
    """
    brick_t, brick_h, brick_w = geometry.brick
    kernel_t, kernel_h, kernel_w = geometry.kernel
    spatial = (
        _joint_banded(brick_h, geometry.span[0], kernel_h, device)[:, None, :, None]
        & _joint_banded(brick_w, geometry.span[1], kernel_w, device)[None, :, None, :]
    ).reshape(brick_h * brick_w, geometry.footprint)
    temporal = _joint_banded(brick_t, geometry.span_t, kernel_t, device)
    video = (temporal[:, None, :, None] & spatial[None, :, None, :]).reshape(
        geometry.queries, geometry.span_t * geometry.footprint
    )
    planes = (
        spatial[None, :, None, :]
        .expand(brick_t, brick_h * brick_w, num_slots, geometry.footprint)
        .reshape(geometry.queries, num_slots * geometry.footprint)
    )
    return torch.cat([video, planes], dim=1)[None, None].contiguous()


def _joint_stage(
    x: torch.Tensor, geometry: _JointGeometry, pad_t: tuple[int, int], with_bias_channel: bool
) -> torch.Tensor:
    """`(B, A, H, W, heads, head_dim)` to a padded head-major `(B, heads, A + pad, Hp, Wp, C)`."""
    batch, axis, height, width, heads, head_dim = x.shape
    channels = _joint_key_channels(head_dim) if with_bias_channel else head_dim
    out = x.new_zeros((batch, heads, axis + sum(pad_t), geometry.padded_height, geometry.padded_width, channels))
    if with_bias_channel:
        out[..., head_dim] = _JOINT_DEAD_KEY
    live = out[
        :,
        :,
        pad_t[0] : pad_t[0] + axis,
        geometry.pad_h[0] : geometry.pad_h[0] + height,
        geometry.pad_w[0] : geometry.pad_w[0] + width,
    ]
    live[..., :head_dim] = x.permute(0, 4, 1, 2, 3, 5)
    if with_bias_channel:
        live[..., head_dim] = 0.0
    return out


def _joint_slabs(
    staged: torch.Tensor, geometry: _JointGeometry, bricks: int, rows: int, blocks: int, group_stride: int
) -> torch.Tensor:
    """Overlapping brick slabs as a view, `(B, bricks, rows, Gw, heads, blocks, eh, ew, C)`."""
    batch, heads = staged.shape[0], staged.shape[1]
    stride_b, stride_nh, stride_a, stride_h, stride_w, _ = staged.stride()
    return staged.as_strided(
        (batch, bricks, rows, geometry.grid[1], heads, blocks, *geometry.span, staged.shape[-1]),
        (
            stride_b,
            group_stride * stride_a,
            geometry.brick[1] * stride_h,
            geometry.brick[2] * stride_w,
            stride_nh,
            stride_a,
            stride_h,
            stride_w,
            1,
        ),
    )


def _joint_query_bricks(x: torch.Tensor, geometry: _JointGeometry, bricks: int, rows: int) -> torch.Tensor:
    """`(B, A, h, W, heads, head_dim)` to `(B * bricks * rows * Gw, heads, Nq, C)`, with a constant bias channel."""
    batch, axis, height, width, heads, head_dim = x.shape
    brick_t, brick_h, brick_w = geometry.brick
    pad_t, pad_h, pad_w = bricks * brick_t - axis, rows * brick_h - height, geometry.grid[1] * brick_w - width
    if pad_t or pad_h or pad_w:
        x = F.pad(x, (0, 0, 0, 0, 0, pad_w, 0, pad_h, 0, pad_t))
    bricked = (
        x.reshape(batch, bricks, brick_t, rows, brick_h, geometry.grid[1], brick_w, heads, head_dim)
        .permute(0, 1, 3, 5, 7, 2, 4, 6, 8)
        .reshape(batch * bricks * rows * geometry.grid[1], heads, geometry.queries, head_dim)
    )
    out = bricked.new_zeros((*bricked.shape[:-1], _joint_key_channels(head_dim)))
    out[..., :head_dim] = bricked
    out[..., head_dim] = 1.0
    return out


def _joint_unbrick(
    attended: torch.Tensor, geometry: _JointGeometry, batch: int, bricks: int, rows: int, extent: tuple[int, int]
) -> torch.Tensor:
    brick_t, brick_h, brick_w = geometry.brick
    heads, head_dim = attended.shape[1], attended.shape[3]
    plane = (
        attended.reshape(batch, bricks, rows, geometry.grid[1], heads, brick_t, brick_h, brick_w, head_dim)
        .permute(0, 1, 5, 2, 6, 3, 7, 4, 8)
        .reshape(batch, bricks * brick_t, rows * brick_h, geometry.grid[1] * brick_w, heads, head_dim)
    )
    return plane[:, : extent[0], : extent[1], : geometry.width]


def _joint_with_null(slots: torch.Tensor, null_index: int) -> torch.Tensor:
    return torch.where(slots < 0, torch.full_like(slots, null_index), slots)


def _joint_slot_runs(slots: torch.Tensor) -> list[tuple[int, int]]:
    """Maximal `[start, stop)` runs of leading-axis positions whose slot row is identical."""
    rows = slots.tolist()
    runs = []
    start = 0
    for index in range(1, len(rows)):
        if rows[index] != rows[start]:
            runs.append((start, index))
            start = index
    runs.append((start, len(rows)))
    return runs


def _joint_attend_group(
    query_slice: torch.Tensor,
    key_views: tuple[torch.Tensor, ...],
    value_views: tuple[torch.Tensor, ...],
    geometry: _JointGeometry,
    shape: tuple[int, int],
    mask: torch.Tensor,
) -> torch.Tensor:
    """Gather one `(bricks, brick rows)` block's keys, attend, and un-brick the result."""
    bricks, rows = shape
    batch = query_slice.shape[0]
    heads, head_dim = query_slice.shape[4], query_slice.shape[5]
    blocks = sum(view.shape[5] for view in key_views)
    channels = _joint_key_channels(head_dim)
    keys = query_slice.new_empty((batch, bricks, rows, geometry.grid[1], heads, blocks, *geometry.span, channels))
    values = query_slice.new_empty((batch, bricks, rows, geometry.grid[1], heads, blocks, *geometry.span, head_dim))
    start = 0
    for key_view, value_view in zip(key_views, value_views):
        stop = start + key_view.shape[5]
        keys[:, :, :, :, :, start:stop].copy_(key_view)
        values[:, :, :, :, :, start:stop].copy_(value_view)
        start = stop
    count = batch * bricks * rows * geometry.grid[1]
    # `scale=1.0`: the query is already scaled in `project_qkv`.
    attended = F.scaled_dot_product_attention(
        _joint_query_bricks(query_slice, geometry, bricks, rows),
        keys.view(count, heads, blocks * geometry.footprint, channels),
        values.view(count, heads, blocks * geometry.footprint, head_dim),
        attn_mask=mask,
        scale=1.0,
    )
    return _joint_unbrick(attended, geometry, batch, bricks, rows, (query_slice.shape[1], query_slice.shape[2]))


def _joint_row_groups(geometry: _JointGeometry, schedule: _JointSchedule) -> list[tuple[int, slice, slice]]:
    """`(rows, staged H slice, output H slice)` per group of brick rows."""
    brick_h = geometry.brick[1]
    groups = []
    for row in range(0, geometry.grid[0], schedule.group_rows):
        rows = min(schedule.group_rows, geometry.grid[0] - row)
        groups.append(
            (
                rows,
                slice(row * brick_h, row * brick_h + geometry.row_extent(rows)),
                slice(row * brick_h, min((row + rows) * brick_h, geometry.height)),
            )
        )
    return groups


def _joint_video_query_pass(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    keyframe_key: torch.Tensor,
    keyframe_value: torch.Tensor,
    slots: torch.Tensor,
    geometry: _JointGeometry,
    factor: float,
) -> torch.Tensor:
    """Video queries: their local `Kt x Kh x Kw` window plus the same `Kh x Kw` window on their nearest planes."""
    num_frames, heads, head_dim = query.shape[1], query.shape[4], query.shape[5]
    brick_t = geometry.brick[0]
    lo_t, hi_t = geometry.pad_t
    num_slots = slots.shape[1]

    plane_keys = _joint_stage(keyframe_key, geometry, (0, 0), with_bias_channel=True)
    plane_values = _joint_stage(keyframe_value, geometry, (0, 0), with_bias_channel=False)
    # One all-dead plane appended for empty (`-1`) slots to point at.
    null_shape = (*plane_keys.shape[:2], 1, *plane_keys.shape[3:])
    null_key = plane_keys.new_zeros(null_shape)
    null_key[..., head_dim] = _JOINT_DEAD_KEY
    plane_keys = torch.cat([plane_keys, null_key], dim=2)
    plane_values = torch.cat([plane_values, plane_values.new_zeros((*null_shape[:-1], head_dim))], dim=2)
    slot_table = _joint_with_null(slots, keyframe_key.shape[1])

    mask = _joint_mask(geometry, num_slots, query.device)
    schedule = _JointSchedule(
        geometry, geometry.span_t + num_slots, heads, head_dim, -(-num_frames // brick_t), query.element_size(), factor
    )
    row_groups = _joint_row_groups(geometry, schedule)

    out = torch.empty_like(query)
    for run_start, run_stop in _joint_slot_runs(slot_table):
        # Every brick inside a run sees the same planes, so they are gathered once per run.
        planes = plane_keys.index_select(2, slot_table[run_start])
        plane_vals = plane_values.index_select(2, slot_table[run_start])
        run_bricks = -(-(run_stop - run_start) // brick_t)
        for staged_brick in range(0, run_bricks, schedule.stage_axis):
            staged_bricks = min(schedule.stage_axis, run_bricks - staged_brick)
            first = run_start + staged_brick * brick_t
            last = first + staged_bricks * brick_t
            source = slice(max(0, first - lo_t), min(num_frames, last + hi_t))
            pad_t = (max(0, lo_t - first), max(0, last + hi_t - num_frames))
            window_keys = _joint_stage(key[:, source], geometry, pad_t, with_bias_channel=True)
            window_values = _joint_stage(value[:, source], geometry, pad_t, with_bias_channel=False)

            for brick in range(staged_brick, staged_brick + staged_bricks, schedule.group_axis):
                count = min(schedule.group_axis, staged_brick + staged_bricks - brick)
                start = run_start + brick * brick_t
                stop = min(start + count * brick_t, run_stop)
                offset = (brick - staged_brick) * brick_t
                for rows, key_rows, out_rows in row_groups:
                    out[:, start:stop, out_rows] = _joint_attend_group(
                        query[:, start:stop, out_rows],
                        (
                            _joint_slabs(
                                window_keys[:, :, offset:, key_rows], geometry, count, rows, geometry.span_t, brick_t
                            ),
                            _joint_slabs(planes[:, :, :, key_rows], geometry, count, rows, num_slots, 0),
                        ),
                        (
                            _joint_slabs(
                                window_values[:, :, offset:, key_rows], geometry, count, rows, geometry.span_t, brick_t
                            ),
                            _joint_slabs(plane_vals[:, :, :, key_rows], geometry, count, rows, num_slots, 0),
                        ),
                        geometry,
                        (count, rows),
                        mask,
                    )
    return out


def _joint_keyframe_query_pass(
    keyframe_query: torch.Tensor,
    keyframe_key: torch.Tensor,
    keyframe_value: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    slots: torch.Tensor,
    geometry: _JointGeometry,
    factor: float,
) -> torch.Tensor:
    """Plane queries: the `Kh x Kw` window on their own plane plus the same window on their nearest video frames."""
    num_planes, heads, head_dim = keyframe_query.shape[1], keyframe_query.shape[4], keyframe_query.shape[5]
    num_slots = slots.shape[1]
    num_frames = key.shape[1]
    flat = _JointGeometry(geometry.height, geometry.width, (1, *geometry.kernel[1:]), (1, *geometry.brick[1:]))

    # Stage only the frames some plane points at; `unique` doubles as the remap of the slot table.
    wanted, inverse = torch.unique(_joint_with_null(slots, num_frames).reshape(-1), return_inverse=True)
    frame_keys = _joint_stage(
        key.index_select(1, wanted.clamp(max=num_frames - 1)), flat, (0, 0), with_bias_channel=True
    )
    frame_values = _joint_stage(
        value.index_select(1, wanted.clamp(max=num_frames - 1)), flat, (0, 0), with_bias_channel=False
    )
    # An empty slot was clamped onto a real frame above; kill it here.
    frame_keys[:, :, wanted == num_frames, ..., head_dim] = _JOINT_DEAD_KEY
    own_keys = _joint_stage(keyframe_key, flat, (0, 0), with_bias_channel=True)
    own_values = _joint_stage(keyframe_value, flat, (0, 0), with_bias_channel=False)
    slot_table = inverse.reshape(num_planes, num_slots)
    mask = _joint_mask(flat, num_slots, keyframe_query.device)
    schedule = _JointSchedule(flat, 1 + num_slots, heads, head_dim, num_planes, keyframe_query.element_size(), factor)
    row_groups = _joint_row_groups(flat, schedule)

    out = torch.empty_like(keyframe_query)
    for start in range(0, num_planes, schedule.group_axis):
        stop = min(start + schedule.group_axis, num_planes)
        count = stop - start
        picked = slot_table[start:stop].reshape(-1)
        frames = frame_keys.index_select(2, picked)
        frame_vals = frame_values.index_select(2, picked)
        for rows, key_rows, out_rows in row_groups:
            out[:, start:stop, out_rows] = _joint_attend_group(
                keyframe_query[:, start:stop, out_rows],
                (
                    _joint_slabs(own_keys[:, :, start:, key_rows], flat, count, rows, 1, 1),
                    _joint_slabs(frames[:, :, :, key_rows], flat, count, rows, num_slots, num_slots),
                ),
                (
                    _joint_slabs(own_values[:, :, start:, key_rows], flat, count, rows, 1, 1),
                    _joint_slabs(frame_vals[:, :, :, key_rows], flat, count, rows, num_slots, num_slots),
                ),
                flat,
                (count, rows),
                mask,
            )
    return out


def _joint_neighborhood_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    keyframe_query: torch.Tensor,
    keyframe_key: torch.Tensor,
    keyframe_value: torch.Tensor,
    keyframe_times: torch.Tensor,
    kernel_size: tuple[int, int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Joint neighborhood attention over a video volume and a stack of keyframe planes, one softmax per query.

    A video query at `(t, h, w)` attends to its `Kt x Kh x Kw` video window plus the `Kh x Kw` window at the same `(h,
    w)` on each of its `_KEYFRAME_CONTEXT_SLOTS` nearest planes, ranked by `|t - keyframe_times|` whatever `Kt` is. A
    plane query attends to the `Kh x Kw` window on its own plane (there is no plane-to-plane attention) plus the same
    window on each of its nearest video frames.

    Unlike [`LTX2VideoVaeNeighborhoodAttnProcessor`] and NATTEN, whose windows shift inward at the volume border, every
    window here is centered and *clamped*: offsets that fall outside the volume are masked out. That is also why no
    axis needs to be at least its kernel size.

    Args:
        query, key, value: `(B, T, H, W, heads, head_dim)` video stream, query already scaled.
        keyframe_query, keyframe_key, keyframe_value: `(B, P, H, W, heads, head_dim)` keyframe stream.
        keyframe_times: `(P,)` plane positions, in the same temporal units and origin as the video stream's RoPE.
        kernel_size: `(Kt, Kh, Kw)`.

    Returns:
        `(video_out, keyframe_out)`, each shaped like its stream's query.
    """
    num_frames, height, width = query.shape[1], query.shape[2], query.shape[3]
    keyframe_times = keyframe_times.to(device=query.device, dtype=torch.float32)
    frame_times = torch.arange(num_frames, dtype=torch.float32, device=query.device)
    video_slots = _nearest_slots(frame_times, keyframe_times, _KEYFRAME_CONTEXT_SLOTS)
    keyframe_slots = _nearest_slots(keyframe_times, frame_times, _KEYFRAME_CONTEXT_SLOTS)

    side = max(1, round(math.sqrt(_JOINT_BRICK_QUERIES)))
    brick = (min(_JOINT_BRICK_DEPTH, num_frames), min(side, height), min(side, width))
    geometry = _JointGeometry(height, width, kernel_size, brick)
    # Only CUDA keeps the score block on chip for this broadcast mask; elsewhere SDPA materializes it.
    factor = _JOINT_STAGING_FACTOR_FUSED if query.device.type == "cuda" else _JOINT_STAGING_FACTOR_MATERIALIZED
    video_out = _joint_video_query_pass(query, key, value, keyframe_key, keyframe_value, video_slots, geometry, factor)
    keyframe_out = _joint_keyframe_query_pass(
        keyframe_query, keyframe_key, keyframe_value, key, value, keyframe_slots, geometry, factor
    )
    return video_out, keyframe_out


class LTX2VideoVaeRotaryPosEmbed3D(nn.Module):
    """Absolute 3D rotary embedding for the diffusion decoder's neighborhood attention.

    `head_dim` is split into (T, H, W) chunks, each rotated by its own axis position. Positions are the tensor's own
    0-based indices: attention here is always a local window with no causal masking, so the score between a query and a
    key depends only on their relative offset and a shared origin shift is a no-op. Rotation is computed in fp32 and
    cast back to the input dtype.
    """

    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()
        if head_dim % 8 != 0:
            raise ValueError(f"head_dim must be a multiple of 8, got {head_dim}.")
        # Split `head_dim` across the (T, H, W) chunks the way the reference decoder does: a quarter to T, the
        # rest halved between H and W, with both halves kept even so each holds whole rotation pairs.
        dim_t = (head_dim // 4) // 2 * 2
        dim_hw = (head_dim - dim_t) // 2
        if dim_hw % 2 != 0:
            dim_t -= 2
            dim_hw = (head_dim - dim_t) // 2
        self.rope_dim_split = (dim_t, dim_hw, dim_hw)
        self.base = base

    def _inv_freqs(self, dim: int, device: torch.device) -> torch.Tensor:
        # The reference builds these in float64 and casts once. float64 is unsupported on some backends, so ask
        # for the device's widest available dtype instead: the difference is at most 1.5e-08 in the frequencies
        # and 1e-06 in the resulting angles, four orders of magnitude under bf16 resolution.
        freqs_dtype = maybe_adjust_dtype_for_device(torch.float64, device)
        exponents = torch.arange(0, dim, 2, dtype=freqs_dtype, device=device) / dim
        return (1.0 / self.base**exponents).to(torch.float32)

    def _rotate_axis(self, x: torch.Tensor, positions: torch.Tensor, inv_freqs: torch.Tensor, axis: int):
        out_dtype = x.dtype
        pairs = x.reshape(*x.shape[:-1], x.shape[-1] // 2, 2)
        even = pairs[..., 0].float()
        odd = pairs[..., 1].float()
        # Broadcast the angle over (B, T, H, W, heads, dim // 2), varying only along `axis`.
        shape = [1, 1, 1, 1, 1, inv_freqs.shape[0]]
        shape[axis] = positions.shape[0]
        angles = (positions[:, None] * inv_freqs[None, :]).reshape(shape)
        cos, sin = angles.cos(), angles.sin()
        rotated = torch.stack([even * cos - odd * sin, even * sin + odd * cos], dim=-1)
        return rotated.reshape(x.shape).to(out_dtype)

    def forward(self, hidden_states: torch.Tensor, positions_t: torch.Tensor | None = None) -> torch.Tensor:
        """`hidden_states`: `(B, T, H, W, heads, head_dim)`.

        `positions_t` overrides the integer positions on the first axis. Keyframe planes pass their (possibly
        fractional) times there, so that both streams of a keyframe decode share one temporal origin.
        """
        dim_t, dim_h, _ = self.rope_dim_split
        num_frames, height, width = hidden_states.shape[1:4]
        device = hidden_states.device
        inv_t, inv_h, inv_w = (self._inv_freqs(dim, device) for dim in self.rope_dim_split)

        if positions_t is None:
            positions_t = torch.arange(num_frames, dtype=torch.float32, device=device)
        else:
            positions_t = positions_t.to(device=device, dtype=torch.float32)
        positions_h = torch.arange(height, dtype=torch.float32, device=device)
        positions_w = torch.arange(width, dtype=torch.float32, device=device)
        rotated_t = self._rotate_axis(hidden_states[..., :dim_t], positions_t, inv_t, axis=1)
        rotated_h = self._rotate_axis(hidden_states[..., dim_t : dim_t + dim_h], positions_h, inv_h, axis=2)
        rotated_w = self._rotate_axis(hidden_states[..., dim_t + dim_h :], positions_w, inv_w, axis=3)
        return torch.cat([rotated_t, rotated_h, rotated_w], dim=-1)


class LTX2VideoVaeNeighborhoodAttnProcessor:
    """Portable neighborhood-attention processor built on FlexAttention via `dispatch_attention_fn`.

    Runs anywhere the flex attention path runs. For bit-exact parity with the reference decoder, use
    [`LTX2VideoVaeNeighborhoodNattenProcessor`] instead.

    Requires the `flex` attention backend: the neighborhood window is expressed as a
    [`~torch.nn.attention.flex_attention.BlockMask`], which only the flex backend consumes. Switching the backend —
    e.g. via `model.set_attention_backend("flash")` — raises a `ValueError` rather than handing the mask to a backend
    that cannot read it.
    """

    _attention_backend = "flex"
    _parallel_config = None

    _SUPPORTED_BACKENDS = ("flex", "_native_flex")

    def __call__(
        self, attn: "LTX2VideoVaeNeighborhoodAttention", hidden_states: torch.Tensor, block_mask=None
    ) -> torch.Tensor:
        if self._attention_backend not in self._SUPPORTED_BACKENDS:
            raise ValueError(
                f"LTX2VideoVaeNeighborhoodAttnProcessor requires the 'flex' attention backend (got "
                f"{self._attention_backend!r}). It builds a flex_attention.BlockMask for the neighborhood "
                f"window, which no other backend in `dispatch_attention_fn` accepts. To use NATTEN's kernels "
                f"instead, set LTX2VideoVaeNeighborhoodNattenProcessor via `set_attn_processor`."
            )

        batch_size, num_frames, height, width, _ = hidden_states.shape
        query, key, value = attn.project_qkv(hidden_states)

        query = query.reshape(batch_size, num_frames * height * width, attn.heads, attn.head_dim)
        key = key.reshape(batch_size, num_frames * height * width, attn.heads, attn.head_dim)
        value = value.reshape(batch_size, num_frames * height * width, attn.heads, attn.head_dim)
        if block_mask is None:
            block_mask = _neighborhood_block_mask(num_frames, height, width, attn.kernel_size, hidden_states.device)
        # `scale=1.0`: the query is already scaled in `project_qkv`, as in the reference.
        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=block_mask,
            scale=1.0,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        hidden_states = hidden_states.reshape(batch_size, num_frames, height, width, attn.heads * attn.head_dim)
        return attn.to_out[0](hidden_states)


class LTX2VideoVaeNeighborhoodNattenProcessor:
    """Neighborhood-attention processor using NATTEN's `na3d`, which is what the reference decoder calls.

    NATTEN is fetched from the Hub (`shi-labs/natten`, a trusted kernel publisher) through the `kernels` package rather
    than imported from a local install, so it needs `kernels`, a supported GPU, and `DIFFUSERS_DISABLE_REMOTE_CODE`
    unset; `backend=None` lets NATTEN pick the fastest kernel for the device. No CPU path — use
    [`LTX2VideoVaeNeighborhoodAttnProcessor`] elsewhere.

    `na3d` encodes the neighborhood window in the kernel itself, so this processor takes no `block_mask` and the
    decoder never builds one for it.
    """

    def __init__(self, backend: str | None = None):
        if DIFFUSERS_DISABLE_REMOTE_CODE:
            raise ValueError(
                "LTX2VideoVaeNeighborhoodNattenProcessor downloads the `shi-labs/natten` kernel from the Hub, which "
                "is disabled globally by the `DIFFUSERS_DISABLE_REMOTE_CODE` environment variable. Unset it, or use "
                "the default `LTX2VideoVaeNeighborhoodAttnProcessor` (FlexAttention) instead."
            )
        if not is_kernels_available():
            raise ImportError(
                "LTX2VideoVaeNeighborhoodNattenProcessor fetches NATTEN from the Hub with the `kernels` package. "
                "Install it with `pip install kernels`, or use the default "
                "`LTX2VideoVaeNeighborhoodAttnProcessor` (FlexAttention) instead."
            )
        from kernels import get_kernel

        self._na3d = get_kernel("shi-labs/natten", version=1).na3d
        self.backend = backend

    def __call__(
        self, attn: "LTX2VideoVaeNeighborhoodAttention", hidden_states: torch.Tensor, block_mask=None
    ) -> torch.Tensor:
        batch_size, num_frames, height, width, channels = hidden_states.shape
        query, key, value = attn.project_qkv(hidden_states)
        # NATTEN's CUTLASS kernels silently produce wrong output for non-contiguous inputs.
        query, key, value = query.contiguous(), key.contiguous(), value.contiguous()
        # `scale=1.0`: the query is already scaled in `project_qkv`, as in the reference.
        hidden_states = self._na3d(query, key, value, kernel_size=attn.kernel_size, scale=1.0, backend=self.backend)
        hidden_states = hidden_states.reshape(batch_size, num_frames, height, width, channels)
        return attn.to_out[0](hidden_states)


class LTX2VideoVaeNeighborhoodAttention(nn.Module, AttentionModuleMixin):
    _default_processor_cls = LTX2VideoVaeNeighborhoodAttnProcessor
    _available_processors = [LTX2VideoVaeNeighborhoodAttnProcessor, LTX2VideoVaeNeighborhoodNattenProcessor]
    # Both processors read `to_q`/`to_k`/`to_v` directly and have no fused path, so QKV fusion would build an
    # unused `to_qkv` and silently no-op. The flex/NATTEN swap goes through `set_attn_processor` instead.
    _supports_qkv_fusion = False

    def __init__(
        self,
        dim: int,
        kernel_size: tuple[int, int, int],
        head_dim: int = 64,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        if dim % head_dim != 0:
            raise ValueError(f"dim {dim} must be divisible by head_dim {head_dim}.")
        self.heads = dim // head_dim
        self.head_dim = head_dim
        self.kernel_size = tuple(kernel_size)
        self.scale = head_dim**-0.5

        self.to_q = nn.Linear(dim, dim, bias=True)
        self.to_k = nn.Linear(dim, dim, bias=True)
        self.to_v = nn.Linear(dim, dim, bias=True)
        self.to_out = nn.ModuleList([nn.Linear(dim, dim, bias=True), nn.Dropout(0.0)])
        self.norm_q = nn.RMSNorm(head_dim, eps=1e-6)
        self.norm_k = nn.RMSNorm(head_dim, eps=1e-6)
        self.rope = LTX2VideoVaeRotaryPosEmbed3D(head_dim, base=rope_base)
        self.set_processor(self._default_processor_cls())

    def project_qkv(
        self, hidden_states: torch.Tensor, positions_t: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Q/K/V as `(B, T, H, W, heads, head_dim)`, RMS-normed, query pre-scaled, then rotated.

        The query carries the `1 / sqrt(head_dim)` factor here so both processors can ask their attention backend for
        `scale=1.0` — this is the order the reference uses (norm, scale, then rotate). `positions_t` overrides the
        temporal RoPE positions, see [`LTX2VideoVaeRotaryPosEmbed3D`].
        """
        batch_size, num_frames, height, width, _ = hidden_states.shape
        shape = (batch_size, num_frames, height, width, self.heads, self.head_dim)
        query = self.to_q(hidden_states).view(shape)
        key = self.to_k(hidden_states).view(shape)
        value = self.to_v(hidden_states).view(shape)

        query = self.norm_q(query)
        key = self.norm_k(key)
        query = query * self.scale
        return self.rope(query, positions_t), self.rope(key, positions_t), value

    def build_block_mask(self, hidden_states: torch.Tensor):
        """The flex `BlockMask` for this grid, or `None` if the processor does not read one.

        The mask depends only on the token grid and the kernel, both of which are fixed within a decoder stage, so a
        stage builds it once and hands it to every block. NATTEN gets `None`: it encodes the window in its kernel, and
        at production grids the mask is not merely wasteful but larger than device memory (a 69x64x96 stage needs 167
        GiB), so it must never be built for a path that would not read it.
        """
        if not isinstance(self.processor, LTX2VideoVaeNeighborhoodAttnProcessor):
            return None
        num_frames, height, width = hidden_states.shape[1:4]
        return _neighborhood_block_mask(num_frames, height, width, self.kernel_size, hidden_states.device)

    def forward(self, hidden_states: torch.Tensor, block_mask=None) -> torch.Tensor:
        """Channels-last in and out: `(B, T, H, W, C)`."""
        num_frames, height, width = hidden_states.shape[1:4]
        kernel_t, kernel_h, kernel_w = self.kernel_size
        if num_frames < kernel_t or height < kernel_h or width < kernel_w:
            raise ValueError(
                f"Neighborhood attention requires each spatial dim to be at least its kernel size; got "
                f"(T, H, W) = ({num_frames}, {height}, {width}) with kernel_size {self.kernel_size}."
            )
        return self.processor(self, hidden_states, block_mask)

    def forward_with_keyframes(
        self, hidden_states: torch.Tensor, keyframe_hidden_states: torch.Tensor, keyframe_times: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Joint attention over the video `(B, T, H, W, C)` and the keyframe planes `(B, P, H, W, C)`.

        This bypasses the attention processor: neither FlexAttention's neighborhood mask nor NATTEN expresses the joint
        window, so a keyframe decode always runs [`_joint_neighborhood_attention`], whatever processor is set.
        `keyframe_times` are the planes' `(P,)` temporal RoPE positions, in the video stream's tile-local origin.
        """
        batch_size, num_frames, height, width, channels = hidden_states.shape
        num_planes = keyframe_hidden_states.shape[1]
        query, key, value = self.project_qkv(hidden_states)
        keyframe_query, keyframe_key, keyframe_value = self.project_qkv(keyframe_hidden_states, keyframe_times)
        hidden_states, keyframe_hidden_states = _joint_neighborhood_attention(
            query.contiguous(),
            key.contiguous(),
            value.contiguous(),
            keyframe_query.contiguous(),
            keyframe_key.contiguous(),
            keyframe_value.contiguous(),
            keyframe_times,
            self.kernel_size,
        )
        hidden_states = self.to_out[0](hidden_states.reshape(batch_size, num_frames, height, width, channels))
        keyframe_hidden_states = self.to_out[0](
            keyframe_hidden_states.reshape(batch_size, num_planes, height, width, channels)
        )
        return hidden_states, keyframe_hidden_states


# Tokens per tile in `LTX2VideoVaeSwiGLU`, matching the reference decoder's own default. `w_gate(x)` and
# `w_up(x)` are both hidden-width and their product makes a third, so evaluating a whole video at once
# holds three hidden-width tensors live at the same time — at 121 frames and 512x768 that is
# 3 x 5.67 GiB, which by itself dominates decode memory. Fixing a token *count* rather than a number of
# tiles keeps that bound independent of resolution.
_SWIGLU_TILE_SIZE = 16384


class LTX2VideoVaeSwiGLU(nn.Module):
    """Gated MLP: `w_down(silu(w_gate(x)) * w_up(x))`, evaluated in tiles of `_SWIGLU_TILE_SIZE` tokens.

    Tiling is not an approximation: the MLP is pointwise across tokens, so splitting it changes only how many
    hidden-width elements exist at once, never what is computed. Outputs can still differ from the untiled evaluation
    by a few ulps, since a matmul over a tile may reduce in a different order than the full-tensor call.
    """

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w_up = nn.Linear(dim, hidden_dim, bias=False)
        self.w_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, *token_dims, channels = hidden_states.shape
        num_tokens = math.prod(token_dims)
        if num_tokens <= _SWIGLU_TILE_SIZE:
            return self.w_down(F.silu(self.w_gate(hidden_states)) * self.w_up(hidden_states))

        flat = hidden_states.reshape(batch_size, num_tokens, channels)
        out = torch.empty_like(flat)
        for start in range(0, num_tokens, _SWIGLU_TILE_SIZE):
            tile = flat[:, start : start + _SWIGLU_TILE_SIZE]
            out[:, start : start + _SWIGLU_TILE_SIZE] = self.w_down(F.silu(self.w_gate(tile)) * self.w_up(tile))
        return out.reshape(hidden_states.shape)


def _swiglu_hidden_dim(dim: int, mlp_ratio: float) -> int:
    return (int(dim * mlp_ratio) + 15) // 16 * 16


class LTX2VideoVaeNABlock(nn.Module):
    """Pre-norm neighborhood-attention block used by the deterministic upsampling stages."""

    def __init__(
        self,
        dim: int,
        kernel_size: tuple[int, int, int],
        head_dim: int = 64,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.norm1 = nn.RMSNorm(dim, eps=1e-6)
        self.attn = LTX2VideoVaeNeighborhoodAttention(dim, kernel_size, head_dim=head_dim)
        self.norm2 = nn.RMSNorm(dim, eps=1e-6)
        self.mlp = LTX2VideoVaeSwiGLU(dim, _swiglu_hidden_dim(dim, mlp_ratio))

    def forward(self, hidden_states: torch.Tensor, block_mask=None) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states), block_mask)
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states

    def forward_with_keyframes(
        self, hidden_states: torch.Tensor, keyframe_hidden_states: torch.Tensor, keyframe_times: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Both streams through the same weights; they only meet inside the joint attention softmax."""
        attn_output, keyframe_attn_output = self.attn.forward_with_keyframes(
            self.norm1(hidden_states), self.norm1(keyframe_hidden_states), keyframe_times
        )
        hidden_states = hidden_states + attn_output
        keyframe_hidden_states = keyframe_hidden_states + keyframe_attn_output
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        keyframe_hidden_states = keyframe_hidden_states + self.mlp(self.norm2(keyframe_hidden_states))
        return hidden_states, keyframe_hidden_states


class LTX2VideoVaeAdaLNZero(nn.Module):
    """Shared AdaLN-Zero modulation: a timestep embedding to seven `(B, 1, 1, 1, C)` chunks.

    Seven chunks (scale/shift/gate for attention and MLP, plus a context gate) is the reference's shape. Only the four
    scale/shift chunks are consumed: the decoder's residuals are ungated, and the static gates the checkpoint used to
    carry are folded into the following linear weights at conversion time.
    """

    def __init__(self, dim: int, t_emb_dim: int, num_chunks: int = 7):
        super().__init__()
        self.num_chunks = num_chunks
        self.proj = nn.Linear(t_emb_dim, num_chunks * dim, bias=True)

    def forward(self, t_emb: torch.Tensor) -> tuple[torch.Tensor, ...]:
        chunks = self.proj(F.silu(t_emb)).chunk(self.num_chunks, dim=-1)
        return tuple(chunk[:, None, None, None, :] for chunk in chunks)


class LTX2VideoVaeDiffusionNABlock(nn.Module):
    """Neighborhood attention + SwiGLU, modulated by the shared AdaLN-Zero scale/shift.

    The decoder owns one `LTX2VideoVaeAdaLNZero`; each block adds its own `scale_shift_table` residual on top of it,
    injects the latent context through `context_proj`, and keeps its residuals ungated.
    """

    def __init__(
        self,
        dim: int,
        kernel_size: tuple[int, int, int],
        context_channels: int,
        head_dim: int = 64,
        mlp_ratio: float = 4.0,
        num_mod_params: int = 7,
    ):
        super().__init__()
        self.context_channels = context_channels
        self.num_mod_params = num_mod_params
        self.context_proj = nn.Linear(context_channels, dim, bias=True)
        self.scale_shift_table = nn.Parameter(torch.zeros(num_mod_params, dim))

        self.norm1 = nn.RMSNorm(dim, eps=1e-6)
        self.attn = LTX2VideoVaeNeighborhoodAttention(dim, kernel_size, head_dim=head_dim)
        self.norm2 = nn.RMSNorm(dim, eps=1e-6)
        self.mlp = LTX2VideoVaeSwiGLU(dim, _swiglu_hidden_dim(dim, mlp_ratio))

    def forward(
        self,
        hidden_states: torch.Tensor,
        latent_context: torch.Tensor,
        modulation: tuple[torch.Tensor, ...],
        block_mask=None,
    ) -> torch.Tensor:
        scale_msa, shift_msa, _, scale_mlp, shift_mlp, _, _ = [
            modulation[i] + self.scale_shift_table[i].view(1, 1, 1, 1, -1) for i in range(self.num_mod_params)
        ]

        hidden_states = hidden_states + self.context_proj(latent_context)
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states) * (1 + scale_msa) + shift_msa, block_mask)
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states) * (1 + scale_mlp) + shift_mlp)
        return hidden_states

    def forward_with_keyframes(
        self,
        hidden_states: torch.Tensor,
        latent_context: torch.Tensor,
        keyframe_hidden_states: torch.Tensor,
        keyframe_latent_context: torch.Tensor,
        modulation: tuple[torch.Tensor, ...],
        keyframe_times: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Both streams through the same weights and the same modulation, each with its own context."""
        scale_msa, shift_msa, _, scale_mlp, shift_mlp, _, _ = [
            modulation[i] + self.scale_shift_table[i].view(1, 1, 1, 1, -1) for i in range(self.num_mod_params)
        ]

        hidden_states = hidden_states + self.context_proj(latent_context)
        keyframe_hidden_states = keyframe_hidden_states + self.context_proj(keyframe_latent_context)
        attn_output, keyframe_attn_output = self.attn.forward_with_keyframes(
            self.norm1(hidden_states) * (1 + scale_msa) + shift_msa,
            self.norm1(keyframe_hidden_states) * (1 + scale_msa) + shift_msa,
            keyframe_times,
        )
        hidden_states = hidden_states + attn_output
        keyframe_hidden_states = keyframe_hidden_states + keyframe_attn_output
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states) * (1 + scale_mlp) + shift_mlp)
        keyframe_hidden_states = keyframe_hidden_states + self.mlp(
            self.norm2(keyframe_hidden_states) * (1 + scale_mlp) + shift_mlp
        )
        return hidden_states, keyframe_hidden_states


class LTX2VideoVaePixelShuffleUpsampler(nn.Module):
    """Linear channel expansion followed by a channels-last pixel shuffle.

    When the temporal stride is 2 the shuffle produces a duplicate leading frame, which is dropped to keep the causal
    1:2 (composed 1:8) frame mapping. `drop_leading_frame=False` keeps it: a tiled decode passes that for temporal
    tiles that do not contain t=0, where the first input frame is an interior frame whose two output frames are both
    real content.
    """

    def __init__(self, in_channels: int, stride: tuple[int, int, int], out_channels_reduction_factor: int = 1):
        super().__init__()
        self.stride = tuple(stride)
        proj_out_channels = math.prod(self.stride) * in_channels // out_channels_reduction_factor
        self.out_channels = proj_out_channels // math.prod(self.stride)
        self.proj = nn.Linear(in_channels, proj_out_channels, bias=True)

    def forward(self, hidden_states: torch.Tensor, drop_leading_frame: bool = True) -> torch.Tensor:
        batch_size, num_frames, height, width, _ = hidden_states.shape
        stride_t, stride_h, stride_w = self.stride
        hidden_states = self.proj(hidden_states)
        hidden_states = hidden_states.reshape(
            batch_size, num_frames, height, width, self.out_channels, stride_t, stride_h, stride_w
        )
        # (b, t, p1, h, p2, w, p3, c) -> merge each stride into its own axis
        hidden_states = hidden_states.permute(0, 1, 5, 2, 6, 3, 7, 4)
        hidden_states = hidden_states.reshape(
            batch_size, num_frames * stride_t, height * stride_h, width * stride_w, self.out_channels
        )
        if stride_t == 2 and drop_leading_frame:
            hidden_states = hidden_states[:, 1:]
        return hidden_states


class LTX2VideoDiffusionDecoder3d(nn.Module):
    """The LTX-2.5 diffusion video decoder.

    Stages 1-4 deterministically upsample the latent into a context volume with neighborhood-attention blocks. Stage 5
    then denoises patchified pixels, conditioned on that context through AdaLN-Zero scale/shift. With
    `model_output_type="x0"` and a single step — how LTX-2.5 ships — stage 5 runs once and its prediction *is* the
    output; more steps add reverse Euler updates.
    """

    def __init__(
        self,
        in_channels: int = 128,
        out_channels: int = 3,
        patch_size: int = 4,
        head_dim: int = 64,
        stage_channels: tuple[int, ...] = (2048, 1024, 512, 512, 256),
        stage_depths: tuple[int, ...] = (4, 6, 4, 2, 8),
        stage_kernels: tuple[tuple[int, int, int], ...] = ((3, 7, 7), (3, 7, 7), (3, 5, 5), (3, 5, 5)),
        upsample_strides: tuple[tuple[int, int, int], ...] = ((1, 2, 2), (2, 1, 1), (2, 2, 2), (2, 2, 2)),
        upsample_channel_reductions: tuple[int, ...] = (2, 2, 1, 2),
        stage5_kernel: tuple[int, int, int] = (11, 11, 11),
        t_emb_dim: int = 384,
        temporal_compression_ratio: int = 8,
        timestep_scale_multiplier: float = 1000.0,
        model_output_type: str = "x0",
        default_num_inference_steps: int = 1,
        keyframe_type_embedding: bool = False,
    ):
        super().__init__()
        if model_output_type not in ("x0", "v"):
            raise ValueError(f"model_output_type must be 'x0' or 'v', got {model_output_type!r}.")
        # Each upsample divides the channel count by its reduction factor, so the stage widths and the
        # reductions are two views of the same thing and an inconsistent pair would only fail deep inside
        # the first block.
        for stage_idx, reduction in enumerate(upsample_channel_reductions):
            expected = stage_channels[stage_idx] // reduction
            if stage_channels[stage_idx + 1] != expected:
                raise ValueError(
                    f"stage_channels[{stage_idx + 1}] must be stage_channels[{stage_idx}] // "
                    f"upsample_channel_reductions[{stage_idx}] = {expected}, got {stage_channels[stage_idx + 1]}."
                )

        self.patch_size = patch_size
        self.out_channels = out_channels
        self.timestep_scale_multiplier = timestep_scale_multiplier
        self.model_output_type = model_output_type
        self.default_num_inference_steps = default_num_inference_steps
        self.temporal_compression_ratio = temporal_compression_ratio
        self.context_channels = stage_channels[-1]
        # NATTEN shifts its window inward at the grid border, so the last latent frame is replicated
        # through stages 1-4 and cropped off the context before stage 5, moving that border past the
        # frames that are kept.
        self.trailing_pad_latent_frames = (stage_kernels[0][0] // 2) * 2

        self.conv_in = nn.Linear(in_channels, stage_channels[0], bias=True)
        # The learned tag of the keyframe stream, added to the keyframe latents right before the shared `conv_in`. It
        # is the only keyframe-specific weight, so a checkpoint trained without keyframes simply does not have one.
        self.type_emb = nn.Parameter(torch.zeros(in_channels)) if keyframe_type_embedding else None
        # Temporal upsampling still to come at each stage input, plus 1 for the diffusion stage: the divisor of
        # `_keyframe_stage_times`. (8, 8, 4, 2, 1) for the production strides.
        self.keyframe_time_strides = tuple(
            math.prod(stride[0] for stride in upsample_strides[stage_idx:])
            for stage_idx in range(len(upsample_strides) + 1)
        )

        self.det_stages = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for stage_idx, stride in enumerate(upsample_strides):
            channels = stage_channels[stage_idx]
            self.det_stages.append(
                nn.ModuleList(
                    [
                        LTX2VideoVaeNABlock(
                            dim=channels,
                            kernel_size=stage_kernels[stage_idx],
                            head_dim=head_dim,
                        )
                        for _ in range(stage_depths[stage_idx])
                    ]
                )
            )
            self.upsamples.append(
                LTX2VideoVaePixelShuffleUpsampler(
                    in_channels=channels,
                    stride=stride,
                    out_channels_reduction_factor=upsample_channel_reductions[stage_idx],
                )
            )

        self.t_embedder = PixArtAlphaCombinedTimestepSizeEmbeddings(embedding_dim=t_emb_dim, size_emb_dim=0)

        stage5_channels = stage_channels[-1]
        noised_pixel_channels = out_channels * patch_size**2
        self.conv_in_x_t = nn.Linear(noised_pixel_channels, stage5_channels, bias=True)
        self.shared_adaln = LTX2VideoVaeAdaLNZero(dim=stage5_channels, t_emb_dim=t_emb_dim)
        self.diff_blocks = nn.ModuleList(
            [
                LTX2VideoVaeDiffusionNABlock(
                    dim=stage5_channels,
                    kernel_size=stage5_kernel,
                    context_channels=self.context_channels,
                    head_dim=head_dim,
                    num_mod_params=self.shared_adaln.num_chunks,
                )
                for _ in range(stage_depths[-1])
            ]
        )
        self.norm_out = nn.RMSNorm(stage5_channels, eps=1e-6)
        self.conv_out = nn.Linear(stage5_channels, noised_pixel_channels, bias=True)

    def forward_stages_1_to_3(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """All deterministic stages but the last: latent `(B, C, T, H, W)` to a channels-last feature volume.

        The trailing ghost frames added for NATTEN's border shift stay in the output; [`forward_stage_4`] crops them.
        The split at this point exists for tiled decoding: these stages are cheap enough to run on the full volume,
        while stage 4 and the diffusion stage — where the grid and the channel-hidden products get large — run per
        tile.
        """
        num_pad = self.trailing_pad_latent_frames
        if num_pad > 0:
            trailing = hidden_states[:, :, -1:].expand(-1, -1, num_pad, -1, -1)
            hidden_states = torch.cat([hidden_states, trailing], dim=2)

        hidden_states = hidden_states.permute(0, 2, 3, 4, 1)
        hidden_states = self.conv_in(hidden_states)
        for blocks, upsample in zip(self.det_stages[:-1], self.upsamples[:-1]):
            # The grid and kernel are fixed within a stage, so every block shares one mask.
            block_mask = blocks[0].attn.build_block_mask(hidden_states)
            for block in blocks:
                hidden_states = block(hidden_states, block_mask)
            hidden_states = upsample(hidden_states)
        return hidden_states

    def forward_stage_4(
        self, hidden_states: torch.Tensor, drop_leading_frame: bool = True, crop_trailing_ghost: bool = True
    ) -> torch.Tensor:
        """Last deterministic stage: [`forward_stages_1_to_3`] output to context `(B, T_5, H_5, W_5, C_5)`.

        The defaults describe the untiled decode. A tiled decode overrides them per temporal tile: only the tile
        containing t=0 drops the upsample's duplicate leading frame, and only the tile containing the video end carries
        the trailing ghost frames to crop.
        """
        blocks = self.det_stages[-1]
        block_mask = blocks[0].attn.build_block_mask(hidden_states)
        for block in blocks:
            hidden_states = block(hidden_states, block_mask)
        hidden_states = self.upsamples[-1](hidden_states, drop_leading_frame=drop_leading_frame)

        num_pad = self.trailing_pad_latent_frames
        if crop_trailing_ghost and num_pad > 0:
            hidden_states = hidden_states[:, : -num_pad * self.temporal_compression_ratio]
        return hidden_states

    def forward_diffusion_step(
        self, latent_context: torch.Tensor, x_t: torch.Tensor, timestep: torch.Tensor
    ) -> torch.Tensor:
        """One stage-5 step. Returns the model's prediction in pixel space, `(B, C, F, H, W)`."""
        t_emb = self.t_embedder(
            self.timestep_scale_multiplier * timestep,
            resolution=None,
            aspect_ratio=None,
            batch_size=timestep.shape[0],
            hidden_dtype=latent_context.dtype,
        )
        modulation = self.shared_adaln(t_emb)

        hidden_states = _patchify(x_t, self.patch_size).permute(0, 2, 3, 4, 1)
        hidden_states = self.conv_in_x_t(hidden_states)
        block_mask = self.diff_blocks[0].attn.build_block_mask(hidden_states)
        for block in self.diff_blocks:
            hidden_states = block(hidden_states, latent_context, modulation, block_mask)

        hidden_states = self.norm_out(hidden_states)
        hidden_states = self.conv_out(hidden_states)
        hidden_states = hidden_states.permute(0, 4, 1, 2, 3).contiguous()
        return _unpatchify(hidden_states, self.patch_size)

    def denoise(self, latent_context: torch.Tensor, x_t: torch.Tensor, num_inference_steps: int) -> torch.Tensor:
        """Denoise `x_t` `(B, C, F, H, W)` through the stage-5 diffusion loop, conditioned on `latent_context`."""
        batch_size = latent_context.shape[0]
        timesteps = torch.linspace(
            1.0, 1.0 / num_inference_steps, num_inference_steps, device=latent_context.device, dtype=torch.float32
        )

        if num_inference_steps == 1 and self.model_output_type == "x0":
            return self.forward_diffusion_step(latent_context, x_t, timesteps[:1].expand(batch_size))

        for step_idx in range(num_inference_steps):
            t_now = timesteps[step_idx].expand(batch_size)
            t_next = timesteps[step_idx + 1] if step_idx + 1 < num_inference_steps else torch.zeros_like(t_now)
            model_out = self.forward_diffusion_step(latent_context, x_t, t_now).float()
            x_t_fp32 = x_t.float()
            if self.model_output_type == "x0":
                sigma = t_now.view(-1, *([1] * (x_t.ndim - 1)))
                model_out = (x_t_fp32 - model_out) / sigma
            dt = (t_now - t_next).view(-1, *([1] * (x_t.ndim - 1)))
            x_t = (x_t_fp32 - dt * model_out).to(x_t.dtype)
        return x_t

    def forward_stages_1_to_3_with_keyframes(
        self, hidden_states: torch.Tensor, keyframe_hidden_states: torch.Tensor, keyframe_frame_indices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Keyframe counterpart of [`forward_stages_1_to_3`], carrying both streams.

        `keyframe_hidden_states` are the denormalized keyframe latents `(B, C, P, H, W)`, one latent frame per plane,
        on the same spatial grid as `hidden_states`. They are tagged with `type_emb`, then share `conv_in` and every
        stage with the video. The trailing ghost frames are a temporal border workaround of the video stream only. The
        keyframe stream comes back channels-last, `(B, P, H_4, W_4, C_4)`.
        """
        num_pad = self.trailing_pad_latent_frames
        if num_pad > 0:
            trailing = hidden_states[:, :, -1:].expand(-1, -1, num_pad, -1, -1)
            hidden_states = torch.cat([hidden_states, trailing], dim=2)

        hidden_states = self.conv_in(hidden_states.permute(0, 2, 3, 4, 1))
        keyframe_hidden_states = keyframe_hidden_states.permute(0, 2, 3, 4, 1)
        if self.type_emb is not None:
            keyframe_hidden_states = keyframe_hidden_states + self.type_emb.view(1, 1, 1, 1, -1)
        keyframe_hidden_states = self.conv_in(keyframe_hidden_states)
        for stage_idx, (blocks, upsample) in enumerate(zip(self.det_stages[:-1], self.upsamples[:-1])):
            keyframe_times = _keyframe_stage_times(keyframe_frame_indices, self.keyframe_time_strides[stage_idx])
            for block in blocks:
                hidden_states, keyframe_hidden_states = block.forward_with_keyframes(
                    hidden_states, keyframe_hidden_states, keyframe_times
                )
            hidden_states = upsample(hidden_states)
            keyframe_hidden_states = _upsample_keyframe_planes(upsample, keyframe_hidden_states)
        return hidden_states, keyframe_hidden_states

    def forward_stage_4_with_keyframes(
        self,
        hidden_states: torch.Tensor,
        keyframe_hidden_states: torch.Tensor,
        keyframe_frame_indices: torch.Tensor,
        drop_leading_frame: bool = True,
        crop_trailing_ghost: bool = True,
        stage_4_time_origin: float = 0.0,
        pixel_time_origin: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Keyframe counterpart of [`forward_stage_4`]. Returns the video context, the keyframe context, and the
        planes' times in the diffusion stage.

        The video stream's RoPE positions are 0-based within a tile at every stage, so the plane times are rebased on
        the tile's first frame. That takes two origins at two scales, both taken from the tile and not derived from one
        another (the causal first frame makes `pixel_time_origin == stride * stage_4_time_origin` an off-by-one trap):
        `stage_4_time_origin` is the tile's first cell entering this stage, `pixel_time_origin` its first pixel frame.
        Both are 0 for an untiled decode. `keyframe_frame_indices` stay global pixel frames.
        """
        keyframe_times = (
            _keyframe_stage_times(keyframe_frame_indices, self.keyframe_time_strides[-2]) - stage_4_time_origin
        )
        for block in self.det_stages[-1]:
            hidden_states, keyframe_hidden_states = block.forward_with_keyframes(
                hidden_states, keyframe_hidden_states, keyframe_times
            )
        hidden_states = self.upsamples[-1](hidden_states, drop_leading_frame=drop_leading_frame)
        keyframe_hidden_states = _upsample_keyframe_planes(self.upsamples[-1], keyframe_hidden_states)

        num_pad = self.trailing_pad_latent_frames
        if crop_trailing_ghost and num_pad > 0:
            hidden_states = hidden_states[:, : -num_pad * self.temporal_compression_ratio]
        keyframe_times = (
            _keyframe_stage_times(keyframe_frame_indices, self.keyframe_time_strides[-1]) - pixel_time_origin
        )
        return hidden_states, keyframe_hidden_states, keyframe_times

    def forward_diffusion_step_with_keyframes(
        self,
        latent_context: torch.Tensor,
        x_t: torch.Tensor,
        keyframe_latent_context: torch.Tensor,
        keyframe_x_t: torch.Tensor,
        timestep: torch.Tensor,
        keyframe_times: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One stage-5 step over both streams. Returns `(video_prediction, keyframe_prediction)` in pixel space.

        The keyframe stream is a second pixel diffusion stream, one pixel frame per plane: its own noised pixels
        through the shared `conv_in_x_t`, its own context, the same modulation. It is stepped along with the video so
        the hidden states the joint attention reads stay at the noise level the decoder was trained on, and is then
        discarded.
        """
        t_emb = self.t_embedder(
            self.timestep_scale_multiplier * timestep,
            resolution=None,
            aspect_ratio=None,
            batch_size=timestep.shape[0],
            hidden_dtype=latent_context.dtype,
        )
        modulation = self.shared_adaln(t_emb)

        hidden_states = self.conv_in_x_t(_patchify(x_t, self.patch_size).permute(0, 2, 3, 4, 1))
        keyframe_hidden_states = self.conv_in_x_t(_patchify(keyframe_x_t, self.patch_size).permute(0, 2, 3, 4, 1))
        for block in self.diff_blocks:
            hidden_states, keyframe_hidden_states = block.forward_with_keyframes(
                hidden_states,
                latent_context,
                keyframe_hidden_states,
                keyframe_latent_context,
                modulation,
                keyframe_times,
            )

        outputs = []
        for states in (hidden_states, keyframe_hidden_states):
            states = self.conv_out(self.norm_out(states))
            outputs.append(_unpatchify(states.permute(0, 4, 1, 2, 3).contiguous(), self.patch_size))
        return outputs[0], outputs[1]

    def denoise_with_keyframes(
        self,
        latent_context: torch.Tensor,
        x_t: torch.Tensor,
        keyframe_latent_context: torch.Tensor,
        keyframe_x_t: torch.Tensor,
        keyframe_times: torch.Tensor,
        num_inference_steps: int,
    ) -> torch.Tensor:
        """Keyframe counterpart of [`denoise`]: both streams through the same Euler loop, only the video returned."""
        batch_size = latent_context.shape[0]
        timesteps = torch.linspace(
            1.0, 1.0 / num_inference_steps, num_inference_steps, device=latent_context.device, dtype=torch.float32
        )

        if num_inference_steps == 1 and self.model_output_type == "x0":
            return self.forward_diffusion_step_with_keyframes(
                latent_context,
                x_t,
                keyframe_latent_context,
                keyframe_x_t,
                timesteps[:1].expand(batch_size),
                keyframe_times,
            )[0]

        for step_idx in range(num_inference_steps):
            t_now = timesteps[step_idx].expand(batch_size)
            t_next = timesteps[step_idx + 1] if step_idx + 1 < num_inference_steps else torch.zeros_like(t_now)
            model_outputs = self.forward_diffusion_step_with_keyframes(
                latent_context, x_t, keyframe_latent_context, keyframe_x_t, t_now, keyframe_times
            )
            sigma = t_now.view(-1, *([1] * (x_t.ndim - 1)))
            dt = (t_now - t_next).view(-1, *([1] * (x_t.ndim - 1)))
            updated = []
            for sample, model_out in zip((x_t, keyframe_x_t), model_outputs):
                sample_fp32, model_out = sample.float(), model_out.float()
                if self.model_output_type == "x0":
                    model_out = (sample_fp32 - model_out) / sigma
                updated.append((sample_fp32 - dt * model_out).to(sample.dtype))
            x_t, keyframe_x_t = updated
        return x_t

    def _decode_with_keyframes(
        self,
        hidden_states: torch.Tensor,
        keyframe_hidden_states: torch.Tensor,
        keyframe_frame_indices: torch.Tensor,
        generator: torch.Generator | None,
        num_inference_steps: int,
    ) -> torch.Tensor:
        features, keyframe_features = self.forward_stages_1_to_3_with_keyframes(
            hidden_states, keyframe_hidden_states, keyframe_frame_indices
        )
        latent_context, keyframe_latent_context, keyframe_times = self.forward_stage_4_with_keyframes(
            features, keyframe_features, keyframe_frame_indices
        )
        batch_size, num_frames, height, width = latent_context.shape[:4]
        pixel_shape = (batch_size, self.out_channels, num_frames, height * self.patch_size, width * self.patch_size)
        x_t = randn_tensor(pixel_shape, generator=generator, device=hidden_states.device, dtype=hidden_states.dtype)
        # The keyframe stream draws its own noise, after the video's: its planes are not part of the video canvas.
        keyframe_shape = (*pixel_shape[:2], keyframe_latent_context.shape[1], *pixel_shape[3:])
        keyframe_x_t = randn_tensor(
            keyframe_shape, generator=generator, device=hidden_states.device, dtype=hidden_states.dtype
        )
        return self.denoise_with_keyframes(
            latent_context, x_t, keyframe_latent_context, keyframe_x_t, keyframe_times, num_inference_steps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        generator: torch.Generator | None = None,
        num_inference_steps: int | None = None,
        keyframe_hidden_states: torch.Tensor | None = None,
        keyframe_frame_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        num_inference_steps = num_inference_steps or self.default_num_inference_steps
        if keyframe_hidden_states is not None:
            return self._decode_with_keyframes(
                hidden_states, keyframe_hidden_states, keyframe_frame_indices, generator, num_inference_steps
            )
        latent_context = self.forward_stage_4(self.forward_stages_1_to_3(hidden_states))
        # The context grid is the stage-5 token grid, so the pixel canvas is its shape times the patch size —
        # temporally that is the causal (T - 1) * ratio + 1 mapping of the LTX-2 latent space.
        pixel_shape = (
            hidden_states.shape[0],
            self.out_channels,
            latent_context.shape[1],
            latent_context.shape[2] * self.patch_size,
            latent_context.shape[3] * self.patch_size,
        )
        x_t = randn_tensor(pixel_shape, generator=generator, device=hidden_states.device, dtype=hidden_states.dtype)
        return self.denoise(latent_context, x_t, num_inference_steps)


def _tile_intervals(length: int, tile_size: int, stride: int, min_size: int) -> list[tuple[int, int]]:
    """Overlapping `[start, end)` tiles covering `[0, length)`, with starts spaced `stride` apart.

    A trailing remnant shorter than `min_size` is merged into the previous tile instead of decoded on its own:
    neighborhood attention rejects any grid smaller than its kernel, so a remnant tile cannot always stand alone.
    """
    if length <= tile_size:
        return [(0, length)]
    starts = list(range(0, length, stride))
    while len(starts) > 1 and length - starts[-1] < min_size:
        starts.pop()
    return [(start, min(start + tile_size, length)) for start in starts[:-1]] + [(starts[-1], length)]


class LTX2VideoDiffusionDecoderModel(ModelMixin, AttentionMixin, ConfigMixin):
    r"""
    The LTX-2 diffusion video decoder, introduced in LTX-2.5.

    This is a decoder, not an autoencoder: it has no encoder and cannot produce latents. Encoding stays with
    [`AutoencoderKLLTX2Video`], whose latent space this consumes unchanged, so latents are interchangeable between the
    convolutional decoder and this one.

    It is also a diffusion model rather than a deterministic decoder — it denoises pixels conditioned on a context
    volume built from the latents — which is why it is driven by [`LTX2VideoDiffusionDecodePipeline`] rather than being
    passed as a pipeline's `vae`.

    The latent statistics are carried here as buffers so the decode pipeline can denormalize without loading a second
    autoencoder just for two vectors.

    Decoding can be anchored on *keyframe planes*: single-frame latents at known pixel frames, each encoded as a
    standalone one-frame clip, passed as `keyframe_latents` / `keyframe_frame_indices` to [`decode`]. Every video
    position then also attends to the same spatial window on its two nearest planes. Checkpoints trained for it carry a
    learned tag added to the plane latents, `decoder.type_emb`, which `decoder_keyframe_type_embedding=True` creates;
    without it the planes are decoded untagged.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for it's generic methods implemented
    for all models (such as downloading or saving).
    """

    # Both block types close over a residual add that combines outputs from different children, so a device split
    # inside either one would separate tensors that have to meet again in the same forward.
    _no_split_modules = ["LTX2VideoVaeNABlock", "LTX2VideoVaeDiffusionNABlock"]
    _supports_gradient_checkpointing = False

    @register_to_config
    def __init__(
        self,
        out_channels: int = 3,
        latent_channels: int = 128,
        patch_size: int = 4,
        scaling_factor: float = 1.0,
        decoder_head_dim: int = 64,
        decoder_stage_channels: tuple[int, ...] = (2048, 1024, 512, 512, 256),
        decoder_stage_depths: tuple[int, ...] = (4, 6, 4, 2, 8),
        decoder_stage_kernels: tuple[tuple[int, int, int], ...] = ((3, 7, 7), (3, 7, 7), (3, 5, 5), (3, 5, 5)),
        decoder_upsample_strides: tuple[tuple[int, int, int], ...] = ((1, 2, 2), (2, 1, 1), (2, 2, 2), (2, 2, 2)),
        decoder_upsample_channel_reductions: tuple[int, ...] = (2, 2, 1, 2),
        decoder_stage5_kernel: tuple[int, int, int] = (11, 11, 11),
        decoder_t_emb_dim: int = 384,
        decoder_timestep_scale_multiplier: float = 1000.0,
        decoder_model_output_type: str = "x0",
        decoder_num_inference_steps: int = 1,
        spatial_compression_ratio: int = 32,
        temporal_compression_ratio: int = 8,
        decoder_keyframe_type_embedding: bool = False,
    ) -> None:
        super().__init__()

        self.decoder = LTX2VideoDiffusionDecoder3d(
            in_channels=latent_channels,
            out_channels=out_channels,
            patch_size=patch_size,
            head_dim=decoder_head_dim,
            stage_channels=decoder_stage_channels,
            stage_depths=decoder_stage_depths,
            stage_kernels=decoder_stage_kernels,
            upsample_strides=decoder_upsample_strides,
            upsample_channel_reductions=decoder_upsample_channel_reductions,
            stage5_kernel=decoder_stage5_kernel,
            t_emb_dim=decoder_t_emb_dim,
            temporal_compression_ratio=temporal_compression_ratio,
            timestep_scale_multiplier=decoder_timestep_scale_multiplier,
            model_output_type=decoder_model_output_type,
            default_num_inference_steps=decoder_num_inference_steps,
            keyframe_type_embedding=decoder_keyframe_type_embedding,
        )

        self.spatial_compression_ratio = spatial_compression_ratio
        self.temporal_compression_ratio = temporal_compression_ratio

        # When decoding a large enough video, the memory-dominant stages (the last deterministic stage and the
        # stage-5 diffusion blocks) can run on overlapping tiles that are blended back together. The earlier
        # stages always see the full latent, so tiling changes the output only near tile borders.
        self.use_tiling = False

        # The tile size and the distance between the starts of two consecutive tiles, in pixels/frames of the
        # decoded video; their difference is the blended overlap. Defaults match the reference implementation.
        self.tile_sample_min_height = 768
        self.tile_sample_min_width = 768
        self.tile_sample_min_num_frames = 80
        self.tile_sample_stride_height = 704
        self.tile_sample_stride_width = 704
        self.tile_sample_stride_num_frames = 56

        latents_mean = torch.zeros((latent_channels,), requires_grad=False)
        latents_std = torch.ones((latent_channels,), requires_grad=False)
        self.register_buffer("latents_mean", latents_mean, persistent=True)
        self.register_buffer("latents_std", latents_std, persistent=True)

    def enable_tiling(
        self,
        tile_sample_min_height: int | None = None,
        tile_sample_min_width: int | None = None,
        tile_sample_min_num_frames: int | None = None,
        tile_sample_stride_height: int | None = None,
        tile_sample_stride_width: int | None = None,
        tile_sample_stride_num_frames: int | None = None,
    ) -> None:
        r"""
        Enable tiled decoding. The deterministic upsampling stages before the last one always process the full latent
        (they run at low resolution and are cheap); the last stage and the stage-5 diffusion blocks — which dominate
        decode memory — run on overlapping tiles whose seams are blended linearly.

        Args:
            tile_sample_min_height (`int`, *optional*):
                The height of one decoded tile, in pixels.
            tile_sample_min_width (`int`, *optional*):
                The width of one decoded tile, in pixels.
            tile_sample_min_num_frames (`int`, *optional*):
                The number of frames of one decoded tile.
            tile_sample_stride_height (`int`, *optional*):
                The distance in pixels between the tops of two consecutive vertical tiles; the difference to
                `tile_sample_min_height` is the blended overlap.
            tile_sample_stride_width (`int`, *optional*):
                The distance in pixels between the left edges of two consecutive horizontal tiles.
            tile_sample_stride_num_frames (`int`, *optional*):
                The distance in frames between the starts of two consecutive temporal tiles.
        """
        self.use_tiling = True
        self.tile_sample_min_height = tile_sample_min_height or self.tile_sample_min_height
        self.tile_sample_min_width = tile_sample_min_width or self.tile_sample_min_width
        self.tile_sample_min_num_frames = tile_sample_min_num_frames or self.tile_sample_min_num_frames
        self.tile_sample_stride_height = tile_sample_stride_height or self.tile_sample_stride_height
        self.tile_sample_stride_width = tile_sample_stride_width or self.tile_sample_stride_width
        self.tile_sample_stride_num_frames = tile_sample_stride_num_frames or self.tile_sample_stride_num_frames

    def disable_tiling(self) -> None:
        r"""Disable tiled decoding, returning to decoding the whole video in one pass."""
        self.use_tiling = False

    # Copied from diffusers.models.autoencoders.autoencoder_kl_ltx2.AutoencoderKLLTX2Video.blend_v
    def blend_v(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        blend_extent = min(a.shape[3], b.shape[3], blend_extent)
        for y in range(blend_extent):
            b[:, :, :, y, :] = a[:, :, :, -blend_extent + y, :] * (1 - y / blend_extent) + b[:, :, :, y, :] * (
                y / blend_extent
            )
        return b

    # Copied from diffusers.models.autoencoders.autoencoder_kl_ltx2.AutoencoderKLLTX2Video.blend_h
    def blend_h(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        blend_extent = min(a.shape[4], b.shape[4], blend_extent)
        for x in range(blend_extent):
            b[:, :, :, :, x] = a[:, :, :, :, -blend_extent + x] * (1 - x / blend_extent) + b[:, :, :, :, x] * (
                x / blend_extent
            )
        return b

    # Copied from diffusers.models.autoencoders.autoencoder_kl_ltx2.AutoencoderKLLTX2Video.blend_t
    def blend_t(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        blend_extent = min(a.shape[-3], b.shape[-3], blend_extent)
        for x in range(blend_extent):
            b[:, :, x, :, :] = a[:, :, -blend_extent + x, :, :] * (1 - x / blend_extent) + b[:, :, x, :, :] * (
                x / blend_extent
            )
        return b

    def tiled_decode(
        self,
        z: torch.Tensor,
        generator: torch.Generator | None = None,
        num_inference_steps: int | None = None,
        keyframe_latents: torch.Tensor | None = None,
        keyframe_frame_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        r"""Decode a batch of latents with the last deterministic stage and the diffusion stage running per tile.

        Tiles live on the grid entering the last deterministic stage, where one cell maps to a fixed block of output
        pixels; the `tile_sample_*` sizes are converted to that grid, so they should be multiples of the cell size (8
        px spatially and 2 frames temporally for the production config). Temporal tiles follow the causal frame
        mapping: the tile containing t=0 drops the temporal upsample's duplicate leading frame and only the tile
        containing the video end carries the NATTEN border padding.

        With keyframe planes, each temporal tile carries the planes inside its pixel-frame span plus the nearest one on
        each side of it, with their times rebased on the tile's first cell and first pixel frame, and each spatial tile
        crops the planes with the same slices as the video.
        """
        if keyframe_latents is not None:
            keyframe_frame_indices = self._check_keyframes(z, keyframe_latents, keyframe_frame_indices)
        decoder = self.decoder
        num_inference_steps = num_inference_steps or decoder.default_num_inference_steps
        batch_size = z.shape[0]
        patch_size = decoder.patch_size

        # Pixels per cell of the tiling grid: the last upsample's stride times the stage-5 patch size.
        upsample_stride = decoder.upsamples[-1].stride
        scale_t, scale_h, scale_w = (
            upsample_stride[0],
            upsample_stride[1] * patch_size,
            upsample_stride[2] * patch_size,
        )
        tile_t, stride_t = self.tile_sample_min_num_frames // scale_t, self.tile_sample_stride_num_frames // scale_t
        tile_h, stride_h = self.tile_sample_min_height // scale_h, self.tile_sample_stride_height // scale_h
        tile_w, stride_w = self.tile_sample_min_width // scale_w, self.tile_sample_stride_width // scale_w
        # Every tile must satisfy both remaining neighborhood-attention kernels: the last deterministic stage
        # sees the tile as-is, stage 5 sees it scaled by the upsample stride.
        min_sizes = [
            max(kernel_4, -(-kernel_5 // stride))
            for kernel_4, kernel_5, stride in zip(
                self.config.decoder_stage_kernels[-1], self.config.decoder_stage5_kernel, upsample_stride
            )
        ]

        if keyframe_latents is None:
            features = decoder.forward_stages_1_to_3(z)
        else:
            features, keyframe_features = decoder.forward_stages_1_to_3_with_keyframes(
                z, keyframe_latents, keyframe_frame_indices
            )
        # The trailing ghost frames replicate through the earlier stages' temporal upsamples, whose composed
        # mapping is affine with slope equal to the product of their strides.
        ghost_frames = decoder.trailing_pad_latent_frames * math.prod(up.stride[0] for up in decoder.upsamples[:-1])
        num_frames = features.shape[1] - ghost_frames
        height, width = features.shape[2], features.shape[3]

        temporal_tiles = _tile_intervals(num_frames, tile_t, stride_t, min_sizes[0])
        height_tiles = _tile_intervals(height, tile_h, stride_h, min_sizes[1])
        width_tiles = _tile_intervals(width, tile_w, stride_w, min_sizes[2])
        blend_frames = (tile_t - stride_t) * scale_t
        blend_height = (tile_h - stride_h) * scale_h
        blend_width = (tile_w - stride_w) * scale_w

        # A single-step x0 decode predicts pixels from pure noise, so each tile draws its own; a multi-step
        # decode integrates its noise across steps, so overlapping tiles must start from the same canvas.
        single_step_x0 = num_inference_steps == 1 and decoder.model_output_type == "x0"
        x_t_full = None
        if not single_step_x0:
            pixel_frames = num_frames * scale_t - (1 if scale_t == 2 else 0)
            x_t_full = randn_tensor(
                (batch_size, decoder.out_channels, pixel_frames, height * scale_h, width * scale_w),
                generator=generator,
                device=z.device,
                dtype=z.dtype,
            )

        frame_groups = []
        for t0, t1 in temporal_tiles:
            is_origin = t0 == 0
            is_trailing = t1 == num_frames
            # The tile containing the video end takes the ghost frames with it into stage 4.
            feature_t1 = features.shape[1] if is_trailing else t1
            # A non-origin tile keeps the duplicate leading frame, placing its first cell one pixel frame earlier
            # than `t0 * scale_t` — the causal 1-then-`scale_t` frame mapping.
            pixel_t0 = t0 * scale_t - (1 if not is_origin and scale_t == 2 else 0)
            if keyframe_latents is not None:
                tile_pixel_frames = (t1 - t0) * scale_t - (1 if is_origin and scale_t == 2 else 0)
                keep = _keyframe_planes_for_tile(keyframe_frame_indices, pixel_t0, pixel_t0 + tile_pixel_frames - 1)
                tile_keyframe_features = keyframe_features[:, keep.to(keyframe_features.device)]
                tile_keyframe_indices = keyframe_frame_indices[keep]
            rows = []
            for h0, h1 in height_tiles:
                row = []
                for w0, w1 in width_tiles:
                    if keyframe_latents is None:
                        context = decoder.forward_stage_4(
                            features[:, t0:feature_t1, h0:h1, w0:w1],
                            drop_leading_frame=is_origin,
                            crop_trailing_ghost=is_trailing,
                        )
                    else:
                        context, keyframe_context, keyframe_times = decoder.forward_stage_4_with_keyframes(
                            features[:, t0:feature_t1, h0:h1, w0:w1],
                            tile_keyframe_features[:, :, h0:h1, w0:w1],
                            tile_keyframe_indices,
                            drop_leading_frame=is_origin,
                            crop_trailing_ghost=is_trailing,
                            stage_4_time_origin=float(t0),
                            pixel_time_origin=float(pixel_t0),
                        )
                    tile_pixel_shape = (
                        batch_size,
                        decoder.out_channels,
                        context.shape[1],
                        context.shape[2] * patch_size,
                        context.shape[3] * patch_size,
                    )
                    if single_step_x0:
                        x_t = randn_tensor(tile_pixel_shape, generator=generator, device=z.device, dtype=z.dtype)
                    else:
                        x_t = x_t_full[
                            :,
                            :,
                            pixel_t0 : pixel_t0 + tile_pixel_shape[2],
                            h0 * scale_h : h0 * scale_h + tile_pixel_shape[3],
                            w0 * scale_w : w0 * scale_w + tile_pixel_shape[4],
                        ]
                    if keyframe_latents is None:
                        row.append(decoder.denoise(context, x_t, num_inference_steps))
                        continue
                    # The keyframe stream always draws its own noise, after the video's.
                    keyframe_x_t = randn_tensor(
                        (*tile_pixel_shape[:2], keyframe_context.shape[1], *tile_pixel_shape[3:]),
                        generator=generator,
                        device=z.device,
                        dtype=z.dtype,
                    )
                    row.append(
                        decoder.denoise_with_keyframes(
                            context, x_t, keyframe_context, keyframe_x_t, keyframe_times, num_inference_steps
                        )
                    )
                rows.append(row)

            result_rows = []
            for i, row in enumerate(rows):
                result_row = []
                for j, tile in enumerate(row):
                    # blend the above tile and the left tile to the current tile and add the current tile to
                    # the result row
                    if i > 0:
                        tile = self.blend_v(rows[i - 1][j], tile, blend_height)
                    if j > 0:
                        tile = self.blend_h(row[j - 1], tile, blend_width)
                    # The last tile can extend past the stride grid (a short remnant is merged into it), so it
                    # keeps its full extent instead of being cropped to the stride.
                    keep_height = stride_h * scale_h if i < len(rows) - 1 else tile.shape[3]
                    keep_width = stride_w * scale_w if j < len(row) - 1 else tile.shape[4]
                    result_row.append(tile[:, :, :, :keep_height, :keep_width])
                result_rows.append(torch.cat(result_row, dim=4))
            frame_groups.append(torch.cat(result_rows, dim=3))

        result = []
        for k, group in enumerate(frame_groups):
            if k > 0:
                group = self.blend_t(frame_groups[k - 1], group, blend_frames)
            if k < len(frame_groups) - 1:
                # The origin group is one frame short of `stride * scale`: its first cell decodes to a single
                # pixel frame under the causal mapping.
                keep_frames = stride_t * scale_t - (1 if k == 0 and scale_t == 2 else 0)
                group = group[:, :, :keep_frames]
            result.append(group)
        return torch.cat(result, dim=2)

    def _check_keyframes(
        self, z: torch.Tensor, keyframe_latents: torch.Tensor, keyframe_frame_indices: torch.Tensor | None
    ) -> torch.Tensor:
        """Validate a keyframe input and return its frame indices as a 1-D int64 tensor."""
        if keyframe_frame_indices is None:
            raise ValueError("`keyframe_frame_indices` is required when `keyframe_latents` is passed.")
        keyframe_frame_indices = torch.as_tensor(keyframe_frame_indices, dtype=torch.int64).cpu()
        if keyframe_latents.ndim != 5:
            raise ValueError(f"`keyframe_latents` must be (B, C, P, H, W), got {tuple(keyframe_latents.shape)}.")
        num_planes = keyframe_latents.shape[2]
        if num_planes == 0:
            raise ValueError("`keyframe_latents` needs at least one plane; omit it for a plain decode.")
        if keyframe_latents.shape[:2] != z.shape[:2] or keyframe_latents.shape[3:] != z.shape[3:]:
            raise ValueError(
                f"`keyframe_latents` {tuple(keyframe_latents.shape)} must match the batch size, channels, height and "
                f"width of `z` {tuple(z.shape)}."
            )
        if keyframe_frame_indices.shape != (num_planes,):
            raise ValueError(
                f"`keyframe_frame_indices` must be 1-D with one pixel frame per plane ({num_planes}), got shape "
                f"{tuple(keyframe_frame_indices.shape)}."
            )
        if int(keyframe_frame_indices.min()) < 0:
            raise ValueError("`keyframe_frame_indices` must be non-negative pixel frame indices.")
        if self.decoder.type_emb is None:
            logger.warning(
                "Decoding with keyframe planes on a checkpoint without a keyframe tag (`decoder.type_emb`): the planes "
                "enter the decoder untagged. Checkpoints trained for keyframe decoding are converted with "
                "`decoder_keyframe_type_embedding=True`."
            )
        return keyframe_frame_indices

    @apply_forward_hook
    def decode(
        self,
        z: torch.Tensor,
        generator: torch.Generator | None = None,
        num_inference_steps: int | None = None,
        return_dict: bool = True,
        keyframe_latents: torch.Tensor | None = None,
        keyframe_frame_indices: torch.Tensor | None = None,
    ) -> DecoderOutput | torch.Tensor:
        """Decode a batch of latents.

        `z` is expected to be denormalized already (the pipeline applies `latents_mean` / `latents_std`), matching
        [`AutoencoderKLLTX2Video`]. This decoder denoises, so pass `generator` for reproducibility.

        `keyframe_latents` `(B, C, P, H, W)`, denormalized like `z`, anchors the decode on `P` keyframe planes at the
        pixel frames `keyframe_frame_indices` `(P,)`. Each plane must be the latent of a standalone one-frame clip.
        Planes may lie outside the decoded clip: they are ranked by temporal distance.
        """
        if keyframe_latents is not None:
            keyframe_frame_indices = self._check_keyframes(z, keyframe_latents, keyframe_frame_indices)
        tile_latent_min_height = self.tile_sample_min_height // self.spatial_compression_ratio
        tile_latent_min_width = self.tile_sample_min_width // self.spatial_compression_ratio
        tile_latent_min_num_frames = self.tile_sample_min_num_frames // self.temporal_compression_ratio
        if self.use_tiling and (
            z.shape[2] > tile_latent_min_num_frames
            or z.shape[3] > tile_latent_min_height
            or z.shape[4] > tile_latent_min_width
        ):
            decoded = self.tiled_decode(
                z,
                generator=generator,
                num_inference_steps=num_inference_steps,
                keyframe_latents=keyframe_latents,
                keyframe_frame_indices=keyframe_frame_indices,
            )
        else:
            decoded = self.decoder(
                z,
                generator=generator,
                num_inference_steps=num_inference_steps,
                keyframe_hidden_states=keyframe_latents,
                keyframe_frame_indices=keyframe_frame_indices,
            )

        if not return_dict:
            return (decoded,)
        return DecoderOutput(sample=decoded)

    def forward(
        self,
        z: torch.Tensor,
        generator: torch.Generator | None = None,
        num_inference_steps: int | None = None,
        return_dict: bool = True,
        keyframe_latents: torch.Tensor | None = None,
        keyframe_frame_indices: torch.Tensor | None = None,
    ) -> DecoderOutput | tuple[torch.Tensor]:
        r"""
        Args:
            z (`torch.Tensor`):
                Latents of shape `(B, C, F, H, W)`, expected to be denormalized already (the pipeline applies
                `latents_mean` / `latents_std`), matching [`AutoencoderKLLTX2Video`].
            generator (`torch.Generator`, *optional*):
                This decoder denoises, so pass a generator to make decoding reproducible.
            num_inference_steps (`int`, *optional*):
                Number of denoising steps. Defaults to the decoder's `decoder_num_inference_steps` config value.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`~models.autoencoders.vae.DecoderOutput`] instead of a plain tuple.
            keyframe_latents (`torch.Tensor`, *optional*):
                Keyframe planes of shape `(B, C, P, H, W)`, denormalized like `z` and on its latent grid, one latent
                frame per plane, each encoded as a standalone one-frame clip. Every video position also attends to the
                same spatial window on its two nearest planes.
            keyframe_frame_indices (`torch.Tensor`, *optional*):
                The `(P,)` pixel frame of each plane in the decoded video. Required with `keyframe_latents`.

        Returns:
            [`~models.autoencoders.vae.DecoderOutput`] or `tuple`
        """
        return self.decode(
            z,
            generator=generator,
            num_inference_steps=num_inference_steps,
            return_dict=return_dict,
            keyframe_latents=keyframe_latents,
            keyframe_frame_indices=keyframe_frame_indices,
        )
