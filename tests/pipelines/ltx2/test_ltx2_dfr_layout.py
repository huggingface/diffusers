# Copyright 2026 The HuggingFace Team.
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

import itertools

import pytest
import torch

from diffusers.pipelines.ltx2.dfr_layout import (
    epilogue_tiles,
    pixel_to_latent_index,
    rectangular_mask_1d,
    resolve_canvas,
    split_by_count,
    split_canvas_at_seams,
    temporal_tile_plan,
    trapezoidal_mask_1d,
)


class TestResolveCanvas:
    def test_exact_multiple_of_a_segment_is_not_padded(self):
        canvas_frames, segment, positions = resolve_canvas(97)

        assert (canvas_frames, segment) == (97, 32)
        assert positions == [32, 64, 96]

    def test_smaller_segment_wins_when_it_pads_strictly_less(self):
        canvas_frames, segment, positions = resolve_canvas(121)

        assert (canvas_frames, segment) == (121, 24)
        assert positions == [24, 48, 72, 96, 120]

    def test_tail_is_padded_up_to_a_whole_segment(self):
        canvas_frames, segment, positions = resolve_canvas(41)

        assert canvas_frames > 41
        assert (canvas_frames - 1) % segment == 0
        assert positions[-1] == canvas_frames - 1

    def test_segment_grid_scales_with_the_vae_temporal_ratio(self):
        canvas_frames, segment, positions = resolve_canvas(9, temporal_compression_ratio=2)

        assert (canvas_frames, segment) == (9, 8)
        assert positions == [8]

    def test_a_request_off_the_latent_grid_is_rejected(self):
        with pytest.raises(ValueError, match="num_frames"):
            resolve_canvas(100)


class TestPixelToLatentIndex:
    def test_a_border_frame_maps_to_its_latent(self):
        assert pixel_to_latent_index(64) == 8

    def test_a_frame_off_the_border_is_rejected(self):
        with pytest.raises(ValueError, match="latent border"):
            pixel_to_latent_index(65)


class TestSegmentDealing:
    def test_leftover_segments_go_to_the_leading_tiles(self):
        # 5 segments over 3 tiles deals 2/2/1, so the tiles end on the 2nd, 4th and 5th seam.
        intervals = split_canvas_at_seams([0, 1, 2, 3, 4, 5], num_tiles=3, overlap=0, dim_size=6)
        assert [interval.end - 1 for interval in intervals] == [2, 4, 5]

    def test_tile_count_is_clamped_to_the_segment_count(self):
        # 2 segments cannot fill 4 tiles; each tile owns one.
        intervals = split_canvas_at_seams([0, 1, 2], num_tiles=4, overlap=0, dim_size=3)
        assert [(interval.start, interval.end) for interval in intervals] == [(0, 2), (2, 3)]


class TestTemporalTilePlan:
    def test_a_single_tile_covers_the_whole_canvas(self):
        (tile,) = temporal_tile_plan([32, 64], 65, 1)

        assert (tile.pixel_start, tile.pixel_end) == (0, 64)
        assert (tile.interval.start, tile.interval.end) == (0, 9)
        assert tile.interval.left_ramp == 0
        assert tile.anchors == (32, 64)
        assert tile.slots == (16, 48)

    def test_tiles_are_gapless_after_dropping_their_ramps(self):
        tiles = temporal_tile_plan([32, 64, 96, 128], 129, 4)

        covered = []
        for tile in tiles:
            covered.extend(range(tile.interval.start + tile.interval.left_ramp, tile.interval.end))
        assert covered == list(range(0, 17))

    def test_non_first_tiles_reach_back_one_segment_for_the_shared_seam(self):
        first, second = temporal_tile_plan([32, 64, 96, 128], 129, 2)

        assert second.pixel_start == 32
        assert first.pixel_end == 64
        assert second.pixel_end == 128
        # The ramp swallows the lead-in *and* the seam latent, so the previous tile keeps the shared keyframe.
        assert second.interval.start + second.interval.left_ramp == first.interval.end

    def test_the_first_tile_window_start_contributes_no_anchor(self):
        tiles = temporal_tile_plan([32, 64, 96], 97, 2)

        assert 0 not in tiles[0].anchors
        assert tiles[1].pixel_start in tiles[1].anchors

    def test_owned_segment_runs_are_balanced_largest_first(self):
        tiles = temporal_tile_plan([32, 64, 96], 97, 2)

        # 3 segments / 2 tiles: first owns 2, second owns 1. Lead-in re-invents slot 48; earlier tile wins.
        assert tiles[0].slots == (16, 48)
        assert tiles[1].slots == (48, 80)

    def test_every_seam_in_a_window_is_an_anchor(self):
        tiles = temporal_tile_plan([32, 64, 96, 128], 129, 2)

        assert tiles[0].anchors == (32, 64)
        assert tiles[1].anchors == (32, 64, 96, 128)

    def test_a_ramp_drop_stitch_reproduces_the_canvas(self):
        tiles = temporal_tile_plan([32, 64, 96, 128], 129, 2)
        tile_latents = [
            torch.arange(tile.interval.start, tile.interval.end, dtype=torch.float32).reshape(1, 1, -1, 1, 1)
            for tile in tiles
        ]

        stitched = torch.cat([latent[:, :, tile.interval.left_ramp :] for latent, tile in zip(tile_latents, tiles)], 2)

        assert stitched.shape == (1, 1, 17, 1, 1)
        assert torch.equal(stitched.flatten(), torch.arange(17, dtype=torch.float32))


class TestSplitCanvasAtSeams:
    def test_seams_must_end_on_the_last_cell(self):
        with pytest.raises(ValueError, match="last cell"):
            split_canvas_at_seams([0, 4, 8], num_tiles=2, overlap=5, dim_size=13)

    def test_seams_must_be_increasing(self):
        with pytest.raises(ValueError, match="strictly increasing"):
            split_canvas_at_seams([0, 8, 4, 12], num_tiles=2, overlap=5, dim_size=13)


class TestSplitByCount:
    @pytest.mark.parametrize(
        ("dim_size", "num_tiles", "overlap"),
        list(itertools.product((13, 16, 17, 21, 34), (1, 2, 3, 4), (0, 2, 6))),
    )
    def test_tiles_cover_the_axis_and_share_exactly_the_overlap(self, dim_size, num_tiles, overlap):
        if num_tiles > 1 and overlap > dim_size - num_tiles:
            pytest.skip("layout the caller is required to clamp first")
        intervals = split_by_count(dim_size, num_tiles, overlap)

        assert len(intervals) == num_tiles
        assert intervals[0].start == 0
        assert intervals[-1].end == dim_size
        for earlier, later in itertools.pairwise(intervals):
            assert earlier.end - later.start == overlap
            assert earlier.right_ramp == overlap
            assert later.left_ramp == overlap

    def test_trapezoidal_weights_over_a_split_sum_to_one_everywhere(self):
        dim_size, num_tiles, overlap = 34, 3, 6
        summed = torch.zeros(dim_size)
        for interval in split_by_count(dim_size, num_tiles, overlap):
            mask = trapezoidal_mask_1d(interval.end - interval.start, interval.left_ramp, interval.right_ramp)
            summed[interval.start : interval.end] += mask

        assert torch.allclose(summed, torch.ones(dim_size), atol=1e-6)

    def test_a_single_tile_is_the_whole_axis_unramped(self):
        (interval,) = split_by_count(9, 1, 6)

        assert (interval.start, interval.end) == (0, 9)
        assert (interval.left_ramp, interval.right_ramp) == (0, 0)


class TestRectangularMask:
    def test_the_lead_in_is_dropped_and_the_rest_kept_whole(self):
        assert torch.equal(rectangular_mask_1d(5, 2), torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0]))
        # The opening tile has no lead-in and so keeps every cell.
        assert torch.equal(rectangular_mask_1d(4, 0), torch.ones(4))

    def test_weights_over_a_seam_cut_sum_to_one_everywhere(self):
        seams, dim_size = [0, 4, 8, 12, 16], 17
        summed = torch.zeros(dim_size)
        for interval in split_canvas_at_seams(seams, 2, overlap=5, dim_size=dim_size):
            summed[interval.start : interval.end] += rectangular_mask_1d(
                interval.end - interval.start, interval.left_ramp
            )

        assert torch.equal(summed, torch.ones(dim_size))


class TestTrapezoidalMask:
    def test_ramps_are_the_interior_of_a_linspace(self):
        assert torch.allclose(trapezoidal_mask_1d(5, 3, 0), torch.tensor([0.25, 0.5, 0.75, 1.0, 1.0]))
        assert torch.allclose(trapezoidal_mask_1d(5, 0, 3), torch.tensor([1.0, 1.0, 0.75, 0.5, 0.25]))
        # A tile with neither neighbour keeps every cell.
        assert torch.equal(trapezoidal_mask_1d(4, 0, 0), torch.ones(4))


class TestEpilogueTiles:
    def test_epilogue_tiles_cover_the_canvas_with_unit_weight(self):
        tiles = epilogue_tiles(latent_shape=(9, 16, 16), frame_tiles=2, frame_seams=[3, 6])

        assert len(tiles) == 2 * 2 * 2
        summed = torch.zeros(9, 16, 16)
        for frames, heights, widths, weight in tiles:
            summed[frames, heights, widths] += weight
        assert torch.allclose(summed, torch.ones_like(summed), atol=1e-6)

    def test_epilogue_temporal_tiles_are_cut_on_the_last_round_seams(self):
        # Ten segments over 61 latent frames dealt to four tiles: the leading two take three each.
        tiles = epilogue_tiles(latent_shape=(61, 8, 8), frame_tiles=4, frame_seams=[6 * i for i in range(1, 11)])

        windows = sorted({(frames.start, frames.stop) for frames, _, _, _ in tiles})
        assert windows == [(0, 19), (12, 37), (30, 49), (42, 61)]
        # The run-up is dropped outright rather than blended, so every non-first window opens on exact zeros and the
        # window before it keeps the seam cell.
        head = next(weight for frames, _, _, weight in tiles if frames.start == 12)[:8, 0, 0]
        assert torch.equal(head, torch.tensor([0.0] * 7 + [1.0]))

    def test_epilogue_temporal_tiles_blend_when_the_seams_are_not_interior(self):
        # A canvas with no refine rounds behind it has only its own end cell as a "seam", which cuts nothing.
        tiles = epilogue_tiles(latent_shape=(9, 8, 8), frame_tiles=2, frame_seams=[8])

        # Nothing is cut, so the later window ramps in instead of opening on zeros.
        head = next(weight for frames, _, _, weight in tiles if frames.start > 0)[0, 0, 0]
        assert 0.0 < head < 1.0
        summed = torch.zeros(9, 8, 8)
        for frames, heights, widths, weight in tiles:
            summed[frames, heights, widths] += weight
        assert torch.allclose(summed, torch.ones_like(summed), atol=1e-6)

    def test_epilogue_tiling_falls_back_on_an_axis_too_short_to_split(self):
        tiles = epilogue_tiles(latent_shape=(3, 16, 16), frame_tiles=4, frame_seams=[1, 2])

        # 3 cells cannot hold 4 tiles, so the temporal axis runs whole and only the spatial split survives.
        assert len(tiles) == 1 * 2 * 2
        assert {(frames.start, frames.stop) for frames, _, _, _ in tiles} == {(0, 3)}

    def test_epilogue_tiling_clamps_an_overlap_the_axis_cannot_hold(self):
        tiles = epilogue_tiles(latent_shape=(9, 3, 16), frame_tiles=1, frame_seams=[])

        # Height is 3 cells, so the requested 12-cell overlap is clamped to 1 and the two tiles still tile it.
        height_slices = sorted({(heights.start, heights.stop) for _, heights, _, _ in tiles})
        assert height_slices == [(0, 2), (1, 3)]
