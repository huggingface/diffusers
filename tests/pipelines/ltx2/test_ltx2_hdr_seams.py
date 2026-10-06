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

"""Seam keyframe positions of the LTX-2.5 SDR-To-HDR IC-LoRA.

Expected values were produced by the reference implementation's own seam function (Lightricks/LTX-2, `hdr_ic_lora.py`
`_dfr_seam_pixel_positions` on top of `dfr_helpers/layout.py` `resolve_canvas`), not by the code under test.
"""

import pytest

from diffusers.pipelines.ltx2.dfr_layout import resolve_canvas, resolve_seam_positions


@pytest.mark.parametrize(
    "num_frames, expected",
    [
        (1, []),
        (9, []),
        (17, []),
        (25, [24]),
        (33, [32]),
        (49, [24, 48]),
        (57, [32]),
        (97, [32, 64, 96]),
        (105, [24, 48, 72, 96]),
        (121, [24, 48, 72, 96, 120]),
        (129, [32, 64, 96, 128]),
        (193, [32, 64, 96, 128, 160, 192]),
        (201, [24, 48, 72, 96, 120, 144, 168, 192]),
        (257, [32, 64, 96, 128, 160, 192, 224, 256]),
    ],
)
def test_seam_positions_match_reference(num_frames, expected):
    assert resolve_seam_positions(num_frames) == expected
    # High quality runs on the frame-doubled `2N - 1` grid: positions are computed on `N`, then doubled.
    assert resolve_seam_positions(num_frames, high_quality=True) == [2 * position for position in expected]


def test_seam_positions_are_the_canvas_keyframes_inside_the_clip():
    # 105 frames pick 24-frame segments and pad the canvas to 121; the seams stop short of the padding, leaving an
    # 8-frame tail segment.
    canvas, segment, positions = resolve_canvas(105)
    assert (canvas, segment, positions) == (121, 24, [24, 48, 72, 96, 120])
    assert resolve_seam_positions(105) == [24, 48, 72, 96]

    for num_frames in range(9, 2002, 8):
        _, _, positions = resolve_canvas(num_frames)
        assert resolve_seam_positions(num_frames) == [p for p in positions if p < num_frames]


def test_single_frame_has_no_seams_where_the_canvas_rejects_it():
    # The one 8k+1 frame count below 9 is a single frame: the reference returns no seams for it, while the DFR canvas
    # (which needs at least one latent step) rejects it.
    assert resolve_seam_positions(1) == []
    with pytest.raises(ValueError):
        resolve_canvas(1)


@pytest.mark.parametrize("num_frames", [10, 96, 98])
def test_seam_positions_reject_off_grid_frame_counts(num_frames):
    with pytest.raises(ValueError):
        resolve_seam_positions(num_frames)
