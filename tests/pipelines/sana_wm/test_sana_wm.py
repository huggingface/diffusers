# Copyright 2025 The HuggingFace Team and SANA-WM Authors. All rights reserved.
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

"""SANA-WM CPU unit tests.

Covers the standalone helpers (action DSL, intrinsics math, resize-and-crop) and the public-surface registration.
"""

import inspect

import numpy as np
import pytest
import torch
from PIL import Image

import diffusers
from diffusers import DiffusionPipeline, SanaWMPipeline, SanaWMPipelineOutput
from diffusers.pipelines.sana_wm import SanaWMLTX2Refiner
from diffusers.pipelines.sana_wm.cam_utils import (
    TARGET_HEIGHT,
    TARGET_WIDTH,
    action_string_to_c2w,
    resize_and_center_crop,
    snap_num_frames,
    transform_intrinsics_for_crop,
)

from ...testing_utils import enable_full_determinism


enable_full_determinism()


class TestSanaWMCamUtils:
    """Pure-numpy/PIL helpers — no torch.cuda required."""

    def test_action_dsl_forward_only(self):
        c2w = action_string_to_c2w("w-5", translation_speed=0.1)
        # 5 action frames + leading identity = 6 total
        assert c2w.shape == (6, 4, 4)
        assert c2w.dtype == np.float32
        # First frame is identity (the anchor).
        np.testing.assert_allclose(c2w[0], np.eye(4, dtype=np.float32), atol=1e-6)
        # 'w' moves forward (+Z in OpenCV convention).
        assert float(c2w[-1, 2, 3]) == pytest.approx(0.5, abs=1e-5)
        # No yaw / pitch -> rotation is identity throughout.
        for i in range(c2w.shape[0]):
            np.testing.assert_allclose(c2w[i, :3, :3], np.eye(3), atol=1e-6)

    def test_action_dsl_concat_segments(self):
        c2w = action_string_to_c2w("w-3,a-2", translation_speed=0.1)
        assert c2w.shape == (6, 4, 4)  # 3 + 2 + identity anchor

    @pytest.mark.parametrize(
        "action",
        ["", "x-5", "w-0"],
        ids=["empty", "unknown-key", "zero-length-segment"],
    )
    def test_action_dsl_rejects_bad_input(self, action):
        with pytest.raises(ValueError):
            action_string_to_c2w(action)

    def test_action_dsl_none_segment_is_idle(self):
        c2w = action_string_to_c2w("none-3", translation_speed=0.1)
        assert c2w.shape == (4, 4, 4)
        # No motion -> all frames are identity.
        for i in range(c2w.shape[0]):
            np.testing.assert_allclose(c2w[i], np.eye(4), atol=1e-6)

    def test_transform_intrinsics_for_crop_scalar(self):
        # (fx, fy, cx, cy) for a 1000x500 source, resized to 1280x704, then
        # center-cropped to 1280x704 (no extra crop offset).
        intr = np.array([800.0, 800.0, 500.0, 250.0], dtype=np.float32)
        out = transform_intrinsics_for_crop(intr, src_size=(1000, 500), resized_size=(1280, 704), crop_offset=(0, 0))
        assert float(out[0]) == pytest.approx(800.0 * 1280 / 1000, abs=1e-4)  # fx scales with x
        assert float(out[1]) == pytest.approx(800.0 * 704 / 500, abs=1e-4)
        assert float(out[2]) == pytest.approx(500.0 * 1280 / 1000, abs=1e-4)
        assert float(out[3]) == pytest.approx(250.0 * 704 / 500, abs=1e-4)

    def test_transform_intrinsics_for_crop_with_offset(self):
        intr = np.array([800.0, 800.0, 500.0, 250.0], dtype=np.float32)
        # After resize, an extra crop offset shifts the principal point.
        out = transform_intrinsics_for_crop(
            intr, src_size=(1000, 500), resized_size=(2000, 1000), crop_offset=(360, 148)
        )
        assert float(out[2]) == pytest.approx(500.0 * 2.0 - 360.0, abs=1e-4)
        assert float(out[3]) == pytest.approx(250.0 * 2.0 - 148.0, abs=1e-4)

    def test_resize_and_center_crop_default_target(self):
        src = Image.new("RGB", (1691, 930))
        cropped, src_size, resized_size, crop_offset = resize_and_center_crop(src)
        assert cropped.size == (TARGET_WIDTH, TARGET_HEIGHT)
        assert src_size == (1691, 930)
        # Resize preserves aspect; one of the resized dimensions equals the target.
        resized_width, resized_height = resized_size
        assert resized_width >= TARGET_WIDTH
        assert resized_height >= TARGET_HEIGHT
        crop_left, crop_top = crop_offset
        assert crop_left >= 0
        assert crop_top >= 0
        # Center crop produces 0 offset on the dimension that hit the target exactly.
        assert crop_left == 0 or crop_top == 0

    # The LTX-2 VAE requires a (8k + 1)-shaped temporal dim, so ``snap_num_frames`` rounds to
    # the nearest such value (ties break to the ceil).
    @pytest.mark.parametrize("num_frames", [1, 9, 17, 81, 161, 321, 801])
    def test_snap_num_frames_is_a_noop_on_8k_plus_1(self, num_frames):
        assert snap_num_frames(num_frames) == num_frames

    @pytest.mark.parametrize(
        ("num_frames", "expected"),
        [
            (2, 1),
            (10, 9),  # 10 is closer to 9 than 17
            (80, 81),  # 80 is closer to 81 than 73
            (100, 97),  # 100 is closer to 97 than 105
        ],
    )
    def test_snap_num_frames_to_8k_plus_1(self, num_frames, expected):
        assert snap_num_frames(num_frames) == expected

    def test_snap_num_frames_respects_upper_bound(self):
        # ``upper_bound`` caps the result (the snap falls back to the floor).
        assert snap_num_frames(100, upper_bound=100) <= 100
        assert snap_num_frames(100, upper_bound=100) == 97


class TestSanaWMRegistration:
    """Verify the SANA-WM symbols are reachable through the public diffusers surface."""

    @pytest.mark.parametrize(
        "name", ["SanaWMPipeline", "SanaWMTransformer3DModel", "SanaWMLTX2Refiner", "SanaWMPipelineOutput"]
    )
    def test_top_level_symbols(self, name):
        assert hasattr(diffusers, name), f"{name!r} not exported from diffusers top-level"

    def test_pipeline_output_dataclass(self):
        frames = np.zeros((3, 8, 8, 3), dtype=np.float32)
        c2w = np.broadcast_to(np.eye(4, dtype=np.float32), (3, 4, 4)).copy()
        latent = torch.zeros(1, 16, 1, 4, 4)
        output = SanaWMPipelineOutput(frames=frames, c2w=c2w, latent=latent)
        assert tuple(output.frames.shape) == (3, 8, 8, 3)
        assert tuple(output.c2w.shape) == (3, 4, 4)
        assert tuple(output.latent.shape) == (1, 16, 1, 4, 4)

    def test_refiner_is_pipeline_with_ar_call_defaults(self):
        # The refiner is a standalone DiffusionPipeline.
        assert issubclass(SanaWMLTX2Refiner, DiffusionPipeline)

        # Its denoising entry point is ``__call__`` with the canonical AR defaults.
        params = inspect.signature(SanaWMLTX2Refiner.__call__).parameters
        assert "block_size" in params
        assert "kv_max_frames" in params
        # AR mode is on by default.
        assert params["block_size"].default == 3
        assert params["kv_max_frames"].default == 11

    @pytest.mark.parametrize(
        "name",
        [
            "intrinsics",
            "c2w",
            "action",
            "use_refiner",
            # Standard diffusers pipeline arguments.
            "generator",
            "prompt_embeds",
            "prompt_attention_mask",
            "negative_prompt_embeds",
            "negative_prompt_attention_mask",
        ],
    )
    def test_pipeline_call_intrinsics_signature(self, name):
        params = inspect.signature(SanaWMPipeline.__call__).parameters
        assert name in params

    def test_pipeline_call_takes_generator_not_seed(self):
        # Pipelines take a `generator`; `seed` shortcuts are not part of the diffusers interface.
        params = inspect.signature(SanaWMPipeline.__call__).parameters
        assert "seed" not in params
        assert "refiner_seed" not in params
