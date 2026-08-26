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

Covers the standalone helpers (action DSL, intrinsics math, resize-and-crop) and
the public-surface registration.
"""

import unittest

import numpy as np
from PIL import Image

from diffusers import SanaWMPipeline, SanaWMPipelineOutput
from diffusers.pipelines.sana_wm import SanaWMLTX2Refiner
from diffusers.pipelines.sana_wm.cam_utils import (
    TARGET_HEIGHT,
    TARGET_WIDTH,
    action_string_to_c2w,
    resize_and_center_crop,
    snap_num_frames,
    transform_intrinsics_for_crop,
)


class SanaWMCamUtilsTests(unittest.TestCase):
    """Pure-numpy/PIL helpers — no torch.cuda required."""

    def test_action_dsl_forward_only(self):
        c2w = action_string_to_c2w("w-5", translation_speed=0.1)
        # 5 action frames + leading identity = 6 total
        self.assertEqual(c2w.shape, (6, 4, 4))
        self.assertEqual(c2w.dtype, np.float32)
        # First frame is identity (the anchor).
        np.testing.assert_allclose(c2w[0], np.eye(4, dtype=np.float32), atol=1e-6)
        # 'w' moves forward (+Z in OpenCV convention).
        self.assertAlmostEqual(float(c2w[-1, 2, 3]), 0.5, places=5)
        # No yaw / pitch -> rotation is identity throughout.
        for i in range(c2w.shape[0]):
            np.testing.assert_allclose(c2w[i, :3, :3], np.eye(3), atol=1e-6)

    def test_action_dsl_concat_segments(self):
        c2w = action_string_to_c2w("w-3,a-2", translation_speed=0.1)
        self.assertEqual(c2w.shape, (6, 4, 4))  # 3 + 2 + identity anchor

    def test_action_dsl_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            action_string_to_c2w("")
        with self.assertRaises(ValueError):
            action_string_to_c2w("x-5")  # 'x' is not in WASD/IJKL
        with self.assertRaises(ValueError):
            action_string_to_c2w("w-0")  # zero-length segment

    def test_action_dsl_none_segment_is_idle(self):
        c2w = action_string_to_c2w("none-3", translation_speed=0.1)
        self.assertEqual(c2w.shape, (4, 4, 4))
        # No motion -> all frames are identity.
        for i in range(c2w.shape[0]):
            np.testing.assert_allclose(c2w[i], np.eye(4), atol=1e-6)

    def test_transform_intrinsics_for_crop_scalar(self):
        # (fx, fy, cx, cy) for a 1000x500 source, resized to 1280x704, then
        # center-cropped to 1280x704 (no extra crop offset).
        intr = np.array([800.0, 800.0, 500.0, 250.0], dtype=np.float32)
        out = transform_intrinsics_for_crop(intr, src_size=(1000, 500), resized_size=(1280, 704), crop_offset=(0, 0))
        self.assertAlmostEqual(float(out[0]), 800.0 * 1280 / 1000, places=4)  # fx scales with x
        self.assertAlmostEqual(float(out[1]), 800.0 * 704 / 500, places=4)
        self.assertAlmostEqual(float(out[2]), 500.0 * 1280 / 1000, places=4)
        self.assertAlmostEqual(float(out[3]), 250.0 * 704 / 500, places=4)

    def test_transform_intrinsics_for_crop_with_offset(self):
        intr = np.array([800.0, 800.0, 500.0, 250.0], dtype=np.float32)
        # After resize, an extra crop offset shifts the principal point.
        out = transform_intrinsics_for_crop(
            intr, src_size=(1000, 500), resized_size=(2000, 1000), crop_offset=(360, 148)
        )
        self.assertAlmostEqual(float(out[2]), 500.0 * 2.0 - 360.0, places=4)
        self.assertAlmostEqual(float(out[3]), 250.0 * 2.0 - 148.0, places=4)

    def test_resize_and_center_crop_default_target(self):
        src = Image.new("RGB", (1691, 930))
        cropped, src_size, resized_size, crop_offset = resize_and_center_crop(src)
        self.assertEqual(cropped.size, (TARGET_WIDTH, TARGET_HEIGHT))
        self.assertEqual(src_size, (1691, 930))
        # Resize preserves aspect; one of the resized dimensions equals the target.
        rw, rh = resized_size
        self.assertTrue(rw >= TARGET_WIDTH and rh >= TARGET_HEIGHT)
        cl, ct = crop_offset
        self.assertGreaterEqual(cl, 0)
        self.assertGreaterEqual(ct, 0)
        # Center crop produces 0 offset on the dimension that hit the target exactly.
        self.assertTrue(cl == 0 or ct == 0)

    def test_snap_num_frames_to_8k_plus_1(self):
        # The LTX-2 VAE requires (8k + 1)-shaped temporal dim. ``snap_num_frames``
        # rounds to the nearest such value (ties break to the ceil).
        for n in [1, 9, 17, 81, 161, 321, 801]:
            self.assertEqual(snap_num_frames(n), n)
        self.assertEqual(snap_num_frames(2), 1)
        self.assertEqual(snap_num_frames(10), 9)  # 10 is closer to 9 than 17
        self.assertEqual(snap_num_frames(80), 81)  # 80 is closer to 81 than 73
        self.assertEqual(snap_num_frames(100), 97)  # 100 is closer to 97 than 105
        # ``upper_bound`` caps the result (the snap falls back to the floor).
        self.assertLessEqual(snap_num_frames(100, upper_bound=100), 100)
        self.assertEqual(snap_num_frames(100, upper_bound=100), 97)


class SanaWMRegistrationTests(unittest.TestCase):
    """Verify the SANA-WM symbols are reachable through the public diffusers surface."""

    def test_top_level_symbols(self):
        import diffusers

        for name in ("SanaWMPipeline", "SanaWMTransformer3DModel", "SanaWMLTX2Refiner", "SanaWMPipelineOutput"):
            self.assertTrue(hasattr(diffusers, name), msg=f"{name!r} not exported from diffusers top-level")

    def test_pipeline_output_dataclass(self):
        import torch

        frames = np.zeros((3, 8, 8, 3), dtype=np.float32)
        c2w = np.broadcast_to(np.eye(4, dtype=np.float32), (3, 4, 4)).copy()
        latent = torch.zeros(1, 16, 1, 4, 4)
        out = SanaWMPipelineOutput(frames=frames, c2w=c2w, latent=latent)
        self.assertEqual(tuple(out.frames.shape), (3, 8, 8, 3))
        self.assertEqual(tuple(out.c2w.shape), (3, 4, 4))
        self.assertEqual(tuple(out.latent.shape), (1, 16, 1, 4, 4))

    def test_refiner_is_pipeline_with_ar_call_defaults(self):
        import inspect

        from diffusers import DiffusionPipeline

        # The refiner is a standalone DiffusionPipeline.
        self.assertTrue(issubclass(SanaWMLTX2Refiner, DiffusionPipeline))

        # Its denoising entry point is ``__call__`` with the canonical AR defaults.
        params = inspect.signature(SanaWMLTX2Refiner.__call__).parameters
        self.assertIn("block_size", params)
        self.assertIn("kv_max_frames", params)
        # AR mode is on by default.
        self.assertEqual(params["block_size"].default, 3)
        self.assertEqual(params["kv_max_frames"].default, 11)

    def test_pipeline_call_intrinsics_signature(self):
        import inspect

        params = inspect.signature(SanaWMPipeline.__call__).parameters
        self.assertIn("intrinsics", params)
        self.assertIn("c2w", params)
        self.assertIn("action", params)
        self.assertIn("use_refiner", params)
