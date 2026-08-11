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

import unittest
from types import SimpleNamespace

import numpy as np
import pytest

from diffusers.pipelines.ltx2.utils import (
    DEFAULT_IMAGE_CRF,
    LTX2_5_IMAGE_CRF,
    apply_image_conditioning_crf,
    resolve_default_image_crf,
)
from diffusers.utils import is_av_available

from ...testing_utils import require_torch


class LTX2ImageCRFTests(unittest.TestCase):
    def test_resolve_default_image_crf_by_text_encoder_model_type(self):
        self.assertEqual(
            resolve_default_image_crf(SimpleNamespace(config=SimpleNamespace(model_type="gemma3"))), DEFAULT_IMAGE_CRF
        )
        self.assertEqual(
            resolve_default_image_crf(SimpleNamespace(config=SimpleNamespace(model_type="gemma4_unified"))),
            LTX2_5_IMAGE_CRF,
        )
        self.assertEqual(
            resolve_default_image_crf(SimpleNamespace(config=SimpleNamespace(model_type="gemma4"))),
            LTX2_5_IMAGE_CRF,
        )
        self.assertEqual(resolve_default_image_crf(None), DEFAULT_IMAGE_CRF)

    def test_crf_zero_is_identity(self):
        image = np.arange(16 * 16 * 3, dtype=np.uint8).reshape(16, 16, 3)
        out = apply_image_conditioning_crf(image, crf=0)
        np.testing.assert_array_equal(out, image)

    def test_unresolved_crf_raises(self):
        image = np.zeros((16, 16, 3), dtype=np.uint8)
        with self.assertRaises(ValueError):
            apply_image_conditioning_crf(image, crf=None)

    @pytest.mark.skipif(not is_av_available(), reason="requires PyAV")
    @require_torch
    def test_nonzero_crf_roundtrips_and_changes_pixels(self):
        # Non-trivial content so libx264 at CRF 18 actually quantizes something.
        rng = np.random.default_rng(0)
        image = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
        out = apply_image_conditioning_crf(image, crf=18)
        self.assertEqual(out.dtype, np.uint8)
        self.assertEqual(out.shape[:2], (64, 64))
        self.assertFalse(np.array_equal(out, image))
