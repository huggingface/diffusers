# Copyright 2026 The HuggingFace Team. All rights reserved.
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

import tempfile
import unittest

import numpy as np
import pytest
import torch

from diffusers.image_processor import VaeImageProcessor
from diffusers.utils import is_invisible_watermark_available, is_opencv_available, load_image

from ...testing_utils import require_torch


# Image from the #15025 repro: float encode fails decode; uint8 encode recovers all 48 bits.
_WATERMARK_REPRO_URL = (
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/"
    "df6c7e6873499cfe63686cb4b6edb05d6f8a5755/diffusers/sdxl_reference_adain_canny_cat.png"
)


def _bits_matched_via_png(pil_image, watermark_bits):
    import cv2
    from imwatermark import WatermarkDecoder

    with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
        pil_image.save(tmp.name)
        bgr = cv2.imread(tmp.name)
        bits = WatermarkDecoder("bits", len(watermark_bits)).decode(bgr, "dwtDct")
    return sum(int(b) == w for b, w in zip(bits, watermark_bits))


@require_torch
@pytest.mark.skipif(not is_invisible_watermark_available(), reason="requires invisible-watermark")
@pytest.mark.skipif(not is_opencv_available(), reason="requires opencv")
class StableDiffusionXLWatermarkTests(unittest.TestCase):
    def test_apply_watermark_decodes_after_uint8_png_roundtrip(self):
        """Regression for #15025: encode on uint8 so Stability's cv2.imread detector recovers the bits."""
        from diffusers.pipelines.stable_diffusion_xl.watermark import WATERMARK_BITS, StableDiffusionXLWatermarker

        rgb = np.asarray(load_image(_WATERMARK_REPRO_URL).convert("RGB"))
        images = torch.from_numpy(rgb.astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1)[None]
        processor = VaeImageProcessor(vae_scale_factor=8)

        watermarked = StableDiffusionXLWatermarker().apply_watermark(images.clone())
        pil = processor.postprocess(watermarked, output_type="pil")[0]
        matched = _bits_matched_via_png(pil, WATERMARK_BITS)

        # detect.py treats <=33/48 as "No watermark detected"; require a clear full match.
        self.assertGreaterEqual(matched, 41)
