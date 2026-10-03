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

from __future__ import annotations

import numpy as np
import PIL.Image
import torch

from ...configuration_utils import register_to_config
from ...image_processor import VaeImageProcessor
from .cam_utils import TARGET_HEIGHT, TARGET_WIDTH, resize_and_center_crop, transform_intrinsics_for_crop


class SanaWMImageProcessor(VaeImageProcessor):
    r"""
    Image processor for SANA-WM's first-frame input.

    SANA-WM was trained at a fixed 704×1280 resolution with an aspect-preserving *resize + center-crop* transform. The
    pipeline also needs to rescale the per-frame camera intrinsics ``[fx, fy, cx, cy]`` to match the crop —
    ``preprocess_with_intrinsics`` does both in one call so the two stay in lockstep.

    Args:
        vae_scale_factor (`int`, defaults to `32`):
            LTX-2 VAE spatial stride.
        do_normalize (`bool`, defaults to `True`):
            Standard `VaeImageProcessor` [-1, 1] normalization.
    """

    @register_to_config
    def __init__(self, vae_scale_factor: int = 32, do_normalize: bool = True) -> None:
        super().__init__(vae_scale_factor=vae_scale_factor, do_normalize=do_normalize)

    def preprocess_with_intrinsics(
        self,
        image: PIL.Image.Image,
        intrinsics: np.ndarray,
        height: int = TARGET_HEIGHT,
        width: int = TARGET_WIDTH,
    ) -> tuple[torch.Tensor, np.ndarray]:
        """Resize + center-crop the image and rescale ``intrinsics`` to match.

        Args:
            image: RGB PIL image (any size).
            intrinsics: ``(F, 4)`` ``[fx, fy, cx, cy]`` per frame in original-image pixel coordinates.
            height / width: Target crop size (defaults to SANA-WM's training resolution).

        Returns:
            ``(pixel_values, intrinsics_cropped)``:
              * ``pixel_values`` — ``(1, 3, H, W)`` tensor in `[-1, 1]` (VaeImageProcessor convention).
              * ``intrinsics_cropped`` — ``(F, 4)`` array rescaled for the resize + crop.
        """
        cropped, src_size, resized_size, crop_offset = resize_and_center_crop(image, height, width)
        pixel_values = self.preprocess(cropped, height=height, width=width)
        intr = transform_intrinsics_for_crop(intrinsics, src_size, resized_size, crop_offset)
        return pixel_values, intr
