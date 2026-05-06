# Copyright 2025 Ant Group and The HuggingFace Team. All rights reserved.
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

"""Image preprocessing utilities for UniLLaDA pipeline."""

from __future__ import annotations

import math

import PIL.Image


def generate_crop_size_list(
    max_num_patches: int,
    patch_size: int = 32,
    min_size: int = 256,
) -> list[tuple[int, int]]:
    """
    Generate a list of valid (height, width) crop sizes.

    Args:
        max_num_patches (`int`):
            Maximum number of patches (e.g., (512 // 32) ** 2 = 256).
        patch_size (`int`, defaults to 32):
            Patch size in pixels.
        min_size (`int`, defaults to 256):
            Minimum image dimension.

    Returns:
        `list[tuple[int, int]]`: Sorted list of (height, width) pairs.
    """
    crop_sizes = []
    for h_patches in range(1, max_num_patches + 1):
        for w_patches in range(1, max_num_patches + 1):
            if h_patches * w_patches <= max_num_patches:
                h = h_patches * patch_size
                w = w_patches * patch_size
                if h >= min_size and w >= min_size:
                    crop_sizes.append((h, w))
    crop_sizes.sort(key=lambda x: x[0] * x[1])
    return crop_sizes


def var_center_crop(
    image: PIL.Image.Image,
    crop_size_list: list[tuple[int, int]],
) -> PIL.Image.Image:
    """
    Center-crop an image to the best matching size from `crop_size_list`,
    preserving aspect ratio as much as possible.

    Args:
        image (`PIL.Image.Image`):
            Input image.
        crop_size_list (`list[tuple[int, int]]`):
            List of valid (height, width) crop sizes.

    Returns:
        `PIL.Image.Image`: Cropped and resized image.
    """
    w, h = image.size
    aspect_ratio = w / h

    # Find best matching crop size
    best_size = min(
        crop_size_list,
        key=lambda s: abs(s[1] / s[0] - aspect_ratio),
    )

    target_h, target_w = best_size

    # Resize to cover target size while maintaining aspect ratio
    scale = max(target_w / w, target_h / h)
    new_w = int(math.ceil(w * scale))
    new_h = int(math.ceil(h * scale))
    image = image.resize((new_w, new_h), PIL.Image.LANCZOS)

    # Center crop
    left = (new_w - target_w) // 2
    top = (new_h - target_h) // 2
    image = image.crop((left, top, left + target_w, top + target_h))

    return image
