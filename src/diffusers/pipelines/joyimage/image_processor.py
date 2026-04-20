# Copyright 2025 The JoyImage Team and The HuggingFace Team. All rights reserved.
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
from typing import Tuple

import torch
import torchvision.transforms.functional as TF
from PIL import Image

from ...configuration_utils import register_to_config
from ...image_processor import PipelineImageInput, VaeImageProcessor


# Mapping from precision string to torch dtype.
PRECISION_TO_TYPE = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


class BucketGroup:
    """Manages dynamic batch grouping buckets for image inference."""

    def __init__(
        self,
        bucket_configs: list[tuple[int, int, int, int, int]],
        prioritize_frame_matching: bool = True,
    ):
        """
        Initialize bucket group with predefined configurations.

        Args:
            bucket_configs: List of (batch_size, num_items, num_frames, height, width) tuples.
            prioritize_frame_matching: Unused, kept for API compatibility.
        """
        self.bucket_configs = [tuple(b) for b in bucket_configs]

    def find_best_bucket(self, media_shape: tuple[int, int, int, int]) -> tuple[int, int, int, int, int]:
        """
        Find the best matching bucket for given media dimensions.

        Selects the bucket whose aspect ratio (height/width) is closest to that of
        the input media. Only image inference (num_frames=1) is supported.

        Args:
            media_shape: (num_items, num_frames, height, width) of the input media.

        Returns:
            Best matching bucket as (batch_size, num_items, num_frames, height, width).

        Raises:
            ValueError: If num_frames != 1 or no valid bucket is found.
        """
        num_items, num_frames, height, width = media_shape
        target_aspect_ratio = height / width

        if num_frames != 1:
            raise ValueError(
                f"Only image inference (num_frames=1) is supported, got num_frames={num_frames}"
            )

        valid_buckets = [
            b for b in self.bucket_configs
            if b[1] == num_items and b[2] == 1
        ]
        if not valid_buckets:
            raise ValueError(f"No image buckets found for shape {media_shape}")

        return min(
            valid_buckets,
            key=lambda bucket: abs((bucket[3] / bucket[4]) - target_aspect_ratio),
        )


def _generate_hw_buckets(
    base_height: int = 256,
    base_width: int = 256,
    step_width: int = 16,
    step_height: int = 16,
    max_ratio: float = 4.0,
) -> list[tuple[int, int, int, int, int]]:
    """
    Generate (batch_size=1, num_items=1, num_frames=1, height, width) bucket tuples
    covering a range of aspect ratios while keeping total pixels close to
    base_height * base_width.

    Args:
        base_height: Reference height in pixels.
        base_width: Reference width in pixels.
        step_width: Width increment per step.
        step_height: Height decrement per step.
        max_ratio: Maximum allowed aspect ratio (long side / short side).

    Returns:
        List of bucket tuples (1, 1, 1, height, width).
    """
    buckets = []
    target_pixels = base_height * base_width

    height = target_pixels // step_width
    width = step_width

    while height >= step_height:
        if max(height, width) / min(height, width) <= max_ratio:
            buckets.append((1, 1, 1, height, width))
        if height * (width + step_width) <= target_pixels:
            width += step_width
        else:
            height -= step_height

    return buckets


def generate_video_image_bucket(
    basesize: int = 256,
    min_temporal: int = 65,
    max_temporal: int = 129,
    bs_img: int = 8,
    bs_vid: int = 1,
    bs_mimg: int = 4,
    min_items: int = 1,
    max_items: int = 1,
) -> list[list[int]]:
    """
    Generate bucket configurations for image inference.

    Each bucket is represented as [batch_size, num_items, num_frames, height, width].
    Spatial dimensions are scaled by (basesize // 256) when basesize > 256.

    Args:
        basesize: Base spatial resolution. Must be one of {256, 512, 768, 1024}.
        min_temporal: Unused; kept for API compatibility.
        max_temporal: Unused; kept for API compatibility.
        bs_img: Batch size for single-image buckets.
        bs_vid: Unused; kept for API compatibility.
        bs_mimg: Batch size for multi-image buckets.
        min_items: Minimum number of items in multi-image buckets.
        max_items: Maximum number of items in multi-image buckets.

    Returns:
        List of bucket configs as [batch_size, num_items, num_frames, height, width].

    Raises:
        AssertionError: If basesize is not in {256, 512, 768, 1024}.
    """
    assert basesize in [256, 512, 768, 1024], (
        f"[generate_video_image_bucket] unsupported basesize {basesize}"
    )
    bucket_list = []
    base_bucket_list = _generate_hw_buckets()

    # Single-image buckets.
    for _bucket in base_bucket_list:
        bucket = list(_bucket)
        bucket[0] = bs_img
        bucket_list.append(bucket)

    # Multi-image buckets.
    for num_items in range(min_items, max_items + 1):
        for _bucket in base_bucket_list:
            bucket = list(_bucket)
            bucket[0] = bs_mimg
            bucket[1] = num_items
            bucket_list.append(bucket)

    # Scale spatial dimensions when basesize exceeds 256.
    if basesize > 256:
        ratio = basesize // 256

        def _scale(bucket: list[int], r: int) -> list[int]:
            bucket[-2] *= r
            bucket[-1] *= r
            return bucket

        bucket_list = [_scale(bucket, ratio) for bucket in bucket_list]

    return bucket_list


class JoyImageEditImageProcessor(VaeImageProcessor):
    """
    Image processor for the JoyImage Edit pipeline.

    Handles bucket-based resolution selection and resize-center-crop preprocessing.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image.
        vae_scale_factor (`int`, *optional*, defaults to `8`):
            VAE spatial scale factor.
        basesize (`int`, *optional*, defaults to `1024`):
            Base resolution for bucket generation.
        resample (`str`, *optional*, defaults to `bilinear`):
            Resampling filter for resizing.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image to [-1,1].
        do_binarize (`bool`, *optional*, defaults to `False`):
            Whether to binarize the image to 0/1.
        do_convert_rgb (`bool`, *optional*, defaults to `False`):
            Whether to convert the images to RGB format.
        do_convert_grayscale (`bool`, *optional*, defaults to `False`):
            Whether to convert the images to grayscale format.
    """

    @register_to_config
    def __init__(
        self,
        do_resize: bool = True,
        vae_scale_factor: int = 8,
        basesize: int = 1024,
        resample: str = "bilinear",
        do_normalize: bool = True,
        do_binarize: bool = False,
        do_convert_rgb: bool = False,
        do_convert_grayscale: bool = False,
    ):
        super().__init__()

    def get_default_height_width(
        self,
        image: Image.Image,
        height: int | None = None,
        width: int | None = None,
    ) -> Tuple[int, int]:
        """
        Compute the target (height, width) from the nearest bucket.

        If height and width are both provided, they are used as the source
        dimensions for bucket matching instead of the image's actual size.

        Args:
            image: Input PIL image (or first image if a list).
            height: Optional override height.
            width: Optional override width.

        Returns:
            Tuple of (target_height, target_width) from the best matching bucket.
        """
        if height is not None and width is not None:
            image_size = (width, height)
        else:
            image_size = image[0].size if isinstance(image, list) else image.size

        bucket_config = generate_video_image_bucket(
            basesize=self.config.basesize,
            min_temporal=56,
            max_temporal=56,
            bs_img=4,
            bs_vid=4,
            bs_mimg=8,
            min_items=2,
            max_items=2,
        )
        bucket_group = BucketGroup(bucket_config)
        src_w, src_h = image_size
        bucket = bucket_group.find_best_bucket((1, 1, src_h, src_w))
        return bucket[-2], bucket[-1]

    def resize_center_crop(
        self,
        img: Image.Image,
        target_size: Tuple[int, int],
    ) -> Image.Image:
        """
        Scale image to cover target_size, then center-crop.

        Args:
            img: Input PIL image.
            target_size: (height, width) to crop to.

        Returns:
            Resized and center-cropped PIL image.
        """
        w, h = img.size
        bh, bw = target_size
        scale = max(bh / h, bw / w)
        resize_h = math.ceil(h * scale)
        resize_w = math.ceil(w * scale)
        img = TF.resize(img, (resize_h, resize_w), interpolation=TF.InterpolationMode.BILINEAR, antialias=True)
        img = TF.center_crop(img, target_size)
        return img
