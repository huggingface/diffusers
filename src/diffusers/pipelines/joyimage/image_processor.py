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
from PIL import Image

from ...configuration_utils import register_to_config
from ...image_processor import VaeImageProcessor


# fmt: off
BUCKETS = {
    1024: [
        (512, 1792), (512, 1856), (512, 1920), (512, 1984), (512, 2048),
        (576, 1600), (576, 1664), (576, 1728), (576, 1792),
        (640, 1472), (640, 1536), (640, 1600),
        (704, 1344), (704, 1408), (704, 1472),
        (768, 1216), (768, 1280), (768, 1344),
        (832, 1152), (832, 1216),
        (896, 1088), (896, 1152),
        (960, 1024), (960, 1088),
        (1024, 960), (1024, 1024),
        (1088, 896), (1088, 960),
        (1152, 832), (1152, 896),
        (1216, 768), (1216, 832),
        (1280, 768),
        (1344, 704), (1344, 768),
        (1408, 704),
        (1472, 640), (1472, 704),
        (1536, 640),
        (1600, 576), (1600, 640),
        (1664, 576),
        (1728, 576),
        (1792, 512), (1792, 576),
        (1856, 512),
        (1920, 512),
        (1984, 512),
        (2048, 512),
    ],
}
# fmt: on


def find_best_bucket(height: int, width: int, basesize: int) -> Tuple[int, int]:
    """Return the (h, w) bucket whose aspect ratio is closest to height/width."""
    target_ratio = height / width
    return min(
        BUCKETS[basesize],
        key=lambda hw: abs(hw[0] / hw[1] - target_ratio),
    )


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
        if height is not None and width is not None:
            src_w, src_h = width, height
        elif isinstance(image, list):
            src_w, src_h = image[0].size
        else:
            src_w, src_h = image.size

        return find_best_bucket(src_h, src_w, self.config.basesize)

    def resize_center_crop(
        self,
        img,
        target_size: Tuple[int, int],
    ):
        """
        Scale image to cover target_size, then center-crop.

        Args:
            img: Input PIL image or list of PIL images.
            target_size: (height, width) to crop to.

        Returns:
            Resized and center-cropped PIL image(s), matching the input type.
        """
        if isinstance(img, list):
            return [self.resize_center_crop(i, target_size) for i in img]

        w, h = img.size
        bh, bw = target_size
        scale = max(bh / h, bw / w)
        resize_h = math.ceil(h * scale)
        resize_w = math.ceil(w * scale)
        img = img.resize((resize_w, resize_h), Image.BILINEAR)
        left = (resize_w - bw) // 2
        top = (resize_h - bh) // 2
        img = img.crop((left, top, left + bw, top + bh))
        return img
