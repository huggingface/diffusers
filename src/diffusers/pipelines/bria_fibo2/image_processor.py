# Copyright 2026 Bria AI and The HuggingFace Team. All rights reserved.
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

import PIL.Image

from ...image_processor import VaeImageProcessor


class BriaFibo2ImageProcessor(VaeImageProcessor):
    r"""
    Image processor for fibo-2: sizes the result and the images to edit as fibo-2 was trained, greys out the masked
    region of an image to edit the way training did, and postprocesses the generated images.

    Args:
        vae_scale_factor (`int`, *optional*, defaults to `16`):
            VAE (spatial) scale factor, including the 2x2 packing of the latents. Sizes are multiples of it.
    """

    def __init__(self, vae_scale_factor: int = 16):
        super().__init__(vae_scale_factor=vae_scale_factor)

    def get_image_size(self, aspect_ratio: float, area: int = 1024 * 1024) -> tuple[int, int]:
        r"""
        The `(width, height)` fibo-2 was trained at for an aspect ratio: of the sizes divisible by `vae_scale_factor`,
        at least 4 of it a side and within 10% of `area`, the one that crops the least from a picture with that aspect
        ratio, then the one closest to `area`.
        """
        multiple = self.config.vae_scale_factor
        min_side = 4 * multiple
        sizes = [
            (width, height)
            for width in range(min_side, 2 * area // min_side, multiple)
            for height in range(min_side, 2 * area // width, multiple)
            if 0.9 * area <= width * height <= 1.1 * area
        ]

        def crop(size):
            ratio = size[0] / size[1]
            return 1 - min(ratio / aspect_ratio, aspect_ratio / ratio)

        return min(sizes, key=lambda size: (round(crop(size), 9), abs(size[0] * size[1] - area), -size[0]))

    def _resize_and_crop(self, image: PIL.Image.Image, width: int, height: int) -> PIL.Image.Image:
        r"""
        Scale the image to cover `width` x `height`, keeping its aspect ratio, then crop the center, as fibo-2's
        training did. `resize` runs it for `resize_mode="crop"`.
        """
        image = image.convert("RGB")
        scale = max(height / image.height, width / image.width)
        resized_width, resized_height = math.ceil(image.width * scale), math.ceil(image.height * scale)
        image = image.resize((resized_width, resized_height), resample=PIL.Image.LANCZOS)
        left, top = (resized_width - width) // 2, (resized_height - height) // 2
        return image.crop((left, top, left + width, top + height))

    @staticmethod
    def grey_out(image: PIL.Image.Image, mask: PIL.Image.Image) -> PIL.Image.Image:
        r"""
        The image with the region `mask` marks, white on black, greyed out as fibo-2's training greyed it: flat (128,
        128, 128) grey through the mask made binary at 128, at the image's own size, before the image is resized.
        """
        grey = PIL.Image.new("RGB", image.size, (128, 128, 128))
        binary_mask = mask.convert("L").point(lambda value: 255 if value >= 128 else 0)
        return PIL.Image.composite(grey, image.convert("RGB"), binary_mask)
