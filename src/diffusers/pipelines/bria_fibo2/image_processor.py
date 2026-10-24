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
import torch

from ...image_processor import VaeImageProcessor


class BriaFibo2ImageProcessor(VaeImageProcessor):
    r"""
    Image processor for Fibo 2.0: preprocesses the images to edit as Fibo 2.0 was trained on them, with the masked
    region of an image greyed out the way training did, gives the size Fibo 2.0 was trained at for an aspect ratio, and
    postprocesses the generated images.

    Args:
        vae_scale_factor (`int`, *optional*, defaults to `16`):
            VAE (spatial) scale factor, including the 2x2 packing of the latents. Sizes are multiples of it.
    """

    def __init__(self, vae_scale_factor: int = 16):
        super().__init__(vae_scale_factor=vae_scale_factor)

    def get_image_size(self, aspect_ratio: float, area: int = 1024 * 1024) -> tuple[int, int]:
        r"""
        The `(width, height)` Fibo 2.0 was trained at for an aspect ratio: of the sizes divisible by
        `vae_scale_factor`, at least 4 of it a side and within 10% of `area`, the one that crops the least from a
        picture with that aspect ratio, then the one closest to `area`.
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
        Scale the image to cover `width` x `height`, keeping its aspect ratio, then crop the center, as Fibo 2.0's
        training did. `resize` runs it for `resize_mode="crop"`.
        """
        image = image.convert("RGB")
        scale = max(height / image.height, width / image.width)
        resized_width, resized_height = math.ceil(image.width * scale), math.ceil(image.height * scale)
        image = image.resize((resized_width, resized_height), resample=PIL.Image.LANCZOS)
        left, top = (resized_width - width) // 2, (resized_height - height) // 2
        return image.crop((left, top, left + width, top + height))

    def preprocess(
        self,
        image: PIL.Image.Image | list[PIL.Image.Image],
        height: int | None = None,
        width: int | None = None,
        mask: PIL.Image.Image | None = None,
        area: int = 1024 * 1024,
        text_encoder_image_size: int = 768,
    ) -> tuple[list[torch.Tensor], list[PIL.Image.Image]]:
        r"""
        Preprocess the images to edit as Fibo 2.0 was trained on them.

        With `mask`, the region it marks in the single image is greyed out first (`grey_out`). A single image is then
        cropped to `width` x `height`, the size of the result, by default the size for its own aspect ratio; several
        images each to the size Fibo 2.0 was trained at for their own aspect ratio, about `area` (`get_image_size`).
        The VAE encodes the crops, and Qwen3-VL sees each one at most `text_encoder_image_size` pixels a side.

        Returns:
            `tuple[list[torch.Tensor], list[PIL.Image.Image]]`:
                The crops as the VAE takes them, one tensor each since their sizes can differ, and Qwen3-VL's copies.
        """
        images = image if isinstance(image, list) else [image]
        if mask is not None:
            images = [self.grey_out(images[0], mask)]
        if len(images) == 1:
            if height is None and width is None:
                width, height = self.get_image_size(images[0].width / images[0].height, area=area)
            sizes = [(width, height)]
        else:
            sizes = [self.get_image_size(img.width / img.height, area=area) for img in images]
        crops = [
            self.resize(img, height=size[1], width=size[0], resize_mode="crop") for img, size in zip(images, sizes)
        ]

        text_encoder_images = []
        for crop in crops:
            text_encoder_image = crop.copy()
            text_encoder_image.thumbnail((text_encoder_image_size, text_encoder_image_size))
            text_encoder_images.append(text_encoder_image)
        # Bound first: a zero-argument super() can't be called inside a comprehension before Python 3.12
        to_tensor = super().preprocess
        return [to_tensor(crop) for crop in crops], text_encoder_images

    @staticmethod
    def grey_out(image: PIL.Image.Image, mask: PIL.Image.Image) -> PIL.Image.Image:
        r"""
        The image with the region `mask` marks, white on black, greyed out as Fibo 2.0's training greyed it: flat (128,
        128, 128) grey through the mask made binary at 128, at the image's own size, before the image is resized.
        """
        grey = PIL.Image.new("RGB", image.size, (128, 128, 128))
        binary_mask = mask.convert("L").point(lambda value: 255 if value >= 128 else 0)
        return PIL.Image.composite(grey, image.convert("RGB"), binary_mask)
