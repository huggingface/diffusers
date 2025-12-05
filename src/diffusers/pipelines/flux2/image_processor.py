# Copyright 2025 The Black Forest Labs Team and The HuggingFace Team. All rights reserved.
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
from typing import List

import PIL.Image

from ...configuration_utils import register_to_config
from ...image_processor import VaeImageProcessor


class Flux2ImageProcessor(VaeImageProcessor):
    r"""
    Image processor to preprocess the reference (character) image for the Flux2 model.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to downscale the image's (height, width) dimensions to multiples of `vae_scale_factor`. Can accept
            `height` and `width` arguments from [`image_processor.VaeImageProcessor.preprocess`] method.
        vae_scale_factor (`int`, *optional*, defaults to `16`):
            VAE (spatial) scale factor. If `do_resize` is `True`, the image is automatically resized to multiples of
            this factor.
        vae_latent_channels (`int`, *optional*, defaults to `32`):
            VAE latent channels.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image to [-1,1].
        do_convert_rgb (`bool`, *optional*, defaults to be `True`):
            Whether to convert the images to RGB format.
    """

    @register_to_config
    def __init__(
        self,
        do_resize: bool = True,
        vae_scale_factor: int = 16,
        vae_latent_channels: int = 32,
        do_normalize: bool = True,
        do_convert_rgb: bool = True,
    ):
        super().__init__(
            do_resize=do_resize,
            vae_scale_factor=vae_scale_factor,
            vae_latent_channels=vae_latent_channels,
            do_normalize=do_normalize,
            do_convert_rgb=do_convert_rgb,
        )

    @staticmethod
    def check_image_input(
        image: PIL.Image.Image, max_aspect_ratio: int = 8, min_side_length: int = 64, max_area: int = 1024 * 1024
    ) -> PIL.Image.Image:
        """
        Check if image meets minimum size and aspect ratio requirements.

        Args:
            image: PIL Image to validate
            max_aspect_ratio: Maximum allowed aspect ratio (width/height or height/width)
            min_side_length: Minimum pixels required for width and height
            max_area: Maximum allowed area in pixels²

        Returns:
            The input image if valid

        Raises:
            ValueError: If image is too small or aspect ratio is too extreme
        """
        if not isinstance(image, PIL.Image.Image):
            raise ValueError(f"Image must be a PIL.Image.Image, got {type(image)}")

        width, height = image.size

        # Check minimum dimensions
        if width < min_side_length or height < min_side_length:
            raise ValueError(
                f"Image too small: {width}×{height}. Both dimensions must be at least {min_side_length}px"
            )

        # Check aspect ratio
        aspect_ratio = max(width / height, height / width)
        if aspect_ratio > max_aspect_ratio:
            raise ValueError(
                f"Aspect ratio too extreme: {width}×{height} (ratio: {aspect_ratio:.1f}:1). "
                f"Maximum allowed ratio is {max_aspect_ratio}:1"
            )

        return image

    @staticmethod
    def _resize_to_target_area(image: PIL.Image.Image, target_area: int = 1024 * 1024) -> PIL.Image.Image:
        image_width, image_height = image.size

        scale = math.sqrt(target_area / (image_width * image_height))
        width = int(image_width * scale)
        height = int(image_height * scale)

        return image.resize((width, height), PIL.Image.Resampling.LANCZOS)

    @staticmethod
    def _resize_if_exceeds_area(image, target_area=1024 * 1024) -> PIL.Image.Image:
        image_width, image_height = image.size
        pixel_count = image_width * image_height
        if pixel_count <= target_area:
            return image
        return Flux2ImageProcessor._resize_to_target_area(image, target_area)

    def _resize_and_crop(
        self,
        image: PIL.Image.Image,
        width: int,
        height: int,
    ) -> PIL.Image.Image:
        r"""
        center crop the image to the specified width and height.

        Args:
            image (`PIL.Image.Image`):
                The image to resize and crop.
            width (`int`):
                The width to resize the image to.
            height (`int`):
                The height to resize the image to.

        Returns:
            `PIL.Image.Image`:
                The resized and cropped image.
        """
        image_width, image_height = image.size

        left = (image_width - width) // 2
        top = (image_height - height) // 2
        right = left + width
        bottom = top + height

        return image.crop((left, top, right, bottom))

    # Taken from
    # https://github.com/black-forest-labs/flux2/blob/5a5d316b1b42f6b59a8c9194b77c8256be848432/src/flux2/sampling.py#L310C1-L339C19
    @staticmethod
    def concatenate_images(images: List[PIL.Image.Image]) -> PIL.Image.Image:
        """
        Concatenate a list of PIL images horizontally with center alignment and white background.
        """

        # If only one image, return a copy of it
        if len(images) == 1:
            return images[0].copy()

        # Convert all images to RGB if not already
        images = [img.convert("RGB") if img.mode != "RGB" else img for img in images]

        # Calculate dimensions for horizontal concatenation
        total_width = sum(img.width for img in images)
        max_height = max(img.height for img in images)

        # Create new image with white background
        background_color = (255, 255, 255)
        new_img = PIL.Image.new("RGB", (total_width, max_height), background_color)

        # Paste images with center alignment
        x_offset = 0
        for img in images:
            y_offset = (max_height - img.height) // 2
            new_img.paste(img, (x_offset, y_offset))
            x_offset += img.width

        return new_img
