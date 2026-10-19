# Copyright 2026 Black Forest Labs and The HuggingFace Team. All rights reserved.
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


class Flux3ImageProcessor(VaeImageProcessor):
    r"""
    Image processor for FLUX.3: preprocesses the reference images of image-to-image generation and postprocesses the
    generated images.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to downscale the image's (height, width) dimensions to multiples of `vae_scale_factor`. Can accept
            `height` and `width` arguments from [`image_processor.VaeImageProcessor.preprocess`] method.
        vae_scale_factor (`int`, *optional*, defaults to `16`):
            VAE (spatial) scale factor, including the 2x2 packing of the latents. If `do_resize` is `True`, the image
            is automatically resized to multiples of this factor.
        vae_latent_channels (`int`, *optional*, defaults to `32`):
            VAE latent channels.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image to [-1,1].
        do_binarize (`bool`, *optional*, defaults to `False`):
            Whether to binarize the image to 0/1.
        do_convert_rgb (`bool`, *optional*, defaults to be `True`):
            Whether to convert the images to RGB format.
        do_convert_grayscale (`bool`, *optional*, defaults to be `False`):
            Whether to convert the images to grayscale format.
    """

    # Copied from diffusers.pipelines.flux2.image_processor.Flux2ImageProcessor.__init__
    def __init__(
        self,
        do_resize: bool = True,
        vae_scale_factor: int = 16,
        vae_latent_channels: int = 32,
        do_normalize: bool = True,
        do_binarize: bool = False,
        do_convert_rgb: bool = True,
        do_convert_grayscale: bool = False,
    ):
        super().__init__(
            do_resize=do_resize,
            vae_scale_factor=vae_scale_factor,
            vae_latent_channels=vae_latent_channels,
            do_normalize=do_normalize,
            do_binarize=do_binarize,
            do_convert_rgb=do_convert_rgb,
            do_convert_grayscale=do_convert_grayscale,
        )

    @staticmethod
    # Copied from diffusers.pipelines.flux2.image_processor.Flux2ImageProcessor._resize_to_target_area
    def _resize_to_target_area(image: PIL.Image.Image, target_area: int = 1024 * 1024) -> PIL.Image.Image:
        image_width, image_height = image.size

        scale = math.sqrt(target_area / (image_width * image_height))
        width = int(image_width * scale)
        height = int(image_height * scale)

        return image.resize((width, height), PIL.Image.Resampling.LANCZOS)

    @staticmethod
    # Copied from diffusers.pipelines.flux2.image_processor.Flux2ImageProcessor._resize_if_exceeds_area with Flux2->Flux3
    def _resize_if_exceeds_area(image, target_area=1024 * 1024) -> PIL.Image.Image:
        image_width, image_height = image.size
        pixel_count = image_width * image_height
        if pixel_count <= target_area:
            return image
        return Flux3ImageProcessor._resize_to_target_area(image, target_area)

    # Copied from diffusers.pipelines.flux2.image_processor.Flux2ImageProcessor._resize_and_crop
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
