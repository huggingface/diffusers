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

import numpy as np
import PIL.Image
import torch

from ...image_processor import PIL_INTERPOLATION, PipelineImageInput, VaeImageProcessor


class Flux3ImageProcessor(VaeImageProcessor):
    r"""
    Image processor for FLUX.3: preprocesses the reference images of image-to-image generation and postprocesses the
    generated images. `resize_mode="fill"` pads the image to the given size with its reflection, and `preprocess` also
    returns the `reference_crop` box of the image inside the padded canvas, which `postprocess` takes to crop the
    generated image back to it.

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

    def _get_fill_region(self, image: PIL.Image.Image, height: int, width: int) -> tuple[int, int, int, int]:
        r"""
        The (left, top, right, bottom) box the image occupies once padded to `height` x `width` with
        `resize_mode="fill"`: downscaled to fit within the dimensions, maintaining the aspect ratio, and centered.
        """
        image_width, image_height = image.size
        if image_width > width or image_height > height:
            scale = min(width / image_width, height / image_height)
            image_width, image_height = max(1, round(image_width * scale)), max(1, round(image_height * scale))
        left, top = (width - image_width) // 2, (height - image_height) // 2
        return left, top, left + image_width, top + image_height

    def _resize_and_fill(self, image: PIL.Image.Image, width: int, height: int) -> PIL.Image.Image:
        r"""
        Resize the image to fit within the specified width and height, maintaining the aspect ratio, and then center
        the image within the dimensions, filling the rest with its reflection.

        Args:
            image (`PIL.Image.Image`):
                The image to resize and fill.
            width (`int`):
                The width to resize the image to.
            height (`int`):
                The height to resize the image to.

        Returns:
            `PIL.Image.Image`:
                The resized and filled image.
        """
        left, top, right, bottom = self._get_fill_region(image, height, width)
        if (right - left, bottom - top) != image.size:
            image = image.resize((right - left, bottom - top), resample=PIL_INTERPOLATION[self.config.resample])
        if image.size == (width, height):
            return image

        padding = ((top, height - bottom), (left, width - right), (0, 0))
        # reflection excludes the edge pixel; an image smaller than one token is mirrored instead
        mode = "reflect" if min(image.size) > self.config.vae_scale_factor else "symmetric"
        return PIL.Image.fromarray(np.pad(np.asarray(image.convert("RGB")), padding, mode=mode))

    def preprocess(
        self,
        image: PipelineImageInput,
        height: int | None = None,
        width: int | None = None,
        resize_mode: str = "default",  # "default", "fill", "crop"
        crops_coords: tuple[int, int, int, int] | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """
        Preprocess the image input as [`~image_processor.VaeImageProcessor.preprocess`] does.

        Returns:
            `tuple[torch.Tensor, dict]`:
                The preprocessed image and the kwargs for [`~Flux3ImageProcessor.postprocess`]: `reference_crop`, the
                (left, top, right, bottom) box of the (first) image inside its canvas when it was padded
                (`resize_mode="fill"`), else `None`.
        """
        reference_crop = None
        if resize_mode == "fill":
            first = image[0] if isinstance(image, list) else image
            if not isinstance(first, PIL.Image.Image):
                raise ValueError(f"Only PIL image input is supported for resize_mode {resize_mode}")
            height, width = self.get_default_height_width(first, height, width)
            reference_crop = self._get_fill_region(first, height, width)
        image = super().preprocess(
            image, height=height, width=width, resize_mode=resize_mode, crops_coords=crops_coords
        )
        return image, {"reference_crop": reference_crop}

    def postprocess(
        self,
        image: torch.Tensor,
        output_type: str = "pil",
        do_denormalize: list[bool] | None = None,
        reference_crop: tuple[int, int, int, int] | list[tuple[int, int, int, int] | None] | None = None,
    ) -> PIL.Image.Image | np.ndarray | torch.Tensor:
        """
        Postprocess the image output as [`~image_processor.VaeImageProcessor.postprocess`] does, cropped to the
        `reference_crop` box returned by [`~Flux3ImageProcessor.preprocess`] when given. For several references, pass
        the list of their boxes: at most one may be a box (the padded reference), the others `None`.
        """
        if isinstance(reference_crop, list):
            boxes = [box for box in reference_crop if box is not None]
            if len(boxes) > 1:
                raise ValueError(
                    f"The generated image can be cropped back to one reference only, got {len(boxes)} boxes in "
                    f"`reference_crop`: {reference_crop}"
                )
            reference_crop = boxes[0] if boxes else None
        if reference_crop is not None:
            left, top, right, bottom = reference_crop
            image = image[..., top:bottom, left:right]
        return super().postprocess(image, output_type=output_type, do_denormalize=do_denormalize)
