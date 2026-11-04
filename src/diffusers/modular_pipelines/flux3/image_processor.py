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

from ...image_processor import VaeImageProcessor


class Flux3ImageProcessor(VaeImageProcessor):
    r"""
    Image processor for FLUX.3: preprocesses the reference images of image-to-image generation and postprocesses the
    generated images. `resize_mode="fill"` keeps a reference whole: it is downscaled so that its canvas (its size
    rounded up to the token grid) fits under 4 megapixels, then reflect-padded to the canvas, and `preprocess` also
    returns the `reference_crop` box of the image inside the canvas, which `postprocess` takes to crop the generated
    image back to it. `resize_mode="crop"` downscales a reference to fit under 1 megapixel and center-crops it to the
    token grid. The sizing follows the released BFL code, `reference_canvas` / `prepare_reference` for one reference
    and `prepare_multi_reference` for several, in
    https://github.com/black-forest-labs/flux-3-image-klein/blob/main/src/flux3/pipeline.py.

    Args:
        vae_scale_factor (`int`, *optional*, defaults to `16`):
            VAE (spatial) scale factor, including the 2x2 packing of the latents: one token. Images are resized to
            multiples of this factor.
        vae_latent_channels (`int`, *optional*, defaults to `32`):
            VAE latent channels.
    """

    def __init__(self, vae_scale_factor: int = 16, vae_latent_channels: int = 32):
        # `preprocess` always passes the size to resize to, and the VAE takes RGB images in [-1, 1]
        super().__init__(
            do_resize=True,
            vae_scale_factor=vae_scale_factor,
            vae_latent_channels=vae_latent_channels,
            do_normalize=True,
            do_binarize=False,
            do_convert_rgb=True,
            do_convert_grayscale=False,
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

    def _get_canvas_height_width(
        self, image_height: int, image_width: int, max_side: int = 4096, min_side: int = 64
    ) -> tuple[int, int]:
        r"""
        The canvas of an image kept whole: its size rounded up to the token grid, clamped to [`min_side`, `max_side`]
        (the `canvas` of BFL's `reference_canvas`).
        """
        multiple_of = self.config.vae_scale_factor
        return tuple(
            min(max_side, max(min_side, -(-side // multiple_of) * multiple_of)) for side in (image_height, image_width)
        )

    def _resize_if_canvas_exceeds_area(
        self, image: PIL.Image.Image, target_area: int = 4 * 1024 * 1024, max_side: int = 4096
    ) -> PIL.Image.Image:
        r"""
        Downscale the image, keeping its aspect ratio, so that its canvas (its size rounded up to the token grid) fits
        under `target_area` and its sides under `max_side`, as BFL's `reference_canvas` does.

        The scale that fits the image itself under `target_area` can still leave the canvas over it once the sides are
        rounded up, so it is backed off by 0.5% until the canvas fits. The first pass already fits unless the image is
        within a token of the cap.
        """
        image_width, image_height = image.size
        scale = min(1.0, math.sqrt(target_area / (image_width * image_height)), max_side / max(image.size))
        while True:
            height, width = (max(1, round(side * scale)) for side in (image_height, image_width))
            canvas_height, canvas_width = self._get_canvas_height_width(height, width, max_side=max_side)
            if canvas_height * canvas_width <= target_area:
                break
            scale *= 0.995
        if (width, height) == image.size:
            return image
        return self.resize(image, height=height, width=width)

    def _get_fill_region(self, image: PIL.Image.Image, height: int, width: int) -> tuple[int, int, int, int]:
        r"""
        The (left, top, right, bottom) box the image occupies once padded to `height` x `width` with
        `resize_mode="fill"`: centered.
        """
        image_width, image_height = image.size
        if image_width > width or image_height > height:
            raise ValueError(
                f"resize_mode='fill' pads the image to the given size, got a {image_width} x {image_height} image for "
                f"a {width} x {height} canvas"
            )
        left, top = (width - image_width) // 2, (height - image_height) // 2
        return left, top, left + image_width, top + image_height

    def _resize_and_fill(self, image: PIL.Image.Image, width: int, height: int) -> PIL.Image.Image:
        r"""
        Center the image within the specified width and height, filling the rest with its reflection. The image is not
        resized: it must fit within the dimensions.

        Args:
            image (`PIL.Image.Image`):
                The image to fill.
            width (`int`):
                The width to pad the image to.
            height (`int`):
                The height to pad the image to.

        Returns:
            `PIL.Image.Image`:
                The filled image.
        """
        left, top, right, bottom = self._get_fill_region(image, height, width)
        if image.size == (width, height):
            return image

        padding = ((top, height - bottom), (left, width - right), (0, 0))
        # reflection excludes the edge pixel; an image smaller than one token is mirrored instead
        mode = "reflect" if min(image.size) > self.config.vae_scale_factor else "symmetric"
        return PIL.Image.fromarray(np.pad(np.asarray(image.convert("RGB")), padding, mode=mode))

    def preprocess(
        self,
        image: PIL.Image.Image | list[PIL.Image.Image],
        resize_mode: str = "fill",  # "fill" (one reference, kept whole), "crop" (several references)
    ) -> tuple[torch.Tensor, dict]:
        """
        Preprocess the reference image(s) as [`~image_processor.VaeImageProcessor.preprocess`] does. With
        `resize_mode="fill"`, the image is downscaled so that its canvas fits under 4 megapixels and reflect-padded to
        the canvas; with `resize_mode="crop"`, it is downscaled to fit under 1 megapixel and center-cropped to the
        token grid.

        Returns:
            `tuple[torch.Tensor, dict]`:
                The preprocessed image and the kwargs for [`~Flux3ImageProcessor.postprocess`]: `reference_crop`, the
                (left, top, right, bottom) box of the (first) image inside its canvas when it was padded
                (`resize_mode="fill"`), else `None`.
        """
        images = image if isinstance(image, list) else [image]
        if not all(isinstance(image, PIL.Image.Image) for image in images):
            raise ValueError(f"Only PIL image input is supported, got {[type(image) for image in images]}")
        height = width = None
        if resize_mode == "fill":
            images = [self._resize_if_canvas_exceeds_area(image) for image in images]
            height, width = self._get_canvas_height_width(images[0].height, images[0].width)
        elif resize_mode == "crop":
            images = [self._resize_if_exceeds_area(image, 1024 * 1024) for image in images]
        else:
            raise ValueError(f"resize_mode must be 'fill' or 'crop', got {resize_mode}")
        image = super().preprocess(images, height=height, width=width, resize_mode=resize_mode)
        reference_crop = self._get_fill_region(images[0], *image.shape[-2:]) if resize_mode == "fill" else None
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
