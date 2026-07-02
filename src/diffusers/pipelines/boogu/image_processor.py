# Copyright (C) 2026 Boogu Team.
# This repository is a fork by Boogu Team; modifications have been made.
#
# Original work: Copyright 2024 The HuggingFace Team. All rights reserved.
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

from typing import Optional, Tuple, Union

import numpy as np
import PIL.Image
import torch

from ...configuration_utils import register_to_config
from ...image_processor import (
    PipelineImageInput,
    VaeImageProcessor,
)


class BooguImageProcessor(VaeImageProcessor):
    """
    Boogu-Image image processor, with resize/crop behavior adapted from PixArt's
    image processor implementation.

    This class keeps a Diffusers-compatible preprocessing contract while adding
    Boogu-Image-specific pixel and side-length constraints.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to downscale the image's (height, width) dimensions to multiples of `vae_scale_factor`. Can accept
            `height` and `width` arguments from [`image_processor.VaeImageProcessor.preprocess`] method.
        vae_scale_factor (`int`, *optional*, defaults to `16`):
            VAE scale factor. If `do_resize` is `True`, the image is automatically resized to multiples of this factor.
        resample (`str`, *optional*, defaults to `lanczos`):
            Resampling filter to use when resizing the image.
        max_pixels (`int`, *optional*):
            Maximum number of pixels; the image is downscaled to fit when set.
        max_side_length (`int`, *optional*):
            Maximum side length; the image is downscaled to fit when set.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image to [-1,1].
        do_binarize (`bool`, *optional*, defaults to `False`):
            Whether to binarize the image to 0/1.
        do_convert_grayscale (`bool`, *optional*, defaults to be `False`):
            Whether to convert the images to grayscale format.
    """

    @register_to_config
    def __init__(
        self,
        do_resize: bool = True,
        vae_scale_factor: int = 16,
        resample: str = "lanczos",
        max_pixels: Optional[int] = None,
        max_side_length: Optional[int] = None,
        do_normalize: bool = True,
        do_binarize: bool = False,
        do_convert_grayscale: bool = False,
    ):
        super().__init__(
            do_resize=do_resize,
            vae_scale_factor=vae_scale_factor,
            resample=resample,
            do_normalize=do_normalize,
            do_binarize=do_binarize,
            do_convert_grayscale=do_convert_grayscale,
        )

        self.max_pixels = max_pixels
        self.max_side_length = max_side_length

    def get_new_height_width(
        self,
        image: Union[PIL.Image.Image, np.ndarray, torch.Tensor],
        height: Optional[int] = None,
        width: Optional[int] = None,
        max_pixels: Optional[int] = None,
        max_side_length: Optional[int] = None,
    ) -> Tuple[int, int]:
        r"""
        Returns target `(height, width)` after optional downscaling and
        rounding to `vae_scale_factor` multiples.

        Args:
            image (`Union[PIL.Image.Image, np.ndarray, torch.Tensor]`):
                The image input, which can be a PIL image, NumPy array, or PyTorch tensor. If it is a NumPy array, it
                should have shape `[batch, height, width]` or `[batch, height, width, channels]`. If it is a PyTorch
                tensor, it should have shape `[batch, channels, height, width]`.
            height (`Optional[int]`, *optional*, defaults to `None`):
                The height of the preprocessed image. If `None`, the height of the `image` input will be used.
            width (`Optional[int]`, *optional*, defaults to `None`):
                The width of the preprocessed image. If `None`, the width of the `image` input will be used.

        Returns:
            `Tuple[int, int]`:
                A tuple containing the height and width, both resized to the nearest integer multiple of
                `vae_scale_factor`.
        """

        if height is None:
            if isinstance(image, PIL.Image.Image):
                height = image.height
            elif isinstance(image, torch.Tensor):
                height = image.shape[2]
            else:
                height = image.shape[1]

        if width is None:
            if isinstance(image, PIL.Image.Image):
                width = image.width
            elif isinstance(image, torch.Tensor):
                width = image.shape[3]
            else:
                width = image.shape[2]

        if max_side_length is None:
            max_side_length = self.max_side_length

        if max_pixels is None:
            max_pixels = self.max_pixels

        # Clamp ratio to <=1 to avoid upscaling input images in preprocessing.
        ratio = 1.0
        if max_side_length is not None:
            longest_side = height if height > width else width
            ratio = min(ratio, max_side_length / longest_side)
        if max_pixels is not None:
            ratio = min(ratio, (max_pixels / (height * width)) ** 0.5)

        new_height, new_width = (
            int(height * ratio) // self.config.vae_scale_factor * self.config.vae_scale_factor,
            int(width * ratio) // self.config.vae_scale_factor * self.config.vae_scale_factor,
        )
        return new_height, new_width

    def preprocess(
        self,
        image: PipelineImageInput,
        height: Optional[int] = None,
        width: Optional[int] = None,
        max_pixels: Optional[int] = None,
        max_side_length: Optional[int] = None,
        resize_mode: str = "default",  # "default", "fill", "crop"
        crops_coords: Optional[Tuple[int, int, int, int]] = None,
    ) -> torch.Tensor:
        """
        Preprocess the image input.

        Identical to [`VaeImageProcessor.preprocess`], except the target size is derived from Boogu's
        `max_pixels` / `max_side_length` downscaling (via [`get_new_height_width`]) instead of a fixed
        default, before delegating the format handling, resize, and normalization to the parent.

        Args:
            image (`PipelineImageInput`):
                The image input, accepted formats are PIL images, NumPy arrays, PyTorch tensors; also a list thereof.
            height (`int`, *optional*):
                Target height. If `None`, derived from the image and the pixel / side-length constraints.
            width (`int`, *optional*):
                Target width. If `None`, derived from the image and the pixel / side-length constraints.
            max_pixels (`int`, *optional*):
                Maximum number of pixels; the image is downscaled to fit. Defaults to `self.max_pixels`.
            max_side_length (`int`, *optional*):
                Maximum side length; the image is downscaled to fit. Defaults to `self.max_side_length`.
            resize_mode (`str`, *optional*, defaults to `default`):
                One of `default`, `fill`, or `crop`; see [`VaeImageProcessor.preprocess`].
            crops_coords (`Tuple[int, int, int, int]`, *optional*):
                The crop coordinates. If `None`, the image is not cropped.

        Returns:
            `torch.Tensor`:
                The preprocessed image tensor with shape `[B, C, H, W]`.
        """
        if self.config.do_resize:
            representative = image[0] if isinstance(image, list) else image
            height, width = self.get_new_height_width(representative, height, width, max_pixels, max_side_length)
        return super().preprocess(
            image, height=height, width=width, resize_mode=resize_mode, crops_coords=crops_coords
        )
