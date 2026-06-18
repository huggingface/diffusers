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

import warnings
from typing import Optional, Tuple, Union

import numpy as np
import PIL.Image
import torch

from ...configuration_utils import register_to_config
from ...image_processor import (
    PipelineImageInput,
    VaeImageProcessor,
    is_valid_image_imagelist,
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
        vae_scale_factor (`int`, *optional*, defaults to `8`):
            VAE scale factor. If `do_resize` is `True`, the image is automatically resized to multiples of this factor.
        resample (`str`, *optional*, defaults to `lanczos`):
            Resampling filter to use when resizing the image.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image to [-1,1].
        do_binarize (`bool`, *optional*, defaults to `False`):
            Whether to binarize the image to 0/1.
        do_convert_rgb (`bool`, *optional*, defaults to be `False`):
            Whether to convert the images to RGB format.
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

        ratio = 1.0
        if max_side_length is not None:
            if height > width:
                max_side_length_ratio = max_side_length / height
            else:
                max_side_length_ratio = max_side_length / width

        cur_pixels = height * width
        max_pixels_ratio = (max_pixels / cur_pixels) ** 0.5
        # Clamp ratio to <=1 to avoid upscaling input images in preprocessing.
        ratio = min(max_pixels_ratio, max_side_length_ratio, 1.0)

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

        Args:
            image (`PipelineImageInput`):
                The image input, accepted formats are PIL images, NumPy arrays, PyTorch tensors; Also accept list of
                supported formats.
            height (`int`, *optional*):
                The height in preprocessed image. If `None`, will use the `get_default_height_width()` to get default
                height.
            width (`int`, *optional*):
                The width in preprocessed. If `None`, will use get_default_height_width()` to get the default width.
            resize_mode (`str`, *optional*, defaults to `default`):
                The resize mode, can be one of `default` or `fill`. If `default`, will resize the image to fit within
                the specified width and height, and it may not maintaining the original aspect ratio. If `fill`, will
                resize the image to fit within the specified width and height, maintaining the aspect ratio, and then
                center the image within the dimensions, filling empty with data from image. If `crop`, will resize the
                image to fit within the specified width and height, maintaining the aspect ratio, and then center the
                image within the dimensions, cropping the excess. Note that resize_mode `fill` and `crop` are only
                supported for PIL image input.
            crops_coords (`List[Tuple[int, int, int, int]]`, *optional*, defaults to `None`):
                The crop coordinates for each image in the batch. If `None`, will not crop the image.

        Returns:
            `torch.Tensor`:
                The preprocessed image tensor with shape `[B, C, H, W]`.
        """

        supported_formats = (PIL.Image.Image, np.ndarray, torch.Tensor)

        # Expand the missing dimension for 3-dimensional pytorch tensor or numpy array that represents grayscale image
        if self.config.do_convert_grayscale and isinstance(image, (torch.Tensor, np.ndarray)) and image.ndim == 3:
            if isinstance(image, torch.Tensor):
                # if image is a pytorch tensor could have 2 possible shapes:
                #    1. batch x height x width: we should insert the channel dimension at position 1
                #    2. channel x height x width: we should insert batch dimension at position 0,
                #       however, since both channel and batch dimension has same size 1, it is same to insert at position 1
                #    for simplicity, we insert a dimension of size 1 at position 1 for both cases
                image = image.unsqueeze(1)
            else:
                # if it is a numpy array, it could have 2 possible shapes:
                #   1. batch x height x width: insert channel dimension on last position
                #   2. height x width x channel: insert batch dimension on first position
                if image.shape[-1] == 1:
                    image = np.expand_dims(image, axis=0)
                else:
                    image = np.expand_dims(image, axis=-1)

        if isinstance(image, list) and isinstance(image[0], np.ndarray) and image[0].ndim == 4:
            warnings.warn(
                "Passing `image` as a list of 4d np.ndarray is deprecated."
                "Please concatenate the list along the batch dimension and pass it as a single 4d np.ndarray",
                FutureWarning,
            )
            image = np.concatenate(image, axis=0)
        if isinstance(image, list) and isinstance(image[0], torch.Tensor) and image[0].ndim == 4:
            warnings.warn(
                "Passing `image` as a list of 4d torch.Tensor is deprecated."
                "Please concatenate the list along the batch dimension and pass it as a single 4d torch.Tensor",
                FutureWarning,
            )
            image = torch.cat(image, axis=0)

        if not is_valid_image_imagelist(image):
            raise ValueError(
                f"Input is in incorrect format. Currently, we only support {', '.join(str(x) for x in supported_formats)}"
            )

        # Normalize to a list so the downstream path handles all input types uniformly.
        if not isinstance(image, list):
            image = [image]

        if isinstance(image[0], PIL.Image.Image):
            if crops_coords is not None:
                image = [i.crop(crops_coords) for i in image]
            if self.config.do_resize:
                height, width = self.get_new_height_width(image[0], height, width, max_pixels, max_side_length)
                image = [self.resize(i, height, width, resize_mode=resize_mode) for i in image]
            if self.config.do_convert_rgb:
                image = [self.convert_to_rgb(i) for i in image]
            elif self.config.do_convert_grayscale:
                image = [self.convert_to_grayscale(i) for i in image]
            image = self.pil_to_numpy(image)  # to np
            image = self.numpy_to_pt(image)  # to pt

        elif isinstance(image[0], np.ndarray):
            image = np.concatenate(image, axis=0) if image[0].ndim == 4 else np.stack(image, axis=0)

            image = self.numpy_to_pt(image)

            height, width = self.get_new_height_width(image, height, width, max_pixels, max_side_length)
            if self.config.do_resize:
                image = self.resize(image, height, width)

        elif isinstance(image[0], torch.Tensor):
            image = torch.cat(image, axis=0) if image[0].ndim == 4 else torch.stack(image, axis=0)

            if self.config.do_convert_grayscale and image.ndim == 3:
                image = image.unsqueeze(1)

            channel = image.shape[1]
            # don't need any preprocess if the image is latents
            if channel == self.config.vae_latent_channels:
                return image

            height, width = self.get_new_height_width(image, height, width, max_pixels, max_side_length)
            if self.config.do_resize:
                image = self.resize(image, height, width)

        # expected range [0,1], normalize to [-1,1]
        do_normalize = self.config.do_normalize
        if do_normalize and image.min() < 0:
            warnings.warn(
                "Passing `image` as torch tensor with value range in [-1,1] is deprecated. The expected value range for image tensor is [0,1] "
                f"when passing as pytorch tensor or numpy Array. You passed `image` with value range [{image.min()},{image.max()}]",
                FutureWarning,
            )
            do_normalize = False
        if do_normalize:
            image = self.normalize(image)

        if self.config.do_binarize:
            image = self.binarize(image)

        return image
