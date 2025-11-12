# Copyright 2025 The Wan Team and The HuggingFace Team. All rights reserved.
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
from ...image_processor import VaeImageProcessor
from ...utils import PIL_INTERPOLATION


class WanAnimateImageProcessor(VaeImageProcessor):
    r"""
    Image processor to preprocess the reference (character) image for the Wan Animate model.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to downscale the image's (height, width) dimensions to multiples of `vae_scale_factor`. Can accept
            `height` and `width` arguments from [`image_processor.VaeImageProcessor.preprocess`] method.
        vae_scale_factor (`int`, *optional*, defaults to `8`):
            VAE (spatial) scale factor. If `do_resize` is `True`, the image is automatically resized to multiples of
            this factor.
        vae_latent_channels (`int`, *optional*, defaults to `16`):
            VAE latent channels.
        spatial_patch_size (`Tuple[int, int]`, *optional*, defaults to `(2, 2)`):
            The spatial patch size used by the diffusion transformer. For Wan models, this is typically (2, 2).
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
        fill_color (`str` or `float` or `Tuple[float, ...]`, *optional*, defaults to `None`):
            An optional fill color when `resize_mode` is set to `"fill"`. This will fill the empty space with that
            color instead of filling with data from the image. Any valid `color` argument to `PIL.Image.new` is valid;
            if `None`, will default to filling with data from `image`.
    """

    @register_to_config
    def __init__(
        self,
        do_resize: bool = True,
        vae_scale_factor: int = 8,
        vae_latent_channels: int = 16,
        spatial_patch_size: Tuple[int, int] = (2, 2),
        resample: str = "lanczos",
        reducing_gap: int = None,
        do_normalize: bool = True,
        do_binarize: bool = False,
        do_convert_rgb: bool = False,
        do_convert_grayscale: bool = False,
        fill_color: Optional[Union[str, float, Tuple[float, ...]]] = 0,
    ):
        super().__init__()
        if do_convert_rgb and do_convert_grayscale:
            raise ValueError(
                "`do_convert_rgb` and `do_convert_grayscale` can not both be set to `True`,"
                " if you intended to convert the image into RGB format, please set `do_convert_grayscale = False`.",
                " if you intended to convert the image into grayscale format, please set `do_convert_rgb = False`",
            )

    def _resize_and_fill(
        self,
        image: PIL.Image.Image,
        width: int,
        height: int,
    ) -> PIL.Image.Image:
        r"""
        Resize the image to fit within the specified width and height, maintaining the aspect ratio, and then center
        the image within the dimensions, filling empty with data from image.

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

        ratio = width / height
        src_ratio = image.width / image.height
        fill_with_image_data = self.config.fill_color is None
        fill_color = self.config.fill_color or 0

        src_w = width if ratio < src_ratio else image.width * height // image.height
        src_h = height if ratio >= src_ratio else image.height * width // image.width

        resized = image.resize((src_w, src_h), resample=PIL_INTERPOLATION[self.config.resample])
        res = PIL.Image.new("RGB", (width, height), color=fill_color)
        res.paste(resized, box=(width // 2 - src_w // 2, height // 2 - src_h // 2))

        if fill_with_image_data:
            if ratio < src_ratio:
                fill_height = height // 2 - src_h // 2
                if fill_height > 0:
                    res.paste(resized.resize((width, fill_height), box=(0, 0, width, 0)), box=(0, 0))
                    res.paste(
                        resized.resize((width, fill_height), box=(0, resized.height, width, resized.height)),
                        box=(0, fill_height + src_h),
                    )
            elif ratio > src_ratio:
                fill_width = width // 2 - src_w // 2
                if fill_width > 0:
                    res.paste(resized.resize((fill_width, height), box=(0, 0, 0, height)), box=(0, 0))
                    res.paste(
                        resized.resize((fill_width, height), box=(resized.width, 0, resized.width, height)),
                        box=(fill_width + src_w, 0),
                    )

        return res

    def get_default_height_width(
        self,
        image: Union[PIL.Image.Image, np.ndarray, torch.Tensor],
        height: Optional[int] = None,
        width: Optional[int] = None,
    ) -> Tuple[int, int]:
        r"""
        Returns the height and width of the image, downscaled to the next integer multiple of `vae_scale_factor`.

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
                `vae_scale_factor * spatial_patch_size`.
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

        max_area = width * height
        aspect_ratio = height / width
        mod_value_h = self.config.vae_scale_factor * self.config.spatial_patch_size[0]
        mod_value_w = self.config.vae_scale_factor * self.config.spatial_patch_size[1]

        # Try to preserve the aspect ratio
        height = round(np.sqrt(max_area * aspect_ratio)) // mod_value_h * mod_value_h
        width = round(np.sqrt(max_area / aspect_ratio)) // mod_value_w * mod_value_w

        return height, width
