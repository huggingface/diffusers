# Copyright 2026 The HuggingFace Team. All rights reserved.
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

import numpy as np
import PIL
import torch

from ...configuration_utils import register_to_config
from ...utils import PIL_INTERPOLATION
from ...video_processor import VideoProcessor


class WanAnimate2VideoProcessor(VideoProcessor):
    r"""
    Letterbox processor for Wan-Animate-2: `preprocess` / `preprocess_video` with `resize_mode="fill"` keep the
    aspect ratio and fill the remainder with `fill_color` (black by default). The resized content is pasted at
    `((height - src_h) // 2, (width - src_w) // 2)` -- the placement convention of the reference implementation --
    instead of `VaeImageProcessor`'s `(height // 2 - src_h // 2, ...)`. The two differ by one row or column
    whenever the frame dimension is even and the content dimension odd, and that one-pixel placement shift is a
    real difference in what the model sees.
    """

    @register_to_config
    def __init__(
        self,
        do_resize: bool = True,
        vae_scale_factor: int = 8,
        vae_latent_channels: int = 16,
        spatial_patch_size: tuple[int, int] = (2, 2),
        resample: str = "lanczos",
        reducing_gap: int = None,
        do_normalize: bool = True,
        do_binarize: bool = False,
        do_convert_rgb: bool = False,
        do_convert_grayscale: bool = False,
        fill_color: str | float | tuple[float, ...] | None = 0,
    ):
        # Deliberately does not chain into the parent `__init__`: it is itself
        # `register_to_config`-decorated, so calling it re-registers every shared field with its
        # defaults and discards what the caller asked for -- `resample` included. The decorator on
        # this method already records the full config; the parent's body only validates.
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
        ratio = width / height
        src_ratio = image.width / image.height

        src_w = width if ratio < src_ratio else image.width * height // image.height
        src_h = height if ratio >= src_ratio else image.height * width // image.width

        resized = image.resize((src_w, src_h), resample=PIL_INTERPOLATION[self.config.resample])
        res = PIL.Image.new("RGB", (width, height), color=self.config.fill_color or 0)
        res.paste(resized, box=((width - src_w) // 2, (height - src_h) // 2))
        return res

    # Copied from diffusers.pipelines.wan.image_processor.WanAnimateImageProcessor.get_default_height_width
    def get_default_height_width(
        self,
        image: PIL.Image.Image | np.ndarray | torch.Tensor,
        height: int | None = None,
        width: int | None = None,
    ) -> tuple[int, int]:
        r"""
        Returns the height and width of the image, downscaled to the next integer multiple of `vae_scale_factor`.

        Args:
            image (`PIL.Image.Image | np.ndarray | torch.Tensor`):
                The image input, which can be a PIL image, NumPy array, or PyTorch tensor. If it is a NumPy array, it
                should have shape `[batch, height, width]` or `[batch, height, width, channels]`. If it is a PyTorch
                tensor, it should have shape `[batch, channels, height, width]`.
            height (`int | None`, *optional*, defaults to `None`):
                The height of the preprocessed image. If `None`, the height of the `image` input will be used.
            width (`int | None`, *optional*, defaults to `None`):
                The width of the preprocessed image. If `None`, the width of the `image` input will be used.

        Returns:
            `tuple[int, int]`:
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
