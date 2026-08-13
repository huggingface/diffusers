# Copyright 2026 NVIDIA Corporation and The HuggingFace Team. All rights reserved.
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
import torch
import torch.nn.functional as F
from PIL import Image

from ...configuration_utils import register_to_config
from ...video_processor import VideoProcessor


class Cosmos3VideoProcessor(VideoProcessor):
    r"""Video processor for Cosmos3 pipelines.

    In addition to the generic video preprocessing supplied by [`VideoProcessor`], this processor implements the
    image-to-video conditioning transform used by Cosmos Framework. Raw PIL images are resized proportionally, center
    cropped, quantized to uint8, and normalized to ``[-1, 1]`` before VAE encoding.
    """

    @register_to_config
    def __init__(self, vae_scale_factor: int = 16, resample: str = "bilinear"):
        super().__init__(do_resize=True, vae_scale_factor=vae_scale_factor, resample=resample)

    @staticmethod
    def _resize_and_center_crop(image: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """Resize an NCHW image proportionally and center crop with Framework-compatible geometry."""
        original_height, original_width = image.shape[-2:]
        scale = max(width / original_width, height / original_height)
        resized_height = math.ceil(scale * original_height)
        resized_width = math.ceil(scale * original_width)
        image = F.interpolate(
            image,
            size=(resized_height, resized_width),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        crop_top = int(round((resized_height - height) / 2.0))
        crop_left = int(round((resized_width - width) / 2.0))
        return image[:, :, crop_top : crop_top + height, crop_left : crop_left + width]

    def preprocess_conditioning_image(self, image, height: int, width: int) -> torch.Tensor:
        """Preprocess an image-to-video reference image for Cosmos3 VAE conditioning.

        The exact Framework path applies to raw PIL input. Tensor and NumPy inputs retain the standard Diffusers input
        contract because they may already be normalized or preprocessed by the caller.
        """
        if not isinstance(image, Image.Image):
            return self.preprocess(image, height=height, width=width)

        pixels = torch.from_numpy(np.array(image.convert("RGB"), copy=True)).permute(2, 0, 1).float().unsqueeze(0)
        pixels = self._resize_and_center_crop(pixels, height, width)
        return pixels.round().clamp(0, 255).to(torch.uint8).float() / 127.5 - 1.0
