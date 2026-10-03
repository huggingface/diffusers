# Copyright 2025 Lightricks and The HuggingFace Team. All rights reserved.
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
import torch
import torch.nn.functional as F

from ...configuration_utils import register_to_config
from ...utils import logging
from ...video_processor import VideoProcessor


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class LTX2VideoHDRProcessor(VideoProcessor):
    r"""
    Video processor for the LTX-2 HDR IC-LoRA pipeline.

    Inherits standard video preprocessing from [`VideoProcessor`] and additionally supports:

    - `preprocess_reference_video_hdr`: aspect-ratio-preserving resize followed by reflect-padding to the target size.
      For LDR (SDR Rec.709) reference videos, `LogC3.compress_ldr` is an identity clamp, so the numerical output is
      equivalent to the standard [-1, 1] normalization used by [`VideoProcessor.preprocess_video`] — only the resize
      strategy differs (reflect-pad vs center-crop).
    - `postprocess_hdr_video`: applies the LogC3 inverse transform to the VAE's decoded output, mapping `[0, 1]` →
      linear HDR `[0, ∞)`.

    Args:
        vae_scale_factor (`int`, *optional*, defaults to `32`):
            VAE (spatial) scale factor for the LTX-2 video VAE.
        resample (`str`, *optional*, defaults to `"bilinear"`):
            Resampling filter used by the base [`VaeImageProcessor`] for PIL/tensor resizing.
        hdr_transform (`str`, *optional*, defaults to `"logc3"`):
            HDR transform identifier. Only `"logc3"` (ARRI EI 800) is currently supported.
    """

    # LogC3 (ARRI EI 800) coefficients, ported from `ltx_core.hdr.LogC3`.
    _LOGC3_A = 5.555556
    _LOGC3_B = 0.052272
    _LOGC3_C = 0.247190
    _LOGC3_D = 0.385537
    _LOGC3_E = 5.367655
    _LOGC3_F = 0.092809
    _LOGC3_CUT = 0.010591

    @register_to_config
    def __init__(
        self,
        vae_scale_factor: int = 32,
        resample: str = "bilinear",
        hdr_transform: str = "logc3",
    ):
        super().__init__(
            do_resize=True,
            vae_scale_factor=vae_scale_factor,
            resample=resample,
        )
        if hdr_transform != "logc3":
            raise ValueError(f"Unsupported HDR transform {hdr_transform!r}. Only 'logc3' is supported.")

    @classmethod
    def _logc3_decompress(cls, logc: torch.Tensor) -> torch.Tensor:
        r"""Decompress LogC3 `[0, 1]` → linear HDR `[0, ∞)`."""
        logc = torch.clamp(logc, 0.0, 1.0)
        cut_log = cls._LOGC3_E * cls._LOGC3_CUT + cls._LOGC3_F
        lin_from_log = (torch.pow(10.0, (logc - cls._LOGC3_D) / cls._LOGC3_C) - cls._LOGC3_B) / cls._LOGC3_A
        lin_from_lin = (logc - cls._LOGC3_F) / cls._LOGC3_E
        return torch.where(logc >= cut_log, lin_from_log, lin_from_lin)

    @staticmethod
    def _resize_and_reflect_pad_video(video: torch.Tensor, height: int, width: int) -> torch.Tensor:
        r"""
        Resize a video tensor preserving aspect ratio, then reflect-pad to the exact target dimensions.

        Args:
            video (`torch.Tensor`): Input of shape `(B, C, F, H, W)`.
            height (`int`), width (`int`): Target spatial dimensions.

        Returns:
            `torch.Tensor`: Resized and padded video of shape `(B, C, F, height, width)`.
        """
        b, c, f, src_h, src_w = video.shape

        if height >= src_h and width >= src_w:
            new_h, new_w = src_h, src_w
        else:
            scale = min(height / src_h, width / src_w)
            new_h = round(src_h * scale)
            new_w = round(src_w * scale)
            # (B, C, F, H, W) → (B, F, C, H, W) → (B*F, C, H, W) for 2D per-frame interpolation.
            video = video.permute(0, 2, 1, 3, 4).reshape(b * f, c, src_h, src_w)
            video = F.interpolate(video, size=(new_h, new_w), mode="bilinear", align_corners=False)
            video = video.reshape(b, f, c, new_h, new_w).permute(0, 2, 1, 3, 4)

        pad_bottom = height - new_h
        pad_right = width - new_w
        if pad_bottom > 0 or pad_right > 0:
            # `reflect` pad requires the pad amount to be strictly less than the corresponding input dim.
            pad_mode = "reflect" if pad_bottom < new_h and pad_right < new_w else "replicate"
            video = video.permute(0, 2, 1, 3, 4).reshape(b * f, c, new_h, new_w)
            video = F.pad(video, (0, pad_right, 0, pad_bottom), mode=pad_mode)
            video = video.reshape(b, f, c, height, width).permute(0, 2, 1, 3, 4)

        return video

    def preprocess_reference_video_hdr(
        self,
        video,
        height: int,
        width: int,
    ) -> torch.Tensor:
        r"""
        Preprocess a reference (SDR) video for HDR IC-LoRA conditioning.

        Runs the input through the standard video preprocessing (normalization to `[-1, 1]`) without resizing, then
        applies reflect-pad resize to the target dimensions. For LDR inputs this is numerically equivalent to
        `load_video_conditioning_hdr` in the reference implementation (since `LogC3.compress_ldr` is an identity clamp
        on `[0, 1]` inputs).

        Args:
            video: Input accepted by `VideoProcessor.preprocess_video` (list of PIL images, 4D/5D tensor/array, etc.).
            height (`int`), width (`int`): Target spatial dimensions.

        Returns:
            `torch.Tensor`: Preprocessed video of shape `(B, C, F, height, width)` with values in `[-1, 1]`.
        """
        video = self.preprocess_video(video, height=None, width=None)  # (B, C, F, src_h, src_w) in [-1, 1]
        video = self._resize_and_reflect_pad_video(video, height, width)
        return video

    def postprocess_hdr_video(self, video: torch.Tensor, output_type: str = "np") -> torch.Tensor | np.ndarray:
        r"""
        Postprocess the VAE's decoded output to linear HDR.

        Args:
            video (`torch.Tensor`):
                VAE decoded output in VAE range `[-1, 1]`, shape `(B, C, F, H, W)`.
            output_type (`str`, *optional*, defaults to `"np"`):
                Output type of post-processed video tensor; should be in `["np", "pt"]`.

        Returns:
            Returns linear HDR video with values in `[0, ∞)`, depending on `output_type`:
              - `output_type="pt"`: `torch.Tensor` with shape `(B, F, H, W, C)` and dtype `float32`.
              - `output_type="np"`: `np.ndarray` with shape `(B, F, H, W, C)` and dtype `float32`.
        """
        if output_type not in ["np", "pt"]:
            logger.warning(
                f"output_type {output_type} is not supported for LTX-2.X HDR postprocessing. Supported types are `np`"
                f" and `pt`; the output_type will be set to `np`."
            )
            output_type = "np"

        video = self.denormalize(video.float())
        # Apply the inverse transform function to get linear HDR light
        video = self._logc3_decompress(video)

        # Permute to channels-last: [B, C, F, H, W] --> [B, F, H, W, C]
        video = video = video.permute(0, 2, 3, 4, 1).contiguous()
        if output_type == "pt":
            return video

        video = video.cpu().numpy()
        return video
