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
import PIL.Image
import torch
import torch.nn.functional as F

from ...configuration_utils import register_to_config
from ...image_processor import is_valid_image, is_valid_image_imagelist
from ...utils import logging
from ...video_processor import VideoProcessor


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# ACEScct constants (Academy S-2016-001), ported from `ltx_core.hdr` in the Lightricks LTX-2 reference.
ACESCCT_A = 10.5402377416545
ACESCCT_B = 0.0729055341958355
ACESCCT_X_BRK = 0.0078125
ACESCCT_Y_BRK = 0.155251141552511
ACESCCT_LOG_M = 17.52
ACESCCT_LOG_B = 9.72

# IEC 61966-2-1 sRGB EOTF constants.
_SRGB_A = 0.055
_SRGB_LINEAR_THRESHOLD = 0.04045
_SRGB_LINEAR_SLOPE = 12.92
_SRGB_GAMMA = 2.4

# Linear RGB -> RGB primaries matrices (Bradford chromatic adaptation), applied as `out[d] = sum_c M[d][c] * in[c]`.
# The LTX-2 reference (`ltx_core.color.primaries`) computes them at import time with colour-science
# (`colour.matrix_RGB_to_RGB(..., chromatic_adaptation_transform="Bradford")`) and stores them as float32;
# `REC709_TO_ACESCG` is the float64 inverse of the float32 `ACESCG_TO_REC709`. The values below are those float32
# matrices (colour-science 0.4.4), written out in full so that no colour-science dependency is needed.
ACESCG_TO_REC709 = (
    (1.7050509452819824, -0.6217921376228333, -0.08325887471437454),
    (-0.13025641441345215, 1.1408047676086426, -0.010548318736255169),
    (-0.02400335669517517, -0.1289689689874649, 1.1529723405838013),
)
REC709_TO_ACESCG = (
    (0.6130974292755127, 0.33952316641807556, 0.04737945273518562),
    (0.07019372284412384, 0.9163538813591003, 0.013452397659420967),
    (0.0206155925989151, 0.10956976562738419, 0.8698146343231201),
)
REC709_TO_REC2020 = (
    (0.6274039149284363, 0.3292830288410187, 0.04331306740641594),
    (0.06909728795289993, 0.9195404052734375, 0.011362315155565739),
    (0.016391439363360405, 0.08801330626010895, 0.8955952525138855),
)
ACESCG_TO_REC2020 = (
    (1.025824785232544, -0.020053191110491753, -0.005771556869149208),
    (-0.0022343695163726807, 1.0045864582061768, -0.002352132461965084),
    (-0.0050133513286709785, -0.025290071964263916, 1.0303034782409668),
)

# Input colour spaces accepted by the ACEScct input transform (`ltx_pipelines` `HDRICLoraInputColorSpace`).
ACESCCT_INPUT_COLORSPACES = ("srgb_gamma", "srgb", "acescg", "acescct")
# Output colour spaces of the ACEScct output transform: scene-linear ACEScg (AP1), scene-linear Rec.709, or the raw
# ACEScct codes.
ACESCCT_OUTPUT_COLORSPACES = ("rec709", "acescg", "acescct")


def apply_primaries_matrix(video: torch.Tensor, matrix) -> torch.Tensor:
    r"""
    Apply a 3x3 linear primaries matrix to an RGB video or image tensor.

    The channel axis is `dim=1` for 5D `(B, C, F, H, W)` inputs and `dim=-3` otherwise (`(..., C, H, W)`), matching the
    reference implementation.

    Args:
        video (`torch.Tensor`):
            Linear RGB tensor of shape `(B, 3, F, H, W)` or `(..., 3, H, W)`.
        matrix (`torch.Tensor` or nested `tuple` of `float`):
            The 3x3 matrix `M`, applied as `out[d] = sum_c M[d][c] * in[c]`.

    Returns:
        `torch.Tensor`: The converted tensor, with the same shape, device and dtype as `video`.
    """
    matrix = torch.as_tensor(matrix, dtype=torch.float32).to(device=video.device, dtype=video.dtype)
    if video.ndim == 5:
        return torch.einsum("bcfhw,dc->bdfhw", video, matrix)
    return torch.einsum("...chw,dc->...dhw", video, matrix)


def srgb_eotf_to_linear(srgb: torch.Tensor) -> torch.Tensor:
    r"""
    sRGB-encoded `[0, 1]` code values to display-linear Rec.709 light (IEC 61966-2-1 EOTF).

    Inputs are cast to float32 and clamped to `[0, 1]` first.

    Args:
        srgb (`torch.Tensor`): sRGB-encoded values.

    Returns:
        `torch.Tensor`: Linear Rec.709 values in `[0, 1]`, as float32.
    """
    x = torch.clamp(srgb.float(), 0.0, 1.0)
    return torch.where(
        x <= _SRGB_LINEAR_THRESHOLD,
        x / _SRGB_LINEAR_SLOPE,
        torch.pow((x + _SRGB_A) / (1.0 + _SRGB_A), _SRGB_GAMMA),
    )


def acescct_encode(linear_acescg: torch.Tensor) -> torch.Tensor:
    r"""
    Encode scene-linear ACEScg (AP1) values to ACEScct `[0, 1]`.

    Follows the reference implementation, which differs from the ACEScct specification in two places: negative inputs
    are clamped to `0` before encoding, and the output is clamped to `[0, 1]` (so linear values above `2 ** (17.52 -
    9.72) ~= 222.86` are clipped).

    Args:
        linear_acescg (`torch.Tensor`): Scene-linear ACEScg values.

    Returns:
        `torch.Tensor`: ACEScct codes in `[0, 1]`.
    """
    x = torch.clamp(linear_acescg, min=0.0)
    log_part = (torch.log2(torch.clamp(x, min=1e-12)) + ACESCCT_LOG_B) / ACESCCT_LOG_M
    lin_part = ACESCCT_A * x + ACESCCT_B
    return torch.clamp(torch.where(x > ACESCCT_X_BRK, log_part, lin_part), 0.0, 1.0)


def acescct_decode(acescct: torch.Tensor) -> torch.Tensor:
    r"""
    Decode ACEScct codes to scene-linear ACEScg (AP1) values.

    The input is clamped to `[0, 1]` first. The linear toe is kept as is, so a code of `0` decodes to a small negative
    value (`-0.0069169`), as in the reference implementation.

    Args:
        acescct (`torch.Tensor`): ACEScct codes.

    Returns:
        `torch.Tensor`: Scene-linear ACEScg values.
    """
    ct = torch.clamp(acescct, 0.0, 1.0)
    lin_from_log = torch.pow(2.0, ct * ACESCCT_LOG_M - ACESCCT_LOG_B)
    lin_from_lin = (ct - ACESCCT_B) / ACESCCT_A
    return torch.where(ct > ACESCCT_Y_BRK, lin_from_log, lin_from_lin)


def to_acescct(video: torch.Tensor, input_colorspace: str = "srgb_gamma") -> torch.Tensor:
    r"""
    Input transform of the LTX-2.5 SDR-To-HDR IC-LoRA: map RGB values to the ACEScct `[0, 1]` working space.

    The supported input colour spaces are those of the reference implementation:

    - `"srgb_gamma"`: sRGB-encoded Rec.709 in `[0, 1]` (e.g. an 8-bit video divided by 255). The sRGB EOTF is applied,
      then the Rec.709 -> AP1 matrix, then the ACEScct encoding.
    - `"srgb"`: scene-linear Rec.709 (e.g. a linear EXR plate). Same as `"srgb_gamma"` without the EOTF. Note that the
      reference also applies this mode to 8-bit video divided by 255, i.e. without linearizing it.
    - `"acescg"`: scene-linear ACEScg (AP1). Only the ACEScct encoding is applied.
    - `"acescct"`: values that are already ACEScct codes. They are only clamped to `[0, 1]`.

    Args:
        video (`torch.Tensor`):
            RGB tensor of shape `(B, 3, F, H, W)` or `(..., 3, H, W)`.
        input_colorspace (`str`, *optional*, defaults to `"srgb_gamma"`):
            One of `"srgb_gamma"`, `"srgb"`, `"acescg"` or `"acescct"`.

    Returns:
        `torch.Tensor`: ACEScct codes in `[0, 1]`, as float32, with the same shape as `video`.
    """
    if input_colorspace not in ACESCCT_INPUT_COLORSPACES:
        raise ValueError(
            f"Unsupported input colorspace {input_colorspace!r}. Expected one of {ACESCCT_INPUT_COLORSPACES}."
        )
    video = video.float()
    if input_colorspace == "acescct":
        return video.clamp(0.0, 1.0)
    if input_colorspace == "srgb_gamma":
        video = srgb_eotf_to_linear(video)
    if input_colorspace in ("srgb_gamma", "srgb"):
        video = apply_primaries_matrix(video, REC709_TO_ACESCG)
    return acescct_encode(video.clamp(min=0.0))


def acescct_to_linear(acescct: torch.Tensor, output_colorspace: str = "rec709") -> torch.Tensor:
    r"""
    Output transform of the LTX-2.5 SDR-To-HDR IC-LoRA: map ACEScct codes to scene-linear HDR.

    The codes are decoded to linear ACEScg, converted to the requested primaries, then clamped to `>= 0`. The clamp
    happens after the primaries matrix, as in the reference implementation, so out-of-gamut Rec.709 values are clipped.

    Args:
        acescct (`torch.Tensor`):
            ACEScct codes of shape `(B, 3, F, H, W)` or `(..., 3, H, W)`.
        output_colorspace (`str`, *optional*, defaults to `"rec709"`):
            `"rec709"` for scene-linear Rec.709 primaries, or `"acescg"` for scene-linear ACEScg (AP1) primaries.

    Returns:
        `torch.Tensor`: Scene-linear HDR values in `[0, inf)`, as float32.
    """
    if output_colorspace not in ("rec709", "acescg"):
        raise ValueError(f"Unsupported output colorspace {output_colorspace!r}. Expected 'rec709' or 'acescg'.")
    linear_acescg = acescct_decode(acescct.float())
    if output_colorspace == "rec709":
        linear_acescg = apply_primaries_matrix(linear_acescg, ACESCG_TO_REC709)
    return linear_acescg.clamp(min=0.0)


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

    With `hdr_transform="acescct"` (LTX-2.5 SDR-To-HDR IC-LoRA), `preprocess_reference_video_hdr` additionally applies
    an input transform to the ACEScct working space (see [`~pipelines.ltx2.image_processor.to_acescct`]) and
    `postprocess_hdr_video` decodes ACEScct to scene-linear HDR (see
    [`~pipelines.ltx2.image_processor.acescct_to_linear`]).

    Args:
        vae_scale_factor (`int`, *optional*, defaults to `32`):
            VAE (spatial) scale factor for the LTX-2 video VAE.
        resample (`str`, *optional*, defaults to `"bilinear"`):
            Resampling filter used by the base [`VaeImageProcessor`] for PIL/tensor resizing.
        hdr_transform (`str`, *optional*, defaults to `"logc3"`):
            HDR transform identifier. `"logc3"` (ARRI LogC3 EI 800, LTX-2.3 HDR) or `"acescct"` (ACEScct, LTX-2.5
            SDR-To-HDR).
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
        if hdr_transform not in ("logc3", "acescct"):
            raise ValueError(f"Unsupported HDR transform {hdr_transform!r}. Expected 'logc3' or 'acescct'.")

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

    @staticmethod
    def _video_to_float_tensor(video) -> tuple[torch.Tensor, bool]:
        r"""
        Convert a video input to a float32 `(B, C, F, H, W)` tensor without resizing or normalizing it.

        Accepts the layouts of [`VideoProcessor.preprocess_video`]. Integer inputs (PIL images, `uint8` arrays or
        tensors) are divided by 255; floating point inputs are kept as is, so scene-linear values above 1 survive.

        Returns:
            `tuple[torch.Tensor, bool]`: The video tensor, and whether the input was integer-valued.
        """
        if isinstance(video, (np.ndarray, torch.Tensor)) and video.ndim == 5:
            videos = list(video)
        elif isinstance(video, list) and is_valid_image(video[0]) or is_valid_image_imagelist(video):
            videos = [video]
        elif isinstance(video, list) and is_valid_image_imagelist(video[0]):
            videos = video
        else:
            raise ValueError(
                "Input is in incorrect format. Currently, we only support numpy.ndarray, torch.Tensor, PIL.Image.Image"
            )

        tensors = []
        is_integer = True
        for frames in videos:
            if isinstance(frames, list) and isinstance(frames[0], PIL.Image.Image):
                frames = np.stack([np.array(frame.convert("RGB")) for frame in frames], axis=0)
            elif isinstance(frames, list) and isinstance(frames[0], np.ndarray):
                frames = np.stack(frames, axis=0)
            elif isinstance(frames, list) and isinstance(frames[0], torch.Tensor):
                frames = torch.stack(frames, dim=0)
            if isinstance(frames, np.ndarray):
                # NumPy frames are channels-last: (F, H, W, C) -> (F, C, H, W).
                frames = torch.from_numpy(np.ascontiguousarray(frames)).permute(0, 3, 1, 2)
            if frames.is_floating_point():
                is_integer = False
                frames = frames.float()
            else:
                frames = frames.float() / 255.0
            tensors.append(frames)
        # (B, F, C, H, W) -> (B, C, F, H, W)
        return torch.stack(tensors, dim=0).permute(0, 2, 1, 3, 4), is_integer

    def preprocess_reference_video_hdr(
        self,
        video,
        height: int,
        width: int,
        input_colorspace: str | None = None,
    ) -> torch.Tensor:
        r"""
        Preprocess a reference (SDR) video for HDR IC-LoRA conditioning.

        With `hdr_transform="logc3"`, runs the input through the standard video preprocessing (normalization to `[-1,
        1]`) without resizing, then applies reflect-pad resize to the target dimensions. For LDR inputs this is
        numerically equivalent to `load_video_conditioning_hdr` in the reference implementation (since
        `LogC3.compress_ldr` is an identity clamp on `[0, 1]` inputs).

        With `hdr_transform="acescct"`, the input is mapped to ACEScct `[0, 1]` with
        [`~pipelines.ltx2.image_processor.to_acescct`], reflect-pad resized, then mapped to `[-1, 1]`. Integer inputs
        (PIL images, `uint8` arrays or tensors) are divided by 255 and transformed before resizing, as the reference
        does for MP4/MOV inputs; floating point inputs are used as is and resized before the transform, as the
        reference does for EXR inputs. The two orders only differ when the video is downscaled.

        Args:
            video: Input accepted by `VideoProcessor.preprocess_video` (list of PIL images, 4D/5D tensor/array, etc.).
            height (`int`), width (`int`): Target spatial dimensions.
            input_colorspace (`str`, *optional*):
                Colour space of `video` for `hdr_transform="acescct"`: `"srgb_gamma"` (default), `"srgb"`, `"acescg"`
                or `"acescct"`. See [`~pipelines.ltx2.image_processor.to_acescct`]. Must be `None` for
                `hdr_transform="logc3"`.

        Returns:
            `torch.Tensor`: Preprocessed video of shape `(B, C, F, height, width)` with values in `[-1, 1]`.
        """
        if self.config.hdr_transform == "logc3":
            if input_colorspace is not None:
                raise ValueError("`input_colorspace` is only supported with `hdr_transform='acescct'`.")
            video = self.preprocess_video(video, height=None, width=None)  # (B, C, F, src_h, src_w) in [-1, 1]
            video = self._resize_and_reflect_pad_video(video, height, width)
            return video

        input_colorspace = "srgb_gamma" if input_colorspace is None else input_colorspace
        if input_colorspace not in ACESCCT_INPUT_COLORSPACES:
            raise ValueError(
                f"Unsupported input colorspace {input_colorspace!r}. Expected one of {ACESCCT_INPUT_COLORSPACES}."
            )
        video, is_integer = self._video_to_float_tensor(video)
        if is_integer:
            video = to_acescct(video, input_colorspace)
            video = self._resize_and_reflect_pad_video(video, height, width)
        else:
            video = self._resize_and_reflect_pad_video(video, height, width)
            video = to_acescct(video, input_colorspace)
        return video * 2.0 - 1.0

    def postprocess_hdr_video(
        self, video: torch.Tensor, output_type: str = "np", output_colorspace: str | None = None
    ) -> torch.Tensor | np.ndarray:
        r"""
        Postprocess the VAE's decoded output to linear HDR.

        Args:
            video (`torch.Tensor`):
                VAE decoded output in VAE range `[-1, 1]`, shape `(B, C, F, H, W)`.
            output_type (`str`, *optional*, defaults to `"np"`):
                Output type of post-processed video tensor; should be in `["np", "pt"]`.
            output_colorspace (`str`, *optional*):
                Output colour space for `hdr_transform="acescct"`: `"rec709"` (default, scene-linear Rec.709),
                `"acescg"` (scene-linear ACEScg) or `"acescct"` (the decoded ACEScct codes in `[0, 1]`, without
                decoding them to linear). See [`~pipelines.ltx2.image_processor.acescct_to_linear`]. Must be `None` for
                `hdr_transform="logc3"`, whose output is linear in the primaries of the input.

        Returns:
            Returns linear HDR video with values in `[0, ∞)` (or ACEScct codes in `[0, 1]` with
            `output_colorspace="acescct"`), depending on `output_type`:
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
        if self.config.hdr_transform == "logc3":
            if output_colorspace is not None:
                raise ValueError("`output_colorspace` is only supported with `hdr_transform='acescct'`.")
            # Apply the inverse transform function to get linear HDR light
            video = self._logc3_decompress(video)
        else:
            output_colorspace = "rec709" if output_colorspace is None else output_colorspace
            if output_colorspace not in ACESCCT_OUTPUT_COLORSPACES:
                raise ValueError(
                    f"Unsupported output colorspace {output_colorspace!r}. Expected one of "
                    f"{ACESCCT_OUTPUT_COLORSPACES}."
                )
            if output_colorspace != "acescct":
                video = acescct_to_linear(video, output_colorspace)

        # Permute to channels-last: [B, C, F, H, W] --> [B, F, H, W, C]
        video = video = video.permute(0, 2, 3, 4, 1).contiguous()
        if output_type == "pt":
            return video

        video = video.cpu().numpy()
        return video
