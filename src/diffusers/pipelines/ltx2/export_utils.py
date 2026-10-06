# Copyright 2025 The Lightricks team and The HuggingFace Team.
# All rights reserved.
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
import os
from fractions import Fraction
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

from ...utils import is_av_available, is_openexr_available
from .image_processor import ACESCG_TO_REC2020, REC709_TO_REC2020


_CAN_USE_AV = is_av_available()
if _CAN_USE_AV:
    import av
else:
    raise ImportError(
        "PyAV is required to use LTX 2.0 video export utilities. You can install it with `pip install av`"
    )


def encode_hdr_tensor_to_mp4(
    frames: torch.Tensor | np.ndarray,
    output_mp4: str | Path,
    frame_rate: float,
    tone_mapping_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    tone_map_in_rgb: bool = True,
    crf: int = 18,
) -> None:
    """
    Converts a linear HDR tensor (for example, as outputted by `LTX2HDRPipeline`) to a SDR `.mp4` file (specifically, a
    sRGB-tonemapped H.264 `.mp4`).

    Args:
        frames (`torch.Tensor` or `np.ndarray`):
            A linear HDR tensors with RGB values in `[0, ∞)` of shape `(F, H, W, 3)`.
        output_mp4 (`str` or `pathlib.Path`):
            Output MP4 path.
        frame_rate (`float`):
            Frame rate for the output video.
        tone_mapping_fn (`Callable[[np.ndarray], np.ndarray]`, *optional*, defaults to `None`):
            An optional tone mapping function which takes a float32 NumPy array of shape `(H, W, 3)` containing linear
            HDR values in `[0, ∞)` and returns tone-mapped linear values in `[0, 1]`. The sRGB transfer function (OETF)
            is applied afterwards — do **not** pre-apply gamma inside this function. If `None`, defaults to
            [`simple_tone_map`], which clips values above `1.0`. The channel ordering of the input array is controlled
            by `tone_map_in_rgb`: RGB by default (matching the `LTX2HDRPipeline` output), or BGR when
            `tone_map_in_rgb=False`. This is the opposite default to `encode_exr_sequence_to_mp4`.
        tone_map_in_rgb (`bool`, *optional*, defaults to `True`):
            When `True` (default), frames are passed as RGB to `tone_mapping_fn`, and the output frame is tagged as
            `rgb24`. Use this when `tone_mapping_fn` expects RGB input (e.g. operators from `colour-science`). When
            `False`, the frames first have their channels flipped to BGR, which is the native format for
            `opencv-python` tone mappers (e.g. `cv2.createTonemapReinhard().process`). Note that this is the opposite
            default to `encode_exr_sequence_to_mp4`.
        crf (`int`, *optional*, defaults to `18`):
            libx264 CRF quality factor. Lower values produce higher quality.
    """
    if isinstance(frames, torch.Tensor):
        frames = frames.cpu().float().numpy()

    container = av.open(str(output_mp4), mode="w")
    stream = container.add_stream("libx264", rate=Fraction(frame_rate).limit_denominator(1000))
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": str(crf), "movflags": "+faststart"}

    pix_fmt = "rgb24" if tone_map_in_rgb else "bgr24"
    if tone_mapping_fn is None:
        # Default to simple tone mapping function which clips values above 1.0 to 1.0. This is what the original
        # LTX-2.X code does, but you may want to do some non-trivial tone-mapping to make the sample look better.
        def simple_tone_map(x: np.ndarray) -> np.ndarray:
            return np.clip(x, 0.0, 1.0)

        tone_mapping_fn = simple_tone_map

    try:
        for i, hdr in enumerate(frames):
            if not tone_map_in_rgb:
                hdr = hdr[..., ::-1]
            hdr_mapped = tone_mapping_fn(hdr)

            hdr_mapped = np.clip(hdr_mapped, 0.0, 1.0)  # Clamp to [0, 1] in case tone mapper does not
            # Apply the sRBG (Rec.709 OETF) transfer function to linear light in [0, 1]
            sdr = np.where(
                hdr_mapped <= 0.0031308, hdr_mapped * 12.92, 1.055 * np.power(hdr_mapped, 1.0 / 2.4) - 0.055
            )
            out8 = (sdr * 255.0 + 0.5).astype(np.uint8)

            if i == 0:
                stream.height, stream.width = out8.shape[:2]

            frame = av.VideoFrame.from_ndarray(out8, format=pix_fmt)
            for packet in stream.encode(frame):
                container.mux(packet)

        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()


# ARIB STD-B67 HLG OETF constants, as used by the LTX-2 reference (`colour` `CONSTANTS_ARIBSTDB67`).
_HLG_A = 0.17883277
_HLG_B = 0.28466892
_HLG_C = 0.55991073

# FFmpeg colour tags (`AVCOL_PRI_BT2020`, `AVCOL_TRC_ARIB_STD_B67`, `AVCOL_SPC_BT2020_NCL`, `AVCOL_RANGE_MPEG`).
_AV_COLOR_PRIMARIES_BT2020 = 9
_AV_COLOR_TRC_ARIB_STD_B67 = 18
_AV_COLORSPACE_BT2020_NCL = 9
_AV_COLOR_RANGE_MPEG = 1

# Full-range RGB -> Y'CbCr matrix with the BT.2020 non-constant-luminance weights (Kr = 0.2627, Kb = 0.0593). The
# reference computes it as the float64 inverse of `colour.matrix_YCbCr(WEIGHTS_YCBCR["ITU-R BT.2020"])` stored as
# float32; these are those float32 values.
_RGB_TO_YCBCR_BT2020 = (
    (0.26269999146461487, 0.6779999732971191, 0.059300001710653305),
    (-0.13963006436824799, -0.3603699505329132, 0.5),
    (0.5, -0.45978569984436035, -0.04021429643034935),
)

_HLG_PRIMARIES_TO_REC2020 = {"rec709": REC709_TO_REC2020, "acescg": ACESCG_TO_REC2020}


def _hlg_inverse_oetf(signal: float) -> float:
    r"""Inverse HLG OETF (ITU-R BT.2100 reference constants): HLG signal `[0, 1]` -> scene-linear `[0, 1]`."""
    a = _HLG_A
    b = 1.0 - 4.0 * a
    c = 0.5 - a * math.log(4.0 * a)
    linear = (signal / 0.5) ** 2 if signal <= 0.5 else math.exp((signal - c) / a) + b
    return linear / 12.0


def _hlg_oetf(x: torch.Tensor) -> torch.Tensor:
    r"""HLG OETF (ARIB STD-B67): scene-linear `[0, 1]` -> HLG signal, clamped to `[0, 1]`."""
    return torch.where(
        x <= 1.0 / 12.0,
        torch.sqrt((3.0 * x).clamp(min=0.0)),
        _HLG_A * torch.log((12.0 * x - _HLG_B).clamp(min=1e-12)) + _HLG_C,
    ).clamp(0.0, 1.0)


def _linear_to_hlg_signal(
    rgb_linear: torch.Tensor, primaries_matrix: torch.Tensor, white_x: float, roll_k: float
) -> torch.Tensor:
    r"""Scene-linear `(..., 3, H, W)` RGB -> Rec.2020 HLG signal `[0, 1]`, with diffuse white mapped to `white_x`."""
    lin = torch.nan_to_num(
        torch.einsum("...chw,dc->...dhw", rgb_linear, primaries_matrix).clamp(min=0.0),
        nan=0.0,
        neginf=0.0,
    )
    # Diffuse white (linear 1.0) maps to `white_x`; highlights roll off exponentially toward 1.0.
    x = torch.where(
        lin <= 1.0,
        lin * white_x,
        1.0 - (1.0 - white_x) * torch.exp(-roll_k * (lin - 1.0)),
    )
    return _hlg_oetf(x)


def _rgb_to_yuv420p10_bt2020_limited(rgb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""
    Float RGB `[0, 1]` `(F, 3, H, W)` -> planar 10-bit limited-range BT.2020 NCL Y, U, V code values (4:2:0).

    Returns `int32` tensors with values in `[0, 1023]`.
    """
    _, _, height, width = rgb.shape
    if height % 2 != 0 or width % 2 != 0:
        raise ValueError(f"HLG export requires an even frame height and width, got {height}x{width}.")
    matrix = torch.as_tensor(_RGB_TO_YCBCR_BT2020, dtype=torch.float32).to(device=rgb.device, dtype=rgb.dtype)
    yuv = (rgb.movedim(-3, -1).flatten(-3, -2) @ matrix.T).unflatten(-2, (height, width)).movedim(-1, -3)
    y = yuv[:, :1]
    uv = F.avg_pool2d(yuv[:, 1:3].contiguous(), kernel_size=2, stride=2)
    # Limited ("MPEG") range at 10 bits: Y = (219 * E' + 16) * 4, Cb/Cr = (224 * E' + 128) * 4.
    y = y * (219 * 4) + 16 * 4
    uv = uv * (224 * 4) + 128 * 4
    y = y[:, 0].round().clamp(0, 1023).to(torch.int32)
    u = uv[:, 0].round().clamp(0, 1023).to(torch.int32)
    v = uv[:, 1].round().clamp(0, 1023).to(torch.int32)
    return y, u, v


def _x265_params(threads: int, width: int, height: int) -> str:
    r"""libx265 `x265-params` for a BT.2020 HLG `hvc1` MP4, as in the reference implementation."""
    params = (
        "colorprim=bt2020:transfer=arib-std-b67:colormatrix=bt2020nc:range=limited:"
        f"repeat-headers=1:info=0:pools={threads}"
    )
    if width <= 32 and height <= 32:
        # With a single CTU per axis, frame-threading and B-frames can flush packets that the MP4 muxer rejects on
        # very short clips.
        return f"{params}:frame-threads=1:bframes=0:lookahead=0"
    return f"{params}:frame-threads=4"


def encode_hdr_tensor_to_hlg_mp4(
    frames: torch.Tensor | np.ndarray,
    output_mp4: str | Path,
    frame_rate: float,
    primaries: str = "rec709",
    white_signal: float = 0.75,
    rolloff_k: float | None = None,
    crf: int = 12,
    preset: str = "ultrafast",
    thread_count: int = 0,
    device: str | torch.device | None = None,
) -> None:
    r"""
    Encodes a scene-linear HDR tensor to a BT.2020 / HLG / 10-bit HEVC `.mp4` file, following the LTX-2 reference HDR
    export.

    Each frame is converted to Rec.2020 primaries; diffuse white (linear `1.0`) is mapped to the HLG signal
    `white_signal` (`0.75` is the ITU-R BT.2408 HDR reference white) and brighter values roll off exponentially toward
    the HLG peak, with a slope that is continuous at diffuse white. The ARIB STD-B67 OETF is then applied, followed by
    a conversion to 10-bit limited-range BT.2020 non-constant-luminance Y'CbCr 4:2:0. The result is encoded with
    `libx265` (`yuv420p10le`, `hvc1` tag) and tagged as BT.2020 primaries, ARIB STD-B67 transfer, BT.2020 NCL matrix
    and limited range. No mastering-display or content-light-level metadata is written.

    Requires a PyAV build whose FFmpeg includes `libx265`.

    Args:
        frames (`torch.Tensor` or `np.ndarray`):
            Scene-linear HDR frames of shape `(F, H, W, 3)` with values in `[0, ∞)`, for example the output of
            [`LTX2VideoHDRProcessor.postprocess_hdr_video`] for a single video. `H` and `W` must be even.
        output_mp4 (`str` or `pathlib.Path`):
            Output MP4 path.
        frame_rate (`float`):
            Frame rate for the output video.
        primaries (`str`, *optional*, defaults to `"rec709"`):
            Primaries of `frames`: `"rec709"` or `"acescg"`.
        white_signal (`float`, *optional*, defaults to `0.75`):
            HLG signal level that diffuse white (linear `1.0`) is mapped to.
        rolloff_k (`float`, *optional*):
            Exponential highlight roll-off rate. Defaults to `white_x / (1 - white_x)`, where `white_x` is the
            scene-linear value of `white_signal`, which makes the mapping C1-continuous at linear `1.0`.
        crf (`int`, *optional*, defaults to `12`):
            libx265 CRF quality factor. Lower values produce higher quality.
        preset (`str`, *optional*, defaults to `"ultrafast"`):
            libx265 preset.
        thread_count (`int`, *optional*, defaults to `0`):
            libx265 thread pool size. `0` uses the number of CPUs, capped at 16.
        device (`str` or `torch.device`, *optional*):
            Device for the colour conversion. Defaults to the device of `frames` (CPU for NumPy input).
    """
    if "libx265" not in av.codecs_available:
        raise RuntimeError(
            "HLG export requires the `libx265` encoder, but the FFmpeg build used by PyAV does not include it."
        )
    if primaries not in _HLG_PRIMARIES_TO_REC2020:
        raise ValueError(f"Unsupported primaries {primaries!r}. Expected 'rec709' or 'acescg'.")
    if not 0.0 < white_signal < 1.0:
        raise ValueError(f"`white_signal` must be in (0, 1), got {white_signal}.")

    frames = torch.as_tensor(frames) if isinstance(frames, np.ndarray) else frames.detach()
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"Expected `frames` of shape (F, H, W, 3), got {tuple(frames.shape)}.")
    num_frames, height, width, _ = frames.shape
    if num_frames == 0:
        raise ValueError("No HDR frames to encode.")
    if height % 2 != 0 or width % 2 != 0:
        raise ValueError(f"HLG export requires an even frame height and width, got {height}x{width}.")

    device = frames.device if device is None else torch.device(device)
    primaries_matrix = torch.as_tensor(_HLG_PRIMARIES_TO_REC2020[primaries], dtype=torch.float32).to(device)
    white_x = _hlg_inverse_oetf(white_signal)
    roll_k = rolloff_k if rolloff_k is not None else white_x / (1.0 - white_x)
    threads = thread_count if thread_count > 0 else max(1, min(os.cpu_count() or 8, 16))

    output_mp4 = Path(output_mp4)
    container = av.open(str(output_mp4), mode="w", options={"movflags": "+faststart"})
    try:
        stream = container.add_stream("libx265", rate=Fraction(frame_rate).limit_denominator(1000))
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p10le"
        stream.codec_tag = "hvc1"
        stream.options = {"crf": str(crf), "preset": preset, "x265-params": _x265_params(threads, width, height)}
        codec_context = stream.codec_context
        codec_context.thread_count = threads
        codec_context.thread_type = "FRAME"
        codec_context.color_primaries = _AV_COLOR_PRIMARIES_BT2020
        codec_context.color_trc = _AV_COLOR_TRC_ARIB_STD_B67
        codec_context.colorspace = _AV_COLORSPACE_BT2020_NCL
        codec_context.color_range = _AV_COLOR_RANGE_MPEG

        for index in range(num_frames):
            rgb = frames[index : index + 1].to(device=device, dtype=torch.float32).movedim(-1, -3)
            hlg = _linear_to_hlg_signal(rgb, primaries_matrix, white_x, roll_k)
            planes_yuv = [plane[0].cpu().numpy().astype(np.uint16) for plane in _rgb_to_yuv420p10_bt2020_limited(hlg)]

            frame = av.VideoFrame(width, height, "yuv420p10le")
            for plane, src in zip(frame.planes, planes_yuv):
                dest = np.frombuffer(plane, dtype=np.uint16).reshape(plane.height, plane.line_size // 2)
                dest[:, : src.shape[1]] = src
            frame.colorspace = _AV_COLORSPACE_BT2020_NCL
            frame.color_range = _AV_COLOR_RANGE_MPEG
            for packet in stream.encode(frame):
                container.mux(packet)

        for packet in stream.encode():
            container.mux(packet)
    except BaseException:
        container.close()
        output_mp4.unlink(missing_ok=True)
        raise
    container.close()


# OpenEXR `chromaticities` (R, G, B and white point xy) and `colorSpace` tags of the reference EXR writer, per EXR
# colour space (`ltx_pipelines` `EXRColorSpace`). ACEScct frames are tagged with AP1 chromaticities.
_EXR_CHROMATICITIES = {
    "rec709": (0.64, 0.33, 0.30, 0.60, 0.15, 0.06, 0.3127, 0.3290),
    "acescg": (0.713, 0.293, 0.165, 0.830, 0.128, 0.044, 0.32168, 0.33767),
}
_EXR_COLORSPACES = {
    "srgb_linear": ("rec709", "sRGB"),
    "acescg": ("acescg", "ACEScg"),
    "acescct": ("acescg", "ACEScct"),
}


def _import_openexr():
    if not is_openexr_available():
        raise ImportError("OpenEXR is required to write EXR frames. You can install it with `pip install OpenEXR`.")
    import OpenEXR

    # `OpenEXR.File` was added in OpenEXR 3.3; older bindings only expose the legacy `OutputFile` API.
    if not hasattr(OpenEXR, "File"):
        raise ImportError(
            "OpenEXR>=3.3 is required to write EXR frames. You can upgrade it with `pip install -U OpenEXR`."
        )
    return OpenEXR


def save_exr_frame(
    frame: torch.Tensor | np.ndarray,
    output_exr: str | Path,
    primaries: str = "rec709",
    color_space: str = "sRGB",
    half: bool = True,
) -> None:
    r"""
    Saves a single RGB frame as an OpenEXR file tagged with its colour space, following the LTX-2 reference EXR writer:
    a scanline image with `R`, `G` and `B` channels, ZIP compression, and `chromaticities` and `colorSpace` header
    attributes.

    Requires the `OpenEXR` package (`pip install OpenEXR`, version 3.3 or later).

    Args:
        frame (`torch.Tensor` or `np.ndarray`):
            Float frame of shape `(H, W, 3)` or `(3, H, W)`.
        output_exr (`str` or `pathlib.Path`):
            Output EXR path.
        primaries (`str`, *optional*, defaults to `"rec709"`):
            Colour primaries written to the `chromaticities` attribute: `"rec709"` or `"acescg"` (AP1).
        color_space (`str`, *optional*, defaults to `"sRGB"`):
            Value of the `colorSpace` string attribute, which describes the encoding (e.g. `"sRGB"` or `"ACEScg"` for
            scene-linear values, `"ACEScct"` for log codes).
        half (`bool`, *optional*, defaults to `True`):
            Write 16-bit half floats. When `False`, 32-bit floats are written, unless `frame` is already float16.
    """
    OpenEXR = _import_openexr()
    if primaries not in _EXR_CHROMATICITIES:
        raise ValueError(f"Unsupported primaries {primaries!r}. Expected 'rec709' or 'acescg'.")

    if isinstance(frame, torch.Tensor):
        use_half = half or frame.dtype == torch.float16
        frame = frame.detach().cpu().float().numpy()
    else:
        use_half = half or frame.dtype == np.float16
    frame = np.asarray(frame, dtype=np.float32)
    if frame.ndim == 3 and frame.shape[0] == 3:
        frame = frame.transpose(1, 2, 0)
    if frame.ndim != 3 or frame.shape[-1] != 3:
        raise ValueError(f"Expected `frame` of shape (H, W, 3) or (3, H, W), got {frame.shape}.")
    frame = frame.astype(np.float16 if use_half else np.float32)

    header = {
        "type": OpenEXR.scanlineimage,
        "compression": OpenEXR.ZIP_COMPRESSION,
        "chromaticities": _EXR_CHROMATICITIES[primaries],
        "colorSpace": color_space,
    }
    channels = {name: np.ascontiguousarray(frame[..., index]) for index, name in enumerate("RGB")}
    with OpenEXR.File(header, channels) as exr_file:
        exr_file.write(str(output_exr))


def export_to_exr_sequence(
    frames: torch.Tensor | np.ndarray,
    output_dir: str | Path,
    exr_colorspace: str = "acescg",
    half: bool = True,
) -> list[str]:
    r"""
    Saves HDR frames as a directory of OpenEXR files named `frame_00000.exr`, `frame_00001.exr`, ..., following the
    LTX-2 reference HDR export. Each file is written with [`~pipelines.ltx2.export_utils.save_exr_frame`].

    Requires the `OpenEXR` package (`pip install OpenEXR`, version 3.3 or later).

    Args:
        frames (`torch.Tensor` or `np.ndarray`):
            Frames of shape `(F, H, W, 3)` in the colour space given by `exr_colorspace`, for example the output of
            [`LTX2VideoHDRProcessor.postprocess_hdr_video`] for a single video with the matching `output_colorspace`.
        output_dir (`str` or `pathlib.Path`):
            Output directory. It is created if it does not exist.
        exr_colorspace (`str`, *optional*, defaults to `"acescg"`):
            Colour space of `frames`, which sets the EXR tags: `"acescg"` (scene-linear ACEScg, AP1 chromaticities,
            `colorSpace="ACEScg"`), `"srgb_linear"` (scene-linear Rec.709, Rec.709 chromaticities, `colorSpace="sRGB"`)
            or `"acescct"` (ACEScct codes, AP1 chromaticities, `colorSpace="ACEScct"`).
        half (`bool`, *optional*, defaults to `True`):
            Write 16-bit half floats.

    Returns:
        `list[str]`: Paths of the written EXR files.
    """
    _import_openexr()
    if exr_colorspace not in _EXR_COLORSPACES:
        raise ValueError(f"Unsupported EXR colorspace {exr_colorspace!r}. Expected one of {tuple(_EXR_COLORSPACES)}.")
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"Expected `frames` of shape (F, H, W, 3), got {tuple(frames.shape)}.")
    primaries, color_space = _EXR_COLORSPACES[exr_colorspace]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for index, frame in enumerate(frames):
        path = output_dir / f"frame_{index:05d}.exr"
        save_exr_frame(frame, path, primaries=primaries, color_space=color_space, half=half)
        paths.append(str(path))
    return paths
