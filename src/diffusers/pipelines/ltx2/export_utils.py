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

from fractions import Fraction
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from ...utils import is_av_available


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
