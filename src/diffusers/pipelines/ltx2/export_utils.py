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

from collections.abc import Iterator
from fractions import Fraction
from itertools import chain
from pathlib import Path
from typing import Callable

import numpy as np
import PIL.Image
import torch
from tqdm import tqdm

from ...utils import get_logger, is_av_available


logger = get_logger(__name__)  # pylint: disable=invalid-name


_CAN_USE_AV = is_av_available()
if _CAN_USE_AV:
    import av
else:
    raise ImportError(
        "PyAV is required to use LTX 2.0 video export utilities. You can install it with `pip install av`"
    )


def _prepare_audio_stream(container: av.container.Container, audio_sample_rate: int) -> av.audio.AudioStream:
    """
    Prepare the audio stream for writing.
    """
    audio_stream = container.add_stream("aac", rate=audio_sample_rate)
    audio_stream.codec_context.sample_rate = audio_sample_rate
    audio_stream.codec_context.layout = "stereo"
    audio_stream.codec_context.time_base = Fraction(1, audio_sample_rate)
    return audio_stream


def _resample_audio(
    container: av.container.Container, audio_stream: av.audio.AudioStream, frame_in: av.AudioFrame
) -> None:
    cc = audio_stream.codec_context

    # Use the encoder's format/layout/rate as the *target*
    target_format = cc.format or "fltp"  # AAC → usually fltp
    target_layout = cc.layout or "stereo"
    target_rate = cc.sample_rate or frame_in.sample_rate

    audio_resampler = av.audio.resampler.AudioResampler(
        format=target_format,
        layout=target_layout,
        rate=target_rate,
    )

    audio_next_pts = 0
    for rframe in audio_resampler.resample(frame_in):
        if rframe.pts is None:
            rframe.pts = audio_next_pts
        audio_next_pts += rframe.samples
        rframe.sample_rate = frame_in.sample_rate
        container.mux(audio_stream.encode(rframe))

    # flush audio encoder
    for packet in audio_stream.encode():
        container.mux(packet)


def _write_audio(
    container: av.container.Container,
    audio_stream: av.audio.AudioStream,
    samples: torch.Tensor,
    audio_sample_rate: int,
) -> None:
    if samples.ndim == 1:
        samples = samples[:, None]

    if samples.shape[1] != 2 and samples.shape[0] == 2:
        samples = samples.T

    if samples.shape[1] != 2:
        raise ValueError(f"Expected samples with 2 channels; got shape {samples.shape}.")

    # Convert to int16 packed for ingestion; resampler converts to encoder fmt.
    if samples.dtype != torch.int16:
        samples = torch.clip(samples, -1.0, 1.0)
        samples = (samples * 32767.0).to(torch.int16)

    frame_in = av.AudioFrame.from_ndarray(
        samples.contiguous().reshape(1, -1).cpu().numpy(),
        format="s16",
        layout="stereo",
    )
    frame_in.sample_rate = audio_sample_rate

    _resample_audio(container, audio_stream, frame_in)


def encode_video(
    video: list[PIL.Image.Image] | np.ndarray | torch.Tensor | Iterator[torch.Tensor],
    fps: int,
    audio: torch.Tensor,
    audio_sample_rate: int,
    output_path: str,
    video_chunks_number: int = 1,
) -> None:
    """
    Encodes a video with audio using the PyAV library. Based on code from the original LTX-2 repo:
    https://github.com/Lightricks/LTX-2/blob/4f410820b198e05074a1e92de793e3b59e9ab5a0/packages/ltx-pipelines/src/ltx_pipelines/utils/media_io.py#L182

    Args:
        video (`List[PIL.Image.Image]` or `np.ndarray` or `torch.Tensor`):
            A video tensor of shape [frames, height, width, channels] with integer pixel values in [0, 255]. If the
            input is a `np.ndarray`, it is expected to be a float array with values in [0, 1] (which is what pipelines
            usually return with `output_type="np"`).
        fps (`int`)
            The frames per second (FPS) of the encoded video.
        audio (`torch.Tensor`, *optional*):
            An audio waveform of shape [audio_channels, samples].
        audio_sample_rate: (`int`, *optional*):
            The sampling rate of the audio waveform. For LTX 2, this is typically 24000 (24 kHz).
        output_path (`str`):
            The path to save the encoded video to.
        video_chunks_number (`int`, *optional*, defaults to `1`):
            The number of chunks to split the video into for encoding. Each chunk will be encoded separately. The
            number of chunks to use often depends on the tiling config for the video VAE.
    """
    if isinstance(video, list) and isinstance(video[0], PIL.Image.Image):
        # Pipeline output_type="pil"; assumes each image is in "RGB" mode
        video_frames = [np.array(frame) for frame in video]
        video = np.stack(video_frames, axis=0)
        video = torch.from_numpy(video)
    elif isinstance(video, np.ndarray):
        # Pipeline output_type="np"
        is_denormalized = np.logical_and(np.zeros_like(video) <= video, video <= np.ones_like(video))
        if np.all(is_denormalized):
            video = (video * 255).round().astype("uint8")
        else:
            logger.warning(
                "Supplied `numpy.ndarray` does not have values in [0, 1]. The values will be assumed to be pixel "
                "values in [0, ..., 255] and will be used as is."
            )
        video = torch.from_numpy(video)

    if isinstance(video, torch.Tensor):
        # Split into video_chunks_number along the frame dimension
        video = torch.tensor_split(video, video_chunks_number, dim=0)
        video = iter(video)

    first_chunk = next(video)

    _, height, width, _ = first_chunk.shape

    container = av.open(output_path, mode="w")
    stream = container.add_stream("libx264", rate=int(fps))
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"

    if audio is not None:
        if audio_sample_rate is None:
            raise ValueError("audio_sample_rate is required when audio is provided")

        audio_stream = _prepare_audio_stream(container, audio_sample_rate)

    for video_chunk in tqdm(chain([first_chunk], video), total=video_chunks_number, desc="Encoding video chunks"):
        video_chunk_cpu = video_chunk.to("cpu").numpy()
        for frame_array in video_chunk_cpu:
            frame = av.VideoFrame.from_ndarray(frame_array, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)

    # Flush encoder
    for packet in stream.encode():
        container.mux(packet)

    if audio is not None:
        _write_audio(container, audio_stream, audio, audio_sample_rate)

    container.close()


# ---------------------------------------------------------------------------
# HDR export helpers (used with LTX2HDRLoraPipeline).
#
# These mirror the reference CLI's `save_exr_tensor`, `_linear_to_srgb`, and
# `encode_exr_sequence_to_mp4` in `ltx_pipelines.utils.media_io`.
# ---------------------------------------------------------------------------


def save_exr_tensor(
    tensor: torch.Tensor | np.ndarray,
    file_path: str | Path,
    half: bool = False,
) -> None:
    r"""
    Save a single linear-HDR frame tensor to an OpenEXR file.

    Args:
        tensor (`torch.Tensor` or `np.ndarray`):
            A float frame of shape `(H, W, C)` or `(C, H, W)` with linear HDR values in `[0, ∞)`. Channels are
            assumed to be RGB.
        file_path (`str` or `pathlib.Path`):
            Output EXR path (e.g. `frame_00000.exr`).
        half (`bool`, *optional*, defaults to `False`):
            When `True`, writes the file as `float16` (HALF) with ZIP compression. `float16` tensors are always
            saved as HALF regardless of this flag.

    The resulting EXR is tagged with Rec.709/sRGB chromaticities and `colorSpace=sRGB` to match the reference.
    Requires [OpenImageIO](https://openimageio.readthedocs.io) with OpenEXR support:
    `pip install OpenImageIO` (or `pip install oiio`).
    """
    try:
        import OpenImageIO
    except ImportError as e:  # pragma: no cover - optional dep
        raise ImportError(
            "`save_exr_tensor` requires `OpenImageIO`. Install with `pip install OpenImageIO` (with OpenEXR support)."
        ) from e

    if isinstance(tensor, torch.Tensor):
        use_half = half or tensor.dtype in (torch.float16, torch.half)
        if tensor.dim() == 3 and tensor.shape[0] == 3:
            tensor = tensor.permute(1, 2, 0)
        arr = np.ascontiguousarray(tensor.detach().cpu().numpy().astype(np.float32))
    else:
        use_half = half or tensor.dtype == np.float16
        if tensor.ndim == 3 and tensor.shape[0] == 3:
            tensor = np.transpose(tensor, (1, 2, 0))
        arr = np.ascontiguousarray(tensor.astype(np.float32))

    file_path = str(file_path)
    h, w = arr.shape[:2]
    fmt = OpenImageIO.HALF if use_half else OpenImageIO.FLOAT
    spec = OpenImageIO.ImageSpec(w, h, 3, fmt)
    spec.channelnames = ("R", "G", "B")
    spec.attribute("compression", "zip")
    spec.attribute(
        "chromaticities", "float[8]", (0.64, 0.33, 0.30, 0.60, 0.15, 0.06, 0.3127, 0.3290)
    )
    spec.attribute("colorSpace", "sRGB")

    out = OpenImageIO.ImageOutput.create(file_path)
    if out is None:
        raise RuntimeError(
            f"Failed to create EXR writer for '{file_path}'. Ensure OpenImageIO is built with OpenEXR support."
        )
    try:
        if not out.open(file_path, spec):
            raise RuntimeError(f"Failed to open EXR file '{file_path}': {out.geterror()}")
        if not out.write_image(arr):
            raise RuntimeError(f"Failed to write EXR image '{file_path}': {out.geterror()}")
    finally:
        out.close()


def simple_tone_map(x: np.ndarray) -> np.ndarray:
    r"""
    Applies a very simple tone-mapping function on (scene-referred) linear light which simply clips values above `1.0`
    to `1.0`. This is what the original LTX-2.X code does, but you probably want to do some non-trivial tone-mapping
    to make the sample look better.
    """
    return np.clip(x, 0.0, 1.0)


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    r"""
    Apply the sRGB (Rec.709) transfer function (OETF; IEC 61966-2-1) to a linear light image. Input values must be in
    `[0, 1]`.
    """
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1.0 / 2.4) - 0.055)


def encode_exr_sequence_to_mp4(
    exr_dir: str | Path,
    output_mp4: str | Path,
    frame_rate: float,
    tone_mapping_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    tone_map_in_rgb: bool = False,
    crf: int = 18,
) -> None:
    r"""
    Convert a linear-HDR EXR frame sequence into an sRGB-tonemapped H.264 `.mp4` preview.

    Each EXR frame is loaded, clipped to `[0, 1]`, passed through the sRGB OETF (no exposure/gain, EV=0), quantized
    to 8-bit, and fed into a libx264 stream at the supplied `frame_rate`.

    Args:
        exr_dir (`str` or `pathlib.Path`):
            Directory containing `frame_*.exr` files (sorted lexicographically).
        output_mp4 (`str` or `pathlib.Path`):
            Output MP4 path.
        frame_rate (`float`):
            Frame rate for the output video.
        tone_mapping_fn (`Callable[[np.ndarray], np.ndarray]`, *optional*, defaults to `None`):
            An optional tone mapping function which takes a float32 NumPy array of shape `(H, W, 3)` containing
            linear HDR values in `[0, ∞)` and returns tone-mapped linear values in `[0, 1]`. The sRGB transfer
            function (OETF) is applied afterwards — do **not** pre-apply gamma inside this function. If `None`,
            defaults to [`simple_tone_map`], which clips values above `1.0`. The channel ordering of the input
            array is controlled by `tone_map_in_rgb`: BGR by default (matching `opencv-python` conventions), or
            RGB when `tone_map_in_rgb=True` (matching `colour-science` and most other libraries).
        tone_map_in_rgb (`bool`, *optional*, defaults to `False`):
            When `True`, each EXR frame is converted from BGR to RGB before being passed to `tone_mapping_fn`,
            and the output frame is tagged as `rgb24`. Use this when `tone_mapping_fn` expects RGB input (e.g.
            operators from `colour-science`). When `False` (default), frames are passed as BGR, which is the
            native format for `opencv-python` tone mappers (e.g. `cv2.createTonemapReinhard().process`).
        crf (`int`, *optional*, defaults to `18`):
            libx264 CRF quality factor. Lower values produce higher quality.

    Requires `opencv-python` (for EXR reading via `OPENCV_IO_ENABLE_OPENEXR`).
    """
    import os
    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

    try:
        import cv2
    except ImportError as e:  # pragma: no cover - optional dep
        raise ImportError(
            "`encode_exr_sequence_to_mp4` requires `opencv-python`. Install with `pip install opencv-python`."
        ) from e

    exr_dir = Path(exr_dir)
    exr_files = sorted(exr_dir.glob("frame_*.exr"))
    if not exr_files:
        raise FileNotFoundError(f"No EXR frames found in {exr_dir}")

    container = av.open(str(output_mp4), mode="w")
    stream = container.add_stream("libx264", rate=Fraction(frame_rate).limit_denominator(1000))
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": str(crf), "movflags": "+faststart"}

    pix_fmt = "rgb24" if tone_map_in_rgb else "bgr24"
    if tone_mapping_fn is None:
        tone_mapping_fn = simple_tone_map

    try:
        for i, exr_path in enumerate(exr_files):
            hdr = cv2.imread(str(exr_path), cv2.IMREAD_UNCHANGED).astype(np.float32)
            if tone_map_in_rgb:
                hdr = hdr[..., ::-1]
            hdr_mapped = tone_mapping_fn(hdr)
            sdr = linear_to_srgb(np.maximum(hdr_mapped, 0.0))
            out8 = (sdr * 255.0 + 0.5).astype(np.uint8)

            if i == 0:
                stream.height = out8.shape[0]
                stream.width = out8.shape[1]

            frame = av.VideoFrame.from_ndarray(out8, format=pix_fmt)
            for packet in stream.encode(frame):
                container.mux(packet)

        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()


def save_hdr_video_frames_as_exr(
    frames: torch.Tensor | np.ndarray,
    exr_dir: str | Path,
    *,
    half: bool = False,
) -> list[Path]:
    r"""
    Save a batch of linear-HDR frames to a directory as `frame_{idx:05d}.exr` files.

    Args:
        frames (`torch.Tensor` or `np.ndarray`):
            HDR video tensor of shape `(F, H, W, C)` or `(F, C, H, W)` with linear HDR values in `[0, ∞)`.
        exr_dir (`str` or `pathlib.Path`):
            Output directory. Created if missing.
        half (`bool`, *optional*, defaults to `False`):
            Forwarded to [`save_exr_tensor`] — when `True`, writes EXR files as `float16`.

    Returns:
        `list[pathlib.Path]`: Paths of the written EXR files, in frame order.
    """
    exr_dir = Path(exr_dir)
    exr_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    num_frames = frames.shape[0]
    for i in range(num_frames):
        frame = frames[i]
        path = exr_dir / f"frame_{i:05d}.exr"
        save_exr_tensor(frame, path, half=half)
        paths.append(path)
    return paths
