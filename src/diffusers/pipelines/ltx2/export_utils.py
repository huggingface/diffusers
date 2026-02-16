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
