# Copyright 2026 The MiniMax and HuggingFace Teams. All rights reserved.
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

r"""
The references of a MiniMax-H3 `ref2va` request: one public dataclass per modality.

`ref2va` conditions on an ordered list of references — images, videos (with their soundtrack) and audio clips —
packed one block per reference ahead of the generated rows. The order is semantic twice over: it fixes the
`"<Picture i>"` / `"<Audio j>"` / `"<Video k>"` labels of the prompt presentation, and it advances the shared
audio/video rotary clock, so a different order is a different request.

Every reference holds in-memory media plus the rate that media carries. No block of this model opens a media file;
each class decodes a path or a URL through its `from_file` classmethod, which brings the rates along. That is the
reason to prefer `from_file` over a bare `load_video`: MiniMax-H3 resamples a reference onto its own 24 fps, so a
video whose real rate is lost on the way in is conditioned on at the wrong speed, silently.
"""

import contextlib
import os
import tempfile
from dataclasses import dataclass
from urllib.parse import unquote, urlparse

import numpy as np
import requests
import torch
from PIL import Image

from ...utils import is_av_available, load_image
from ...utils.constants import DIFFUSERS_REQUEST_TIMEOUT
from .modular_pipeline import MINIMAX_H3_FPS


@dataclass
class MiniMaxH3Reference:
    r"""
    Base class of the three references a [`MiniMaxH3Ref2VABlocks`] request conditions on:
    [`MiniMaxH3ImageReference`], [`MiniMaxH3VideoReference`] and [`MiniMaxH3AudioReference`].

    References are passed to the blocks as a list, **in the order the model should read them**: the order labels them
    in the prompt presentation and lays them out on the shared rotary clock, so a different order is a different
    request.

    Every reference holds in-memory media, and the rate that media carries where there is one — MiniMax-H3 resamples a
    reference onto its own 24 fps and onto the audio VAE's sample rate, so a rate lost on the way in is a request
    conditioned at the wrong speed. Each class decodes a file through its `from_file` classmethod, along with the
    rates.

    ```py
    >>> import numpy as np
    >>> from diffusers.modular_pipelines.minimax_h3 import (
    ...     MiniMaxH3AudioReference,
    ...     MiniMaxH3ImageReference,
    ...     MiniMaxH3VideoReference,
    ... )

    >>> references = [
    ...     MiniMaxH3ImageReference.from_file("subject.png"),
    ...     MiniMaxH3VideoReference.from_file("motion_ref.mp4"),  # frames, their `fps`, and the soundtrack
    ...     MiniMaxH3AudioReference.from_file("voice.wav"),  # waveform and its `sample_rate`
    ... ]

    >>> # Media a request produced itself declares the rate it was produced at.
    >>> frames = np.zeros((30, 480, 854, 3), dtype="uint8")
    >>> reference = MiniMaxH3VideoReference(frames=frames, fps=30.0)
    ```
    """


@dataclass
class MiniMaxH3ImageReference(MiniMaxH3Reference):
    r"""
    A subject, style or scene reference: at most 9 per request.

    Attributes:
        image (`PIL.Image.Image`):
            The reference image. It never binds the generated geometry — it is encoded at a 2048 pixel short edge of
            its own aspect ratio, whatever canvas the request generates at.
    """

    image: Image.Image

    kind = "image"
    has_audio = False

    @classmethod
    def from_file(cls, media) -> "MiniMaxH3ImageReference":
        r"""
        Load an image file into a [`MiniMaxH3ImageReference`], through [`~utils.load_image`].

        Args:
            media (`str` or `os.PathLike`): Path or URL of the image.
        """
        return cls(image=load_image(str(media)))


@dataclass
class MiniMaxH3VideoReference(MiniMaxH3Reference):
    r"""
    A motion and camera reference: at most 3 per request, conditioned on together with its own soundtrack.

    Attributes:
        frames (`list[PIL.Image.Image]`, `np.ndarray` or `torch.Tensor`):
            The reference frames: a list of images, a `(num_frames, height, width, 3)` array or a `(num_frames, 3,
            height, width)` tensor, `uint8` or floating point over `[0, 1]`.
        fps (`float`, *optional*, defaults to 24.0):
            The frame rate `frames` carries, which is what places the reference's vision blocks on the conditioner's 2
            fps grid. MiniMax-H3's own clock is 24 fps, so any other rate is resampled onto it by dropping and
            duplicating whole frames — which makes this the field to get right when the frames came from a file.
        audio (`torch.Tensor` of shape `(channels, num_samples)`, *optional*):
            This video's soundtrack, mono or stereo, conditioned on as the reference's own rather than as a reference
            of its own. Left out, the reference conditions on motion alone.
        sample_rate (`int`, *optional*):
            The rate `audio` carries its samples at. Left out, it is the audio VAE's own, which leaves the samples
            untouched; any other rate is resampled onto it.
    """

    frames: list[Image.Image] | np.ndarray | torch.Tensor
    fps: float | None = None
    audio: torch.Tensor | None = None
    sample_rate: int | None = None

    kind = "video"

    def __post_init__(self):
        if self.fps is None:
            self.fps = float(MINIMAX_H3_FPS)

    @property
    def has_audio(self) -> bool:
        r"""Whether this reference contributes audio rows, i.e. whether it carries a soundtrack."""
        return self.audio is not None

    @classmethod
    def from_file(cls, media) -> "MiniMaxH3VideoReference":
        r"""
        Decode a video file into a [`MiniMaxH3VideoReference`], at the resolution, the frame rate and the soundtrack
        it carries.

        The rates land on the reference, which is the point of decoding this way rather than with
        [`~utils.load_video`]: MiniMax-H3 resamples a reference onto its own 24 fps, so a frame rate lost on the way
        in is a request conditioned at the wrong speed, with nothing to raise about it. A container whose metadata is
        wrong is corrected by overriding `fps` or `sample_rate` on the returned reference.

        Needs [PyAV](https://github.com/PyAV-Org/PyAV).

        Args:
            media (`str` or `os.PathLike`): Path or URL of the video.

        Returns:
            [`MiniMaxH3VideoReference`]: the `(num_frames, height, width, 3)` `uint8` frames at the frame rate the
            container reports, carrying its soundtrack and that soundtrack's own sample rate when it has an audio
            stream.
        """
        frames, fps, audio, sample_rate = _decode_video_file(media)
        return cls(frames=frames, fps=fps, audio=audio, sample_rate=sample_rate)


@dataclass
class MiniMaxH3AudioReference(MiniMaxH3Reference):
    r"""
    A voice or music reference: at most 3 per request, and never on its own — an audio reference has to be paired with
    at least one image or video reference. It never reaches the conditioner and is encoded by the audio VAE alone.

    Attributes:
        audio (`torch.Tensor` of shape `(channels, num_samples)`):
            The reference waveform, mono or stereo.
        sample_rate (`int`, *optional*):
            The rate `audio` carries its samples at. Left out, it is the audio VAE's own, which leaves the samples
            untouched; any other rate is resampled onto it.
    """

    audio: torch.Tensor
    sample_rate: int | None = None

    kind = "audio"
    has_audio = True

    @classmethod
    def from_file(cls, media) -> "MiniMaxH3AudioReference":
        r"""
        Decode an audio file into a [`MiniMaxH3AudioReference`], at the sample rate it carries.

        Needs [PyAV](https://github.com/PyAV-Org/PyAV).

        Args:
            media (`str` or `os.PathLike`): Path or URL of the audio, or of a video whose soundtrack is taken.

        Returns:
            [`MiniMaxH3AudioReference`]: the `(channels, num_samples)` float32 waveform at the sample rate the
            container reports.
        """
        audio, sample_rate = _decode_audio_file(media)
        return cls(audio=audio, sample_rate=sample_rate)


@contextlib.contextmanager
def _local_media_file(media):
    r"""The reference media as a local file: a URL is downloaded to a temporary file, removed on the way out."""
    path = str(media)
    if not path.startswith(("http://", "https://")):
        if not os.path.isfile(path):
            raise ValueError(
                f"Incorrect path or URL. URLs must start with `http://` or `https://`, and {path} is not a valid path."
            )
        yield path
        return

    response = requests.get(path, stream=True, timeout=DIFFUSERS_REQUEST_TIMEOUT)
    if response.status_code != 200:
        raise ValueError(f"Failed to download {path}. Status code: {response.status_code}")
    suffix = os.path.splitext(os.path.basename(unquote(urlparse(path).path)))[1]
    download = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        with download as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        yield download.name
    finally:
        os.remove(download.name)


def _import_av():
    r"""PyAV, the soft dependency a media file is decoded with."""
    if not is_av_available():
        raise ImportError(
            "Decoding a MiniMax-H3 reference from a file needs PyAV. You can install it with `pip install av`, or "
            "build the reference from decoded media itself: frames and the `fps` they carry for a video, a "
            "`(channels, num_samples)` waveform and its `sample_rate` for audio."
        )

    import av

    return av


def _decode_reference_soundtrack(av, container, stream) -> tuple[torch.Tensor, int]:
    r"""
    An audio stream's samples as a `(channels, num_samples)` float32 waveform, at the rate the container carries them.

    Args:
        av (`module`): PyAV.
        container (`av.container.InputContainer`): The open container.
        stream (`av.audio.stream.AudioStream`): The stream to decode.

    Returns:
        `tuple[torch.Tensor, int]`: the waveform and its sample rate.
    """
    sample_rate = int(stream.codec_context.sample_rate)
    # Planar float is a format conversion only: the sample rate and the channel layout stay the container's own, and a
    # mono soundtrack is upmixed later, by the setup step's audio normalization.
    resampler = av.audio.resampler.AudioResampler(format="fltp", layout=stream.layout, rate=sample_rate)
    chunks = []
    for frame in container.decode(stream):
        chunks += [torch.from_numpy(resampled.to_ndarray()) for resampled in resampler.resample(frame)]
    # Whatever the resampler is still holding.
    chunks += [torch.from_numpy(resampled.to_ndarray()) for resampled in resampler.resample(None)]
    return torch.cat(chunks, dim=-1).to(torch.float32), sample_rate


def _decode_video_file(media) -> tuple[np.ndarray, float, torch.Tensor | None, int | None]:
    r"""
    A video file's frames as `(num_frames, height, width, 3)` `uint8`, the frame rate the container reports, and its
    soundtrack with that soundtrack's sample rate (`None, None` without an audio stream). The machinery behind
    [`MiniMaxH3VideoReference.from_file`].
    """
    av = _import_av()
    with _local_media_file(media) as path, av.open(path) as container:
        stream = container.streams.video[0]
        frames, rotation = [], 0.0
        for frame in container.decode(stream):
            # The display matrix rotation belongs to the stream, and PyAV surfaces it on every frame of it.
            rotation = frame.rotation
            frames.append(frame.to_ndarray(format="rgb24"))
        frame_rate = float(stream.average_rate or stream.guessed_rate)
        soundtrack = None
        if container.streams.audio:
            # Decoding the frames drained the container, so the soundtrack is read in a second pass over it.
            container.seek(0)
            soundtrack = _decode_reference_soundtrack(av, container, container.streams.audio[0])

    if not frames:
        raise ValueError(f"No video frames to decode in {media}.")
    frames = np.stack(frames)
    # `ffmpeg` displays a frame upright by undoing the counterclockwise rotation the display matrix carries, which is
    # what this reproduces, snapped to the nearest quarter turn. A non-square pixel aspect ratio is left alone: the
    # reference implementation resolved a reference's canvas from its *display* geometry, so a stream that carries a
    # sample aspect ratio is conditioned on at the wrong shape, and correcting it is untested guesswork here.
    turns = round(rotation / 90.0) % 4
    if turns:
        frames = np.ascontiguousarray(np.rot90(frames, k=-turns, axes=(1, 2)))
    waveform, sample_rate = soundtrack if soundtrack is not None else (None, None)
    return frames, frame_rate, waveform, sample_rate


def _decode_audio_file(media) -> tuple[torch.Tensor, int]:
    r"""
    An audio file's `(channels, num_samples)` float32 waveform, at the sample rate the container reports. The
    machinery behind [`MiniMaxH3AudioReference.from_file`].
    """
    av = _import_av()
    with _local_media_file(media) as path, av.open(path) as container:
        if not container.streams.audio:
            raise ValueError(f"No audio stream to decode in {media}.")
        waveform, sample_rate = _decode_reference_soundtrack(av, container, container.streams.audio[0])
    return waveform, sample_rate
