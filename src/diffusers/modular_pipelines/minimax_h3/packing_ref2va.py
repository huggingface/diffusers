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
The `ref2va` half of MiniMax-H3's packed sequence: omni-reference conditioning.

`ref2va` takes an ordered list of references — images, videos (with their soundtrack) and audio clips — and packs one
block per reference ahead of the generated rows:

```
[ text (L) | reference block 1 | reference block 2 | ... | target audio (A) | target video (V) ]
```

The blocks are packed in **request order**, and a video reference packs its soundtrack rows immediately *before* its
own video rows. Order is semantic twice over: it fixes the `"<Picture i>"` / `"<Audio j>"` / `"<Video k>"` labels of
the prompt presentation, and it advances the shared rotary clock — an image reference occupies one rotary slot, an
audio reference as many as it has latents, and a video reference the longer of its own two spans, its soundtrack and
its frames sharing one origin.

Unlike a `fl2va` keyframe, a reference never binds the target geometry: every reference is prepared at its own
resolution (2048 pixel short edge for images, MiniMax-H3's 768 pixel canvas for videos) and carries its own
aspect-normalized spatial grid.

References reach the blocks as in-memory media: a video is decoded frames plus the `fps` they carry, and a soundtrack
a waveform plus its sample rate. A [`MiniMaxH3Reference`] takes a path or a URL too, but it decodes it when it is
built, so that no block of this model ever opens a media file.
"""

import contextlib
import math
import os
import tempfile
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import unquote, urlparse

import numpy as np
import requests
import torch
from PIL import Image

from ...utils import is_av_available, load_image
from ...utils.constants import DIFFUSERS_REQUEST_TIMEOUT
from .packing import (
    _ROPE_FRAME_RESCALE,
    _ROPE_FRAMES_PER_LATENT,
    MINIMAX_H3_AUDIO_CHANNELS,
    MINIMAX_H3_AUDIO_TAG,
    MINIMAX_H3_CANVAS_MULTIPLE,
    MINIMAX_H3_FPS,
    MINIMAX_H3_FRAMES_PER_CHUNK,
    MINIMAX_H3_LATENTS_PER_CHUNK,
    MINIMAX_H3_TEXT_TAG,
    MINIMAX_H3_VIDEO_TAG,
    MiniMaxH3PackedSequence,
    _spatial_position_grid,
    _temporal_position_grid,
    resolve_canvas_size,
)


# Reference images are resized to a 2048 pixel short edge — upscaling included — and both axes are rounded to a
# multiple of 32 independently. There is no area cap, so a 4:1 reference is 8192x2048.
MINIMAX_H3_REFERENCE_IMAGE_SHORT_EDGE = 2048

# The conditioner sees a reference video at 2 fps, and Qwen3-VL merges every two of those frames into one vision
# block labelled with the mean timestamp of the pair.
MINIMAX_H3_QWEN_VIDEO_SAMPLE_FPS = 2.0
MINIMAX_H3_QWEN_TEMPORAL_PATCH = 2

# Documented per-request limits of the omni-reference task.
MINIMAX_H3_MAX_REFERENCE_IMAGES = 9
MINIMAX_H3_MAX_REFERENCE_VIDEOS = 3
MINIMAX_H3_MAX_REFERENCE_AUDIOS = 3
MINIMAX_H3_MAX_REFERENCES = 12


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
    r"""PyAV, the soft dependency a reference decodes a media file with."""
    if not is_av_available():
        raise ImportError(
            "Decoding a MiniMax-H3 reference from a file needs PyAV. You can install it with `pip install av`, or "
            "pass the decoded media itself: frames and the `fps` they carry for a video, a `(channels, num_samples)` "
            "waveform and its `sample_rate` for audio."
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
    # mono soundtrack is upmixed later, by `prepare_reference_waveform`.
    resampler = av.audio.resampler.AudioResampler(format="fltp", layout=stream.layout, rate=sample_rate)
    chunks = []
    for frame in container.decode(stream):
        chunks += [torch.from_numpy(resampled.to_ndarray()) for resampled in resampler.resample(frame)]
    # Whatever the resampler is still holding.
    chunks += [torch.from_numpy(resampled.to_ndarray()) for resampled in resampler.resample(None)]
    return torch.cat(chunks, dim=-1).to(torch.float32), sample_rate


def decode_reference_video(media) -> tuple[np.ndarray, float, tuple[torch.Tensor, int] | None]:
    r"""
    Decode a reference video file into `uint8` RGB frames, at the resolution and the frame rate it carries.

    Args:
        media (`str` or `os.PathLike`): Path or URL of the video.

    Returns:
        `tuple[np.ndarray, float, tuple[torch.Tensor, int]]`: the `(num_frames, height, width, 3)` frames, the frame
        rate the container reports, and its soundtrack with that soundtrack's own sample rate, `None` when the
        container carries no audio stream.
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
    return frames, frame_rate, soundtrack


def decode_reference_audio(media) -> tuple[torch.Tensor, int]:
    r"""
    Decode a reference audio file into a waveform, at the sample rate it carries.

    Args:
        media (`str` or `os.PathLike`): Path or URL of the audio, or of a video whose soundtrack is taken.

    Returns:
        `tuple[torch.Tensor, int]`: the `(channels, num_samples)` float32 waveform and its sample rate.
    """
    av = _import_av()
    with _local_media_file(media) as path, av.open(path) as container:
        if not container.streams.audio:
            raise ValueError(f"No audio stream to decode in {media}.")
        return _decode_reference_soundtrack(av, container, container.streams.audio[0])


@dataclass
class MiniMaxH3Reference:
    r"""
    One omni-reference of a [`MiniMaxH3Ref2VABlocks`] request: an image, a video, or an audio clip.

    A reference carries exactly one medium — plus, for a video, the `audio` of its own soundtrack, which is then
    conditioned on as that reference's own. References are passed to the blocks as a list, **in the order the model
    should read them**: the order labels them in the prompt presentation and lays them out on the shared rotary clock,
    so a different order is a different request.

    Every medium may be a path or a URL as well as in-memory media. A path is decoded here, when the reference is
    built, and with it the rates that come with it: no MiniMax-H3 block opens a media file. Decoding a video or an
    audio file needs [PyAV](https://github.com/PyAV-Org/PyAV).

    ```py
    >>> from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3Reference

    >>> # A file or a URL is decoded on the spot, at the rate the container reports.
    >>> references = [
    ...     MiniMaxH3Reference(video="motion_ref.mp4"),
    ...     MiniMaxH3Reference(image="subject.png"),
    ...     MiniMaxH3Reference(audio="voice.wav"),
    ... ]

    >>> # In-memory media instead carries the rates it was produced at, MiniMax-H3's own by default.
    >>> import numpy as np

    >>> frames = np.zeros((30, 480, 854, 3), dtype="uint8")
    >>> reference = MiniMaxH3Reference(video=frames, fps=30.0)
    ```

    Attributes:
        image (`str`, `os.PathLike`, `PIL.Image.Image`, `np.ndarray` or `torch.Tensor`, *optional*):
            A subject, style or scene reference: at most 9 per request. A path or a URL, which is read with
            [`~utils.load_image`], a `(height, width, 3)` array or a `(3, height, width)` tensor, `uint8` or floating
            point over `[0, 1]`. Mutually exclusive with `video`.
        video (`str`, `os.PathLike`, `list[PIL.Image.Image]`, `np.ndarray` or `torch.Tensor`, *optional*):
            A motion and camera reference: at most 3 per request. A path or a URL, which PyAV decodes into frames, a
            list of images, a `(num_frames, height, width, 3)` array or a `(num_frames, 3, height, width)` tensor.
            Mutually exclusive with `image`. A decoded file brings its soundtrack along, as this reference's own, so
            conditioning on a file's motion alone means decoding its frames first, with [`~utils.load_video`].
        fps (`float`, *optional*):
            The frame rate `video` carries its frames at, which is what places its vision blocks on the conditioner's
            2 fps grid. Left out, it is the rate the container reports for a decoded file and MiniMax-H3's own 24 fps
            for in-memory frames, and `fps` holds that resolved rate once the reference is built. Passing it wins over
            both, which is only needed when a container's metadata is wrong. MiniMax-H3's clock is 24 fps, so any other
            rate is resampled onto it by dropping and duplicating whole frames.
        audio (`str`, `os.PathLike` or `torch.Tensor` of shape `(channels, num_samples)`, *optional*):
            A voice or music reference, mono or stereo: at most 3 per request, and never on its own — an audio
            reference has to be paired with at least one image or video reference. A path or a URL, which PyAV decodes
            into a waveform, or the waveform itself. Passed next to `video`, it is that video's soundtrack instead of a
            reference of its own. An audio reference never reaches the conditioner and is encoded by the audio VAE
            alone.
        sample_rate (`int`, *optional*):
            The rate `audio` carries its samples at. Left out, it is the rate the container reports for a decoded file,
            and for an in-memory waveform the audio VAE's own, which leaves the samples untouched. Passing it wins over
            both, which is only needed when a container's metadata is wrong. Any other rate is resampled onto the audio
            VAE's own.
    """

    image: str | os.PathLike | Image.Image | np.ndarray | torch.Tensor | None = None
    video: str | os.PathLike | list[Image.Image] | np.ndarray | torch.Tensor | None = None
    fps: float | None = None
    audio: str | os.PathLike | torch.Tensor | None = None
    sample_rate: int | None = None

    def __post_init__(self):
        # A video reference conditions on its soundtrack too, so `audio` is a second medium of a video reference.
        media = [name for name in ("image", "video", "audio") if getattr(self, name) is not None]
        if media not in (["image"], ["video"], ["audio"], ["video", "audio"]):
            raise ValueError(
                "A `MiniMaxH3Reference` must carry exactly one of `image`, `video` or `audio` — plus, for a video, "
                f"the `audio` of its soundtrack — got {media if media else 'none of them'}."
            )

        # A path is decoded on the spot, so that the blocks only ever see in-memory media. A rate the request
        # passed explicitly wins over the one the container reports, for a container whose metadata is wrong.
        if isinstance(self.image, (str, os.PathLike)):
            self.image = load_image(str(self.image))
        if isinstance(self.video, (str, os.PathLike)):
            frames, frame_rate, soundtrack = decode_reference_video(self.video)
            self.video = frames
            self.fps = frame_rate if self.fps is None else self.fps
            if soundtrack is not None and self.audio is None:
                self.audio, soundtrack_sample_rate = soundtrack
                self.sample_rate = soundtrack_sample_rate if self.sample_rate is None else self.sample_rate
        if isinstance(self.audio, (str, os.PathLike)):
            self.audio, sample_rate = decode_reference_audio(self.audio)
            self.sample_rate = sample_rate if self.sample_rate is None else self.sample_rate
        if self.fps is None:
            self.fps = float(MINIMAX_H3_FPS)

    @property
    def kind(self) -> str:
        r"""The modality this reference is packed as: `"image"`, `"video"` or `"audio"`."""
        if self.image is not None:
            return "image"
        return "video" if self.video is not None else "audio"

    @property
    def has_audio(self) -> bool:
        r"""Whether this reference contributes audio rows, i.e. whether it carries a waveform."""
        return self.audio is not None


def reference_kind(index: int, entry: Any) -> str:
    r"""
    The modality of one `references` entry, which the [`MiniMaxH3Reference`] validated at construction.
    """
    if not isinstance(entry, MiniMaxH3Reference):
        raise ValueError(
            f"`references[{index}]` must be a [`MiniMaxH3Reference`], got {type(entry)}. A request is built from "
            "the dataclass: `MiniMaxH3Reference(image=...)`, `MiniMaxH3Reference(video=...)` or "
            "`MiniMaxH3Reference(audio=...)`."
        )
    # A reference decodes a path when it is built, so the blocks only ever see in-memory media.
    for name in ("image", "video", "audio"):
        if isinstance(getattr(entry, name), (str, os.PathLike)):
            raise ValueError(
                f"`references[{index}].{name}` is a path. MiniMax-H3 blocks never open media files: rebuild "
                "the reference, which decodes a path as it is built."
            )
    return entry.kind


@dataclass
class MiniMaxH3PreparedReference:
    r"""
    One `ref2va` reference prepared for packing, in packed order.

    A [`MiniMaxH3Reference`] is resolved in three passes: the blocks read the modality off the request (`kind`,
    `has_audio`), prepares the pixels or samples (`image`, `frames`, `waveform`), and finally encodes them, which is
    what fixes the latent geometry (`num_latent_frames`, `latent_height`, `latent_width`, `num_audio_latents`) the
    packed layout is built from.

    Attributes:
        kind (`str`):
            `"image"`, `"video"` or `"audio"`.
        has_audio (`bool`):
            Whether this reference contributes audio rows. Always `True` for `"audio"`, and `True` for a `"video"` the
            request passed a soundtrack with.
        image (`PIL.Image.Image`):
            The prepared reference image.
        frames (`np.ndarray` of shape `(num_frames, height, width, 3)`):
            The prepared reference video, `uint8` RGB at 24 fps.
        waveform (`torch.Tensor` of shape `(2, num_samples)`):
            The prepared soundtrack, stereo at the audio VAE's sample rate.
        block_timestamps (`list[float]`):
            The timestamp of every vision block the conditioner sees for a video reference.
        num_latent_frames (`int`), latent_height (`int`), latent_width (`int`):
            Latent geometry of the visual rows.
        num_audio_latents (`int`):
            Number of audio latents per channel.
    """

    kind: str
    has_audio: bool = False
    image: Any = None
    frames: Any = None
    waveform: torch.Tensor | None = None
    block_timestamps: list[float] = field(default_factory=list)
    num_latent_frames: int = 1
    latent_height: int = 0
    latent_width: int = 0
    num_audio_latents: int = 0

    @property
    def num_video_rows(self) -> int:
        r"""The number of packed video rows, for the `(1, 2, 2)` patch MiniMax-H3 packs video latents with."""
        return self.num_latent_frames * (self.latent_height // 2) * (self.latent_width // 2)

    @property
    def num_audio_rows(self) -> int:
        r"""The number of packed audio rows: one per latent and per stereo channel."""
        return self.num_audio_latents * MINIMAX_H3_AUDIO_CHANNELS


def _temporal_position_span(num_latent_frames: int) -> float:
    r"""
    The rotary time a video reference advances the clock by.

    Summed sequentially in float64, which is *not* how [`~modular_pipelines.minimax_h3.packing._temporal_position_span`] sums
    the same series: that one reproduces a numpy pairwise sum, and the two orders differ in the last ulp from 16
    latent frames onwards. The reference implementation keeps both, one per call site, so the port has to as well.
    """
    return sum(
        _ROPE_FRAME_RESCALE * _ROPE_FRAMES_PER_LATENT[index % len(_ROPE_FRAMES_PER_LATENT)]
        for index in range(num_latent_frames)
    )


def _frame_position_grid(
    latent_height: int, latent_width: int, patch_h: int, patch_w: int
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""The `(h, w)` rotary coordinates of one latent frame, and the width axis they were built from."""
    sqrt_area = np.sqrt(latent_height * latent_width)
    height_grid = _spatial_position_grid(latent_height, patch_h, sqrt_area)
    width_grid = _spatial_position_grid(latent_width, patch_w, sqrt_area)
    grids = torch.meshgrid(height_grid, width_grid, indexing="ij")
    return torch.stack([grid.reshape(-1) for grid in grids], dim=-1), width_grid


def _fill_audio_positions(
    position_ids: torch.Tensor,
    rows: slice,
    num_audio_latents: int,
    rotary_time: float,
    width_grid: torch.Tensor,
) -> None:
    r"""
    Place one channel-major audio block.

    Audio rows carry no height coordinate and are pinned to the two extremes of the width grid of *their own* block —
    the target grid for a standalone audio reference, the video's grid for a soundtrack.
    """
    time = rotary_time + torch.arange(num_audio_latents, dtype=torch.float64)
    position_ids[rows, 0] = time.repeat(MINIMAX_H3_AUDIO_CHANNELS)
    position_ids[rows, 2] = torch.cat(
        [
            torch.full((num_audio_latents,), float(width_grid[0]), dtype=torch.float64),
            torch.full((num_audio_latents,), float(width_grid[-1]), dtype=torch.float64),
        ]
    )


def build_ref2va_packed_sequence(
    text_token_tags: torch.Tensor,
    references: list[MiniMaxH3PreparedReference],
    num_latent_frames: int,
    latent_height: int,
    latent_width: int,
    num_audio_latents: int,
    patch_size: tuple[int, int, int],
) -> MiniMaxH3PackedSequence:
    r"""
    Build the `[text | reference blocks | target audio | target video]` layout of the `ref2va` task.

    Args:
        text_token_tags (`torch.Tensor` of shape `(num_text_tokens,)`):
            The modality tag of every text row. Text is tagged `1`, except for the rows of a reference's vision block,
            which MiniMax-H3 tags `0` (video).
        references (`list[MiniMaxH3PreparedReference]`):
            The references, in packed order, with their latent geometry already resolved.
        num_latent_frames (`int`): Number of target latent frames.
        latent_height (`int`): Target latent height.
        latent_width (`int`): Target latent width.
        num_audio_latents (`int`): Number of target audio latents per channel.
        patch_size (`tuple[int, int, int]`): The transformer's `(t, h, w)` patch.

    Returns:
        [`MiniMaxH3PackedSequence`]
    """
    _, patch_h, patch_w = patch_size
    num_text_tokens = text_token_tags.shape[0]
    num_target_video_rows = num_latent_frames * (latent_height // patch_h) * (latent_width // patch_w)
    num_target_audio_rows = num_audio_latents * MINIMAX_H3_AUDIO_CHANNELS
    num_reference_video_rows = sum(reference.num_video_rows for reference in references if reference.kind != "audio")
    num_reference_audio_rows = sum(reference.num_audio_rows for reference in references)
    sequence_length = (
        num_text_tokens
        + num_reference_video_rows
        + num_reference_audio_rows
        + num_target_audio_rows
        + num_target_video_rows
    )

    position_ids = torch.zeros(sequence_length, 3, dtype=torch.float64)
    position_ids[:num_text_tokens, 0] = torch.arange(num_text_tokens, dtype=torch.float64)
    target_frame_grid, target_width_grid = _frame_position_grid(latent_height, latent_width, patch_h, patch_w)

    # Reference blocks, in request order. `rotary_time` is the shared audio/video clock: it starts where the text
    # rows end and every block pushes it forward by the time that block occupies.
    video_indices, audio_indices = [], []
    cursor = num_text_tokens
    rotary_time = float(num_text_tokens)
    for reference in references:
        if reference.kind == "image":
            rows = slice(cursor, cursor + reference.num_video_rows)
            cursor = rows.stop
            video_indices.append(torch.arange(rows.start, rows.stop))
            frame_grid, _ = _frame_position_grid(reference.latent_height, reference.latent_width, patch_h, patch_w)
            position_ids[rows, 0] = rotary_time
            position_ids[rows, 1:] = frame_grid
            # An image is a single frame and takes a single integer rotary slot, not a latent frame's 5/3 units.
            rotary_time += 1.0
        elif reference.kind == "audio":
            rows = slice(cursor, cursor + reference.num_audio_rows)
            cursor = rows.stop
            audio_indices.append(torch.arange(rows.start, rows.stop))
            _fill_audio_positions(position_ids, rows, reference.num_audio_latents, rotary_time, target_width_grid)
            rotary_time += float(reference.num_audio_latents)
        elif reference.kind == "video":
            # A video reference's soundtrack rows are packed immediately before its video rows and share their
            # origin, so the two are rotary-aligned exactly as the generated audio and video are.
            audio_rows = slice(cursor, cursor + reference.num_audio_rows)
            video_rows = slice(audio_rows.stop, audio_rows.stop + reference.num_video_rows)
            cursor = video_rows.stop
            audio_indices.append(torch.arange(audio_rows.start, audio_rows.stop))
            video_indices.append(torch.arange(video_rows.start, video_rows.stop))

            frame_grid, width_grid = _frame_position_grid(
                reference.latent_height, reference.latent_width, patch_h, patch_w
            )
            _fill_audio_positions(position_ids, audio_rows, reference.num_audio_latents, rotary_time, width_grid)
            frame_time = _temporal_position_grid(reference.num_latent_frames, rotary_time)
            position_ids[video_rows, 0] = frame_time.repeat_interleave(frame_grid.shape[0])
            position_ids[video_rows, 1:] = frame_grid.repeat(reference.num_latent_frames, 1)
            rotary_time += max(
                float(reference.num_audio_latents), _temporal_position_span(reference.num_latent_frames)
            )
        else:
            raise ValueError(f"A reference must be an 'image', a 'video' or an 'audio', got {reference.kind!r}.")

    # The generated rows. Target audio and target video share the origin the reference blocks left behind.
    audio_start = cursor
    video_start = audio_start + num_target_audio_rows
    _fill_audio_positions(
        position_ids, slice(audio_start, video_start), num_audio_latents, rotary_time, target_width_grid
    )
    frame_time = _temporal_position_grid(num_latent_frames, rotary_time)
    position_ids[video_start:, 0] = frame_time.repeat_interleave(target_frame_grid.shape[0])
    position_ids[video_start:, 1:] = target_frame_grid.repeat(num_latent_frames, 1)

    video_indices = torch.cat(video_indices + [torch.arange(video_start, sequence_length)])
    audio_indices = torch.cat(audio_indices + [torch.arange(audio_start, video_start)])
    text_indices = torch.arange(num_text_tokens)

    token_tags = torch.empty(sequence_length, dtype=torch.long)
    token_tags[text_indices] = text_token_tags.to(torch.long)
    token_tags[audio_indices] = MINIMAX_H3_AUDIO_TAG
    token_tags[video_indices] = MINIMAX_H3_VIDEO_TAG

    return MiniMaxH3PackedSequence(
        sequence_length=sequence_length,
        position_ids=position_ids,
        token_tags=token_tags,
        video_indices=video_indices,
        audio_indices=audio_indices,
        text_indices=text_indices,
        num_condition_video_rows=num_reference_video_rows,
        num_condition_audio_rows=num_reference_audio_rows,
    )


def resolve_reference_image_size(width: int, height: int) -> tuple[int, int]:
    r"""
    Resolve the resolution a reference image is encoded at: a 2048 pixel short edge, both axes rounded to a multiple
    of 32. Upscaling is intended, and unlike the target canvas there is no area cap.

    Args:
        width (`int`): Width of the source image.
        height (`int`): Height of the source image.

    Returns:
        `tuple[int, int]`: the `(height, width)` the reference is resized to.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"A reference image must have a positive size, got {width}x{height}.")
    if width > 4 * height or height > 4 * width:
        raise ValueError(f"A reference image must be within 1:4 and 4:1, got {width}x{height}.")

    scale = MINIMAX_H3_REFERENCE_IMAGE_SHORT_EDGE / min(width, height)
    multiple = MINIMAX_H3_CANVAS_MULTIPLE
    return (
        max(multiple, round(height * scale / multiple) * multiple),
        max(multiple, round(width * scale / multiple) * multiple),
    )


def reference_media_to_uint8(media) -> np.ndarray:
    r"""
    An in-memory reference image or video as channels-last `uint8` RGB.

    Args:
        media (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor` or `list`):
            One image, one video, or a list of either. A `torch.Tensor` is channels-first, as everywhere else in
            diffusers, and a `np.ndarray` channels-last; floating point values are read over `[0, 1]`.

    Returns:
        `np.ndarray`: the `uint8` RGB pixels, `(height, width, 3)` for a single image and `(num_frames, height, width,
        3)` for a video or a list.
    """
    if isinstance(media, list):
        return np.stack([reference_media_to_uint8(item) for item in media])
    if isinstance(media, Image.Image):
        return np.asarray(media.convert("RGB"))
    if isinstance(media, torch.Tensor):
        media = media.movedim(-3, -1).cpu().numpy()
    media = np.asarray(media)
    if media.dtype != np.uint8:
        media = (media * 255.0).round().clip(0, 255).astype(np.uint8)
    return media


def prepare_reference_image(image, height: int, width: int):
    r"""
    Resize a reference image onto its own resolution with LANCZOS. An image that already is that size is returned
    untouched, without a resampling pass.

    Args:
        image (`PIL.Image.Image`): The reference, in RGB and already EXIF-transposed.
        height (`int`): Target height, as resolved by [`resolve_reference_image_size`].
        width (`int`): Target width.

    Returns:
        `PIL.Image.Image`: The prepared reference.
    """
    if image.size == (width, height):
        return image
    return image.resize((width, height), Image.Resampling.LANCZOS)


def resample_reference_frames(frames: np.ndarray, fps: float) -> np.ndarray:
    r"""
    Resample a reference video onto MiniMax-H3's own 24 fps, dropping and duplicating whole frames.

    The reference implementation decoded every video reference through `ffmpeg`'s `fps` filter, a constant frame rate
    resampler: every source frame lands on the output slot its timestamp rounds to, `round(index * 24 / fps)`, and a
    slot holds the last frame that landed on it, so a frame whose successor lands on the same slot is dropped and a
    frame whose successor skips slots is repeated. The end of the stream is rounded onto the grid the very same way,
    which is what fixes the length at `round(num_frames * 24 / fps)` slots and decides whether the last frame is
    repeated to reach it or dropped for overshooting it.

    Reproducing that selection is what makes a constant frame rate reference give the very frames the reference's
    decode gave, and it is an exact identity — the same array, no copy — for frames already at 24 fps.

    Args:
        frames (`np.ndarray` of shape `(num_frames, height, width, 3)`): The reference video, `uint8` RGB.
        fps (`float`): The frame rate `frames` was decoded at.

    Returns:
        `np.ndarray`: the frames on MiniMax-H3's 24 fps grid, `frames` itself when they already are.
    """
    if fps <= 0:
        raise ValueError(f"A reference video must have a positive frame rate, got {fps}.")
    if fps == MINIMAX_H3_FPS:
        return frames

    scale = MINIMAX_H3_FPS / fps
    slots = np.floor(np.arange(frames.shape[0]) * scale + 0.5).astype(np.int64)
    # Every frame is held until the slot of the next one, and the last one until the slot the stream's end rounds to.
    return np.repeat(frames, np.diff(slots, append=math.floor(frames.shape[0] * scale + 0.5)), axis=0)


def prepare_reference_frames(frames: np.ndarray, num_frames: int) -> np.ndarray:
    r"""
    Put a reference video onto the canvas its own aspect ratio resolves to, and cap it at the generated frame count.

    Frames that already are that canvas flow through untouched, with no resampling pass and no copy, and that is the
    parity-exact route: the reference implementation rescaled the frames with `ffmpeg`'s own LANCZOS scaler while
    decoding them, so only frames decoded at the canvas reproduce its pixels bit for bit. Any other size is resized
    frame by frame with PIL, the very path [`prepare_reference_image`] takes.

    Args:
        frames (`np.ndarray` of shape `(num_frames, height, width, 3)`):
            The reference video, `uint8` RGB at 24 fps, as returned by [`resample_reference_frames`].
        num_frames (`int`): The frame count the reference is truncated to, i.e. the target's own frame count.

    Returns:
        `np.ndarray` of shape `(num_frames, height, width, 3)`: The prepared reference.
    """
    if frames.ndim != 4 or frames.shape[3] != 3:
        raise ValueError(
            f"A reference video must be `(num_frames, height, width, 3)` RGB frames, got {tuple(frames.shape)}."
        )
    frames = frames[:num_frames]
    height, width = resolve_canvas_size(frames.shape[2], frames.shape[1])
    if frames.shape[1:3] == (height, width):
        return frames
    return np.stack(
        [np.asarray(Image.fromarray(frame).resize((width, height), Image.Resampling.LANCZOS)) for frame in frames]
    )


def sample_reference_video_frames(frames: np.ndarray) -> tuple[list[np.ndarray], list[float]]:
    r"""
    Sample the frames the conditioner sees from a prepared reference video, and label their vision blocks.

    The conditioner reads a reference at 2 fps: every 12th of the 24 fps frames, deduplicated. Qwen3-VL then merges
    the sampled frames in pairs — repeating the last one when there is an odd number of them — and a merged pair is
    labelled with the mean of its two timestamps, which `"<{timestamp:.1f} seconds>"` renders with Python's
    round-half-to-even, so the first block of a 2 fps pair is `"<0.2 seconds>"` rather than `"<0.3 seconds>"`.

    Args:
        frames (`np.ndarray` of shape `(num_frames, height, width, 3)`): The prepared reference video, at 24 fps.

    Returns:
        `tuple[list[np.ndarray], list[float]]`: the sampled frames and one timestamp per vision block.
    """
    stride = MINIMAX_H3_FPS / MINIMAX_H3_QWEN_VIDEO_SAMPLE_FPS
    indices, cursor = [], 0.0
    while round(cursor) < frames.shape[0]:
        if not indices or round(cursor) > indices[-1]:
            indices.append(round(cursor))
        cursor += stride

    timestamps = [index / MINIMAX_H3_QWEN_VIDEO_SAMPLE_FPS for index in range(len(indices))]
    timestamps += [timestamps[-1]] * (-len(timestamps) % MINIMAX_H3_QWEN_TEMPORAL_PATCH)
    block_timestamps = [
        (timestamps[index] + timestamps[index + MINIMAX_H3_QWEN_TEMPORAL_PATCH - 1]) / 2
        for index in range(0, len(timestamps), MINIMAX_H3_QWEN_TEMPORAL_PATCH)
    ]
    return [frames[index] for index in indices], block_timestamps


def prepare_reference_waveform(
    waveform: torch.Tensor, sample_rate: int, target_sample_rate: int, max_duration: float
) -> torch.Tensor:
    r"""
    Put a reference soundtrack on the audio VAE's sample rate, as a stereo waveform.

    The reference implementation extracts a soundtrack at a native rate, truncates it there and resamples it once, in
    torch, which this mirrors: the truncation is applied at `sample_rate` and the resampling is a single `torchaudio`
    pass. A mono waveform is upmixed by repeating its channel.

    Args:
        waveform (`torch.Tensor` of shape `(channels, num_samples)`): The soundtrack, mono or stereo.
        sample_rate (`int`): The sample rate `waveform` carries its samples at.
        target_sample_rate (`int`): The audio VAE's sample rate, i.e. what the waveform is resampled to.
        max_duration (`float`): Truncate the reference to this many seconds.

    Returns:
        `torch.Tensor` of shape `(2, num_samples)`: the float32 waveform.
    """
    waveform = torch.as_tensor(waveform)
    if waveform.ndim != 2 or waveform.shape[0] not in (1, MINIMAX_H3_AUDIO_CHANNELS):
        raise ValueError(
            "A reference soundtrack must be a `(channels, num_samples)` mono or stereo waveform, got "
            f"{tuple(waveform.shape)}."
        )
    waveform = waveform.to(torch.float32)[:, : int(max_duration * sample_rate)]
    if waveform.shape[0] != MINIMAX_H3_AUDIO_CHANNELS:
        waveform = waveform.expand(MINIMAX_H3_AUDIO_CHANNELS, -1).contiguous()
    if sample_rate == target_sample_rate:
        return waveform

    try:
        import torchaudio
    except ImportError as error:
        raise ImportError(
            f"Resampling a MiniMax-H3 reference soundtrack from {sample_rate} Hz to {target_sample_rate} Hz needs "
            "`torchaudio`. Pass a waveform already at the audio VAE's sample rate to do without it."
        ) from error
    return torchaudio.transforms.Resample(sample_rate, target_sample_rate)(waveform)


def build_ref2va_presentation(
    tokenizer,
    prompt: str,
    references: list[MiniMaxH3PreparedReference],
    image_token_counts: list[int],
    video_block_token_counts: list[int],
) -> tuple[list[int], list[int]]:
    r"""
    Tokenize MiniMax-H3's presentation of a `ref2va` request.

    Every reference prepends a label, in packed order and numbered per modality: `"<Picture i>: "` plus a vision block
    for an image, `"<Audio j>: "` alone for audio — a waveform never reaches the conditioner — and `"<Video k>: "`
    plus one timestamped vision block per merged frame pair for a video. A video that carries sound is labelled
    `"<Audio j>: "` *before* `"<Video k>: "`, mirroring the order its rows are packed in. The prompt follows verbatim,
    with no chat template and no special tokens.

    Args:
        tokenizer (`Qwen2TokenizerFast`): Tokenizer of the conditioner.
        prompt (`str`): The prompt, appended verbatim.
        references (`list[MiniMaxH3PreparedReference]`): The prepared references, in packed order.
        image_token_counts (`list[int]`): Number of vision tokens of every image reference's block.
        video_block_token_counts (`list[int]`): Number of vision tokens per block of every video reference.

    Returns:
        `tuple[list[int], list[int]]`: the token ids and their modality tags. A vision block is tagged `0` (video) and
        everything else `1` (text).
    """

    def text(value: str) -> tuple[list[int], list[int]]:
        token_ids = tokenizer(value, add_special_tokens=False)["input_ids"]
        return token_ids, [MINIMAX_H3_TEXT_TAG] * len(token_ids)

    def vision(pad_token: str, num_tokens: int) -> tuple[list[int], list[int]]:
        token_ids = (
            [tokenizer.convert_tokens_to_ids("<|vision_start|>")]
            + [tokenizer.convert_tokens_to_ids(pad_token)] * num_tokens
            + [tokenizer.convert_tokens_to_ids("<|vision_end|>")]
        )
        return token_ids, [MINIMAX_H3_VIDEO_TAG] * len(token_ids)

    token_ids, token_tags = [], []

    def emit(segment: tuple[list[int], list[int]]) -> None:
        token_ids.extend(segment[0])
        token_tags.extend(segment[1])

    counts = {"image": 0, "video": 0, "audio": 0}
    for reference in references:
        if reference.has_audio:
            counts["audio"] += 1
            emit(text(f"<Audio {counts['audio']}>: "))
        if reference.kind == "image":
            counts["image"] += 1
            emit(text(f"<Picture {counts['image']}>: "))
            emit(vision("<|image_pad|>", image_token_counts[counts["image"] - 1]))
        elif reference.kind == "video":
            counts["video"] += 1
            emit(text(f"<Video {counts['video']}>: "))
            for timestamp in reference.block_timestamps:
                # `"{:.1f}"` rounds half to even, so the mean of a 2 fps pair renders as "<0.2 seconds>".
                emit(text(f"<{timestamp:.1f} seconds>"))
                emit(vision("<|video_pad|>", video_block_token_counts[counts["video"] - 1]))
    emit(text(prompt))
    return token_ids, token_tags


def trim_reference_num_frames(num_frames: int) -> int:
    r"""
    Snap a reference video's frame count *down* to a `17 * n + 5` the video VAE encodes without padding.

    A reference is truncated to the target's own frame count, which already is of that form, so this only bites when
    the reference is shorter than the video being generated.

    Args:
        num_frames (`int`): The number of frames the reference carries.

    Returns:
        `int`: The number of frames to encode.
    """
    if num_frames < 1:
        raise ValueError(f"A reference video must have at least one frame, got {num_frames}.")
    return (
        max(1, (num_frames - MINIMAX_H3_LATENTS_PER_CHUNK) // MINIMAX_H3_FRAMES_PER_CHUNK)
        * MINIMAX_H3_FRAMES_PER_CHUNK
        + MINIMAX_H3_LATENTS_PER_CHUNK
    )
