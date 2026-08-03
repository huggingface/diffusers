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

References reach the blocks as in-memory media, one dataclass per modality: a [`MiniMaxH3VideoReference`] is decoded
frames plus the `fps` they carry, a [`MiniMaxH3AudioReference`] a waveform plus its sample rate, and a
[`MiniMaxH3ImageReference`] a single image. No block of this model opens a media file; decoding a path is the caller's
job, and [`~modular_pipelines.minimax_h3.reference_loading`] is the convenience for doing it.
"""

import math
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

from .packing import (
    _ROPE_FRAME_RESCALE,
    _ROPE_FRAMES_PER_LATENT,
    MINIMAX_H3_AUDIO_CHANNELS,
    MINIMAX_H3_AUDIO_TAG,
    MINIMAX_H3_FPS,
    MINIMAX_H3_TEXT_TAG,
    MINIMAX_H3_VIDEO_TAG,
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
    conditioned at the wrong speed. Decoding a file is the caller's job:
    [`~modular_pipelines.minimax_h3.decode_reference_video`] and
    [`~modular_pipelines.minimax_h3.decode_reference_audio`] do it along with the rates, and
    [`MiniMaxH3Ref2VALoadReferencesStep`] wraps them in a block.

    ```py
    >>> import numpy as np
    >>> from diffusers.utils import load_image
    >>> from diffusers.modular_pipelines.minimax_h3 import (
    ...     MiniMaxH3AudioReference,
    ...     MiniMaxH3ImageReference,
    ...     MiniMaxH3VideoReference,
    ...     decode_reference_audio,
    ...     decode_reference_video,
    ... )

    >>> references = [
    ...     MiniMaxH3ImageReference(image=load_image("subject.png")),
    ...     decode_reference_video("motion_ref.mp4"),  # frames, their `fps`, and the soundtrack
    ...     decode_reference_audio("voice.wav"),  # waveform and its `sample_rate`
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
    references: list[MiniMaxH3Reference],
    condition_latents: list[torch.Tensor],
    audio_condition_latents: list[torch.Tensor],
    num_latent_frames: int,
    latent_height: int,
    latent_width: int,
    num_audio_latents: int,
    patch_size: tuple[int, int, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    r"""
    Build the `[text | reference blocks | target audio | target video]` layout of the `ref2va` task.

    Args:
        text_token_tags (`torch.Tensor` of shape `(num_text_tokens,)`):
            The modality tag of every text row. Text is tagged `1`, except for the rows of a reference's vision block,
            which MiniMax-H3 tags `0` (video).
        references (`list[MiniMaxH3Reference]`):
            The references, in packed order. Only their modality is read here; the geometry comes from the latents.
        condition_latents (`list[torch.Tensor]`):
            One `(1, channels, num_latent_frames, latent_height, latent_width)` tensor per image and video reference,
            in packed order, as [`MiniMaxH3Ref2VAReferenceEncoderStep`] produced them.
        audio_condition_latents (`list[torch.Tensor]`):
            One `(num_audio_latents * 2, audio_latent_channels)` tensor per audio-bearing reference, in packed order.
        num_latent_frames (`int`): Number of target latent frames.
        latent_height (`int`): Target latent height.
        latent_width (`int`): Target latent width.
        num_audio_latents (`int`): Number of target audio latents per channel.
        patch_size (`tuple[int, int, int]`): The transformer's `(t, h, w)` patch.

    Returns:
        `tuple`: `position_ids`, `token_tags`, `video_indices`, `audio_indices`, `text_indices`, and the number of
        leading video and audio rows that are references rather than generated.
    """
    _, patch_h, patch_w = patch_size
    num_text_tokens = text_token_tags.shape[0]
    num_target_video_rows = num_latent_frames * (latent_height // patch_h) * (latent_width // patch_w)
    num_target_audio_rows = num_audio_latents * MINIMAX_H3_AUDIO_CHANNELS

    # The geometry of every reference block is the shape of what the encoder produced for it, so the two can never
    # disagree. Both lists are in packed order but skip the references they do not apply to, so they are consumed as
    # iterators alongside the reference list rather than indexed by it.
    visual_geometry = iter(tuple(latents.shape[2:5]) for latents in condition_latents)
    audio_row_counts = iter(rows.shape[0] for rows in audio_condition_latents)
    num_reference_video_rows = sum(
        frames * (height // patch_h) * (width // patch_w)
        for frames, height, width in (tuple(latents.shape[2:5]) for latents in condition_latents)
    )
    num_reference_audio_rows = sum(rows.shape[0] for rows in audio_condition_latents)
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
            num_latent_frames_, reference_height, reference_width = next(visual_geometry)
            num_video_rows = (
                num_latent_frames_ * (reference_height // patch_h) * (reference_width // patch_w)
            )
            rows = slice(cursor, cursor + num_video_rows)
            cursor = rows.stop
            video_indices.append(torch.arange(rows.start, rows.stop))
            frame_grid, _ = _frame_position_grid(reference_height, reference_width, patch_h, patch_w)
            position_ids[rows, 0] = rotary_time
            position_ids[rows, 1:] = frame_grid
            # An image is a single frame and takes a single integer rotary slot, not a latent frame's 5/3 units.
            rotary_time += 1.0
        elif reference.kind == "audio":
            num_audio_rows = next(audio_row_counts)
            reference_audio_latents = num_audio_rows // MINIMAX_H3_AUDIO_CHANNELS
            rows = slice(cursor, cursor + num_audio_rows)
            cursor = rows.stop
            audio_indices.append(torch.arange(rows.start, rows.stop))
            _fill_audio_positions(position_ids, rows, reference_audio_latents, rotary_time, target_width_grid)
            rotary_time += float(reference_audio_latents)
        elif reference.kind == "video":
            # A video reference's soundtrack rows are packed immediately before its video rows and share their
            # origin, so the two are rotary-aligned exactly as the generated audio and video are.
            num_audio_rows = next(audio_row_counts) if reference.has_audio else 0
            reference_audio_latents = num_audio_rows // MINIMAX_H3_AUDIO_CHANNELS
            num_latent_frames_, reference_height, reference_width = next(visual_geometry)
            num_video_rows = (
                num_latent_frames_ * (reference_height // patch_h) * (reference_width // patch_w)
            )
            audio_rows = slice(cursor, cursor + num_audio_rows)
            video_rows = slice(audio_rows.stop, audio_rows.stop + num_video_rows)
            cursor = video_rows.stop
            audio_indices.append(torch.arange(audio_rows.start, audio_rows.stop))
            video_indices.append(torch.arange(video_rows.start, video_rows.stop))

            frame_grid, width_grid = _frame_position_grid(reference_height, reference_width, patch_h, patch_w)
            _fill_audio_positions(position_ids, audio_rows, reference_audio_latents, rotary_time, width_grid)
            frame_time = _temporal_position_grid(num_latent_frames_, rotary_time)
            position_ids[video_rows, 0] = frame_time.repeat_interleave(frame_grid.shape[0])
            position_ids[video_rows, 1:] = frame_grid.repeat(num_latent_frames_, 1)
            rotary_time += max(float(reference_audio_latents), _temporal_position_span(num_latent_frames_))
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

    return (
        position_ids,
        token_tags,
        video_indices,
        audio_indices,
        text_indices,
        num_reference_video_rows,
        num_reference_audio_rows,
    )


def resolve_reference_image_size(width: int, height: int, canvas_multiple: int) -> tuple[int, int]:
    r"""
    Resolve the resolution a reference image is encoded at: a 2048 pixel short edge, both axes rounded to
    `canvas_multiple`. Upscaling is intended, and unlike the target canvas there is no area cap.

    Args:
        width (`int`): Width of the source image.
        height (`int`): Height of the source image.
        canvas_multiple (`int`):
            What both axes round to, i.e. `components.canvas_multiple` — 32 for the released checkpoint.

    Returns:
        `tuple[int, int]`: the `(height, width)` the reference is resized to.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"A reference image must have a positive size, got {width}x{height}.")
    if width > 4 * height or height > 4 * width:
        raise ValueError(f"A reference image must be within 1:4 and 4:1, got {width}x{height}.")

    scale = MINIMAX_H3_REFERENCE_IMAGE_SHORT_EDGE / min(width, height)
    multiple = canvas_multiple
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


def prepare_reference_frames(frames: np.ndarray, num_frames: int, canvas_multiple: int) -> np.ndarray:
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
        canvas_multiple (`int`):
            What both axes of its own canvas round to, i.e. `components.canvas_multiple`.

    Returns:
        `np.ndarray` of shape `(num_frames, height, width, 3)`: The prepared reference.
    """
    if frames.ndim != 4 or frames.shape[3] != 3:
        raise ValueError(
            f"A reference video must be `(num_frames, height, width, 3)` RGB frames, got {tuple(frames.shape)}."
        )
    frames = frames[:num_frames]
    height, width = resolve_canvas_size(frames.shape[2], frames.shape[1], canvas_multiple)
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
    references: list[MiniMaxH3Reference],
    image_token_counts: list[int],
    video_block_token_counts: list[int],
    video_block_timestamps: list[list[float]],
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
        references (`list[MiniMaxH3Reference]`): The normalized references, in packed order.
        image_token_counts (`list[int]`): Number of vision tokens of every image reference's block.
        video_block_token_counts (`list[int]`): Number of vision tokens per block of every video reference.
        video_block_timestamps (`list[list[float]]`): The timestamp of every vision block, per video reference.

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
            for timestamp in video_block_timestamps[counts["video"] - 1]:
                # `"{:.1f}"` rounds half to even, so the mean of a 2 fps pair renders as "<0.2 seconds>".
                emit(text(f"<{timestamp:.1f} seconds>"))
                emit(vision("<|video_pad|>", video_block_token_counts[counts["video"] - 1]))
    emit(text(prompt))
    return token_ids, token_tags


def trim_reference_num_frames(num_frames: int, frames_per_chunk: int, latents_per_chunk: int) -> int:
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
    return max(1, (num_frames - latents_per_chunk) // frames_per_chunk) * frames_per_chunk + latents_per_chunk
