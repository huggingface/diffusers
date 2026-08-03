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
Reading `ref2va` references off the filesystem.

MiniMax-H3's blocks take references as in-memory media, and the three reference dataclasses take nothing else: a
[`MiniMaxH3VideoReference`] carries decoded frames and the `fps` they were decoded at, a [`MiniMaxH3AudioReference`] a
waveform and its sample rate. Turning a path into one of those is the caller's job, the same way it is for every other
pipeline in the library.

This module is the convenience for doing it: [`decode_reference_video`] and [`decode_reference_audio`] decode a file
or a URL along with the rates it carries, and [`MiniMaxH3Ref2VALoadReferencesStep`] wraps them in a block that can be
put in front of [`MiniMaxH3Blocks`]. It is deliberately *not* one of that blockset's own steps — decoding media
is not part of generating video, and a request that already holds its media should not pay for the machinery.

The rates are the reason to prefer these over a bare `load_video`: MiniMax-H3 resamples a reference onto its own 24
fps, so a video whose real rate is lost on the way in is conditioned on at the wrong speed, silently.
"""

import contextlib
import os
import tempfile
from urllib.parse import unquote, urlparse

import numpy as np
import requests
import torch

from ...utils import is_av_available, load_image, logging
from ...utils.constants import DIFFUSERS_REQUEST_TIMEOUT
from ..modular_pipeline import ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import InputParam, OutputParam
from .modular_pipeline import MiniMaxH3ModularPipeline
from .references import (
    MiniMaxH3AudioReference,
    MiniMaxH3ImageReference,
    MiniMaxH3Reference,
    MiniMaxH3VideoReference,
)


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


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
    # mono soundtrack is upmixed later, by `prepare_reference_waveform`.
    resampler = av.audio.resampler.AudioResampler(format="fltp", layout=stream.layout, rate=sample_rate)
    chunks = []
    for frame in container.decode(stream):
        chunks += [torch.from_numpy(resampled.to_ndarray()) for resampled in resampler.resample(frame)]
    # Whatever the resampler is still holding.
    chunks += [torch.from_numpy(resampled.to_ndarray()) for resampled in resampler.resample(None)]
    return torch.cat(chunks, dim=-1).to(torch.float32), sample_rate


def decode_reference_video(media) -> MiniMaxH3VideoReference:
    r"""
    Decode a video file into a [`MiniMaxH3VideoReference`], at the resolution, the frame rate and the soundtrack it
    carries.

    The rates land on the reference, which is the point of decoding this way rather than with [`~utils.load_video`]:
    MiniMax-H3 resamples a reference onto its own 24 fps, so a frame rate lost on the way in is a request conditioned
    at the wrong speed, with nothing to raise about it. A container whose metadata is wrong is corrected by overriding
    `fps` or `sample_rate` on the returned reference.

    Needs [PyAV](https://github.com/PyAV-Org/PyAV).

    Args:
        media (`str` or `os.PathLike`): Path or URL of the video.

    Returns:
        [`MiniMaxH3VideoReference`]: the `(num_frames, height, width, 3)` `uint8` frames at the frame rate the
        container reports, carrying its soundtrack and that soundtrack's own sample rate when it has an audio stream.
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
    return MiniMaxH3VideoReference(frames=frames, fps=frame_rate, audio=waveform, sample_rate=sample_rate)


def decode_reference_audio(media) -> MiniMaxH3AudioReference:
    r"""
    Decode an audio file into a [`MiniMaxH3AudioReference`], at the sample rate it carries.

    Needs [PyAV](https://github.com/PyAV-Org/PyAV).

    Args:
        media (`str` or `os.PathLike`): Path or URL of the audio, or of a video whose soundtrack is taken.

    Returns:
        [`MiniMaxH3AudioReference`]: the `(channels, num_samples)` float32 waveform at the sample rate the container
        reports.
    """
    av = _import_av()
    with _local_media_file(media) as path, av.open(path) as container:
        if not container.streams.audio:
            raise ValueError(f"No audio stream to decode in {media}.")
        waveform, sample_rate = _decode_reference_soundtrack(av, container, container.streams.audio[0])
    return MiniMaxH3AudioReference(audio=waveform, sample_rate=sample_rate)


class MiniMaxH3Ref2VALoadReferencesStep(ModularPipelineBlocks):
    r"""
    Reads `ref2va` references off the filesystem, so that a request can name its media instead of decoding it.

    Not one of [`MiniMaxH3Blocks`]' own steps — put it in front of them when the request holds paths:

    ```py
    >>> from diffusers.modular_pipelines import SequentialPipelineBlocks
    >>> from diffusers.modular_pipelines.minimax_h3 import (
    ...     MiniMaxH3Blocks,
    ...     MiniMaxH3Ref2VALoadReferencesStep,
    ... )

    >>> blocks = SequentialPipelineBlocks.from_blocks_dict(
    ...     {"load_references": MiniMaxH3Ref2VALoadReferencesStep(), "generate": MiniMaxH3Blocks()}
    ... )
    >>> pipe = blocks.init_pipeline("MiniMaxAI/MiniMax-H3-Ref2VA-Diffusers")
    >>> output = pipe(
    ...     prompt="the subject walks through the frame",
    ...     reference_media=[{"image": "subject.png"}, {"video": "motion_ref.mp4"}, {"audio": "voice.wav"}],
    ... )
    ```
    """

    model_name = "minimax-h3"

    @property
    def description(self) -> str:
        return (
            "Decodes `ref2va` references from paths or URLs, along with the rates they carry — a video's frame rate "
            "and its soundtrack, an audio clip's sample rate. A convenience in front of the `ref2va` blocks, which "
            "themselves only ever see in-memory media."
        )

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                name="reference_media",
                type_hint=list[dict],
                required=True,
                description=(
                    "The media to condition on, **in the order the model should read them**, one `{medium: path}` "
                    "mapping each: `{'image': ...}`, `{'video': ...}` or `{'audio': ...}`, where the value is a path "
                    "or a URL. The medium is named rather than guessed from the file. A video brings its soundtrack "
                    "along, so conditioning on a file's motion alone means building a "
                    "[`MiniMaxH3VideoReference`] without one."
                ),
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "references",
                type_hint=list[MiniMaxH3Reference],
                description="The decoded references, in packed order, carrying the rates their containers report.",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components: MiniMaxH3ModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        references = []
        for index, entry in enumerate(block_state.reference_media):
            if not isinstance(entry, dict) or len(entry) != 1 or next(iter(entry)) not in ("image", "video", "audio"):
                raise ValueError(
                    f"`reference_media[{index}]` must name exactly one medium, as `{{'image': path}}`, "
                    f"`{{'video': path}}` or `{{'audio': path}}`, got {entry}."
                )
            [(medium, media)] = entry.items()
            if medium == "image":
                references.append(MiniMaxH3ImageReference(image=load_image(str(media))))
            elif medium == "video":
                references.append(decode_reference_video(media))
            else:
                references.append(decode_reference_audio(media))
        block_state.references = references

        self.set_block_state(state, block_state)
        return components, state
