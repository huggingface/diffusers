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
decoding a path is the caller's job, and [`~modular_pipelines.minimax_h3.reference_loading`] is the convenience for
doing it.
"""

from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

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
