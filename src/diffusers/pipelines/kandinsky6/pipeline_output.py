"""Output containers for the Kandinsky 6 Diffusers pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from ...utils import BaseOutput

Audio = list[np.ndarray] | np.ndarray | None


@dataclass
class Kandinsky6TI2VAPipelineOutput(BaseOutput):
    """Output of the K6 video-and-audio pipeline."""

    frames: Any
    audio: list[np.ndarray] | None


@dataclass
class Kandinsky6SRPipelineOutput(BaseOutput):
    """Output of the K6 super-resolution pipeline."""

    frames: Any
    audio: Audio = None
    path: str | list[str] | None = None
    metadata: dict[str, Any] | None = None


__all__ = [
    "Audio",
    "Kandinsky6SRPipelineOutput",
    "Kandinsky6TI2VAPipelineOutput",
]
