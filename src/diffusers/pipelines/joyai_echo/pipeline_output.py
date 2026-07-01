from dataclasses import dataclass

import torch

from ...utils import BaseOutput


@dataclass
class JoyAIEchoShotOutput(BaseOutput):
    r"""
    Output class for one JoyAI-Echo shot.

    Args:
        frames (`torch.Tensor`, `np.ndarray`, or list[list[PIL.Image.Image]]):
            Generated video frames for the shot.
        audio (`torch.Tensor` or `np.ndarray`):
            Generated waveform for the shot.
        latents (`torch.Tensor`, *optional*):
            Generated packed video latents before decoding.
        audio_latents (`torch.Tensor`, *optional*):
            Generated packed audio latents before decoding.
    """

    frames: torch.Tensor
    audio: torch.Tensor
    latents: torch.Tensor | None = None
    audio_latents: torch.Tensor | None = None


@dataclass
class JoyAIEchoPipelineOutput(BaseOutput):
    r"""
    Output class for JoyAI-Echo multi-shot generation.

    Args:
        frames (`list`):
            Generated video frames for each shot.
        audio (`list`):
            Generated waveform for each shot.
        shots (`list[JoyAIEchoShotOutput]`):
            Per-shot structured outputs.
    """

    frames: list
    audio: list
    shots: list[JoyAIEchoShotOutput]
