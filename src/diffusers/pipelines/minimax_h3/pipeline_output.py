from dataclasses import dataclass

import numpy as np
import PIL.Image
import torch

from ...utils import BaseOutput


@dataclass
class MiniMaxH3PipelineOutput(BaseOutput):
    r"""
    Output class for MiniMax-H3 pipelines, which always generate video and audio jointly.

    Args:
        frames (`torch.Tensor`, `np.ndarray`, or `list[list[PIL.Image.Image]]`):
            The generated video. A nested list of length `batch_size` holding `num_frames` PIL images each, or a NumPy
            array / torch tensor of shape `(batch_size, num_frames, channels, height, width)`.
        audio (`torch.Tensor`):
            The generated soundtrack, of shape `(batch_size, 2, num_samples)` at the audio VAE's sampling rate
            (32 kHz). Muxing it onto the frames is left to the caller, e.g. with
            [`~utils.export_utils.encode_video`].
    """

    frames: torch.Tensor | np.ndarray | list[list[PIL.Image.Image]]
    audio: torch.Tensor
