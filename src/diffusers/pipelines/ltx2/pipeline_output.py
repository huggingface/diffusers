from dataclasses import dataclass

import torch

from diffusers.utils import BaseOutput


@dataclass
class LTX2PipelineOutput(BaseOutput):
    r"""
    Output class for LTX pipelines.

    Args:
        frames (`torch.Tensor`, `np.ndarray`, or list[list[PIL.Image.Image]]):
            List of video outputs - It can be a nested list of length `batch_size,` with each sub-list containing
            denoised PIL image sequences of length `num_frames.` It can also be a NumPy array or Torch tensor of shape
            `(batch_size, num_frames, channels, height, width)`.
        audio (`torch.Tensor`, `np.ndarray`):
            Generated audio aligned with the returned video. When returned as a tensor or NumPy array, the shape is
            `(batch_size, channels, samples)`.
    """

    frames: torch.Tensor
    audio: torch.Tensor
