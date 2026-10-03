from dataclasses import dataclass

import numpy as np
import PIL.Image
import torch

from ...utils import BaseOutput


@dataclass
class AnimateDiffPipelineOutput(BaseOutput):
    r"""
     Output class for AnimateDiff pipelines.

    Args:
         frames (`torch.Tensor`, `np.ndarray`, or list[list[PIL.Image.Image]]):
             list of video outputs - It can be a nested list of length `batch_size,` with each sub-list containing
             denoised
     PIL image sequences of length `num_frames.` It can also be a NumPy array or Torch tensor of shape
    `(batch_size, num_frames, channels, height, width)`
    """

    frames: torch.Tensor | np.ndarray | list[list[PIL.Image.Image]]
