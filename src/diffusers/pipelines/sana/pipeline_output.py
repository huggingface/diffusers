from dataclasses import dataclass
from typing import List, Union

import numpy as np
import PIL.Image
import torch

from ...utils import BaseOutput


@dataclass
class SanaPipelineOutput(BaseOutput):
    """
    Output class for Sana pipelines.

    Args:
        images (`List[PIL.Image.Image]` or `np.ndarray`)
            List of denoised PIL images of length `batch_size` or numpy array of shape `(batch_size, height, width,
            num_channels)`. PIL images or numpy array present the denoised images of the diffusion pipeline.
    """

    images: Union[List[PIL.Image.Image], np.ndarray]


@dataclass
class SanaVideoPipelineOutput(BaseOutput):
    r"""
    Output class for Sana-Video pipelines.

    Args:
        frames (`torch.Tensor`, `np.ndarray`, or List[List[PIL.Image.Image]]):
            List of video outputs - It can be a nested list of length `batch_size,` with each sub-list containing
            denoised PIL image sequences of length `num_frames.` It can also be a NumPy array or Torch tensor of shape
            `(batch_size, num_frames, channels, height, width)`.
    """

    frames: torch.Tensor
