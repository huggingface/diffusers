from dataclasses import dataclass

import torch

from diffusers.utils import BaseOutput


@dataclass
class ChronoEditPipelineOutput(BaseOutput):
    r"""
    Output class for ChronoEdit pipelines.

    Args:
        frames (`torch.Tensor`, `np.ndarray`, or List[List[PIL.Image.Image]]):
            List of video outputs - It can be a nested list of length `batch_size,` with each sub-list containing
            denoised PIL image sequences of length `num_frames.` It can also be a NumPy array or Torch tensor of shape
            `(batch_size, num_frames, channels, height, width)`.
    """

    frames: torch.Tensor
