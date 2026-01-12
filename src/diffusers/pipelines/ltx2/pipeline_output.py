from dataclasses import dataclass

import torch

from diffusers.utils import BaseOutput


@dataclass
class LTX2PipelineOutput(BaseOutput):
    r"""
    Output class for LTX pipelines.

    Args:
        frames (`torch.Tensor`, `np.ndarray`, or List[List[PIL.Image.Image]]):
            List of video outputs - It can be a nested list of length `batch_size,` with each sub-list containing
            denoised PIL image sequences of length `num_frames.` It can also be a NumPy array or Torch tensor of shape
            `(batch_size, num_frames, channels, height, width)`.
        audio (`torch.Tensor`, `np.ndarray`):
            TODO
    """

    frames: torch.Tensor
    audio: torch.Tensor
