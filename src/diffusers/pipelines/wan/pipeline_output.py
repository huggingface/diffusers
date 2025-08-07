from dataclasses import dataclass

import torch

from diffusers.utils import BaseOutput


@dataclass
class WanPipelineOutput(BaseOutput):
    r"""
    Output class for Wan pipelines.

    Args:
        frames (`torch.Tensor`, `np.ndarray`, or List[List[PIL.Image.Image]]):
            List of video outputs - It can be a nested list of length `batch_size,` with each sub-list containing
            denoised PIL image sequences of length `num_frames.` It can also be a NumPy array or Torch tensor of shape
            `(batch_size, num_frames, channels, height, width)`.
    """

    frames: torch.Tensor


@dataclass
class WanImagePipelineOutput(BaseOutput):
    r"""
    Output class for Wan text-to-image pipelines.

    Args:
        images (`torch.Tensor`, `np.ndarray`, or List[PIL.Image.Image]):
            List of image outputs - It can be a list of PIL images or a NumPy array or Torch tensor of shape
            `(batch_size, channels, height, width)`. The tensor dimensions are automatically squeezed to remove
            the temporal dimension for single-frame outputs.
    """

    images: torch.Tensor
