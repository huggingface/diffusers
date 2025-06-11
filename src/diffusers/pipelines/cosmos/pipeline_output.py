from dataclasses import dataclass
from typing import List, Union

import numpy as np
import PIL.Image
import torch

from diffusers.utils import BaseOutput, get_logger


logger = get_logger(__name__)


@dataclass
class CosmosPipelineOutput(BaseOutput):
    r"""
    Output class for Cosmos any-to-world/video pipelines.

    Args:
        frames (`torch.Tensor`, `np.ndarray`, or List[List[PIL.Image.Image]]):
            List of video outputs - It can be a nested list of length `batch_size,` with each sub-list containing
            denoised PIL image sequences of length `num_frames.` It can also be a NumPy array or Torch tensor of shape
            `(batch_size, num_frames, channels, height, width)`.
    """

    frames: torch.Tensor


@dataclass
class CosmosImagePipelineOutput(BaseOutput):
    """
    Output class for Cosmos any-to-image pipelines.

    Args:
        images (`List[PIL.Image.Image]` or `np.ndarray`)
            List of denoised PIL images of length `batch_size` or numpy array of shape `(batch_size, height, width,
            num_channels)`. PIL images or numpy array present the denoised images of the diffusion pipeline.
    """

    images: Union[List[PIL.Image.Image], np.ndarray]
