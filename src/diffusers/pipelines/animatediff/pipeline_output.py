from dataclasses import dataclass
from typing import List, Union

import numpy as np
import PIL.Image
import torch

from ...utils import BaseOutput


@dataclass
class AnimateDiffPipelineOutput(BaseOutput):
    r"""
    Output class for AnimateDiff pipelines.

    Args:
        frames (`List[List[PIL.Image.Image]]` or `torch.Tensor` or `np.ndarray`):
            List of PIL Images of length `batch_size` or torch.Tensor or np.ndarray of shape
            `(batch_size, num_frames, height, width, num_channels)`.
    """

    frames: Union[List[List[PIL.Image.Image]], torch.Tensor, np.ndarray]
