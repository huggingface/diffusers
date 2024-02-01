from dataclasses import dataclass
from typing import List, Union

import numpy as np
import PIL
import torch

from ...utils import (
    BaseOutput,
)


@dataclass
class TextToVideoSDPipelineOutput(BaseOutput):
    """
    Output class for text-to-video pipelines.

    Args:
        frames (`torch.Tensor`, `np.ndarray`, or List[List[PIL.Image.Image]]):
        Nested list of length `batch_size` with denoised PIL image sequences of length `num_frames`,
        NumPy array of shape `(batch_size, num_frames, channels, height, width,
        Torch tensor of shape `(batch_size, num_frames, channels, height, width)`.
    """

    frames: Union[torch.Tensor, np.ndarray, List[List[PIL.Image.Image]]]
