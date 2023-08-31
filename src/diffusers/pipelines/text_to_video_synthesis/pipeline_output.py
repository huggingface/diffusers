from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import torch

from ...utils import (
    _LazyModule,
    BaseOutput,
    OptionalDependencyNotAvailable,
    is_torch_available,
    is_transformers_available,
)


@dataclass
class TextToVideoSDPipelineOutput(BaseOutput):
    """
    Output class for text-to-video pipelines.

    Args:
        frames (`List[np.ndarray]` or `torch.FloatTensor`)
            List of denoised frames (essentially images) as NumPy arrays of shape `(height, width, num_channels)` or as
            a `torch` tensor. The length of the list denotes the video length (the number of frames).
    """

    frames: Union[List[np.ndarray], torch.FloatTensor]
