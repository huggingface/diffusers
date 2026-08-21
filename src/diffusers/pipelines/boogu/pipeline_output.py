from dataclasses import dataclass
from typing import List, Union

import numpy as np
import PIL.Image

from ...utils import BaseOutput


@dataclass
class BooguImagePipelineOutput(BaseOutput):
    """
    Output class for Boogu-Image pipelines.

    Args:
        images (`List[PIL.Image.Image]` or `np.ndarray`):
            List of denoised PIL images of length `batch_size` or numpy array of shape
            `(batch_size, height, width, num_channels)`. Contains the generated images.
    """

    images: Union[List[PIL.Image.Image], np.ndarray]
