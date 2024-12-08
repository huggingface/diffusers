from dataclasses import dataclass
from typing import List, Union

import numpy as np
import PIL.Image

from ...utils import BaseOutput


@dataclass
class CogView3PipelineOutput(BaseOutput):
    """
    Output class for CogView3 pipelines.

    Args:
        images (`List[PIL.Image.Image]` or `np.ndarray`)
            List of denoised PIL images of length `batch_size` or numpy array of shape `(batch_size, height, width,
            num_channels)`. PIL images or numpy array present the denoised images of the diffusion pipeline.
    """

    images: Union[List[PIL.Image.Image], np.ndarray]
