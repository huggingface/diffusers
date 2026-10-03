from dataclasses import dataclass

import numpy as np
import PIL.Image

from ...utils import BaseOutput


@dataclass
class HunyuanImagePipelineOutput(BaseOutput):
    """
    Output class for HunyuanImage pipelines.

    Args:
        images (`list[PIL.Image.Image]` or `np.ndarray`)
            List of denoised PIL images of length `batch_size` or numpy array of shape `(batch_size, height, width,
            num_channels)`. PIL images or numpy array present the denoised images of the diffusion pipeline.
    """

    images: list[PIL.Image.Image, np.ndarray]
