from dataclasses import dataclass

import numpy as np
import PIL.Image

from diffusers.utils import BaseOutput


@dataclass
class LongCatImagePipelineOutput(BaseOutput):
    """
    Output class for Stable Diffusion pipelines.

    Args:
        images (`list[PIL.Image.Image]` or `np.ndarray`)
            List of denoised PIL images of length `batch_size` or numpy array of shape `(batch_size, height, width,
            num_channels)`. PIL images or numpy array present the denoised images of the diffusion pipeline.
    """

    images: list[PIL.Image.Image, np.ndarray]
