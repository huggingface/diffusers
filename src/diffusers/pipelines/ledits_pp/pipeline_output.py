from dataclasses import dataclass
from typing import Optional

import numpy as np
import PIL.Image

from ...utils import BaseOutput


@dataclass
class LEditsPPDiffusionPipelineOutput(BaseOutput):
    """
    Output class for LEdits++ Diffusion pipelines.

    Args:
        images (`list[PIL.Image.Image]` or `np.ndarray`)
            List of denoised PIL images of length `batch_size` or NumPy array of shape `(batch_size, height, width,
            num_channels)`.
        nsfw_content_detected (`list[bool]`)
            List indicating whether the corresponding generated image contains “not-safe-for-work” (nsfw) content or
            `None` if safety checking could not be performed.
    """

    images: list[PIL.Image.Image] | np.ndarray
    nsfw_content_detected: Optional[list[bool]]


@dataclass
class LEditsPPInversionPipelineOutput(BaseOutput):
    """
    Output class for LEdits++ Diffusion pipelines.

    Args:
        input_images (`list[PIL.Image.Image]` or `np.ndarray`)
            List of the cropped and resized input images as PIL images of length `batch_size` or NumPy array of shape `
            (batch_size, height, width, num_channels)`.
        vae_reconstruction_images (`list[PIL.Image.Image]` or `np.ndarray`)
            List of VAE reconstruction of all input images as PIL images of length `batch_size` or NumPy array of shape
            ` (batch_size, height, width, num_channels)`.
    """

    images: list[PIL.Image.Image] | np.ndarray
    vae_reconstruction_images: list[PIL.Image.Image] | np.ndarray
