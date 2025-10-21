from dataclasses import dataclass
from typing import Optional

import numpy as np
import PIL.Image

from ...utils import BaseOutput


@dataclass
class IFPipelineOutput(BaseOutput):
    r"""
    Output class for Stable Diffusion pipelines.

    Args:
        images (`list[PIL.Image.Image]` or `np.ndarray`):
            List of denoised PIL images of length `batch_size` or numpy array of shape `(batch_size, height, width,
            num_channels)`. PIL images or numpy array present the denoised images of the diffusion pipeline.
        nsfw_detected (`list[bool]`):
            List of flags denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content or a watermark. `None` if safety checking could not be performed.
        watermark_detected (`list[bool]`):
            List of flags denoting whether the corresponding generated image likely has a watermark. `None` if safety
            checking could not be performed.
    """

    images: list[PIL.Image.Image] | np.ndarray
    nsfw_detected: Optional[list[bool]]
    watermark_detected: Optional[list[bool]]
