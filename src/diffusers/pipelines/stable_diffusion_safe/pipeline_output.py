from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import PIL.Image

from ...utils import (
    BaseOutput,
)


@dataclass
class StableDiffusionSafePipelineOutput(BaseOutput):
    """
    Output class for Safe Stable Diffusion pipelines.

    Args:
        images (`list[PIL.Image.Image]` or `np.ndarray`)
            List of denoised PIL images of length `batch_size` or numpy array of shape `(batch_size, height, width,
            num_channels)`. PIL images or numpy array present the denoised images of the diffusion pipeline.
        nsfw_content_detected (`list[bool]`)
            List of flags denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, or `None` if safety checking could not be performed.
        images (`list[PIL.Image.Image]` or `np.ndarray`)
            List of denoised PIL images that were flagged by the safety checker any may contain "not-safe-for-work"
            (nsfw) content, or `None` if no safety check was performed or no images were flagged.
        applied_safety_concept (`str`)
            The safety concept that was applied for safety guidance, or `None` if safety guidance was disabled
    """

    images: Union[list[PIL.Image.Image], np.ndarray]
    nsfw_content_detected: Optional[list[bool]]
    unsafe_images: Optional[Union[list[PIL.Image.Image], np.ndarray]]
    applied_safety_concept: Optional[str]
