from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import PIL

from ...utils import (
    BaseOutput,
    is_flax_available,
    is_transformers_available,
)


@dataclass
class StableDiffusionPipelineOutput(BaseOutput):
    """
    Output class for Stable Diffusion pipelines.

    Args:
        images (`List[PIL.Image.Image]` or `np.ndarray`)
            List of denoised PIL images of length `batch_size` or NumPy array of shape `(batch_size, height, width,
            num_channels)`.
        nsfw_content_detected (`List[bool]`)
            List indicating whether the corresponding generated image contains "not-safe-for-work" (nsfw) content or
            `None` if safety checking could not be performed.
    """

    images: Union[List[PIL.Image.Image], np.ndarray]
    nsfw_content_detected: Optional[List[bool]]


if is_transformers_available() and is_flax_available():
    import flax

    @flax.struct.dataclass
    class FlaxStableDiffusionPipelineOutput(BaseOutput):
        """
        Output class for Flax-based Stable Diffusion pipelines.

        Args:
            images (`np.ndarray`):
                Denoised images of array shape of `(batch_size, height, width, num_channels)`.
            nsfw_content_detected (`List[bool]`):
                List indicating whether the corresponding generated image contains "not-safe-for-work" (nsfw) content
                or `None` if safety checking could not be performed.
        """

        images: np.ndarray
        nsfw_content_detected: List[bool]
