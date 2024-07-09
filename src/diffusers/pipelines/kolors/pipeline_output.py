from dataclasses import dataclass
from typing import List, Union

import numpy as np
import PIL.Image

from ...utils import BaseOutput, is_flax_available


@dataclass
class KolorsPipelineOutput(BaseOutput):
    """
    Output class for Kolors pipelines.

    Args:
        images (`List[PIL.Image.Image]` or `np.ndarray`)
            List of denoised PIL images of length `batch_size` or numpy array of shape `(batch_size, height, width,
            num_channels)`. PIL images or numpy array present the denoised images of the diffusion pipeline.
    """

    images: Union[List[PIL.Image.Image], np.ndarray]


if is_flax_available():
    import flax

    @flax.struct.dataclass
    class FlaxStableKolorsPipelineOutput(BaseOutput):
        """
        Output class for Flax Kolors pipelines.

        Args:
            images (`np.ndarray`)
                Array of shape `(batch_size, height, width, num_channels)` with images from the diffusion pipeline.
        """

        images: np.ndarray
