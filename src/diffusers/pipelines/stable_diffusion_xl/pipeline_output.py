from dataclasses import dataclass

import numpy as np
import PIL.Image

from ...utils import BaseOutput, is_flax_available


@dataclass
class StableDiffusionXLPipelineOutput(BaseOutput):
    """
    Output class for Stable Diffusion pipelines.

    Args:
        images (`list[PIL.Image.Image]` or `np.ndarray`)
            list of denoised PIL images of length `batch_size` or numpy array of shape `(batch_size, height, width,
            num_channels)`. PIL images or numpy array present the denoised images of the diffusion pipeline.
    """

    images: list[PIL.Image.Image] | np.ndarray


if is_flax_available():
    import flax

    @flax.struct.dataclass
    class FlaxStableDiffusionXLPipelineOutput(BaseOutput):
        """
        Output class for Flax Stable Diffusion XL pipelines.

        Args:
            images (`np.ndarray`)
                Array of shape `(batch_size, height, width, num_channels)` with images from the diffusion pipeline.
        """

        images: np.ndarray
