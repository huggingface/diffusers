from dataclasses import dataclass

import numpy as np
import PIL.Image

from ...utils import BaseOutput


@dataclass
class SeFiPipelineOutput(BaseOutput):
    """
    Output class for SeFi-Image pipelines.

    Args:
        images (`list[PIL.Image.Image]` or `np.ndarray`)
            Generated images.
    """

    images: list[PIL.Image.Image] | np.ndarray
