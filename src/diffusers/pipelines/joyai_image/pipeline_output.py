from dataclasses import dataclass
from typing import Union

import numpy as np
from PIL import Image

from diffusers.utils import BaseOutput


@dataclass
class JoyAIImagePipelineOutput(BaseOutput):
    images: Union[Image.Image, np.ndarray]


__all__ = ["JoyAIImagePipelineOutput"]
