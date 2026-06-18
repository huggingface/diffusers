from dataclasses import dataclass
from typing import List, Union

import numpy as np
import PIL.Image

from ...utils import BaseOutput


@dataclass
class JoyImageEditPipelineOutput(BaseOutput):
    """
    Output class for JoyImageEdit generation pipelines.
    """

    images: Union[List[PIL.Image.Image], np.ndarray]

@dataclass
class JoyImageEditPlusPipelineOutput(BaseOutput):
    """
    Output class for JoyImage Edit Plus multi-image editing pipelines.
    """

    images: Union[List[PIL.Image.Image], np.ndarray]