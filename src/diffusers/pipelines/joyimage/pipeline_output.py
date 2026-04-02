from dataclasses import dataclass

import torch

from ...utils import BaseOutput
import PIL.Image
from typing import Union, List, Tuple, Optional
import numpy as np

@dataclass
class JoyImageEditPipelineOutput(BaseOutput):
    """
    Output class for JoyImageEdit generation pipelines.
    """

    images: Union[List[PIL.Image.Image], np.ndarray]
