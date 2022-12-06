from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Union

import numpy as np

import PIL
from PIL import Image

from ...utils import BaseOutput, is_torch_available, is_transformers_available


if is_transformers_available() and is_torch_available():
    from .image_encoder import InpaintByExampleImageEncoder
    from .pipeline_inpaint_by_example import InpaintByExamplePipeline
