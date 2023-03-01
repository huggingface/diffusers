from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np

import PIL
from PIL import Image

from ...utils import is_torch_available, is_transformers_available


if is_transformers_available() and is_torch_available():
    from .image_encoder import PaintByExampleImageEncoder
    from .pipeline_paint_by_example import PaintByExamplePipeline
