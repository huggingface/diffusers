from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import PIL
from PIL import Image

from ...utils import OptionalDependencyNotAvailable, is_torch_available, is_transformers_available


try:
    if not (is_transformers_available() and is_torch_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ...utils.dummy_torch_and_transformers_objects import ShapEPipeline
else:
    from .blip_image_processing import BlipImageProcessor
    from .modeling_blip2 import Blip2QFormerModel
    from .modeling_ctx_clip import ContextCLIPTextModel
    from .pipeline_blip_diffusion import BlipDiffusionPipeline
