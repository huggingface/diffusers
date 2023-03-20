from dataclasses import dataclass
from typing import Union

import numpy as np
import torch

from ...utils import (
    BaseOutput,
    OptionalDependencyNotAvailable,
    is_flax_available,
    is_k_diffusion_available,
    is_k_diffusion_version,
    is_onnx_available,
    is_torch_available,
    is_transformers_available,
    is_transformers_version,
)


@dataclass
class TuneAVideoPipelineOutput(BaseOutput):
    videos: Union[torch.Tensor, np.ndarray]


if is_transformers_available() and is_torch_available():
    from .tune_a_video_pipeline import TuneAVideoPipeline
