from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np

import PIL
from PIL import Image

from ...utils import (
    BaseOutput,
    OptionalDependencyNotAvailable,
    is_k_diffusion_version,
    is_onnx_available,
    is_torch_available,
    is_transformers_available,
    is_transformers_version,
)


@dataclass
class MultiDiffusionText2PanoramaPipelineOutput(BaseOutput):
    """
    Output class for Multi Diffusion Text2Panorama pipelines.

    Args:
        images (`List[PIL.Image.Image]` or `np.ndarray`)
            List of denoised PIL images of length `batch_size` or numpy array of shape `(batch_size, height, width,
            num_channels)`. PIL images or numpy array present the denoised images of the diffusion pipeline.
        nsfw_content_detected (`List[bool]`)
            List of flags denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, or `None` if safety checking could not be performed.
    """

    images: Union[List[PIL.Image.Image], np.ndarray]
    nsfw_content_detected: Optional[List[bool]]


if is_transformers_available() and is_torch_available():
    from .pipeline_multidiffusion_text2panorama import MultiDiffusionText2PanoramaPipeline
    from .safety_checker import StableDiffusionSafetyChecker
