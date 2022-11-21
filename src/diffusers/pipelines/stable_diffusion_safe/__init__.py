from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Union

import numpy as np

import PIL
from PIL import Image

from ...utils import BaseOutput, is_torch_available, is_transformers_available


@dataclass
class SafetyConfig(object):
    WEAK = {
        "sld_warmup_steps": 15,
        "sld_guidance_scale": 20,
        "sld_threshold": 0.0,
        "sld_momentum_scale": 0.0,
        "sld_mom_beta": 0.0,
    }
    MEDIUM = {
        "sld_warmup_steps": 10,
        "sld_guidance_scale": 1000,
        "sld_threshold": 0.01,
        "sld_momentum_scale": 0.3,
        "sld_mom_beta": 0.4,
    }
    STRONG = {
        "sld_warmup_steps": 7,
        "sld_guidance_scale": 2000,
        "sld_threshold": 0.025,
        "sld_momentum_scale": 0.5,
        "sld_mom_beta": 0.7,
    }
    MAX = {
        "sld_warmup_steps": 0,
        "sld_guidance_scale": 5000,
        "sld_threshold": 1.0,
        "sld_momentum_scale": 0.5,
        "sld_mom_beta": 0.7,
    }


@dataclass
class StableDiffusionSafePipelineOutput(BaseOutput):
    """
    Output class for Safe Stable Diffusion pipelines.

    Args:
        images (`List[PIL.Image.Image]` or `np.ndarray`)
            List of denoised PIL images of length `batch_size` or numpy array of shape `(batch_size, height, width,
            num_channels)`. PIL images or numpy array present the denoised images of the diffusion pipeline.
        nsfw_content_detected (`List[bool]`)
            List of flags denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, or `None` if safety checking could not be performed.
        images (`List[PIL.Image.Image]` or `np.ndarray`)
            List of denoised PIL images that were flagged by the safety checker any may contain "not-safe-for-work"
            (nsfw) content, or `None` if no safety check was performed or no images were flagged.
        applied_safety_concept (`str`)
            The safety concept that was applied for safety guidance, or `None` if safety guidance was disabled
    """

    images: Union[List[PIL.Image.Image], np.ndarray]
    nsfw_content_detected: Optional[List[bool]]
    unsafe_images: Optional[Union[List[PIL.Image.Image], np.ndarray]]
    applied_safety_concept: Optional[str]


if is_transformers_available() and is_torch_available():
    from .pipeline_stable_diffusion_safe import StableDiffusionPipelineSafe
    from .safety_checker import SafeStableDiffusionSafetyChecker
