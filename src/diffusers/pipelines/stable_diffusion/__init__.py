from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np

import PIL
from PIL import Image

from ...utils import (
    BaseOutput,
    is_flax_available,
    is_onnx_available,
    is_torch_available,
    is_transformers_available,
    is_transformers_version,
)


@dataclass
class StableDiffusionPipelineOutput(BaseOutput):
    """
    Output class for Stable Diffusion pipelines.

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
    from .pipeline_cycle_diffusion import CycleDiffusionPipeline
    from .pipeline_stable_diffusion import StableDiffusionPipeline
    from .pipeline_stable_diffusion_img2img import StableDiffusionImg2ImgPipeline
    from .pipeline_stable_diffusion_inpaint import StableDiffusionInpaintPipeline
    from .pipeline_stable_diffusion_inpaint_legacy import StableDiffusionInpaintPipelineLegacy
    from .safety_checker import StableDiffusionSafetyChecker

if is_transformers_available() and is_torch_available() and is_transformers_version(">=", "4.25.0.dev0"):
    from .pipeline_stable_diffusion_image_variation import StableDiffusionImageVariationPipeline
else:
    from ...utils.dummy_torch_and_transformers_objects import StableDiffusionImageVariationPipeline

if is_transformers_available() and is_onnx_available():
    from .pipeline_onnx_stable_diffusion import OnnxStableDiffusionPipeline, StableDiffusionOnnxPipeline
    from .pipeline_onnx_stable_diffusion_img2img import OnnxStableDiffusionImg2ImgPipeline
    from .pipeline_onnx_stable_diffusion_inpaint import OnnxStableDiffusionInpaintPipeline
    from .pipeline_onnx_stable_diffusion_inpaint_legacy import OnnxStableDiffusionInpaintPipelineLegacy

if is_transformers_available() and is_flax_available():
    import flax

    @flax.struct.dataclass
    class FlaxStableDiffusionPipelineOutput(BaseOutput):
        """
        Output class for Stable Diffusion pipelines.

        Args:
            images (`List[PIL.Image.Image]` or `np.ndarray`)
                List of denoised PIL images of length `batch_size` or numpy array of shape `(batch_size, height, width,
                num_channels)`. PIL images or numpy array present the denoised images of the diffusion pipeline.
            nsfw_content_detected (`List[bool]`)
                List of flags denoting whether the corresponding generated image likely represents "not-safe-for-work"
                (nsfw) content.
        """

        images: Union[List[PIL.Image.Image], np.ndarray]
        nsfw_content_detected: List[bool]

    from ...schedulers.scheduling_pndm_flax import PNDMSchedulerState
    from .pipeline_flax_stable_diffusion import FlaxStableDiffusionPipeline
    from .safety_checker_flax import FlaxStableDiffusionSafetyChecker
