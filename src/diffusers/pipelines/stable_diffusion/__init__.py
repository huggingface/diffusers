from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import PIL
from PIL import Image

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


try:
    if not (is_transformers_available() and is_torch_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ...utils.dummy_torch_and_transformers_objects import *  # noqa F403
else:
    from .pipeline_cycle_diffusion import CycleDiffusionPipeline
    from .pipeline_stable_diffusion import StableDiffusionPipeline
    from .pipeline_stable_diffusion_attend_and_excite import StableDiffusionAttendAndExcitePipeline
    from .pipeline_stable_diffusion_img2img import StableDiffusionImg2ImgPipeline
    from .pipeline_stable_diffusion_inpaint import StableDiffusionInpaintPipeline
    from .pipeline_stable_diffusion_inpaint_legacy import StableDiffusionInpaintPipelineLegacy
    from .pipeline_stable_diffusion_instruct_pix2pix import StableDiffusionInstructPix2PixPipeline
    from .pipeline_stable_diffusion_latent_upscale import StableDiffusionLatentUpscalePipeline
    from .pipeline_stable_diffusion_panorama import StableDiffusionPanoramaPipeline
    from .pipeline_stable_diffusion_sag import StableDiffusionSAGPipeline
    from .pipeline_stable_diffusion_upscale import StableDiffusionUpscalePipeline
    from .pipeline_stable_unclip import StableUnCLIPPipeline
    from .pipeline_stable_unclip_img2img import StableUnCLIPImg2ImgPipeline
    from .safety_checker import StableDiffusionSafetyChecker
    from .stable_unclip_image_normalizer import StableUnCLIPImageNormalizer

try:
    if not (is_transformers_available() and is_torch_available() and is_transformers_version(">=", "4.25.0")):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ...utils.dummy_torch_and_transformers_objects import StableDiffusionImageVariationPipeline
else:
    from .pipeline_stable_diffusion_image_variation import StableDiffusionImageVariationPipeline


try:
    if not (is_transformers_available() and is_torch_available() and is_transformers_version(">=", "4.26.0")):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ...utils.dummy_torch_and_transformers_objects import (
        StableDiffusionDepth2ImgPipeline,
        StableDiffusionPix2PixZeroPipeline,
    )
else:
    from .pipeline_stable_diffusion_depth2img import StableDiffusionDepth2ImgPipeline
    from .pipeline_stable_diffusion_pix2pix_zero import StableDiffusionPix2PixZeroPipeline


try:
    if not (
        is_torch_available()
        and is_transformers_available()
        and is_k_diffusion_available()
        and is_k_diffusion_version(">=", "0.0.12")
    ):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ...utils.dummy_torch_and_transformers_and_k_diffusion_objects import *  # noqa F403
else:
    from .pipeline_stable_diffusion_k_diffusion import StableDiffusionKDiffusionPipeline

try:
    if not (is_transformers_available() and is_onnx_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ...utils.dummy_onnx_objects import *  # noqa F403
else:
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
            images (`np.ndarray`)
                Array of shape `(batch_size, height, width, num_channels)` with images from the diffusion pipeline.
            nsfw_content_detected (`List[bool]`)
                List of flags denoting whether the corresponding generated image likely represents "not-safe-for-work"
                (nsfw) content.
        """

        images: np.ndarray
        nsfw_content_detected: List[bool]

    from ...schedulers.scheduling_pndm_flax import PNDMSchedulerState
    from .pipeline_flax_stable_diffusion import FlaxStableDiffusionPipeline
    from .pipeline_flax_stable_diffusion_img2img import FlaxStableDiffusionImg2ImgPipeline
    from .pipeline_flax_stable_diffusion_inpaint import FlaxStableDiffusionInpaintPipeline
    from .safety_checker_flax import FlaxStableDiffusionSafetyChecker
