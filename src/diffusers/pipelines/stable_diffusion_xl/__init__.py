from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import PIL

from ...utils import (
    BaseOutput,
    is_invisible_watermark_available,
    is_torch_available,
    is_transformers_available,
    is_flax_available,
)


@dataclass
class StableDiffusionXLPipelineOutput(BaseOutput):
    """
    Output class for Stable Diffusion pipelines.

    Args:
        images (`List[PIL.Image.Image]` or `np.ndarray`)
            List of denoised PIL images of length `batch_size` or numpy array of shape `(batch_size, height, width,
            num_channels)`. PIL images or numpy array present the denoised images of the diffusion pipeline.
    """

    images: Union[List[PIL.Image.Image], np.ndarray]


if is_transformers_available() and is_torch_available() and is_invisible_watermark_available():
    from .pipeline_stable_diffusion_xl import StableDiffusionXLPipeline
    from .pipeline_stable_diffusion_xl_img2img import StableDiffusionXLImg2ImgPipeline
    from .pipeline_stable_diffusion_xl_inpaint import StableDiffusionXLInpaintPipeline
    from .pipeline_stable_diffusion_xl_instruct_pix2pix import StableDiffusionXLInstructPix2PixPipeline


if is_transformers_available() and is_flax_available():
    import flax

    @flax.struct.dataclass
    class FlaxStableDiffusionXLPipelineOutput(BaseOutput):
        """
        Output class for Flax Stable Diffusion XL pipelines.

        Args:
            images (`np.ndarray`)
                Array of shape `(batch_size, height, width, num_channels)` with images from the diffusion pipeline.
        """
        images: np.ndarray

    from ...schedulers.scheduling_pndm_flax import PNDMSchedulerState
    from .pipeline_flax_stable_diffusion_xl import FlaxStableDiffusionXLPipeline

