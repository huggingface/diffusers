from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import PIL

from ...utils import (
    BaseOutput,
    OptionalDependencyNotAvailable,
    is_torch_available,
    is_transformers_available,
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


try:
    if not (is_transformers_available() and is_torch_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ...utils.dummy_torch_and_transformers_objects import *  # noqa F403
else:
    from .pipeline_stable_diffusion_xl import StableDiffusionXLPipeline
    from .pipeline_stable_diffusion_xl_img2img import StableDiffusionXLImg2ImgPipeline
    from .pipeline_stable_diffusion_xl_inpaint import StableDiffusionXLInpaintPipeline
    from .pipeline_stable_diffusion_xl_instruct_pix2pix import StableDiffusionXLInstructPix2PixPipeline
