from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import PIL
from PIL import Image

from ...utils import BaseOutput, OptionalDependencyNotAvailable, is_torch_available, is_transformers_available


@dataclass
# Copied from diffusers.pipelines.stable_diffusion.__init__.StableDiffusionPipelineOutput with Stable->Alt
class AltDiffusionPipelineOutput(BaseOutput):
    """
    Output class for Alt Diffusion pipelines.

    Args:
        images (`List[PIL.Image.Image]` or `np.ndarray`)
            List of denoised PIL images of length `batch_size` or NumPy array of shape `(batch_size, height, width,
            num_channels)`.
        nsfw_content_detected (`List[bool]`)
            List indicating whether the corresponding generated image contains "not-safe-for-work" (nsfw) content or
            `None` if safety checking could not be performed.
    """

    images: Union[List[PIL.Image.Image], np.ndarray]
    nsfw_content_detected: Optional[List[bool]]


try:
    if not (is_transformers_available() and is_torch_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ...utils.dummy_torch_and_transformers_objects import ShapEPipeline
else:
    from .modeling_roberta_series import RobertaSeriesModelWithTransformation
    from .pipeline_alt_diffusion import AltDiffusionPipeline
    from .pipeline_alt_diffusion_img2img import AltDiffusionImg2ImgPipeline
