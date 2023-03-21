from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np

from ...utils import BaseOutput, OptionalDependencyNotAvailable, is_torch_available, is_transformers_available


@dataclass
class TextToVideoMSPipelineOutput(BaseOutput):
    """
    Output class for text to video pipelines.

    Args:
        images (`List[PIL.Image.Image]` or `np.ndarray`)
            List of denoised PIL images of length `batch_size` or numpy array of shape `(batch_size, height, width,
            num_channels)`. PIL images or numpy array present the denoised images of the diffusion pipeline.
    """

    image: np.ndarray


try:
    if not (is_transformers_available() and is_torch_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ...utils.dummy_torch_and_transformers_objects import *  # noqa F403
else:
    from .pipeline_text_to_video_synth import TextToVideoMSPipeline  # noqa: F401
