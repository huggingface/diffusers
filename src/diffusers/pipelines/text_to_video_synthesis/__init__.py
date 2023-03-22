from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np

from ...utils import BaseOutput, OptionalDependencyNotAvailable, is_torch_available, is_transformers_available


@dataclass
class TextToVideoSDPipelineOutput(BaseOutput):
    """
    Output class for text to video pipelines.

    Args:
        frames (`List[np.ndarray]`)
            List of denoised frames (essentially images) as NumPy arrays of shape `(height, width, num_channels)`.
            NumPy array present the denoised images of the diffusion pipeline. The length of the list denotes the video
            length i.e., the number of frames.
    """

    frames: List[np.ndarray]


try:
    if not (is_transformers_available() and is_torch_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ...utils.dummy_torch_and_transformers_objects import *  # noqa F403
else:
    from .pipeline_text_to_video_synth import TextToVideoSDPipeline  # noqa: F401
