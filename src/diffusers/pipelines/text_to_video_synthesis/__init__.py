from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import torch

from ...utils import BaseOutput, OptionalDependencyNotAvailable, is_torch_available, is_transformers_available


@dataclass
class TextToVideoSDPipelineOutput(BaseOutput):
    """
    Output class for text to video pipelines.

    Args:
        frames (`List[np.ndarray]` or `torch.FloatTensor`)
            List of denoised frames (essentially images) as NumPy arrays of shape `(height, width, num_channels)` or as
            a `torch` tensor. NumPy array present the denoised images of the diffusion pipeline. The length of the list
            denotes the video length i.e., the number of frames.
    """

    frames: Union[List[np.ndarray], torch.FloatTensor]


try:
    if not (is_transformers_available() and is_torch_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ...utils.dummy_torch_and_transformers_objects import *  # noqa F403
else:
    from .pipeline_text_to_video_synth import TextToVideoSDPipeline  # noqa: F401
    from .pipeline_text_to_video_zero import TextToVideoZeroPipeline
