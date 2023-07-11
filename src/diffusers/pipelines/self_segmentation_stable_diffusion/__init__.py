from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np
import PIL

from diffusers.utils import BaseOutput


@dataclass
class SelfSegmentationStableDiffusionPipelineOutput(BaseOutput):
    """
    Output class for Stable Diffusion pipelines.

    Args:
        images (`List[PIL.Image.Image]` or `np.ndarray`)
            List of denoised PIL images of length `batch_size` or numpy array of shape `(batch_size, height, width,
            num_channels)`. PIL images or numpy array present the denoised images of the diffusion pipeline.
        nsfw_content_detected (`List[bool]`)
            List of flags denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, or `None` if safety checking could not be performed.
        seg_map (`np.ndarray`)
            Segmentation map of the input image. Each segment is represented by a unique integer.
        seg_labels (`Dict[int, Union[int, str]]`)
            Dictionary mapping each segment to its corresponding label.
        merged_seg_map (`np.ndarray`)

    """

    images: Union[List[PIL.Image.Image], np.ndarray]
    nsfw_content_detected: Optional[List[bool]]
    seg_map: List[np.ndarray]
    seg_labels: List[Dict[int, Union[int, str]]]
    merged_seg_map: List[np.ndarray]
    merged_seg_labels: List[Dict[int, Union[int, str]]]


from .self_segmentation_stable_diffusion import SelfSegmentationStableDiffusionPipeline
