from dataclasses import dataclass

import numpy as np
import PIL.Image

from ....utils import (
    BaseOutput,
)


@dataclass
# Copied from diffusers.pipelines.stable_diffusion.pipeline_output.StableDiffusionPipelineOutput with Stable->Alt
class AltDiffusionPipelineOutput(BaseOutput):
    """
    Output class for Alt Diffusion pipelines.

    Args:
        images (`list[PIL.Image.Image]` or `np.ndarray`)
            list of denoised PIL images of length `batch_size` or NumPy array of shape `(batch_size, height, width,
            num_channels)`.
        nsfw_content_detected (`list[bool]`)
            list indicating whether the corresponding generated image contains "not-safe-for-work" (nsfw) content or
            `None` if safety checking could not be performed.
    """

    images: list[PIL.Image.Image] | np.ndarray
    nsfw_content_detected: list[bool] | None
