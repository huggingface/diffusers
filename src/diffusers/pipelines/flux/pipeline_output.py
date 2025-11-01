from dataclasses import dataclass

import numpy as np
import PIL.Image
import torch

from ...utils import BaseOutput


@dataclass
class FluxPipelineOutput(BaseOutput):
    """
    Output class for Flux image generation pipelines.

    Args:
        images (`list[PIL.Image.Image]` or `torch.Tensor` or `np.ndarray`)
            list of denoised PIL images of length `batch_size` or numpy array or torch tensor of shape `(batch_size,
            height, width, num_channels)`. PIL images or numpy array present the denoised images of the diffusion
            pipeline. Torch tensors can represent either the denoised images or the intermediate latents ready to be
            passed to the decoder.
    """

    images: list[PIL.Image.Image] | np.ndarray


@dataclass
class FluxPriorReduxPipelineOutput(BaseOutput):
    """
    Output class for Flux Prior Redux pipelines.

    Args:
        images (`list[PIL.Image.Image]` or `np.ndarray`)
            list of denoised PIL images of length `batch_size` or numpy array of shape `(batch_size, height, width,
            num_channels)`. PIL images or numpy array present the denoised images of the diffusion pipeline.
    """

    prompt_embeds: torch.Tensor
    pooled_prompt_embeds: torch.Tensor
