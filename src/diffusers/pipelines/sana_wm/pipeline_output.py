from dataclasses import dataclass

import numpy as np
import PIL.Image
import torch

from ...utils import BaseOutput


@dataclass
class SanaWMPipelineOutput(BaseOutput):
    """
    Output class for SANA-WM image-to-video pipeline.

    Args:
        frames (`torch.Tensor`, `np.ndarray`, or `list[list[PIL.Image.Image]]`):
            Generated video as a list of frame batches per prompt. Shape ``(B, T, H, W, C)`` when
            returned as tensor / numpy array; uint8 frames.
        c2w (`np.ndarray`):
            Camera-to-world poses ``(T, 4, 4)`` aligned with ``frames`` (the refiner drops the
            sink anchor frame; this array is realigned accordingly when the refiner ran).
        latent (`torch.Tensor`, optional):
            Latent tensor in LTX-2 VAE space, shape ``(B, C, T_lat, H_lat, W_lat)``. Returned
            when ``output_type="latent"``.
    """

    frames: torch.Tensor | np.ndarray | list[list[PIL.Image.Image]]
    c2w: np.ndarray | None = None
    latent: torch.Tensor | None = None
