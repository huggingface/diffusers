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
        frames (`torch.Tensor`, `np.ndarray`, or `list[PIL.Image.Image]`):
            Generated video. Shape ``(T, H, W, 3)`` as a float ``np.ndarray`` /
            ``torch.Tensor`` in ``[0, 1]`` when ``output_type="np"`` / ``"latent"``,
            or a list of ``PIL.Image`` of length ``T`` when ``output_type="pil"``.
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
