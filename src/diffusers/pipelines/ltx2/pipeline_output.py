from dataclasses import dataclass

import torch

from diffusers.utils import BaseOutput


@dataclass
class LTX2PipelineOutput(BaseOutput):
    r"""
    Output class for LTX pipelines.

    Args:
        frames (`torch.Tensor`, `np.ndarray`, or list[list[PIL.Image.Image]]):
            List of video outputs - It can be a nested list of length `batch_size,` with each sub-list containing
            denoised PIL image sequences of length `num_frames.` It can also be a NumPy array or Torch tensor of shape
            `(batch_size, num_frames, channels, height, width)`.
        audio (`torch.Tensor`, `np.ndarray`):
            TODO
    """

    frames: torch.Tensor
    audio: torch.Tensor


@dataclass
class LTX2DFRPipelineOutput(LTX2PipelineOutput):
    r"""
    Output class for DFR pipelines.

    Args:
        frames (`torch.Tensor`, `np.ndarray`, or list[list[PIL.Image.Image]]):
            Denoised video. Latent output is the untrimmed canvas, shape
            `(batch_size, num_channels, latent_frames, latent_height, latent_width)`.
        audio (`torch.Tensor`, `np.ndarray`):
            Accompanying audio latents or waveform.
        keyframes (`torch.Tensor`, *optional*):
            Generated or carried keyframe latents of shape `(batch_size, num_channels, num_keyframes, latent_height,
            latent_width)`. `None` when the pass did not produce slots (e.g. a tiled epilogue).
        keyframe_positions (`list[int]`, *optional*):
            Pixel-frame index of each keyframe on this pass's canvas. After a temporal round these cannot be
            re-derived from the original `num_frames` and must be passed into the next stage.
    """

    keyframes: torch.Tensor | None = None
    keyframe_positions: list[int] | None = None


@dataclass
class LTX2VideoDecodeOutput(BaseOutput):
    r"""
    Output class for the LTX-2 diffusion decode pipeline, which produces video only.

    Args:
        frames (`torch.Tensor`, `np.ndarray`, or list[list[PIL.Image.Image]]):
            List of video outputs - It can be a nested list of length `batch_size,` with each sub-list containing
            denoised PIL image sequences of length `num_frames.` It can also be a NumPy array or Torch tensor of shape
            `(batch_size, num_frames, channels, height, width)`.
    """

    frames: torch.Tensor
