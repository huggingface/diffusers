# Copyright 2025 The Kandinsky Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass

import numpy as np
import PIL.Image
import torch

from ...utils import BaseOutput


@dataclass
class Kandinsky6TI2VAPipelineOutput(BaseOutput):
    r"""
    Output class for [`Kandinsky6TI2VAPipeline`].

    Args:
        frames (`torch.Tensor`, `np.ndarray`, or `list[list[PIL.Image.Image]]`):
            The generated video. A nested list of length `batch_size` holding `num_frames` PIL images each, or a NumPy
            array or torch tensor of shape `(batch_size, num_frames, height, width, channels)` / `(batch_size,
            num_frames, channels, height, width)`. With `output_type="latent"`, the video latents of shape
            `(batch_size, channels, num_latent_frames, latent_height, latent_width)`.
        audio (`torch.Tensor` or `np.ndarray`, *optional*):
            The generated waveforms of shape `(batch_size, num_samples)` in `[-1, 1]` at the audio VAE's sample rate,
            or `None` when audio was not sampled. With `output_type="latent"`, the audio latents of shape `(batch_size,
            channels, audio_length)`.
    """

    frames: torch.Tensor | np.ndarray | list[list[PIL.Image.Image]]
    audio: torch.Tensor | np.ndarray | None = None


@dataclass
class Kandinsky6SRPipelineOutput(BaseOutput):
    r"""
    Output class for [`Kandinsky6SRPipeline`].

    Args:
        frames (`torch.Tensor`, `np.ndarray`, or `list[list[PIL.Image.Image]]`):
            The super-resolved video. A nested list of length `batch_size` holding `num_frames` PIL images each, or a
            NumPy array or torch tensor of shape `(batch_size, num_frames, height, width, channels)` / `(batch_size,
            num_frames, channels, height, width)`.
    """

    frames: torch.Tensor | np.ndarray | list[list[PIL.Image.Image]]
