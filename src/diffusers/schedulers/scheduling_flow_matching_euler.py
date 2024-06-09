# Copyright 2024 The HuggingFace Team. All rights reserved.
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

# DISCLAIMER: This code is strongly influenced by https://github.com/leffff/euler-scheduler

from dataclasses import dataclass
from typing import Tuple, Any, Optional, Union

import torch
from ..configuration_utils import ConfigMixin, register_to_config
from ..utils import BaseOutput
from .scheduling_utils import KarrasDiffusionSchedulers, SchedulerMixin


@dataclass
class FlowMatchingEulerSchedulerOutput(BaseOutput):
    """
    Output class for the scheduler's `step` function output.

    Args:
        prev_sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample `(x_{t-1})` of previous timestep (which in flow-matching notation should be noted as
            `(x_{t+h})`). `prev_sample` should be used as next model input in the denoising loop.
        pred_original_sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width)` for images):
            The predicted denoised sample `(x_{0})` (which in flow-matching notation should be noted as
            `(x_{1})`) based on the model output from the current timestep.
            `pred_original_sample` can be used to preview progress or for guidance.
    """

    prev_sample: torch.Tensor
    pred_original_sample: Optional[torch.Tensor] = None


def get_time_coefficients(timestep: torch.Tensor, ndim: int) -> torch.Tensor:
    return timestep.reshape((timestep.shape[0], *([1] * (ndim - 1))))


class FlowMatchingEulerScheduler(SchedulerMixin, ConfigMixin):
    """
    `FlowMatchingEulerScheduler` is a scheduler for training and inferencing Conditional Flow Matching models (CFMs).
    
    Flow Matching (FM) is a novel, simulation-free methodology for training Continuous Normalizing Flows (CNFs) by 
    regressing vector fields of predetermined conditional probability paths, facilitating scalable training and 
    efficient sample generation through the utilization of various probability paths, including Gaussian and 
    Optimal Transport (OT) paths, thereby enhancing model performance and generalization capabilities

    Args:
        num_inference_steps (`int`, defaults to 100):
            The number of steps on inference.
    """

    @register_to_config
    def __init__(self, num_inference_steps: int = 100):
        self.timesteps = None
        self.num_inference_steps = None
        self.h = None

        if num_inference_steps is not None:
            self.set_timesteps(num_inference_steps)

    @staticmethod
    def add_noise(original_samples: torch.Tensor, noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        Add noise to the given sample

        Args:
            original_samples (`torch.Tensor`):
                The original sample that is to be noised
            noise (`torch.Tensor`):
                The noise that is used to noise the image
            timestep (`torch.Tensor`):
                Timestep used to create linear interpolation `x_t = t * x_1 + (1 - t) * x_0`.
                Where x_1 is a target distribution, x_0 is a source distribution and t (timestep) ∈ [0, 1]
        """

        t = get_time_coefficients(timestep, original_samples.ndim)

        noised_sample = t * original_samples + (1 - t) * noise

        return noised_sample

    def set_timesteps(self, num_inference_steps: int = 100) -> None:
        """
        Set number of inference steps (Euler intagration steps)

        Args:
            num_inference_steps (`int`, defaults to 100):
                The number of steps on inference.
        """

        self.num_inference_steps = num_inference_steps
        self.h = 1 / num_inference_steps
        self.timesteps = torch.arange(0, 1, self.h)

    def step(self, model_output: torch.Tensor, timestep: torch.Tensor, sample: torch.Tensor,
             return_dict: bool = True) -> Union[FlowMatchingEulerSchedulerOutput, Tuple]:
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.Tensor`):
                The direct output from learned diffusion model.
            timestep (`float`):
                Timestep used to perform Euler Method `x_t = h * f(x_t, t) + x_{t-1}`.
                Where x_1 is a target distribution, x_0 is a source distribution and t (timestep) ∈ [0, 1]
            sample (`torch.Tensor`):
                A current instance of a sample created by the diffusion process.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~schedulers.scheduling_ddpm.DDPMSchedulerOutput`] or `tuple`.

        Returns:
            [`~schedulers.scheduling_ddpm.DDPMSchedulerOutput`] or `tuple`:
                If return_dict is `True`, [`~schedulers.scheduling_ddpm.DDPMSchedulerOutput`] is returned, otherwise a
                tuple is returned where the first element is the sample tensor.
        """

        step = FlowMatchingEulerSchedulerOutput(
            prev_sample=sample + self.h * model_output,
            pred_original_sample=sample + (1 - get_time_coefficients(timestep, model_output.ndim)) * model_output
        )

        if return_dict:
            return step

        return step.prev_sample,

    @staticmethod
    def get_velocity(original_samples: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            original_samples (`torch.Tensor`):
                The original sample that is to be noised
            noise (`torch.Tensor`):
                The noise that is used to noise the image

        Returns:
            `torch.Tensor`
        """

        return original_samples - noise

    @staticmethod
    def scale_model_input(sample: torch.Tensor, timestep: Optional[int] = None) -> torch.Tensor:
        """
         Ensures interchangeability with schedulers that need to scale the denoising model input depending on the
         current timestep.

        Args:
            sample (`torch.Tensor`):
                The input sample.
            timestep (`int`, *optional*):
                The current timestep in the diffusion chain.

        Returns:
            `torch.Tensor`:
                A scaled input sample.
        """

        return sample
