# Copyright 2024 Google Brain and The HuggingFace Team. All rights reserved.
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

# DISCLAIMER: This file is strongly influenced by https://github.com/yang-song/score_sde_pytorch

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch

from ..configuration_utils import ConfigMixin, register_to_config
from ..utils import BaseOutput
from ..utils.torch_utils import randn_tensor
from .scheduling_utils import SchedulerMixin, SchedulerOutput


@dataclass
class SdeVeOutput(BaseOutput):
    """
    Output class for the scheduler's `step` function output.

    Args:
        prev_sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample `(x_{t-1})` of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
        prev_sample_mean (`torch.Tensor` of shape `(batch_size, num_channels, height, width)` for images):
            Mean averaged `prev_sample` over previous timesteps.
    """

    prev_sample: torch.Tensor
    prev_sample_mean: torch.Tensor


class ScoreSdeVeScheduler(SchedulerMixin, ConfigMixin):
    """
    `ScoreSdeVeScheduler` is a variance exploding stochastic differential equation (SDE) scheduler.

    This model inherits from [`SchedulerMixin`] and [`ConfigMixin`]. Check the superclass documentation for the generic
    methods the library implements for all schedulers such as loading and saving.

    Args:
        num_train_timesteps (`int`, defaults to 1000):
            The number of diffusion steps to train the model.
        snr (`float`, defaults to 0.15):
            A coefficient weighting the step from the `model_output` sample (from the network) to the random noise.
        sigma_min (`float`, defaults to 0.01):
            The initial noise scale for the sigma sequence in the sampling procedure. The minimum sigma should mirror
            the distribution of the data.
        sigma_max (`float`, defaults to 1348.0):
            The maximum value used for the range of continuous timesteps passed into the model.
        sampling_eps (`float`, defaults to 1e-5):
            The end value of sampling where timesteps decrease progressively from 1 to epsilon.
        correct_steps (`int`, defaults to 1):
            The number of correction steps performed on a produced sample.
    """

    order = 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 2000,
        snr: float = 0.15,
        sigma_min: float = 0.01,
        sigma_max: float = 1348.0,
        sampling_eps: float = 1e-5,
        correct_steps: int = 1,
    ):
        # standard deviation of the initial noise distribution
        self.init_noise_sigma = sigma_max

        # setable values
        self.timesteps = None

        self.set_sigmas(num_train_timesteps, sigma_min, sigma_max, sampling_eps)

    def scale_model_input(self, sample: torch.Tensor, timestep: Optional[int] = None) -> torch.Tensor:
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

    def set_timesteps(
        self, num_inference_steps: int, sampling_eps: float = None, device: Union[str, torch.device] = None
    ):
        """
        Sets the continuous timesteps used for the diffusion chain (to be run before inference).

        Args:
            num_inference_steps (`int`):
                The number of diffusion steps used when generating samples with a pre-trained model.
            sampling_eps (`float`, *optional*):
                The final timestep value (overrides value given during scheduler instantiation).
            device (`str` or `torch.device`, *optional*):
                The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.

        """
        sampling_eps = sampling_eps if sampling_eps is not None else self.config.sampling_eps

        self.timesteps = torch.linspace(1, sampling_eps, num_inference_steps, device=device)

    def set_sigmas(
        self, num_inference_steps: int, sigma_min: float = None, sigma_max: float = None, sampling_eps: float = None
    ):
        """
        Sets the noise scales used for the diffusion chain (to be run before inference). The sigmas control the weight
        of the `drift` and `diffusion` components of the sample update.

        Args:
            num_inference_steps (`int`):
                The number of diffusion steps used when generating samples with a pre-trained model.
            sigma_min (`float`, optional):
                The initial noise scale value (overrides value given during scheduler instantiation).
            sigma_max (`float`, optional):
                The final noise scale value (overrides value given during scheduler instantiation).
            sampling_eps (`float`, optional):
                The final timestep value (overrides value given during scheduler instantiation).

        """
        sigma_min = sigma_min if sigma_min is not None else self.config.sigma_min
        sigma_max = sigma_max if sigma_max is not None else self.config.sigma_max
        sampling_eps = sampling_eps if sampling_eps is not None else self.config.sampling_eps
        if self.timesteps is None:
            self.set_timesteps(num_inference_steps, sampling_eps)

        self.sigmas = sigma_min * (sigma_max / sigma_min) ** (self.timesteps / sampling_eps)
        self.discrete_sigmas = torch.exp(torch.linspace(math.log(sigma_min), math.log(sigma_max), num_inference_steps))
        self.sigmas = torch.tensor([sigma_min * (sigma_max / sigma_min) ** t for t in self.timesteps])

    def get_adjacent_sigma(self, timesteps, t):
        return torch.where(
            timesteps == 0,
            torch.zeros_like(t.to(timesteps.device)),
            self.discrete_sigmas[timesteps - 1].to(timesteps.device),
        )

    def step_pred(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ) -> Union[SdeVeOutput, Tuple]:
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.Tensor`):
                The direct output from learned diffusion model.
            timestep (`int`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.Tensor`):
                A current instance of a sample created by the diffusion process.
            generator (`torch.Generator`, *optional*):
                A random number generator.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~schedulers.scheduling_sde_ve.SdeVeOutput`] or `tuple`.

        Returns:
            [`~schedulers.scheduling_sde_ve.SdeVeOutput`] or `tuple`:
                If return_dict is `True`, [`~schedulers.scheduling_sde_ve.SdeVeOutput`] is returned, otherwise a tuple
                is returned where the first element is the sample tensor.

        """
        if self.timesteps is None:
            raise ValueError(
                "`self.timesteps` is not set, you need to run 'set_timesteps' after creating the scheduler"
            )

        timestep = timestep * torch.ones(
            sample.shape[0], device=sample.device
        )  # torch.repeat_interleave(timestep, sample.shape[0])
        timesteps = (timestep * (len(self.timesteps) - 1)).long()

        # mps requires indices to be in the same device, so we use cpu as is the default with cuda
        timesteps = timesteps.to(self.discrete_sigmas.device)

        sigma = self.discrete_sigmas[timesteps].to(sample.device)
        adjacent_sigma = self.get_adjacent_sigma(timesteps, timestep).to(sample.device)
        drift = torch.zeros_like(sample)
        diffusion = (sigma**2 - adjacent_sigma**2) ** 0.5

        # equation 6 in the paper: the model_output modeled by the network is grad_x log pt(x)
        # also equation 47 shows the analog from SDE models to ancestral sampling methods
        diffusion = diffusion.flatten()
        while len(diffusion.shape) < len(sample.shape):
            diffusion = diffusion.unsqueeze(-1)
        drift = drift - diffusion**2 * model_output

        #  equation 6: sample noise for the diffusion term of
        noise = randn_tensor(
            sample.shape, layout=sample.layout, generator=generator, device=sample.device, dtype=sample.dtype
        )
        prev_sample_mean = sample - drift  # subtract because `dt` is a small negative timestep
        # TODO is the variable diffusion the correct scaling term for the noise?
        prev_sample = prev_sample_mean + diffusion * noise  # add impact of diffusion field g

        if not return_dict:
            return (prev_sample, prev_sample_mean)

        return SdeVeOutput(prev_sample=prev_sample, prev_sample_mean=prev_sample_mean)

    def step_correct(
        self,
        model_output: torch.Tensor,
        sample: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ) -> Union[SchedulerOutput, Tuple]:
        """
        Correct the predicted sample based on the `model_output` of the network. This is often run repeatedly after
        making the prediction for the previous timestep.

        Args:
            model_output (`torch.Tensor`):
                The direct output from learned diffusion model.
            sample (`torch.Tensor`):
                A current instance of a sample created by the diffusion process.
            generator (`torch.Generator`, *optional*):
                A random number generator.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~schedulers.scheduling_sde_ve.SdeVeOutput`] or `tuple`.

        Returns:
            [`~schedulers.scheduling_sde_ve.SdeVeOutput`] or `tuple`:
                If return_dict is `True`, [`~schedulers.scheduling_sde_ve.SdeVeOutput`] is returned, otherwise a tuple
                is returned where the first element is the sample tensor.

        """
        if self.timesteps is None:
            raise ValueError(
                "`self.timesteps` is not set, you need to run 'set_timesteps' after creating the scheduler"
            )

        # For small batch sizes, the paper "suggest replacing norm(z) with sqrt(d), where d is the dim. of z"
        # sample noise for correction
        noise = randn_tensor(sample.shape, layout=sample.layout, generator=generator).to(sample.device)

        # compute step size from the model_output, the noise, and the snr
        grad_norm = torch.norm(model_output.reshape(model_output.shape[0], -1), dim=-1).mean()
        noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
        step_size = (self.config.snr * noise_norm / grad_norm) ** 2 * 2
        step_size = step_size * torch.ones(sample.shape[0]).to(sample.device)
        # self.repeat_scalar(step_size, sample.shape[0])

        # compute corrected sample: model_output term and noise term
        step_size = step_size.flatten()
        while len(step_size.shape) < len(sample.shape):
            step_size = step_size.unsqueeze(-1)
        prev_sample_mean = sample + step_size * model_output
        prev_sample = prev_sample_mean + ((step_size * 2) ** 0.5) * noise

        if not return_dict:
            return (prev_sample,)

        return SchedulerOutput(prev_sample=prev_sample)

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        # Make sure sigmas and timesteps have the same device and dtype as original_samples
        timesteps = timesteps.to(original_samples.device)
        sigmas = self.discrete_sigmas.to(original_samples.device)[timesteps]
        noise = (
            noise * sigmas[:, None, None, None]
            if noise is not None
            else torch.randn_like(original_samples) * sigmas[:, None, None, None]
        )
        noisy_samples = noise + original_samples
        return noisy_samples

    def __len__(self):
        return self.config.num_train_timesteps
