# Copyright 2022 Google Brain and The HuggingFace Team. All rights reserved.
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

# TODO(Patrick, Anton, Suraj) - make scheduler framework indepedent and clean-up a bit
from typing import Union

import numpy as np
import torch

from ..configuration_utils import ConfigMixin, register_to_config
from .scheduling_utils import SchedulerMixin


class ScoreSdeVeScheduler(SchedulerMixin, ConfigMixin):
    """
    The variance exploding stochastic differential equation (SDE) scheduler.

    :param snr: coefficient weighting the step from the model_output sample (from the network) to the random noise.
    :param sigma_min: initial noise scale for sigma sequence in sampling procedure. The minimum sigma should mirror the
            distribution of the data.
    :param sigma_max: :param sampling_eps: the end value of sampling, where timesteps decrease progessively from 1 to
    epsilon. :param correct_steps: number of correction steps performed on a produced sample. :param tensor_format:
    "np" or "pt" for the expected format of samples passed to the Scheduler.
    """

    @register_to_config
    def __init__(
        self,
        num_train_timesteps=2000,
        snr=0.15,
        sigma_min=0.01,
        sigma_max=1348,
        sampling_eps=1e-5,
        correct_steps=1,
        tensor_format="pt",
    ):
        # self.sigmas = None
        # self.discrete_sigmas = None
        #
        # # setable values
        # self.num_inference_steps = None
        self.timesteps = None

        self.set_sigmas(num_train_timesteps, sigma_min, sigma_max, sampling_eps)

        self.tensor_format = tensor_format
        self.set_format(tensor_format=tensor_format)

    def set_timesteps(self, num_inference_steps, sampling_eps=None):
        sampling_eps = sampling_eps if sampling_eps is not None else self.config.sampling_eps
        tensor_format = getattr(self, "tensor_format", "pt")
        if tensor_format == "np":
            self.timesteps = np.linspace(1, sampling_eps, num_inference_steps)
        elif tensor_format == "pt":
            self.timesteps = torch.linspace(1, sampling_eps, num_inference_steps)
        else:
            raise ValueError(f"`self.tensor_format`: {self.tensor_format} is not valid.")

    def set_sigmas(self, num_inference_steps, sigma_min=None, sigma_max=None, sampling_eps=None):
        sigma_min = sigma_min if sigma_min is not None else self.config.sigma_min
        sigma_max = sigma_max if sigma_max is not None else self.config.sigma_max
        sampling_eps = sampling_eps if sampling_eps is not None else self.config.sampling_eps
        if self.timesteps is None:
            self.set_timesteps(num_inference_steps, sampling_eps)

        tensor_format = getattr(self, "tensor_format", "pt")
        if tensor_format == "np":
            self.discrete_sigmas = np.exp(np.linspace(np.log(sigma_min), np.log(sigma_max), num_inference_steps))
            self.sigmas = np.array([sigma_min * (sigma_max / sigma_min) ** t for t in self.timesteps])
        elif tensor_format == "pt":
            self.discrete_sigmas = torch.exp(torch.linspace(np.log(sigma_min), np.log(sigma_max), num_inference_steps))
            self.sigmas = torch.tensor([sigma_min * (sigma_max / sigma_min) ** t for t in self.timesteps])
        else:
            raise ValueError(f"`self.tensor_format`: {self.tensor_format} is not valid.")

    def get_adjacent_sigma(self, timesteps, t):
        tensor_format = getattr(self, "tensor_format", "pt")
        if tensor_format == "np":
            return np.where(timesteps == 0, np.zeros_like(t), self.discrete_sigmas[timesteps - 1])
        elif tensor_format == "pt":
            return torch.where(
                timesteps == 0, torch.zeros_like(t), self.discrete_sigmas[timesteps - 1].to(timesteps.device)
            )

        raise ValueError(f"`self.tensor_format`: {self.tensor_format} is not valid.")

    def set_seed(self, seed):
        tensor_format = getattr(self, "tensor_format", "pt")
        if tensor_format == "np":
            np.random.seed(seed)
        elif tensor_format == "pt":
            torch.manual_seed(seed)
        else:
            raise ValueError(f"`self.tensor_format`: {self.tensor_format} is not valid.")

    def step_pred(
        self,
        model_output: Union[torch.FloatTensor, np.ndarray],
        timestep: int,
        sample: Union[torch.FloatTensor, np.ndarray],
        seed=None,
    ):
        """
        Predict the sample at the previous timestep by reversing the SDE.
        """
        if seed is not None:
            self.set_seed(seed)
        # TODO(Patrick) non-PyTorch

        if self.timesteps is None:
            raise ValueError(
                "`self.timesteps` is not set, you need to run 'set_timesteps' after creating the scheduler"
            )

        timestep = timestep * torch.ones(
            sample.shape[0], device=sample.device
        )  # torch.repeat_interleave(timestep, sample.shape[0])
        timesteps = (timestep * (len(self.timesteps) - 1)).long()

        sigma = self.discrete_sigmas[timesteps].to(sample.device)
        adjacent_sigma = self.get_adjacent_sigma(timesteps, timestep)
        drift = self.zeros_like(sample)
        diffusion = (sigma**2 - adjacent_sigma**2) ** 0.5

        # equation 6 in the paper: the model_output modeled by the network is grad_x log pt(x)
        # also equation 47 shows the analog from SDE models to ancestral sampling methods
        drift = drift - diffusion[:, None, None, None] ** 2 * model_output

        #  equation 6: sample noise for the diffusion term of
        noise = self.randn_like(sample)
        prev_sample_mean = sample - drift  # subtract because `dt` is a small negative timestep
        # TODO is the variable diffusion the correct scaling term for the noise?
        prev_sample = prev_sample_mean + diffusion[:, None, None, None] * noise  # add impact of diffusion field g

        return {"prev_sample": prev_sample, "prev_sample_mean": prev_sample_mean}

    def step_correct(
        self,
        model_output: Union[torch.FloatTensor, np.ndarray],
        sample: Union[torch.FloatTensor, np.ndarray],
        seed=None,
    ):
        """
        Correct the predicted sample based on the output model_output of the network. This is often run repeatedly
        after making the prediction for the previous timestep.
        """
        if seed is not None:
            self.set_seed(seed)

        if self.timesteps is None:
            raise ValueError(
                "`self.timesteps` is not set, you need to run 'set_timesteps' after creating the scheduler"
            )

        # For small batch sizes, the paper "suggest replacing norm(z) with sqrt(d), where d is the dim. of z"
        # sample noise for correction
        noise = self.randn_like(sample)

        # compute step size from the model_output, the noise, and the snr
        grad_norm = self.norm(model_output)
        noise_norm = self.norm(noise)
        step_size = (self.config.snr * noise_norm / grad_norm) ** 2 * 2
        step_size = step_size * torch.ones(sample.shape[0]).to(sample.device)
        # self.repeat_scalar(step_size, sample.shape[0])

        # compute corrected sample: model_output term and noise term
        prev_sample_mean = sample + step_size[:, None, None, None] * model_output
        prev_sample = prev_sample_mean + ((step_size * 2) ** 0.5)[:, None, None, None] * noise

        return {"prev_sample": prev_sample}

    def __len__(self):
        return self.config.num_train_timesteps
