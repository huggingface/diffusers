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

import numpy as np
import torch

from ..configuration_utils import ConfigMixin
from .scheduling_utils import SchedulerMixin


class ScoreSdeVeScheduler(SchedulerMixin, ConfigMixin):
    """
    The variance exploding stochastic differential equation (SDE) scheduler.

    :param snr: coefficient weighting the step from the score sample (from the network) to the random noise. :param
    sigma_min: initial noise scale for sigma sequence in sampling procedure. The minimum sigma should mirror the
            distribution of the data.
    :param sigma_max: :param sampling_eps: the end value of sampling, where timesteps decrease progessively from 1 to
    epsilon. :param correct_steps: number of correction steps performed on a produced sample. :param tensor_format:
    "np" or "pt" for the expected format of samples passed to the Scheduler.
    """

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
        super().__init__()
        self.register_to_config(
            num_train_timesteps=num_train_timesteps,
            snr=snr,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            sampling_eps=sampling_eps,
            correct_steps=correct_steps,
        )

        self.sigmas = None
        self.discrete_sigmas = None
        self.timesteps = None

        # TODO - update step to be torch-independant
        self.set_format(tensor_format=tensor_format)

    def set_timesteps(self, num_inference_steps):
        tensor_format = getattr(self, "tensor_format", "pt")
        if tensor_format == "np":
            self.timesteps = np.linspace(1, self.config.sampling_eps, num_inference_steps)
        elif tensor_format == "pt":
            self.timesteps = torch.linspace(1, self.config.sampling_eps, num_inference_steps)
        else:
            raise ValueError(f"`self.tensor_format`: {self.tensor_format} is not valid.")

    def set_sigmas(self, num_inference_steps):
        if self.timesteps is None:
            self.set_timesteps(num_inference_steps)

        tensor_format = getattr(self, "tensor_format", "pt")
        if tensor_format == "np":
            self.discrete_sigmas = np.exp(
                np.linspace(np.log(self.config.sigma_min), np.log(self.config.sigma_max), num_inference_steps)
            )
            self.sigmas = np.array(
                [self.config.sigma_min * (self.config.sigma_max / self.sigma_min) ** t for t in self.timesteps]
            )
        elif tensor_format == "pt":
            self.discrete_sigmas = torch.exp(
                torch.linspace(np.log(self.config.sigma_min), np.log(self.config.sigma_max), num_inference_steps)
            )
            self.sigmas = torch.tensor(
                [self.config.sigma_min * (self.config.sigma_max / self.sigma_min) ** t for t in self.timesteps]
            )
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

    def step_pred(self, score, x, t):
        """
        Predict the sample at the previous timestep by reversing the SDE.
        """
        # TODO(Patrick) better comments + non-PyTorch
        t = self.repeat_scalar(t, x.shape[0]).to(x.device)
        timesteps = self.long((t * (len(self.timesteps) - 1))).to(x.device)

        sigma = self.discrete_sigmas[timesteps].to(x.device)
        adjacent_sigma = self.get_adjacent_sigma(timesteps, t)
        drift = self.zeros_like(x)
        diffusion = (sigma**2 - adjacent_sigma**2) ** 0.5

        # equation 6 in the paper: the score modeled by the network is grad_x log pt(x)
        # also equation 47 shows the analog from SDE models to ancestral sampling methods
        drift = drift - diffusion[:, None, None, None] ** 2 * score

        #  equation 6: sample noise for the diffusion term of
        noise = self.randn_like(x)
        x_mean = x - drift  # subtract because `dt` is a small negative timestep
        # TODO is the variable diffusion the correct scaling term for the noise?
        x = x_mean + diffusion[:, None, None, None] * noise  # add impact of diffusion field g
        return x, x_mean

    def step_correct(self, score, x):
        """
        Correct the predicted sample based on the output score of the network. This is often run repeatedly after
        making the prediction for the previous timestep.
        """
        # TODO(Patrick) non-PyTorch

        # For small batch sizes, the paper "suggest replacing norm(z) with sqrt(d), where d is the dim. of z"
        # sample noise for correction
        noise = self.randn_like(x)

        # compute step size from the score, the noise, and the snr
        grad_norm = self.norm(score)
        noise_norm = self.norm(noise)
        step_size = (self.config.snr * noise_norm / grad_norm) ** 2 * 2
        step_size = self.repeat_scalar(step_size, x.shape[0])  # * self.ones(x.shape[0], device=x.device)

        # compute corrected sample: score term and noise term
        x_mean = x + step_size[:, None, None, None] * score
        x = x_mean + ((step_size * 2) ** 0.5)[:, None, None, None] * noise

        return x

    def __len__(self):
        return self.config.num_train_timesteps
