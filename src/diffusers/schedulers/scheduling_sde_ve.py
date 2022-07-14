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

    :param snr:
    :param sigma_min:
    :param sigma_max:
    :param sampling_eps:
    :param correct_steps:
    :param tensor_format:
    """
    def __init__(self, snr=0.15, sigma_min=0.01, sigma_max=1348, sampling_eps=1e-5, correct_steps=1, tensor_format="np"):
        super().__init__()
        self.register_to_config(
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
        self.timesteps = torch.linspace(1, self.config.sampling_eps, num_inference_steps)

    def set_sigmas(self, num_inference_steps):
        if self.timesteps is None:
            self.set_timesteps(num_inference_steps)

        self.discrete_sigmas = torch.exp(
            torch.linspace(np.log(self.config.sigma_min), np.log(self.config.sigma_max), num_inference_steps)
        )
        self.sigmas = torch.tensor(
            [self.config.sigma_min * (self.config.sigma_max / self.sigma_min) ** t for t in self.timesteps]
        )

    def step_pred(self, score, x, t):
        """
        Predict the sample at the previous timestep by reversing the SDE.
        """
        # TODO(Patrick) better comments + non-PyTorch
        t = t * torch.ones(x.shape[0], device=x.device)
        timestep = (t * (len(self.timesteps) - 1)).long()

        sigma = self.discrete_sigmas.to(t.device)[timestep]
        adjacent_sigma = torch.where(
            timestep == 0, torch.zeros_like(t), self.discrete_sigmas[timestep - 1].to(timestep.device)
        )
        f = torch.zeros_like(x)
        G = torch.sqrt(sigma**2 - adjacent_sigma**2)

        # equation 6 in the paper: the score modeled by the network is grad_x log pt(x)
        f = f - G[:, None, None, None] ** 2 * score

        #  equation 6: sample noise for the diffusion term of
        z = torch.randn_like(x)
        x_mean = x - f                              # substract because `dt` is a small negative timestep
        x = x_mean + G[:, None, None, None] * z     # add impact of diffusion field g
        return x, x_mean

    def step_correct(self, result, x):
        """
        Correct the predicted sample based on the output of the network.
        This is often run repeatedly after making the prediction for the previous timestep.
        """
        # TODO(Patrick) better comments + non-PyTorch

        # TODO for small batch sizes, the paper "suggest replacing nomr(z) with sqrt(d), where d is the dim. of z".
        noise = torch.randn_like(x)
        grad_norm = torch.norm(result.reshape(result.shape[0], -1), dim=-1).mean()
        noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
        step_size = (self.config.snr * noise_norm / grad_norm) ** 2 * 2
        step_size = step_size * torch.ones(x.shape[0], device=x.device)
        x_mean = x + step_size[:, None, None, None] * result

        x = x_mean + torch.sqrt(step_size * 2)[:, None, None, None] * noise

        return x
