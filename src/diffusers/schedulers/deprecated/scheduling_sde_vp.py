# Copyright 2023 Google Brain and The HuggingFace Team. All rights reserved.
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
from typing import Union

import torch

from ...configuration_utils import ConfigMixin, register_to_config
from ...utils.torch_utils import randn_tensor
from ..scheduling_utils import SchedulerMixin


class ScoreSdeVpScheduler(SchedulerMixin, ConfigMixin):
    """
    `ScoreSdeVpScheduler` is a variance preserving stochastic differential equation (SDE) scheduler.

    This model inherits from [`SchedulerMixin`] and [`ConfigMixin`]. Check the superclass documentation for the generic
    methods the library implements for all schedulers such as loading and saving.

    Args:
        num_train_timesteps (`int`, defaults to 2000):
            The number of diffusion steps to train the model.
        beta_min (`int`, defaults to 0.1):
        beta_max (`int`, defaults to 20):
        sampling_eps (`int`, defaults to 1e-3):
            The end value of sampling where timesteps decrease progressively from 1 to epsilon.
    """

    order = 1

    @register_to_config
    def __init__(self, num_train_timesteps=2000, beta_min=0.1, beta_max=20, sampling_eps=1e-3):
        self.sigmas = None
        self.discrete_sigmas = None
        self.timesteps = None

    def set_timesteps(self, num_inference_steps, device: Union[str, torch.device] = None):
        """
        Sets the continuous timesteps used for the diffusion chain (to be run before inference).

        Args:
            num_inference_steps (`int`):
                The number of diffusion steps used when generating samples with a pre-trained model.
            device (`str` or `torch.device`, *optional*):
                The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        """
        self.timesteps = torch.linspace(1, self.config.sampling_eps, num_inference_steps, device=device)

    def step_pred(self, score, x, t, generator=None):
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            score ():
            x ():
            t ():
            generator (`torch.Generator`, *optional*):
                A random number generator.
        """
        if self.timesteps is None:
            raise ValueError(
                "`self.timesteps` is not set, you need to run 'set_timesteps' after creating the scheduler"
            )

        # TODO(Patrick) better comments + non-PyTorch
        # postprocess model score
        log_mean_coeff = -0.25 * t**2 * (self.config.beta_max - self.config.beta_min) - 0.5 * t * self.config.beta_min
        std = torch.sqrt(1.0 - torch.exp(2.0 * log_mean_coeff))
        std = std.flatten()
        while len(std.shape) < len(score.shape):
            std = std.unsqueeze(-1)
        score = -score / std

        # compute
        dt = -1.0 / len(self.timesteps)

        beta_t = self.config.beta_min + t * (self.config.beta_max - self.config.beta_min)
        beta_t = beta_t.flatten()
        while len(beta_t.shape) < len(x.shape):
            beta_t = beta_t.unsqueeze(-1)
        drift = -0.5 * beta_t * x

        diffusion = torch.sqrt(beta_t)
        drift = drift - diffusion**2 * score
        x_mean = x + drift * dt

        # add noise
        noise = randn_tensor(x.shape, layout=x.layout, generator=generator, device=x.device, dtype=x.dtype)
        x = x_mean + diffusion * math.sqrt(-dt) * noise

        return x, x_mean

    def __len__(self):
        return self.config.num_train_timesteps
