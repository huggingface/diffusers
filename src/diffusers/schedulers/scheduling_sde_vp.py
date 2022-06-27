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


class ScoreSdeVpScheduler(SchedulerMixin, ConfigMixin):
    def __init__(self, beta_min=0.1, beta_max=20, sampling_eps=1e-3, tensor_format="np"):
        super().__init__()
        self.register_to_config(
            beta_min=beta_min,
            beta_max=beta_max,
            sampling_eps=sampling_eps,
        )

        self.sigmas = None
        self.discrete_sigmas = None
        self.timesteps = None

    def set_timesteps(self, num_inference_steps):
        self.timesteps = torch.linspace(1, self.config.sampling_eps, num_inference_steps)

    def step_pred(self, result, x, t):
        # TODO(Patrick) better comments + non-PyTorch
        # postprocess model result
        log_mean_coeff = (
            -0.25 * t**2 * (self.config.beta_max - self.config.beta_min) - 0.5 * t * self.config.beta_min
        )
        std = torch.sqrt(1.0 - torch.exp(2.0 * log_mean_coeff))
        result = -result / std[:, None, None, None]

        # compute
        dt = -1.0 / len(self.timesteps)

        beta_t = self.config.beta_min + t * (self.config.beta_max - self.config.beta_min)
        drift = -0.5 * beta_t[:, None, None, None] * x
        diffusion = torch.sqrt(beta_t)
        drift = drift - diffusion[:, None, None, None] ** 2 * result
        x_mean = x + drift * dt

        # add noise
        z = torch.randn_like(x)
        x = x_mean + diffusion[:, None, None, None] * np.sqrt(-dt) * z

        return x, x_mean
