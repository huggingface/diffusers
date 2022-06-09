# Copyright 2022 The HuggingFace Team. All rights reserved.
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
import torch
import numpy as np
from torch import nn

from ..configuration_utils import ConfigMixin
from .schedulers_utils import linear_beta_schedule, betas_for_alpha_bar


SAMPLING_CONFIG_NAME = "scheduler_config.json"


class GlideDDIMScheduler(nn.Module, ConfigMixin):

    config_name = SAMPLING_CONFIG_NAME

    def __init__(
        self,
        timesteps=1000,
        beta_schedule="linear",
        variance_type="fixed_large"
    ):
        super().__init__()
        self.register(
            timesteps=timesteps,
            beta_schedule=beta_schedule,
        )
        self.num_timesteps = int(timesteps)

        if beta_schedule == "linear":
            # Linear schedule from Ho et al, extended to work for any number of
            # diffusion steps.
            scale = 1000 / self.num_timesteps
            beta_start = scale * 0.0001
            beta_end = scale * 0.02
            betas = linear_beta_schedule(timesteps, beta_start=beta_start, beta_end=beta_end)
        else:
            raise NotImplementedError(f"{beta_schedule} does is not implemented for {self.__class__}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.nn.functional.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

        if variance_type == "fixed_small":
            log_variance = torch.log(variance.clamp(min=1e-20))
        elif variance_type == "fixed_large":
            log_variance = torch.log(torch.cat([variance[1:2], betas[1:]], dim=0))

        self.register_buffer("betas", betas.to(torch.float32))
        self.register_buffer("alphas", alphas.to(torch.float32))
        self.register_buffer("alphas_cumprod", alphas_cumprod.to(torch.float32))

        self.register_buffer("log_variance", log_variance.to(torch.float32))

    def get_alpha(self, time_step):
        return self.alphas[time_step]

    def get_beta(self, time_step):
        return self.betas[time_step]

    def get_alpha_prod(self, time_step):
        if time_step < 0:
            return torch.tensor(1.0)
        return self.alphas_cumprod[time_step]

    def sample_variance(self, time_step, shape, device, generator=None):
        variance = self.log_variance[time_step]
        nonzero_mask = torch.tensor([1 - (time_step == 0)], device=device).float()[None, :]

        noise = self.sample_noise(shape, device=device, generator=generator)

        sampled_variance = nonzero_mask * (0.5 * variance).exp()
        sampled_variance = sampled_variance * noise

        return sampled_variance

    def sample_noise(self, shape, device, generator=None):
        # always sample on CPU to be deterministic
        return torch.randn(shape, generator=generator).to(device)

    def __len__(self):
        return self.num_timesteps