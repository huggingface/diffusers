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
import math

import numpy as np

from ..configuration_utils import ConfigMixin
from .scheduling_utils import SchedulerMixin


def betas_for_alpha_bar(num_diffusion_timesteps, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    def alpha_bar(time_step):
        return math.cos((time_step + 0.008) / 1.008 * math.pi / 2) ** 2

    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas, dtype=np.float32)


class DDPMScheduler(SchedulerMixin, ConfigMixin):
    def __init__(
        self,
        timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear",
        trained_betas=None,
        timestep_values=None,
        variance_type="fixed_small",
        clip_sample=True,
        tensor_format="np",
    ):
        super().__init__()
        self.register_to_config(
            timesteps=timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule=beta_schedule,
            trained_betas=trained_betas,
            timestep_values=timestep_values,
            variance_type=variance_type,
            clip_sample=clip_sample,
        )

        if trained_betas is not None:
            self.betas = np.asarray(trained_betas)
        elif beta_schedule == "linear":
            self.betas = np.linspace(beta_start, beta_end, timesteps, dtype=np.float32)
        elif beta_schedule == "squaredcos_cap_v2":
            # GLIDE cosine schedule
            self.betas = betas_for_alpha_bar(timesteps)
        else:
            raise NotImplementedError(f"{beta_schedule} does is not implemented for {self.__class__}")

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas, axis=0)
        self.one = np.array(1.0)

        self.set_format(tensor_format=tensor_format)

    #        self.register_buffer("betas", betas.to(torch.float32))
    #        self.register_buffer("alphas", alphas.to(torch.float32))
    #        self.register_buffer("alphas_cumprod", alphas_cumprod.to(torch.float32))

    #        alphas_cumprod_prev = torch.nn.functional.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    # TODO(PVP) - check how much of these is actually necessary!
    # LDM only uses "fixed_small"; glide seems to use a weird mix of the two, ...
    # https://github.com/openai/glide-text2im/blob/69b530740eb6cef69442d6180579ef5ba9ef063e/glide_text2im/gaussian_diffusion.py#L246
    #        variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    #        if variance_type == "fixed_small":
    #            log_variance = torch.log(variance.clamp(min=1e-20))
    #        elif variance_type == "fixed_large":
    #            log_variance = torch.log(torch.cat([variance[1:2], betas[1:]], dim=0))
    #
    #
    #        self.register_buffer("log_variance", log_variance.to(torch.float32))
    def get_timestep_values(self):
        return self.config.timestep_values

    def get_alpha(self, time_step):
        return self.alphas[time_step]

    def get_beta(self, time_step):
        return self.betas[time_step]

    def get_alpha_prod(self, time_step):
        if time_step < 0:
            return self.one
        return self.alphas_cumprod[time_step]

    def get_variance(self, t):
        alpha_prod_t = self.get_alpha_prod(t)
        alpha_prod_t_prev = self.get_alpha_prod(t - 1)

        # For t > 0, compute predicted variance βt (see formala (6) and (7) from https://arxiv.org/pdf/2006.11239.pdf)
        # and sample from it to get previous sample
        # x_{t-1} ~ N(pred_prev_sample, variance) == add variane to pred_sample
        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * self.get_beta(t)

        # hacks - were probs added for training stability
        if self.config.variance_type == "fixed_small":
            variance = self.clip(variance, min_value=1e-20)
        # for rl-diffuser https://arxiv.org/abs/2205.09991
        elif self.config.variance_type == "fixed_small_log":
            variance = self.log(self.clip(variance, min_value=1e-20))
        elif self.config.variance_type == "fixed_large":
            variance = self.get_beta(t)

        return variance

    def step(self, residual, sample, t, predict_epsilon=True):
        # 1. compute alphas, betas
        alpha_prod_t = self.get_alpha_prod(t)
        alpha_prod_t_prev = self.get_alpha_prod(t - 1)
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        # 2. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (15) from https://arxiv.org/pdf/2006.11239.pdf
        if predict_epsilon:
            pred_original_sample = (sample - beta_prod_t ** (0.5) * residual) / alpha_prod_t ** (0.5)
        else:
            pred_original_sample = residual

        # 3. Clip "predicted x_0"
        if self.config.clip_sample:
            pred_original_sample = self.clip(pred_original_sample, -1, 1)

        # 4. Compute coefficients for pred_original_sample x_0 and current sample x_t
        # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
        pred_original_sample_coeff = (alpha_prod_t_prev ** (0.5) * self.get_beta(t)) / beta_prod_t
        current_sample_coeff = self.get_alpha(t) ** (0.5) * beta_prod_t_prev / beta_prod_t

        # 5. Compute predicted previous sample µ_t
        # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample

        return pred_prev_sample

    def forward_step(self, original_sample, noise, t):
        sqrt_alpha_prod = self.get_alpha_prod(t) ** 0.5
        sqrt_one_minus_alpha_prod = (1 - self.get_alpha_prod(t)) ** 0.5
        noisy_sample = sqrt_alpha_prod * original_sample + sqrt_one_minus_alpha_prod * noise
        return noisy_sample

    def __len__(self):
        return self.config.timesteps
