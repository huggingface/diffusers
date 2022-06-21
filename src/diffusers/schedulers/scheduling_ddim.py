# Copyright 2022 Stanford University Team and The HuggingFace Team. All rights reserved.
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

# DISCLAIMER: This code is strongly influenced by https://github.com/pesser/pytorch_diffusion
# and https://github.com/hojonathanho/diffusion

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


class DDIMScheduler(SchedulerMixin, ConfigMixin):
    def __init__(
        self,
        timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear",
        trained_betas=None,
        timestep_values=None,
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
            clip_sample=clip_sample,
        )

        if beta_schedule == "linear":
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

    def get_variance(self, t, num_inference_steps):
        orig_t = self.config.timesteps // num_inference_steps * t
        orig_prev_t = self.config.timesteps // num_inference_steps * (t - 1) if t > 0 else -1

        alpha_prod_t = self.alphas_cumprod[orig_t]
        alpha_prod_t_prev = self.alphas_cumprod[orig_prev_t] if orig_prev_t >= 0 else self.one
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)

        return variance

    def step(self, residual, sample, t, num_inference_steps, eta, use_clipped_residual=False):
        # See formulas (12) and (16) of DDIM paper https://arxiv.org/pdf/2010.02502.pdf
        # Ideally, read DDIM paper in-detail understanding

        # Notation (<variable name> -> <name in paper>
        # - pred_noise_t -> e_theta(x_t, t)
        # - pred_original_sample -> f_theta(x_t, t) or x_0
        # - std_dev_t -> sigma_t
        # - eta -> η
        # - pred_sample_direction -> "direction pointingc to x_t"
        # - pred_prev_sample -> "x_t-1"

        # 1. get actual t and t-1
        orig_t = self.config.timesteps // num_inference_steps * t
        orig_prev_t = self.config.timesteps // num_inference_steps * (t - 1) if t > 0 else -1

        # 2. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[orig_t]
        alpha_prod_t_prev = self.alphas_cumprod[orig_prev_t] if orig_prev_t >= 0 else self.one
        beta_prod_t = 1 - alpha_prod_t

        # 3. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        pred_original_sample = (sample - beta_prod_t ** (0.5) * residual) / alpha_prod_t ** (0.5)

        # 4. Clip "predicted x_0"
        if self.config.clip_sample:
            pred_original_sample = self.clip(pred_original_sample, -1, 1)

        # 5. compute variance: "sigma_t(η)" -> see formula (16)
        # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
        variance = self.get_variance(t, num_inference_steps)
        std_dev_t = eta * variance ** (0.5)

        if use_clipped_residual:
            # the residual is always re-derived from the clipped x_0 in GLIDE
            residual = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)

        # 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * residual

        # 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        pred_prev_sample = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction

        return pred_prev_sample

    def __len__(self):
        return self.config.timesteps
