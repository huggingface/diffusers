# Copyright 2022 UC Berkely Team and The HuggingFace Team. All rights reserved.
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

# DISCLAIMER: This file is strongly influenced by https://github.com/ermongroup/ddim

import math
from typing import Union

import numpy as np
import torch

from ..configuration_utils import ConfigMixin
from .scheduling_utils import SchedulerMixin


def betas_for_alpha_bar(num_diffusion_timesteps, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function, which defines the cumulative product of
    (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce. :param alpha_bar: a lambda that takes an argument t
    from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that part of the diffusion process.
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
        num_train_timesteps=1000,
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
            num_train_timesteps=num_train_timesteps,
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
            self.betas = np.linspace(beta_start, beta_end, num_train_timesteps, dtype=np.float32)
        elif beta_schedule == "squaredcos_cap_v2":
            # Glide cosine schedule
            self.betas = betas_for_alpha_bar(num_train_timesteps)
        else:
            raise NotImplementedError(f"{beta_schedule} does is not implemented for {self.__class__}")

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas, axis=0)
        self.one = np.array(1.0)

        # setable values
        self.num_inference_steps = None
        self.timesteps = np.arange(0, num_train_timesteps)[::-1].copy()

        self.tensor_format = tensor_format
        self.set_format(tensor_format=tensor_format)

    def set_timesteps(self, num_inference_steps):
        self.num_inference_steps = num_inference_steps
        self.timesteps = np.arange(
            0, self.config.num_train_timesteps, self.config.num_train_timesteps // self.num_inference_steps
        )[::-1].copy()
        self.set_format(tensor_format=self.tensor_format)

    def get_variance(self, t, variance_type=None):
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[t - 1] if t > 0 else self.one

        # For t > 0, compute predicted variance βt (see formula (6) and (7) from https://arxiv.org/pdf/2006.11239.pdf)
        # and sample from it to get previous sample
        # x_{t-1} ~ N(pred_prev_sample, variance) == add variance to pred_sample
        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * self.betas[t]

        if variance_type is None:
            variance_type = self.config.variance_type

        # hacks - were probs added for training stability
        if variance_type == "fixed_small":
            variance = self.clip(variance, min_value=1e-20)
        # for rl-diffuser https://arxiv.org/abs/2205.09991
        elif variance_type == "fixed_small_log":
            variance = self.log(self.clip(variance, min_value=1e-20))
        elif variance_type == "fixed_large":
            variance = self.betas[t]
        elif variance_type == "fixed_large_log":
            # Glide max_log
            variance = self.log(self.betas[t])

        return variance

    def step(
        self,
        model_output: Union[torch.FloatTensor, np.ndarray],
        timestep: int,
        sample: Union[torch.FloatTensor, np.ndarray],
        predict_epsilon=True,
    ):
        t = timestep
        # 1. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[t - 1] if t > 0 else self.one
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        # 2. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (15) from https://arxiv.org/pdf/2006.11239.pdf
        if predict_epsilon:
            pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        else:
            pred_original_sample = model_output

        # 3. Clip "predicted x_0"
        if self.config.clip_sample:
            pred_original_sample = self.clip(pred_original_sample, -1, 1)

        # 4. Compute coefficients for pred_original_sample x_0 and current sample x_t
        # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
        pred_original_sample_coeff = (alpha_prod_t_prev ** (0.5) * self.betas[t]) / beta_prod_t
        current_sample_coeff = self.alphas[t] ** (0.5) * beta_prod_t_prev / beta_prod_t

        # 5. Compute predicted previous sample µ_t
        # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample

        return {"prev_sample": pred_prev_sample}

    def add_noise(self, original_samples, noise, timesteps):
        sqrt_alpha_prod = self.alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = self.match_shape(sqrt_alpha_prod, original_samples)
        sqrt_one_minus_alpha_prod = (1 - self.alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = self.match_shape(sqrt_one_minus_alpha_prod, original_samples)

        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples

    def __len__(self):
        return self.config.num_train_timesteps
