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
from typing import Union

import numpy as np
import torch

from ..configuration_utils import ConfigMixin, register_to_config
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


class RePaintScheduler(SchedulerMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear",
        trained_betas=None,
        timestep_values=None,
        clip_sample=True,
        tensor_format="pt",
    ):

        if beta_schedule == "linear":
            self.betas = np.linspace(beta_start, beta_end, num_train_timesteps, dtype=np.float32)
        elif beta_schedule == "scaled_linear":
            # this schedule is very specific to the latent diffusion model.
            self.betas = np.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=np.float32) ** 2
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

    def _get_variance(self, timestep, prev_timestep):
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.one
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)

        return variance

    def set_timesteps(self, num_inference_steps, jump_length=10, jump_n_sample=10):
        self.num_inference_steps = num_inference_steps
        timesteps = []

        jumps = {}
        for j in range(0, num_inference_steps - jump_length, jump_length):
            jumps[j] = jump_n_sample - 1

        t = num_inference_steps
        while t >= 1:
            t = t - 1
            timesteps.append(t)

            if jumps.get(t, 0) > 0:
                jumps[t] = jumps[t] - 1
                for _ in range(jump_length):
                    t = t + 1
                    timesteps.append(t)

        self.timesteps = np.array(timesteps) * (self.config.num_train_timesteps // self.num_inference_steps)

        self.set_format(tensor_format=self.tensor_format)

    def step(
        self,
        model_output: Union[torch.FloatTensor, np.ndarray],
        timestep: int,
        sample: Union[torch.FloatTensor, np.ndarray],
        original_sample: Union[torch.FloatTensor, np.ndarray],
        mask: Union[torch.FloatTensor, np.ndarray],
        generator=None,
    ):
        device = model_output.device
        prev_timestep = timestep - self.config.num_train_timesteps // self.num_inference_steps

        alpha = self.alphas[timestep]
        alpha_prod = self.alphas_cumprod[timestep]
        beta = self.betas[timestep]
        alpha_prod_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.one
        std_dev = self.sqrt(self._get_variance(timestep, prev_timestep))

        if timestep > 1:
            noise = torch.randn(model_output.shape, generator=generator).to(device)
        else:
            noise = torch.zeros(model_output.shape, device=device)

        # compute predicted original sample from predicted noise
        pred_original_sample = (sample - self.sqrt(1 - alpha_prod) * model_output) / self.sqrt(alpha_prod)

        #  clip "predicted x_0"
        if self.config.clip_sample:
            pred_original_sample = self.clip(pred_original_sample, -1, 1)

        # add noise to the known pixels of the image
        prev_known_part = self.sqrt(alpha_prod) * original_sample + self.sqrt(1 - alpha_prod) * noise

        # add noise to the unknown pixels of the image
        posterior_mean_coef1 = (
                beta * self.sqrt(alpha_prod_prev) /
                (1.0 - alpha_prod)
        )
        posterior_mean_coef2 = (
                (1.0 - alpha_prod_prev)
                * self.sqrt(alpha)
                / (1.0 - alpha_prod)
        )
        prev_unknown_part = posterior_mean_coef1 * pred_original_sample + posterior_mean_coef2 * sample
        prev_unknown_part = prev_unknown_part + std_dev * noise
        #pred_sample_direction = self.sqrt(1 - alpha_prod_prev - std_dev ** 2) * model_output
        #prev_unknown_part = self.sqrt(alpha_prod_prev) * pred_original_sample + pred_sample_direction
        #prev_unknown_part = prev_unknown_part + std_dev * noise

        prev_sample = mask * prev_known_part + (1 - mask) * prev_unknown_part

        return {"prev_sample": prev_sample}

    def undo_step(self, sample, timestep, generator=None):
        beta = self.betas[timestep]

        noise = torch.randn(sample.shape, generator=generator).to(sample.device)
        next_sample = self.sqrt(1 - beta) * sample + self.sqrt(beta) * noise

        return next_sample

    def add_noise(self, original_samples, noise, timesteps):
        raise NotImplementedError("Use `DDPMScheduler.add_noise()` to train for sampling with RePaint.")

    def __len__(self):
        return self.config.num_train_timesteps
