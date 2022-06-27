# Copyright 2022 Zhejiang University Team and The HuggingFace Team. All rights reserved.
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

import numpy as np

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


class PNDMScheduler(SchedulerMixin, ConfigMixin):
    def __init__(
        self,
        timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear",
        tensor_format="np",
    ):
        super().__init__()
        self.register_to_config(
            timesteps=timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule=beta_schedule,
        )

        if beta_schedule == "linear":
            self.betas = np.linspace(beta_start, beta_end, timesteps, dtype=np.float32)
        elif beta_schedule == "squaredcos_cap_v2":
            # Glide cosine schedule
            self.betas = betas_for_alpha_bar(timesteps)
        else:
            raise NotImplementedError(f"{beta_schedule} does is not implemented for {self.__class__}")

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas, axis=0)

        self.one = np.array(1.0)

        self.set_format(tensor_format=tensor_format)

        # For now we only support F-PNDM, i.e. the runge-kutta method
        # For more information on the algorithm please take a look at the paper: https://arxiv.org/pdf/2202.09778.pdf
        # mainly at formula (9), (12), (13) and the Algorithm 2.
        self.pndm_order = 4

        # running values
        self.cur_residual = 0
        self.cur_sample = None
        self.ets = []
        self.prk_time_steps = {}
        self.time_steps = {}
        self.set_prk_mode()

    def get_prk_time_steps(self, num_inference_steps):
        if num_inference_steps in self.prk_time_steps:
            return self.prk_time_steps[num_inference_steps]

        inference_step_times = list(range(0, self.config.timesteps, self.config.timesteps // num_inference_steps))

        prk_time_steps = np.array(inference_step_times[-self.pndm_order :]).repeat(2) + np.tile(
            np.array([0, self.config.timesteps // num_inference_steps // 2]), self.pndm_order
        )
        self.prk_time_steps[num_inference_steps] = list(reversed(prk_time_steps[:-1].repeat(2)[1:-1]))

        return self.prk_time_steps[num_inference_steps]

    def get_time_steps(self, num_inference_steps):
        if num_inference_steps in self.time_steps:
            return self.time_steps[num_inference_steps]

        inference_step_times = list(range(0, self.config.timesteps, self.config.timesteps // num_inference_steps))
        self.time_steps[num_inference_steps] = list(reversed(inference_step_times[:-3]))

        return self.time_steps[num_inference_steps]

    def set_prk_mode(self):
        self.mode = "prk"

    def set_plms_mode(self):
        self.mode = "plms"

    def step(self, *args, **kwargs):
        if self.mode == "prk":
            return self.step_prk(*args, **kwargs)
        if self.mode == "plms":
            return self.step_plms(*args, **kwargs)

        raise ValueError(f"mode {self.mode} does not exist.")

    def step_prk(self, residual, sample, t, num_inference_steps):
        prk_time_steps = self.get_prk_time_steps(num_inference_steps)

        t_orig = prk_time_steps[t // 4 * 4]
        t_orig_prev = prk_time_steps[min(t + 1, len(prk_time_steps) - 1)]

        if t % 4 == 0:
            self.cur_residual += 1 / 6 * residual
            self.ets.append(residual)
            self.cur_sample = sample
        elif (t - 1) % 4 == 0:
            self.cur_residual += 1 / 3 * residual
        elif (t - 2) % 4 == 0:
            self.cur_residual += 1 / 3 * residual
        elif (t - 3) % 4 == 0:
            residual = self.cur_residual + 1 / 6 * residual
            self.cur_residual = 0

        # cur_sample should not be `None`
        cur_sample = self.cur_sample if self.cur_sample is not None else sample

        return self.get_prev_sample(cur_sample, t_orig, t_orig_prev, residual)

    def step_plms(self, residual, sample, t, num_inference_steps):
        if len(self.ets) < 3:
            raise ValueError(
                f"{self.__class__} can only be run AFTER scheduler has been run "
                "in 'prk' mode for at least 12 iterations "
                "See: https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/pipeline_pndm.py "
                "for more information."
            )

        timesteps = self.get_time_steps(num_inference_steps)

        t_orig = timesteps[t]
        t_orig_prev = timesteps[min(t + 1, len(timesteps) - 1)]
        self.ets.append(residual)

        residual = (1 / 24) * (55 * self.ets[-1] - 59 * self.ets[-2] + 37 * self.ets[-3] - 9 * self.ets[-4])

        return self.get_prev_sample(sample, t_orig, t_orig_prev, residual)

    def get_prev_sample(self, sample, t_orig, t_orig_prev, residual):
        # See formula (9) of PNDM paper https://arxiv.org/pdf/2202.09778.pdf
        # this function computes x_(t−δ) using the formula of (9)
        # Note that x_t needs to be added to both sides of the equation

        # Notation (<variable name> -> <name in paper>
        # alpha_prod_t -> α_t
        # alpha_prod_t_prev -> α_(t−δ)
        # beta_prod_t -> (1 - α_t)
        # beta_prod_t_prev -> (1 - α_(t−δ))
        # sample -> x_t
        # residual -> e_θ(x_t, t)
        # prev_sample -> x_(t−δ)
        alpha_prod_t = self.alphas_cumprod[t_orig + 1]
        alpha_prod_t_prev = self.alphas_cumprod[t_orig_prev + 1]
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        # corresponds to (α_(t−δ) - α_t) divided by
        # denominator of x_t in formula (9) and plus 1
        # Note: (α_(t−δ) - α_t) / (sqrt(α_t) * (sqrt(α_(t−δ)) + sqr(α_t))) =
        # sqrt(α_(t−δ)) / sqrt(α_t))
        sample_coeff = (alpha_prod_t_prev / alpha_prod_t) ** (0.5)

        # corresponds to denominator of e_θ(x_t, t) in formula (9)
        residual_denom_coeff = alpha_prod_t * beta_prod_t_prev ** (0.5) + (
            alpha_prod_t * beta_prod_t * alpha_prod_t_prev
        ) ** (0.5)

        # full formula (9)
        prev_sample = sample_coeff * sample - (alpha_prod_t_prev - alpha_prod_t) * residual / residual_denom_coeff

        return prev_sample

    def __len__(self):
        return self.config.timesteps
