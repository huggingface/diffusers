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
from .scheduling_utils import SchedulerMixin, betas_for_alpha_bar, linear_beta_schedule


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
        self.register(
            timesteps=timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule=beta_schedule,
        )
        self.timesteps = int(timesteps)

        if beta_schedule == "linear":
            self.betas = linear_beta_schedule(timesteps, beta_start=beta_start, beta_end=beta_end)
        elif beta_schedule == "squaredcos_cap_v2":
            # GLIDE cosine schedule
            self.betas = betas_for_alpha_bar(
                timesteps,
                lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
            )
        else:
            raise NotImplementedError(f"{beta_schedule} does is not implemented for {self.__class__}")

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas, axis=0)

        self.one = np.array(1.0)

        self.set_format(tensor_format=tensor_format)

        # for now we only support F-PNDM, i.e. the runge-kutta method
        self.pndm_order = 4

        # running values
        self.cur_residual = 0
        self.ets = []
        self.warmup_time_steps = {}
        self.time_steps = {}

    def get_alpha(self, time_step):
        return self.alphas[time_step]

    def get_beta(self, time_step):
        return self.betas[time_step]

    def get_alpha_prod(self, time_step):
        if time_step < 0:
            return self.one
        return self.alphas_cumprod[time_step]

    def get_warmup_time_steps(self, num_inference_steps):
        if num_inference_steps in self.warmup_time_steps:
            return self.warmup_time_steps[num_inference_steps]

        inference_step_times = list(range(0, self.timesteps, self.timesteps // num_inference_steps))

        warmup_time_steps = np.array(inference_step_times[-self.pndm_order:]).repeat(2) + np.tile(np.array([0, self.timesteps // num_inference_steps // 2]), self.pndm_order)
        self.warmup_time_steps[num_inference_steps] = list(reversed(warmup_time_steps[:-1].repeat(2)[1:-1]))

        return self.warmup_time_steps[num_inference_steps]

    def get_time_steps(self, num_inference_steps):
        if num_inference_steps in self.time_steps:
            return self.time_steps[num_inference_steps]

        inference_step_times = list(range(0, self.timesteps, self.timesteps // num_inference_steps))
        self.time_steps[num_inference_steps] = list(reversed(inference_step_times[:-3]))

        return self.time_steps[num_inference_steps]

    def step_warm_up(self, residual, image, t, num_inference_steps):
        # TODO(Patrick) - need to rethink whether the "warmup" way is the correct API design here
        warmup_time_steps = self.get_warmup_time_steps(num_inference_steps)

        t_prev = warmup_time_steps[t // 4 * 4]
        t_next = warmup_time_steps[min(t + 1, len(warmup_time_steps) - 1)]

        if t % 4 == 0:
            self.cur_residual += 1 / 6 * residual
            self.ets.append(residual)
        elif (t - 1) % 4 == 0:
            self.cur_residual += 1 / 3 * residual
        elif (t - 2) % 4 == 0:
            self.cur_residual += 1 / 3 * residual
        elif (t - 3) % 4 == 0:
            residual = self.cur_residual + 1 / 6 * residual
            self.cur_residual = 0

        return self.transfer(image, t_prev, t_next, residual)

    def step(self, residual, image, t, num_inference_steps):
        timesteps = self.get_time_steps(num_inference_steps)

        t_prev = timesteps[t]
        t_next = timesteps[min(t + 1, len(timesteps) - 1)]
        self.ets.append(residual)

        residual = (1 / 24) * (55 * self.ets[-1] - 59 * self.ets[-2] + 37 * self.ets[-3] - 9 * self.ets[-4])

        return self.transfer(image, t_prev, t_next, residual)

    def transfer(self, x, t, t_next, et):
        # TODO(Patrick): clean up to be compatible with numpy and give better names

        alphas_cump = self.alphas_cumprod.to(x.device)
        at = alphas_cump[t + 1].view(-1, 1, 1, 1)
        at_next = alphas_cump[t_next + 1].view(-1, 1, 1, 1)

        x_delta = (at_next - at) * ((1 / (at.sqrt() * (at.sqrt() + at_next.sqrt()))) * x - 1 / (at.sqrt() * (((1 - at_next) * at).sqrt() + ((1 - at) * at_next).sqrt())) * et)

        x_next = x + x_delta
        return x_next

    def __len__(self):
        return self.timesteps
