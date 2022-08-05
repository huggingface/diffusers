# Copyright 2022 NVIDIA and The HuggingFace Team. All rights reserved.
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
from typing import Union

import numpy as np
import torch

from ..configuration_utils import ConfigMixin, register_to_config
from .scheduling_utils import SchedulerMixin


class KarrasScheduler(SchedulerMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        sigma_min=0.02,
        sigma_max=100,
        s_churn=80,
        s_min=0.05,
        s_max=50,
        tensor_format="pt",
    ):
        # setable values
        self.num_inference_steps = None
        self.timesteps = None
        self.schedule = None  # sigma(t_i)

        self.tensor_format = tensor_format
        self.set_format(tensor_format=tensor_format)

    def set_timesteps(self, num_inference_steps):
        self.num_inference_steps = num_inference_steps
        self.timesteps = np.arange(0, self.num_inference_steps)[::-1].copy()

        # self.schedule = [
        #     (
        #         self.sigma_max ** (1 / self.rho)
        #         + (i / (num_inference_steps - 1))
        #         * (self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho))
        #     )
        #     ** self.rho
        #     for i in self.timesteps
        # ]

        self.schedule = [
            (
                    self.sigma_max
                    * (self.sigma_min ** 2 / self.sigma_max ** 2) ** (i / (num_inference_steps - 1))
            )
            for i in self.timesteps
        ]

        # t[N] = 0
        #self.schedule += [0]
        self.schedule = np.array(self.schedule, dtype=np.float32)

        self.set_format(tensor_format=self.tensor_format)

    def get_model_inputs(self, sample, t, s_noise=1.007, generator=None):
        if self.s_min <= t <= self.s_max:
            gamma = min(self.s_churn / self.num_inference_steps, 2**0.5 - 1)
        else:
            gamma = 0

        eps = s_noise * torch.randn(sample.shape, generator=generator).to(sample.device)
        t_hat = t + gamma * t
        x_hat = sample + (self.sqrt(t_hat**2 - t**2) * eps)

        return x_hat, t_hat

    def step(
        self,
        model_output: Union[torch.FloatTensor, np.ndarray],
        timestep: int,
        sample: Union[torch.FloatTensor, np.ndarray],
        generator=None,
    ):
        return {"prev_sample": None}

    def add_noise(self, original_samples, noise, timesteps):
        raise NotImplementedError()
