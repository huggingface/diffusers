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


class KarrasVeScheduler(SchedulerMixin, ConfigMixin):
    """
    Stochastic sampling from Karras et al. [1] tailored to the Variance-Expanding (VE) models [2].
    Use Algorithm 2 and the VE column of Table 1 from [1] for reference.

    [1] Karras, Tero, et al. "Elucidating the Design Space of Diffusion-Based Generative Models." arXiv preprint arXiv:2206.00364 (2022).
    [2] Song, Yang, et al. "Score-based generative modeling through stochastic differential equations." In Proc. ICLR (2021).
    """

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
        self.schedule = [
            (self.sigma_max * (self.sigma_min**2 / self.sigma_max**2) ** (i / (num_inference_steps - 1)))
            for i in self.timesteps
        ]
        self.schedule = np.array(self.schedule, dtype=np.float32)

        self.set_format(tensor_format=self.tensor_format)

    def get_model_inputs(self, sample, sigma, s_noise=1.007, generator=None):
        """
        Explicit Langevin-like "churn" step of adding noise to the sample according to
        a factor gamma_i ≥ 0 to reach a higher noise level sigma_hat = sigma_i + gamma_i*sigma_i.
        """
        if self.s_min <= sigma <= self.s_max:
            gamma = min(self.s_churn / self.num_inference_steps, 2**0.5 - 1)
        else:
            gamma = 0

        # sample eps ~ N(0, S_noise^2 * I)
        eps = s_noise * torch.randn(sample.shape, generator=generator).to(sample.device)
        sigma_hat = sigma + gamma * sigma
        sample_hat = sample + (self.sqrt(sigma_hat**2 - sigma**2) * eps)

        return sample_hat, sigma_hat

    def step(
        self,
        model_output: Union[torch.FloatTensor, np.ndarray],
        sigma_hat: float,
        sigma_prev: float,
        sample_hat: Union[torch.FloatTensor, np.ndarray],
    ):
        D = sample_hat + sigma_hat * model_output
        derivative = (sample_hat - D) / sigma_hat
        sample_prev = sample_hat + (sigma_prev - sigma_hat) * derivative

        return {"prev_sample": sample_prev, "derivative": derivative}

    def step_correct(
        self,
        model_output: Union[torch.FloatTensor, np.ndarray],
        sigma_hat: float,
        sigma_prev: float,
        sample_hat: Union[torch.FloatTensor, np.ndarray],
        sample_prev: Union[torch.FloatTensor, np.ndarray],
        derivative: Union[torch.FloatTensor, np.ndarray],
    ):
        D_prev = sample_prev + sigma_prev * model_output

        derivative_corr = (sample_prev - D_prev) / sigma_prev
        sample_prev = sample_hat + (sigma_prev - sigma_hat) * (0.5 * derivative + 0.5 * derivative_corr)
        return {"prev_sample": sample_prev, "derivative": derivative_corr}

    def add_noise(self, original_samples, noise, timesteps):
        raise NotImplementedError()
