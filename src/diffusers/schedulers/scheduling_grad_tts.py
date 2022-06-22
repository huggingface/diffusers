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

import numpy as np

from ..configuration_utils import ConfigMixin
from .scheduling_utils import SchedulerMixin


class GradTTSScheduler(SchedulerMixin, ConfigMixin):
    def __init__(
        self,
        beta_start=0.05,
        beta_end=20,
        tensor_format="np",
    ):
        super().__init__()
        self.register_to_config(
            beta_start=beta_start,
            beta_end=beta_end,
        )
        self.set_format(tensor_format=tensor_format)
        self.betas = None

    def get_timesteps(self, num_inference_steps):
        return np.array([(t + 0.5) / num_inference_steps for t in range(num_inference_steps)])

    def set_betas(self, num_inference_steps):
        timesteps = self.get_timesteps(num_inference_steps)
        self.betas = np.array([self.beta_start + (self.beta_end - self.beta_start) * t for t in timesteps])

    def step(self, residual, sample, t, num_inference_steps):
        # This is a VE scheduler from https://arxiv.org/pdf/2011.13456.pdf (see Algorithm 2 in Appendix)
        if self.betas is None:
            self.set_betas(num_inference_steps)

        beta_t = self.betas[t]
        beta_t_deriv = beta_t / num_inference_steps

        sample_deriv = residual * beta_t_deriv / 2

        sample = sample + sample_deriv
        return sample
