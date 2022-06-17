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

from ..configuration_utils import ConfigMixin
from .scheduling_utils import SchedulerMixin


class GradTTSScheduler(SchedulerMixin, ConfigMixin):
    def __init__(
        self,
        timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        tensor_format="np",
    ):
        super().__init__()
        self.register_to_config(
            timesteps=timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
        )
        self.timesteps = int(timesteps)

        self.set_format(tensor_format=tensor_format)

    def sample_noise(self, timestep):
        noise = self.beta_start + (self.beta_end - self.beta_start) * timestep
        return noise

    def step(self, xt, residual, mu, h, timestep):
        noise_t = self.sample_noise(timestep)
        dxt = 0.5 * (mu - xt - residual)
        dxt = dxt * noise_t * h
        xt = xt - dxt
        return xt

    def __len__(self):
        return self.timesteps
