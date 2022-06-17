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
import torch


SCHEDULER_CONFIG_NAME = "scheduler_config.json"


def linear_beta_schedule(timesteps, beta_start, beta_end):
    return np.linspace(beta_start, beta_end, timesteps, dtype=np.float32)


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
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
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas, dtype=np.float32)


class SchedulerMixin:

    config_name = SCHEDULER_CONFIG_NAME

    def set_format(self, tensor_format="pt"):
        self.tensor_format = tensor_format
        if tensor_format == "pt":
            for key, value in vars(self).items():
                if isinstance(value, np.ndarray):
                    setattr(self, key, torch.from_numpy(value))

        return self

    def clip(self, tensor, min_value=None, max_value=None):
        tensor_format = getattr(self, "tensor_format", "pt")

        if tensor_format == "np":
            return np.clip(tensor, min_value, max_value)
        elif tensor_format == "pt":
            return torch.clamp(tensor, min_value, max_value)

        raise ValueError(f"`self.tensor_format`: {self.tensor_format} is not valid.")

    def log(self, tensor):
        tensor_format = getattr(self, "tensor_format", "pt")

        if tensor_format == "np":
            return np.log(tensor)
        elif tensor_format == "pt":
            return torch.log(tensor)

        raise ValueError(f"`self.tensor_format`: {self.tensor_format} is not valid.")
