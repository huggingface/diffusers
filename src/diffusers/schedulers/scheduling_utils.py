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
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import torch

from ..configuration_utils import ConfigMixin, register_to_config
from ..utils import BaseOutput


SCHEDULER_CONFIG_NAME = "scheduler_config.json"


def betas_for_alpha_bar(num_diffusion_timesteps, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function, which defines the cumulative product of
    (1-beta) over time from t = [0,1].

    Contains a function alpha_bar that takes an argument t and transforms it to the cumulative product of (1-beta) up
    to that part of the diffusion process.


    Args:
        num_diffusion_timesteps (`int`): the number of betas to produce.
        max_beta (`float`): the maximum beta to use; use values lower than 1 to
                     prevent singularities.

    Returns:
        betas (`np.ndarray`): the betas used by the scheduler to step the model outputs
    """

    def alpha_bar(time_step):
        return math.cos((time_step + 0.008) / 1.008 * math.pi / 2) ** 2

    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas, dtype=np.float32)


@dataclass
class SchedulerOutput(BaseOutput):
    """
    Base class for the scheduler's step function output.

    Args:
        prev_sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample (x_{t-1}) of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
    """

    prev_sample: torch.FloatTensor


class BaseScheduler(ConfigMixin):
    """
    Mixin containing common functions for the schedulers.
    """

    config_name = SCHEDULER_CONFIG_NAME
    ignore_for_config = ["tensor_format"]

    @register_to_config
    def __init__(
        self,
        beta_start: Optional[float] = None,
        beta_end: Optional[float] = None,
        sigma_min: Optional[float] = None,
        sigma_max: Optional[float] = None,
        num_train_timesteps: int = 1000,
        beta_schedule: Optional[str] = None,
        trained_betas: Optional[np.ndarray] = None,
        tensor_format: str = "pt",
        **kwargs,
    ):
        if beta_start is not None and beta_end is not None:
            if trained_betas is not None:
                self.betas = np.asarray(trained_betas)
            elif beta_schedule == "linear":
                self.betas = np.linspace(beta_start, beta_end, num_train_timesteps, dtype=np.float32)
            elif beta_schedule == "scaled_linear":
                # this schedule is very specific to the latent diffusion model.
                self.betas = (
                    np.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=np.float32) ** 2
                )
            elif beta_schedule == "squaredcos_cap_v2":
                # Glide cosine schedule
                self.betas = betas_for_alpha_bar(num_train_timesteps)
            else:
                raise NotImplementedError(f"{beta_schedule} does is not implemented for {self.__class__}")

            self.alphas = 1.0 - self.betas
            self.alphas_cumprod = np.cumprod(self.alphas, axis=0)
            self.one = np.array(1.0)
        elif sigma_min is not None and sigma_max is not None:
            self.set_sigmas(num_train_timesteps, sigma_min, sigma_max)
        else:
            raise ValueError("Either beta_start and beta_end or sigma_min and sigma_max must be provided.")

        # setable values
        self.num_inference_steps = None
        self.timesteps = np.arange(0, num_train_timesteps)[::-1].copy()

        self.tensor_format = tensor_format
        self.set_format(tensor_format=tensor_format)

    def set_sigmas(self, num_train_timesteps, sigma_min, sigma_max):
        raise NotImplementedError("set_sigmas is not implemented for this scheduler.")

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

    def match_shape(self, values: Union[np.ndarray, torch.Tensor], broadcast_array: Union[np.ndarray, torch.Tensor]):
        """
        Turns a 1-D array into an array or tensor with len(broadcast_array.shape) dims.

        Args:
            values: an array or tensor of values to extract.
            broadcast_array: an array with a larger shape of K dimensions with the batch
                dimension equal to the length of timesteps.
        Returns:
            a tensor of shape [batch_size, 1, ...] where the shape has K dims.
        """

        tensor_format = getattr(self, "tensor_format", "pt")
        values = values.flatten()

        while len(values.shape) < len(broadcast_array.shape):
            values = values[..., None]
        if tensor_format == "pt":
            values = values.to(broadcast_array.device)

        return values

    def norm(self, tensor):
        tensor_format = getattr(self, "tensor_format", "pt")
        if tensor_format == "np":
            return np.linalg.norm(tensor)
        elif tensor_format == "pt":
            return torch.norm(tensor.reshape(tensor.shape[0], -1), dim=-1).mean()

        raise ValueError(f"`self.tensor_format`: {self.tensor_format} is not valid.")

    def randn_like(self, tensor, generator=None):
        tensor_format = getattr(self, "tensor_format", "pt")
        if tensor_format == "np":
            return np.random.randn(*np.shape(tensor))
        elif tensor_format == "pt":
            # return torch.randn_like(tensor)
            return torch.randn(tensor.shape, layout=tensor.layout, generator=generator).to(tensor.device)

        raise ValueError(f"`self.tensor_format`: {self.tensor_format} is not valid.")

    def zeros_like(self, tensor):
        tensor_format = getattr(self, "tensor_format", "pt")
        if tensor_format == "np":
            return np.zeros_like(tensor)
        elif tensor_format == "pt":
            return torch.zeros_like(tensor)

        raise ValueError(f"`self.tensor_format`: {self.tensor_format} is not valid.")
