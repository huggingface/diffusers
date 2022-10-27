# Copyright 2022 UC Berkeley Team and The HuggingFace Team. All rights reserved.
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
from typing import Optional, Tuple, Union

import numpy as np
import torch

from ..configuration_utils import ConfigMixin, register_to_config
from ..utils import BaseOutput
from .scheduling_utils import SchedulerMixin


@dataclass
class ALDSchedulerOutput(BaseOutput):
    """
    Output class for the scheduler's step function output.

    Args:
        prev_sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample (x_{t-1}) of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
        pred_original_sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` for images):
            The predicted denoised sample (x_{0}) based on the model output from the current timestep.
            `pred_original_sample` can be used to preview progress or for guidance.
    """

    prev_sample: torch.FloatTensor
    pred_original_sample: Optional[torch.FloatTensor] = None



class ALDScheduler(SchedulerMixin, ConfigMixin):
    """
    The Annealed Langevin Dynamics sampler was popularized in the paper on Noise Conditional Score Networks (NCSNs). For more details, refer to the paper https://arxiv.org/abs/1907.05600

    [`~ConfigMixin`] takes care of storing all config attributes that are passed in the scheduler's `__init__`
    function, such as `num_train_timesteps`. They can be accessed via `scheduler.config.num_train_timesteps`.
    [`~ConfigMixin`] also provides general loading and saving functionality via the [`~ConfigMixin.save_config`] and
    [`~ConfigMixin.from_config`] functions.

    For more details, see the original paper: https://arxiv.org/abs/2006.11239

    Args:
        num_train_timesteps (`int`): number of diffusion steps used to train the model.
        sigma_min (`float`):
                initial noise scale for sigma sequence in sampling procedure. The minimum sigma should mirror the
                distribution of the data.
        sigma_max (`float`): maximum value used for the range of continuous timesteps passed into the model.
        step_lr (`float`): learning rate for stepping through noise.

    """

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 100,
        sigma_min: float = 0.01,
        sigma_max: float = 1.0,
        step_lr: float = 0.00002,
    ):
        # standard deviation of the initial noise distribution
        self.final_noise_sigma = None
        self.step_lr = step_lr

        # setable values
        self.num_inference_steps = None
        self.timesteps = torch.from_numpy(np.arange(0, num_train_timesteps)[::-1].copy())

        self.set_sigmas(num_train_timesteps, sigma_min, sigma_max)

    def scale_model_input(self, sample: torch.FloatTensor, timestep: Optional[int] = None) -> torch.FloatTensor:
        """
        Ensures interchangeability with schedulers that need to scale the denoising model input depending on the
        current timestep.

        Args:
            sample (`torch.FloatTensor`): input sample
            timestep (`int`, optional): current timestep

        Returns:
            `torch.FloatTensor`: scaled input sample
        """
        return sample

    def set_timesteps(
        self, num_inference_steps: int, device: Union[str, torch.device] = None
    ):
        """
        Sets the continuous timesteps used for the diffusion chain. Supporting function to be run before inference.

        Args:
            num_inference_steps (`int`):
                the number of diffusion steps used when generating samples with a pre-trained model.

        """
        num_inference_steps = min(self.config.num_train_timesteps, num_inference_steps)
        self.num_inference_steps = num_inference_steps
        timesteps = np.arange(
            0, self.config.num_train_timesteps, self.config.num_train_timesteps // self.num_inference_steps
        )[::-1].copy()
        self.timesteps = torch.from_numpy(timesteps).to(device)

    def set_sigmas(
        self, num_inference_steps: int, sigma_min: float = None, sigma_max: float = None
    ):
        """
        Sets the noise scales used for the diffusion chain. Supporting function to be run before inference.

        The sigmas control the weight of the `drift` and `diffusion` components of sample update.

        Args:
            num_inference_steps (`int`):
                the number of diffusion steps used when generating samples with a pre-trained model.
            sigma_min (`float`, optional):
                initial noise scale value (overrides value given at Scheduler instantiation).
            sigma_max (`float`, optional): final noise scale value (overrides value given at Scheduler instantiation).

        """
        sigma_min = sigma_min if sigma_min is not None else self.config.sigma_min
        sigma_max = sigma_max if sigma_max is not None else self.config.sigma_max

        if self.timesteps is None:
            self.set_timesteps(num_inference_steps)

        self.sigmas = torch.tensor(
            torch.exp(torch.linspace(torch.log(sigma_min), torch.log(sigma_max),
                               num_inference_steps)), dtype=torch.float32)

        self.final_noise_sigma = self.sigmas[-1]

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        sample: torch.FloatTensor,
        generator=None,
        return_dict: bool = True,
    ) -> Union[ALDSchedulerOutput, Tuple]:
        """
        Predict the sample at the previous timestep by reversing the SDE. Core function to propagate the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.FloatTensor`): direct output from learned diffusion model.
            timestep (`int`): current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                current instance of sample being created by diffusion process.
            generator: random number generator.
            return_dict (`bool`): option for returning tuple rather than ALDSchedulerOutput class

        Returns:
            [`~schedulers.scheduling_utils.ALDSchedulerOutput`] or `tuple`:
            [`~schedulers.scheduling_utils.ALDSchedulerOutput`] if `return_dict` is True, otherwise a `tuple`. When
            returning a tuple, the first element is the sample tensor.

        """
        t = timestep

        # 1. get sigma
        sigma = self.sigmas[t]

        # 2. compute step_size
        step_size = self.step_lr * (sigma / self.final_noise_sigma) ** 2

        # 3. create new output
        pred_prev_sample = sample + step_size * model_output

        # 4. Add noise except last step
        variance = 0
        if t > 0:
            noise = torch.randn(
                model_output.size(), dtype=model_output.dtype, layout=model_output.layout, generator=generator
            ).to(model_output.device)
            variance = noise * torch.sqrt(step_size * 2)

        pred_prev_sample = pred_prev_sample + variance

        if not return_dict:
            return (pred_prev_sample,)

        return ALDSchedulerOutput(prev_sample=pred_prev_sample)

    def __len__(self):
        return self.config.num_train_timesteps
