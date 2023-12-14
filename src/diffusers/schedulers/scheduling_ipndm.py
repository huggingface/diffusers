# Copyright 2023 Zhejiang University Team and The HuggingFace Team. All rights reserved.
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
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from ..configuration_utils import ConfigMixin, register_to_config
from .scheduling_utils import SchedulerMixin, SchedulerOutput


class IPNDMScheduler(SchedulerMixin, ConfigMixin):
    """
    A fourth-order Improved Pseudo Linear Multistep scheduler.

    This model inherits from [`SchedulerMixin`] and [`ConfigMixin`]. Check the superclass documentation for the generic
    methods the library implements for all schedulers such as loading and saving.

    Args:
        num_train_timesteps (`int`, defaults to 1000):
            The number of diffusion steps to train the model.
        trained_betas (`np.ndarray`, *optional*):
            Pass an array of betas directly to the constructor to bypass `beta_start` and `beta_end`.
    """

    order = 1

    @register_to_config
    def __init__(
        self, num_train_timesteps: int = 1000, trained_betas: Optional[Union[np.ndarray, List[float]]] = None
    ):
        # set `betas`, `alphas`, `timesteps`
        self.set_timesteps(num_train_timesteps)

        # standard deviation of the initial noise distribution
        self.init_noise_sigma = 1.0

        # For now we only support F-PNDM, i.e. the runge-kutta method
        # For more information on the algorithm please take a look at the paper: https://arxiv.org/pdf/2202.09778.pdf
        # mainly at formula (9), (12), (13) and the Algorithm 2.
        self.pndm_order = 4

        # running values
        self.ets = []
        self._step_index = None

    @property
    def step_index(self):
        """
        The index counter for current timestep. It will increae 1 after each scheduler step.
        """
        return self._step_index

    def set_timesteps(self, num_inference_steps: int, device: Union[str, torch.device] = None):
        """
        Sets the discrete timesteps used for the diffusion chain (to be run before inference).

        Args:
            num_inference_steps (`int`):
                The number of diffusion steps used when generating samples with a pre-trained model.
            device (`str` or `torch.device`, *optional*):
                The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        """
        self.num_inference_steps = num_inference_steps
        steps = torch.linspace(1, 0, num_inference_steps + 1)[:-1]
        steps = torch.cat([steps, torch.tensor([0.0])])

        if self.config.trained_betas is not None:
            self.betas = torch.tensor(self.config.trained_betas, dtype=torch.float32)
        else:
            self.betas = torch.sin(steps * math.pi / 2) ** 2

        self.alphas = (1.0 - self.betas**2) ** 0.5

        timesteps = (torch.atan2(self.betas, self.alphas) / math.pi * 2)[:-1]
        self.timesteps = timesteps.to(device)

        self.ets = []
        self._step_index = None

    # Copied from diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler._init_step_index
    def _init_step_index(self, timestep):
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.to(self.timesteps.device)

        index_candidates = (self.timesteps == timestep).nonzero()

        # The sigma index that is taken for the **very** first `step`
        # is always the second index (or the last index if there is only 1)
        # This way we can ensure we don't accidentally skip a sigma in
        # case we start in the middle of the denoising schedule (e.g. for image-to-image)
        if len(index_candidates) > 1:
            step_index = index_candidates[1]
        else:
            step_index = index_candidates[0]

        self._step_index = step_index.item()

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        sample: torch.FloatTensor,
        return_dict: bool = True,
    ) -> Union[SchedulerOutput, Tuple]:
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the sample with
        the linear multistep method. It performs one forward pass multiple times to approximate the solution.

        Args:
            model_output (`torch.FloatTensor`):
                The direct output from learned diffusion model.
            timestep (`int`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                A current instance of a sample created by the diffusion process.
            return_dict (`bool`):
                Whether or not to return a [`~schedulers.scheduling_utils.SchedulerOutput`] or tuple.

        Returns:
            [`~schedulers.scheduling_utils.SchedulerOutput`] or `tuple`:
                If return_dict is `True`, [`~schedulers.scheduling_utils.SchedulerOutput`] is returned, otherwise a
                tuple is returned where the first element is the sample tensor.
        """
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )
        if self.step_index is None:
            self._init_step_index(timestep)

        timestep_index = self.step_index
        prev_timestep_index = self.step_index + 1

        ets = sample * self.betas[timestep_index] + model_output * self.alphas[timestep_index]
        self.ets.append(ets)

        if len(self.ets) == 1:
            ets = self.ets[-1]
        elif len(self.ets) == 2:
            ets = (3 * self.ets[-1] - self.ets[-2]) / 2
        elif len(self.ets) == 3:
            ets = (23 * self.ets[-1] - 16 * self.ets[-2] + 5 * self.ets[-3]) / 12
        else:
            ets = (1 / 24) * (55 * self.ets[-1] - 59 * self.ets[-2] + 37 * self.ets[-3] - 9 * self.ets[-4])

        prev_sample = self._get_prev_sample(sample, timestep_index, prev_timestep_index, ets)

        # upon completion increase step index by one
        self._step_index += 1

        if not return_dict:
            return (prev_sample,)

        return SchedulerOutput(prev_sample=prev_sample)

    def scale_model_input(self, sample: torch.FloatTensor, *args, **kwargs) -> torch.FloatTensor:
        """
        Ensures interchangeability with schedulers that need to scale the denoising model input depending on the
        current timestep.

        Args:
            sample (`torch.FloatTensor`):
                The input sample.

        Returns:
            `torch.FloatTensor`:
                A scaled input sample.
        """
        return sample

    def _get_prev_sample(self, sample, timestep_index, prev_timestep_index, ets):
        alpha = self.alphas[timestep_index]
        sigma = self.betas[timestep_index]

        next_alpha = self.alphas[prev_timestep_index]
        next_sigma = self.betas[prev_timestep_index]

        pred = (sample - sigma * ets) / max(alpha, 1e-8)
        prev_sample = next_alpha * pred + ets * next_sigma

        return prev_sample

    def __len__(self):
        return self.config.num_train_timesteps
