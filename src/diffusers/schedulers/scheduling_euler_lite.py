# Copyright 2025 Katherine Crowson and The HuggingFace Team. All rights reserved.
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

from typing import Optional, Tuple, Union

import torch

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import SchedulerMixin, SchedulerOutput


class EulerLiteScheduler(SchedulerMixin, ConfigMixin):
    """First-order Euler ODE sampler over a linear-beta sigma schedule.

    Lite subset of [`EulerDiscreteScheduler`]: epsilon prediction, leading timesteps, and a
    terminal sigma of 0. The update is the Karras Euler step from
    `scheduling_euler_discrete.py` (no ancestral noise, no Karras/exponential/beta sigma
    conversions).

    This model inherits from [`SchedulerMixin`] and [`ConfigMixin`]. Check the superclass
    documentation for the generic methods the library implements for all schedulers such as
    loading and saving.

    Args:
        num_train_timesteps (`int`, defaults to `1000`):
            The number of diffusion steps to train the model.
        beta_start (`float`, defaults to `0.0001`):
            The starting `beta` value of the linear schedule.
        beta_end (`float`, defaults to `0.02`):
            The final `beta` value of the linear schedule.
    """

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
    ):
        self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        # Same sigma definition as EulerDiscreteScheduler: sqrt((1 - bar_alpha) / bar_alpha).
        self.sigmas_train = ((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5
        self.timesteps = torch.arange(num_train_timesteps - 1, -1, -1)
        self.sigmas = torch.cat([self.sigmas_train.flip(0), torch.zeros(1)])
        self.num_inference_steps: Optional[int] = None

    def set_timesteps(self, num_inference_steps: int, device: Union[str, torch.device, None] = None):
        """Set the discrete timesteps used for the denoising loop.

        Leading spacing matches `EulerDiscreteScheduler` with `timestep_spacing="leading"`:
        `t = (arange(num_inference_steps) * (num_train_timesteps // num_inference_steps))`
        reversed. Sigmas are the training-schedule values at those indices plus a trailing 0.

        Args:
            num_inference_steps: Number of diffusion steps used at inference.
            device: Device the timesteps should be moved to.
        """
        self.num_inference_steps = num_inference_steps
        step_ratio = self.config.num_train_timesteps // num_inference_steps
        timesteps = (torch.arange(0, num_inference_steps) * step_ratio).round().flip(0).to(torch.long)
        self.timesteps = timesteps.to(device) if device is not None else timesteps
        sigmas = self.sigmas_train[self.timesteps.cpu()]
        self.sigmas = torch.cat([sigmas, torch.zeros(1, dtype=sigmas.dtype)])

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ) -> Union[SchedulerOutput, Tuple]:
        """Predict the sample at the previous timestep with a first-order Euler ODE step.

        For epsilon prediction this is the update in `EulerDiscreteScheduler.step` with
        `s_churn=0` (no stochastic churn, so `generator` is unused):

            x0 = sample - sigma * epsilon
            d = (sample - x0) / sigma
            prev_sample = sample + d * (sigma_next - sigma)

        Args:
            model_output: Direct output from the learned diffusion model (epsilon).
            timestep: The current discrete timestep in the diffusion chain.
            sample: A current instance of a sample created by the diffusion process.
            generator: A torch.Generator for reproducible sampling. The Euler ODE step is
                deterministic; this argument exists to match the scheduler `step` contract.
            return_dict: Whether to return a SchedulerOutput or a plain tuple.

        Returns:
            SchedulerOutput or tuple with the predicted previous sample.
        """
        schedule = self.timesteps
        if isinstance(timestep, torch.Tensor):
            t_val = timestep.to(device=schedule.device, dtype=schedule.dtype).reshape(-1)[0]
        else:
            t_val = torch.tensor(timestep, device=schedule.device, dtype=schedule.dtype)

        matches = (schedule == t_val).nonzero()
        if len(matches) > 0:
            step_index = int(matches[0].item())
            sigma = self.sigmas[step_index]
            sigma_next = self.sigmas[step_index + 1]
        else:
            t_int = int(t_val.item())
            sigma = self.sigmas_train[t_int]
            if t_int > 0:
                sigma_next = self.sigmas_train[t_int - 1]
            else:
                sigma_next = torch.zeros((), dtype=sigma.dtype, device=sigma.device)

        # Upcast to avoid precision issues when computing prev_sample (EulerDiscreteScheduler).
        sample = sample.to(torch.float32)
        sigma = sigma.to(device=sample.device, dtype=sample.dtype)
        sigma_next = sigma_next.to(device=sample.device, dtype=sample.dtype)

        pred_original_sample = sample - sigma * model_output.to(sample.dtype)
        derivative = (sample - pred_original_sample) / sigma
        prev_sample = sample + derivative * (sigma_next - sigma)
        prev_sample = prev_sample.to(model_output.dtype)

        if not return_dict:
            return (prev_sample,)
        return SchedulerOutput(prev_sample=prev_sample)
