# Copyright 2024 NVIDIA and The HuggingFace Team. All rights reserved.
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


from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import torch

from ...configuration_utils import ConfigMixin, register_to_config
from ...utils import BaseOutput
from ...utils.torch_utils import randn_tensor
from ..scheduling_utils import SchedulerMixin


@dataclass
class KarrasVeOutput(BaseOutput):
    """
    Output class for the scheduler's step function output.

    Args:
        prev_sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample (x_{t-1}) of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
        derivative (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` for images):
            Derivative of predicted original image sample (x_0).
        pred_original_sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` for images):
            The predicted denoised sample (x_{0}) based on the model output from the current timestep.
            `pred_original_sample` can be used to preview progress or for guidance.
    """

    prev_sample: torch.FloatTensor
    derivative: torch.FloatTensor
    pred_original_sample: Optional[torch.FloatTensor] = None


class KarrasVeScheduler(SchedulerMixin, ConfigMixin):
    """
    A stochastic scheduler tailored to variance-expanding models.

    This model inherits from [`SchedulerMixin`] and [`ConfigMixin`]. Check the superclass documentation for the generic
    methods the library implements for all schedulers such as loading and saving.

    <Tip>

    For more details on the parameters, see [Appendix E](https://arxiv.org/abs/2206.00364). The grid search values used
    to find the optimal `{s_noise, s_churn, s_min, s_max}` for a specific model are described in Table 5 of the paper.

    </Tip>

    Args:
        sigma_min (`float`, defaults to 0.02):
            The minimum noise magnitude.
        sigma_max (`float`, defaults to 100):
            The maximum noise magnitude.
        s_noise (`float`, defaults to 1.007):
            The amount of additional noise to counteract loss of detail during sampling. A reasonable range is [1.000,
            1.011].
        s_churn (`float`, defaults to 80):
            The parameter controlling the overall amount of stochasticity. A reasonable range is [0, 100].
        s_min (`float`, defaults to 0.05):
            The start value of the sigma range to add noise (enable stochasticity). A reasonable range is [0, 10].
        s_max (`float`, defaults to 50):
            The end value of the sigma range to add noise. A reasonable range is [0.2, 80].
    """

    order = 2

    @register_to_config
    def __init__(
        self,
        sigma_min: float = 0.02,
        sigma_max: float = 100,
        s_noise: float = 1.007,
        s_churn: float = 80,
        s_min: float = 0.05,
        s_max: float = 50,
    ):
        # standard deviation of the initial noise distribution
        self.init_noise_sigma = sigma_max

        # setable values
        self.num_inference_steps: int = None
        self.timesteps: np.IntTensor = None
        self.schedule: torch.FloatTensor = None  # sigma(t_i)

    def scale_model_input(self, sample: torch.FloatTensor, timestep: Optional[int] = None) -> torch.FloatTensor:
        """
        Ensures interchangeability with schedulers that need to scale the denoising model input depending on the
        current timestep.

        Args:
            sample (`torch.FloatTensor`):
                The input sample.
            timestep (`int`, *optional*):
                The current timestep in the diffusion chain.

        Returns:
            `torch.FloatTensor`:
                A scaled input sample.
        """
        return sample

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
        timesteps = np.arange(0, self.num_inference_steps)[::-1].copy()
        self.timesteps = torch.from_numpy(timesteps).to(device)
        schedule = [
            (
                self.config.sigma_max**2
                * (self.config.sigma_min**2 / self.config.sigma_max**2) ** (i / (num_inference_steps - 1))
            )
            for i in self.timesteps
        ]
        self.schedule = torch.tensor(schedule, dtype=torch.float32, device=device)

    def add_noise_to_input(
        self, sample: torch.FloatTensor, sigma: float, generator: Optional[torch.Generator] = None
    ) -> Tuple[torch.FloatTensor, float]:
        """
        Explicit Langevin-like "churn" step of adding noise to the sample according to a `gamma_i ≥ 0` to reach a
        higher noise level `sigma_hat = sigma_i + gamma_i*sigma_i`.

        Args:
            sample (`torch.FloatTensor`):
                The input sample.
            sigma (`float`):
            generator (`torch.Generator`, *optional*):
                A random number generator.
        """
        if self.config.s_min <= sigma <= self.config.s_max:
            gamma = min(self.config.s_churn / self.num_inference_steps, 2**0.5 - 1)
        else:
            gamma = 0

        # sample eps ~ N(0, S_noise^2 * I)
        eps = self.config.s_noise * randn_tensor(sample.shape, generator=generator).to(sample.device)
        sigma_hat = sigma + gamma * sigma
        sample_hat = sample + ((sigma_hat**2 - sigma**2) ** 0.5 * eps)

        return sample_hat, sigma_hat

    def step(
        self,
        model_output: torch.FloatTensor,
        sigma_hat: float,
        sigma_prev: float,
        sample_hat: torch.FloatTensor,
        return_dict: bool = True,
    ) -> Union[KarrasVeOutput, Tuple]:
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.FloatTensor`):
                The direct output from learned diffusion model.
            sigma_hat (`float`):
            sigma_prev (`float`):
            sample_hat (`torch.FloatTensor`):
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~schedulers.scheduling_karras_ve.KarrasVESchedulerOutput`] or `tuple`.

        Returns:
            [`~schedulers.scheduling_karras_ve.KarrasVESchedulerOutput`] or `tuple`:
                If return_dict is `True`, [`~schedulers.scheduling_karras_ve.KarrasVESchedulerOutput`] is returned,
                otherwise a tuple is returned where the first element is the sample tensor.

        """

        pred_original_sample = sample_hat + sigma_hat * model_output
        derivative = (sample_hat - pred_original_sample) / sigma_hat
        sample_prev = sample_hat + (sigma_prev - sigma_hat) * derivative

        if not return_dict:
            return (sample_prev, derivative)

        return KarrasVeOutput(
            prev_sample=sample_prev, derivative=derivative, pred_original_sample=pred_original_sample
        )

    def step_correct(
        self,
        model_output: torch.FloatTensor,
        sigma_hat: float,
        sigma_prev: float,
        sample_hat: torch.FloatTensor,
        sample_prev: torch.FloatTensor,
        derivative: torch.FloatTensor,
        return_dict: bool = True,
    ) -> Union[KarrasVeOutput, Tuple]:
        """
        Corrects the predicted sample based on the `model_output` of the network.

        Args:
            model_output (`torch.FloatTensor`):
                The direct output from learned diffusion model.
            sigma_hat (`float`): TODO
            sigma_prev (`float`): TODO
            sample_hat (`torch.FloatTensor`): TODO
            sample_prev (`torch.FloatTensor`): TODO
            derivative (`torch.FloatTensor`): TODO
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~schedulers.scheduling_ddpm.DDPMSchedulerOutput`] or `tuple`.

        Returns:
            prev_sample (TODO): updated sample in the diffusion chain. derivative (TODO): TODO

        """
        pred_original_sample = sample_prev + sigma_prev * model_output
        derivative_corr = (sample_prev - pred_original_sample) / sigma_prev
        sample_prev = sample_hat + (sigma_prev - sigma_hat) * (0.5 * derivative + 0.5 * derivative_corr)

        if not return_dict:
            return (sample_prev, derivative)

        return KarrasVeOutput(
            prev_sample=sample_prev, derivative=derivative, pred_original_sample=pred_original_sample
        )

    def add_noise(self, original_samples, noise, timesteps):
        raise NotImplementedError()
