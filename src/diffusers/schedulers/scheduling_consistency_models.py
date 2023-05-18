# Copyright 2023 NVIDIA and The HuggingFace Team. All rights reserved.
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

from ..configuration_utils import ConfigMixin, register_to_config
from ..utils import BaseOutput, randn_tensor
from .scheduling_utils import SchedulerMixin


def append_zero(x):
    return torch.cat([x, x.new_zeros([1])])


@dataclass
class CMStochasticIterativeSchedulerOutput(BaseOutput):
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
    # derivative: torch.FloatTensor
    # pred_original_sample: Optional[torch.FloatTensor] = None


class CMStochasticIterativeScheduler(SchedulerMixin, ConfigMixin):
    """
    Stochastic sampling from Karras et al. [1] tailored to the Variance-Expanding (VE) models [2]. Use Algorithm 2 and
    the VE column of Table 1 from [1] for reference.
    [1] Karras, Tero, et al. "Elucidating the Design Space of Diffusion-Based Generative Models."
    https://arxiv.org/abs/2206.00364 [2] Song, Yang, et al. "Score-based generative modeling through stochastic
    differential equations." https://arxiv.org/abs/2011.13456
    [`~ConfigMixin`] takes care of storing all config attributes that are passed in the scheduler's `__init__`
    function, such as `num_train_timesteps`. They can be accessed via `scheduler.config.num_train_timesteps`.
    [`SchedulerMixin`] provides general loading and saving functionality via the [`SchedulerMixin.save_pretrained`] and
    [`~SchedulerMixin.from_pretrained`] functions.
    For more details on the parameters, see the original paper's Appendix E.: "Elucidating the Design Space of
    Diffusion-Based Generative Models." https://arxiv.org/abs/2206.00364. The grid search values used to find the
    optimal {s_noise, s_churn, s_min, s_max} for a specific model are described in Table 5 of the paper.
    Args:
        sigma_min (`float`): minimum noise magnitude
        sigma_max (`float`): maximum noise magnitude
        s_noise (`float`): the amount of additional noise to counteract loss of detail during sampling.
            A reasonable range is [1.000, 1.011].
        s_churn (`float`): the parameter controlling the overall amount of stochasticity.
            A reasonable range is [0, 100].
        s_min (`float`): the start value of the sigma range where we add noise (enable stochasticity).
            A reasonable range is [0, 10].
        s_max (`float`): the end value of the sigma range where we add noise.
            A reasonable range is [0.2, 80].
    """

    @register_to_config
    def __init__(
        self,
        sigma_data: float = 0.5,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        rho: float = 7.0,
        s_noise: float = 1.0,
        s_churn: float = 0.0,
        s_min: float = 0.0,
        s_max: float = float('inf'),
    ):
        # standard deviation of the initial noise distribution
        self.init_noise_sigma = sigma_max

        # setable values
        self.num_inference_steps: int = None
        self.timesteps: np.IntTensor = None
        self.schedule: torch.FloatTensor = None  # sigma(t_i)

        self.sigma_data = sigma_data
        self.rho = rho
    
    def index_for_timestep(self, timestep, schedule_timesteps=None):
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps

        indices = (schedule_timesteps == timestep).nonzero()
        return indices.item()

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
    
    def get_sigmas_karras(self):
        """Constructs the noise schedule of Karras et al. (2022)."""
        ramp = np.linspace(0, 1, self.num_inference_steps)
        min_inv_rho = self.sigma_min ** (1 / self.rho)
        max_inv_rho = self.sigma_max ** (1 / self.rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** self.rho
        return append_zero(sigmas)

    def set_timesteps(self, num_inference_steps: int, device: Union[str, torch.device] = None):
        """
        Sets the continuous timesteps used for the diffusion chain. Supporting function to be run before inference.
        Args:
            num_inference_steps (`int`):
                the number of diffusion steps used when generating samples with a pre-trained model.
        """
        self.num_inference_steps = num_inference_steps
        # TODO: how should timesteps be set? the original code seems to either solely work in sigma space or have
        # hardcoded timesteps (see e.g. https://github.com/openai/consistency_models/blob/main/scripts/launch.sh#L74)
        # TODO: should add num_train_timesteps here???
        timesteps = np.arange(0, self.num_inference_steps)[::-1].copy()
        sigmas = self.get_sigmas_karras()

        self.timesteps = torch.from_numpy(timesteps).to(device)
        self.sigmas = torch.tensor(sigmas, dtype=torch.float32, device=device)
    
    def add_noise(self, original_samples, noise, timesteps):
        """Add noise for training."""
        raise NotImplementedError()

    def add_noise_to_input(
        self,
        sample: torch.FloatTensor,
        sigma: float,
        generator: Optional[torch.Generator] = None
    ) -> Tuple[torch.FloatTensor, float]:
        """
        Explicit Langevin-like "churn" step of adding noise to the sample according to a factor gamma_i ≥ 0 to reach a
        higher noise level sigma_hat = sigma_i + gamma_i*sigma_i.
        TODO Args:
        """
        sigma_min = self.config.sigma_min
        sigma_max = self.config.sigma_max

        step_idx = (self.sigmas == sigma).nonzero().item()
        sigma_hat = self.sigmas[step_idx + 1].clamp(min=sigma_min, max=sigma_max)

        # sample z ~ N(0, s_noise^2 * I)
        z = self.config.s_noise * randn_tensor(sample.shape, generator=generator, device=sample.device)

        # tau = sigma_hat, eps = sigma_min
        sample_hat = sample + ((sigma_hat**2 - sigma_min**2) ** 0.5 * z)

        return sample_hat, sigma_hat

    def step(
        self,
        model_output: torch.FloatTensor,
        sigma_hat: float,
        sigma_prev: float,
        sample_hat: torch.FloatTensor,
        return_dict: bool = True,
    ) -> Union[CMStochasticIterativeSchedulerOutput, Tuple]:
        """
        Predict the sample at the previous timestep by reversing the SDE. Core function to propagate the diffusion
        process from the learned model outputs (most often the predicted noise).
        Args:
            model_output (`torch.FloatTensor`): direct output from learned diffusion model.
            sigma_hat (`float`): TODO
            sigma_prev (`float`): TODO
            sample_hat (`torch.FloatTensor`): TODO
            return_dict (`bool`): option for returning tuple rather than KarrasVeOutput class
            KarrasVeOutput: updated sample in the diffusion chain and derivative (TODO double check).
        Returns:
            [`~schedulers.scheduling_karras_ve.KarrasVeOutput`] or `tuple`:
            [`~schedulers.scheduling_karras_ve.KarrasVeOutput`] if `return_dict` is True, otherwise a `tuple`. When
            returning a tuple, the first element is the sample tensor.
        """
        # Assume model output is the consistency model evaluated at sample_hat.
        sample_prev = model_output

        if not return_dict:
            return (sample_prev,)

        return CMStochasticIterativeSchedulerOutput(
            prev_sample=sample_prev,
        )
    
