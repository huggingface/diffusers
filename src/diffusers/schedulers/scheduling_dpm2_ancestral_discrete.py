# Copyright 2022 Katherine Crowson, The HuggingFace Team and hlky. All rights reserved.
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
from diffusers.models.unet_2d_condition import UNet2DConditionModel

import numpy as np
import torch

from scipy import integrate

from ..configuration_utils import ConfigMixin, register_to_config
from .scheduling_utils import SchedulerMixin, SchedulerOutput


class DPM2AncestralDiscreteScheduler(SchedulerMixin, ConfigMixin):
    """
    Ancestral sampling with DPM-Solver inspired second-order steps.
    for discrete beta schedules. Based on the original k-diffusion implementation by
    Katherine Crowson:
    https://github.com/crowsonkb/k-diffusion/blob/481677d114f6ea445aa009cf5bd7a9cdee909e47/k_diffusion/sampling.py#L145

    [`~ConfigMixin`] takes care of storing all config attributes that are passed in the scheduler's `__init__`
    function, such as `num_train_timesteps`. They can be accessed via `scheduler.config.num_train_timesteps`.
    [`~ConfigMixin`] also provides general loading and saving functionality via the [`~ConfigMixin.save_config`] and
    [`~ConfigMixin.from_config`] functions.

    Args:
        num_train_timesteps (`int`): number of diffusion steps used to train the model.
        beta_start (`float`): the starting `beta` value of inference.
        beta_end (`float`): the final `beta` value.
        beta_schedule (`str`):
            the beta schedule, a mapping from a beta range to a sequence of betas for stepping the model. Choose from
            `linear` or `scaled_linear`.
        trained_betas (`np.ndarray`, optional):
            option to pass an array of betas directly to the constructor to bypass `beta_start`, `beta_end` etc.
            options to clip the variance used when adding noise to the denoised sample. Choose from `fixed_small`,
            `fixed_small_log`, `fixed_large`, `fixed_large_log`, `learned` or `learned_range`.
        tensor_format (`str`): whether the scheduler expects pytorch or numpy arrays.

    """

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.00085, #sensible defaults
        beta_end: float = 0.012,
        beta_schedule: str = "linear",
        trained_betas: Optional[np.ndarray] = None,
    ):
        if trained_betas is not None:
            self.betas = torch.from_numpy(trained_betas)
        elif beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        elif beta_schedule == "scaled_linear":
            # this schedule is very specific to the latent diffusion model.
            self.betas = (
                torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32) ** 2
            )
        else:
            raise NotImplementedError(f"{beta_schedule} does is not implemented for {self.__class__}")

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        sigmas = np.array(((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5)
        sigmas = np.concatenate([sigmas[::-1], [0.0]]).astype(np.float32)
        self.sigmas = torch.from_numpy(sigmas)

        self.init_noise_sigma = None

        # setable values
        self.num_inference_steps = None
        timesteps = np.arange(0, num_train_timesteps)[::-1].copy()
        self.timesteps = torch.from_numpy(timesteps)
        self.derivatives = []
        self.is_scale_input_called = False

    def set_timesteps(self, num_inference_steps: int, device: Union[str, torch.device] = None):
        """
        Sets the timesteps used for the diffusion chain. Supporting function to be run before inference.

        Args:
            num_inference_steps (`int`):
                the number of diffusion steps used when generating samples with a pre-trained model.
        """
        self.num_inference_steps = num_inference_steps
        self.timesteps = np.linspace(self.num_train_timesteps - 1, 0, num_inference_steps, dtype=float)

        low_idx = np.floor(self.timesteps).astype(int)
        high_idx = np.ceil(self.timesteps).astype(int)
        frac = np.mod(self.timesteps, 1.0)
        sigmas = np.array(((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5)
        sigmas = (1 - frac) * sigmas[low_idx] + frac * sigmas[high_idx]
        sigmas = np.concatenate([sigmas, [0.0]]).astype(np.float32)
        self.sigmas = torch.from_numpy(sigmas)
        self.timesteps = torch.from_numpy(self.timesteps)
        self.init_noise_sigma = self.sigmas[0]
        self.derivatives = []

    def scale_model_input(
        self, sample: torch.FloatTensor, 
        timestep: Union[float, torch.FloatTensor], step_index: Union[int, torch.IntTensor],
        **kwargs) -> torch.FloatTensor:
        """
        Ensures interchangeability with schedulers that need to scale the denoising model input depending on the
        current timestep.

        Args:
            sample (`torch.FloatTensor`): input sample
            timestep (`int`, optional): current timestep

        Returns:
            `torch.FloatTensor`: scaled input sample
        """
        sigma = self.sigmas[step_index]
        sample = sample / ((sigma**2 + 1) ** 0.5)
        return sample

    def step(
        self,
        model_output: Union[torch.FloatTensor, np.ndarray],
        timestep: Union[float, torch.FloatTensor],
        step_index: Union[int, torch.IntTensor],
        sample: Union[torch.FloatTensor, np.ndarray],
        # unet: UNet2DConditionModel = None,
        # encoder_hidden_states: torch.Tensor,
        s_churn: float = 0.,
        s_tmin:  float = 0.,
        s_tmax: float = float('inf'),
        s_noise:  float = 1.,
        return_dict: bool = True,
    ) -> Union[SchedulerOutput, Tuple]:
        """
        Predict the sample at the previous timestep by reversing the SDE. Core function to propagate the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.FloatTensor` or `np.ndarray`): direct output from learned diffusion model.
            timestep (`int`): current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor` or `np.ndarray`):
                current instance of sample being created by diffusion process.
            s_churn (`float`)
            s_tmin  (`float`)
            s_tmax  (`float`)
            s_noise (`float`)
            return_dict (`bool`): option for returning tuple rather than SchedulerOutput class

        Returns:
            [`~schedulers.scheduling_utils.SchedulerOutput`] or `tuple`:
            [`~schedulers.scheduling_utils.SchedulerOutput`] if `return_dict` is True, otherwise a `tuple`. When
            returning a tuple, the first element is the sample tensor.

        """
        sigma = self.sigmas[step_index]

        # 1. compute predicted original sample (x_0) from sigma-scaled predicted noise
        pred_original_sample = sample - sigma * model_output
        sigma_from = self.sigmas[step_index]
        sigma_to = self.sigmas[step_index + 1]
        sigma_up = (sigma_to ** 2 * (sigma_from ** 2 - sigma_to ** 2) / sigma_from ** 2) ** 0.5
        sigma_down = (sigma_to ** 2 - sigma_up ** 2) ** 0.5
        # 2. Convert to an ODE derivative
        derivative = (sample - pred_original_sample) / sigma
        self.derivatives.append(derivative)
        # Midpoint method, where the midpoint is chosen according to a rho=3 Karras schedule
        sigma_mid = ((sigma ** (1 / 3) + sigma_down ** (1 / 3)) / 2) ** 3

        dt_1 = sigma_mid - sigma
        dt_2 = sigma_down - sigma
        sample_2 = sample + derivative * dt_1
        pred_original_sample_2 = sample_2 - sigma_mid * model_output
        derivative_2 = (sample_2 - pred_original_sample_2) / sigma_mid
        sample = sample + derivative_2 * dt_2
        sample = sample + torch.randn_like(sample) * sigma_up

        prev_sample = sample
        # if sigma_to == 0:
        #     # Euler method
        #     dt = sigma_down - sigma
        #     sample = sample + derivative * dt
        # else:
        #     # DPM-Solver-2
        #     # Midpoint method, where the midpoint is chosen according to a rho=3 Karras schedule
        #     #     sigma_mid = sigmas[i].log().lerp(sigma_down.log(), 0.5).exp()
        #     sigma_mid = sigma.log().lerp(sigma_down.log(), 0.5).exp()
        #     # sigma_mid = ((sigma ** (1 / 3) + sigma_down ** (1 / 3)) / 2) ** 3
        #     dt_1 = sigma_mid - sigma
        #     dt_2 = sigma_down - sigma
        #     sample_2 = sample + derivative * dt_1
        #     pred_original_sample_2 = sample_2 - sigma_mid * model_output
            
        #     derivative_2 = unet(pred_original_sample_2, sigma_mid, encoder_hidden_states=encoder_hidden_states)
        #     sample = sample + derivative_2 * dt_2
        #     sample = sample + torch.randn_like(sample) * sigma_up

        # prev_sample = sample

        # if sigma_down == 0:
        #     # Euler method
        #     dt = sigma_down - sigmas[i]
        #     x = x + d * dt
        # else:
        #     # DPM-Solver-2
        #     sigma_mid = sigmas[i].log().lerp(sigma_down.log(), 0.5).exp()
        #     dt_1 = sigma_mid - sigmas[i]
        #     dt_2 = sigma_down - sigmas[i]
        #     x_2 = x + d * dt_1
        #     denoised_2 = model(x_2, sigma_mid * s_in, **extra_args)
        #     d_2 = to_d(x_2, sigma_mid, denoised_2)
        #     x = x + d_2 * dt_2
        #     x = x + torch.randn_like(x) * sigma_up

        if not return_dict:
            return (prev_sample,)

        return SchedulerOutput(prev_sample=prev_sample)

    def add_noise(
        self,
        original_samples: Union[torch.FloatTensor, np.ndarray],
        noise: Union[torch.FloatTensor, np.ndarray],
        timesteps: Union[torch.IntTensor, np.ndarray],
    ) -> Union[torch.FloatTensor, np.ndarray]:
        # Make sure sigmas and timesteps have the same device and dtype as original_samples
        self.sigmas = self.sigmas.to(device=original_samples.device, dtype=original_samples.dtype)
        self.timesteps = self.timesteps.to(original_samples.device)
        sigma = self.sigmas[timesteps].flatten()
        while len(sigma.shape) < len(original_samples.shape):
            sigma = sigma.unsqueeze(-1)

        noisy_samples = original_samples + noise * sigma
        return noisy_samples

    def __len__(self):
        return self.config.num_train_timesteps
