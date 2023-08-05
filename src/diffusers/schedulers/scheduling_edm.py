# Copyright 2023 The HuggingFace Team. All rights reserved.
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
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from ..configuration_utils import ConfigMixin, register_to_config
from ..utils import BaseOutput, logging, randn_tensor
from .scheduling_utils import SchedulerMixin


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class KarrasEDMSchedulerOutput(BaseOutput):
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


class KarrasEDMScheduler(SchedulerMixin, ConfigMixin):
    """
    Implements the "EDM" column of Table 1 and Algorithm 2 in Karras et al. 2022 [1]/

    [1] Karras, Tero, et al. "Elucidating the Design Space of Diffusion-Based Generative Models."
    https://arxiv.org/abs/2206.00364

    [`~ConfigMixin`] takes care of storing all config attributes that are passed in the scheduler's `__init__`
    function, such as `num_train_timesteps`. They can be accessed via `scheduler.config.num_train_timesteps`.
    [`SchedulerMixin`] provides general loading and saving functionality via the [`SchedulerMixin.save_pretrained`] and
    [`~SchedulerMixin.from_pretrained`] functions.

    Args:
        num_train_timesteps (`int`, *optional*, defaults t0 40):
            Number of diffusion steps used to train the model.
        sigma_min (`float`, *optional*, defaults to 0.002):
            Minimum noise magnitude in the sigma schedule. This was set to 0.002 in the EDM paper [1]; a reasonable
            range is [0, 10].
        sigma_max (`float`, *optional*, defaults to 80.0):
            Maximum noise magnitude in the sigma schedule. This was set to 80.0 in the EDM paper [1]; a reasonable
            range is [0.2, 100]. (TODO: is this correct?)
        sigma_data (`float`, *optional*, defaults to 0.5):
            The standard deviation of the data distribution. This is set to 0.5 in the EDM paper [1].
        s_noise (`float`, *optional*, defaults to 1.0):
            The amount of additional noise to counteract loss of detail during sampling. This was set to 1.0 in the
            EDM paper [1]; a reasonable range is [1.000, 1.011].
        s_churn (`float`, *optional*, defaults to 0.0):
            The parameter controlling the overall amount of stochasticity if we add noise during sampling. Defaults to
            0.0; a reasonable range is [0, 100].
        s_min (`float`, *optional*, defaults to 0.0):
            The start value of the sigma range where we add noise. Defaults to 0.0; a reasonable range is [0, 10].
        s_max (`float`, *optional*, defaults to `float('inf')`):
            The end value of the sigma range where we add noise. Defaults to `float('inf')`; a reasonable
            range is [0.2, float('inf')].
        rho (`float`, *optional*, defaults to 7.0):
            The rho parameter used for calculating the Karras sigma schedule, which is set to 7.0 in the EDM
            paper [1].
        clip_denoised (`bool`, *optional*, defaults to `True`):
            Whether to clip the denoised outputs to `(-1, 1)`. Defaults to `True`.
        clip_sample (`bool`, *optional*, defaults to `True`):
            option to clip predicted sample for numerical stability.
        clip_sample_range (`float`, *optional*, defaults to `1.0`):
            the maximum magnitude for sample clipping. Valid only when `clip_sample=True`.
        precondition_type (`str`, *optional*, defaults to `edm`):
            The preconditioning type for the model, which determines the preconditioning scalings used. See the
            "Network and precondtioning" section of Table 1 in [1]. Currently only `edm` is supported.
        prediction_type (`str`, *optional*, defaults to `sample`):
            prediction type of the scheduler function; currently only `sample` (directly predicting the noisy sample`)
            is supported (the prediction type used in the original EDM paper/implementation).
    """

    order = 1 # TODO: should be 2?

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 40,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        sigma_data: float = 0.5,
        s_noise: float = 1.0,
        s_churn: float = 0.0,
        s_min: float = 0.0,
        s_max: float = float('inf'),
        rho: float = 7.0,
        clip_denoised: bool = True,
        clip_sample: Optional[bool] = False,
        clip_sample_range: float = 1.0,
        precondition_type: str = "edm",
        prediction_type: str = "sample",
    ):
        # Initial latents are not scaled.
        self.init_noise_sigma = 1.0

        # Follows timestep discretization in edm.generate.edm_sampler
        timesteps = torch.arange(num_train_timesteps)
        ramp = timesteps / (num_train_timesteps - 1)
        sigmas = self._convert_to_karras(ramp)
        # Apply noise preconditioning within scheduler
        timesteps = self.sigma_to_t(sigmas)

        # setable values
        self.num_inference_steps = None
        self.sigmas = torch.from_numpy(sigmas)
        self.timesteps = torch.from_numpy(timesteps)
        self.custom_timesteps = False
        self.is_scale_input_called = False

    def index_for_timestep(self, timestep, schedule_timesteps=None):
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps

        indices = (schedule_timesteps == timestep).nonzero()
        return indices.item()
    
    def index_for_sigma(self, sigma, sigma_schedule=None):
        if sigma_schedule is None:
            sigma_schedule = self.sigmas
        
        indices = (sigma_schedule == sigma).nonzero()
        return indices.item()
    
    def _set_precondition_type(self, precondition_type=None):
        if precondition_type is None:
            precondition_type = self.config.precondition_type
        
        if precondition_type not in ["edm", "cm_edm"]:
            raise ValueError(f"Precondition type {precondition_type} is not supported.")
        
        return precondition_type
    
    def precondition_noise(self, sigmas: Union[float, np.ndarray], precondition_type=None):
        precondition_type = self._set_precondition_type(precondition_type)

        if precondition_type == "edm":
            scaled_noise = 0.25 * np.log(sigmas)
        elif precondition_type == "cm_edm":
            scaled_noise = 1000 * 0.25 * np.log(sigmas + 1e-44)
        
        return scaled_noise
    
    def precondition_inputs(self, sample, sigma, precondition_type=None):
        precondition_type = self._set_precondition_type(precondition_type)

        if precondition_type in ["edm", "cm_edm"]:
            scaled_sample = sample / ((sigma**2 + self.config.sigma_data**2) ** 0.5)

        return scaled_sample
    
    def precondition_outputs(self, sample, model_output, sigma, precondition_type=None):
        precondition_type = self._set_precondition_type(precondition_type)

        if self.config.precondition_type in ["edm", "cm_edm"]:
            sigma_data = self.config.sigma_data
            c_skip = sigma_data**2 / (sigma**2 + sigma_data**2)
            c_out = sigma * sigma_data / (sigma**2 + sigma_data**2) ** 0.5
        
        denoised = c_skip * sample + c_out * model_output
        return denoised

    def scale_model_input(
        self, sample: torch.FloatTensor, timestep: Union[float, torch.FloatTensor]
    ) -> torch.FloatTensor:
        """
        Preconditions the sample input to the neural network $F_\theta$, following c_in(sigma) for the EDM column in
        Table 1.

        Args:
            sample (`torch.FloatTensor`): input sample
            timestep (`float` or `torch.FloatTensor`): the current timestep in the diffusion chain
        Returns:
            `torch.FloatTensor`: scaled input sample
        """
        # Get sigma corresponding to timestep
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.to(self.timesteps.device)
        step_idx = self.index_for_timestep(timestep)
        sigma = self.sigmas[step_idx]

        sample = self.precondition_inputs(sample, sigma)

        self.is_scale_input_called = True
        return sample

    def sigma_to_t(self, sigmas: Union[float, np.ndarray]):
        """
        Preconditions the timestep input tothe neural network $F_\theta$, following c_noise(sigma) for the EDM column
        in Table 1.

        Args:
            sigmas (`float` or `np.ndarray`): single Karras sigma or array of Karras sigmas
        Returns:
            `float` or `np.ndarray`: scaled input timestep or scaled input timestep array
        """
        if not isinstance(sigmas, np.ndarray):
            sigmas = np.array(sigmas, dtype=np.float64)

        timesteps = self.precondition_noise(sigmas)

        return timesteps

    def set_timesteps(
        self,
        num_inference_steps: Optional[int] = None,
        device: Union[str, torch.device] = None,
        timesteps: Optional[List[int]] = None,
    ):
        """
        Sets the timesteps used for the diffusion chain. Supporting function to be run before inference.

        Args:
            num_inference_steps (`int`):
                the number of diffusion steps used when generating samples with a pre-trained model.
            device (`str` or `torch.device`, optional):
                the device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
            timesteps (`List[int]`, optional):
                custom timesteps used to support arbitrary spacing between timesteps. If `None`, then the default
                timestep spacing strategy of equal spacing between timesteps is used. If passed, `num_inference_steps`
                must be `None`.
        """
        if num_inference_steps is None and timesteps is None:
            raise ValueError("Exactly one of `num_inference_steps` or `timesteps` must be supplied.")

        if num_inference_steps is not None and timesteps is not None:
            raise ValueError("Can only pass one of `num_inference_steps` or `timesteps`.")

        # Follow DDPMScheduler custom timesteps logic
        if timesteps is not None:
            for i in range(1, len(timesteps)):
                if timesteps[i] >= timesteps[i - 1]:
                    raise ValueError("`timesteps` must be in descending order.")

            if timesteps[0] >= self.config.num_train_timesteps:
                raise ValueError(
                    f"`timesteps` must start before `self.config.train_timesteps`:"
                    f" {self.config.num_train_timesteps}."
                )

            timesteps = np.array(timesteps, dtype=np.int64)
            self.custom_timesteps = True
        else:
            if num_inference_steps > self.config.num_train_timesteps:
                raise ValueError(
                    f"`num_inference_steps`: {num_inference_steps} cannot be larger than `self.config.train_timesteps`:"
                    f" {self.config.num_train_timesteps} as the unet model trained with this scheduler can only handle"
                    f" maximal {self.config.num_train_timesteps} timesteps."
                )

            self.num_inference_steps = num_inference_steps

            step_ratio = self.config.num_train_timesteps // self.num_inference_steps
            timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
            self.custom_timesteps = False

        # Map timesteps to Karras sigmas
        num_train_timesteps = self.config.num_train_timesteps
        # Ramp function should be in increasing order, unlike timesteps which should be in decreasing order
        ramp = timesteps[::-1].copy()
        ramp = ramp / (num_train_timesteps - 1)
        sigmas = self._convert_to_karras(ramp)
        # Apply noise preconditioning c_noise(sigma)
        timesteps = self.sigma_to_t(sigmas)

        # TODO: should append 0.0 to sigmas instead of sigma_min?
        # see https://github.com/NVlabs/edm/blob/main/generate.py#L37
        sigmas = np.concatenate([sigmas, [0.0]]).astype(np.float32)
        self.sigmas = torch.from_numpy(sigmas).to(device=device)

        if str(device).startswith("mps"):
            # mps does not support float64
            self.timesteps = torch.from_numpy(timesteps).to(device, dtype=torch.float32)
        else:
            self.timesteps = torch.from_numpy(timesteps).to(device=device)

    # Modified _convert_to_karras implementation that takes in ramp as argument
    def _convert_to_karras(self, ramp):
        """Constructs the noise schedule of Karras et al. (2022)."""

        sigma_min: float = self.config.sigma_min
        sigma_max: float = self.config.sigma_max

        rho = self.config.rho
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        return sigmas

    def step(
        self,
        model_output: torch.FloatTensor,
        sigma_hat: float,
        sample_hat: torch.FloatTensor,
        sigma_prev: float, 
        return_dict: bool = True,
    ) -> Union[KarrasEDMSchedulerOutput, Tuple]:
        """
        Predict the sample at the previous timestep by reversing the SDE. Core function to propagate the diffusion
        process from the learned model outputs (most often the predicted noise).

        This takes an Euler step to reverse the ODE from sigma_hat to sigma (the Karras sigma corresponding to
        timestep).

        Args:
            model_output (`torch.FloatTensor`): direct output from learned diffusion model.
            timestep (`float`): current timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                current instance of sample being created by diffusion process.
            return_dict (`bool`): option for returning tuple rather than EulerDiscreteSchedulerOutput class
        Returns:
            [`~schedulers.scheduling_utils.CMStochasticIterativeSchedulerOutput`] or `tuple`:
            [`~schedulers.scheduling_utils.CMStochasticIterativeSchedulerOutput`] if `return_dict` is True, otherwise a
            `tuple`. When returning a tuple, the first element is the sample tensor.
        """
        # 1. Convert network output to denoiser output using output preconditioning scalings c_skip, c_out
        denoised = self.precondition_outputs(sample_hat, model_output, sigma_hat)
        # Used in consistency models EDM implementation
        if self.config.clip_denoised:
            denoised = denoised.clamp(-1, 1)

        # 2. Handle prediction types
        # TODO: not sure how to support other prediction types like `epsilon`
        if self.config.prediction_type == "sample":
            pred_original_sample = denoised
        else:
            raise ValueError(
                f"prediction_type {self.config.prediction_type} must be `sample` (for now)."
            )
        
        # TODO: Kind of weird to have two clipping steps but this follows HeunDiscreteScheduler
        if self.config.clip_sample:
            pred_original_sample = pred_original_sample.clamp(
                -self.config.clip_sample_range, self.config.clip_sample_range
            )
        
        # 3. Get Karras ODE derivative at sigma_hat.
        derivative = (sample_hat - pred_original_sample) / sigma_hat

        # 4. Take a Euler step backwards from sigma to sigma_prev.
        prev_sample = sample_hat + (sigma_prev - sigma_hat) * derivative

        if not return_dict:
            return (prev_sample, derivative)

        return KarrasEDMSchedulerOutput(
            prev_sample=prev_sample, derivative=derivative, pred_original_sample=pred_original_sample
        )
    
    def step_correct(
        self,
        model_output: torch.FloatTensor,
        sigma_prev: float,
        sample_prev: torch.FloatTensor,
        sigma_hat: float,
        sample_hat: torch.FloatTensor,
        derivative: torch.FloatTensor,
        return_dict: bool = True,
    ) -> Union[KarrasEDMSchedulerOutput, Tuple]:
        """
        Correct the predicted sample based on the output model_output of the network. TODO complete description

        Args:
            model_output (`torch.FloatTensor`): direct output from learned diffusion model.
            sigma_hat (`float`): TODO
            sigma_prev (`float`): TODO
            sample_hat (`torch.FloatTensor`): TODO
            sample_prev (`torch.FloatTensor`): TODO
            derivative (`torch.FloatTensor`): TODO
            return_dict (`bool`): option for returning tuple rather than KarrasVeOutput class

        Returns:
            prev_sample (TODO): updated sample in the diffusion chain. derivative (TODO): TODO

        """
        # 1. Convert network output to denoiser output using output preconditioning scalings c_skip, c_out
        denoised = self.precondition_outputs(sample_prev, model_output, sigma_prev)
        # Used in consistency models EDM implementation
        if self.config.clip_denoised:
            denoised = denoised.clamp(-1, 1)

        # 2. Handle prediction types
        # TODO: not sure how to support other prediction types like `epsilon`
        if self.config.prediction_type == "sample":
            pred_original_sample = denoised
        else:
            raise ValueError(
                f"prediction_type {self.config.prediction_type} must be `sample` (for now)."
            )
        
        # TODO: Kind of weird to have two clipping steps but this follows HeunDiscreteScheduler
        if self.config.clip_sample:
            pred_original_sample = pred_original_sample.clamp(
                -self.config.clip_sample_range, self.config.clip_sample_range
            )
        
        # 3. Get the Karras ODE derivative at sigma_prev.
        derivative_corr = (sample_prev - pred_original_sample) / sigma_prev

        # 4. Apply 2nd order correction.
        prev_sample = sample_hat + (sigma_prev - sigma_hat) * (0.5 * derivative + 0.5 * derivative_corr)

        if not return_dict:
            return (prev_sample, derivative)

        return KarrasEDMSchedulerOutput(
            prev_sample=prev_sample, derivative=derivative, pred_original_sample=pred_original_sample
        )
    
    # TODO: change to match noise added for EDM training???
    # Copied from diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler.add_noise
    def add_noise(
        self,
        original_samples: torch.FloatTensor,
        noise: torch.FloatTensor,
        timesteps: torch.FloatTensor,
    ) -> torch.FloatTensor:
        # Make sure sigmas and timesteps have the same device and dtype as original_samples
        sigmas = self.sigmas.to(device=original_samples.device, dtype=original_samples.dtype)
        if original_samples.device.type == "mps" and torch.is_floating_point(timesteps):
            # mps does not support float64
            schedule_timesteps = self.timesteps.to(original_samples.device, dtype=torch.float32)
            timesteps = timesteps.to(original_samples.device, dtype=torch.float32)
        else:
            schedule_timesteps = self.timesteps.to(original_samples.device)
            timesteps = timesteps.to(original_samples.device)

        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < len(original_samples.shape):
            sigma = sigma.unsqueeze(-1)

        noisy_samples = original_samples + noise * sigma
        return noisy_samples
    
    # Copied from diffusers.schedulers.scheduling_euler_discrete.KarrasVeScheduler.add_noise_to_input
    def add_noise_to_input(
        self, sample: torch.FloatTensor, sigma: float, generator: Optional[torch.Generator] = None
    ) -> Tuple[torch.FloatTensor, float]:
        """
        Explicit Langevin-like "churn" step of adding noise to the sample according to a factor gamma_i ≥ 0 to reach a
        higher noise level sigma_hat = sigma_i + gamma_i*sigma_i.

        TODO Args:
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

    def __len__(self):
        return self.config.num_train_timesteps
