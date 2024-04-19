# Copyright 2024 UC Berkeley Team and The HuggingFace Team. All rights reserved.
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

# DISCLAIMER: This file is strongly influenced by https://github.com/ermongroup/ddim

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
from functools import partial

import numpy as np
import torch

from ..configuration_utils import ConfigMixin, register_to_config
from ..utils import BaseOutput
from ..utils.torch_utils import randn_tensor
from .scheduling_utils import SchedulerMixin


def log_snr(t: torch.FloatTensor, beta_schedule: str) -> torch.FloatTensor:
    if t.min() < 0 or t.max() > 1:
        raise ValueError("`t` must be in range [0, 1].")

    # From https://github.com/Zhengxinyang/LAS-Diffusion/blob/a7eb304a24dec2eb85a8d3899c73338e10435bba/network/model_utils.py#L345
    if beta_schedule == "linear":
        return -torch.log(torch.special.expm1(1e-4 + 10 * t ** 2))
    elif beta_schedule == "squaredcos_cap_v2":
        return -torch.log(torch.clamp((torch.cos((t + 0.008) / (1 + 0.008) * math.pi * 0.5) ** -2) - 1, min=1e-5))
    elif beta_schedule == "sigmoid":
        # From https://colab.research.google.com/github/google-research/vdm/blob/main/colab/SimpleDiffusionColab.ipynb
        gamma_min = -6  # -13.3 in VDM CIFAR10 experiments
        gamma_max = 6  # 5.0 in VDM CIFAR10 experiments
        return gamma_max + (gamma_min - gamma_max) * t

    raise NotImplementedError(f"{beta_schedule} does is not implemented for {VDMScheduler.__class__}")


@dataclass
class VDMSchedulerOutput(BaseOutput):
    """
    Output class for the scheduler's `step` function output.

    Args:
        prev_sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample `(x_{t-1})` of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
        pred_original_sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` for images):
            The predicted denoised sample `(x_{0})` based on the model output from the current timestep.
            `pred_original_sample` can be used to preview progress or for guidance.
    """

    prev_sample: torch.FloatTensor
    pred_original_sample: Optional[torch.FloatTensor] = None


class VDMScheduler(SchedulerMixin, ConfigMixin):
    @register_to_config
    def __init__(self,
                 num_train_timesteps: Optional[int] = None,
                 beta_schedule: str = "linear",
                 clip_sample: bool = True,
                 prediction_type: str = "epsilon",
                 thresholding: bool = False,
                 dynamic_thresholding_ratio: float = 0.995,
                 clip_sample_range: float = 1.0,
                 sample_max_value: float = 1.0,
                 timestep_spacing: str = "leading",
                 steps_offset: Union[int, float] = 0):
        # Hardcoded as continuous schedules in self._log_snr are fitted to these values
        self.beta_start = 1e-4
        self.beta_end = 0.02
        self.init_noise_sigma = 1.0

        # For linear beta schedule equivalent to torch.exp(-1e-4 - 10 * t ** 2)
        self.alphas_cumprod = lambda t: torch.sigmoid(self.log_snr(t))  # Equivalent to 1 - self.sigmas
        self.sigmas = lambda t: torch.sigmoid(-self.log_snr(t))  # Equivalent to 1 - self.alphas_cumprod

        self.num_inference_steps = None
        self.timesteps = None
        if num_train_timesteps:
            # TODO: Might not be exact
            self.timesteps = torch.from_numpy(self.get_timesteps(len(self)))
            alphas_cumprod = self.alphas_cumprod(torch.flip(self.timesteps, dims=(0,)))
            alphas = alphas_cumprod[1:] / alphas_cumprod[:-1]
            self.alphas = torch.cat([alphas_cumprod[:1], alphas])
            self.betas = 1 - self.alphas

    def __len__(self) -> int:
        return self.num_inference_steps or self.config.num_train_timesteps or 1000

    def log_snr(self, timesteps: torch.Tensor) -> torch.FloatTensor:
        if not timesteps.is_floating_point():
            if not self.config.num_train_timesteps:
                raise TypeError("Discrete timesteps require `self.config.num_train_timesteps` to be set.")
            timesteps = timesteps / self.config.num_train_timesteps  # Normalize to [0, 1]

        return log_snr(timesteps, beta_schedule=self.config.beta_schedule)

    def get_timesteps(self, num_steps: Optional[int] = None) -> np.ndarray:
        if num_steps is None:
            num_steps = self.config.num_train_timesteps
        if self.config.timestep_spacing in ["linspace", "leading"]:
            timesteps = np.linspace(0, 1, num_steps,
                                    endpoint=self.config.timestep_spacing == "linspace")[::-1]
        elif self.config.time_spacing == "trailing":
            timesteps = np.arange(1, 0, -1 / num_steps) - 1 / num_steps
        else:
            raise ValueError(f"`{self.config.timestep_spacing}` timestep spacing is not supported."
                             "Choose one of 'linspace', 'leading' or 'trailing'.")
        return timesteps.astype(np.float32).copy()

    def set_timesteps(self, num_inference_steps: int, device: Optional[Union[str, torch.device]] = None):
        if not self.config.num_train_timesteps:
            timesteps = self.get_timesteps(num_inference_steps)
        else:
            if self.config.timestep_spacing in ["linspace", "leading"]:
                start = 0
                stop = self.config.num_train_timesteps
                timesteps = np.linspace(start,
                                        stop - 1 if self.config.timestep_spacing == "linspace" else stop,
                                        num_inference_steps,
                                        endpoint=self.config.timestep_spacing == "linspace")[::-1]
            elif self.config.timestep_spacing == "trailing":
                timesteps = np.arange(self.config.num_train_timesteps,
                                      0,
                                      -self.config.num_train_timesteps / num_inference_steps) - 1
            else:
                raise ValueError(f"`{self.config.timestep_spacing}` timestep spacing is not supported."
                                 "Choose one of 'linspace', 'leading' or 'trailing'.")
            timesteps = timesteps.round().astype(np.int64).copy()

        self.num_inference_steps = num_inference_steps
        timesteps += self.config.steps_offset
        self.timesteps = torch.from_numpy(timesteps).to(device)

    # Copied from diffusers.schedulers.scheduling_ddpm.DDPMScheduler._threshold_sample
    def _threshold_sample(self, sample: torch.FloatTensor) -> torch.FloatTensor:
        """
        "Dynamic thresholding: At each sampling step we set s to a certain percentile absolute pixel value in xt0 (the
        prediction of x_0 at timestep t), and if s > 1, then we threshold xt0 to the range [-s, s] and then divide by
        s. Dynamic thresholding pushes saturated pixels (those near -1 and 1) inwards, thereby actively preventing
        pixels from saturation at each step. We find that dynamic thresholding results in significantly better
        photorealism as well as better image-text alignment, especially when using very large guidance weights."

        https://arxiv.org/abs/2205.11487
        """
        dtype = sample.dtype
        batch_size, channels, *remaining_dims = sample.shape

        if dtype not in (torch.float32, torch.float64):
            sample = sample.float()  # upcast for quantile calculation, and clamp not implemented for cpu half

        # Flatten sample for doing quantile calculation along each image
        sample = sample.reshape(batch_size, channels * np.prod(remaining_dims))

        abs_sample = sample.abs()  # "a certain percentile absolute pixel value"

        s = torch.quantile(abs_sample, self.config.dynamic_thresholding_ratio, dim=1)
        s = torch.clamp(
            s, min=1, max=self.config.sample_max_value
        )  # When clamped to min=1, equivalent to standard clipping to [-1, 1]
        s = s.unsqueeze(1)  # (batch_size, 1) because clamp will broadcast along dim=0
        sample = torch.clamp(sample, -s, s) / s  # "we threshold xt0 to the range [-s, s] and then divide by s"

        sample = sample.reshape(batch_size, channels, *remaining_dims)
        sample = sample.to(dtype)

        return sample

    # Copied from diffusers.schedulers.scheduling_ddpm.DDPMScheduler.scale_model_input
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

    def add_noise(self,
                  original_samples: torch.Tensor,
                  noise: torch.Tensor,
                  timesteps: torch.Tensor) -> torch.FloatTensor:

        log_snr = self.log_snr(timesteps)
        log_snr = log_snr.view(timesteps.size(0), *((1,) * (original_samples.ndim - 1)))

        sqrt_alpha_prod = torch.sqrt(torch.sigmoid(log_snr))
        sqrt_one_minus_alpha_prod = torch.sqrt(torch.sigmoid(-log_snr))  # sqrt(sigma)

        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples

    def step(self,
             model_output: torch.Tensor,
             timestep: Union[int, float, torch.Tensor],
             sample: torch.Tensor,
             generator: Optional[torch.Generator] = None,
             return_dict: bool = True) -> Union[VDMSchedulerOutput, Tuple]:
        # From https://github.com/addtt/variational-diffusion-models/blob/7f81074dfdfc897178ad3d471458ea03e16197e8/vdm.py#L29

        if isinstance(timestep, (int, float)):
            timestep = torch.tensor(timestep,
                                    dtype=torch.float32 if isinstance(timestep, float) else torch.int64,
                                    device=sample.device)

        if not timestep.is_floating_point():
            if not self.config.num_train_timesteps:
                raise TypeError("Discrete timesteps require `self.config.num_train_timesteps` to be set.")
            timestep = timestep / self.config.num_train_timesteps  # Normalize to [0, 1]
        prev_timestep = (timestep - 1 / len(self)).clamp(0, 1)

        # 1. Compute current and previous alpha and sigma values
        log_snr = self.log_snr(timestep)
        prev_log_snr = self.log_snr(prev_timestep)

        # Allow for batched inputs
        if timestep.ndim > 0:
            log_snr = log_snr.view(timestep.size(0), *((1,) * (sample.ndim - 1)))
            prev_log_snr = prev_log_snr.view(timestep.size(0), *((1,) * (sample.ndim - 1)))

        alpha, sigma = torch.sigmoid(log_snr), torch.sigmoid(-log_snr)
        prev_alpha, prev_sigma = torch.sigmoid(prev_log_snr), torch.sigmoid(-prev_log_snr)

        # 2. Compute predicted original sample x_0
        if self.config.prediction_type == "epsilon":
            pred_original_sample = (sample - torch.sqrt(sigma) * model_output) / torch.sqrt(alpha)  # Sec. 3.4, eq. 10
        elif self.config.prediction_type == "sample":
            pred_original_sample = model_output
        else:
            raise ValueError("`prediction_type` must be either `epsilon` or `sample`.")

        # 3. Clip or threshold "predicted x_0"
        if self.config.thresholding:
            pred_original_sample = self._threshold_sample(pred_original_sample)
        elif self.config.clip_sample:
            pred_original_sample = pred_original_sample.clamp(-self.config.clip_sample_range,
                                                              self.config.clip_sample_range)

        # 4. Computed predicted previous sample x_{t-1}
        c = -torch.expm1(log_snr - prev_log_snr)
        if self.config.thresholding or self.config.clip_sample:
            pred_prev_sample = torch.sqrt(prev_alpha) * (sample * (1 - c) / torch.sqrt(alpha) + c * pred_original_sample)
        else:
            pred_prev_sample = torch.sqrt(prev_alpha / alpha) * (sample - c * torch.sqrt(sigma) * model_output)

        # 5. (Maybe) add noise
        noise_scale = torch.sqrt(prev_sigma * c)  # Becomes 0 for prev_timestep = 0
        if torch.any(noise_scale > 0):
            noise = randn_tensor(model_output.shape,
                                 generator=generator,
                                 device=model_output.device,
                                 dtype=model_output.dtype)
            pred_prev_sample += noise_scale * noise

        if not return_dict:
            return (pred_prev_sample,)

        return VDMSchedulerOutput(prev_sample=pred_prev_sample, pred_original_sample=pred_original_sample)
