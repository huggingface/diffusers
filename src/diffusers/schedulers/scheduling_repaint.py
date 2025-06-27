# Copyright 2025 ETH Zurich Computer Vision Lab and The HuggingFace Team. All rights reserved.
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
from ..utils.torch_utils import randn_tensor
from .scheduling_utils import SchedulerMixin


@dataclass
class RePaintSchedulerOutput(BaseOutput):
    """
    Output class for the scheduler's step function output.

    Args:
        prev_sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample (x_{t-1}) of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
        pred_original_sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width)` for images):
            The predicted denoised sample (x_{0}) based on the model output from
             the current timestep. `pred_original_sample` can be used to preview progress or for guidance.
    """

    prev_sample: torch.Tensor
    pred_original_sample: torch.Tensor


# Copied from diffusers.schedulers.scheduling_ddpm.betas_for_alpha_bar
def betas_for_alpha_bar(
    num_diffusion_timesteps,
    max_beta=0.999,
    alpha_transform_type="cosine",
):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function, which defines the cumulative product of
    (1-beta) over time from t = [0,1].

    Contains a function alpha_bar that takes an argument t and transforms it to the cumulative product of (1-beta) up
    to that part of the diffusion process.


    Args:
        num_diffusion_timesteps (`int`): the number of betas to produce.
        max_beta (`float`): the maximum beta to use; use values lower than 1 to
                     prevent singularities.
        alpha_transform_type (`str`, *optional*, default to `cosine`): the type of noise schedule for alpha_bar.
                     Choose from `cosine` or `exp`

    Returns:
        betas (`np.ndarray`): the betas used by the scheduler to step the model outputs
    """
    if alpha_transform_type == "cosine":

        def alpha_bar_fn(t):
            return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2

    elif alpha_transform_type == "exp":

        def alpha_bar_fn(t):
            return math.exp(t * -12.0)

    else:
        raise ValueError(f"Unsupported alpha_transform_type: {alpha_transform_type}")

    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar_fn(t2) / alpha_bar_fn(t1), max_beta))
    return torch.tensor(betas, dtype=torch.float32)


class RePaintScheduler(SchedulerMixin, ConfigMixin):
    """
    `RePaintScheduler` is a scheduler for DDPM inpainting inside a given mask.

    This model inherits from [`SchedulerMixin`] and [`ConfigMixin`]. Check the superclass documentation for the generic
    methods the library implements for all schedulers such as loading and saving.

    Args:
        num_train_timesteps (`int`, defaults to 1000):
            The number of diffusion steps to train the model.
        beta_start (`float`, defaults to 0.0001):
            The starting `beta` value of inference.
        beta_end (`float`, defaults to 0.02):
            The final `beta` value.
        beta_schedule (`str`, defaults to `"linear"`):
            The beta schedule, a mapping from a beta range to a sequence of betas for stepping the model. Choose from
            `linear`, `scaled_linear`, `squaredcos_cap_v2`, or `sigmoid`.
        eta (`float`):
            The weight of noise for added noise in diffusion step. If its value is between 0.0 and 1.0 it corresponds
            to the DDIM scheduler, and if its value is between -0.0 and 1.0 it corresponds to the DDPM scheduler.
        trained_betas (`np.ndarray`, *optional*):
            Pass an array of betas directly to the constructor to bypass `beta_start` and `beta_end`.
        clip_sample (`bool`, defaults to `True`):
            Clip the predicted sample between -1 and 1 for numerical stability.

    """

    order = 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        eta: float = 0.0,
        trained_betas: Optional[np.ndarray] = None,
        clip_sample: bool = True,
    ):
        if trained_betas is not None:
            self.betas = torch.from_numpy(trained_betas)
        elif beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        elif beta_schedule == "scaled_linear":
            # this schedule is very specific to the latent diffusion model.
            self.betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32) ** 2
        elif beta_schedule == "squaredcos_cap_v2":
            # Glide cosine schedule
            self.betas = betas_for_alpha_bar(num_train_timesteps)
        elif beta_schedule == "sigmoid":
            # GeoDiff sigmoid schedule
            betas = torch.linspace(-6, 6, num_train_timesteps)
            self.betas = torch.sigmoid(betas) * (beta_end - beta_start) + beta_start
        else:
            raise NotImplementedError(f"{beta_schedule} is not implemented for {self.__class__}")

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.one = torch.tensor(1.0)

        self.final_alpha_cumprod = torch.tensor(1.0)

        # standard deviation of the initial noise distribution
        self.init_noise_sigma = 1.0

        # setable values
        self.num_inference_steps = None
        self.timesteps = torch.from_numpy(np.arange(0, num_train_timesteps)[::-1].copy())

        self.eta = eta

    def scale_model_input(self, sample: torch.Tensor, timestep: Optional[int] = None) -> torch.Tensor:
        """
        Ensures interchangeability with schedulers that need to scale the denoising model input depending on the
        current timestep.

        Args:
            sample (`torch.Tensor`):
                The input sample.
            timestep (`int`, *optional*):
                The current timestep in the diffusion chain.

        Returns:
            `torch.Tensor`:
                A scaled input sample.
        """
        return sample

    def set_timesteps(
        self,
        num_inference_steps: int,
        jump_length: int = 10,
        jump_n_sample: int = 10,
        device: Union[str, torch.device] = None,
    ):
        """
        Sets the discrete timesteps used for the diffusion chain (to be run before inference).

        Args:
            num_inference_steps (`int`):
                The number of diffusion steps used when generating samples with a pre-trained model. If used,
                `timesteps` must be `None`.
            jump_length (`int`, defaults to 10):
                The number of steps taken forward in time before going backward in time for a single jump (“j” in
                RePaint paper). Take a look at Figure 9 and 10 in the paper.
            jump_n_sample (`int`, defaults to 10):
                The number of times to make a forward time jump for a given chosen time sample. Take a look at Figure 9
                and 10 in the paper.
            device (`str` or `torch.device`, *optional*):
                The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.

        """
        num_inference_steps = min(self.config.num_train_timesteps, num_inference_steps)
        self.num_inference_steps = num_inference_steps

        timesteps = []

        jumps = {}
        for j in range(0, num_inference_steps - jump_length, jump_length):
            jumps[j] = jump_n_sample - 1

        t = num_inference_steps
        while t >= 1:
            t = t - 1
            timesteps.append(t)

            if jumps.get(t, 0) > 0:
                jumps[t] = jumps[t] - 1
                for _ in range(jump_length):
                    t = t + 1
                    timesteps.append(t)

        timesteps = np.array(timesteps) * (self.config.num_train_timesteps // self.num_inference_steps)
        self.timesteps = torch.from_numpy(timesteps).to(device)

    def _get_variance(self, t):
        prev_timestep = t - self.config.num_train_timesteps // self.num_inference_steps

        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        # For t > 0, compute predicted variance βt (see formula (6) and (7) from
        # https://huggingface.co/papers/2006.11239) and sample from it to get
        # previous sample x_{t-1} ~ N(pred_prev_sample, variance) == add
        # variance to pred_sample
        # Is equivalent to formula (16) in https://huggingface.co/papers/2010.02502
        # without eta.
        # variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * self.betas[t]
        variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)

        return variance

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        original_image: torch.Tensor,
        mask: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ) -> Union[RePaintSchedulerOutput, Tuple]:
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.Tensor`):
                The direct output from learned diffusion model.
            timestep (`int`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.Tensor`):
                A current instance of a sample created by the diffusion process.
            original_image (`torch.Tensor`):
                The original image to inpaint on.
            mask (`torch.Tensor`):
                The mask where a value of 0.0 indicates which part of the original image to inpaint.
            generator (`torch.Generator`, *optional*):
                A random number generator.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~schedulers.scheduling_repaint.RePaintSchedulerOutput`] or `tuple`.

        Returns:
            [`~schedulers.scheduling_repaint.RePaintSchedulerOutput`] or `tuple`:
                If return_dict is `True`, [`~schedulers.scheduling_repaint.RePaintSchedulerOutput`] is returned,
                otherwise a tuple is returned where the first element is the sample tensor.

        """
        t = timestep
        prev_timestep = timestep - self.config.num_train_timesteps // self.num_inference_steps

        # 1. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t

        # 2. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (15) from https://huggingface.co/papers/2006.11239
        pred_original_sample = (sample - beta_prod_t**0.5 * model_output) / alpha_prod_t**0.5

        # 3. Clip "predicted x_0"
        if self.config.clip_sample:
            pred_original_sample = torch.clamp(pred_original_sample, -1, 1)

        # We choose to follow RePaint Algorithm 1 to get x_{t-1}, however we
        # substitute formula (7) in the algorithm coming from DDPM paper
        # (formula (4) Algorithm 2 - Sampling) with formula (12) from DDIM paper.
        # DDIM schedule gives the same results as DDPM with eta = 1.0
        # Noise is being reused in 7. and 8., but no impact on quality has
        # been observed.

        # 5. Add noise
        device = model_output.device
        noise = randn_tensor(model_output.shape, generator=generator, device=device, dtype=model_output.dtype)
        std_dev_t = self.eta * self._get_variance(timestep) ** 0.5

        variance = 0
        if t > 0 and self.eta > 0:
            variance = std_dev_t * noise

        # 6. compute "direction pointing to x_t" of formula (12)
        # from https://huggingface.co/papers/2010.02502
        pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** 0.5 * model_output

        # 7. compute x_{t-1} of formula (12) from https://huggingface.co/papers/2010.02502
        prev_unknown_part = alpha_prod_t_prev**0.5 * pred_original_sample + pred_sample_direction + variance

        # 8. Algorithm 1 Line 5 https://huggingface.co/papers/2201.09865
        # The computation reported in Algorithm 1 Line 5 is incorrect. Line 5 refers to formula (8a) of the same paper,
        # which tells to sample from a Gaussian distribution with mean "(alpha_prod_t_prev**0.5) * original_image"
        # and variance "(1 - alpha_prod_t_prev)". This means that the standard Gaussian distribution "noise" should be
        # scaled by the square root of the variance (as it is done here), however Algorithm 1 Line 5 tells to scale by the variance.
        prev_known_part = (alpha_prod_t_prev**0.5) * original_image + ((1 - alpha_prod_t_prev) ** 0.5) * noise

        # 9. Algorithm 1 Line 8 https://huggingface.co/papers/2201.09865
        pred_prev_sample = mask * prev_known_part + (1.0 - mask) * prev_unknown_part

        if not return_dict:
            return (
                pred_prev_sample,
                pred_original_sample,
            )

        return RePaintSchedulerOutput(prev_sample=pred_prev_sample, pred_original_sample=pred_original_sample)

    def undo_step(self, sample, timestep, generator=None):
        n = self.config.num_train_timesteps // self.num_inference_steps

        for i in range(n):
            beta = self.betas[timestep + i]
            if sample.device.type == "mps":
                # randn does not work reproducibly on mps
                noise = randn_tensor(sample.shape, dtype=sample.dtype, generator=generator)
                noise = noise.to(sample.device)
            else:
                noise = randn_tensor(sample.shape, generator=generator, device=sample.device, dtype=sample.dtype)

            # 10. Algorithm 1 Line 10 https://huggingface.co/papers/2201.09865
            sample = (1 - beta) ** 0.5 * sample + beta**0.5 * noise

        return sample

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.IntTensor,
    ) -> torch.Tensor:
        raise NotImplementedError("Use `DDPMScheduler.add_noise()` to train for sampling with RePaint.")

    def __len__(self):
        return self.config.num_train_timesteps
