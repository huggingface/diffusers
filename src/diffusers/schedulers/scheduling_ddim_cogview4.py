# Copyright 2024 The CogView team, Tsinghua University & ZhipuAI and The HuggingFace Team.
# All rights reserved.
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

# DISCLAIMER: This code is strongly influenced by https://github.com/pesser/pytorch_diffusion
# and https://github.com/hojonathanho/diffusion

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from ..configuration_utils import ConfigMixin, register_to_config
from ..utils import BaseOutput
from ..utils.torch_utils import randn_tensor
from .scheduling_utils import KarrasDiffusionSchedulers, SchedulerMixin


# Copied from diffusers.schedulers.scheduling_flow_match_heun_discrete.FlowMatchHeunDiscreteSchedulerOutput
@dataclass
class CogView4DDIMSchedulerOutput(BaseOutput):
    """
    Output class for the scheduler's `step` function output.

    Args:
        prev_sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample `(x_{t-1})` of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
    """

    prev_sample: torch.FloatTensor


class CogView4DDIMScheduler(SchedulerMixin, ConfigMixin):
    """
    CogView4 DDIM Scheduler.

    This scheduler is a modified version of the DDIM scheduler specifically designed for use with the CogView4 model.
    It implements the denoising process using a deterministic approach based on the DDIM (Denoising Diffusion Implicit Models)
    framework.

    The scheduler maintains the core DDIM functionality while being optimized for the CogView4 architecture and its specific
    requirements for image generation tasks.

    Args:
        num_train_timesteps (int, optional): The number of diffusion steps to train the model. Defaults to 1000.
        beta_start (float, optional): The starting value of beta for the noise schedule. Defaults to 0.0001.
        beta_end (float, optional): The ending value of beta for the noise schedule. Defaults to 0.02.
        set_alpha_to_one (bool, optional): Whether to set the final alpha cumprod value to 1. Defaults to True.
    """

    _compatibles = [e.name for e in KarrasDiffusionSchedulers]
    order = 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        set_alpha_to_one: bool = True,
    ):
        self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        self.final_alpha_cumprod = torch.tensor(1.0) if set_alpha_to_one else self.alphas_cumprod[0]

        # standard deviation of the initial noise distribution
        self.init_noise_sigma = 1.0

        # setable values
        self.num_inference_steps = None

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

    @staticmethod
    def calculate_shift(
        image_seq_len,
        base_seq_len: int = 256,
    ):
        if isinstance(image_seq_len, int):
            mu = math.sqrt(image_seq_len / base_seq_len)
        elif isinstance(image_seq_len, torch.Tensor):
            mu = torch.sqrt(image_seq_len / base_seq_len)
        else:
            raise ValueError(f"Invalid type for image_seq_len: {type(image_seq_len)}")

        mu = mu * 0.75 + 0.25

        return mu

    @staticmethod
    def time_shift(mu: float, shift_sigma: float, sigmas: torch.Tensor):
        return mu / (mu + (1 / sigmas - 1) ** shift_sigma)

    def set_timesteps(self, num_inference_steps: int, image_seq_len: int, device: Union[str, torch.device] = None):
        """
        Sets the discrete timesteps used for the diffusion chain. Supporting to be called in every batch.

        Args:
            num_inference_steps (`int`):
                The number of diffusion steps used when generating samples with a pre-trained model.
            image_seq_len (`int`):
                The length of the image sequence.
            device (`str` or `torch.device`, *optional*):
                The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        """

        # Check if the requested number of steps is valid
        if num_inference_steps > self.config.num_train_timesteps:
            raise ValueError(
                f"`num_inference_steps`: {num_inference_steps} cannot be larger than `self.config.num_train_timesteps`:"
                f" {self.config.num_train_timesteps} as the unet model trained with this scheduler can only handle"
                f" maximal {self.config.num_train_timesteps} timesteps."
            )

        # Set the current number of inference steps
        self.num_inference_steps = num_inference_steps

        timesteps = np.linspace(self.config.num_train_timesteps, 1, num_inference_steps).astype(np.int64)
        self.timestep2idx = {timestep: i for i, timestep in enumerate(timesteps)}

        # Convert the numpy array of timesteps into a PyTorch tensor
        timesteps = torch.from_numpy(timesteps).to(device)

        mu = self.calculate_shift(image_seq_len)
        sigmas = timesteps / self.config.num_train_timesteps
        sigmas = torch.cat([sigmas, torch.zeros(1, device=sigmas.device)])  # Append zero at the end

        self.timesteps = timesteps
        self.sigmas = self.time_shift(mu, 1.0, sigmas).to("cpu")
        self._num_timesteps = len(timesteps)

    def step(
        self,
        model_output: torch.Tensor,
        sample: torch.Tensor,
        timestep: int,
        return_dict: bool = True,
    ) -> Union[CogView4DDIMSchedulerOutput, Tuple]:
        """
        Predict the sample from the previous timestep by applying the flow matching update.

        This method implements the flow matching step for the CogView4 DDIM scheduler. It takes the model output
        (predicted noise) and current sample, and computes the previous sample by following the flow matching
        update rule.

        Args:
            model_output (`torch.Tensor`):
                The output from the diffusion model, typically the predicted noise or velocity.
            sample (`torch.Tensor`):
                The current sample at the current timestep.
            timestep (`int`):
                The current timestep in the diffusion process.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a `CogView4DDIMSchedulerOutput` or a tuple.

        Returns:
            `CogView4DDIMSchedulerOutput` or `tuple`:
                If `return_dict` is True, returns a `CogView4DDIMSchedulerOutput` containing the predicted
                sample at the previous timestep. Otherwise, returns a tuple with the predicted sample.
        """

        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )
        idx = self.timestep2idx[timestep.item()]
        sigma = self.sigmas[idx]
        sigma_next = self.sigmas[idx + 1]
        dt = sigma_next - sigma

        prev_sample = sample + dt * model_output

        if not return_dict:
            return (prev_sample,)

        return CogView4DDIMSchedulerOutput(prev_sample=prev_sample)

    # Copied from diffusers.schedulers.scheduling_ddpm.DDPMScheduler.add_noise
    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.IntTensor,
    ) -> torch.Tensor:
        # Make sure alphas_cumprod and timestep have same device and dtype as original_samples
        # Move the self.alphas_cumprod to device to avoid redundant CPU to GPU data movement
        # for the subsequent add_noise calls
        self.alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device)
        alphas_cumprod = self.alphas_cumprod.to(dtype=original_samples.dtype)
        timesteps = timesteps.to(original_samples.device)

        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples

    # Copied from diffusers.schedulers.scheduling_ddpm.DDPMScheduler.get_velocity
    def get_velocity(self, sample: torch.Tensor, noise: torch.Tensor, timesteps: torch.IntTensor) -> torch.Tensor:
        # Make sure alphas_cumprod and timestep have same device and dtype as sample
        self.alphas_cumprod = self.alphas_cumprod.to(device=sample.device)
        alphas_cumprod = self.alphas_cumprod.to(dtype=sample.dtype)
        timesteps = timesteps.to(sample.device)

        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(sample.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(sample.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        velocity = sqrt_alpha_prod * noise - sqrt_one_minus_alpha_prod * sample
        return velocity

    def __len__(self):
        return self.config.num_train_timesteps
