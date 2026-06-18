# Copyright 2025 Antgroup and The HuggingFace Inc. team. All rights reserved.
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

"""
Flow Matching Euler ODE Scheduler for UniLLaDA.

Implements linear interpolation path (ICPlan) with velocity prediction and Euler ODE integration.
Supports both standard ODE sampling (50 steps) and DDPM-style re-noising (8 steps, decoder-turbo).
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch

from ..configuration_utils import ConfigMixin, register_to_config
from ..utils import BaseOutput
from .scheduling_utils import SchedulerMixin


@dataclass
class LLaDA2UniFlowMatchEulerSchedulerOutput(BaseOutput):
    """
    Output class for the scheduler's `step` function.

    Args:
        prev_sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample `(x_{t-1})` of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
    """

    prev_sample: torch.Tensor


class LLaDA2UniFlowMatchEulerScheduler(SchedulerMixin, ConfigMixin):
    """
    Flow Matching scheduler using Euler ODE integration with linear interpolation path for UniLLaDA.

    This scheduler implements the flow matching framework with:
    - Linear path: x_t = t * x_1 + (1 - t) * x_0
    - Velocity prediction: v_t = x_1 - x_0
    - Euler ODE integration

    Args:
        num_train_timesteps (`int`, defaults to 1000):
            The number of diffusion steps to train the model (not used during inference).
        shift_factor (`float`, defaults to 6.0):
            Time shifting factor for improved sampling at high resolutions.
        use_dynamic_shifting (`bool`, defaults to `True`):
            Whether to apply dynamic time shifting based on sequence length.
        stochastic_ratio (`float`, defaults to 0.0):
            Ratio of stochastic (DDPM-style) vs deterministic (ODE) sampling.
            0.0 = pure ODE, 1.0 = pure DDPM re-noising (decoder-turbo mode).
    """

    order = 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift_factor: float = 6.0,
        use_dynamic_shifting: bool = True,
        stochastic_ratio: float = 0.0,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.shift_factor = shift_factor
        self.use_dynamic_shifting = use_dynamic_shifting
        self.stochastic_ratio = stochastic_ratio

        # Will be set in set_timesteps
        self.timesteps = None
        self.num_inference_steps = None
        self._step_index = None

    def time_shift(self, t: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Apply time shifting for improved high-resolution sampling."""
        if not self.use_dynamic_shifting:
            return t

        # Dynamic shifting based on sequence length
        base_shift = 0.5
        max_shift = 1.15
        mu = base_shift + (max_shift - base_shift) * (seq_len - 256) / (4096 - 256)
        mu = max(base_shift, min(max_shift, mu))

        # Shift formula (original uses t=0:clean, t=1:noise; we use reverse)
        t_shifted = 1 - t
        t_shifted = math.exp(mu) / (math.exp(mu) + (1 / (t_shifted + 1e-10) - 1) ** 1.0)
        t_shifted = 1 - t_shifted

        return t_shifted

    def set_timesteps(
        self,
        num_inference_steps: int,
        device: Union[str, torch.device] = None,
        seq_len: Optional[int] = None,
    ):
        """
        Sets the discrete timesteps used for the diffusion chain (to be run before inference).

        Args:
            num_inference_steps (`int`):
                The number of diffusion steps used when generating samples with a pre-trained model.
            device (`str` or `torch.device`, *optional*):
                The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
            seq_len (`int`, *optional*):
                Sequence length for dynamic time shifting. If None, uses default shifting.
        """
        self.num_inference_steps = num_inference_steps

        # Linear timesteps from 0 to 1
        timesteps = torch.linspace(0, 1, num_inference_steps, device=device)

        # Apply time shifting if enabled
        if self.use_dynamic_shifting and seq_len is not None:
            timesteps = torch.tensor(
                [self.time_shift(t.item(), seq_len) for t in timesteps],
                device=device,
            )
        elif self.shift_factor > 0:
            # Apply fixed shifting
            timesteps = timesteps / (timesteps + self.shift_factor - self.shift_factor * timesteps)

        self.timesteps = timesteps
        self._step_index = 0

    def step(
        self,
        model_output: torch.Tensor,
        timestep: Union[float, torch.Tensor],
        sample: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ) -> Union[LLaDA2UniFlowMatchEulerSchedulerOutput, Tuple]:
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.Tensor`):
                The direct output from learned diffusion model (velocity prediction).
            timestep (`float` or `torch.Tensor`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.Tensor`):
                A current instance of a sample created by the diffusion process.
            generator (`torch.Generator`, *optional*):
                A random number generator for stochastic sampling.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~schedulers.scheduling_utils.LLaDA2UniFlowMatchEulerSchedulerOutput`] or `tuple`.

        Returns:
            [`~schedulers.scheduling_utils.LLaDA2UniFlowMatchEulerSchedulerOutput`] or `tuple`:
                If return_dict is `True`, [`~schedulers.scheduling_utils.LLaDA2UniFlowMatchEulerSchedulerOutput`] is returned,
                otherwise a tuple is returned where the first element is the sample tensor.
        """
        if self.timesteps is None:
            raise ValueError("Timesteps must be set before calling step(). Call set_timesteps() first.")

        # Get current and next timestep
        step_idx = self._step_index
        if step_idx == len(self.timesteps) - 1:
            # Last step
            dt = 0.0
            prev_sample = sample + model_output * dt
        else:
            t_curr = self.timesteps[step_idx]
            t_next = self.timesteps[step_idx + 1]
            dt = (t_next - t_curr).item()

            # Euler step: x_{t+dt} = x_t + v_t * dt
            prev_sample = sample + model_output * dt

            # Add stochastic noise if stochastic_ratio > 0 (decoder-turbo mode)
            if self.stochastic_ratio > 0 and step_idx < len(self.timesteps) - 1:
                noise = torch.randn_like(sample, generator=generator)
                # Scale noise by stochastic ratio and timestep
                noise_scale = self.stochastic_ratio * math.sqrt(abs(dt))
                prev_sample = prev_sample + noise * noise_scale

        self._step_index += 1

        if not return_dict:
            return (prev_sample,)

        return LLaDA2UniFlowMatchEulerSchedulerOutput(prev_sample=prev_sample)

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Add noise to the original samples according to the flow matching forward process.

        Args:
            original_samples (`torch.Tensor`):
                The original samples (x_1).
            noise (`torch.Tensor`):
                The noise to add (x_0).
            timesteps (`torch.Tensor`):
                The timesteps (t) for each sample.

        Returns:
            `torch.Tensor`: The noisy samples x_t = t * x_1 + (1 - t) * x_0.
        """
        # Ensure timesteps are in [0, 1]
        timesteps = timesteps.float()
        if timesteps.dim() == 0:
            timesteps = timesteps.unsqueeze(0)

        # Reshape timesteps to broadcast correctly
        while timesteps.dim() < original_samples.dim():
            timesteps = timesteps.unsqueeze(-1)

        # Linear interpolation: x_t = t * x_1 + (1 - t) * x_0
        noisy_samples = timesteps * original_samples + (1 - timesteps) * noise

        return noisy_samples

    def __len__(self):
        return self.config.num_train_timesteps
