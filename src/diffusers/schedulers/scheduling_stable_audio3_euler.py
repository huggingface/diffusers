# Copyright 2025 HuggingFace Inc.
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
Rectified-flow Euler scheduler for Stable Audio 3 (base model).

Reference: ``sample_discrete_euler`` in
  stable_audio_3/inference/sampling.py
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch

from ..configuration_utils import ConfigMixin, register_to_config
from ..utils import BaseOutput
from .scheduling_utils import SchedulerMixin


@dataclass
class StableAudio3EulerSchedulerOutput(BaseOutput):
    """
    Output class for `StableAudio3EulerScheduler.step`.

    Args:
        prev_sample (`torch.Tensor`):
            Computed sample at the previous (less noisy) timestep.
        pred_original_sample (`torch.Tensor`, *optional*):
            The predicted de-noised sample (x₀ prediction) at the current step.
    """

    prev_sample: torch.Tensor
    pred_original_sample: Optional[torch.Tensor] = None


class StableAudio3EulerScheduler(SchedulerMixin, ConfigMixin):
    """
    Deterministic Euler sampler for Stable Audio 3's rectified-flow **base** model.

    Unlike [`PingPongScheduler`] (used by the distilled model, which resamples fresh noise each step), this scheduler
    performs a deterministic first-order flow-matching update:

        ``x_{t+1} = x_t + (σ_{i+1} − σ_i) · v(x_t, t)``

    where ``v`` is the model's velocity prediction. The base model is *not* distilled, so it requires many more steps
    than ping-pong — typically ``num_inference_steps=100``.

    The schedule is identical to [`PingPongScheduler`]: ``N+1`` breakpoints equally spaced in log-SNR space
    ``[logsnr_min, logsnr_max]``, converted to the flow-matching *t* variable via ``t = sigmoid(−λ)``. Because
    ``sigmoid`` is decreasing, the first sigma is the highest (most noisy) and the last is the lowest.

    This model inherits from [`SchedulerMixin`] and [`ConfigMixin`].

    Args:
        num_inference_steps (`int`, defaults to 100):
            Number of denoising steps. The rectified-flow base model is not distilled and needs a large step count.
        logsnr_min (`float`, defaults to −6.2):
            Minimum log-SNR value — maps to the high-noise start of the schedule.
        logsnr_max (`float`, defaults to 2.0):
            Maximum log-SNR value — maps to the low-noise end of the schedule.
    """

    order = 1

    @register_to_config
    def __init__(
        self,
        num_inference_steps: int = 100,
        logsnr_min: float = -6.2,
        logsnr_max: float = 2.0,
    ) -> None:
        self.num_inference_steps: Optional[int] = None
        self.sigmas: Optional[torch.Tensor] = None
        self.timesteps: Optional[torch.Tensor] = None

    def set_timesteps(
        self,
        num_inference_steps: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        """
        Build the logSNR-uniform noise schedule and store it.

        Sets:
          - ``self.sigmas``: shape ``(N+1,)``, decreasing from ~1 to ~0.
          - ``self.timesteps``: shape ``(N,)`` — the *t* values at which the model is called (``sigmas[:-1]``).

        Args:
            num_inference_steps (`int`, *optional*):
                Override the configured step count.
            device (`str` or `torch.device`, *optional*):
                Device for the schedule tensors.
        """
        n = num_inference_steps if num_inference_steps is not None else self.config.num_inference_steps
        self.num_inference_steps = n

        logsnr = torch.linspace(self.config.logsnr_min, self.config.logsnr_max, n + 1)
        sigmas = torch.sigmoid(-logsnr)  # (N+1,), decreasing
        sigmas[0] = 1.0  # sigma_max: start from pure noise
        sigmas[-1] = 0.0  # end fully denoised

        if device is not None:
            sigmas = sigmas.to(device)

        self.sigmas = sigmas
        self.timesteps = sigmas[:-1]  # model is called at these t values
        self._step_index = 0

    def step(
        self,
        model_output: torch.Tensor,
        timestep: Union[float, torch.Tensor],
        sample: torch.Tensor,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        return_dict: bool = True,
    ) -> Union[StableAudio3EulerSchedulerOutput, Tuple[torch.Tensor, ...]]:
        """
        Perform one deterministic Euler flow-matching step.

        Args:
            model_output (`torch.Tensor`):
                Velocity prediction ``v(x_t, t)`` from the diffusion model.
            timestep (`float` or `torch.Tensor`):
                Current *t* value. Should match one entry of ``self.timesteps``.
            sample (`torch.Tensor`):
                Current noisy sample ``x_t``.
            generator (`torch.Generator` or `list[torch.Generator]`, *optional*):
                Unused (this sampler is deterministic); accepted for interface compatibility.
            return_dict (`bool`, defaults to `True`):
                Return a `StableAudio3EulerSchedulerOutput`; if ``False`` return a tuple.

        Returns:
            `StableAudio3EulerSchedulerOutput` or `tuple`:
                - **prev_sample** — less-noisy sample ``x_{t+1}``
                - **pred_original_sample** — predicted clean sample ``x̂₀``
        """
        if self.sigmas is None:
            raise RuntimeError("Call `set_timesteps` before calling `step`.")

        step_index = self._step_index
        t_curr = self.sigmas[step_index]
        t_next = self.sigmas[step_index + 1]
        self._step_index += 1

        # Broadcast sigmas over (batch, channel, time, ...) dimensions
        t_curr_b = t_curr.to(sample.dtype).reshape(*([1] * sample.ndim))
        dt_b = (t_next - t_curr).to(sample.dtype).reshape(*([1] * sample.ndim))

        # Deterministic Euler update along the flow ODE: x_{t+1} = x_t + (σ_{i+1} − σ_i)·v
        prev_sample = sample + dt_b * model_output

        # x̂₀ = x_t − t·v (reported for callbacks / inspection)
        pred_original_sample = sample - t_curr_b * model_output

        if not return_dict:
            return (prev_sample, pred_original_sample)
        return StableAudio3EulerSchedulerOutput(prev_sample=prev_sample, pred_original_sample=pred_original_sample)

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward diffusion: ``x_t = (1 − t)·x₀ + t·ε``.

        Consistent with the rectified-flow interpolant used by Stable Audio 3.
        """
        t = timesteps.to(dtype=original_samples.dtype, device=original_samples.device)
        while t.ndim < original_samples.ndim:
            t = t.unsqueeze(-1)
        return (1.0 - t) * original_samples + t * noise

    def scale_model_input(
        self,
        sample: torch.Tensor,
        timestep: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Identity — SA3 does not pre-condition model inputs."""
        return sample

    def __len__(self) -> int:
        return self.config.num_inference_steps
