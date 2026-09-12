# Copyright 2025 SandAI and The HuggingFace Team. All rights reserved.
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

import torch

from ..configuration_utils import ConfigMixin, register_to_config
from ..utils import BaseOutput
from .scheduling_utils import SchedulerMixin


@dataclass
class MagiEulerSchedulerOutput(BaseOutput):
    """
    Output of a MAGI Euler update.

    Args:
        prev_sample (`torch.Tensor`): The FP32 sample at the next, cleaner timestep.
    """

    prev_sample: torch.Tensor


class MagiEulerScheduler(SchedulerMixin, ConfigMixin):
    """
    Euler integration of MAGI velocity predictions, with time increasing from noise (0) to clean data (1).

    Args:
        shift (`float`, defaults to 3.0): Shift applied after squaring time when `time_schedule="sd3"`.
        time_schedule (`str`, defaults to `"sd3"`): One of `"sd3"`, `"square"`, `"piecewise"`, or `"linear"`.
        shortcut_mode (`str`, defaults to `"8,16,16"`): The 12-step grid ordering, either `"8,16,16"` or `"16,16,8"`.
    """

    order = 1
    _compatibles = []

    @register_to_config
    def __init__(self, shift: float = 3.0, time_schedule: str = "sd3", shortcut_mode: str = "8,16,16"):
        if not math.isfinite(shift) or shift < 1:
            raise ValueError("shift must be finite and at least 1.")
        if time_schedule not in {"sd3", "square", "piecewise", "linear"}:
            raise ValueError("time_schedule must be sd3, square, piecewise, or linear.")
        if shortcut_mode not in {"8,16,16", "16,16,8"}:
            raise ValueError("shortcut_mode must be 8,16,16 or 16,16,8.")
        self.init_noise_sigma = 1.0
        self.num_inference_steps = None
        self.timesteps = None
        self.timestep_schedule = None
        self._step_index = None

    @property
    def step_index(self):
        """Index of the next sequential update; explicit endpoint updates do not change it."""
        return self._step_index

    def set_timesteps(self, num_inference_steps: int, device: str | torch.device = None):
        """
        Build the official FP32 grid on the execution device and reset sequential stepping.

        Args:
            num_inference_steps (`int`): Positive number of updates per chunk, not total chunk-window model calls.
            device (`str` or `torch.device`, optional): Device on which to compute the time grid.

        `timesteps` contains model evaluation times. `timestep_schedule` also includes the integration endpoint. The
        endpoint retains the reference's floating-point rounding instead of being clamped to exactly 1.
        """
        if (
            isinstance(num_inference_steps, bool)
            or not isinstance(num_inference_steps, int)
            or num_inference_steps <= 0
        ):
            raise ValueError("num_inference_steps must be a positive integer.")
        if num_inference_steps == 12:
            base_t = torch.linspace(0, 1, 5, device=device, dtype=torch.float32) / 4
            offsets = torch.linspace(0, 1, 5, device=device, dtype=torch.float32)
            if self.config.shortcut_mode == "16,16,8":
                base_t = base_t[:3]
            else:
                base_t = torch.cat([base_t[:1], base_t[2:4]], dim=0)
            timesteps = torch.cat([base_t + offset for offset in offsets], dim=0)[:13]
        else:
            timesteps = torch.linspace(0, 1, num_inference_steps + 1, device=device, dtype=torch.float32)
        if self.config.time_schedule == "sd3":
            timesteps = timesteps**2
            inverse_shift = 1.0 / self.config.shift
            timesteps = inverse_shift * timesteps / (1 + (inverse_shift - 1) * timesteps)
        elif self.config.time_schedule == "square":
            timesteps = timesteps**2
        elif self.config.time_schedule == "piecewise":
            mask = timesteps < 0.875
            timesteps[mask] = timesteps[mask] * (0.5 / 0.875)
            timesteps[~mask] = 0.5 + (timesteps[~mask] - 0.875) * (0.5 / (1 - 0.875))
        self.num_inference_steps = num_inference_steps
        self.timestep_schedule = timesteps
        self.timesteps = timesteps[:-1]
        self._step_index = None

    def step(
        self,
        model_output: torch.Tensor,
        timestep: float | torch.Tensor,
        sample: torch.Tensor,
        next_timestep: float | torch.Tensor | None = None,
        return_dict: bool = True,
    ) -> MagiEulerSchedulerOutput | tuple:
        """
        Advance the sample with `sample + velocity * (next_timestep - timestep)` in FP32.

        Args:
            model_output (`torch.Tensor`): Predicted velocity, after guidance, with the same shape as `sample`.
            timestep (`float` or `torch.Tensor`): Normalized time, not a loop index or a time multiplied by 1000.
            sample (`torch.Tensor`):
                Current sample. Chunk-wise updates require `(batch, channels, frames, height, width)`.
            next_timestep (`float` or `torch.Tensor`, optional):
                Explicit target time. Endpoint tensors can be scalars, `(chunks,)`, or `(batch, chunks)` and must
                broadcast together. Chunks split frames equally. Explicit updates do not advance `step_index`. If
                omitted, advance sequentially from a scalar schedule timestep.
            return_dict (`bool`, defaults to `True`): Return a structured output instead of a tuple.

        Returns:
            `MagiEulerSchedulerOutput` or `tuple`: The next FP32 sample. No prediction-to-velocity conversion is
            applied.
        """
        if self.num_inference_steps is None:
            raise ValueError("Call set_timesteps before step.")
        if model_output.shape != sample.shape:
            raise ValueError("model_output and sample must have the same shape.")
        timestep = torch.as_tensor(timestep, dtype=torch.float32, device=sample.device)
        sequential = next_timestep is None
        step_index = self._step_index
        if sequential:
            if timestep.numel() != 1:
                raise ValueError("Chunk-wise updates require next_timestep.")
            timestep = timestep.reshape(())
            if step_index is None:
                indices = (self.timesteps == timestep.to(self.timesteps.device)).nonzero().flatten()
                if indices.numel() != 1:
                    raise ValueError("timestep must be a scalar from scheduler.timesteps.")
                step_index = indices.item()
            if step_index >= self.num_inference_steps:
                raise ValueError("The schedule is complete; call set_timesteps to restart.")
            if timestep != self.timesteps[step_index].to(sample.device):
                raise ValueError("timestep does not match the next sequential step.")
            next_timestep = self.timestep_schedule[step_index + 1]
        next_timestep = torch.as_tensor(next_timestep, dtype=torch.float32, device=sample.device)
        timestep, next_timestep = torch.broadcast_tensors(timestep, next_timestep)
        if (
            not torch.isfinite(timestep).all()
            or not torch.isfinite(next_timestep).all()
            or (timestep < 0).any()
            or (next_timestep > 1 + 1e-5).any()
            or (next_timestep < timestep).any()
        ):
            raise ValueError("Timesteps must be finite, increasing from 0 to 1 (allowing endpoint rounding).")
        delta_t = next_timestep - timestep
        if delta_t.ndim == 0:
            prev_sample = sample.float() + model_output.float() * delta_t
        else:
            if delta_t.ndim == 1:
                delta_t = delta_t[None]
            if (
                sample.ndim != 5
                or delta_t.ndim != 2
                or delta_t.shape[0] not in (1, sample.shape[0])
                or delta_t.shape[1] == 0
                or sample.shape[2] % delta_t.shape[1]
            ):
                raise ValueError(
                    "Chunk times must have shape (chunks,) or (batch, chunks), dividing video frames evenly."
                )
            chunk_shape = (*sample.shape[:2], delta_t.shape[1], -1, *sample.shape[3:])
            prev_sample = (
                sample.float().reshape(chunk_shape)
                + model_output.float().reshape(chunk_shape) * delta_t[:, None, :, None, None, None]
            )
            prev_sample = prev_sample.reshape(sample.shape)
        if sequential:
            self._step_index = step_index + 1
        if not return_dict:
            return (prev_sample,)
        return MagiEulerSchedulerOutput(prev_sample=prev_sample)
