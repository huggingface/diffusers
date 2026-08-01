# Copyright 2025 The MiniMax authors and The HuggingFace Team. All rights reserved.
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

"""Rectified-flow Euler scheduler for MiniMax-H3.

Three things make this incompatible with a [`FlowMatchEulerDiscreteScheduler`] config, which is why
it is a separate class:

1. **The velocity sign is reversed.** MiniMax-H3's transformer predicts a *data-ward* velocity, so
   `x0 = x_t + sigma * v` instead of diffusers' `x0 = x_t - sigma * v`.
2. **Timesteps are `t = 1 - sigma` in `[0, 1]`**, with `t = 1` meaning *clean*. Flow-match
   schedulers expose `timesteps = sigma * num_train_timesteps`, i.e. the opposite direction on a
   1000x scale. The transformer's AdaLN consumes the H3 convention directly.
3. **The sigma grid starts from `linspace(1, 0, num_inference_steps)`** — the terminal zero is part
   of the requested step count, and duplicates created by the shift are collapsed with
   `unique_consecutive`. `FlowMatchEulerDiscreteScheduler` instead builds
   `linspace(1, 1/num_train_timesteps, ...)` and appends a terminal sigma afterwards, so
   `len(sigmas)` and every interior value differ.

Everything else is ordinary rectified flow: the exponential shift `sigma' = s*sigma / (1 + (s-1)*sigma)`,
and an Euler update written as the `x_t` / `x0` blend `x_next = r*x_t + (1 - r)*x0` with
`r = sigma_next / sigma`, evaluated in float32. Despite the reference class being named
"euler ancestral", `eta` is 0 — no noise is ever re-injected.

MiniMax-H3 runs **two schedules per request**, one per modality (`shift=12.0` for video,
`shift=3.0` for audio). The modality is not a property of the scheduler: a pipeline holds two
instances, e.g. `scheduler` and `audio_scheduler`.
"""

from dataclasses import dataclass

import torch

from ..configuration_utils import ConfigMixin, register_to_config
from ..utils import BaseOutput
from .scheduling_utils import SchedulerMixin


@dataclass
class MiniMaxH3SchedulerOutput(BaseOutput):
    """
    Output class for the scheduler's `step` function output.

    Args:
        prev_sample (`torch.FloatTensor`):
            Computed sample `x_{t+1}` for the next step of the denoising loop.
    """

    prev_sample: torch.FloatTensor


class MiniMaxH3Scheduler(SchedulerMixin, ConfigMixin):
    r"""
    Rectified-flow Euler scheduler (`eta = 0`) with an exponential sigma shift, as used by MiniMax-H3.

    Args:
        shift (`float`, defaults to `12.0`):
            Exponential shift applied to the sigma grid, `sigma' = s*sigma / (1 + (s-1)*sigma)`. The
            released checkpoints use `12.0` for video latents and `3.0` for audio latents.
    """

    _compatibles = []
    order = 1

    @register_to_config
    def __init__(self, shift: float = 12.0):
        if shift <= 0:
            raise ValueError(f"`shift` must be positive, got {shift}.")

        self.num_inference_steps: int | None = None
        self.sigmas: torch.Tensor | None = None
        self.timesteps: torch.Tensor | None = None
        self._shift = float(shift)
        self._step_index: int | None = None
        self._begin_index: int | None = None

    @property
    def shift(self) -> float:
        """The exponential shift currently applied to the sigma grid."""
        return self._shift

    @property
    def step_index(self) -> int | None:
        """Index of the step the scheduler is about to take. Increases by one after each `step`."""
        return self._step_index

    @property
    def begin_index(self) -> int | None:
        """Index of the first step, set from a pipeline through [`~MiniMaxH3Scheduler.set_begin_index`]."""
        return self._begin_index

    def set_begin_index(self, begin_index: int = 0) -> None:
        """
        Sets the begin index for the scheduler.

        Args:
            begin_index (`int`, defaults to `0`):
                The begin index for the scheduler.
        """
        self._begin_index = begin_index

    def set_shift(self, shift: float) -> None:
        """
        Overrides the configured sigma shift; call before [`~MiniMaxH3Scheduler.set_timesteps`].

        MiniMax-H3 exposes this per request as `flow_shift` (video) / `audio_flow_shift` (audio).

        Args:
            shift (`float`):
                The exponential shift to use for the next schedule.
        """
        if shift <= 0:
            raise ValueError(f"`shift` must be positive, got {shift}.")
        self._shift = float(shift)

    def set_timesteps(
        self,
        num_inference_steps: int | None = None,
        device: str | torch.device | None = None,
        sigmas: list[float] | torch.Tensor | None = None,
    ) -> None:
        r"""
        Build the sigma / timestep schedule.

        The grid is `linspace(1, 0, num_inference_steps)` pushed through the exponential shift, with
        consecutive duplicates collapsed. The terminal `0` is already part of that grid — the shift maps
        `0` to exactly `0` — so the schedule holds `num_inference_steps` sigmas and drives
        `num_inference_steps - 1` model evaluations, exposed as `self.timesteps = 1 - sigmas[:-1]`.

        Args:
            num_inference_steps (`int`, *optional*):
                Number of sigma grid points, terminal `0` included. Ignored when `sigmas` is given.
            device (`str` or `torch.device`, *optional*):
                Device the schedule tensors are moved to. The grid itself is always built on CPU in
                float32 so the schedule does not depend on the accelerator.
            sigmas (`list[float]` or `torch.Tensor`, *optional*):
                A fully-formed sigma schedule, used verbatim (no shifting, no deduplication). It must be
                strictly decreasing and terminate at `0.0`.
        """
        if sigmas is None:
            if num_inference_steps is None or num_inference_steps < 2:
                raise ValueError(
                    "`set_timesteps` requires either an explicit `sigmas` schedule or "
                    f"`num_inference_steps` >= 2, got {num_inference_steps}."
                )

            # The rectified-flow sigma range is fixed at [1.0, 0.0].
            base = torch.linspace(1.0, 0.0, int(num_inference_steps), dtype=torch.float32)
            sigmas = self._shift * base / (1 + (self._shift - 1) * base)
            # The shift compresses the grid near sigma = 1; collapse any float32 collisions it creates.
            sigmas = torch.unique_consecutive(sigmas)
        else:
            sigmas = torch.as_tensor(sigmas, dtype=torch.float32).flatten().cpu()
            if sigmas.numel() < 2 or not bool((sigmas[1:] < sigmas[:-1]).all()) or sigmas[-1].item() != 0.0:
                raise ValueError("`sigmas` must hold at least two strictly decreasing values ending at 0.0.")

        self.sigmas = sigmas.to(device=device)
        # t = 1 - sigma, and t = 1 is clean. The terminal sigma has no model evaluation.
        self.timesteps = (1.0 - sigmas[:-1]).to(device=device)
        self.num_inference_steps = int(self.timesteps.numel())
        self._step_index = None
        self._begin_index = None

    def index_for_timestep(self, timestep: float | torch.Tensor) -> int:
        """
        Map a timestep value to its index in the schedule.

        Args:
            timestep (`float` or `torch.Tensor`):
                A value taken from `self.timesteps`. The schedule is strictly increasing in `t`, so the
                match is unique.

        Returns:
            `int`: The index of `timestep`.
        """
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.to(self.timesteps.device)
        indices = (self.timesteps == timestep).nonzero()
        if len(indices) == 0:
            raise ValueError(
                "Passed `timestep` is not in `self.timesteps`. Make sure to use values from `scheduler.timesteps`."
            )
        return indices[0].item()

    def scale_noise(
        self,
        sample: torch.FloatTensor,
        timestep: float | torch.FloatTensor,
        noise: torch.FloatTensor,
    ) -> torch.FloatTensor:
        r"""
        Rectified-flow forward process, in MiniMax-H3's `t` convention: `x_t = t*x_0 + (1 - t)*noise`.

        MiniMax-H3 uses this to noise its conditioning anchors, where `t` is the `noise_aug` level
        rather than a schedule entry, so `timestep` is taken at face value and is *not* looked up in
        `self.timesteps`.

        Args:
            sample (`torch.FloatTensor`):
                The clean sample `x_0`.
            timestep (`float` or `torch.FloatTensor`):
                The target time in `[0, 1]`; `1` returns `sample` unchanged.
            noise (`torch.FloatTensor`):
                The noise to mix in.

        Returns:
            `torch.FloatTensor`: The noised sample.
        """
        if not isinstance(timestep, torch.Tensor):
            timestep = torch.tensor(timestep, dtype=sample.dtype, device=sample.device)
        timestep = timestep.to(device=sample.device, dtype=sample.dtype)
        while timestep.ndim < sample.ndim:
            timestep = timestep.unsqueeze(-1)
        return timestep * sample + (1.0 - timestep) * noise

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: float | torch.FloatTensor,
        sample: torch.FloatTensor,
        return_dict: bool = True,
    ) -> MiniMaxH3SchedulerOutput | tuple:
        r"""
        Take one Euler (`eta = 0`) step.

        The model output is a data-ward velocity, so the denoised estimate is
        `x0 = x_t + (1 - t) * v` — note the `+`, the opposite of the usual flow-match convention.
        The update is then the blend `x_next = r*x_t + (1 - r)*x0` with `r = sigma_next / sigma`,
        evaluated in float32 for half-precision samples.

        Args:
            model_output (`torch.FloatTensor`):
                The transformer's velocity prediction at `timestep`.
            timestep (`float` or `torch.FloatTensor`):
                The current timestep, one of `self.timesteps` (so `timestep == 1 - sigma`).
            sample (`torch.FloatTensor`):
                The current sample `x_t`.
            return_dict (`bool`, defaults to `True`):
                Whether to return a [`MiniMaxH3SchedulerOutput`] instead of a plain tuple.

        Returns:
            [`MiniMaxH3SchedulerOutput`] or `tuple`: the sample for the next step.
        """
        if isinstance(timestep, int) or (isinstance(timestep, torch.Tensor) and not timestep.is_floating_point()):
            raise ValueError(
                "Passing integer indices (e.g. from `enumerate(timesteps)`) as timesteps to"
                " `MiniMaxH3Scheduler.step()` is not supported. Make sure to pass one of the"
                " `scheduler.timesteps` values."
            )

        if self._step_index is None:
            self._step_index = self.index_for_timestep(timestep) if self._begin_index is None else self._begin_index

        # x0 from the data-ward velocity. The sigma used here is recovered from the *timestep* the
        # transformer was conditioned on, whereas the Euler ratio below uses the sigma grid: for
        # sigma < 0.5 the float32 round trip `1 - (1 - sigma)` is not exact, and the reference keeps
        # the two sources apart.
        if not isinstance(timestep, torch.Tensor):
            timestep = torch.tensor(timestep, dtype=sample.dtype)
        sigma_from_timestep = 1 - timestep.to(device=sample.device, dtype=sample.dtype)
        while sigma_from_timestep.ndim < sample.ndim:
            sigma_from_timestep = sigma_from_timestep.unsqueeze(-1)
        denoised = sample + sigma_from_timestep * model_output

        # Euler with eta = 0, written as an x_t / x0 blend and evaluated in float32.
        compute_dtype = torch.float32 if sample.dtype in (torch.float16, torch.bfloat16) else sample.dtype
        sigma = self.sigmas[self._step_index].to(device=sample.device, dtype=compute_dtype)
        sigma_next = self.sigmas[self._step_index + 1].to(device=sample.device, dtype=compute_dtype)
        ratio = sigma_next / sigma
        prev_sample = ratio * sample.to(dtype=compute_dtype) + (1.0 - ratio) * denoised.to(dtype=compute_dtype)
        prev_sample = prev_sample.to(dtype=sample.dtype)

        self._step_index += 1

        if not return_dict:
            return (prev_sample,)
        return MiniMaxH3SchedulerOutput(prev_sample=prev_sample)
