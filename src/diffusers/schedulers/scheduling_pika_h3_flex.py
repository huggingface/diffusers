# Copyright 2026 Mellis, Inc. (Pika), The MiniMax authors and The HuggingFace Team. All rights reserved.
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

"""Few-step re-noising sampler for Pika H3 Flex.

Pika H3 Flex post-trains MiniMax-H3 into a six-evaluation generator. Its sampler is not the Euler update of
[`MiniMaxH3Scheduler`] but a *consistency re-noising ladder*: at every knot of a fixed schedule the transformer
predicts the clean sample, and the next state is that prediction re-noised with **fresh** Gaussian noise to the next
knot. The final evaluation returns the clean prediction. Generic Euler sampling over the same knots is not equivalent,
and neither is a different knot count, so the ladder is part of the released adapter's contract and lives in the
config.

Three conventions carry over from MiniMax-H3:

1. **The velocity is data-ward**: `x0 = x_t + sigma * v`.
2. **Timesteps are `t = 1 - sigma` in `[0, 1]`**, with `t = 1` meaning *clean*, which is what the transformer's AdaLN
   consumes directly.
3. **The exponential shift** `sigma' = s*sigma / (1 + (s-1)*sigma)` is applied per modality: the released adapters use
   `shift = 12.0` for video latents and `shift = 3.0` for audio latents. The modality is not a property of the
   scheduler: a pipeline holds two instances, `scheduler` and `audio_scheduler`, and steps both inside a single
   transformer call.

The knots are the base-grid values `[1000, 750, 500, 250, 100, 40, 0]`, i.e. `K + 1` entries ending in the terminal `0`
that is never evaluated, pushed through the modality's shift. Every state of the trajectory is held in float32.
"""

from dataclasses import dataclass

import torch

from ..configuration_utils import ConfigMixin, register_to_config
from ..utils import BaseOutput
from ..utils.torch_utils import randn_tensor
from .scheduling_utils import SchedulerMixin


@dataclass
# Copied from diffusers.schedulers.scheduling_minimax_h3.MiniMaxH3SchedulerOutput with MiniMaxH3->PikaH3Flex
class PikaH3FlexSchedulerOutput(BaseOutput):
    """
    Output class for the scheduler's `step` function output.

    Args:
        prev_sample (`torch.FloatTensor`):
            Computed sample `x_{t+1}` for the next step of the denoising loop.
    """

    prev_sample: torch.FloatTensor


class PikaH3FlexScheduler(SchedulerMixin, ConfigMixin):
    r"""
    Consistency re-noising scheduler over a fixed knot ladder with an exponential sigma shift, as used by Pika H3 Flex.

    Args:
        shift (`float`, defaults to `12.0`):
            Exponential shift applied to the knots, `sigma' = s*sigma / (1 + (s-1)*sigma)`. The released adapters use
            `12.0` for video latents and `3.0` for audio latents.
        denoising_step_list (`tuple[float, ...]`, defaults to `(1000, 750, 500, 250, 100, 40, 0)`):
            The base-grid knots on a `1000` scale, `K + 1` values starting at `1000`, strictly decreasing and ending in
            the terminal `0`. Every knot but the last drives one transformer evaluation, so the released adapters run
            six.
    """

    _compatibles = []
    order = 1

    @register_to_config
    def __init__(
        self, shift: float = 12.0, denoising_step_list: tuple[float, ...] = (1000, 750, 500, 250, 100, 40, 0)
    ):
        if shift <= 0:
            raise ValueError(f"`shift` must be positive, got {shift}.")
        knots = [float(knot) for knot in denoising_step_list]
        if len(knots) < 2 or knots[0] != 1000.0 or knots[-1] != 0.0:
            raise ValueError(
                "`denoising_step_list` must hold at least two knots on the 1000 scale, starting at 1000 and ending "
                f"with the terminal 0, got {denoising_step_list}."
            )
        if any(current <= following for current, following in zip(knots[:-1], knots[1:])):
            raise ValueError(f"`denoising_step_list` must be strictly decreasing, got {denoising_step_list}.")

        self.num_inference_steps: int | None = None
        self.sigmas: torch.Tensor | None = None
        self.timesteps: torch.Tensor | None = None
        self._step_index: int | None = None
        self._begin_index: int | None = None

    @property
    def step_index(self) -> int | None:
        """Index of the step the scheduler is about to take. Increases by one after each `step`."""
        return self._step_index

    @property
    def begin_index(self) -> int | None:
        """Index of the first step, set from a pipeline through [`~PikaH3FlexScheduler.set_begin_index`]."""
        return self._begin_index

    def set_begin_index(self, begin_index: int = 0) -> None:
        """
        Sets the begin index for the scheduler.

        Args:
            begin_index (`int`, defaults to `0`):
                The begin index for the scheduler.
        """
        self._begin_index = begin_index

    def set_timesteps(self, device: str | torch.device | None = None) -> None:
        r"""
        Build the sigma / timestep ladder.

        The configured knots are divided by `1000` and pushed through the exponential shift, in float64 before the
        float32 cast, as the released sampler builds them. The terminal `0` is part of the ladder — the shift maps `0`
        to exactly `0` — so the ladder holds `K + 1` sigmas and drives exactly `K` transformer evaluations, exposed as
        `self.timesteps = 1 - sigmas[:-1]`. There is no `num_inference_steps` argument: the evaluation count is fixed
        by the knots the adapter was trained for.

        Args:
            device (`str` or `torch.device`, *optional*):
                Device the schedule tensors are moved to. The ladder itself is always built on CPU so the schedule does
                not depend on the accelerator.
        """
        knots = torch.tensor(self.config.denoising_step_list, dtype=torch.float64) / 1000.0
        shift = float(self.config.shift)
        sigmas = (shift * knots / (1.0 + (shift - 1.0) * knots)).to(torch.float32)

        self.sigmas = sigmas.to(device=device)
        # t = 1 - sigma, and t = 1 is clean. The terminal knot has no model evaluation.
        self.timesteps = (1.0 - sigmas[:-1]).to(device=device)
        self.num_inference_steps = int(self.timesteps.numel())
        self._step_index = None
        self._begin_index = None

    # Copied from diffusers.schedulers.scheduling_minimax_h3.MiniMaxH3Scheduler.index_for_timestep
    def index_for_timestep(self, timestep: float | torch.Tensor) -> int:
        """
        Map a timestep value to its index in the schedule.

        Args:
            timestep (`float` or `torch.Tensor`):
                A value taken from `self.timesteps`. The schedule is strictly increasing in `t`, so the match is
                unique.

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

    # Copied from diffusers.schedulers.scheduling_minimax_h3.MiniMaxH3Scheduler.scale_noise
    def scale_noise(
        self,
        sample: torch.FloatTensor,
        timestep: float | torch.FloatTensor,
        noise: torch.FloatTensor,
    ) -> torch.FloatTensor:
        r"""
        Rectified-flow forward process, in MiniMax-H3's `t` convention: `x_t = t*x_0 + (1 - t)*noise`.

        MiniMax-H3 uses this to noise its conditioning anchors, where `t` is the `noise_aug` level rather than a
        schedule entry, so `timestep` is taken at face value and is *not* looked up in `self.timesteps`.

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
        generator: torch.Generator | None = None,
        return_dict: bool = True,
    ) -> PikaH3FlexSchedulerOutput | tuple:
        r"""
        Take one re-noising step down the ladder.

        The model output is a data-ward velocity, so the clean prediction is `x0 = x_t + sigma * v`. Unless this is the
        last evaluation, the next state re-noises that prediction to the next knot with fresh Gaussian noise drawn from
        `generator`: `x_next = (1 - sigma_next) * x0 + sigma_next * noise`. The last evaluation returns `x0` itself.
        Everything is evaluated in float32 and returned in the sample's dtype.

        Args:
            model_output (`torch.FloatTensor`):
                The transformer's velocity prediction at `timestep`.
            timestep (`float` or `torch.FloatTensor`):
                The current timestep, one of `self.timesteps` (so `timestep == 1 - sigma`).
            sample (`torch.FloatTensor`):
                The current sample `x_t`.
            generator (`torch.Generator`, *optional*):
                The generator the fresh noise of the next knot is drawn from.
            return_dict (`bool`, defaults to `True`):
                Whether to return a [`PikaH3FlexSchedulerOutput`] instead of a plain tuple.

        Returns:
            [`PikaH3FlexSchedulerOutput`] or `tuple`: the sample for the next step.
        """
        if isinstance(timestep, int) or (isinstance(timestep, torch.Tensor) and not timestep.is_floating_point()):
            raise ValueError(
                "Passing integer indices (e.g. from `enumerate(timesteps)`) as timesteps to"
                " `PikaH3FlexScheduler.step()` is not supported. Make sure to pass one of the"
                " `scheduler.timesteps` values."
            )

        if self._step_index is None:
            self._step_index = self.index_for_timestep(timestep) if self._begin_index is None else self._begin_index

        sigma = self.sigmas[self._step_index].to(device=sample.device, dtype=torch.float32)
        denoised = sample.to(torch.float32) + sigma * model_output.to(torch.float32)

        if self._step_index + 1 < self.num_inference_steps:
            # Re-noise the clean prediction to the next knot. The noise is fresh at every knot: this is what makes
            # the trajectory a consistency ladder rather than an Euler chain.
            sigma_next = self.sigmas[self._step_index + 1].to(device=sample.device, dtype=torch.float32)
            noise = randn_tensor(sample.shape, generator=generator, device=sample.device, dtype=torch.float32)
            prev_sample = (1.0 - sigma_next) * denoised + sigma_next * noise
        else:
            prev_sample = denoised
        prev_sample = prev_sample.to(dtype=sample.dtype)

        self._step_index += 1

        if not return_dict:
            return (prev_sample,)
        return PikaH3FlexSchedulerOutput(prev_sample=prev_sample)
