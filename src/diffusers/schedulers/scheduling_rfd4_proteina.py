# Copyright 2026 The HuggingFace Team. All rights reserved.
# Diffusers integration additions are licensed under the Apache License, Version 2.0.
#
# RFD4 sampling is adapted from RFD4-Proteina (BSD-3-Clause).
# Copyright (c) 2026, Institute for Protein Design, University of Washington
# Full third-party copyright notices, license conditions and disclaimers for the RFD4-Proteina
# integration are in diffusers/models/transformers/transformer_rfd4_proteina.py.
from dataclasses import dataclass

import torch

from ..configuration_utils import ConfigMixin, register_to_config
from ..utils import BaseOutput
from ..utils.torch_utils import randn_tensor
from .scheduling_utils import SchedulerMixin


@dataclass
class RFD4ProteinaSchedulerOutput(BaseOutput):
    """Updated state of one modality.

    Args:
        prev_sample (`torch.Tensor`):
            The state after the step, shaped `(B, N, D)`: `D` is 3 for the backbone (coordinates in nanometers) and the
            latent width for the local latents.
    """

    prev_sample: torch.Tensor


def _schedule(
    mode: str, steps: int, *, start: float = 0.0, tail_steps: int = 0, tail_multiplier: int = 1
) -> torch.Tensor:
    """Times from `start` to 1 with `steps` intervals; the last `tail_steps` intervals are `tail_multiplier` times finer."""
    if tail_steps:
        if type(tail_multiplier) is not int or tail_multiplier < 2:
            raise ValueError("tail_step_multiplier must be an integer >= 2.")
        base_steps = steps - (tail_multiplier - 1) * tail_steps
        if base_steps < tail_steps:
            raise ValueError("steps must be at least tail_steps * tail_step_multiplier.")
        fine = _schedule(mode, base_steps * tail_multiplier, start=start)
        tail_start = (base_steps - tail_steps) * tail_multiplier
        indices = [*range(0, tail_start, tail_multiplier), *range(tail_start, len(fine))]
        return fine[torch.tensor(indices)]
    if mode == "power":
        return torch.linspace(start**0.5, 1, steps + 1) ** 2
    times = 1 - torch.logspace(-2, start, steps + 1).flip(0)
    times -= times.min() - start
    return times / times.max()


def _noise_schedule(times: torch.Tensor, mode: str) -> torch.Tensor:
    times = times.clamp(0, 1 - 1e-5)
    if mode == "tan":
        angle = (1 - times) * torch.pi / 2
        noise = (torch.pi / 2) * torch.sin(angle) / (torch.cos(angle) + 1e-2)
    else:
        noise = 1 / (times + 1e-2)
    return noise.clamp(min=0)


def _score(sample: torch.Tensor, velocity: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
    return (times[:, None, None] * velocity - sample) / (1 - times)[:, None, None]


def _velocity(sample: torch.Tensor, score: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
    times = times[:, None, None]
    return (sample + (1 - times) * score) / times


def _tsr_ratio(times: torch.Tensor, k: float, sigma: float) -> torch.Tensor:
    snr = times.square() / (1 - times).square()
    return (snr * sigma**2 + 1) / (snr * sigma**2 / k + 1)


class RFD4ProteinaScheduler(SchedulerMixin, ConfigMixin):
    """Temporal-score-rescaled flow sampler for one modality of RFD4-Proteína.

    The pipeline integrates two modalities, the backbone and the per-token local latents, with one scheduler each. They
    differ in their time schedule, noise schedule and rescaling.

    Args:
        schedule_mode (`str`):
            Spacing of the times between noise and data: `"log"` or `"power"`.
        noise_mode (`str`):
            Shape of the noise injected in the stochastic window: `"1/t"` or `"tan"`.
        tsr_k (`float`):
            How low the sampling temperature goes; higher is lower.
        tsr_sigma (`float`):
            How fast the low temperature is reached; higher is faster.
        tail_steps (`int`, defaults to `10`):
            Number of final intervals that are finer than the rest by `tail_step_multiplier`.
        tail_step_multiplier (`int`, defaults to `2`):
            How much finer the final intervals are.
        t_lim_ode (`float`, defaults to `0.95`):
            Above this time the step is deterministic.
        t_lim_ode_below (`float`, defaults to `0.02`):
            Below this time the step is deterministic.

    Call `set_timesteps` before sampling. Times increase from noise toward clean data. A scheduler tracks the current
    interval and should not be shared by concurrent generations.
    """

    order = 1
    _compatibles = []

    @register_to_config
    def __init__(
        self,
        schedule_mode: str,
        noise_mode: str,
        tsr_k: float,
        tsr_sigma: float,
        tail_steps: int = 10,
        tail_step_multiplier: int = 2,
        t_lim_ode: float = 0.95,
        t_lim_ode_below: float = 0.02,
    ):
        if schedule_mode not in {"log", "power"}:
            raise ValueError(f"schedule_mode must be 'log' or 'power', got {schedule_mode!r}.")
        if noise_mode not in {"1/t", "tan"}:
            raise ValueError(f"noise_mode must be '1/t' or 'tan', got {noise_mode!r}.")
        self.timesteps = None
        self._step_index = None
        self._times = None
        self._noise = None

    @property
    def step_index(self) -> int | None:
        """Index of the next integration interval."""
        return self._step_index

    def set_timesteps(
        self,
        num_inference_steps: int,
        device: str | torch.device | None = None,
        start_time: float | None = None,
    ) -> None:
        """Resolve the schedule and reset the integration cursor.

        Args:
            num_inference_steps (`int`):
                Number of integration intervals.
            device (`str` or `torch.device`, *optional*):
                Device for `timesteps`. The interval lengths and noise coefficients stay on the CPU.
            start_time (`float`, *optional*):
                Partial-diffusion start time in `[0, 1)`. Omit for full generation.
        """
        if type(num_inference_steps) is not int or num_inference_steps < 1:
            raise ValueError("num_inference_steps must be a positive integer.")
        if start_time is not None and not 0 <= float(start_time) < 1:
            raise ValueError("start_time must be in [0, 1).")

        def schedule_for(steps: int) -> torch.Tensor:
            return _schedule(
                self.config.schedule_mode,
                steps,
                start=1e-3,
                tail_steps=self.config.tail_steps,
                tail_multiplier=self.config.tail_step_multiplier,
            )

        if start_time is None:
            full_times = times = schedule_for(num_inference_steps)
        else:
            dense_steps = max(num_inference_steps * 100, 1000)
            above = schedule_for(dense_steps) >= float(start_time) - 1e-6
            first = above.nonzero(as_tuple=True)[0][0].item() if above.any() else dense_steps
            points_above = max(dense_steps - first, 1)
            full_times = schedule_for(round(num_inference_steps * dense_steps / points_above))
            times = full_times[-(num_inference_steps + 1) :]
        # The noise schedule carries the temporal score rescaling twice, once for the drift and once for the noise.
        noise = _noise_schedule(full_times[:-1], self.config.noise_mode)
        noise /= _tsr_ratio(full_times[:-1], self.config.tsr_k, self.config.tsr_sigma) ** 2
        # The interval lengths and noise coefficients stay CPU scalars; only the model times move to the device.
        self._times, self._noise = times, noise[-num_inference_steps:]
        self.timesteps = times[:-1].to(device)
        self._step_index = 0

    @staticmethod
    def _generators(
        generator: torch.Generator | list[torch.Generator] | None, batch_size: int
    ) -> tuple[torch.Generator, ...] | None:
        """Normalise per-row generators; `None` means draw from the global stream."""
        if generator is None:
            return None
        generators = (generator,) if isinstance(generator, torch.Generator) else tuple(generator)
        if len(generators) != batch_size or any(not isinstance(item, torch.Generator) for item in generators):
            raise ValueError(f"Provide one torch.Generator per batch row ({batch_size}).")
        if len({id(item) for item in generators}) != len(generators):
            raise ValueError("Each batch row requires a distinct torch.Generator.")
        return generators

    def add_noise(
        self, original_samples: torch.Tensor, noise: torch.Tensor, timesteps: float | torch.Tensor
    ) -> torch.Tensor:
        """Sample the forward marginal `(1 - t) * noise + t * clean`.

        Args:
            original_samples (`torch.Tensor`): Clean state `(batch, tokens, D)`.
            noise (`torch.Tensor`): Noise of the same shape, drawn by the caller.
            timesteps (`float` or `torch.Tensor`): Time in `[0, 1)`, a scalar or `(batch,)`.
        """
        batch_size = original_samples.shape[0]
        time = torch.as_tensor(timesteps, device=original_samples.device, dtype=original_samples.dtype)
        if time.ndim == 0:
            time = time.expand(batch_size)
        if time.shape != (batch_size,) or torch.any((time < 0) | (time >= 1)):
            raise ValueError(f"timesteps must have shape ({batch_size},) with values in [0, 1).")
        return (1 - time[:, None, None]) * noise + time[:, None, None] * original_samples

    def step(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor,
        sample: torch.Tensor,
        emit_mask: torch.Tensor,
        generator: torch.Generator | list[torch.Generator] | None = None,
        return_dict: bool = True,
    ) -> RFD4ProteinaSchedulerOutput | tuple[torch.Tensor]:
        """Advance one integration interval and update `step_index`.

        Args:
            model_output (`torch.Tensor`):
                Predicted velocity, shaped like `sample`.
            timestep (`torch.Tensor`):
                The next element of `timesteps`. Out-of-order steps are rejected.
            sample (`torch.Tensor`):
                Current state `(B, N, D)`.
            emit_mask (`torch.Tensor`):
                Boolean `(B, N)`, true for the tokens that are updated (not context or padding).
            generator (`torch.Generator` or `list[torch.Generator]`, *optional*):
                A generator for a single row or distinct generators for each batch row. Omitting them uses the global
                stream.
            return_dict (`bool`, defaults to `True`):
                Return `RFD4ProteinaSchedulerOutput`; otherwise a one-item tuple holding the state.

        Returns:
            `RFD4ProteinaSchedulerOutput` or `tuple`: The updated state.
        """
        if self._step_index is None:
            raise ValueError("Call set_timesteps before step.")
        if self._step_index >= len(self.timesteps):
            raise ValueError("The trajectory is complete; call set_timesteps to start another.")
        if not torch.equal(timestep, self.timesteps[self._step_index]):
            raise ValueError("timestep must be the next element of timesteps.")

        generators = self._generators(generator, sample.shape[0])
        index = self._step_index
        t = timestep.expand(sample.shape[0])
        dt = self._times[index + 1] - self._times[index]
        gt = self._noise[index]
        ratio = _tsr_ratio(t[0], self.config.tsr_k, self.config.tsr_sigma)
        if t[0] == 0:
            # The score is undefined at zero time; the released sampler holds the state.
            prev_sample = sample * emit_mask[..., None]
        else:
            score = _score(sample, model_output, t)
            rescaled = _velocity(sample, score * ratio, t)
            if t[0] > self.config.t_lim_ode or t[0] < self.config.t_lim_ode_below:
                change = rescaled * dt
            else:
                random = randn_tensor(
                    sample.shape,
                    generator=list(generators) if generators is not None else None,
                    device=sample.device,
                    dtype=sample.dtype,
                )
                # The drift and the noise schedule both carry the temporal score rescaling, which keeps the injected
                # noise consistent with the lower temperature.
                change = (rescaled + gt * score * ratio) * dt + torch.sqrt(2 * gt * dt) * random
            prev_sample = (sample + change) * emit_mask[..., None]
        self._step_index += 1
        if not return_dict:
            return (prev_sample,)
        return RFD4ProteinaSchedulerOutput(prev_sample=prev_sample)
