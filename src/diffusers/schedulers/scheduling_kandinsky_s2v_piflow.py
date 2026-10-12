# Copyright 2025 The Kandinsky Team and The HuggingFace Team. All rights reserved.
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
"""pi-Flow scheduler for distilled Kandinsky S2V 1.0 checkpoints.

pi-Flow (arXiv 2510.14974) replaces the per-step velocity with a *policy*: the student predicts a grid of x0 values
spanning the segment ``[t_dst, t_src]`` in a single forward, and the ODE is then integrated across the segment without
further network evaluations. One segment therefore costs exactly one model evaluation, and 16 segments replace the 50
Euler steps of the base model.

The distilled checkpoints widen the output head ``n_grid`` times; this scheduler splits that axis, so the transformer
itself is unchanged and its config only differs in ``out_visual_dim``.
"""

from __future__ import annotations

import torch
from torch import Tensor

from ..configuration_utils import register_to_config
from .scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
    FlowMatchEulerDiscreteSchedulerOutput,
)


_MIN_GRID_POINTS = 2


def shift_timesteps(t: Tensor, shift: float) -> Tensor:
    """Flow-matching timestep warp."""
    return shift * t / (1 + (shift - 1) * t)


class DXPolicy:
    """Dynamic-x0 policy: a grid of x0 predictions, linearly interpolated in raw time."""

    def __init__(  # noqa: PLR0913
        self,
        denoising_output: Tensor,
        x_t_src: Tensor,
        sigma_t_src: Tensor,
        segment_size: Tensor,
        shift: float = 1.0,
        eps: float = 1e-4,
    ):
        self.x_t_src = x_t_src
        self.ndim = x_t_src.dim()
        self.shift = shift
        self.eps = eps

        self.sigma_t_src = sigma_t_src.reshape(*sigma_t_src.size(), *((self.ndim - sigma_t_src.dim()) * [1]))
        self.raw_t_src = self._unwarp_t(self.sigma_t_src)
        seg = segment_size
        if isinstance(seg, Tensor) and seg.dim() < self.raw_t_src.dim():
            seg = seg.reshape(*seg.size(), *((self.ndim - seg.dim()) * [1]))
        # clamp(min=0) here while the sampler clamps to eps — that asymmetry comes
        # from the distillation fork (piflow_s2v.py vs utils_s2v_piflow.py) and the
        # students were trained against it.  Do not align the two.
        self.raw_t_dst = (self.raw_t_src - seg).clamp(min=0)
        self.segment_size = (self.raw_t_src - self.raw_t_dst).clamp(min=eps)

        self.denoising_output_x_0 = self._u_to_x_0(denoising_output, self.x_t_src, self.sigma_t_src)

    @staticmethod
    def _interpolate(x: Tensor, t: Tensor) -> Tensor:
        n = x.size(1)
        if n < _MIN_GRID_POINTS:
            return x.squeeze(1)
        t = t.clamp(min=0, max=1) * (n - 1)
        t0 = t.floor().to(torch.long).clamp(min=0, max=n - 2)
        t1 = t0 + 1
        t0t1 = torch.stack([t0, t1], dim=1)
        x0x1 = torch.gather(x, dim=1, index=t0t1.expand(-1, -1, *x.shape[2:]))
        return (t1 - t) * x0x1[:, 0] + (t - t0) * x0x1[:, 1]

    def _unwarp_t(self, sigma_t: Tensor) -> Tensor:
        return sigma_t / (self.shift + (1 - self.shift) * sigma_t)

    @staticmethod
    def _u_to_x_0(denoising_output: Tensor, x_t: Tensor, sigma_t: Tensor) -> Tensor:
        return x_t.unsqueeze(1) - sigma_t.unsqueeze(1) * denoising_output

    def pi(self, x_t: Tensor, sigma_t: Tensor) -> Tensor:
        sigma_t = sigma_t.reshape(*sigma_t.size(), *((self.ndim - sigma_t.dim()) * [1]))
        raw_t = self._unwarp_t(sigma_t)
        x_0 = self._interpolate(self.denoising_output_x_0, (raw_t - self.raw_t_dst) / self.segment_size)
        return (x_t - x_0) / sigma_t.clamp(min=self.eps)


def policy_rollout_fm(  # noqa: PLR0913
    x_t_start: Tensor,
    sigma_t_start: Tensor,
    raw_t_start: Tensor,
    raw_t_end: Tensor,
    total_substeps: int,
    policy: DXPolicy,
) -> Tensor:
    """Integrate the policy ODE across a segment with a fixed substep size."""
    num_tokens = x_t_start.size(0)
    ndim = x_t_start.dim()
    raw_t_start = raw_t_start.reshape(num_tokens, *((ndim - 1) * [1]))
    raw_t_end = raw_t_end.reshape(num_tokens, *((ndim - 1) * [1]))

    delta_raw_t = raw_t_start - raw_t_end
    num_substeps = (delta_raw_t * total_substeps).round().to(torch.long).clamp(min=1)
    substep_size = delta_raw_t / num_substeps

    raw_t, sigma_t, x_t = raw_t_start, sigma_t_start, x_t_start
    for substep_id in range(int(num_substeps.max().item())):
        u = policy.pi(x_t, sigma_t)

        raw_t_minus = (raw_t - substep_size).clamp(min=0)
        sigma_t_minus = shift_timesteps(raw_t_minus, policy.shift)
        x_t_minus = x_t + u * (sigma_t_minus - sigma_t)

        active = num_substeps > substep_id
        x_t = torch.where(active, x_t_minus, x_t)
        sigma_t = torch.where(active, sigma_t_minus, sigma_t)
        raw_t = torch.where(active, raw_t_minus, raw_t)
    return x_t


class KandinskyS2VPiflowScheduler(FlowMatchEulerDiscreteScheduler):
    """Few-step pi-Flow scheduler for distilled Kandinsky S2V 1.0 checkpoints.

    Args:
        num_train_timesteps (`int`, *optional*, defaults to 1000):
            Number of training diffusion steps.
        shift (`float`, *optional*, defaults to 10.0):
            Flow-matching timestep shift.
        n_grid (`int`, *optional*, defaults to 5):
            Number of x0 predictions packed into the model's output channels.
        eps (`float`, *optional*, defaults to 1e-6):
            Minimum raw timestep and policy denominator.
        final_step_size_scale (`float`, *optional*, defaults to 0.5):
            Size of the last segment relative to the others.
        num_policy_substeps (`int`, *optional*, defaults to 128):
            Substeps per unit of raw time used to integrate the policy.
    """

    is_piflow = True

    @register_to_config
    def __init__(  # noqa: PLR0913
        self,
        num_train_timesteps: int = 1000,
        shift: float = 10.0,
        n_grid: int = 5,
        eps: float = 1e-6,
        final_step_size_scale: float = 0.5,
        num_policy_substeps: int = 128,
    ):
        if n_grid < _MIN_GRID_POINTS:
            raise ValueError(f"n_grid must be >= {_MIN_GRID_POINTS}, got {n_grid}")
        if eps <= 0:
            raise ValueError(f"eps must be > 0, got {eps}")
        if not 0 < final_step_size_scale <= 1:
            raise ValueError(f"final_step_size_scale must be in (0, 1], got {final_step_size_scale}")
        if num_policy_substeps < 1:
            raise ValueError(f"num_policy_substeps must be >= 1, got {num_policy_substeps}")
        super().__init__(num_train_timesteps=num_train_timesteps, shift=shift)
        self._raw_timesteps = torch.empty(0)
        self._segment_sizes = torch.empty(0)

    def set_timesteps(  # noqa: PLR0913
        self,
        num_inference_steps: int | None = None,
        device: str | torch.device | None = None,
        sigmas: list[float] | None = None,
        mu: float | None = None,
        timesteps: list[float] | None = None,
    ) -> None:
        """Build the distilled segment schedule.

        Every segment has the same size except the last one, which is scaled by ``final_step_size_scale``; together
        they cover raw time from 1 down to 0.
        """
        if sigmas is not None or mu is not None or timesteps is not None:
            raise ValueError("this scheduler only supports its own distilled schedule")
        if num_inference_steps is None or num_inference_steps < 1:
            raise ValueError(f"num_inference_steps must be positive, got {num_inference_steps}")

        config = self.config
        base_segment = 1.0 / (num_inference_steps - (1.0 - config.final_step_size_scale))

        # Walked segment by segment rather than as `1 - cumsum(segments)`: the
        # native sampler subtracts one segment at a time, and in float32 the two
        # orders differ in the last bits, which the policy rollout then amplifies.
        raws, segments = [], []
        raw_t = torch.ones(1, dtype=torch.float32, device=device)
        for step in range(num_inference_steps):
            is_final = step == num_inference_steps - 1
            segment = base_segment * (config.final_step_size_scale if is_final else 1.0)
            raws.append(raw_t)
            segments.append(segment)
            raw_t = (raw_t - segment).clamp(min=config.eps)

        raw = torch.cat(raws)
        segments = torch.tensor(segments, dtype=torch.float32, device=device)
        raw_sigmas = shift_timesteps(raw, float(config.shift))

        self.num_inference_steps = int(num_inference_steps)
        self._raw_timesteps = raw
        self._segment_sizes = segments
        self.timesteps = raw_sigmas * config.num_train_timesteps
        self.sigmas = torch.cat([raw_sigmas, raw_sigmas.new_zeros(1)])
        self._step_index = None
        self._begin_index = None

    def _to_grid(self, model_output: Tensor, sample: Tensor) -> Tensor:
        """``(T, H, W, n_grid * C)`` -> ``(T, n_grid, H, W, C)``.

        Grid-major, exactly as the distilled head emits it. Do not reorder: a channel-major split would silently
        shuffle the policy grid while every shape still checks out.
        """
        n_grid = self.config.n_grid
        if model_output.shape[:-1] != sample.shape[:-1]:
            raise ValueError(f"model output {tuple(model_output.shape)} does not match sample {tuple(sample.shape)}")
        if model_output.shape[-1] != sample.shape[-1] * n_grid:
            raise ValueError(
                f"expected {sample.shape[-1] * n_grid} output channels for n_grid={n_grid}, "
                f"got {model_output.shape[-1]}"
            )
        return model_output.reshape(*model_output.shape[:-1], n_grid, sample.shape[-1]).movedim(-2, 1)

    def _resolve_step_index(self, timestep: Tensor | float) -> int:
        if self.step_index is None:
            if self.begin_index is not None:
                self._step_index = self.begin_index
            else:
                schedule = self.timesteps
                value = torch.as_tensor(timestep, device=schedule.device, dtype=schedule.dtype)
                indices = torch.nonzero(schedule == value).flatten()
                if not indices.numel():
                    raise ValueError(f"timestep {float(value)} is not in the pi-Flow schedule")
                self._step_index = int(indices[-1].item())
        if self.step_index >= self.num_inference_steps:
            raise ValueError("step() called after the schedule was exhausted")
        return int(self.step_index)

    def step(
        self,
        model_output: Tensor,
        timestep: float | Tensor,
        sample: Tensor,
        return_dict: bool = True,
    ) -> FlowMatchEulerDiscreteSchedulerOutput | tuple:
        """Advance one segment by integrating the policy.

        Args:
            model_output: the widened prediction, ``n_grid`` x0 values per channel.
            timestep: a value from ``scheduler.timesteps``.
            sample: the current noisy latent.
            return_dict: return a [`FlowMatchEulerDiscreteSchedulerOutput`].
        """
        if isinstance(timestep, int) or (isinstance(timestep, Tensor) and not timestep.is_floating_point()):
            raise ValueError("pass a value from scheduler.timesteps, not an index")

        step_index = self._resolve_step_index(timestep)
        config = self.config
        # The policy rollout runs in float32, as the native sampler does.
        sample = sample.to(torch.float32)
        grid = self._to_grid(model_output.to(torch.float32), sample)

        device = sample.device
        num_tokens = sample.shape[0]
        extra_dims = (sample.dim() - 1) * [1]

        raw_src = self._raw_timesteps[step_index].to(device)
        segment = self._segment_sizes[step_index].to(device)
        raw_dst = (raw_src - segment).clamp(min=config.eps)

        sigma_src = self.sigmas[step_index].to(device).repeat_interleave(num_tokens).reshape(num_tokens, *extra_dims)
        policy = DXPolicy(
            denoising_output=grid,
            x_t_src=sample,
            sigma_t_src=sigma_src,
            segment_size=torch.full_like(sigma_src, float(segment)),
            shift=float(config.shift),
            eps=config.eps,
        )
        updated = policy_rollout_fm(
            x_t_start=sample,
            sigma_t_start=sigma_src,
            raw_t_start=raw_src.repeat_interleave(num_tokens),
            raw_t_end=raw_dst.repeat_interleave(num_tokens),
            total_substeps=config.num_policy_substeps,
            policy=policy,
        )

        self._step_index += 1
        if return_dict:
            return FlowMatchEulerDiscreteSchedulerOutput(prev_sample=updated)
        return (updated,)
