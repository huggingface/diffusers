# Copyright 2025 The HuggingFace Team. All rights reserved.
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

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from ..configuration_utils import ConfigMixin, register_to_config
from ..utils import BaseOutput
from .scheduling_utils import SchedulerMixin


@dataclass
class TokenDiffusionSchedulerOutput(BaseOutput):
    """
    Output class for discrete token schedulers.

    Args:
        prev_sample (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Sample at the previous timestep. This should be fed into the model at the next denoising iteration.
    """

    prev_sample: torch.LongTensor


def _gumbel_argmax(logits: torch.Tensor, generator: Optional[torch.Generator] = None) -> torch.LongTensor:
    """
    Sample from a categorical distribution defined by (unnormalized) logits via Gumbel-max.

    Args:
        logits: Tensor of shape `(..., vocab_size)`.
        generator: Optional torch generator for determinism.

    Returns:
        `torch.LongTensor` of shape `logits.shape[:-1]` with sampled indices.
    """
    # Gumbel(0,1) noise: -log(-log(U))
    uniform = torch.rand(logits.shape, device=logits.device, dtype=logits.dtype, generator=generator).clamp_(1e-30, 1)
    gumbel = -torch.log(-torch.log(uniform))
    return (logits + gumbel).argmax(dim=-1)


class TokenDiffusionScheduler(SchedulerMixin, ConfigMixin):
    """
    Discrete diffusion scheduler over token IDs (categorical states).

    This scheduler is designed for *token-space* diffusion (e.g. masked/absorbing diffusion language models) and
    follows the diffusers scheduler API where possible: `set_timesteps()` for inference and `step()` for reverse
    updates.

    Currently implemented:
    - Forward process:
        - `absorbing`: with probability `1 - alpha(t)` replace token with `mask_token_id`.
        - `uniform`: with probability `1 - alpha(t)` replace token with a uniform random token.
    - Noise schedule: selectable `alpha(t)` families with `t in [0, 1]`.

    Notes:
    - `step()` expects the model to return logits over vocabulary for `x0` reconstruction.
    - The mask token is treated as an *absorbing state* and is never sampled as an `x0` prediction.
    """

    order = 1

    @register_to_config
    def __init__(
        self,
        vocab_size: int,
        mask_token_id: int,
        num_train_timesteps: int = 1000,
        alpha_schedule: str = "log_linear",
        eps: float = 1e-3,
        sigma_min: float = 1e-4,
        sigma_max: float = 20.0,
        forward_process: str = "absorbing",
        exclude_mask_from_uniform: bool = True,
    ):
        if vocab_size <= 0:
            raise ValueError(f"`vocab_size` must be > 0, got {vocab_size}.")
        if num_train_timesteps <= 1:
            raise ValueError(f"`num_train_timesteps` must be > 1, got {num_train_timesteps}.")
        if not (0.0 < eps < 1.0):
            raise ValueError(f"`eps` must be in (0, 1), got {eps}.")
        if not (0 <= mask_token_id < vocab_size):
            raise ValueError(f"`mask_token_id` must be in [0, vocab_size), got {mask_token_id}.")
        alpha_schedule = str(alpha_schedule).lower()
        if alpha_schedule not in {"log_linear", "linear", "cosine", "geometric"}:
            raise ValueError(
                "`alpha_schedule` must be one of {'log_linear','linear','cosine','geometric'}, got"
                f" {alpha_schedule!r}."
            )
        if sigma_min <= 0 or sigma_max <= 0:
            raise ValueError(
                f"`sigma_min` and `sigma_max` must be > 0, got sigma_min={sigma_min}, sigma_max={sigma_max}."
            )
        if sigma_max <= sigma_min:
            raise ValueError(f"`sigma_max` must be > `sigma_min`, got sigma_min={sigma_min}, sigma_max={sigma_max}.")
        if forward_process not in {"absorbing", "uniform"}:
            raise ValueError(f"`forward_process` must be one of {{'absorbing','uniform'}}, got {forward_process!r}.")

        self.vocab_size = int(vocab_size)
        self.mask_token_id = int(mask_token_id)
        self.num_train_timesteps = int(num_train_timesteps)
        self.alpha_schedule = alpha_schedule
        self.eps = float(eps)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.forward_process = str(forward_process)
        self.exclude_mask_from_uniform = bool(exclude_mask_from_uniform)

        self.num_inference_steps = None
        self.timesteps = torch.from_numpy(np.arange(0, num_train_timesteps)[::-1].copy())

    def _effective_vocab_size(self) -> int:
        if self.forward_process == "uniform" and self.exclude_mask_from_uniform:
            return self.vocab_size - 1
        return self.vocab_size

    def _sample_uniform_tokens(
        self, shape: torch.Size, device: torch.device, dtype: torch.dtype, generator: Optional[torch.Generator] = None
    ) -> torch.LongTensor:
        """
        Sample uniform token IDs, optionally excluding `mask_token_id` (by shifting indices around it).
        """
        if self.forward_process != "uniform":
            raise ValueError("Uniform token sampling is only valid for `forward_process='uniform'`.")

        if not self.exclude_mask_from_uniform:
            return torch.randint(0, self.vocab_size, shape, device=device, dtype=dtype, generator=generator)

        # Sample in [0, vocab_size-1) and shift around mask_token_id.
        v_eff = self.vocab_size - 1
        draw = torch.randint(0, v_eff, shape, device=device, dtype=dtype, generator=generator)
        return torch.where(draw >= self.mask_token_id, draw + 1, draw)

    def sample_prior(
        self,
        shape: torch.Size,
        device: torch.device,
        generator: Optional[torch.Generator] = None,
    ) -> torch.LongTensor:
        """
        Sample from the prior distribution of the forward process at t=1.

        For `forward_process="absorbing"`, returns a tensor filled with `mask_token_id`. For
        `forward_process="uniform"`, returns uniform random token IDs (optionally excluding `mask_token_id`).

        Args:
            shape (`torch.Size`):
                Desired output shape, e.g. `(batch_size, seq_len)`.
            device (`torch.device`):
                Device for the output tensor.
            generator (`torch.Generator`, *optional*):
                Optional generator for determinism (only used for the uniform process).

        Returns:
            `torch.LongTensor` of shape `shape` with sampled prior token IDs.
        """
        if self.forward_process == "uniform":
            return self._sample_uniform_tokens(shape, device=device, dtype=torch.long, generator=generator)
        return torch.full(shape, self.mask_token_id, device=device, dtype=torch.long)

    def set_timesteps(self, num_inference_steps: int, device: Union[str, torch.device, None] = None) -> None:
        """
        Sets the discrete timesteps used for the diffusion chain (to be run before inference).

        Timesteps are stored in descending order, so `timesteps[0]` is the noisiest step.
        """
        if num_inference_steps <= 0:
            raise ValueError(f"`num_inference_steps` must be > 0, got {num_inference_steps}.")
        self.num_inference_steps = int(num_inference_steps)

        # Standard diffusers behavior: map inference steps onto training step indices.
        timesteps = torch.linspace(
            self.num_train_timesteps - 1, 0, self.num_inference_steps, dtype=torch.float32
        ).round()
        self.timesteps = timesteps.to(dtype=torch.long, device=device)

    def scale_model_input(self, sample: torch.Tensor, timestep: Optional[int] = None) -> torch.Tensor:
        return sample

    def _t_from_timestep(self, timestep: Union[int, torch.Tensor], device: torch.device) -> torch.Tensor:
        """
        Convert an integer training timestep index into continuous time `t in [0, 1]`.
        """
        if isinstance(timestep, torch.Tensor):
            t_idx = timestep.to(device=device, dtype=torch.float32)
        else:
            t_idx = torch.tensor(float(timestep), device=device, dtype=torch.float32)
        denom = float(self.num_train_timesteps - 1)
        return (t_idx / denom).clamp_(0.0, 1.0)

    def _alpha_t(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute alpha(t) for the configured schedule.

        The returned tensor is expected to be in (0, 1] and monotone decreasing in `t`.
        """
        if self.alpha_schedule == "log_linear":
            # alpha(t) = 1 - (1 - eps) * t
            return 1.0 - (1.0 - self.eps) * t

        if self.alpha_schedule == "linear":
            # alpha(t) = (1 - 2*eps) * (1 - t) + eps
            return (1.0 - 2.0 * self.eps) * (1.0 - t) + self.eps

        if self.alpha_schedule == "cosine":
            # alpha_base(t) = 1 - cos(pi/2 * (1 - t))
            # alpha(t) = (1 - 2*eps) * alpha_base(t) + eps
            base = 1.0 - torch.cos(torch.pi / 2.0 * (1.0 - t))
            return (1.0 - 2.0 * self.eps) * base + self.eps

        if self.alpha_schedule == "geometric":
            # total_noise(t) = sigma_min^(1-t) * sigma_max^t
            # alpha(t) = exp(-total_noise(t))
            sigma_min = torch.as_tensor(self.sigma_min, device=t.device, dtype=t.dtype)
            sigma_max = torch.as_tensor(self.sigma_max, device=t.device, dtype=t.dtype)
            total_noise = (sigma_min ** (1.0 - t)) * (sigma_max**t)
            return (-total_noise).exp()

        raise ValueError(f"Unsupported alpha schedule: {self.alpha_schedule!r}")

    def _alpha_prime_t(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute d/dt alpha(t) for the configured schedule.
        """
        if self.alpha_schedule == "log_linear":
            return -(1.0 - self.eps) * torch.ones_like(t)

        if self.alpha_schedule == "linear":
            return -(1.0 - 2.0 * self.eps) * torch.ones_like(t)

        if self.alpha_schedule == "cosine":
            base_prime = -(torch.pi / 2.0) * torch.sin(torch.pi / 2.0 * (1.0 - t))
            return (1.0 - 2.0 * self.eps) * base_prime

        if self.alpha_schedule == "geometric":
            sigma_min = torch.as_tensor(self.sigma_min, device=t.device, dtype=t.dtype)
            sigma_max = torch.as_tensor(self.sigma_max, device=t.device, dtype=t.dtype)
            total_noise = (sigma_min ** (1.0 - t)) * (sigma_max**t)
            alpha = (-total_noise).exp()
            rate = total_noise * (sigma_max.log() - sigma_min.log())
            return -alpha * rate

        raise ValueError(f"Unsupported alpha schedule: {self.alpha_schedule!r}")

    def get_mdlm_loss_weights(self, timesteps: torch.LongTensor) -> torch.Tensor:
        """
        Return per-example positive loss weights for masked-token reconstruction objectives.

        The weight corresponds to `-alpha'(t) / (1 - alpha(t))`, which is positive for monotone decreasing alpha(t).

        Args:
            timesteps (`torch.LongTensor` of shape `(batch_size,)`):
                Training timestep indices in `[0, num_train_timesteps-1]`.

        Returns:
            `torch.FloatTensor` of shape `(batch_size, 1)`:
                Positive weights to multiply token-level cross-entropy by.
        """
        if timesteps.dtype not in (torch.int32, torch.int64):
            raise ValueError(f"`timesteps` must be an integer tensor, got dtype={timesteps.dtype}.")
        device = timesteps.device
        t = self._t_from_timestep(timesteps.to(device), device=device)
        t = t.to(dtype=torch.float32)
        alpha = self._alpha_t(t).to(dtype=torch.float32)
        dalpha = self._alpha_prime_t(t).to(dtype=torch.float32)
        denom = (1.0 - alpha).clamp_min(torch.finfo(torch.float32).eps)
        w = (-dalpha / denom).clamp_min(torch.finfo(torch.float32).tiny)
        return w.view(-1, 1)

    def get_alpha(self, timesteps: torch.LongTensor) -> torch.Tensor:
        """
        Return per-example alpha(t) values for the configured schedule.

        Args:
            timesteps (`torch.LongTensor` of shape `(batch_size,)`):
                Training timestep indices in `[0, num_train_timesteps-1]`.

        Returns:
            `torch.FloatTensor` of shape `(batch_size, 1)`:
                Alpha values in `(0, 1]` for each example.
        """
        if timesteps.dtype not in (torch.int32, torch.int64):
            raise ValueError(f"`timesteps` must be an integer tensor, got dtype={timesteps.dtype}.")
        device = timesteps.device
        t = self._t_from_timestep(timesteps.to(device), device=device).to(dtype=torch.float32)
        alpha = self._alpha_t(t).to(dtype=torch.float32)
        return alpha.view(-1, 1)

    def get_alpha_prime(self, timesteps: torch.LongTensor) -> torch.Tensor:
        """
        Return per-example time derivative alpha'(t) for the configured schedule.

        Args:
            timesteps (`torch.LongTensor` of shape `(batch_size,)`):
                Training timestep indices in `[0, num_train_timesteps-1]`.

        Returns:
            `torch.FloatTensor` of shape `(batch_size, 1)`:
                Alpha derivatives with respect to continuous time `t in [0, 1]`.
        """
        if timesteps.dtype not in (torch.int32, torch.int64):
            raise ValueError(f"`timesteps` must be an integer tensor, got dtype={timesteps.dtype}.")
        device = timesteps.device
        t = self._t_from_timestep(timesteps.to(device), device=device).to(dtype=torch.float32)
        dalpha = self._alpha_prime_t(t).to(dtype=torch.float32)
        return dalpha.view(-1, 1)

    def add_noise(
        self,
        original_samples: torch.LongTensor,
        noise: Optional[torch.Tensor],
        timesteps: torch.LongTensor,
    ) -> torch.LongTensor:
        """
        Apply the (absorbing) forward process q(x_t | x_0).

        The `noise` argument is accepted for API compatibility but is not used for the absorbing kernel.
        """
        del noise

        if original_samples.dtype != torch.long:
            raise ValueError(f"`original_samples` must be int64 token IDs, got dtype={original_samples.dtype}.")
        if timesteps.dtype not in (torch.int32, torch.int64):
            raise ValueError(f"`timesteps` must be an integer tensor, got dtype={timesteps.dtype}.")

        batch_size, seq_len = original_samples.shape
        device = original_samples.device

        # Convert per-example timesteps into alpha(t) in [eps, 1].
        t = self._t_from_timestep(timesteps.to(device), device=device).view(batch_size, 1)
        alpha = self._alpha_t(t).to(dtype=torch.float32)

        p_replace = (1.0 - alpha).expand(batch_size, seq_len)
        rand = torch.rand((batch_size, seq_len), device=device, dtype=torch.float32)
        replace_positions = rand < p_replace

        if self.forward_process == "absorbing":
            replacement = torch.full_like(original_samples, self.mask_token_id)
        elif self.forward_process == "uniform":
            replacement = self._sample_uniform_tokens(
                original_samples.shape, device=device, dtype=original_samples.dtype, generator=None
            )
        else:
            raise ValueError(f"Unsupported forward process: {self.forward_process!r}")

        return torch.where(replace_positions, replacement, original_samples)

    def step(
        self,
        model_output: torch.Tensor,
        timestep: Union[int, torch.Tensor],
        sample: torch.LongTensor,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ) -> Union[TokenDiffusionSchedulerOutput, Tuple[torch.LongTensor]]:
        """
        Reverse diffusion step for the configured forward process.

        For `forward_process="absorbing"`, the update mirrors the common absorbing posterior:
        - Keep all unmasked positions fixed.
        - For masked positions, with probability p_denoise replace mask by a sample from p_theta(x0 | x_t, t).

        For `forward_process="uniform"`, this implements the discrete posterior used by UDLM-style uniform token
        diffusion.
        """
        if sample.dtype != torch.long:
            raise ValueError(f"`sample` must be int64 token IDs, got dtype={sample.dtype}.")
        if model_output.ndim != 3 or model_output.shape[-1] != self.vocab_size:
            raise ValueError(
                f"`model_output` must have shape [batch, seq_len, vocab_size={self.vocab_size}], got {tuple(model_output.shape)}."
            )
        if model_output.shape[0] != sample.shape[0] or model_output.shape[1] != sample.shape[1]:
            raise ValueError(
                f"`model_output` batch/seq dims {tuple(model_output.shape[:2])} must match `sample` {tuple(sample.shape)}."
            )

        device = sample.device
        batch_size, seq_len = sample.shape

        # Figure out the previous timestep in the configured inference schedule.
        if self.num_inference_steps is None:
            raise ValueError("Call `set_timesteps(num_inference_steps, ...)` before calling `step()`.")

        if isinstance(timestep, torch.Tensor):
            timestep_int = int(timestep.item())
        else:
            timestep_int = int(timestep)

        # Find current index in timesteps and use the next value as "previous" time (less noisy).
        # If we are at the end, perform a "noise removal" step (alpha_prev = 1).
        current_indices = (self.timesteps == timestep_int).nonzero(as_tuple=False)
        if current_indices.numel() == 0:
            raise ValueError(f"`timestep` ({timestep_int}) must be one of `self.timesteps`.")
        step_index = int(current_indices[0].item())
        is_noise_removal_step = step_index + 1 >= len(self.timesteps)
        prev_timestep_int = int(self.timesteps[step_index + 1].item()) if not is_noise_removal_step else 0

        t = self._t_from_timestep(timestep_int, device=device)
        alpha_t = self._alpha_t(t).to(dtype=torch.float32)
        if is_noise_removal_step:
            alpha_prev = torch.tensor(1.0, device=device, dtype=torch.float32)
        else:
            t_prev = self._t_from_timestep(prev_timestep_int, device=device)
            alpha_prev = self._alpha_t(t_prev).to(dtype=torch.float32)

        if self.forward_process == "uniform":
            # Convert logits to probabilities for x0; optionally forbid mask token.
            logits = model_output.to(dtype=torch.float32)
            if self.exclude_mask_from_uniform:
                logits = logits.clone()
                logits[..., self.mask_token_id] = torch.finfo(logits.dtype).min
            p_x0 = logits.softmax(dim=-1)

            V = self.vocab_size
            x = sample
            xt_one_hot = F.one_hot(x, V).to(dtype=p_x0.dtype)

            alpha_ts = (alpha_t / alpha_prev).clamp_min(torch.finfo(torch.float32).eps)

            if self.exclude_mask_from_uniform:
                limiting = torch.full((V,), 1.0 / float(V - 1), device=device, dtype=p_x0.dtype)
                limiting[self.mask_token_id] = 0.0
            else:
                limiting = torch.full((V,), 1.0 / float(V), device=device, dtype=p_x0.dtype)
            limiting = limiting.view(1, 1, -1)

            alpha_t3 = alpha_t.view(1, 1, 1)
            alpha_s3 = alpha_prev.view(1, 1, 1)
            alpha_ts3 = alpha_ts.view(1, 1, 1)

            numerator = (
                (alpha_t3 * V * p_x0 * xt_one_hot)
                + ((alpha_ts3 - alpha_t3) * xt_one_hot)
                + ((alpha_s3 - alpha_t3) * p_x0)
                + ((1.0 - alpha_ts3) * (1.0 - alpha_s3) * limiting)
            )
            denom = (alpha_t3 * V * p_x0.gather(-1, x.unsqueeze(-1)) + (1.0 - alpha_t3)).clamp_min(
                torch.finfo(torch.float32).eps
            )

            q_xs = numerator / denom
            q_xs = q_xs.clamp_min(torch.finfo(torch.float32).tiny)
            q_xs = q_xs / q_xs.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(torch.float32).eps)

            x_prev = _gumbel_argmax(torch.log(q_xs), generator=generator).to(dtype=torch.long)

            if not return_dict:
                return (x_prev,)
            return TokenDiffusionSchedulerOutput(prev_sample=x_prev)

        if self.forward_process != "absorbing":
            raise ValueError(f"Unsupported forward process for `step()`: {self.forward_process!r}")

        # p_denoise = (alpha_prev - alpha_t) / (1 - alpha_t)
        denom = (1.0 - alpha_t).clamp_min(torch.finfo(torch.float32).eps)
        p_denoise = ((alpha_prev - alpha_t) / denom).clamp(0.0, 1.0)

        # Sample x0 predictions (never sample the mask token).
        logits = model_output.to(dtype=torch.float32)
        logits[..., self.mask_token_id] = torch.finfo(logits.dtype).min
        sampled_x0 = _gumbel_argmax(logits, generator=generator).to(dtype=torch.long)

        # Only masked positions can change.
        is_masked = sample == self.mask_token_id

        # Bernoulli draw for whether to denoise at this step (only matters on masked positions).
        rand = torch.rand((batch_size, seq_len), device=device, dtype=torch.float32, generator=generator)
        should_denoise = rand < float(p_denoise.item())

        x_prev = torch.where(is_masked & should_denoise, sampled_x0, sample)

        if not return_dict:
            return (x_prev,)
        return TokenDiffusionSchedulerOutput(prev_sample=x_prev)


__all__ = ["TokenDiffusionScheduler", "TokenDiffusionSchedulerOutput"]
