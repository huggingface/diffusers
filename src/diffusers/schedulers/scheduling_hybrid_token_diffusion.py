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

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from ..configuration_utils import ConfigMixin, register_to_config
from ..utils import BaseOutput
from .scheduling_token_diffusion import _gumbel_argmax
from .scheduling_utils import SchedulerMixin


@dataclass
class HybridTokenDiffusionSchedulerOutput(BaseOutput):
    prev_sample: torch.LongTensor


class HybridTokenDiffusionScheduler(SchedulerMixin, ConfigMixin):
    """
    Hybrid-transition discrete token diffusion scheduler.

    This scheduler defines a forward transition kernel that mixes:
    - keeping the current token (scaled by alpha(t))
    - moving toward a mixture distribution over tokens (beta_pi(t))

    The scheduler exposes:
    - `add_noise(...)` for forward corruption
    - `step(...)` for reverse updates using the model's predicted token distribution
    """

    order = 1

    @register_to_config
    def __init__(
        self,
        vocab_size: int,
        mask_token_id: int,
        num_train_timesteps: int = 1000,
        t_eps: float = 1e-4,
        p_uniform: float = 0.0,
        clip_noise: float = 20.0,
        gamma: float = 1.0,
    ):
        if vocab_size <= 0:
            raise ValueError(f"`vocab_size` must be > 0, got {vocab_size}.")
        if not (0 <= mask_token_id < vocab_size):
            raise ValueError(f"`mask_token_id` must be in [0, vocab_size), got {mask_token_id}.")
        if num_train_timesteps <= 1:
            raise ValueError(f"`num_train_timesteps` must be > 1, got {num_train_timesteps}.")
        if not (0.0 < t_eps < 0.5):
            raise ValueError(f"`t_eps` must be in (0, 0.5), got {t_eps}.")
        if gamma <= 0:
            raise ValueError(f"`gamma` must be > 0, got {gamma}.")

        self.vocab_size = int(vocab_size)
        self.mask_token_id = int(mask_token_id)
        self.num_train_timesteps = int(num_train_timesteps)
        self.t_eps = float(t_eps)

        p_uniform = max(math.exp(-float(clip_noise)), float(p_uniform))
        log_B = float(gamma) * math.log(2.0) + math.log(p_uniform) - math.log(1.0 - p_uniform)
        log_B = float(np.clip(log_B, -float(clip_noise), float(clip_noise)))
        self.log_B = float(log_B)
        self.log_gamma = float(math.log(float(gamma)))

        self.num_inference_steps = None
        self.timesteps = None
        self._timesteps_with_end = None

        mask = torch.zeros(self.vocab_size, dtype=torch.float32)
        mask[self.mask_token_id] = 1.0
        self.mask = mask

        unif = (1.0 - mask) / max(self.vocab_size - 1, 1)
        self.unif = unif

    def set_timesteps(self, num_inference_steps: int, device: Union[str, torch.device, None] = None) -> None:
        if num_inference_steps <= 0:
            raise ValueError(f"`num_inference_steps` must be > 0, got {num_inference_steps}.")
        self.num_inference_steps = int(num_inference_steps)

        t0 = 1.0 - float(self.t_eps)
        t1 = float(self.t_eps)
        timesteps = torch.linspace(t0, t1, self.num_inference_steps + 1, dtype=torch.float32, device=device)
        self._timesteps_with_end = timesteps
        self.timesteps = timesteps[:-1]

    def scale_model_input(
        self, sample: torch.Tensor, timestep: Optional[Union[int, torch.Tensor]] = None
    ) -> torch.Tensor:
        return sample

    def _to_continuous_t(self, timesteps: torch.Tensor, device: torch.device) -> torch.Tensor:
        if timesteps.dtype in (torch.float16, torch.float32, torch.float64, torch.bfloat16):
            t = timesteps.to(device=device, dtype=torch.float32)
            return t.clamp(float(self.t_eps), 1.0 - float(self.t_eps))

        if timesteps.dtype not in (torch.int32, torch.int64):
            raise ValueError(f"`timesteps` must be float or int, got dtype={timesteps.dtype}.")

        t = timesteps.to(device=device, dtype=torch.float32) / float(self.num_train_timesteps - 1)
        t = (1.0 - 2.0 * float(self.t_eps)) * t + float(self.t_eps)
        return t.clamp(float(self.t_eps), 1.0 - float(self.t_eps))

    def _get_alpha_betapi(self, t: torch.Tensor, eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
        t = t.view(-1, 1)
        t1m = 1.0 - t

        gamma = float(math.exp(self.log_gamma))
        B = float(math.exp(self.log_B))
        c_t = (t.pow(gamma / 2.0) * t1m.pow(gamma / 2.0) * B).to(dtype=torch.float32)
        C_t = (1.0 + c_t).clamp_min(eps)

        alpha_t = t1m / C_t
        beta_pi = (
            t * self.mask.to(device=t.device, dtype=torch.float32)
            + c_t * self.unif.to(device=t.device, dtype=torch.float32)
        ) / C_t
        return alpha_t, beta_pi

    def _probs_at_t(self, probs: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        alpha_t, beta_pi = self._get_alpha_betapi(t)
        alpha_t = alpha_t.to(dtype=probs.dtype)
        beta_pi = beta_pi.to(dtype=probs.dtype)

        out = probs.mul(alpha_t.unsqueeze(1))
        out[..., : beta_pi.shape[-1]].add_(beta_pi.unsqueeze(1))
        return out

    def _sample_categorical(self, probs: torch.Tensor, generator: Optional[torch.Generator]) -> torch.LongTensor:
        bsz, seqlen, vocab = probs.shape
        flat = probs.view(-1, vocab).clamp_min(torch.finfo(probs.dtype).tiny)
        flat = flat / flat.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(probs.dtype).eps)
        sample = torch.multinomial(flat, num_samples=1, generator=generator).view(bsz, seqlen)
        return sample.to(dtype=torch.long)

    def add_noise(
        self,
        original_samples: torch.LongTensor,
        noise: Optional[torch.Tensor],
        timesteps: torch.Tensor,
    ) -> torch.LongTensor:
        del noise
        if original_samples.dtype != torch.long:
            raise ValueError(f"`original_samples` must be int64 token IDs, got dtype={original_samples.dtype}.")

        device = original_samples.device
        t = self._to_continuous_t(timesteps.to(device=device), device=device)
        onehot = F.one_hot(original_samples, num_classes=self.vocab_size).to(dtype=torch.float32)
        probs = self._probs_at_t(onehot, t)
        return self._sample_categorical(probs, generator=None)

    def _index_for_timestep(self, timestep: Union[float, torch.Tensor]) -> int:
        if self.timesteps is None:
            raise ValueError("Call `set_timesteps(...)` before calling `step()`.")

        if isinstance(timestep, torch.Tensor):
            t = float(timestep.detach().cpu().item())
        else:
            t = float(timestep)

        idx = int(torch.argmin(torch.abs(self.timesteps.detach().cpu() - torch.tensor(t))).item())
        return idx

    def step(
        self,
        model_output: torch.Tensor,
        timestep: Union[float, torch.Tensor],
        sample: torch.LongTensor,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ) -> Union[HybridTokenDiffusionSchedulerOutput, Tuple[torch.LongTensor]]:
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

        if self._timesteps_with_end is None:
            raise ValueError("Call `set_timesteps(...)` before calling `step()`.")

        device = sample.device
        batch_size, seq_len = sample.shape

        step_index = self._index_for_timestep(timestep)
        t_val = self._timesteps_with_end[step_index].to(device=device)
        s_val = self._timesteps_with_end[step_index + 1].to(device=device)

        t = t_val * torch.ones(batch_size, device=device, dtype=torch.float32)
        s = s_val * torch.ones(batch_size, device=device, dtype=torch.float32)

        logits = model_output.to(dtype=torch.float32)
        logits = logits.clone()
        logits[..., self.mask_token_id] = torch.finfo(logits.dtype).min
        probs = logits.softmax(dim=-1)

        q_s = self._probs_at_t(probs, s)
        q_t = self._probs_at_t(probs, t)
        q_zt = q_t.gather(-1, sample.unsqueeze(-1)).clamp_min(torch.finfo(torch.float32).eps)

        alpha_t, beta_pi_t = self._get_alpha_betapi(t)
        alpha_s, beta_pi_s = self._get_alpha_betapi(s)

        alpha_ts = (alpha_t / alpha_s).clamp_min(torch.finfo(torch.float32).eps)
        beta_pi_ts = beta_pi_t - (alpha_t / alpha_s) * beta_pi_s

        vz_t = F.one_hot(sample, num_classes=self.vocab_size).to(dtype=torch.float32)
        beta_pi_ts_at_zt = beta_pi_ts.unsqueeze(1).expand_as(vz_t).gather(-1, sample.unsqueeze(-1))
        q_ts = alpha_ts.view(batch_size, 1, 1) * vz_t + beta_pi_ts_at_zt

        q_st = q_ts * q_s / q_zt
        q_st = q_st.clamp_min(torch.finfo(torch.float32).tiny)
        q_st = q_st / q_st.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(torch.float32).eps)

        x_prev = _gumbel_argmax(torch.log(q_st), generator=generator).to(dtype=torch.long)

        if not return_dict:
            return (x_prev,)
        return HybridTokenDiffusionSchedulerOutput(prev_sample=x_prev)


__all__ = ["HybridTokenDiffusionScheduler", "HybridTokenDiffusionSchedulerOutput"]
