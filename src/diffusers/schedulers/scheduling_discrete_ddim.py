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

import torch

from ..configuration_utils import ConfigMixin, register_to_config
from ..utils import BaseOutput
from .scheduling_utils import SchedulerMixin


@dataclass
class DiscreteDDIMSchedulerOutput(BaseOutput):
    """
    Output class for the discrete DDIM scheduler.

    Args:
        prev_sample (`torch.LongTensor` of shape `(batch_size, block_length)`):
            Updated block tokens after the current denoising step.
        sampled_tokens (`torch.LongTensor` of shape `(batch_size, block_length)`):
            Token IDs sampled from the model logits, i.e. the predicted clean tokens `x0`.
        sampled_probs (`torch.Tensor` of shape `(batch_size, block_length)`):
            Probabilities of the sampled tokens.
    """

    prev_sample: torch.LongTensor
    sampled_tokens: torch.LongTensor
    sampled_probs: torch.Tensor


class DiscreteDDIMScheduler(SchedulerMixin, ConfigMixin):
    """
    Discrete DDIM scheduler for the uniform corruption process, following "Structured Denoising Diffusion Models in
    Discrete State-Spaces" (D3PM, https://huggingface.co/papers/2107.03006).

    On the linear schedule the survival probability of a clean token at time `t` is `alpha(t) = 1 - t`. One denoising
    step from time `t` to `s < t` samples every block position from the exact posterior `q(x_s | x_t, x0)`, which for
    the uniform kernel decomposes into three routes: jump to the predicted clean token `x0`, stay on the current token,
    or jump to a uniformly random token. Unlike masked diffusion, there is no mask token; uncommitted positions carry
    random tokens.

    Args:
        num_inference_steps (`int`, defaults to 32):
            The number of denoising steps, defining the linear time grid the posterior is evaluated on.
    """

    order = 1

    @register_to_config
    def __init__(self, num_inference_steps: int = 32):
        self.num_inference_steps = num_inference_steps
        self.timesteps = torch.arange(num_inference_steps, dtype=torch.long)

    def set_timesteps(self, num_inference_steps: int, device: str | torch.device | None = None) -> None:
        if num_inference_steps <= 0:
            raise ValueError(f"`num_inference_steps` must be > 0, got {num_inference_steps}.")
        self.num_inference_steps = num_inference_steps
        self.timesteps = torch.arange(num_inference_steps, device=device, dtype=torch.long)

    @staticmethod
    def _sample_from_logits(
        logits: torch.Tensor,
        *,
        temperature: float,
        generator: torch.Generator | None,
    ) -> tuple[torch.LongTensor, torch.Tensor]:
        """Sample one token per position with optional temperature, returning tokens and their probabilities."""
        if temperature < 0:
            raise ValueError(f"`temperature` must be >= 0, got {temperature}.")

        vocab_size = logits.shape[-1]
        flat_logits = logits.reshape(-1, vocab_size)
        probs = torch.softmax(flat_logits.float(), dim=-1)

        if temperature == 0.0:
            token = flat_logits.argmax(dim=-1, keepdim=True)
        else:
            scaled_probs = torch.softmax(flat_logits.float() / temperature, dim=-1)
            token = torch.multinomial(scaled_probs, num_samples=1, generator=generator)

        token_prob = torch.gather(probs, -1, token)
        return token.view(*logits.shape[:-1]), token_prob.view(*logits.shape[:-1])

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int | torch.Tensor,
        sample: torch.LongTensor,
        *,
        temperature: float = 0.0,
        generator: torch.Generator | None = None,
        return_dict: bool = True,
    ) -> DiscreteDDIMSchedulerOutput | tuple[torch.LongTensor, torch.LongTensor, torch.Tensor]:
        """
        Sample the next block from the posterior `q(x_s | x_t, x0)` of the uniform corruption process.

        With `a = alpha_t / alpha_s` (survival probability from `s` to `t`) and `b = alpha_s`, the posterior mass of
        each route is

            clean: `b * (1 - a) / K + a * b * 1[x_t = x0]`, stay: `a * (1 - b) / K`, noise: `(1 - a) * (1 - b) / K`,

        so the last step (`b = 1`) deterministically commits the predicted clean tokens.

        Args:
            model_output (`torch.Tensor` of shape `(batch_size, block_length, vocab_size)`):
                Raw logits from the model for the current block.
            timestep (`int` or `torch.Tensor`):
                Current step index within the denoising schedule, in `[0, num_inference_steps - 1]`.
            sample (`torch.LongTensor` of shape `(batch_size, block_length)`):
                Current block token IDs `x_t`.
            temperature (`float`):
                Sampling temperature applied to the logits when drawing `x0`.
            generator (`torch.Generator`, *optional*):
                RNG for sampling.
            return_dict (`bool`):
                Whether to return a [`DiscreteDDIMSchedulerOutput`] or a plain tuple.
        """
        if isinstance(timestep, torch.Tensor):
            step_index = int(timestep.item())
        else:
            step_index = int(timestep)

        sampled_tokens, sampled_probs = self._sample_from_logits(
            model_output, temperature=temperature, generator=generator
        )

        vocab_size = model_output.shape[-1]
        num_steps = self.num_inference_steps
        # `step_index` counts up from 0 to `num_inference_steps - 1`: alpha(t) = 1 - t increases towards the clean end,
        # with alpha_s = 1 on the final step so the predicted clean tokens are committed deterministically.
        alpha_t = step_index / num_steps
        alpha_s = (step_index + 1) / num_steps
        survival = alpha_t / alpha_s

        same = (sample == sampled_tokens).float()
        clean_mass = alpha_s * (1 - survival) / vocab_size + survival * alpha_s * same
        stay_mass = survival * (1 - alpha_s) / vocab_size * torch.ones_like(same)
        noise_mass = (1 - survival) * (1 - alpha_s) / vocab_size * torch.ones_like(same)

        route_probs = torch.stack([clean_mass, stay_mass, noise_mass], dim=-1)
        route_probs = route_probs / route_probs.sum(dim=-1, keepdim=True)
        routes = torch.multinomial(route_probs.view(-1, 3), num_samples=1, generator=generator).view_as(sample)

        random_tokens = torch.randint(
            low=0, high=vocab_size, size=sample.shape, device=sample.device, generator=generator
        )
        prev_sample = torch.where(routes == 0, sampled_tokens, sample)
        prev_sample = torch.where(routes == 2, random_tokens, prev_sample)

        if not return_dict:
            return prev_sample, sampled_tokens, sampled_probs
        return DiscreteDDIMSchedulerOutput(
            prev_sample=prev_sample,
            sampled_tokens=sampled_tokens,
            sampled_probs=sampled_probs,
        )


__all__ = ["DiscreteDDIMScheduler", "DiscreteDDIMSchedulerOutput"]
