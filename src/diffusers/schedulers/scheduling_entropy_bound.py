# Copyright 2026 The HuggingFace Team. All rights reserved.
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

from typing import Optional, Union

import torch

from ..configuration_utils import ConfigMixin, register_to_config
from .scheduling_utils import DiscreteSchedulerOutput, SchedulerMixin


class EntropyBoundScheduler(SchedulerMixin, ConfigMixin):
    """
    Entropy bound scheduler for the uniform corruption process.

    At each step the scheduler samples a candidate token per position and accepts the `k` lowest-entropy positions such
    that `sum_i^k entropy_i - max(entropy_1, ..., entropy_k) <= entropy_bound`. The left-hand side upper-bounds the
    joint mutual information between the accepted tokens, so they are approximately independent. Accepted positions
    keep their sampled token; the rest are renoised with uniformly random tokens (there is no mask token).

    Proposed in "Accelerated Sampling from Masked Diffusion Models via Entropy Bounded Unmasking"
    (https://huggingface.co/papers/2505.24857).

    The sampling temperature is annealed from `t_max` on the first step down to `t_min` on the last, matching the
    released checkpoint's sampler (sharper sampling as denoising advances). It is applied to the logits before both the
    candidate sampling and the entropy that drives acceptance.

    Args:
        entropy_bound (`float`, defaults to 0.1):
            The maximum tolerated joint entropy of the accepted tokens. Larger values accept more tokens per step.
        t_max (`float`, defaults to 0.8):
            Sampling temperature on the first denoising step.
        t_min (`float`, defaults to 0.4):
            Sampling temperature on the last denoising step.
        num_inference_steps (`int`, defaults to 32):
            The maximum number of denoising steps.
    """

    order = 1

    @register_to_config
    def __init__(
        self, entropy_bound: float = 0.1, t_max: float = 0.8, t_min: float = 0.4, num_inference_steps: int = 32
    ):
        # The annealed temperature divides the logits, so it must stay strictly positive across the whole
        # schedule. Since it moves linearly between `t_min` and `t_max`, bounding both ends is what guarantees
        # that -- `t_max > 0` alone would still let a negative `t_min` cross exactly zero mid-schedule, where
        # the division yields `inf` and the entropy becomes `nan`.
        if t_max <= 0.0 or t_min < 0.0:
            raise ValueError(
                f"`t_max` must be > 0 and `t_min` must be >= 0, got t_max={t_max}, t_min={t_min}. The annealed "
                "temperature divides the logits, so it can never reach zero. For greedy sampling use "
                "`DiscreteDDIMScheduler`."
            )
        self._step_index = None
        self._begin_index = None
        self.set_timesteps(num_inference_steps)

    @property
    def step_index(self):
        """
        The index counter for current timestep. It will increase 1 after each scheduler step.
        """
        return self._step_index

    @property
    def begin_index(self):
        """
        The index for the first timestep. It should be set from pipeline with `set_begin_index` method.
        """
        return self._begin_index

    # Copied from diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler.set_begin_index
    def set_begin_index(self, begin_index: int = 0):
        """
        Sets the begin index for the scheduler. This function should be run from pipeline before the inference.

        Args:
            begin_index (`int`, defaults to `0`):
                The begin index for the scheduler.
        """
        self._begin_index = begin_index

    def set_timesteps(self, num_inference_steps: int, device: str | torch.device | None = None) -> None:
        """
        Set the discrete timestep grid, as the decreasing corruption level `t` in `[0, 1]`.

        The grid matches [`~DiscreteDDIMScheduler.set_timesteps`] — `1.0` down to `1 / num_inference_steps` — so the
        two schedulers are interchangeable in a pipeline loop. `timesteps` is the public loop variable; the annealed
        sampling temperature is derived from the integer `step_index` so it does not inherit float rounding.

        Args:
            num_inference_steps (`int`):
                The maximum number of denoising steps.
            device (`str` or `torch.device`, *optional*):
                The device the timesteps should be moved to.
        """
        if num_inference_steps <= 0:
            raise ValueError(f"`num_inference_steps` must be > 0, got {num_inference_steps}.")
        self.num_inference_steps = num_inference_steps
        self.timesteps = (
            1.0 - torch.arange(num_inference_steps, device=device, dtype=torch.float32) / num_inference_steps
        )
        self._step_index = None
        self._begin_index = None

    # Copied from diffusers.schedulers.scheduling_flow_match_euler_discrete.FlowMatchEulerDiscreteScheduler.index_for_timestep
    def index_for_timestep(
        self,
        timestep: Union[float, torch.FloatTensor],
        schedule_timesteps: Optional[torch.FloatTensor] = None,
    ) -> int:
        """
        Get the index for the given timestep.

        Args:
            timestep (`float` or `torch.FloatTensor`):
                The timestep to find the index for.
            schedule_timesteps (`torch.FloatTensor`, *optional*):
                The schedule timesteps to validate against. If `None`, the scheduler's timesteps are used.

        Returns:
            `int`:
                The index of the timestep.
        """
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps

        indices = (schedule_timesteps == timestep).nonzero()

        # The sigma index that is taken for the **very** first `step`
        # is always the second index (or the last index if there is only 1)
        # This way we can ensure we don't accidentally skip a sigma in
        # case we start in the middle of the denoising schedule (e.g. for image-to-image)
        pos = 1 if len(indices) > 1 else 0

        return indices[pos].item()

    # Copied from diffusers.schedulers.scheduling_flow_match_euler_discrete.FlowMatchEulerDiscreteScheduler._init_step_index
    def _init_step_index(self, timestep: Union[float, torch.FloatTensor]) -> None:
        if self.begin_index is None:
            if isinstance(timestep, torch.Tensor):
                timestep = timestep.to(self.timesteps.device)
            self._step_index = self.index_for_timestep(timestep)
        else:
            self._step_index = self._begin_index

    @staticmethod
    # Copied from diffusers.schedulers.scheduling_discrete_ddim.DiscreteDDIMScheduler._sample_from_logits
    def _sample_from_logits(
        logits: torch.Tensor,
        *,
        temperature: float,
        generator: torch.Generator | None,
    ) -> tuple[torch.LongTensor, torch.Tensor]:
        """
        Draw one token per position, returning the tokens and their probabilities.

        The draw is temperature-scaled; the returned probabilities are gathered from the softmax of `logits` *before*
        that scaling, so a confidence threshold does not move with `temperature`.
        """
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
        timestep: float | torch.Tensor,
        sample: torch.LongTensor,
        *,
        generator: torch.Generator | None = None,
        return_dict: bool = True,
    ) -> DiscreteSchedulerOutput | tuple:
        """
        Accept the lowest-entropy positions under the entropy bound and renoise the rest.

        Args:
            model_output (`torch.Tensor` of shape `(batch_size, block_length, vocab_size)`):
                Raw logits from the model for the current block.
            timestep (`float` or `torch.Tensor`):
                The current corruption level, one entry of [`~EntropyBoundScheduler.timesteps`]. Sets the annealed
                sampling temperature.
            sample (`torch.LongTensor` of shape `(batch_size, block_length)`):
                Current block token IDs.
            generator (`torch.Generator`, *optional*):
                RNG for sampling.
            return_dict (`bool`):
                Whether to return a [`DiscreteSchedulerOutput`] or a plain tuple.
        """
        if self.step_index is None:
            self._init_step_index(timestep)

        entropy_bound = float(self.config.entropy_bound)

        # Anneal the temperature from `t_max` on the first step down to `t_min` on the last. The fraction comes from
        # the integer `step_index`, not from `timestep`, so it is exact for any `num_inference_steps` (see
        # `set_timesteps`).
        fraction = (self.num_inference_steps - self.step_index) / self.num_inference_steps
        temperature = self.config.t_min + (self.config.t_max - self.config.t_min) * fraction

        # The candidates are drawn from the annealed distribution, and the acceptance entropy is measured on that
        # same distribution — the annealing is part of this scheduler's rule, not a user-facing knob. The reported
        # `sampled_probs`, however, come from the *unshaped* denoiser distribution, which is what
        # `DiscreteSchedulerOutput` promises; `_sample_from_logits` does both in one pass.
        sampled_tokens, sampled_probs = self._sample_from_logits(
            model_output, temperature=temperature, generator=generator
        )
        scaled_logits = model_output / temperature

        token_entropy = torch.distributions.Categorical(logits=scaled_logits).entropy()  # (batch, block_length)
        sorted_token_entropy, sorted_indices = torch.sort(token_entropy, dim=-1, descending=False)
        cumulative_entropy = torch.cumsum(sorted_token_entropy, dim=-1)

        # `sorted_token_entropy` is the running maximum entropy (ascending order), so the left-hand side bounds the
        # joint mutual information of the accepted tokens.
        sorted_accepted = cumulative_entropy - sorted_token_entropy <= entropy_bound
        accepted_index = torch.scatter(
            input=torch.zeros_like(sorted_accepted), dim=-1, index=sorted_indices, src=sorted_accepted
        )

        random_tokens = torch.randint(
            low=0, high=model_output.shape[-1], size=sample.shape, device=sample.device, generator=generator
        )
        prev_sample = torch.where(accepted_index, sampled_tokens, random_tokens)

        self._step_index += 1

        if not return_dict:
            return prev_sample, sampled_tokens, sampled_probs, scaled_logits, accepted_index, None
        return DiscreteSchedulerOutput(
            prev_sample=prev_sample,
            pred_original_sample=sampled_tokens,
            sampled_probs=sampled_probs,
            # Self-conditioning wants the distribution the candidates were actually drawn from.
            pred_logits=scaled_logits,
            committed_mask=accepted_index,
        )


__all__ = ["EntropyBoundScheduler"]
