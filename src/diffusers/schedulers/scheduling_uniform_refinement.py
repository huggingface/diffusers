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


class UniformRefinementScheduler(SchedulerMixin, ConfigMixin):
    """
    Commit-by-confidence scheduler for the uniform corruption process.

    There is no mask token: every position always holds a real token, so which positions are still undecided cannot be
    read off the sequence and is tracked as scheduler state instead. At each step the scheduler samples a candidate per
    position, commits the most confident undecided ones until the step's cumulative quota is met (plus any whose
    confidence exceeds `threshold`), and renoises everything still undecided with uniformly random tokens.

    Optionally supports editing: once a position is committed it can still be overwritten if the model predicts a
    different token with confidence above a positive `editing_threshold` (`None`, `0.0`, or negative disables editing).

    Because the undecided set is state, the scheduler must be reset between blocks by calling
    [`~UniformRefinementScheduler.set_timesteps`] at the start of each one. Denoising past `num_inference_steps` raises
    rather than silently over-committing.

    Args:
        num_inference_steps (`int`, defaults to 32):
            The number of denoising steps the commit quota is spread across.
        temperature (`float`, defaults to 0.0):
            Sampling temperature applied to the logits when drawing candidates. `0.0` takes the argmax. The confidence
            driving the quota is always measured on the unscaled distribution, so `threshold` and `editing_threshold`
            do not move with this value.
        threshold (`float`, defaults to 0.95):
            Confidence above which an undecided position commits even if the quota is already met.
        editing_threshold (`float`, *optional*):
            Confidence above which an already-committed position is overwritten with a different predicted token. Must
            be positive to enable editing; `None`, `0.0`, or negative disables it.
    """

    order = 1

    @register_to_config
    def __init__(
        self,
        num_inference_steps: int = 32,
        temperature: float = 0.0,
        threshold: float = 0.95,
        editing_threshold: float | None = None,
    ):
        if threshold < 0.0:
            raise ValueError(f"`threshold` must be >= 0 (use > 1 to commit on quota alone), got {threshold}.")
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
        Set the discrete timestep grid, as the decreasing corruption level `t` in `[0, 1]`, and clear the committed
        state so the next block starts undecided.

        The grid matches [`~DiscreteDDIMScheduler.set_timesteps`] — `1.0` down to `1 / num_inference_steps` — so the
        two schedulers are interchangeable in a pipeline loop. `timesteps` is the public loop variable; the commit
        quota is derived from the integer `step_index` so it stays exact for any `num_inference_steps`.

        Args:
            num_inference_steps (`int`):
                The number of denoising steps.
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
        # Committed positions, tracked because the uniform process leaves no mark on the sequence itself.
        self._committed: torch.BoolTensor | None = None

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
        Commit the most confident undecided positions and renoise the rest.

        Args:
            model_output (`torch.Tensor` of shape `(batch_size, block_length, vocab_size)`):
                Raw logits from the model for the current block.
            timestep (`float` or `torch.Tensor`):
                The current corruption level, one entry of [`~UniformRefinementScheduler.timesteps`].
            sample (`torch.LongTensor` of shape `(batch_size, block_length)`):
                Current block token IDs.
            generator (`torch.Generator`, *optional*):
                RNG for sampling.
            return_dict (`bool`):
                Whether to return a [`DiscreteSchedulerOutput`] or a plain tuple.
        """
        if self.step_index is None:
            self._init_step_index(timestep)
        if self.step_index >= self.num_inference_steps:
            raise ValueError(
                f"`step` was called {self.step_index + 1} times for a schedule of {self.num_inference_steps} steps. "
                "This scheduler carries per-block committed state, so `set_timesteps` must be called again at the "
                "start of each block."
            )

        threshold = float(self.config.threshold)
        editing_threshold = self.config.editing_threshold

        sampled_tokens, sampled_probs = self._sample_from_logits(
            model_output, temperature=float(self.config.temperature), generator=generator
        )

        batch_size, block_length = sample.shape
        if self._committed is None:
            # First step of this block: `set_timesteps` cleared the state, so every position is undecided.
            self._committed = torch.zeros_like(sample, dtype=torch.bool)
        elif self._committed.shape != sample.shape:
            raise ValueError(
                f"`sample` changed shape mid-schedule, from {tuple(self._committed.shape)} to "
                f"{tuple(sample.shape)}. Call `set_timesteps` to start a new block."
            )
        committed = self._committed
        confidence = sampled_probs.to(dtype=torch.float32)

        # Cumulative quota: spread the block evenly across the steps and commit whatever is still owed. Integer
        # arithmetic off `step_index`, so the boundary is exact for any `(block_length, num_inference_steps)`.
        steps_done = self.step_index + 1
        target = (steps_done * block_length + self.num_inference_steps - 1) // self.num_inference_steps
        needed = (target - committed.sum(dim=-1)).clamp(min=0)

        masked_confidence = confidence.masked_fill(committed, float("-inf"))
        ranks = masked_confidence.argsort(dim=-1, descending=True).argsort(dim=-1)
        committed_mask = ~committed & ((ranks < needed[:, None]) | (confidence > threshold))

        edited_mask = torch.zeros_like(committed_mask)
        if editing_threshold is not None and editing_threshold > 0.0:
            edited_mask = committed & (sampled_tokens != sample) & (confidence > float(editing_threshold))

        prev_sample = torch.where(committed_mask | edited_mask, sampled_tokens, sample)
        self._committed = committed | committed_mask
        random_tokens = torch.randint(
            low=0, high=model_output.shape[-1], size=sample.shape, device=sample.device, generator=generator
        )
        prev_sample = torch.where(self._committed, prev_sample, random_tokens)

        self._step_index += 1

        if not return_dict:
            return prev_sample, sampled_tokens, sampled_probs, model_output, committed_mask, edited_mask
        return DiscreteSchedulerOutput(
            prev_sample=prev_sample,
            pred_original_sample=sampled_tokens,
            sampled_probs=sampled_probs,
            pred_logits=model_output,
            committed_mask=committed_mask,
            edited_mask=edited_mask,
        )


__all__ = ["UniformRefinementScheduler"]
