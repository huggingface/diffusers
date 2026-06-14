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
class EntropyBoundSchedulerOutput(BaseOutput):
    """
    Output class for the entropy bound scheduler.

    Args:
        prev_sample (`torch.LongTensor` of shape `(batch_size, block_length)`):
            Updated block tokens after the current denoising step.
        accepted_index (`torch.BoolTensor` of shape `(batch_size, block_length)`):
            Boolean mask of the positions accepted (committed) in this step.
        sampled_tokens (`torch.LongTensor` of shape `(batch_size, block_length)`):
            Token IDs sampled from the model logits.
        sampled_probs (`torch.Tensor` of shape `(batch_size, block_length)`):
            Probabilities of the sampled tokens.
    """

    prev_sample: torch.LongTensor
    accepted_index: torch.BoolTensor
    sampled_tokens: torch.LongTensor
    sampled_probs: torch.Tensor


class EntropyBoundScheduler(SchedulerMixin, ConfigMixin):
    """
    Entropy bound scheduler for the uniform corruption process.

    At each step the scheduler samples a candidate token per position and accepts the `k` lowest-entropy positions
    such that `sum_i^k entropy_i - max(entropy_1, ..., entropy_k) <= entropy_bound`. The left-hand side upper-bounds
    the joint mutual information between the accepted tokens, so they are approximately independent. Accepted positions
    keep their sampled token; the rest are renoised with uniformly random tokens (there is no mask token).

    Proposed in "Beyond Next-Token Prediction" (https://huggingface.co/papers/2505.24857).

    Args:
        entropy_bound (`float`, defaults to 0.1):
            The maximum tolerated joint entropy of the accepted tokens. Larger values accept more tokens per step.
        num_inference_steps (`int`, defaults to 32):
            The maximum number of denoising steps.
    """

    order = 1

    @register_to_config
    def __init__(self, entropy_bound: float = 0.1, num_inference_steps: int = 32):
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
        entropy_bound: float | None = None,
        temperature: float = 1.0,
        generator: torch.Generator | None = None,
        return_dict: bool = True,
    ) -> EntropyBoundSchedulerOutput | tuple[torch.LongTensor, torch.BoolTensor, torch.LongTensor, torch.Tensor]:
        """
        Accept the lowest-entropy positions under the entropy bound and renoise the rest.

        Args:
            model_output (`torch.Tensor` of shape `(batch_size, block_length, vocab_size)`):
                Raw logits from the model for the current block.
            timestep (`int` or `torch.Tensor`):
                Current step index within the denoising schedule. Unused; kept for API consistency.
            sample (`torch.LongTensor` of shape `(batch_size, block_length)`):
                Current block token IDs.
            entropy_bound (`float`, *optional*):
                Overrides the configured entropy bound for this step.
            temperature (`float`):
                Sampling temperature applied to the logits when drawing the candidate tokens.
            generator (`torch.Generator`, *optional*):
                RNG for sampling.
            return_dict (`bool`):
                Whether to return an [`EntropyBoundSchedulerOutput`] or a plain tuple.
        """
        if entropy_bound is None:
            entropy_bound = float(self.config.entropy_bound)

        sampled_tokens, sampled_probs = self._sample_from_logits(
            model_output, temperature=temperature, generator=generator
        )

        token_entropy = torch.distributions.Categorical(logits=model_output).entropy()  # (batch, block_length)
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

        if not return_dict:
            return prev_sample, accepted_index, sampled_tokens, sampled_probs
        return EntropyBoundSchedulerOutput(
            prev_sample=prev_sample,
            accepted_index=accepted_index,
            sampled_tokens=sampled_tokens,
            sampled_probs=sampled_probs,
        )


__all__ = ["EntropyBoundScheduler", "EntropyBoundSchedulerOutput"]
