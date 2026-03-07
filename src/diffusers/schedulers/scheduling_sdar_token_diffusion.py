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

import torch
import torch.nn.functional as F

from ..configuration_utils import ConfigMixin, register_to_config
from ..utils import BaseOutput
from .scheduling_utils import SchedulerMixin


@dataclass
class SDARTokenDiffusionSchedulerOutput(BaseOutput):
    """
    Output class for SDAR-style block diffusion scheduling.

    Args:
        prev_sample (`torch.LongTensor` of shape `(batch_size, block_length)`):
            Updated block tokens after the current denoising step.
        transfer_index (`torch.BoolTensor` of shape `(batch_size, block_length)`):
            Boolean mask indicating which tokens were updated.
        sampled_tokens (`torch.LongTensor` of shape `(batch_size, block_length)`):
            Sampled token IDs from the model logits.
        sampled_probs (`torch.Tensor` of shape `(batch_size, block_length)`):
            Probabilities of the sampled tokens.
    """

    prev_sample: torch.LongTensor
    transfer_index: torch.BoolTensor
    sampled_tokens: torch.LongTensor
    sampled_probs: torch.Tensor


class SDARTokenDiffusionScheduler(SchedulerMixin, ConfigMixin):
    """
    Scheduler for SDAR-style block diffusion decoding.
    """

    order = 1

    @register_to_config
    def __init__(
        self,
        block_length: int = 4,
        denoising_steps: int = 4,
        remasking_strategy: str = "low_confidence_dynamic",
        confidence_threshold: float = 0.9,
        entropy_threshold: float = 0.35,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
    ):
        self.num_inference_steps = int(denoising_steps)
        self.timesteps = torch.arange(self.num_inference_steps - 1, -1, -1, dtype=torch.long)

    def set_timesteps(self, num_inference_steps: int, device: Optional[Union[str, torch.device]] = None) -> None:
        if num_inference_steps <= 0:
            raise ValueError(f"`num_inference_steps` must be > 0, got {num_inference_steps}.")
        self.num_inference_steps = int(num_inference_steps)
        self.timesteps = torch.arange(self.num_inference_steps - 1, -1, -1, device=device, dtype=torch.long)

    def get_num_transfer_tokens(self, block_length: int, num_inference_steps: int) -> torch.LongTensor:
        if num_inference_steps <= 0:
            raise ValueError(f"`num_inference_steps` must be > 0, got {num_inference_steps}.")
        base = int(block_length) // int(num_inference_steps)
        remainder = int(block_length) % int(num_inference_steps)
        num_transfer_tokens = torch.zeros(int(num_inference_steps), dtype=torch.long)
        num_transfer_tokens += base
        if remainder > 0:
            num_transfer_tokens[:remainder] += 1
        return num_transfer_tokens

    def _top_k_logits(self, logits: torch.Tensor, k: int) -> torch.Tensor:
        if k <= 0:
            return logits
        values, _ = torch.topk(logits, k)
        min_values = values[..., -1, None]
        return torch.where(logits < min_values, torch.full_like(logits, float("-inf")), logits)

    def _top_p_logits(self, logits: torch.Tensor, p: float) -> torch.Tensor:
        if p >= 1.0:
            return logits
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_mask = cumulative_probs > p
        sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
        sorted_mask[..., 0] = False
        mask_indices = torch.scatter(torch.zeros_like(logits, dtype=torch.bool), -1, sorted_indices, sorted_mask)
        return logits.masked_fill(mask_indices, float("-inf"))

    def sample(
        self,
        logits: torch.Tensor,
        *,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.LongTensor, torch.Tensor]:
        if temperature is None:
            temperature = float(self.config.temperature)
        if top_k is None:
            top_k = int(self.config.top_k)
        if top_p is None:
            top_p = float(self.config.top_p)

        orig_shape = logits.shape[:-1]
        vocab_size = logits.shape[-1]
        flat = logits.view(-1, vocab_size)

        if temperature < 1e-5:
            probs = F.softmax(flat, dim=-1)
            tokens = torch.argmax(flat, dim=-1, keepdim=True)
            token_probs = torch.gather(probs, -1, tokens)
            return tokens.view(*orig_shape), token_probs.view(*orig_shape)

        flat = flat / float(temperature)
        flat = self._top_k_logits(flat, int(top_k))
        flat = self._top_p_logits(flat, float(top_p))
        probs = F.softmax(flat, dim=-1)
        tokens = torch.multinomial(probs, num_samples=1, generator=generator)
        token_probs = torch.gather(probs, -1, tokens)
        return tokens.view(*orig_shape), token_probs.view(*orig_shape)

    def step(
        self,
        model_output: torch.Tensor,
        timestep: Union[int, torch.Tensor],
        sample: torch.LongTensor,
        *,
        mask_token_id: int,
        num_transfer_tokens: torch.LongTensor,
        remasking_strategy: Optional[str] = None,
        confidence_threshold: Optional[float] = None,
        entropy_threshold: Optional[float] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ) -> Union[
        SDARTokenDiffusionSchedulerOutput, Tuple[torch.LongTensor, torch.BoolTensor, torch.LongTensor, torch.Tensor]
    ]:
        if remasking_strategy is None:
            remasking_strategy = str(self.config.remasking_strategy)
        if confidence_threshold is None:
            confidence_threshold = float(self.config.confidence_threshold)
        if entropy_threshold is None:
            entropy_threshold = float(self.config.entropy_threshold)

        sampled_tokens, sampled_probs = self.sample(
            model_output, temperature=temperature, top_k=top_k, top_p=top_p, generator=generator
        )
        mask_index = sample == int(mask_token_id)
        transfer_index = torch.zeros_like(mask_index)

        if isinstance(timestep, torch.Tensor):
            step_index = int(timestep.item())
        else:
            step_index = int(timestep)

        if step_index >= int(num_transfer_tokens.numel()):
            step_index = int(num_transfer_tokens.numel()) - 1
        step_transfer = int(num_transfer_tokens[step_index].item())

        if remasking_strategy == "sequential":
            for j in range(sample.shape[0]):
                if not mask_index[j].any():
                    continue
                num_masked = int(mask_index[j].sum().item())
                k = min(step_transfer, num_masked)
                first_mask_index = mask_index[j].nonzero(as_tuple=True)[0].min().item()
                transfer_index[j, first_mask_index : first_mask_index + k] = True

        elif remasking_strategy in {"low_confidence_static", "low_confidence_dynamic"}:
            confidence = torch.where(mask_index, sampled_probs, torch.full_like(sampled_probs, float("-inf")))
            for j in range(confidence.shape[0]):
                if not mask_index[j].any():
                    continue
                num_masked = int(mask_index[j].sum().item())
                k = min(step_transfer, num_masked)
                if remasking_strategy == "low_confidence_dynamic":
                    high_conf_mask = confidence[j] > confidence_threshold
                    if int(high_conf_mask.sum().item()) >= k:
                        transfer_index[j] = high_conf_mask
                        continue
                _, idx = torch.topk(confidence[j], k)
                transfer_index[j, idx] = True

        elif remasking_strategy == "entropy_bounded":
            eps = 1e-12
            entropies = -(sampled_probs.clamp_min(eps) * sampled_probs.clamp_min(eps).log()).sum(dim=-1)
            entropies = torch.where(mask_index, entropies, torch.full_like(sampled_probs, float("inf")))
            ent_sorted, order = torch.sort(entropies, dim=1, descending=False)
            cumsum = torch.cumsum(ent_sorted, dim=1)
            for j in range(sampled_probs.shape[0]):
                if not mask_index[j].any():
                    continue
                threshold_tensor = torch.tensor(entropy_threshold, device=sampled_probs.device)
                k = int(torch.searchsorted(cumsum[j], threshold_tensor, right=False).item())
                num_masked = int(mask_index[j].sum().item())
                k = max(1, min(k, num_masked))
                selected_token_indices = order[j, :k]
                transfer_index[j, selected_token_indices] = True

        else:
            raise ValueError(f"Unknown remasking strategy: {remasking_strategy}")

        prev_sample = sample.clone()
        prev_sample[transfer_index] = sampled_tokens[transfer_index]

        if not return_dict:
            return prev_sample, transfer_index, sampled_tokens, sampled_probs
        return SDARTokenDiffusionSchedulerOutput(
            prev_sample=prev_sample,
            transfer_index=transfer_index,
            sampled_tokens=sampled_tokens,
            sampled_probs=sampled_probs,
        )


__all__ = ["SDARTokenDiffusionScheduler", "SDARTokenDiffusionSchedulerOutput"]
