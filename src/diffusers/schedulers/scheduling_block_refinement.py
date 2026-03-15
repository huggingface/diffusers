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

from ..configuration_utils import ConfigMixin, register_to_config
from ..utils import BaseOutput
from .scheduling_utils import SchedulerMixin


@dataclass
class BlockRefinementSchedulerOutput(BaseOutput):
    """
    Output class for block refinement scheduling.

    Args:
        prev_sample (`torch.LongTensor` of shape `(batch_size, block_length)`):
            Updated block tokens after the current refinement step.
        transfer_index (`torch.BoolTensor` of shape `(batch_size, block_length)`):
            Boolean mask indicating which tokens were committed (mask-filling).
        editing_transfer_index (`torch.BoolTensor` of shape `(batch_size, block_length)`):
            Boolean mask indicating which tokens were edited (non-mask replacement).
        sampled_tokens (`torch.LongTensor` of shape `(batch_size, block_length)`):
            Sampled token IDs from the model logits.
        sampled_probs (`torch.Tensor` of shape `(batch_size, block_length)`):
            Probabilities of the sampled tokens.
    """

    prev_sample: torch.LongTensor
    transfer_index: torch.BoolTensor
    editing_transfer_index: torch.BoolTensor
    sampled_tokens: torch.LongTensor
    sampled_probs: torch.Tensor


class BlockRefinementScheduler(SchedulerMixin, ConfigMixin):
    """
    Scheduler for block-wise iterative refinement (commit-by-confidence).

    At each step, the scheduler samples candidate tokens and commits those with the highest confidence. The number of
    tokens to commit per step is determined by evenly distributing the block length across the number of refinement
    steps.

    Optionally supports editing: after all mask tokens are resolved, tokens can be replaced if the model predicts a
    different token with confidence above `editing_threshold`.
    """

    order = 1

    @register_to_config
    def __init__(
        self,
        block_length: int = 32,
        num_inference_steps: int = 32,
        threshold: float = 0.95,
        editing_threshold: Optional[float] = None,
        minimal_topk: int = 1,
    ):
        self.num_inference_steps = int(num_inference_steps)
        self.timesteps = torch.arange(self.num_inference_steps - 1, -1, -1, dtype=torch.long)
        self._transfer_schedule: Optional[torch.LongTensor] = None

    def set_timesteps(self, num_inference_steps: int, device: Optional[Union[str, torch.device]] = None) -> None:
        if num_inference_steps <= 0:
            raise ValueError(f"`num_inference_steps` must be > 0, got {num_inference_steps}.")
        self.num_inference_steps = int(num_inference_steps)
        self.timesteps = torch.arange(self.num_inference_steps - 1, -1, -1, device=device, dtype=torch.long)
        self._transfer_schedule = self.get_num_transfer_tokens(
            int(self.config.block_length), self.num_inference_steps
        ).to(device=device if device is not None else "cpu")

    def get_num_transfer_tokens(self, block_length: int, num_inference_steps: int) -> torch.LongTensor:
        """Evenly distribute `block_length` token commits across `num_inference_steps` steps."""
        if num_inference_steps <= 0:
            return torch.zeros((0,), dtype=torch.long)
        base = int(block_length) // int(num_inference_steps)
        remainder = int(block_length) % int(num_inference_steps)
        out = torch.full((int(num_inference_steps),), base, dtype=torch.long)
        out[:remainder] += 1
        return out

    def step(
        self,
        sampled_tokens: torch.LongTensor,
        sampled_probs: torch.Tensor,
        timestep: Union[int, torch.Tensor],
        sample: torch.LongTensor,
        *,
        mask_token_id: int,
        threshold: Optional[float] = None,
        editing_threshold: Optional[float] = None,
        minimal_topk: Optional[int] = None,
        prompt_mask: Optional[torch.BoolTensor] = None,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ) -> Union[
        BlockRefinementSchedulerOutput,
        Tuple[torch.LongTensor, torch.BoolTensor, torch.BoolTensor, torch.LongTensor, torch.Tensor],
    ]:
        """
        Perform a single refinement step: commit confident tokens and optionally edit existing ones.

        Args:
            sampled_tokens (`torch.LongTensor` of shape `(batch_size, block_length)`):
                Candidate token IDs sampled from model logits.
            sampled_probs (`torch.Tensor` of shape `(batch_size, block_length)`):
                Confidence probabilities for the sampled tokens.
            timestep (`int` or `torch.Tensor`):
                Current step index within the block's refinement schedule.
            sample (`torch.LongTensor` of shape `(batch_size, block_length)`):
                Current block token IDs (contains mask tokens for uncommitted positions).
            mask_token_id (`int`):
                Token ID used for masked positions.
            threshold (`float`, *optional*):
                Confidence threshold for committing tokens. Defaults to config value.
            editing_threshold (`float`, *optional*):
                Confidence threshold for editing non-mask tokens. Defaults to config value.
            minimal_topk (`int`, *optional*):
                Minimum tokens to commit per step. Defaults to config value.
            prompt_mask (`torch.BoolTensor`, *optional*):
                Boolean mask of shape `(block_length,)` where `True` marks prompt (non-editable) positions.
            generator (`torch.Generator`, *optional*):
                Unused, kept for API consistency.
            return_dict (`bool`):
                Whether to return a `BlockRefinementSchedulerOutput` or a tuple.
        """
        if threshold is None:
            threshold = float(self.config.threshold)
        if editing_threshold is None:
            editing_threshold = self.config.editing_threshold
        if minimal_topk is None:
            minimal_topk = int(self.config.minimal_topk)

        batch_size, block_length = sample.shape
        active_block = sample == int(mask_token_id)
        masks_remaining = active_block.any()

        if isinstance(timestep, torch.Tensor):
            step_index = int(timestep.item())
        else:
            step_index = int(timestep)

        # --- Mask-filling transfer ---
        transfer_index = torch.zeros_like(sampled_tokens, dtype=torch.bool)
        if masks_remaining and self._transfer_schedule is not None:
            clamped_step = min(step_index, len(self._transfer_schedule) - 1)
            num_to_transfer = int(self._transfer_schedule[clamped_step].item())

            confidence = torch.where(
                active_block,
                sampled_probs.to(dtype=torch.float32),
                torch.full_like(sampled_probs, -torch.inf, dtype=torch.float32),
            )

            for b in range(batch_size):
                high_conf = confidence[b] > float(threshold)
                if high_conf.sum().item() >= num_to_transfer:
                    transfer_index[b] = high_conf
                else:
                    k = min(num_to_transfer, int(active_block[b].sum().item()))
                    if k > 0:
                        _, idx = torch.topk(confidence[b], k=k)
                        transfer_index[b, idx] = True

        # --- Editing transfer (non-mask, non-prompt positions) ---
        editing_enabled = editing_threshold is not None and editing_threshold >= 0.0
        editing_transfer_index = torch.zeros_like(sampled_tokens, dtype=torch.bool)
        if editing_enabled:
            if prompt_mask is None:
                prompt_mask = torch.zeros(block_length, device=sample.device, dtype=torch.bool)
            editable = (~active_block) & (~prompt_mask.unsqueeze(0))
            editing_conf = torch.where(
                editable,
                sampled_probs.to(dtype=torch.float32),
                torch.full_like(sampled_probs, -torch.inf, dtype=torch.float32),
            )
            high_conf_edit = editing_conf > float(editing_threshold)
            token_changed = sampled_tokens != sample
            editing_transfer_index = high_conf_edit & token_changed & editable

        # Apply transfers
        final_transfer = transfer_index | editing_transfer_index
        prev_sample = sample.clone()
        if final_transfer.any():
            prev_sample[final_transfer] = sampled_tokens[final_transfer]

        if not return_dict:
            return prev_sample, transfer_index, editing_transfer_index, sampled_tokens, sampled_probs
        return BlockRefinementSchedulerOutput(
            prev_sample=prev_sample,
            transfer_index=transfer_index,
            editing_transfer_index=editing_transfer_index,
            sampled_tokens=sampled_tokens,
            sampled_probs=sampled_probs,
        )


__all__ = ["BlockRefinementScheduler", "BlockRefinementSchedulerOutput"]
