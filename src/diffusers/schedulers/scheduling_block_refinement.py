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

    At each step, the scheduler samples candidate tokens from model logits and commits those with the highest
    confidence. The number of tokens to commit per step is determined by evenly distributing the block length across
    the number of refinement steps.

    Optionally supports editing: after all mask tokens are resolved, tokens can be replaced if the model predicts a
    different token with confidence above a positive `editing_threshold` (`None`, `0.0`, or negative disables editing).
    """

    order = 1

    @register_to_config
    def __init__(
        self,
        block_length: int = 32,
        num_inference_steps: int = 32,
        threshold: float = 0.95,
        editing_threshold: float | None = None,
        minimal_topk: int = 1,
    ):
        self.num_inference_steps = num_inference_steps
        self.timesteps = torch.arange(self.num_inference_steps - 1, -1, -1, dtype=torch.long)
        self._transfer_schedule: torch.LongTensor | None = None

    def set_timesteps(
        self,
        num_inference_steps: int,
        device: str | torch.device | None = None,
        block_length: int | None = None,
    ) -> None:
        if num_inference_steps <= 0:
            raise ValueError(f"`num_inference_steps` must be > 0, got {num_inference_steps}.")
        if block_length is None:
            block_length = self.config.block_length
        elif block_length <= 0:
            raise ValueError(f"`block_length` must be > 0, got {block_length}.")
        self.num_inference_steps = num_inference_steps
        self.timesteps = torch.arange(self.num_inference_steps - 1, -1, -1, device=device, dtype=torch.long)
        self._transfer_schedule = self.get_num_transfer_tokens(block_length, self.num_inference_steps).to(
            device=device if device is not None else "cpu"
        )

    def get_num_transfer_tokens(self, block_length: int, num_inference_steps: int) -> torch.LongTensor:
        """Evenly distribute `block_length` token commits across `num_inference_steps` steps."""
        if num_inference_steps <= 0:
            return torch.zeros((0,), dtype=torch.long)
        base = block_length // num_inference_steps
        remainder = block_length % num_inference_steps
        out = torch.full((num_inference_steps,), base, dtype=torch.long)
        out[:remainder] += 1
        return out

    # --- SAR sampling utilities ---

    @staticmethod
    def _top_p_filtering(logits: torch.Tensor, top_p: float | None) -> torch.Tensor:
        """Nucleus (top-p) logit filtering."""
        if top_p is None or top_p >= 1.0:
            return logits
        if not (0.0 < top_p <= 1.0):
            raise ValueError(f"`top_p` must be in (0, 1], got {top_p}.")

        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = sorted_probs.cumsum(dim=-1)

        sorted_indices_to_remove = cumulative_probs > float(top_p)
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        sorted_logits = sorted_logits.masked_fill(sorted_indices_to_remove, torch.finfo(sorted_logits.dtype).min)
        filtered = logits.scatter(-1, sorted_indices, sorted_logits)
        return filtered

    @staticmethod
    def _top_k_filtering(logits: torch.Tensor, top_k: int | None) -> torch.Tensor:
        """Top-k logit filtering."""
        if top_k is None or top_k <= 0:
            return logits
        if top_k >= logits.shape[-1]:
            return logits
        values, _ = torch.topk(logits, k=top_k, dim=-1)
        min_keep = values[..., -1, None]
        return logits.masked_fill(logits < min_keep, torch.finfo(logits.dtype).min)

    @staticmethod
    def _sample_from_logits(
        logits: torch.Tensor,
        *,
        temperature: float,
        top_k: int | None,
        top_p: float | None,
        generator: torch.Generator | None,
        use_multinomial: bool,
    ) -> tuple[torch.LongTensor, torch.Tensor]:
        """Sample tokens from logits with temperature scaling, top-k, and top-p."""
        if temperature < 0:
            raise ValueError(f"`temperature` must be >= 0, got {temperature}.")

        vocab_size = logits.shape[-1]
        flat_logits = logits.reshape(-1, vocab_size)

        if temperature == 0.0 or not use_multinomial:
            probs = torch.softmax(flat_logits.float(), dim=-1)
            token = flat_logits.argmax(dim=-1, keepdim=True)
            token_prob = torch.gather(probs, -1, token)
            return token.view(*logits.shape[:-1]), token_prob.view(*logits.shape[:-1])

        scaled = flat_logits
        if temperature != 1.0:
            scaled = flat_logits / temperature

        filtered = BlockRefinementScheduler._top_k_filtering(scaled, top_k=top_k)
        filtered = BlockRefinementScheduler._top_p_filtering(filtered, top_p=top_p)

        probs = torch.softmax(filtered.float(), dim=-1)
        token = torch.multinomial(probs, num_samples=1, generator=generator)
        token_prob = torch.gather(probs, -1, token)

        return token.view(*logits.shape[:-1]), token_prob.view(*logits.shape[:-1])

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int | torch.Tensor,
        sample: torch.LongTensor,
        *,
        mask_token_id: int,
        temperature: float = 0.0,
        top_p: float | None = None,
        top_k: int | None = None,
        sampling_method: str = "auto",
        threshold: float | None = None,
        editing_threshold: float | None = None,
        minimal_topk: int | None = None,
        prompt_mask: torch.BoolTensor | None = None,
        generator: torch.Generator | None = None,
        return_dict: bool = True,
    ) -> (
        BlockRefinementSchedulerOutput
        | tuple[torch.LongTensor, torch.BoolTensor, torch.BoolTensor, torch.LongTensor, torch.Tensor]
    ):
        """
        Perform a single refinement step: sample from logits, commit confident tokens, and optionally edit existing
        ones.

        Args:
            model_output (`torch.Tensor` of shape `(batch_size, block_length, vocab_size)`):
                Raw logits from the model for the current block.
            timestep (`int` or `torch.Tensor`):
                Current step index within the block's refinement schedule.
            sample (`torch.LongTensor` of shape `(batch_size, block_length)`):
                Current block token IDs (contains mask tokens for uncommitted positions).
            mask_token_id (`int`):
                Token ID used for masked positions.
            temperature (`float`):
                Sampling temperature.
            top_p (`float`, *optional*):
                Nucleus sampling cutoff.
            top_k (`int`, *optional*):
                Top-k sampling cutoff.
            sampling_method (`str`):
                Sampling method (`auto`, `greedy`, `multinomial`).
            threshold (`float`, *optional*):
                Confidence threshold for committing tokens. Defaults to config value.
            editing_threshold (`float`, *optional*):
                Confidence threshold for editing non-mask tokens; must be positive to enable editing. Defaults to
                config value.
            minimal_topk (`int`, *optional*):
                Minimum tokens to commit per step. Defaults to config value.
            prompt_mask (`torch.BoolTensor`, *optional*):
                Boolean mask of shape `(block_length,)` where `True` marks prompt (non-editable) positions.
            generator (`torch.Generator`, *optional*):
                RNG for sampling.
            return_dict (`bool`):
                Whether to return a `BlockRefinementSchedulerOutput` or a tuple.
        """
        if threshold is None:
            threshold = float(self.config.threshold)
        if editing_threshold is None:
            editing_threshold = self.config.editing_threshold
        if minimal_topk is None:
            minimal_topk = self.config.minimal_topk

        # Sample from logits
        use_multinomial = sampling_method == "multinomial" or (sampling_method == "auto" and temperature != 0.0)
        sampled_tokens, sampled_probs = self._sample_from_logits(
            model_output,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            generator=generator,
            use_multinomial=use_multinomial,
        )

        batch_size, block_length = sample.shape
        active_block = sample == mask_token_id
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
                high_conf = confidence[b] > threshold
                if high_conf.sum().item() >= num_to_transfer:
                    transfer_index[b] = high_conf
                else:
                    k = min(num_to_transfer, int(active_block[b].sum().item()))
                    if k > 0:
                        _, idx = torch.topk(confidence[b], k=k)
                        transfer_index[b, idx] = True

        # --- Editing transfer (non-mask, non-prompt positions) ---
        editing_enabled = editing_threshold is not None and editing_threshold > 0.0
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

    @staticmethod
    def check_eos_finished(
        cur_x: torch.LongTensor,
        sampled_tokens: torch.LongTensor,
        final_transfer: torch.BoolTensor,
        finished: torch.BoolTensor,
        eos_token_id: int,
        mask_token_id: int,
        prompt_length: int,
    ) -> torch.BoolTensor:
        """
        Update per-batch finished flags when EOS tokens are committed.

        Args:
            cur_x (`torch.LongTensor` of shape `(batch_size, seq_len)`):
                Current full sequence including all blocks up to the current window.
            sampled_tokens (`torch.LongTensor` of shape `(batch_size, block_length)`):
                Tokens sampled by the scheduler in this step.
            final_transfer (`torch.BoolTensor` of shape `(batch_size, block_length)`):
                Combined mask of committed and edited positions.
            finished (`torch.BoolTensor` of shape `(batch_size,)`):
                Current per-batch finished flags.
            eos_token_id (`int`):
                EOS token ID.
            mask_token_id (`int`):
                Mask token ID.
            prompt_length (`int`):
                Number of prompt tokens at the start of the sequence.

        Returns:
            `torch.BoolTensor`: Updated finished flags.
        """
        batch_size = cur_x.shape[0]
        for b in range(batch_size):
            if finished[b]:
                continue
            eos_in_commits = (sampled_tokens[b][final_transfer[b]] == eos_token_id).any().item()
            if not eos_in_commits:
                continue
            eos_pos = (cur_x[b] == eos_token_id).nonzero(as_tuple=True)
            if len(eos_pos[0]) == 0:
                continue
            eos_pos = int(eos_pos[0][0].item())
            # The first generated token sits at index `prompt_length`; allow EOS there.
            if eos_pos < prompt_length:
                continue
            if (cur_x[b, prompt_length:eos_pos] != mask_token_id).all().item():
                finished[b] = True
        return finished

    def check_block_should_continue(
        self,
        step_idx: int,
        masks_remaining: bool,
        editing_enabled: bool,
        editing_transfer_index: torch.BoolTensor,
        post_steps: int,
        max_post_steps: int,
        finished: torch.BoolTensor,
    ) -> bool:
        """
        Determine whether the inner refinement loop should continue for the current block.

        Args:
            step_idx (`int`):
                Current refinement step index within this block.
            masks_remaining (`bool`):
                Whether any mask tokens remain in the block.
            editing_enabled (`bool`):
                Whether editing mode is active.
            editing_transfer_index (`torch.BoolTensor`):
                Which tokens were edited in this step.
            post_steps (`int`):
                Number of post-mask editing steps taken so far.
            max_post_steps (`int`):
                Maximum allowed post-mask editing steps.
            finished (`torch.BoolTensor`):
                Per-batch finished flags (from EOS detection).

        Returns:
            `bool`: `True` if refinement should continue, `False` to break.
        """
        if finished.all():
            return False
        if not masks_remaining and not editing_enabled:
            return False
        if not masks_remaining and not editing_transfer_index.any():
            return False
        if masks_remaining and step_idx >= self.num_inference_steps:
            return False
        if not masks_remaining and post_steps > max_post_steps:
            return False
        return True

    def add_noise(
        self,
        original_samples: torch.LongTensor,
        attention_mask: torch.LongTensor,
        *,
        prompt_length: int,
        block_length: int,
        mask_token_id: int,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.LongTensor, torch.LongTensor, torch.BoolTensor, torch.BoolTensor]:
        """
        Apply the forward (noising) process for semi-autoregressive block masking.

        For each block after the prompt, a random fraction of valid (non-padding) tokens are replaced with
        `mask_token_id`. Two complementary views are returned: `noisy` and `noisy_rev`, where the masked positions in
        one are the unmasked positions in the other.

        Args:
            original_samples (`torch.LongTensor` of shape `(batch_size, seq_len)`):
                Clean token IDs.
            attention_mask (`torch.LongTensor` of shape `(batch_size, seq_len)`):
                Padding mask (1 for valid, 0 for padding).
            prompt_length (`int`):
                Number of leading prompt tokens to keep unmasked.
            block_length (`int`):
                Block size for masking.
            mask_token_id (`int`):
                Token ID to use for masked positions.
            generator (`torch.Generator`, *optional*):
                RNG for reproducibility.

        Returns:
            `tuple[torch.LongTensor, torch.LongTensor, torch.BoolTensor, torch.BoolTensor]`:
                `(noisy, noisy_rev, masked, masked_rev)` — the two complementary noisy sequences and their
                corresponding boolean masks.
        """
        batch_size, seq_len = original_samples.shape
        device = original_samples.device

        noisy = original_samples.clone()
        noisy_rev = original_samples.clone()
        masked = torch.zeros_like(original_samples, dtype=torch.bool)
        masked_rev = torch.zeros_like(original_samples, dtype=torch.bool)

        valid = attention_mask.to(dtype=torch.bool)
        for block_start in range(prompt_length, seq_len, block_length):
            block_end = min(seq_len, block_start + block_length)
            seg_len = block_end - block_start
            if seg_len <= 0:
                continue

            p_mask = torch.rand((batch_size, 1), device=device, generator=generator)
            seg = torch.rand((batch_size, seg_len), device=device, generator=generator) < p_mask
            seg = seg & valid[:, block_start:block_end]
            seg_rev = (~seg) & valid[:, block_start:block_end]

            masked[:, block_start:block_end] = seg
            masked_rev[:, block_start:block_end] = seg_rev

        noisy = torch.where(masked, torch.full_like(noisy, mask_token_id), noisy)
        noisy_rev = torch.where(masked_rev, torch.full_like(noisy_rev, mask_token_id), noisy_rev)
        return noisy, noisy_rev, masked, masked_rev


__all__ = ["BlockRefinementScheduler", "BlockRefinementSchedulerOutput"]
