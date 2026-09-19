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

from dataclasses import dataclass
from typing import Optional, Union

import torch

from ..configuration_utils import ConfigMixin, register_to_config
from ..utils import deprecate
from .scheduling_utils import DiscreteSchedulerOutput, SchedulerMixin


@dataclass
class BlockRefinementSchedulerOutput(DiscreteSchedulerOutput):
    """
    Deprecated output class for [`BlockRefinementScheduler`], kept for one release.

    It is now a [`DiscreteSchedulerOutput`] with two deprecated aliases: `transfer_index` for `committed_mask` and
    `editing_transfer_index` for `edited_mask`. The aliases resolve on attribute access only — `BaseOutput.__getitem__`
    and iteration see dataclass fields, so `output["transfer_index"]` will not work.
    """

    @property
    def transfer_index(self) -> torch.BoolTensor:
        deprecate(
            "transfer_index",
            "1.0.0",
            "`transfer_index` is deprecated; use `committed_mask`, which every discrete scheduler now returns.",
        )
        return self.committed_mask

    @property
    def editing_transfer_index(self) -> torch.BoolTensor | None:
        deprecate(
            "editing_transfer_index",
            "1.0.0",
            "`editing_transfer_index` is deprecated; use `edited_mask`, which every discrete scheduler now returns.",
        )
        return self.edited_mask


class BlockRefinementScheduler(SchedulerMixin, ConfigMixin):
    """
    Scheduler for block-wise iterative refinement (commit-by-confidence).

    At each step, the scheduler samples candidate tokens from model logits and commits those with the highest
    confidence. The number of tokens to commit per step is determined by evenly distributing the block length across
    the number of refinement steps.

    Optionally supports editing: after all mask tokens are resolved, tokens can be replaced if the model predicts a
    different token with confidence above a positive `editing_threshold` (`None`, `0.0`, or negative disables editing).

    This scheduler models the absorbing (masked) corruption process. For the uniform process, where every position
    always holds a real token and there is no mask token, use [`UniformRefinementScheduler`].

    Args:
        block_length (`int`, defaults to 32):
            The block size this scheduler is configured for. Pipelines read it as their default block size; the commit
            quota itself is taken from the width of `sample`.
        num_inference_steps (`int`, defaults to 32):
            The number of refinement steps the commit quota is spread across.
        mask_token_id (`int`, *optional*):
            Token ID marking an undecided position. Required by [`~BlockRefinementScheduler.step`]; it lives in the
            config because it is a property of the tokenizer, matching [`AmusedScheduler`].
        temperature (`float`, defaults to 0.0):
            Sampling temperature applied to the logits when drawing candidates. `0.0` takes the argmax. The confidence
            driving the quota is measured on the unscaled distribution, so `threshold` and `editing_threshold` do not
            move with this value.
        top_p (`float`, *optional*):
            Nucleus sampling cutoff.
        top_k (`int`, *optional*):
            Top-k sampling cutoff.
        sampling_method (`str`, defaults to `"auto"`):
            One of `"auto"`, `"greedy"`, `"multinomial"`. `"auto"` draws multinomially when `temperature != 0`.
        threshold (`float`, defaults to 0.95):
            Confidence above which a masked position commits even if the quota is already met.
        editing_threshold (`float`, *optional*):
            Confidence above which an already-resolved position is overwritten with a different predicted token. Must
            be positive to enable editing; `None`, `0.0`, or negative disables it.
    """

    order = 1

    @register_to_config
    def __init__(
        self,
        block_length: int = 32,
        num_inference_steps: int = 32,
        mask_token_id: int | None = None,
        temperature: float = 0.0,
        top_p: float | None = None,
        top_k: int | None = None,
        sampling_method: str = "auto",
        threshold: float = 0.95,
        editing_threshold: float | None = None,
    ):
        if sampling_method not in {"auto", "greedy", "multinomial"}:
            raise ValueError(
                f"`sampling_method` must be one of {{'auto', 'greedy', 'multinomial'}}, got {sampling_method!r}."
            )
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

    def set_timesteps(self, num_inference_steps: int, device: str | torch.device | None = None, **kwargs) -> None:
        """
        Set the discrete timestep grid, as the decreasing corruption level `t` in `[0, 1]`.

        The grid matches [`~DiscreteDDIMScheduler.set_timesteps`] — `1.0` down to `1 / num_inference_steps` — so the
        discrete schedulers are interchangeable in a pipeline loop. `timesteps` is the public loop variable; the commit
        quota is derived from the integer `step_index` so it stays exact for any `num_inference_steps`.

        Args:
            num_inference_steps (`int`):
                The number of refinement steps.
            device (`str` or `torch.device`, *optional*):
                The device the timesteps should be moved to.
        """
        if kwargs.pop("block_length", None) is not None:
            deprecate(
                "block_length",
                "1.0.0",
                "Passing `block_length` to `set_timesteps` is deprecated and has no effect: the commit quota is now "
                "taken from the width of `sample` on each `step`.",
            )
        if kwargs:
            raise TypeError(f"Unexpected keyword arguments: {sorted(kwargs)}.")
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

    def get_num_transfer_tokens(self, block_length: int, num_inference_steps: int) -> torch.LongTensor:
        """
        Evenly distribute `block_length` token commits across `num_inference_steps` steps.

        Deprecated: the per-step quota is now computed inline in [`~BlockRefinementScheduler.step`] from `step_index`,
        so there is no schedule tensor to build.
        """
        deprecate(
            "get_num_transfer_tokens",
            "1.0.0",
            "`get_num_transfer_tokens` is deprecated; the per-step commit quota is computed inline in `step`.",
        )
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
        # Confidence is always read off the unmodified distribution, so `threshold` and `editing_threshold`
        # keep their meaning under any `temperature` / `top_k` / `top_p` (the `sampled_probs` contract of
        # `DiscreteSchedulerOutput`).
        probs = torch.softmax(flat_logits.float(), dim=-1)

        if temperature == 0.0 or not use_multinomial:
            token = flat_logits.argmax(dim=-1, keepdim=True)
            token_prob = torch.gather(probs, -1, token)
            return token.view(*logits.shape[:-1]), token_prob.view(*logits.shape[:-1])

        scaled = flat_logits
        if temperature != 1.0:
            scaled = flat_logits / temperature

        filtered = BlockRefinementScheduler._top_k_filtering(scaled, top_k=top_k)
        filtered = BlockRefinementScheduler._top_p_filtering(filtered, top_p=top_p)

        draw_probs = torch.softmax(filtered.float(), dim=-1)
        token = torch.multinomial(draw_probs, num_samples=1, generator=generator)
        token_prob = torch.gather(probs, -1, token)

        return token.view(*logits.shape[:-1]), token_prob.view(*logits.shape[:-1])

    def _pop_deprecated_step_kwargs(self, kwargs: dict) -> dict:
        """
        Accept the pre-1.0 per-call `step` arguments for one release, honouring them with a warning.

        They moved to the config (D6: the scheduler owns logit shaping), except `prompt_mask`, whose job moved to the
        pipeline — restoring frozen positions after the step, the way inpainting pipelines blend by mask.
        """
        moved_to_config = ("mask_token_id", "temperature", "top_p", "top_k", "sampling_method", "threshold")
        overrides = {}
        for name in moved_to_config:
            if name in kwargs:
                value = kwargs.pop(name)
                if value is not None:
                    deprecate(
                        name,
                        "1.0.0",
                        f"Passing `{name}` to `step` is deprecated; set it on the scheduler config instead, e.g. "
                        f"`BlockRefinementScheduler.from_config(scheduler.config, {name}=...)`.",
                    )
                    overrides[name] = value
        for name in ("editing_threshold", "prompt_mask"):
            if name in kwargs:
                value = kwargs.pop(name)
                if value is not None:
                    hint = (
                        "set it on the scheduler config instead"
                        if name == "editing_threshold"
                        else "restore frozen positions in the pipeline after `step` instead"
                    )
                    deprecate(name, "1.0.0", f"Passing `{name}` to `step` is deprecated; {hint}.")
                    overrides[name] = value
        if "minimal_topk" in kwargs:
            kwargs.pop("minimal_topk")
            deprecate(
                "minimal_topk",
                "1.0.0",
                "`minimal_topk` is deprecated and has no effect; it was read from the config but never used.",
            )
        if kwargs:
            raise TypeError(f"Unexpected keyword arguments: {sorted(kwargs)}.")
        return overrides

    def _draw(self, model_output: torch.Tensor, overrides: dict, generator: torch.Generator | None):
        """Sample candidate tokens using the configured shaping, with any deprecated per-call overrides applied."""
        temperature = float(overrides.get("temperature", self.config.temperature))
        sampling_method = overrides.get("sampling_method", self.config.sampling_method)
        use_multinomial = sampling_method == "multinomial" or (sampling_method == "auto" and temperature != 0.0)
        return self._sample_from_logits(
            model_output,
            temperature=temperature,
            top_k=overrides.get("top_k", self.config.top_k),
            top_p=overrides.get("top_p", self.config.top_p),
            generator=generator,
            use_multinomial=use_multinomial,
        )

    def _edited_mask(
        self,
        sampled_tokens: torch.LongTensor,
        sampled_probs: torch.Tensor,
        sample: torch.LongTensor,
        resolved: torch.BoolTensor,
        editing_threshold: float | None,
        prompt_mask: torch.BoolTensor | None,
    ) -> torch.BoolTensor:
        """Positions that overwrite an already-resolved token with a different, confident prediction."""
        edited = torch.zeros_like(sampled_tokens, dtype=torch.bool)
        if editing_threshold is None or editing_threshold <= 0.0:
            return edited
        editable = resolved
        if prompt_mask is not None:
            editable = editable & (~prompt_mask.unsqueeze(0))
        editing_conf = torch.where(
            editable,
            sampled_probs.to(dtype=torch.float32),
            torch.full_like(sampled_probs, -torch.inf, dtype=torch.float32),
        )
        return (editing_conf > float(editing_threshold)) & (sampled_tokens != sample) & editable

    def step(
        self,
        model_output: torch.Tensor,
        timestep: float | torch.Tensor,
        sample: torch.LongTensor,
        *,
        generator: torch.Generator | None = None,
        return_dict: bool = True,
        **kwargs,
    ) -> BlockRefinementSchedulerOutput | tuple:
        """
        Perform a single refinement step: sample from logits, commit confident masked positions, and optionally edit
        already-resolved ones.

        Args:
            model_output (`torch.Tensor` of shape `(batch_size, block_length, vocab_size)`):
                Raw logits from the model for the current block.
            timestep (`float` or `torch.Tensor`):
                The current corruption level, one entry of [`~BlockRefinementScheduler.timesteps`].
            sample (`torch.LongTensor` of shape `(batch_size, block_length)`):
                Current block token IDs, with `mask_token_id` at the positions still undecided.
            generator (`torch.Generator`, *optional*):
                RNG for sampling.
            return_dict (`bool`):
                Whether to return a [`DiscreteSchedulerOutput`] or a plain tuple.
        """
        overrides = self._pop_deprecated_step_kwargs(kwargs)
        mask_token_id = overrides.get("mask_token_id", self.config.mask_token_id)
        if mask_token_id is None:
            raise ValueError(
                "`mask_token_id` is required. Set it on the scheduler config, e.g. "
                "`BlockRefinementScheduler.from_config(scheduler.config, mask_token_id=tokenizer.mask_token_id)`. "
                "For the uniform corruption process, which has no mask token, use `UniformRefinementScheduler`."
            )
        if self.step_index is None:
            self._init_step_index(timestep)

        threshold = float(overrides.get("threshold", self.config.threshold))
        editing_threshold = overrides.get("editing_threshold", self.config.editing_threshold)

        sampled_tokens, sampled_probs = self._draw(model_output, overrides, generator)

        batch_size, block_length = sample.shape
        active_block = sample == mask_token_id
        masks_remaining = active_block.any()

        # --- Mask-filling transfer ---
        committed_mask = torch.zeros_like(sampled_tokens, dtype=torch.bool)
        if masks_remaining:
            # Per-step quota: `block_length` commits spread evenly over the schedule, remainder on the first steps.
            # Integer arithmetic off `step_index`, and clamped so a schedule overrun keeps the final step's quota
            # (matching the previous `_transfer_schedule` lookup).
            quota_step = min(self.step_index, self.num_inference_steps - 1)
            base, remainder = divmod(block_length, self.num_inference_steps)
            num_to_transfer = base + (1 if quota_step < remainder else 0)

            confidence = torch.where(
                active_block,
                sampled_probs.to(dtype=torch.float32),
                torch.full_like(sampled_probs, -torch.inf, dtype=torch.float32),
            )

            for b in range(batch_size):
                high_conf = confidence[b] > threshold
                if high_conf.sum().item() >= num_to_transfer:
                    committed_mask[b] = high_conf
                else:
                    k = min(num_to_transfer, int(active_block[b].sum().item()))
                    if k > 0:
                        _, idx = torch.topk(confidence[b], k=k)
                        committed_mask[b, idx] = True

        # --- Editing transfer (already-resolved, non-prompt positions) ---
        edited_mask = self._edited_mask(
            sampled_tokens,
            sampled_probs,
            sample,
            ~active_block,
            editing_threshold,
            overrides.get("prompt_mask"),
        )

        prev_sample = sample.clone()
        final_transfer = committed_mask | edited_mask
        if final_transfer.any():
            prev_sample[final_transfer] = sampled_tokens[final_transfer]

        self._step_index += 1

        if not return_dict:
            return prev_sample, sampled_tokens, sampled_probs, model_output, committed_mask, edited_mask
        return BlockRefinementSchedulerOutput(
            prev_sample=prev_sample,
            pred_original_sample=sampled_tokens,
            sampled_probs=sampled_probs,
            pred_logits=model_output,
            committed_mask=committed_mask,
            edited_mask=edited_mask,
        )

    def step_edit(
        self,
        model_output: torch.Tensor,
        sample: torch.LongTensor,
        *,
        generator: torch.Generator | None = None,
        return_dict: bool = True,
    ) -> BlockRefinementSchedulerOutput | tuple:
        """
        Overwrite already-resolved positions whose prediction is both different and confident.

        This is the post-mask refinement phase: once no `mask_token_id` remains there is nothing left to unmask, so the
        step is a confidence-thresholded overwrite rather than a diffusion step and takes **no** `timestep`. It also
        does not advance `step_index`, so a pipeline can run as many sweeps as it likes after exhausting the schedule.

        Args:
            model_output (`torch.Tensor` of shape `(batch_size, block_length, vocab_size)`):
                Raw logits from the model for the current block.
            sample (`torch.LongTensor` of shape `(batch_size, block_length)`):
                Current block token IDs, with every position resolved.
            generator (`torch.Generator`, *optional*):
                RNG for sampling.
            return_dict (`bool`):
                Whether to return a [`DiscreteSchedulerOutput`] or a plain tuple.
        """
        mask_token_id = self.config.mask_token_id
        if mask_token_id is None:
            raise ValueError(
                "`mask_token_id` is required. Set it on the scheduler config, e.g. "
                "`BlockRefinementScheduler.from_config(scheduler.config, mask_token_id=tokenizer.mask_token_id)`."
            )

        sampled_tokens, sampled_probs = self._draw(model_output, {}, generator)
        edited_mask = self._edited_mask(
            sampled_tokens,
            sampled_probs,
            sample,
            ~(sample == mask_token_id),
            self.config.editing_threshold,
            None,
        )

        prev_sample = sample.clone()
        if edited_mask.any():
            prev_sample[edited_mask] = sampled_tokens[edited_mask]

        committed_mask = torch.zeros_like(edited_mask)
        if not return_dict:
            return prev_sample, sampled_tokens, sampled_probs, model_output, committed_mask, edited_mask
        return BlockRefinementSchedulerOutput(
            prev_sample=prev_sample,
            pred_original_sample=sampled_tokens,
            sampled_probs=sampled_probs,
            pred_logits=model_output,
            committed_mask=committed_mask,
            edited_mask=edited_mask,
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
        deprecate(
            "check_eos_finished",
            "1.0.0",
            "`check_eos_finished` is deprecated and moves to the pipeline: it is denoising-loop control, not "
            "scheduling, and requiring it on the scheduler prevents swapping in another discrete scheduler.",
        )
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
        deprecate(
            "check_block_should_continue",
            "1.0.0",
            "`check_block_should_continue` is deprecated and moves to the pipeline: it is denoising-loop control, "
            "not scheduling, and requiring it on the scheduler prevents swapping in another discrete scheduler.",
        )
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
        timesteps: float | torch.Tensor,
        *,
        generator: torch.Generator | None = None,
        **kwargs,
    ) -> tuple[torch.LongTensor, torch.BoolTensor]:
        """
        Apply the forward (noising) process: replace each position with `mask_token_id` with probability `timesteps`.

        `timesteps` is the corruption level in `[0, 1]`, so it *is* the expected masking fraction — `1` masks
        everything, `0` masks nothing. The caller chooses it, matching every other `add_noise` in the library.

        Args:
            original_samples (`torch.LongTensor` of shape `(batch_size, seq_len)`):
                Clean token IDs.
            timesteps (`float` or `torch.Tensor`):
                Masking probability. A scalar applies to the whole batch; a tensor of shape `(batch_size,)` or
                `(batch_size, 1)` gives a per-example rate.
            generator (`torch.Generator`, *optional*):
                RNG for reproducibility.

        Returns:
            `tuple[torch.LongTensor, torch.BoolTensor]`: the noisy tokens and the boolean mask of noised positions.
        """
        if any(key in kwargs for key in ("attention_mask", "prompt_length", "block_length", "mask_token_id")):
            raise ValueError(
                "`add_noise` no longer takes `attention_mask`/`prompt_length`/`block_length`/`mask_token_id`, and "
                "its second positional argument is now `timesteps` (the masking probability) rather than "
                "`attention_mask`. It returns `(noisy, mask)` instead of `(noisy, noisy_rev, masked, masked_rev)`; "
                "build the complementary view in the caller as `~mask & valid`. This is a hard break rather than a "
                "deprecation because the meaning of the second positional argument changed, so a shim would "
                "silently corrupt data."
            )
        if kwargs:
            raise TypeError(f"Unexpected keyword arguments: {sorted(kwargs)}.")

        mask_token_id = self.config.mask_token_id
        if mask_token_id is None:
            raise ValueError(
                "`mask_token_id` is required. Set it on the scheduler config, e.g. "
                "`BlockRefinementScheduler.from_config(scheduler.config, mask_token_id=tokenizer.mask_token_id)`."
            )

        if not isinstance(timesteps, torch.Tensor):
            timesteps = torch.full(
                (original_samples.shape[0], 1), float(timesteps), device=original_samples.device, dtype=torch.float32
            )
        else:
            timesteps = timesteps.to(device=original_samples.device, dtype=torch.float32)
            if timesteps.ndim == 1:
                timesteps = timesteps[:, None]

        draw = torch.rand(original_samples.shape, device=original_samples.device, generator=generator)
        mask = draw < timesteps
        noisy = torch.where(mask, torch.full_like(original_samples, mask_token_id), original_samples)
        return noisy, mask


__all__ = ["BlockRefinementScheduler", "BlockRefinementSchedulerOutput"]
