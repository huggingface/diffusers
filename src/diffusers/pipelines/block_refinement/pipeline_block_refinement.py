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
from typing import Any, Callable, Dict, List, Optional, Union

import torch

from ...callbacks import MultiPipelineCallbacks, PipelineCallback
from ...utils import BaseOutput
from ..pipeline_utils import DiffusionPipeline, DiscreteDiffusionPipelineMixin


@dataclass
class BlockRefinementPipelineOutput(BaseOutput):
    sequences: torch.LongTensor
    texts: Optional[List[str]] = None


def _get_num_transfer_tokens(block_length: int, steps: int) -> torch.LongTensor:
    if steps <= 0:
        return torch.zeros((0,), dtype=torch.long)
    base = int(block_length) // int(steps)
    remainder = int(block_length) % int(steps)
    out = torch.full((int(steps),), base, dtype=torch.long)
    out[:remainder] += 1
    return out


class BlockRefinementPipeline(DiffusionPipeline, DiscreteDiffusionPipelineMixin):
    """
    Block-wise iterative refinement pipeline for token generation.

    This pipeline maintains a template sequence filled with a `mask_token_id` and refines it in blocks. In each
    refinement step, it samples candidate tokens for the active block and commits a subset based on confidence.

    The model is expected to accept an additive attention mask of shape `[batch, 1, seq, seq]` (0 for allowed, `-inf`
    for disallowed) and `position_ids`, and to return logits of shape `[batch, seq, vocab_size]`.
    """

    model: Any
    tokenizer: Any

    _callback_tensor_inputs = ["cur_x", "x0", "x0_p", "transfer_index", "confidence", "active_block"]

    def __init__(
        self,
        model: Any,
        tokenizer: Optional[Any] = None,
    ):
        super().__init__()
        self.register_modules(model=model, tokenizer=tokenizer)

    @property
    def num_timesteps(self):
        return self._num_timesteps

    def _model_forward_logits(
        self,
        input_ids: torch.LongTensor,
        *,
        attention_mask_4d: Optional[torch.Tensor],
        attention_mask_2d: Optional[torch.Tensor],
        position_ids: torch.LongTensor,
        attention_mask_mode: str,
    ) -> tuple[torch.Tensor, str]:
        if attention_mask_mode not in {"auto", "4d", "2d", "none"}:
            raise ValueError(
                f"`attention_mask_mode` must be one of {{'auto','4d','2d','none'}}, got {attention_mask_mode!r}."
            )

        def _call(mask):
            return self.model(input_ids, attention_mask=mask, position_ids=position_ids).logits

        if attention_mask_mode == "none":
            return _call(None), "none"
        if attention_mask_mode == "2d":
            return _call(attention_mask_2d), "2d"
        if attention_mask_mode == "4d":
            return _call(attention_mask_4d), "4d"

        # auto: try 4d additive mask first, then fall back to 2d padding mask, then no mask.
        try:
            return _call(attention_mask_4d), "4d"
        except (TypeError, ValueError, RuntimeError):
            pass
        try:
            return _call(attention_mask_2d), "2d"
        except (TypeError, ValueError, RuntimeError):
            return _call(None), "none"

    def _build_block_attention_mask(
        self,
        *,
        num_blocks: int,
        block_length: int,
        total_length: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        block_mask = torch.tril(torch.ones(num_blocks, num_blocks, device=device, dtype=torch.bool))
        attn = (
            block_mask.repeat_interleave(block_length, dim=0)
            .repeat_interleave(block_length, dim=1)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        attn = attn[:, :, :total_length, :total_length]
        return torch.where(
            attn,
            torch.zeros((), device=device, dtype=dtype),
            torch.full((), float("-inf"), device=device, dtype=dtype),
        )

    def _encode_prompt(
        self,
        prompt: Optional[Union[str, List[str]]],
        prompt_ids: Optional[torch.LongTensor],
        *,
        device: torch.device,
    ) -> torch.LongTensor:
        if prompt_ids is not None:
            if prompt_ids.ndim == 1:
                prompt_ids = prompt_ids.unsqueeze(0)
            if prompt_ids.ndim != 2:
                raise ValueError(
                    f"`prompt_ids` must have shape [prompt_len] or [batch, prompt_len], got {prompt_ids.shape}."
                )
            if prompt_ids.dtype != torch.long:
                raise ValueError(f"`prompt_ids` must be int64 token IDs, got dtype={prompt_ids.dtype}.")
            return prompt_ids.to(device=device)

        if prompt is None:
            return torch.zeros((1, 0), device=device, dtype=torch.long)
        if getattr(self, "tokenizer", None) is None:
            raise ValueError("`prompt` requires a tokenizer, but no tokenizer was provided to the pipeline.")

        encoded = self.tokenizer(prompt, return_tensors="pt", padding=True)
        return encoded["input_ids"].to(device=device)

    def prepare_latents(
        self,
        batch_size: int,
        total_length: int,
        mask_token_id: int,
        device: torch.device,
    ) -> torch.LongTensor:
        return torch.full((batch_size, total_length), int(mask_token_id), device=device, dtype=torch.long)

    def check_inputs(
        self,
        gen_length: int,
        block_length: int,
        steps: int,
        minimal_topk: int,
        threshold: float,
        sampling_method: str,
        callback_on_step_end: Optional[Union[Callable, PipelineCallback, MultiPipelineCallbacks]],
        callback_on_step_end_tensor_inputs: Optional[List[str]],
    ):
        if gen_length <= 0:
            raise ValueError(f"`gen_length` must be > 0, got {gen_length}.")
        if block_length <= 0:
            raise ValueError(f"`block_length` must be > 0, got {block_length}.")
        if steps <= 0:
            raise ValueError(f"`steps` must be > 0, got {steps}.")
        if minimal_topk <= 0:
            raise ValueError(f"`minimal_topk` must be > 0, got {minimal_topk}.")
        if not (0.0 <= threshold <= 1.0) and not (threshold > 1.0):
            raise ValueError(f"`threshold` must be in [0, 1] (or > 1 to force top-k commits), got {threshold}.")
        if sampling_method not in {"auto", "greedy", "multinomial"}:
            raise ValueError(
                f"`sampling_method` must be one of {{'auto','greedy','multinomial'}}, got {sampling_method!r}."
            )
        if callback_on_step_end is not None and isinstance(
            callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)
        ):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs
        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found "
                f"{[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

    @torch.no_grad()
    def __call__(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        prompt_ids: Optional[torch.LongTensor] = None,
        gen_length: int = 128,
        block_length: int = 32,
        steps: int = 32,
        temperature: float = 0.0,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        sampling_method: str = "auto",
        threshold: float = 0.95,
        editing_threshold: Optional[float] = None,
        max_post_steps: int = 0,
        minimal_topk: int = 1,
        eos_early_stop: bool = False,
        eos_token_id: Optional[int] = None,
        mask_token_id: Optional[int] = None,
        attention_mask_mode: str = "auto",
        generator: Optional[torch.Generator] = None,
        return_text: bool = True,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: Optional[List[str]] = None,
    ) -> BlockRefinementPipelineOutput:
        """
        Generate tokens with block-wise refinement.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                Prompt text to encode with the tokenizer.
            prompt_ids (`torch.LongTensor`, *optional*):
                Pre-tokenized prompt IDs with shape `[prompt_len]` or `[batch, prompt_len]`.
            gen_length (`int`):
                Number of tokens to generate.
            block_length (`int`):
                Block size for refinement.
            steps (`int`):
                Refinement steps per block.
            temperature (`float`):
                Sampling temperature.
            top_p (`float`, *optional*):
                Nucleus sampling cutoff.
            top_k (`int`, *optional*):
                Top-k sampling cutoff.
            sampling_method (`str`):
                Sampling method (`auto`, `greedy`, `multinomial`).
            threshold (`float`):
                Confidence threshold for committing tokens.
            editing_threshold (`float`, *optional*):
                Confidence threshold for editing already-committed (non-mask) tokens. When set, after all mask tokens
                in a block are resolved, the pipeline continues refining: if the model predicts a different token with
                confidence above this threshold, the existing token is replaced. Set to `None` or a negative value to
                disable editing. Defaults to `None` (disabled).
            max_post_steps (`int`):
                Maximum number of additional refinement iterations after all mask tokens in a block are resolved. Only
                used when `editing_threshold` is enabled. Defaults to `0` (no post-mask editing steps).
            minimal_topk (`int`):
                Minimum number of tokens to commit per step.
            eos_early_stop (`bool`):
                Whether to stop after committing EOS in a block.
            eos_token_id (`int`, *optional*):
                EOS token ID to use for early stopping.
            mask_token_id (`int`, *optional*):
                Mask token ID to use for the template.
            attention_mask_mode (`str`):
                Attention mask mode (`auto`, `4d`, `2d`, `none`).
            generator (`torch.Generator`, *optional*):
                RNG for sampling.
            return_text (`bool`, *optional*, defaults to `True`):
                Whether to decode sequences into text when a tokenizer is available.
            callback_on_step_end (`Callable` or `PipelineCallback`, *optional*):
                Callback executed after each refinement step with signature `callback_on_step_end(self, step: int,
                timestep: int, callback_kwargs: Dict)`.
            callback_on_step_end_tensor_inputs (`List[str]`, *optional*):
                Tensor keys to pass to the callback. Allowed keys: `cur_x`, `x0`, `x0_p`, `transfer_index`,
                `confidence`, `active_block`.
        """
        if callback_on_step_end is not None and isinstance(
            callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)
        ):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs
        if callback_on_step_end_tensor_inputs is None:
            callback_on_step_end_tensor_inputs = ["cur_x"]

        self.check_inputs(
            gen_length=gen_length,
            block_length=block_length,
            steps=steps,
            minimal_topk=minimal_topk,
            threshold=threshold,
            sampling_method=sampling_method,
            callback_on_step_end=callback_on_step_end,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        )

        model_params = list(self.model.parameters()) if hasattr(self.model, "parameters") else []
        model_device = model_params[0].device if len(model_params) > 0 else torch.device("cpu")

        prompt_ids = self._encode_prompt(prompt, prompt_ids, device=model_device)
        batch_size, prompt_length = prompt_ids.shape

        if eos_token_id is None:
            eos_token_id = getattr(getattr(self, "tokenizer", None), "eos_token_id", None)
        if mask_token_id is None:
            mask_token_id = getattr(getattr(self, "tokenizer", None), "mask_token_id", None)
        if mask_token_id is None:
            raise ValueError("`mask_token_id` must be provided (or available on the tokenizer).")

        steps = min(int(steps), int(gen_length) // int(minimal_topk))

        num_blocks = (prompt_length + int(gen_length) + int(block_length) - 1) // int(block_length)
        total_length = int(num_blocks) * int(block_length)

        dtype = getattr(self.model, "dtype", torch.float32)
        attn_dtype = torch.bfloat16 if dtype in (torch.bfloat16, torch.float16) else torch.float32
        attn_mask_4d = self._build_block_attention_mask(
            num_blocks=num_blocks,
            block_length=block_length,
            total_length=total_length,
            device=model_device,
            dtype=attn_dtype,
        )
        attn_mask_2d_full = torch.ones((batch_size, total_length), device=model_device, dtype=torch.long)
        position_ids = (
            torch.arange(total_length, device=model_device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
        )

        x = self.prepare_latents(batch_size, total_length, int(mask_token_id), model_device)
        if prompt_length > 0:
            x[:, :prompt_length] = prompt_ids.to(device=model_device)

        prefill_blocks = prompt_length // int(block_length)
        self._num_timesteps = int(steps) * max(int(num_blocks) - int(prefill_blocks), 0)
        transfer_schedule = _get_num_transfer_tokens(int(block_length), int(steps)).to(device=model_device)

        finished = torch.zeros((batch_size,), device=model_device, dtype=torch.bool)
        resolved_attention_mode: str = str(attention_mask_mode)

        use_multinomial = sampling_method == "multinomial" or (sampling_method == "auto" and float(temperature) != 0.0)
        editing_enabled = editing_threshold is not None and editing_threshold >= 0.0
        global_step = 0

        for num_block in range(int(prefill_blocks), int(num_blocks)):
            current_window_end = (num_block + 1) * int(block_length)
            cur_x = x[:, :current_window_end]
            cur_attn_mask_4d = attn_mask_4d[:, :, :current_window_end, :current_window_end]
            cur_attn_mask_2d = attn_mask_2d_full[:, :current_window_end]
            cur_position_ids = position_ids[:, :current_window_end]

            # Identify which positions in the block are prompt (non-editable).
            block_start_pos = num_block * int(block_length)
            prompt_mask_in_block = torch.zeros(int(block_length), device=model_device, dtype=torch.bool)
            if block_start_pos < prompt_length:
                prompt_end_in_block = min(prompt_length - block_start_pos, int(block_length))
                prompt_mask_in_block[:prompt_end_in_block] = True

            post_steps = 0
            step_idx = 0
            while step_idx < int(steps) or (editing_enabled and post_steps <= int(max_post_steps)):
                if finished.all():
                    break

                active_block = cur_x[:, -int(block_length) :] == int(mask_token_id)
                masks_remaining = active_block.sum() > 0

                if not masks_remaining and not editing_enabled:
                    break
                if not masks_remaining:
                    post_steps += 1
                    if post_steps > int(max_post_steps):
                        break

                logits, resolved_attention_mode = self._model_forward_logits(
                    cur_x,
                    attention_mask_4d=cur_attn_mask_4d,
                    attention_mask_2d=cur_attn_mask_2d,
                    position_ids=cur_position_ids,
                    attention_mask_mode=resolved_attention_mode,
                )
                block_logits = logits[:, -int(block_length) :, :]

                x0, x0_p = self._sample_with_temperature_topk_topp(
                    block_logits,
                    temperature=float(temperature),
                    top_k=top_k,
                    top_p=top_p,
                    generator=generator,
                    use_multinomial=use_multinomial,
                )

                # --- Mask-filling transfer ---
                transfer_index = torch.zeros_like(x0, dtype=torch.bool)
                if masks_remaining and step_idx < int(steps):
                    clamped_step = min(step_idx, len(transfer_schedule) - 1)
                    num_to_transfer = int(transfer_schedule[clamped_step].item())

                    confidence = torch.where(
                        active_block,
                        x0_p.to(dtype=torch.float32),
                        torch.full_like(x0_p, -torch.inf, dtype=torch.float32),
                    )

                    for b in range(batch_size):
                        if finished[b]:
                            continue
                        high_conf = confidence[b] > float(threshold)
                        if high_conf.sum().item() >= num_to_transfer:
                            transfer_index[b] = high_conf
                        else:
                            k = min(num_to_transfer, int(active_block[b].sum().item()))
                            if k > 0:
                                _, idx = torch.topk(confidence[b], k=k)
                                transfer_index[b, idx] = True

                # --- Editing transfer (non-mask, non-prompt positions) ---
                editing_transfer_index = torch.zeros_like(x0, dtype=torch.bool)
                if editing_enabled:
                    old_block_tokens = cur_x[:, -int(block_length) :]
                    editable = (~active_block) & (~prompt_mask_in_block.unsqueeze(0))
                    editing_conf = torch.where(
                        editable, x0_p.to(dtype=torch.float32), torch.full_like(x0_p, -torch.inf, dtype=torch.float32)
                    )
                    high_conf_edit = editing_conf > float(editing_threshold)
                    token_changed = x0 != old_block_tokens
                    editing_transfer_index = high_conf_edit & token_changed & editable

                final_transfer = transfer_index | editing_transfer_index
                if final_transfer.any():
                    updated = cur_x[:, -int(block_length) :].clone()
                    updated[final_transfer] = x0[final_transfer]
                    cur_x[:, -int(block_length) :] = updated

                # Break if no masks remain and no edits were made.
                if not masks_remaining and not editing_transfer_index.any():
                    break

                if eos_early_stop and eos_token_id is not None:
                    for b in range(batch_size):
                        if finished[b]:
                            continue
                        eos_in_commits = (x0[b][final_transfer[b]] == int(eos_token_id)).any().item()
                        if not eos_in_commits:
                            continue
                        eos_pos = (cur_x[b] == int(eos_token_id)).nonzero(as_tuple=True)
                        if len(eos_pos[0]) == 0:
                            continue
                        eos_pos = int(eos_pos[0][0].item())
                        if prompt_length >= eos_pos:
                            continue
                        if (cur_x[b, prompt_length:eos_pos] != int(mask_token_id)).all().item():
                            finished[b] = True

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, global_step, step_idx, callback_kwargs)
                    cur_x = callback_outputs.pop("cur_x", cur_x)

                global_step += 1
                if masks_remaining:
                    step_idx += 1

            x[:, :current_window_end] = cur_x
            if eos_token_id is not None and (x[:, prompt_length:current_window_end] == int(eos_token_id)).any().item():
                if eos_early_stop:
                    break

        generated = x[:, : prompt_length + int(gen_length)]
        sequences = generated[:, prompt_length:]
        if eos_token_id is not None and batch_size == 1:
            eos_positions = (sequences[0] == int(eos_token_id)).nonzero(as_tuple=True)[0]
            if len(eos_positions) > 0:
                sequences = sequences[:, : int(eos_positions[0].item()) + 1]

        texts = None
        if return_text and getattr(self, "tokenizer", None) is not None:
            texts = self.tokenizer.batch_decode(sequences, skip_special_tokens=True)

        return BlockRefinementPipelineOutput(sequences=sequences.to(device=model_device), texts=texts)
