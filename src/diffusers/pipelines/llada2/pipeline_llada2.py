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
from typing import Any, Callable

import torch
from tqdm.auto import tqdm

from ...callbacks import MultiPipelineCallbacks, PipelineCallback
from ...schedulers import BlockRefinementScheduler
from ...utils import BaseOutput, logging, replace_example_docstring
from ..pipeline_utils import DiffusionPipeline


logger = logging.get_logger(__name__)


EXAMPLE_DOC_STRING = """
    Examples:
        ```python
        >>> import torch
        >>> from transformers import AutoModelForCausalLM, AutoTokenizer
        >>> from diffusers import BlockRefinementScheduler, LLaDA2Pipeline

        >>> model_id = "inclusionAI/LLaDA2.1-mini"
        >>> model = AutoModelForCausalLM.from_pretrained(
        ...     model_id, trust_remote_code=True, dtype=torch.bfloat16, device_map="auto"
        ... )
        >>> tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        >>> scheduler = BlockRefinementScheduler()

        >>> pipe = LLaDA2Pipeline(model=model, scheduler=scheduler, tokenizer=tokenizer)
        >>> output = pipe(prompt="What is the meaning of life?", gen_length=256)
        >>> print(output.texts[0])
        ```
"""


@dataclass
class LLaDA2PipelineOutput(BaseOutput):
    sequences: torch.LongTensor
    texts: list[str] | None = None


class LLaDA2Pipeline(DiffusionPipeline):
    r"""
    Pipeline for LLaDA2-style discrete diffusion text generation via block-wise iterative refinement.

    This pipeline maintains a template sequence filled with a `mask_token_id` and refines it in blocks. In each
    refinement step, it samples candidate tokens for the active block and commits a subset based on confidence.

    The model is expected to accept an attention mask and `position_ids`, and to return logits of shape `[batch, seq,
    vocab_size]`.
    """

    model: Any
    scheduler: BlockRefinementScheduler
    tokenizer: Any

    _callback_tensor_inputs = [
        "block_x",
        "transfer_index",
        "editing_transfer_index",
        "sampled_tokens",
        "sampled_probs",
        "active_block",
    ]

    def __init__(
        self,
        model: Any,
        scheduler: BlockRefinementScheduler,
        tokenizer: Any | None = None,
    ):
        super().__init__()
        self.register_modules(model=model, scheduler=scheduler, tokenizer=tokenizer)
        self.eos_token_id = getattr(self.tokenizer, "eos_token_id", None) if self.tokenizer is not None else None
        self.mask_token_id = getattr(self.tokenizer, "mask_token_id", None) if self.tokenizer is not None else None

    @property
    def num_timesteps(self):
        return self._num_timesteps

    # --- Prompt encoding ---

    def _prepare_input_ids(
        self,
        *,
        prompt: str | list[str] | None,
        messages: list[dict[str, str]] | None,
        input_ids: torch.LongTensor | None,
        use_chat_template: bool,
        add_generation_prompt: bool,
        chat_template_kwargs: dict[str, Any] | None,
        attention_mask: torch.LongTensor | None = None,
    ) -> tuple[torch.LongTensor, torch.LongTensor]:
        """Convert prompt/messages/input_ids to `(input_ids, attention_mask)` tensors of shape `[batch, seq]`."""
        if input_ids is not None:
            if input_ids.ndim == 1:
                input_ids = input_ids.unsqueeze(0)
            if input_ids.ndim != 2:
                raise ValueError(f"`input_ids` must be 2D, got shape {tuple(input_ids.shape)}.")
            if input_ids.dtype != torch.long:
                raise ValueError(f"`input_ids` must be int64 token IDs, got dtype={input_ids.dtype}.")
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids, dtype=torch.long)
            else:
                if attention_mask.ndim == 1:
                    attention_mask = attention_mask.unsqueeze(0)
                if attention_mask.shape != input_ids.shape:
                    raise ValueError(
                        f"`attention_mask` shape {tuple(attention_mask.shape)} must match `input_ids` shape "
                        f"{tuple(input_ids.shape)}."
                    )
                attention_mask = attention_mask.to(dtype=torch.long)
            return input_ids, attention_mask

        if self.tokenizer is None:
            raise ValueError("Tokenizer is required when `input_ids` is not provided.")

        if messages is not None and prompt is not None:
            raise ValueError("Provide either `prompt` or `messages`, not both.")
        if messages is None and prompt is None:
            raise ValueError("Provide one of `prompt`, `messages`, or `input_ids`.")

        chat_template_kwargs = chat_template_kwargs or {}

        if messages is not None:
            encoded = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=add_generation_prompt,
                tokenize=True,
                return_tensors="pt",
                return_dict=True,
                **chat_template_kwargs,
            )
            ids = encoded["input_ids"]
            mask = encoded.get("attention_mask")
            if mask is None:
                mask = torch.ones_like(ids, dtype=torch.long)
            return ids, mask.to(dtype=torch.long)

        if use_chat_template and getattr(self.tokenizer, "chat_template", None):
            if isinstance(prompt, list):
                raise ValueError("`prompt` must be a string when `use_chat_template=True`.")
            encoded = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=add_generation_prompt,
                tokenize=True,
                return_tensors="pt",
                return_dict=True,
                **chat_template_kwargs,
            )
            ids = encoded["input_ids"]
            mask = encoded.get("attention_mask")
            if mask is None:
                mask = torch.ones_like(ids, dtype=torch.long)
            return ids, mask.to(dtype=torch.long)

        encoded = self.tokenizer(prompt, return_tensors="pt", padding=isinstance(prompt, list))
        ids = encoded["input_ids"]
        mask = encoded.get("attention_mask")
        if mask is None:
            mask = torch.ones_like(ids, dtype=torch.long)
        return ids, mask.to(dtype=torch.long)

    def check_inputs(
        self,
        prompt: str | list[str] | None,
        messages: list[dict[str, str]] | None,
        input_ids: torch.LongTensor | None,
        gen_length: int,
        block_length: int,
        num_inference_steps: int,
        minimal_topk: int,
        threshold: float,
        sampling_method: str,
        output_type: str,
        callback_on_step_end: Callable | PipelineCallback | MultiPipelineCallbacks | None,
        callback_on_step_end_tensor_inputs: list[str] | None,
    ):
        # Input source validation
        if prompt is None and messages is None and input_ids is None:
            raise ValueError("Provide one of `prompt`, `messages`, or `input_ids`.")
        if prompt is not None and messages is not None:
            raise ValueError("Provide either `prompt` or `messages`, not both.")
        if input_ids is not None:
            if input_ids.ndim not in (1, 2):
                raise ValueError(f"`input_ids` must be 1D or 2D, got shape {tuple(input_ids.shape)}.")
            if input_ids.dtype != torch.long:
                raise ValueError(f"`input_ids` must be int64 token IDs, got dtype={input_ids.dtype}.")
        if prompt is not None and input_ids is None and self.tokenizer is None:
            raise ValueError("Tokenizer is required when `input_ids` is not provided.")
        if messages is not None and input_ids is None and self.tokenizer is None:
            raise ValueError("Tokenizer is required when `input_ids` is not provided.")

        # Generation parameter validation
        if gen_length <= 0:
            raise ValueError(f"`gen_length` must be > 0, got {gen_length}.")
        if block_length <= 0:
            raise ValueError(f"`block_length` must be > 0, got {block_length}.")
        if num_inference_steps <= 0:
            raise ValueError(f"`num_inference_steps` must be > 0, got {num_inference_steps}.")
        if minimal_topk <= 0:
            raise ValueError(f"`minimal_topk` must be > 0, got {minimal_topk}.")
        if not (0.0 <= threshold <= 1.0) and not (threshold > 1.0):
            raise ValueError(f"`threshold` must be in [0, 1] (or > 1 to force top-k commits), got {threshold}.")
        if sampling_method not in {"auto", "greedy", "multinomial"}:
            raise ValueError(
                f"`sampling_method` must be one of {{'auto','greedy','multinomial'}}, got {sampling_method!r}."
            )
        if output_type not in {"seq", "text"}:
            raise ValueError(f"`output_type` must be 'seq' or 'text', got {output_type!r}.")

        # Callback validation
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
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: str | list[str] | None = None,
        messages: list[dict[str, str]] | None = None,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.LongTensor | None = None,
        use_chat_template: bool = True,
        add_generation_prompt: bool = True,
        gen_length: int = 2048,
        block_length: int | None = None,
        num_inference_steps: int = 32,
        temperature: float = 0.0,
        top_p: float | None = None,
        top_k: int | None = None,
        sampling_method: str = "multinomial",
        threshold: float = 0.7,
        editing_threshold: float | None = 0.5,
        max_post_steps: int = 16,
        minimal_topk: int = 1,
        eos_early_stop: bool = True,
        eos_token_id: int | None = None,
        mask_token_id: int | None = None,
        generator: torch.Generator | None = None,
        output_type: str = "text",
        return_dict: bool = True,
        callback_on_step_end: Callable[[int, int, dict], None]
        | PipelineCallback
        | MultiPipelineCallbacks
        | None = None,
        callback_on_step_end_tensor_inputs: list[str] | None = None,
    ) -> LLaDA2PipelineOutput | tuple[torch.LongTensor, list[str] | None]:
        """
        Generate text with block-wise refinement.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                Prompt text. When `use_chat_template` is `True` (default) and a tokenizer with a chat template is
                available, the prompt is wrapped in a chat message before tokenization.
            messages (`List[Dict[str, str]]`, *optional*):
                Chat messages to encode (e.g. `[{"role": "user", "content": "Hello"}]`). Takes precedence over `prompt`
                when provided. Requires a tokenizer with `apply_chat_template`.
            input_ids (`torch.LongTensor`, *optional*):
                Pre-tokenized input IDs. Takes precedence over `prompt` and `messages`.
            attention_mask (`torch.LongTensor`, *optional*):
                Per-token mask (1 for valid prompt tokens, 0 for padding) matching the shape of `input_ids`. Only used
                when `input_ids` is provided. When omitted (and `input_ids` is given), all positions are treated as
                valid. When constructing inputs from `prompt` / `messages`, the tokenizer's mask is carried through
                automatically.
            use_chat_template (`bool`, defaults to `True`):
                Whether to wrap the prompt in a chat template.
            add_generation_prompt (`bool`, defaults to `True`):
                Whether to add the generation prompt when using chat templates.
            gen_length (`int`):
                Number of tokens to generate.
            block_length (`int`, *optional*):
                Block size for refinement. If not provided, the scheduler's configured `block_length` is used.
            num_inference_steps (`int`):
                Number of refinement steps per block.
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
                Confidence threshold for editing already-committed (non-mask) tokens. When positive, after all mask
                tokens in a block are resolved, the pipeline continues refining: if the model predicts a different
                token with confidence above this threshold, the existing token is replaced. Set to `None`, `0.0`, or a
                negative value to disable editing. Defaults to `0.5`.
            max_post_steps (`int`):
                Maximum number of additional refinement iterations after all mask tokens in a block are resolved. Only
                used when `editing_threshold` is enabled. Defaults to `16`.
            minimal_topk (`int`):
                Minimum number of tokens to commit per step.
            eos_early_stop (`bool`):
                Whether to stop after committing EOS in a block.
            eos_token_id (`int`, *optional*):
                EOS token ID to use for early stopping.
            mask_token_id (`int`, *optional*):
                Mask token ID to use for the template.
            generator (`torch.Generator`, *optional*):
                RNG for sampling.
            output_type (`str`, defaults to `"text"`):
                Output format. `"text"` decodes sequences into strings (requires a tokenizer). `"seq"` returns raw
                token ID sequences only.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`LLaDA2PipelineOutput`] instead of a tuple.
            callback_on_step_end (`Callable` or `PipelineCallback`, *optional*):
                Callback executed after each refinement step with signature `callback_on_step_end(self, step: int,
                timestep: int, callback_kwargs: Dict)`.
            callback_on_step_end_tensor_inputs (`List[str]`, *optional*):
                Tensor keys to pass to the callback. Allowed keys: `block_x`, `transfer_index`,
                `editing_transfer_index`, `sampled_tokens`, `sampled_probs`, `active_block`.

        Examples:
        """
        # 1. Check inputs early
        if callback_on_step_end is not None and isinstance(
            callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)
        ):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs
        if callback_on_step_end_tensor_inputs is None:
            callback_on_step_end_tensor_inputs = ["block_x"]

        if block_length is None:
            block_length = self.scheduler.config.block_length

        self.check_inputs(
            prompt=prompt,
            messages=messages,
            input_ids=input_ids,
            gen_length=gen_length,
            block_length=block_length,
            num_inference_steps=num_inference_steps,
            minimal_topk=minimal_topk,
            threshold=threshold,
            sampling_method=sampling_method,
            output_type=output_type,
            callback_on_step_end=callback_on_step_end,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        )

        # 2. Prepare input IDs from prompt/messages/input_ids
        prompt_ids, prompt_attention_mask = self._prepare_input_ids(
            prompt=prompt,
            messages=messages,
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_chat_template=use_chat_template,
            add_generation_prompt=add_generation_prompt,
            chat_template_kwargs=None,
        )

        device = self._execution_device

        if prompt_ids.ndim == 1:
            prompt_ids = prompt_ids.unsqueeze(0)
        prompt_ids = prompt_ids.to(device=device)
        prompt_attention_mask = prompt_attention_mask.to(device=device)
        batch_size, prompt_length = prompt_ids.shape

        if eos_token_id is None:
            eos_token_id = self.eos_token_id
        if mask_token_id is None:
            mask_token_id = self.mask_token_id
        if mask_token_id is None:
            raise ValueError("`mask_token_id` must be provided (or available on the tokenizer).")

        num_inference_steps = min(num_inference_steps, gen_length // minimal_topk)

        self.scheduler.set_timesteps(num_inference_steps, device=device, block_length=block_length)

        # 3. Build attention mask and position IDs
        num_blocks = (prompt_length + gen_length + block_length - 1) // block_length
        total_length = num_blocks * block_length

        # 2D attention mask: prompt tokenizer mask + ones over generated positions + zeros over the
        # block-aligned tail past `prompt_length + gen_length`. The model handles backend-specific
        # conversion internally; this just tells it which positions are real context.
        attn_mask = torch.zeros((batch_size, total_length), device=device, dtype=torch.long)
        attn_mask[:, :prompt_length] = prompt_attention_mask
        attn_mask[:, prompt_length : prompt_length + gen_length] = 1

        position_ids = torch.arange(total_length, device=device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)

        # 4. Prepare latents (fully masked sequence)
        x = torch.full((batch_size, total_length), mask_token_id, device=device, dtype=torch.long)
        if prompt_length > 0:
            x[:, :prompt_length] = prompt_ids

        prefill_blocks = prompt_length // block_length
        self._num_timesteps = num_inference_steps * max(num_blocks - prefill_blocks, 0)

        finished = torch.zeros((batch_size,), device=device, dtype=torch.bool)
        editing_enabled = editing_threshold is not None and editing_threshold > 0.0
        global_step = 0

        # 5. Block-wise refinement loop
        outer_progress_bar_config = getattr(self, "_progress_bar_config", {}).copy()
        block_progress_bar_config = {**outer_progress_bar_config, "position": 0, "desc": "Blocks"}
        for num_block in tqdm(range(prefill_blocks, num_blocks), **block_progress_bar_config):
            current_window_end = (num_block + 1) * block_length
            block_x = x[:, :current_window_end]
            block_attn_mask = attn_mask[:, :current_window_end]
            block_position_ids = position_ids[:, :current_window_end]

            # Identify which positions in the block are prompt (non-editable).
            block_start_pos = num_block * block_length
            prompt_mask_in_block = torch.zeros(block_length, device=device, dtype=torch.bool)
            if block_start_pos < prompt_length:
                prompt_end_in_block = min(prompt_length - block_start_pos, block_length)
                prompt_mask_in_block[:prompt_end_in_block] = True

            post_steps = 0
            step_idx = 0
            should_continue = True
            inner_progress_bar_config = {
                **outer_progress_bar_config,
                "position": 1,
                "leave": False,
                "desc": f"Block {num_block} Inference Steps",
            }
            progress_bar = tqdm(total=num_inference_steps, **inner_progress_bar_config)

            while should_continue:
                block_tokens = block_x[:, -block_length:]
                masks_remaining = (block_tokens == mask_token_id).any()

                if not masks_remaining:
                    post_steps += 1

                logits = self.model(block_x, attention_mask=block_attn_mask, position_ids=block_position_ids).logits
                block_logits = logits[:, -block_length:, :]

                scheduler_output = self.scheduler.step(
                    model_output=block_logits,
                    timestep=step_idx,
                    sample=block_tokens,
                    mask_token_id=mask_token_id,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    sampling_method=sampling_method,
                    threshold=threshold,
                    editing_threshold=editing_threshold,
                    minimal_topk=minimal_topk,
                    prompt_mask=prompt_mask_in_block,
                    generator=generator,
                    return_dict=True,
                )

                transfer_index = scheduler_output.transfer_index
                editing_transfer_index = scheduler_output.editing_transfer_index
                sampled_tokens = scheduler_output.sampled_tokens
                sampled_probs = scheduler_output.sampled_probs
                active_block = block_tokens == mask_token_id
                final_transfer = transfer_index | editing_transfer_index

                # Freeze rows that already emitted EOS so further blocks don't extend them.
                if eos_early_stop and finished.any():
                    final_transfer = final_transfer & ~finished[:, None]

                if final_transfer.any():
                    block_x[:, -block_length:] = torch.where(
                        final_transfer, scheduler_output.prev_sample, block_tokens
                    )

                if eos_early_stop and eos_token_id is not None:
                    finished = self.scheduler.check_eos_finished(
                        cur_x=block_x,
                        sampled_tokens=scheduler_output.sampled_tokens,
                        final_transfer=final_transfer,
                        finished=finished,
                        eos_token_id=eos_token_id,
                        mask_token_id=mask_token_id,
                        prompt_length=prompt_length,
                    )

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, global_step, step_idx, callback_kwargs)
                    block_x = callback_outputs.pop("block_x", block_x)

                global_step += 1
                if masks_remaining:
                    step_idx += 1
                    progress_bar.update(1)

                should_continue = self.scheduler.check_block_should_continue(
                    step_idx=step_idx,
                    masks_remaining=masks_remaining,
                    editing_enabled=editing_enabled,
                    editing_transfer_index=editing_transfer_index,
                    post_steps=post_steps,
                    max_post_steps=max_post_steps,
                    finished=finished,
                )

            progress_bar.close()
            x[:, :current_window_end] = block_x
            if eos_early_stop and finished.all():
                break

        # 6. Post-process output
        generated = x[:, : prompt_length + gen_length]
        sequences = generated[:, prompt_length:]

        # For decode, trim each row at the first EOS so post-EOS positions (which may still hold
        # mask tokens or refined content for unfinished blocks) don't leak into the decoded text.
        decode_sequences: list[torch.LongTensor] | torch.LongTensor = sequences
        if eos_token_id is not None:
            decode_sequences = [
                seq[: int((seq == eos_token_id).nonzero(as_tuple=True)[0][0]) + 1]
                if (seq == eos_token_id).any()
                else seq
                for seq in sequences
            ]

        texts = None
        if output_type == "text" and self.tokenizer is not None:
            texts = self.tokenizer.batch_decode(decode_sequences, skip_special_tokens=True)

        if not return_dict:
            return sequences.to(device=device), texts
        return LLaDA2PipelineOutput(sequences=sequences.to(device=device), texts=texts)


__all__ = ["LLaDA2Pipeline", "LLaDA2PipelineOutput"]
