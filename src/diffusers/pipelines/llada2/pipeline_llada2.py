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

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable

import torch
from tqdm.auto import tqdm

from ...callbacks import MultiPipelineCallbacks, PipelineCallback
from ...schedulers import BlockRefinementScheduler, DiscreteSchedulerOutput
from ...utils import BaseOutput, deprecate, logging, replace_example_docstring
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
        >>> # The scheduler owns sampling: temperature, top-k/p, and the confidence thresholds.
        >>> scheduler = BlockRefinementScheduler(threshold=0.7, editing_threshold=0.5)

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
        "active_block",
        "pred_original_sample",
        "sampled_probs",
        "committed_mask",
        "edited_mask",
    ]

    # Pre-1.0 callback keys, still resolved for one release: old name -> current name.
    _deprecated_callback_tensor_inputs = {
        "transfer_index": "committed_mask",
        "editing_transfer_index": "edited_mask",
        "sampled_tokens": "pred_original_sample",
    }

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
        minimal_topk: int | None,
        threshold: float | None,
        sampling_method: str | None,
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
        # `threshold` / `sampling_method` / `minimal_topk` are deprecated here and validated by the scheduler's
        # own `__init__` -- but this call path reaches the config through `register_to_config`, which merges
        # values in without re-running that validation. These checks are what catches a bad *argument*, so they
        # stay until the arguments themselves go at 1.0.0.
        if minimal_topk is not None and minimal_topk <= 0:
            raise ValueError(f"`minimal_topk` must be > 0, got {minimal_topk}.")
        if threshold is not None and threshold < 0.0:
            raise ValueError(f"`threshold` must be in [0, 1] (or > 1 to force top-k commits), got {threshold}.")
        if sampling_method is not None and sampling_method not in {"auto", "greedy", "multinomial"}:
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
        allowed_tensor_inputs = self._callback_tensor_inputs + list(self._deprecated_callback_tensor_inputs)
        if callback_on_step_end_tensor_inputs is not None and not all(
            k in allowed_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found "
                f"{[k for k in callback_on_step_end_tensor_inputs if k not in allowed_tensor_inputs]}"
            )

    @contextmanager
    def _scheduler_config_overrides(self, overrides: dict[str, Any]):
        """
        Apply `overrides` to the scheduler's config for the duration of a call, then restore it.

        Two things need this: handing the scheduler the tokenizer's `mask_token_id`, and honouring the deprecated
        per-call sampling arguments for one release. Overriding in place rather than running against a reconfigured
        copy keeps `self.scheduler` the object the loop actually steps, which is what callbacks, `step_index`
        inspection, and anything else keyed on identity rely on.
        """
        if not overrides:
            yield
            return
        previous = {key: self.scheduler.config.get(key, None) for key in overrides}
        self.scheduler.register_to_config(**overrides)
        try:
            yield
        finally:
            self.scheduler.register_to_config(**previous)

    # --- Denoising-loop control ---

    @staticmethod
    def _update_finished(
        cur_x: torch.LongTensor,
        pred_original_sample: torch.LongTensor,
        final_transfer: torch.BoolTensor,
        finished: torch.BoolTensor,
        eos_token_id: int,
        mask_token_id: int,
        prompt_length: int,
    ) -> torch.BoolTensor:
        """
        Mark rows finished once they commit an EOS with no unresolved position before it.

        This is loop control, not scheduling, so it lives here rather than on the scheduler (where it used to sit as
        `BlockRefinementScheduler.check_eos_finished`) — the pipeline can then drive any discrete scheduler.
        """
        for b in range(cur_x.shape[0]):
            if finished[b]:
                continue
            eos_in_commits = (pred_original_sample[b][final_transfer[b]] == eos_token_id).any().item()
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

    @staticmethod
    def _resolve_transfer(
        scheduler_output: DiscreteSchedulerOutput,
        editable: torch.BoolTensor,
        finished: torch.BoolTensor,
        eos_early_stop: bool,
    ) -> tuple[torch.BoolTensor, torch.BoolTensor]:
        """
        Turn a scheduler output into the positions this step is allowed to write.

        The scheduler proposes; the pipeline decides. Prompt positions are frozen (`editable`) and rows that already
        emitted EOS are frozen wholesale, so later blocks cannot extend them.
        """
        committed_mask = scheduler_output.committed_mask
        edited_mask = scheduler_output.edited_mask
        if edited_mask is None:
            edited_mask = torch.zeros_like(committed_mask)
        edited_mask = edited_mask & editable
        final_transfer = (committed_mask | edited_mask) & editable
        if eos_early_stop and finished.any():
            final_transfer = final_transfer & ~finished[:, None]
        return edited_mask, final_transfer

    def _run_step_callback(
        self,
        callback_on_step_end: Callable | PipelineCallback | MultiPipelineCallbacks,
        tensor_inputs: list[str],
        *,
        step: int,
        timestep: torch.Tensor,
        block_x: torch.LongTensor,
        active_block: torch.BoolTensor,
        scheduler_output: DiscreteSchedulerOutput,
        edited_mask: torch.BoolTensor,
    ) -> torch.LongTensor:
        """Run `callback_on_step_end`, resolving the advertised tensor keys and their pre-1.0 aliases."""
        available = {
            "block_x": block_x,
            "active_block": active_block,
            "pred_original_sample": scheduler_output.pred_original_sample,
            "sampled_probs": scheduler_output.sampled_probs,
            "committed_mask": scheduler_output.committed_mask,
            "edited_mask": edited_mask,
        }
        callback_kwargs = {}
        for key in tensor_inputs:
            renamed = self._deprecated_callback_tensor_inputs.get(key)
            if renamed is not None:
                deprecate(
                    key,
                    "1.0.0",
                    f"The callback tensor input `{key}` is deprecated; use `{renamed}` instead.",
                )
            callback_kwargs[key] = available[renamed or key]
        callback_outputs = callback_on_step_end(self, step, timestep, callback_kwargs)
        return callback_outputs.pop("block_x", block_x)

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
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        sampling_method: str | None = None,
        threshold: float | None = None,
        editing_threshold: float | None = None,
        max_post_steps: int = 16,
        minimal_topk: int | None = None,
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
            temperature (`float`, *optional*):
                Deprecated. Sampling temperature; set it on the scheduler instead.
            top_p (`float`, *optional*):
                Deprecated. Nucleus sampling cutoff; set it on the scheduler instead.
            top_k (`int`, *optional*):
                Deprecated. Top-k sampling cutoff; set it on the scheduler instead.
            sampling_method (`str`, *optional*):
                Deprecated. Sampling method (`auto`, `greedy`, `multinomial`); set it on the scheduler instead.
            threshold (`float`, *optional*):
                Deprecated. Confidence threshold for committing tokens; set it on the scheduler instead.
            editing_threshold (`float`, *optional*):
                Deprecated. Confidence threshold for editing already-committed (non-mask) tokens; set it on the
                scheduler instead. When positive, after all mask tokens in a block are resolved the pipeline keeps
                refining: if the model predicts a different token with confidence above this threshold, the existing
                token is replaced. `None`, `0.0`, or a negative value disables editing.
            max_post_steps (`int`):
                Maximum number of additional refinement iterations after all mask tokens in a block are resolved. Only
                used when the scheduler's `editing_threshold` is enabled. Defaults to `16`.
            minimal_topk (`int`, *optional*):
                Deprecated and unused by the scheduler; it only ever capped `num_inference_steps` at `gen_length //
                minimal_topk`.
            eos_early_stop (`bool`):
                Whether to stop after committing EOS in a block.
            eos_token_id (`int`, *optional*):
                EOS token ID to use for early stopping.
            mask_token_id (`int`, *optional*):
                Mask token ID to use for the template. Falls back to the tokenizer's `mask_token_id`, then to the
                scheduler's configured `mask_token_id`.
            generator (`torch.Generator`, *optional*):
                RNG for sampling.
            output_type (`str`, defaults to `"text"`):
                Output format. `"text"` decodes sequences into strings (requires a tokenizer). `"seq"` returns raw
                token ID sequences only.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`LLaDA2PipelineOutput`] instead of a tuple.
            callback_on_step_end (`Callable` or `PipelineCallback`, *optional*):
                Callback executed after each refinement step with signature `callback_on_step_end(self, step: int,
                timestep: float, callback_kwargs: Dict)`. During the editing phase there is no schedule left to report,
                so `timestep` repeats the last mask-filling timestep.
            callback_on_step_end_tensor_inputs (`List[str]`, *optional*):
                Tensor keys to pass to the callback. Allowed keys: `block_x`, `active_block`, `pred_original_sample`,
                `sampled_probs`, `committed_mask`, `edited_mask`. The pre-1.0 names `transfer_index`,
                `editing_transfer_index` and `sampled_tokens` still resolve, with a warning.

        Examples:

        Returns:
            [`~pipelines.llada2.pipeline_llada2.LLaDA2PipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.llada2.pipeline_llada2.LLaDA2PipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is the generated token IDs (`torch.LongTensor`)
                and the second element is the decoded texts (`list[str]`), or `None` when `output_type` is `"seq"`.
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
            mask_token_id = getattr(self.scheduler.config, "mask_token_id", None)
        if mask_token_id is None:
            raise ValueError(
                "`mask_token_id` must be provided (or available on the tokenizer, or configured on the scheduler)."
            )

        # The scheduler owns logit shaping and the confidence thresholds. The per-call knobs below are
        # deprecated; while they are still honoured -- and to hand the scheduler the tokenizer's
        # `mask_token_id` -- they are applied to the scheduler config for the duration of the call.
        scheduler_overrides: dict[str, Any] = {}
        for name, value in (
            ("temperature", temperature),
            ("top_p", top_p),
            ("top_k", top_k),
            ("sampling_method", sampling_method),
            ("threshold", threshold),
            ("editing_threshold", editing_threshold),
        ):
            if value is None:
                continue
            deprecate(
                name,
                "1.0.0",
                f"Passing `{name}` to `LLaDA2Pipeline.__call__` is deprecated; the scheduler owns sampling now. "
                f"Set it there instead: "
                f"`pipe.scheduler = BlockRefinementScheduler.from_config(pipe.scheduler.config, {name}=...)`.",
            )
            scheduler_overrides[name] = value
        if minimal_topk is not None:
            deprecate(
                "minimal_topk",
                "1.0.0",
                "`minimal_topk` is deprecated and has no replacement: the scheduler never used it, and its only "
                "effect here was to cap `num_inference_steps` at `gen_length // minimal_topk`.",
            )
        if getattr(self.scheduler.config, "mask_token_id", None) != mask_token_id:
            scheduler_overrides["mask_token_id"] = mask_token_id

        num_inference_steps = min(num_inference_steps, gen_length // (minimal_topk or 1))

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
        global_step = 0

        # 5. Block-wise refinement loop
        with self._scheduler_config_overrides(scheduler_overrides):
            active_editing_threshold = self.scheduler.config.get("editing_threshold", None)
            editing_enabled = active_editing_threshold is not None and active_editing_threshold > 0.0

            outer_progress_bar_config = getattr(self, "_progress_bar_config", {}).copy()
            block_progress_bar_config = {**outer_progress_bar_config, "position": 0, "desc": "Blocks"}
            for num_block in tqdm(range(prefill_blocks, num_blocks), **block_progress_bar_config):
                current_window_end = (num_block + 1) * block_length
                block_x = x[:, :current_window_end]
                block_attn_mask = attn_mask[:, :current_window_end]
                block_position_ids = position_ids[:, :current_window_end]

                # Prompt positions inside this block are frozen. The scheduler is free to propose edits
                # there; the pipeline drops them, the way inpainting pipelines blend the original latents
                # back in by mask. This replaces the scheduler's old `prompt_mask` argument.
                block_start_pos = num_block * block_length
                editable = torch.ones((1, block_length), device=device, dtype=torch.bool)
                if block_start_pos < prompt_length:
                    editable[:, : min(prompt_length - block_start_pos, block_length)] = False

                # `step_index` advances on every `step`, so every block needs a fresh schedule; this also
                # clears whatever per-block state the scheduler keeps.
                self.scheduler.set_timesteps(num_inference_steps, device=device)

                inner_progress_bar_config = {
                    **outer_progress_bar_config,
                    "position": 1,
                    "leave": False,
                    "desc": f"Block {num_block} Inference Steps",
                }
                progress_bar = tqdm(total=num_inference_steps, **inner_progress_bar_config)

                masks_cleared = False

                # --- Mask-filling phase: one step per entry of the schedule ---
                for t in self.scheduler.timesteps:
                    # Kept for the editing phase, which has no timestep of its own to report to a callback.
                    timestep = t
                    block_tokens = block_x[:, -block_length:]
                    active_block = block_tokens == mask_token_id

                    logits = self.model(
                        block_x, attention_mask=block_attn_mask, position_ids=block_position_ids
                    ).logits
                    scheduler_output = self.scheduler.step(
                        logits[:, -block_length:, :],
                        t,
                        block_tokens,
                        generator=generator,
                        return_dict=True,
                    )

                    edited_mask, final_transfer = self._resolve_transfer(
                        scheduler_output, editable, finished, eos_early_stop
                    )
                    if final_transfer.any():
                        block_x[:, -block_length:] = torch.where(
                            final_transfer, scheduler_output.prev_sample, block_tokens
                        )

                    if eos_early_stop and eos_token_id is not None:
                        finished = self._update_finished(
                            cur_x=block_x,
                            pred_original_sample=scheduler_output.pred_original_sample,
                            final_transfer=final_transfer,
                            finished=finished,
                            eos_token_id=eos_token_id,
                            mask_token_id=mask_token_id,
                            prompt_length=prompt_length,
                        )

                    if callback_on_step_end is not None:
                        block_x = self._run_step_callback(
                            callback_on_step_end,
                            callback_on_step_end_tensor_inputs,
                            step=global_step,
                            timestep=t,
                            block_x=block_x,
                            active_block=active_block,
                            scheduler_output=scheduler_output,
                            edited_mask=edited_mask,
                        )

                    global_step += 1
                    progress_bar.update(1)

                    if finished.all():
                        break
                    if not (block_x[:, -block_length:] == mask_token_id).any():
                        masks_cleared = True
                        break

                # --- Editing phase: with every position resolved there is no unmasking left to schedule, so
                # this sweeps for confident overwrites instead. `step_edit` takes no timestep and does not
                # consume the schedule, which is what lets the phase run past `num_inference_steps`.
                if editing_enabled and masks_cleared:
                    post_steps = 0
                    while post_steps <= max_post_steps and not finished.all():
                        block_tokens = block_x[:, -block_length:]
                        active_block = block_tokens == mask_token_id

                        logits = self.model(
                            block_x, attention_mask=block_attn_mask, position_ids=block_position_ids
                        ).logits
                        scheduler_output = self.scheduler.step_edit(
                            logits[:, -block_length:, :],
                            block_tokens,
                            generator=generator,
                            return_dict=True,
                        )

                        edited_mask, final_transfer = self._resolve_transfer(
                            scheduler_output, editable, finished, eos_early_stop
                        )
                        if final_transfer.any():
                            block_x[:, -block_length:] = torch.where(
                                final_transfer, scheduler_output.prev_sample, block_tokens
                            )

                        if eos_early_stop and eos_token_id is not None:
                            finished = self._update_finished(
                                cur_x=block_x,
                                pred_original_sample=scheduler_output.pred_original_sample,
                                final_transfer=final_transfer,
                                finished=finished,
                                eos_token_id=eos_token_id,
                                mask_token_id=mask_token_id,
                                prompt_length=prompt_length,
                            )

                        if callback_on_step_end is not None:
                            block_x = self._run_step_callback(
                                callback_on_step_end,
                                callback_on_step_end_tensor_inputs,
                                step=global_step,
                                timestep=timestep,
                                block_x=block_x,
                                active_block=active_block,
                                scheduler_output=scheduler_output,
                                edited_mask=edited_mask,
                            )

                        global_step += 1
                        post_steps += 1

                        if not edited_mask.any():
                            break

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
