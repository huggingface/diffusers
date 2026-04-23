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
from transformers import DynamicCache

from ...callbacks import MultiPipelineCallbacks, PipelineCallback
from ...schedulers import IDLMBlockDiffusionScheduler
from ...utils import BaseOutput, logging, replace_example_docstring
from ..pipeline_utils import DiffusionPipeline, DiscreteDiffusionPipelineMixin


logger = logging.get_logger(__name__)


EXAMPLE_DOC_STRING = """
    Examples:
        ```python
        >>> import torch
        >>> from transformers import AutoModelForCausalLM, AutoTokenizer
        >>> from diffusers import IDLMPipeline

        >>> model_id = "yifanyu/I-DLM-8B"
        >>> model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, dtype=torch.bfloat16)
        >>> tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

        >>> pipe = IDLMPipeline(model=model, tokenizer=tokenizer)
        >>> out = pipe(prompt="Prove that sqrt(2) is irrational.", max_new_tokens=128)
        >>> print(out.texts[0])
        ```
"""


@dataclass
class IDLMPipelineOutput(BaseOutput):
    sequences: torch.LongTensor
    texts: list[str] | None = None


class IDLMPipeline(DiffusionPipeline, DiscreteDiffusionPipelineMixin):
    r"""
    Introspective Diffusion Language Model (I-DLM) pipeline.

    Each round is a single target-model forward over `[pending, spec_0, ..., spec_{K-1}, MASK, ..., MASK]` (length
    `2*gen_block_size - 1`) under strict causal attention. Under I-DLM's Dream-style logit shift, `logits[:, i, :]`
    predicts the token at input position `i+1`; this lets the same forward both *verify* the pending specs (via `min(1,
    p/(alpha*q))` against stored proposal probs) and *sample* the next batch of specs from the MASK-position anchors.
    On partial accept, the corrected token seeds a cold-start next round. See `IDLMBlockDiffusionScheduler.step` for
    the verify/resample math.

    The model is expected to be a standard causal LM (Qwen3-family for the published checkpoints). No custom
    attention-mask construction is needed — the pipeline relies on the model's default causal mask.
    """

    model: Any
    scheduler: IDLMBlockDiffusionScheduler
    tokenizer: Any

    _callback_tensor_inputs = ["committed_tokens", "accepted_length", "next_pending", "next_specs"]

    def __init__(
        self,
        model: Any,
        scheduler: IDLMBlockDiffusionScheduler | None = None,
        tokenizer: Any | None = None,
    ):
        super().__init__()
        if scheduler is None:
            scheduler = IDLMBlockDiffusionScheduler()
        self.register_modules(model=model, tokenizer=tokenizer, scheduler=scheduler)

    def check_inputs(
        self,
        prompt,
        messages,
        input_ids,
        max_new_tokens: int,
        mask_token_id: int | None,
        output_type: str,
        callback_on_step_end,
        callback_on_step_end_tensor_inputs,
    ):
        if prompt is None and messages is None and input_ids is None:
            raise ValueError("Provide one of `prompt`, `messages`, or `input_ids`.")
        if prompt is not None and messages is not None:
            raise ValueError("Provide either `prompt` or `messages`, not both.")
        if input_ids is not None:
            if input_ids.ndim not in (1, 2):
                raise ValueError(f"`input_ids` must be 1D or 2D, got shape {tuple(input_ids.shape)}.")
            if input_ids.dtype != torch.long:
                raise ValueError(f"`input_ids` must be int64 token IDs, got dtype={input_ids.dtype}.")
        if max_new_tokens <= 0:
            raise ValueError(f"`max_new_tokens` must be > 0, got {max_new_tokens}.")
        if output_type not in {"seq", "text"}:
            raise ValueError(f"`output_type` must be 'seq' or 'text', got {output_type!r}.")
        if mask_token_id is None:
            raise ValueError("`mask_token_id` must be provided (via tokenizer.mask_token_id or argument).")
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
        max_new_tokens: int = 256,
        gen_block_size: int | None = None,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        verify_alpha: float = 1.0,
        stop_token_ids: list[int] | None = None,
        mask_token_id: int | None = None,
        use_chat_template: bool = True,
        add_generation_prompt: bool = True,
        chat_template_kwargs: dict[str, object] | None = None,
        generator: torch.Generator | None = None,
        output_type: str = "text",
        return_dict: bool = True,
        callback_on_step_end: Callable[[int, int, dict], None]
        | PipelineCallback
        | MultiPipelineCallbacks
        | None = None,
        callback_on_step_end_tensor_inputs: list[str] | None = None,
    ) -> IDLMPipelineOutput | tuple[torch.LongTensor, list[str] | None]:
        """
        Generate via I-DLM introspective strided decoding (ISD).

        Args:
            prompt (`str`, *optional*): Prompt text. Wrapped in the tokenizer's chat template when
                `use_chat_template` is `True`.
            messages (`list[dict]`, *optional*): Chat messages; takes precedence over `prompt`.
            input_ids (`torch.LongTensor`, *optional*): Pre-tokenized input; takes precedence over the above.
            max_new_tokens (`int`): Upper bound on tokens emitted after the prompt.
            gen_block_size (`int`, *optional*): Override the scheduler's `gen_block_size` (N). Each round
                commits at most `N` tokens.
            temperature, top_k, top_p: Sampling params for proposal and resample distributions.
            verify_alpha (`float`): Leniency in the `min(1, p/(alpha*q))` accept criterion.
            stop_token_ids (`list[int]`, *optional*): IDs that terminate generation.
            mask_token_id (`int`, *optional*): MASK token ID. Defaults to `tokenizer.mask_token_id`.
            use_chat_template (`bool`), add_generation_prompt (`bool`), chat_template_kwargs (`dict`):
                Chat-template controls (same semantics as transformers).
            generator (`torch.Generator`, *optional*): RNG for sampling.
            output_type (`str`): `"text"` (decoded strings) or `"seq"` (token IDs only).
            return_dict (`bool`): Return `IDLMPipelineOutput` or a tuple.
            callback_on_step_end (`Callable`), callback_on_step_end_tensor_inputs (`list[str]`):
                Per-round callback hook.

        Examples:
        """
        if callback_on_step_end is not None and isinstance(
            callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)
        ):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs
        if callback_on_step_end_tensor_inputs is None:
            callback_on_step_end_tensor_inputs = ["committed_tokens"]

        if mask_token_id is None:
            mask_token_id = getattr(getattr(self, "tokenizer", None), "mask_token_id", None)
            if mask_token_id is None:
                cfg = getattr(self.model, "config", None)
                mask_token_id = getattr(cfg, "mask_token_id", None)

        self.check_inputs(
            prompt=prompt,
            messages=messages,
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            mask_token_id=mask_token_id,
            output_type=output_type,
            callback_on_step_end=callback_on_step_end,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        )

        # Allow per-call override of gen_block_size (recomputes block_size via the scheduler property).
        if gen_block_size is not None and int(gen_block_size) != int(self.scheduler.config.gen_block_size):
            self.scheduler.register_to_config(gen_block_size=int(gen_block_size))

        input_ids = self._prepare_input_ids(
            prompt=prompt,
            messages=messages,
            input_ids=input_ids,
            use_chat_template=use_chat_template,
            add_generation_prompt=add_generation_prompt,
            chat_template_kwargs=chat_template_kwargs,
        )
        device = self._execution_device
        input_ids = input_ids.to(device=device)
        if input_ids.shape[0] != 1:
            raise ValueError("IDLMPipeline currently supports batch_size=1 input_ids.")

        if stop_token_ids is None:
            eos_token_id = getattr(getattr(self, "tokenizer", None), "eos_token_id", None)
            stop_token_ids = [int(eos_token_id)] if eos_token_id is not None else None

        self.model.eval()

        prompt_length = input_ids.shape[1]
        mask_id = int(mask_token_id)
        num_masks = self.scheduler.num_masks
        N = int(self.scheduler.config.gen_block_size)

        # Cache for the committed prefix. We append one block per round and then crop back to the committed
        # length after each round so MASK / rejected-spec KVs don't leak into subsequent forwards.
        past_key_values = DynamicCache()

        # 1. Prefill the prompt (length `prompt_length`) into the cache. Its last logit under Dream shift
        #    predicts the first committed token (`pending_0`).
        prompt_position_ids = torch.arange(prompt_length, device=device).unsqueeze(0)
        # Rely on the model's built-in strict-causal path (`config.use_regular_causal=True`) — no need
        # to pass a per-token attention_mask; the model constructs an offset-aware causal mask internally.
        prefill_out = self.model(
            input_ids=input_ids,
            position_ids=prompt_position_ids,
            past_key_values=past_key_values,
            use_cache=True,
        )
        pending_logits = prefill_out.logits[0, -1:, :]  # [1, vocab]
        pending_ids, _ = self.scheduler.sample(
            pending_logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            generator=generator,
        )
        pending: int = int(pending_ids[0].item())

        # 2. ISD decoding loop: each round extends the sequence by 1 .. N committed tokens.
        generated: list[int] = []
        spec_tokens: list[int] = []
        spec_draft_probs: torch.Tensor | None = None

        global_step = 0
        progress_bar_config = getattr(self, "_progress_bar_config", {}).copy()
        progress_bar_config["desc"] = "ISD rounds"
        pbar = tqdm(total=max_new_tokens, **progress_bar_config)

        while len(generated) < max_new_tokens:
            # Build round input: [pending, *specs, MASK * num_masks]. Length K + 1 + num_masks, where K is 0
            # on cold-start and len(spec_tokens) on verify rounds.
            K = len(spec_tokens)
            round_tokens = [pending] + list(spec_tokens) + [mask_id] * num_masks
            round_input = torch.tensor([round_tokens], dtype=torch.long, device=device)
            committed_so_far = len(generated)
            base = prompt_length + committed_so_far
            round_pos = torch.arange(base, base + round_input.shape[1], device=device).unsqueeze(0)

            out = self.model(
                input_ids=round_input,
                position_ids=round_pos,
                past_key_values=past_key_values,
                use_cache=True,
            )

            step_output = self.scheduler.step(
                model_output=out.logits,
                timestep=global_step,
                pending=pending,
                spec_tokens=spec_tokens,
                spec_draft_probs=spec_draft_probs,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                verify_alpha=verify_alpha,
                generator=generator,
                return_dict=True,
            )
            committed_tokens = step_output.committed_tokens
            accepted_length = step_output.accepted_length
            next_pending = step_output.next_pending
            next_specs = step_output.next_specs

            # Commit this round's tokens (truncated to max_new_tokens).
            num_to_commit = min(int(committed_tokens.shape[0]), max_new_tokens - len(generated))
            generated.extend(committed_tokens[:num_to_commit].tolist())

            # Crop cache back to the committed length. The forward advanced it by `1 + K + num_masks`
            # positions; we keep only positions for the `num_to_commit` newly committed tokens.
            past_key_values.crop(prompt_length + len(generated))

            pending = next_pending
            spec_tokens = next_specs
            spec_draft_probs = step_output.next_draft_probs

            if callback_on_step_end is not None:
                callback_kwargs = {}
                for k in callback_on_step_end_tensor_inputs:
                    callback_kwargs[k] = locals().get(k)
                callback_on_step_end(self, global_step, 0, callback_kwargs)

            global_step += 1
            pbar.update(num_to_commit)

            # Stop-token check against the committed sequence so far.
            if stop_token_ids is not None and any(tok in generated for tok in stop_token_ids):
                # Truncate generated at the first stop token (inclusive).
                for idx, tok in enumerate(generated):
                    if tok in stop_token_ids:
                        generated = generated[: idx + 1]
                        break
                break

        pbar.close()

        sequences = torch.tensor([generated], dtype=torch.long, device=device)
        texts = None
        if output_type == "text" and self.tokenizer is not None:
            texts = self.tokenizer.batch_decode(sequences, skip_special_tokens=True)

        if not return_dict:
            return sequences, texts
        return IDLMPipelineOutput(sequences=sequences, texts=texts)
