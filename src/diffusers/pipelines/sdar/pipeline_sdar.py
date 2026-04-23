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
from ...schedulers import SDARTokenDiffusionScheduler
from ...utils import BaseOutput, logging, replace_example_docstring
from ..pipeline_utils import DiffusionPipeline, DiscreteDiffusionPipelineMixin


logger = logging.get_logger(__name__)


EXAMPLE_DOC_STRING = """
    Examples:
        ```python
        >>> import torch
        >>> from transformers import AutoModelForCausalLM, AutoTokenizer
        >>> from diffusers import SDARPipeline

        >>> model_id = "JetLM/SDAR-1.7B-Chat"
        >>> model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, dtype=torch.bfloat16)
        >>> tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        >>> tokenizer.add_special_tokens({"mask_token": "<|MASK|>"})

        >>> pipe = SDARPipeline(model=model, tokenizer=tokenizer)
        >>> out = pipe(prompt="Explain what reinforcement learning is in simple terms.")
        >>> print(out.texts[0])
        ```
"""


@dataclass
class SDARPipelineOutput(BaseOutput):
    sequences: torch.LongTensor
    texts: list[str] | None = None


class SDARPipeline(DiffusionPipeline, DiscreteDiffusionPipelineMixin):
    r"""
    Block diffusion pipeline for SDAR-style token generation.

    This pipeline generates text by processing blocks of tokens in a semi-autoregressive fashion. Each block is
    iteratively denoised using a masked diffusion process, where tokens are progressively revealed based on model
    confidence.

    The model is expected to accept an attention mask and `position_ids`, and to return logits of shape `[batch, seq,
    vocab_size]`.
    """

    model: Any
    scheduler: SDARTokenDiffusionScheduler
    tokenizer: Any

    _callback_tensor_inputs = ["block_x", "logits", "sampled_tokens", "sampled_probs", "transfer_index"]

    def __init__(
        self,
        model: Any,
        scheduler: SDARTokenDiffusionScheduler | None = None,
        tokenizer: Any | None = None,
    ):
        super().__init__()
        if scheduler is None:
            scheduler = SDARTokenDiffusionScheduler()
        self.register_modules(model=model, tokenizer=tokenizer, scheduler=scheduler)

    @property
    def num_timesteps(self):
        return self._num_timesteps

    def check_inputs(
        self,
        prompt: str | list[str] | None,
        messages: list[dict[str, str]] | None,
        input_ids: torch.LongTensor | None,
        block_length: int,
        num_inference_steps: int,
        mask_token_id: int | None,
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
            if input_ids.ndim == 2 and input_ids.shape[0] != 1:
                raise ValueError("SDARPipeline currently supports batch_size=1 input_ids.")
        if prompt is not None and input_ids is None and self.tokenizer is None:
            raise ValueError("Tokenizer is required when `input_ids` is not provided.")
        if messages is not None and input_ids is None and self.tokenizer is None:
            raise ValueError("Tokenizer is required when `input_ids` is not provided.")

        # Generation parameter validation
        if block_length <= 0:
            raise ValueError(f"`block_length` must be > 0, got {block_length}.")
        if num_inference_steps <= 0:
            raise ValueError(f"`num_inference_steps` must be > 0, got {num_inference_steps}.")
        if mask_token_id is None:
            raise ValueError("`mask_token_id` must be provided (or available on the tokenizer).")
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

    def prepare_latents(
        self,
        total_length: int,
        mask_token_id: int,
        device: torch.device,
    ) -> torch.LongTensor:
        return torch.full(
            (1, total_length),
            mask_token_id,
            dtype=torch.long,
            device=device,
        )

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: str | list[str] | None = None,
        messages: list[dict[str, str]] | None = None,
        input_ids: torch.LongTensor | None = None,
        max_new_tokens: int = 256,
        block_length: int = 4,
        num_inference_steps: int = 4,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        remasking_strategy: str = "low_confidence_dynamic",
        confidence_threshold: float = 0.9,
        entropy_threshold: float = 0.35,
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
    ) -> SDARPipelineOutput | tuple[torch.LongTensor, list[str] | None]:
        """
        Generate text using SDAR-style block diffusion decoding.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                Prompt text. When `use_chat_template` is `True` (default) and a tokenizer with a chat template is
                available, the prompt is wrapped in a chat message before tokenization.
            messages (`List[Dict[str, str]]`, *optional*):
                Chat messages to encode (e.g. `[{"role": "user", "content": "Hello"}]`). Takes precedence over `prompt`
                when provided. Requires a tokenizer with `apply_chat_template`.
            input_ids (`torch.LongTensor`, *optional*):
                Pre-tokenized input IDs. Takes precedence over `prompt` and `messages`.
            max_new_tokens (`int`):
                Number of tokens to generate.
            block_length (`int`):
                Block size for denoising.
            num_inference_steps (`int`):
                Number of denoising steps per block.
            temperature (`float`):
                Sampling temperature.
            top_k (`int`):
                Top-k sampling cutoff.
            top_p (`float`):
                Nucleus sampling cutoff.
            remasking_strategy (`str`):
                Strategy for selecting which tokens to commit (`sequential`, `low_confidence_static`,
                `low_confidence_dynamic`, `entropy_bounded`).
            confidence_threshold (`float`):
                Confidence threshold for dynamic remasking.
            entropy_threshold (`float`):
                Entropy threshold for entropy-bounded remasking.
            stop_token_ids (`list[int]`, *optional*):
                Token IDs that signal generation should stop.
            mask_token_id (`int`, *optional*):
                Mask token ID to use for the template.
            use_chat_template (`bool`, defaults to `True`):
                Whether to wrap the prompt in a chat template.
            add_generation_prompt (`bool`, defaults to `True`):
                Whether to add the generation prompt when using chat templates.
            chat_template_kwargs (`dict`, *optional*):
                Extra kwargs for `apply_chat_template`.
            generator (`torch.Generator`, *optional*):
                RNG for sampling.
            output_type (`str`, defaults to `"text"`):
                Output format. `"text"` decodes sequences into strings (requires a tokenizer). `"seq"` returns raw
                token ID sequences only.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`SDARPipelineOutput`] instead of a tuple.
            callback_on_step_end (`Callable` or `PipelineCallback`, *optional*):
                Callback executed after each denoising step.
            callback_on_step_end_tensor_inputs (`List[str]`, *optional*):
                Tensor keys to pass to the callback.

        Examples:
        """
        # 1. Check inputs early
        if callback_on_step_end is not None and isinstance(
            callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)
        ):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs
        if callback_on_step_end_tensor_inputs is None:
            callback_on_step_end_tensor_inputs = ["block_x"]

        # Resolve block_length from model if not explicitly overridden by the user
        model_block_length = getattr(self.model, "block_length", None)
        if model_block_length is None:
            model_block_length = getattr(getattr(self.model, "config", None), "block_length", None)
        if model_block_length is not None:
            block_length = model_block_length

        if mask_token_id is None:
            mask_token_id = getattr(getattr(self, "tokenizer", None), "mask_token_id", None)

        self.check_inputs(
            prompt=prompt,
            messages=messages,
            input_ids=input_ids,
            block_length=block_length,
            num_inference_steps=num_inference_steps,
            mask_token_id=mask_token_id,
            output_type=output_type,
            callback_on_step_end=callback_on_step_end,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        )

        # 2. Prepare input IDs from prompt/messages/input_ids
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

        if stop_token_ids is None:
            eos_token_id = getattr(getattr(self, "tokenizer", None), "eos_token_id", None)
            stop_token_ids = [eos_token_id] if eos_token_id is not None else None

        self.model.eval()
        self.scheduler.set_timesteps(num_inference_steps, device=device)

        prompt_length = input_ids.shape[1]
        num_blocks = (prompt_length + max_new_tokens + block_length - 1) // block_length
        total_length = num_blocks * block_length

        # 3. Build 2D attention mask — the model handles backend-specific conversion internally.
        attn_mask = self._build_block_attention_mask_2d(
            num_blocks=num_blocks,
            block_length=block_length,
            total_length=total_length,
            device=device,
        )

        x = self.prepare_latents(total_length, mask_token_id, device)
        x[:, :prompt_length] = input_ids

        position_ids = torch.arange(total_length, device=device).unsqueeze(0)
        past_key_values = DynamicCache()

        prefill_blocks = prompt_length // block_length
        prefill_length = prefill_blocks * block_length

        self._num_timesteps = num_inference_steps * max(num_blocks - prefill_blocks, 0)

        if prefill_length > 0:
            block_x = x[:, :prefill_length]
            block_position_ids = position_ids[:, :prefill_length]
            block_attn_mask = attn_mask[:prefill_length, :prefill_length].unsqueeze(0)
            self._model_forward_logits(
                input_ids=block_x,
                attention_mask=block_attn_mask,
                position_ids=block_position_ids,
                past_key_values=past_key_values,
                store_kv=True,
            )

        num_transfer_tokens = self.scheduler.get_num_transfer_tokens(block_length, num_inference_steps).to(
            device=device
        )

        global_step = 0

        # 4. Block-wise generation loop
        block_progress_bar_config = getattr(self, "_progress_bar_config", {}).copy()
        block_progress_bar_config["position"] = 0
        block_progress_bar_config["desc"] = "Blocks"
        for block_idx in tqdm(range(prefill_blocks, num_blocks), **block_progress_bar_config):
            start = block_idx * block_length
            end = start + block_length
            block_x = x[:, start:end].clone()
            block_position_ids = position_ids[:, start:end]
            block_attn_mask = attn_mask[start:end, :end].unsqueeze(0)

            self.set_progress_bar_config(position=1, leave=False, desc=f"Block {block_idx} Inference Steps")
            progress_bar = self.progress_bar(total=num_inference_steps)

            for step in range(num_inference_steps + 1):
                mask_index = block_x == mask_token_id
                if mask_index.sum() == 0:
                    self._model_forward_logits(
                        input_ids=block_x,
                        attention_mask=block_attn_mask,
                        position_ids=block_position_ids,
                        past_key_values=past_key_values,
                        store_kv=True,
                    )
                    break

                logits = self._model_forward_logits(
                    input_ids=block_x,
                    attention_mask=block_attn_mask,
                    position_ids=block_position_ids,
                    past_key_values=past_key_values,
                    store_kv=False,
                )

                step_output = self.scheduler.step(
                    logits,
                    step,
                    block_x,
                    mask_token_id=mask_token_id,
                    num_transfer_tokens=num_transfer_tokens,
                    remasking_strategy=remasking_strategy,
                    confidence_threshold=confidence_threshold,
                    entropy_threshold=entropy_threshold,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    generator=generator,
                    return_dict=True,
                )
                block_x = step_output.prev_sample
                transfer_index = step_output.transfer_index
                sampled_tokens = step_output.sampled_tokens
                sampled_probs = step_output.sampled_probs

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, global_step, step, callback_kwargs)
                    block_x = callback_outputs.pop("block_x", block_x)

                global_step += 1
                progress_bar.update(1)

            progress_bar.close()
            x[:, start:end] = block_x

            if self.scheduler.check_should_stop(x, prompt_length, stop_token_ids):
                break

        # 5. Post-process output
        output_ids = x[:, : prompt_length + max_new_tokens]
        if stop_token_ids is not None:
            stop_tensor = torch.tensor(stop_token_ids, device=device, dtype=torch.long)
            stop_positions = torch.isin(output_ids[0, prompt_length:], stop_tensor).nonzero(as_tuple=True)[0]
            if stop_positions.numel() > 0:
                output_ids = output_ids[:, : prompt_length + stop_positions[0].item() + 1]

        if output_ids.shape[0] == 1:
            output_ids = output_ids[:, output_ids[0] != mask_token_id]

        sequences = output_ids[:, prompt_length:]
        texts = None
        if output_type == "text" and self.tokenizer is not None:
            texts = self.tokenizer.batch_decode(sequences, skip_special_tokens=True)

        if not return_dict:
            return sequences, texts
        return SDARPipelineOutput(sequences=sequences, texts=texts)

    def _model_forward_logits(
        self,
        *,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        position_ids: torch.LongTensor,
        past_key_values: DynamicCache,
        store_kv: bool,
    ) -> torch.Tensor:
        """Run the model forward pass and return logits.

        Follows the transformers v5 cache convention: `use_cache=True + past_key_values` always commits the appended
        KVs to the cache. When `store_kv=False` (the SDAR denoising inner loop, which wants logits without growing the
        cache), we snapshot the pre-forward seq length and `.crop()` back after the forward so the append is
        effectively undone.
        """
        prev_seq_len = past_key_values.get_seq_length() if past_key_values is not None else 0
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=True,
        )
        if not store_kv and past_key_values is not None:
            past_key_values.crop(prev_seq_len)
        return output.logits if hasattr(output, "logits") else output[0]

    def _build_block_attention_mask_2d(
        self,
        *,
        num_blocks: int,
        block_length: int,
        total_length: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Build a 2D block-causal attention mask of shape `(total_length, total_length)`.

        Each position can attend to all positions in the same or earlier blocks.
        """
        block_mask = torch.tril(torch.ones(num_blocks, num_blocks, device=device, dtype=torch.long))
        attn = block_mask.repeat_interleave(block_length, dim=0).repeat_interleave(block_length, dim=1)
        return attn[:total_length, :total_length]


__all__ = ["SDARPipeline", "SDARPipelineOutput"]
