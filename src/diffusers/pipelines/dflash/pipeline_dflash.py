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
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, DynamicCache

from ...callbacks import MultiPipelineCallbacks, PipelineCallback
from ...schedulers import DFlashTokenDiffusionScheduler
from ...utils import BaseOutput, logging, replace_example_docstring
from ..pipeline_utils import DiffusionPipeline, DiscreteDiffusionPipelineMixin


logger = logging.get_logger(__name__)


EXAMPLE_DOC_STRING = """
    Examples:
        ```python
        >>> import torch
        >>> from diffusers import DFlashPipeline

        >>> draft_id = "z-lab/Qwen3-8B-DFlash-b16"
        >>> target_id = "Qwen/Qwen3-8B"
        >>> pipe = DFlashPipeline.from_pretrained(
        ...     draft_model_id=draft_id,
        ...     target_model_id=target_id,
        ...     draft_model_kwargs={"trust_remote_code": True, "dtype": torch.bfloat16},
        ...     target_model_kwargs={"dtype": torch.bfloat16},
        ... )
        >>> out = pipe(prompt="How many positive whole-number divisors does 196 have?")
        >>> print(out.texts[0])
        ```
"""


@dataclass
class DFlashPipelineOutput(BaseOutput):
    sequences: torch.LongTensor
    texts: list[str] | None = None


def _build_target_layer_ids(num_target_layers: int, num_draft_layers: int) -> list[int]:
    if num_draft_layers == 1:
        return [int(num_target_layers // 2)]
    start = 1
    end = int(num_target_layers) - 3
    span = end - start
    return [int(round(start + (i * span) / (num_draft_layers - 1))) for i in range(int(num_draft_layers))]


def _extract_context_feature(hidden_states: list[torch.Tensor], layer_ids: list[int]) -> torch.Tensor:
    offset = 1
    selected_states = [hidden_states[layer_id + offset] for layer_id in layer_ids]
    return torch.cat(selected_states, dim=-1)


class DFlashPipeline(DiffusionPipeline, DiscreteDiffusionPipelineMixin):
    r"""
    Block diffusion pipeline for speculative decoding with a DFlash draft model and a target causal LM.
    """

    draft_model: Any
    target_model: Any
    tokenizer: Any
    scheduler: DFlashTokenDiffusionScheduler
    _callback_tensor_inputs = ["block_output_ids", "draft_logits", "accepted_length", "next_token", "output_ids"]

    def __init__(
        self,
        draft_model: torch.nn.Module,
        target_model: torch.nn.Module,
        tokenizer: Any | None = None,
        scheduler: DFlashTokenDiffusionScheduler | None = None,
    ):
        super().__init__()
        if scheduler is None:
            scheduler = DFlashTokenDiffusionScheduler()
        self.register_modules(
            draft_model=draft_model, target_model=target_model, tokenizer=tokenizer, scheduler=scheduler
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | None = None,
        *,
        draft_model_id: str | None = None,
        target_model_id: str | None = None,
        tokenizer_id: str | None = None,
        mask_token: str | None = "<|MASK|>",
        scheduler: DFlashTokenDiffusionScheduler | None = None,
        draft_model_kwargs: dict[str, object] | None = None,
        target_model_kwargs: dict[str, object] | None = None,
        tokenizer_kwargs: dict[str, object] | None = None,
        **pipeline_kwargs,
    ) -> "DFlashPipeline":
        if draft_model_id is None and target_model_id is None and pretrained_model_name_or_path is not None:
            return super().from_pretrained(pretrained_model_name_or_path, **pipeline_kwargs)

        if draft_model_id is None:
            if pretrained_model_name_or_path is None:
                raise ValueError("Provide `draft_model_id` or `pretrained_model_name_or_path`.")
            draft_model_id = str(pretrained_model_name_or_path)
        if target_model_id is None:
            raise ValueError("`target_model_id` must be provided when loading draft/target models separately.")

        draft_model_kwargs = dict(draft_model_kwargs or {})
        draft_model_kwargs.setdefault("trust_remote_code", True)
        target_model_kwargs = dict(target_model_kwargs or {})
        tokenizer_kwargs = dict(tokenizer_kwargs or {})

        draft = AutoModel.from_pretrained(draft_model_id, **draft_model_kwargs)
        target = AutoModelForCausalLM.from_pretrained(target_model_id, **target_model_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_id or target_model_id, **tokenizer_kwargs)

        if mask_token is not None and tokenizer.mask_token_id is None:
            tokenizer.add_special_tokens({"mask_token": mask_token})

        return cls(
            draft_model=draft,
            target_model=target,
            tokenizer=tokenizer,
            scheduler=scheduler,
            **pipeline_kwargs,
        )

    def check_inputs(
        self,
        prompt: str | list[str] | None,
        messages: list[dict[str, str]] | None,
        input_ids: torch.LongTensor | None,
        max_new_tokens: int,
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
        if max_new_tokens <= 0:
            raise ValueError(f"`max_new_tokens` must be > 0, got {max_new_tokens}.")
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
        max_length: int,
        block_size: int,
        mask_token_id: int,
        device: torch.device,
    ) -> torch.LongTensor:
        return torch.full(
            (1, max_length + int(block_size)),
            int(mask_token_id),
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
        max_new_tokens: int = 2048,
        temperature: float = 0.0,
        stop_token_ids: list[int] | None = None,
        mask_token_id: int | None = None,
        use_chat_template: bool = True,
        add_generation_prompt: bool = True,
        chat_template_kwargs: dict[str, object] | None = None,
        output_type: str = "text",
        return_dict: bool = True,
        callback_on_step_end: Callable[[int, int, dict], None]
        | PipelineCallback
        | MultiPipelineCallbacks
        | None = None,
        callback_on_step_end_tensor_inputs: list[str] | None = None,
    ) -> DFlashPipelineOutput | tuple[torch.LongTensor, list[str] | None]:
        """
        Generate text using block-diffusion speculative decoding.

        Args:
            prompt (`str` or `list[str]`, *optional*):
                Prompt text. When `use_chat_template` is `True` (default) and a tokenizer with a chat template is
                available, the prompt is wrapped in a chat message before tokenization.
            messages (`list[dict[str, str]]`, *optional*):
                Chat messages to encode. Takes precedence over `prompt` when provided.
            input_ids (`torch.LongTensor`, *optional*):
                Pre-tokenized input IDs. Takes precedence over `prompt` and `messages`.
            max_new_tokens (`int`):
                Maximum number of new tokens to generate.
            temperature (`float`):
                Sampling temperature.
            stop_token_ids (`list[int]`, *optional*):
                Token IDs that signal generation should stop.
            mask_token_id (`int`, *optional*):
                Mask token ID for the draft model.
            use_chat_template (`bool`, defaults to `True`):
                Whether to wrap the prompt in a chat template.
            add_generation_prompt (`bool`, defaults to `True`):
                Whether to add the generation prompt when using chat templates.
            chat_template_kwargs (`dict[str, object]`, *optional*):
                Additional keyword arguments for the chat template.
            output_type (`str`, defaults to `"text"`):
                Output format. `"text"` decodes sequences into strings (requires a tokenizer). `"seq"` returns raw
                token ID sequences only.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`DFlashPipelineOutput`] instead of a tuple.
            callback_on_step_end (`Callable` or `PipelineCallback`, *optional*):
                Callback executed after each speculative decoding step.
            callback_on_step_end_tensor_inputs (`list[str]`, *optional*):
                Tensor keys to pass to the callback.

        Examples:
        """
        # 1. Check inputs early
        if callback_on_step_end is not None and isinstance(
            callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)
        ):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs
        if callback_on_step_end_tensor_inputs is None:
            callback_on_step_end_tensor_inputs = ["block_output_ids"]

        self.check_inputs(
            prompt=prompt,
            messages=messages,
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
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

        if mask_token_id is None:
            mask_token_id = getattr(getattr(self, "tokenizer", None), "mask_token_id", None)
        if mask_token_id is None:
            raise ValueError("`mask_token_id` must be provided (or available on the tokenizer).")
        if input_ids.shape[0] != 1:
            raise ValueError("DFlashPipeline currently supports batch_size=1 input_ids.")

        target_params = list(self.target_model.parameters()) if hasattr(self.target_model, "parameters") else []
        device = target_params[0].device if len(target_params) > 0 else torch.device("cpu")
        input_ids = input_ids.to(device=device)
        draft_params = list(self.draft_model.parameters()) if hasattr(self.draft_model, "parameters") else []
        draft_device = draft_params[0].device if len(draft_params) > 0 else device
        if draft_device != device:
            logger.warning(
                "Draft model is on %s while target model is on %s. For best performance, place both on the same device.",
                draft_device,
                device,
            )

        if stop_token_ids is None:
            eos_token_id = getattr(getattr(self, "tokenizer", None), "eos_token_id", None)
            stop_token_ids = [int(eos_token_id)] if eos_token_id is not None else None
        if stop_token_ids is not None:
            stop_token_ids = [int(token_id) for token_id in stop_token_ids]

        # 3. Setup models and scheduler
        self.draft_model.eval()
        self.target_model.eval()
        self.scheduler.set_timesteps(1, device=device)

        block_size = self._get_block_size()
        target_layer_ids = self._get_target_layer_ids()
        input_embeddings = self._get_target_input_embeddings()
        output_embeddings = self._get_target_output_embeddings()

        num_input_tokens = input_ids.shape[1]
        max_length = num_input_tokens + int(max_new_tokens)

        output_ids = self.prepare_latents(max_length, block_size, int(mask_token_id), device)
        position_ids = torch.arange(output_ids.shape[1], device=device).unsqueeze(0)

        past_key_values_target = DynamicCache()
        past_key_values_draft = DynamicCache()

        # 4. Prefill step
        output = self._target_forward(
            input_ids=input_ids,
            position_ids=position_ids[:, :num_input_tokens],
            past_key_values=past_key_values_target,
            output_hidden_states=True,
            logits_to_keep=1,
        )
        output_ids[:, :num_input_tokens] = input_ids
        output_ids[:, num_input_tokens : num_input_tokens + 1] = self.scheduler.sample(
            output.logits[:, -1:], temperature=temperature
        )
        target_hidden = _extract_context_feature(output.hidden_states, target_layer_ids)

        start = num_input_tokens
        global_step = 0
        num_blocks = (max_length - num_input_tokens + block_size - 1) // block_size

        # 5. Block-wise speculative decoding loop
        block_progress_bar_config = getattr(self, "_progress_bar_config", {}).copy()
        block_progress_bar_config["position"] = 0
        block_progress_bar_config["desc"] = "Blocks"
        block_iter = tqdm(range(num_blocks), **block_progress_bar_config)

        for _block_idx in block_iter:
            if start >= max_length:
                break

            block_output_ids = output_ids[:, start : start + int(block_size)].clone()
            block_position_ids = position_ids[:, start : start + int(block_size)]
            noise_embedding = input_embeddings(block_output_ids)
            draft_hidden = self.draft_model(
                target_hidden=target_hidden,
                noise_embedding=noise_embedding,
                position_ids=position_ids[:, past_key_values_draft.get_seq_length() : start + int(block_size)],
                past_key_values=past_key_values_draft,
                use_cache=True,
                is_causal=False,
            )
            if not torch.is_tensor(draft_hidden):
                draft_hidden = getattr(draft_hidden, "last_hidden_state", draft_hidden[0])
            draft_logits = output_embeddings(draft_hidden[:, -int(block_size) + 1 :, :])
            past_key_values_draft.crop(start)
            block_output_ids[:, 1:] = self.scheduler.sample(draft_logits, temperature=temperature)

            output = self._target_forward(
                input_ids=block_output_ids,
                position_ids=block_position_ids,
                past_key_values=past_key_values_target,
                output_hidden_states=True,
                logits_to_keep=None,
            )
            step_output = self.scheduler.step(
                block_output_ids, output.logits, temperature=temperature, return_dict=True
            )
            accepted_length = step_output.accepted_length
            next_token = step_output.next_token
            acceptance_length = int(step_output.accepted_length[0].item())
            output_ids[:, start : start + acceptance_length + 1] = block_output_ids[:, : acceptance_length + 1]
            output_ids[:, start + acceptance_length + 1] = step_output.next_token
            start += acceptance_length + 1
            past_key_values_target.crop(start)
            target_hidden = _extract_context_feature(output.hidden_states, target_layer_ids)[
                :, : acceptance_length + 1, :
            ]

            if callback_on_step_end is not None:
                callback_kwargs = {}
                for k in callback_on_step_end_tensor_inputs:
                    callback_kwargs[k] = locals()[k]
                callback_outputs = callback_on_step_end(self, global_step, 0, callback_kwargs)
                output_ids = callback_outputs.pop("output_ids", output_ids)
                global_step += 1

            if self.scheduler.check_should_stop(output_ids, stop_token_ids, num_input_tokens):
                break

        # 6. Post-process output
        output_ids = output_ids[:, :max_length]
        output_ids = output_ids[:, output_ids[0] != int(mask_token_id)]
        if stop_token_ids is not None:
            stop_tensor = torch.tensor(stop_token_ids, device=device, dtype=torch.long)
            stop_positions = torch.isin(output_ids[0, num_input_tokens:], stop_tensor).nonzero(as_tuple=True)[0]
            if stop_positions.numel() > 0:
                output_ids = output_ids[:, : num_input_tokens + int(stop_positions[0].item()) + 1]

        prompt_len = input_ids.shape[1]
        sequences = output_ids[:, prompt_len:]

        texts = None
        if output_type == "text" and getattr(self, "tokenizer", None) is not None:
            texts = self.tokenizer.batch_decode(sequences, skip_special_tokens=True)

        if not return_dict:
            return sequences, texts
        return DFlashPipelineOutput(sequences=sequences, texts=texts)

    def _get_block_size(self) -> int:
        block_size = getattr(self.draft_model, "block_size", None)
        if block_size is None:
            block_size = getattr(getattr(self.draft_model, "config", None), "block_size", None)
        if block_size is None:
            raise ValueError("`draft_model` must define `block_size` on the module or its config.")
        return int(block_size)

    def _get_target_layer_ids(self) -> list[int]:
        layer_ids = getattr(self.draft_model, "target_layer_ids", None)
        if layer_ids is not None:
            return list(layer_ids)
        cfg = getattr(self.draft_model, "config", None)
        num_target_layers = getattr(cfg, "num_target_layers", None)
        num_hidden_layers = getattr(cfg, "num_hidden_layers", None)
        if num_target_layers is None or num_hidden_layers is None:
            raise ValueError("`draft_model` must define `target_layer_ids` or expose `num_target_layers` in config.")
        return _build_target_layer_ids(int(num_target_layers), int(num_hidden_layers))

    def _get_target_input_embeddings(self) -> torch.nn.Module:
        embeddings = self.target_model.get_input_embeddings()
        if embeddings is None:
            base_model = getattr(self.target_model, "model", None)
            embeddings = getattr(base_model, "embed_tokens", None)
        if embeddings is None:
            raise ValueError("`target_model` must provide input embeddings for DFlash decoding.")
        return embeddings

    def _get_target_output_embeddings(self) -> torch.nn.Module:
        embeddings = self.target_model.get_output_embeddings()
        if embeddings is None:
            embeddings = getattr(self.target_model, "lm_head", None)
        if embeddings is None:
            raise ValueError("`target_model` must provide output embeddings for DFlash decoding.")
        return embeddings

    def _target_forward(
        self,
        *,
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor,
        past_key_values: DynamicCache,
        output_hidden_states: bool,
        logits_to_keep: int | None,
    ):
        kwargs = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": True,
            "output_hidden_states": output_hidden_states,
        }
        if logits_to_keep is not None:
            try:
                return self.target_model(**kwargs, logits_to_keep=logits_to_keep)
            except TypeError:
                pass
        return self.target_model(**kwargs)


__all__ = ["DFlashPipeline", "DFlashPipelineOutput"]
