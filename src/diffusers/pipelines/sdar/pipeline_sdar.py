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
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
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
    texts: Optional[List[str]] = None


class SDARPipeline(DiffusionPipeline, DiscreteDiffusionPipelineMixin):
    r"""
    Block diffusion pipeline for SDAR-style token generation.
    """

    model: torch.nn.Module
    tokenizer: Optional[object]
    scheduler: SDARTokenDiffusionScheduler
    _callback_tensor_inputs = ["cur_x", "logits", "sampled_tokens", "sampled_probs", "transfer_index"]

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Optional[object] = None,
        scheduler: Optional[SDARTokenDiffusionScheduler] = None,
    ):
        super().__init__()
        if scheduler is None:
            scheduler = SDARTokenDiffusionScheduler()
        self.register_modules(model=model, tokenizer=tokenizer, scheduler=scheduler)
        self._store_kv_supported: Optional[bool] = None

    def check_inputs(
        self,
        input_ids: torch.LongTensor,
        block_length: int,
        denoising_steps: int,
        mask_token_id: Optional[int],
        callback_on_step_end: Optional[Union[Callable, PipelineCallback, MultiPipelineCallbacks]],
        callback_on_step_end_tensor_inputs: Optional[List[str]],
    ):
        if input_ids.shape[0] != 1:
            raise ValueError("SDARPipeline currently supports batch_size=1 input_ids.")
        if block_length <= 0:
            raise ValueError(f"`block_length` must be > 0, got {block_length}.")
        if denoising_steps <= 0:
            raise ValueError(f"`denoising_steps` must be > 0, got {denoising_steps}.")
        if mask_token_id is None:
            raise ValueError("`mask_token_id` must be provided (or available on the tokenizer).")
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
            int(mask_token_id),
            dtype=torch.long,
            device=device,
        )

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        input_ids: Optional[torch.LongTensor] = None,
        max_new_tokens: int = 256,
        block_length: int = 4,
        denoising_steps: int = 4,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        remasking_strategy: str = "low_confidence_dynamic",
        confidence_threshold: float = 0.9,
        entropy_threshold: float = 0.35,
        stop_token_ids: Optional[List[int]] = None,
        mask_token_id: Optional[int] = None,
        attention_mask_mode: str = "3d",
        use_chat_template: bool = True,
        add_generation_prompt: bool = True,
        chat_template_kwargs: Optional[Dict[str, object]] = None,
        return_text: bool = True,
        return_dict: bool = True,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: Optional[List[str]] = None,
    ) -> Union[SDARPipelineOutput, Tuple[torch.LongTensor, Optional[List[str]]]]:
        """
        Generate text using SDAR-style block diffusion decoding.

        Examples:
        """
        if callback_on_step_end is not None and isinstance(
            callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)
        ):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs
        if callback_on_step_end_tensor_inputs is None:
            callback_on_step_end_tensor_inputs = ["cur_x"]

        # Resolve block_length from model if not explicitly overridden by the user
        model_block_length = getattr(self.model, "block_length", None)
        if model_block_length is None:
            model_block_length = getattr(getattr(self.model, "config", None), "block_length", None)
        if model_block_length is not None:
            block_length = int(model_block_length)

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

        self.check_inputs(
            input_ids=input_ids,
            block_length=block_length,
            denoising_steps=denoising_steps,
            mask_token_id=mask_token_id,
            callback_on_step_end=callback_on_step_end,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        )

        params = list(self.model.parameters()) if hasattr(self.model, "parameters") else []
        device = params[0].device if len(params) > 0 else torch.device("cpu")
        input_ids = input_ids.to(device=device)

        if stop_token_ids is None:
            eos_token_id = getattr(getattr(self, "tokenizer", None), "eos_token_id", None)
            stop_token_ids = [int(eos_token_id)] if eos_token_id is not None else None
        if stop_token_ids is not None:
            stop_token_ids = [int(token_id) for token_id in stop_token_ids]

        self.model.eval()
        self.scheduler.set_timesteps(int(denoising_steps), device=device)

        prompt_length = input_ids.shape[1]
        num_blocks = (prompt_length + int(max_new_tokens) + int(block_length) - 1) // int(block_length)
        total_length = int(num_blocks) * int(block_length)

        block_mask_3d = self._build_block_attention_mask_3d(
            num_blocks=num_blocks,
            block_length=block_length,
            total_length=total_length,
            device=device,
            dtype=torch.float32,
        )
        block_mask_4d = self._build_block_attention_mask_4d(block_mask_3d, dtype=torch.float32)
        block_mask_2d = block_mask_3d[0]

        x = self.prepare_latents(total_length, int(mask_token_id), device)
        x[:, :prompt_length] = input_ids

        position_ids = torch.arange(total_length, device=device).unsqueeze(0)
        past_key_values = DynamicCache()

        prefill_blocks = prompt_length // int(block_length)
        prefill_length = int(prefill_blocks) * int(block_length)
        resolved_attention_mode = str(attention_mask_mode)

        if prefill_length > 0:
            cur_x = x[:, :prefill_length]
            cur_position_ids = position_ids[:, :prefill_length]
            cur_attn_mask = block_mask_3d[:, :prefill_length, :prefill_length]
            cur_attn_mask_4d = block_mask_4d[:, :, :prefill_length, :prefill_length]
            cur_attn_mask_2d = block_mask_2d[:prefill_length, :prefill_length]
            _, resolved_attention_mode = self._model_forward_logits(
                input_ids=cur_x,
                attention_mask_3d=cur_attn_mask,
                attention_mask_4d=cur_attn_mask_4d,
                attention_mask_2d=cur_attn_mask_2d,
                position_ids=cur_position_ids,
                attention_mask_mode=resolved_attention_mode,
                past_key_values=past_key_values,
                store_kv=True,
            )

        num_transfer_tokens = self.scheduler.get_num_transfer_tokens(int(block_length), int(denoising_steps)).to(
            device=device
        )

        stop_tensor = None
        if stop_token_ids is not None:
            stop_tensor = torch.tensor(stop_token_ids, device=device, dtype=torch.long)

        global_step = 0
        for block_idx in range(prefill_blocks, int(num_blocks)):
            start = int(block_idx) * int(block_length)
            end = start + int(block_length)
            cur_x = x[:, start:end].clone()
            cur_position_ids = position_ids[:, start:end]
            cur_attn_mask = block_mask_3d[:, start:end, :end]
            cur_attn_mask_4d = block_mask_4d[:, :, start:end, :end]
            cur_attn_mask_2d = block_mask_2d[start:end, :end]

            for step in range(int(denoising_steps) + 1):
                mask_index = cur_x == int(mask_token_id)
                if mask_index.sum() == 0:
                    _, resolved_attention_mode = self._model_forward_logits(
                        input_ids=cur_x,
                        attention_mask_3d=cur_attn_mask,
                        attention_mask_4d=cur_attn_mask_4d,
                        attention_mask_2d=cur_attn_mask_2d,
                        position_ids=cur_position_ids,
                        attention_mask_mode=resolved_attention_mode,
                        past_key_values=past_key_values,
                        store_kv=True,
                    )
                    break

                logits, resolved_attention_mode = self._model_forward_logits(
                    input_ids=cur_x,
                    attention_mask_3d=cur_attn_mask,
                    attention_mask_4d=cur_attn_mask_4d,
                    attention_mask_2d=cur_attn_mask_2d,
                    position_ids=cur_position_ids,
                    attention_mask_mode=resolved_attention_mode,
                    past_key_values=past_key_values,
                    store_kv=False,
                )

                step_output = self.scheduler.step(
                    logits,
                    step,
                    cur_x,
                    mask_token_id=int(mask_token_id),
                    num_transfer_tokens=num_transfer_tokens,
                    remasking_strategy=remasking_strategy,
                    confidence_threshold=confidence_threshold,
                    entropy_threshold=entropy_threshold,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    return_dict=True,
                )
                cur_x = step_output.prev_sample
                transfer_index = step_output.transfer_index
                sampled_tokens = step_output.sampled_tokens
                sampled_probs = step_output.sampled_probs

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, global_step, step, callback_kwargs)
                    cur_x = callback_outputs.pop("cur_x", cur_x)

                global_step += 1

            x[:, start:end] = cur_x
            if stop_tensor is not None and torch.isin(x[:, prompt_length:], stop_tensor).any():
                break

        output_ids = x[:, : prompt_length + int(max_new_tokens)]
        if stop_tensor is not None:
            stop_positions = torch.isin(output_ids[0, prompt_length:], stop_tensor).nonzero(as_tuple=True)[0]
            if stop_positions.numel() > 0:
                output_ids = output_ids[:, : prompt_length + int(stop_positions[0].item()) + 1]

        if output_ids.shape[0] == 1:
            output_ids = output_ids[:, output_ids[0] != int(mask_token_id)]

        sequences = output_ids[:, prompt_length:]
        texts = None
        if return_text and getattr(self, "tokenizer", None) is not None:
            texts = self.tokenizer.batch_decode(sequences, skip_special_tokens=True)

        if not return_dict:
            return sequences, texts
        return SDARPipelineOutput(sequences=sequences, texts=texts)

    def _model_forward_logits(
        self,
        *,
        input_ids: torch.LongTensor,
        attention_mask_3d: Optional[torch.Tensor],
        attention_mask_4d: Optional[torch.Tensor],
        attention_mask_2d: Optional[torch.Tensor],
        position_ids: torch.LongTensor,
        attention_mask_mode: str,
        past_key_values: DynamicCache,
        store_kv: bool,
    ) -> Tuple[torch.Tensor, str]:
        if attention_mask_mode not in {"auto", "3d", "4d", "2d", "none"}:
            raise ValueError(
                f"`attention_mask_mode` must be one of {{'auto','3d','4d','2d','none'}}, got {attention_mask_mode!r}."
            )

        def _call(mask):
            kwargs = {
                "input_ids": input_ids,
                "attention_mask": mask,
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": True,
            }
            if self._store_kv_supported is False:
                output = self.model(**kwargs)
                return output.logits if hasattr(output, "logits") else output[0]
            if self._store_kv_supported is True:
                kwargs["store_kv"] = store_kv
                output = self.model(**kwargs)
                return output.logits if hasattr(output, "logits") else output[0]
            try:
                kwargs["store_kv"] = store_kv
                output = self.model(**kwargs)
                self._store_kv_supported = True
                return output.logits if hasattr(output, "logits") else output[0]
            except TypeError:
                output = self.model(**kwargs)
                self._store_kv_supported = False
                return output.logits if hasattr(output, "logits") else output[0]

        if attention_mask_mode == "none":
            return _call(None), "none"
        if attention_mask_mode == "2d":
            return _call(attention_mask_2d), "2d"
        if attention_mask_mode == "3d":
            return _call(attention_mask_3d), "3d"
        if attention_mask_mode == "4d":
            return _call(attention_mask_4d), "4d"

        try:
            return _call(attention_mask_3d), "3d"
        except (TypeError, ValueError, RuntimeError):
            pass
        try:
            return _call(attention_mask_4d), "4d"
        except (TypeError, ValueError, RuntimeError):
            pass
        try:
            return _call(attention_mask_2d), "2d"
        except (TypeError, ValueError, RuntimeError):
            return _call(None), "none"

    def _build_block_attention_mask_3d(
        self,
        *,
        num_blocks: int,
        block_length: int,
        total_length: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        block_mask = torch.tril(torch.ones(num_blocks, num_blocks, device=device, dtype=dtype))
        attn = block_mask.repeat_interleave(block_length, dim=0).repeat_interleave(block_length, dim=1).unsqueeze(0)
        return attn[:, :total_length, :total_length]

    def _build_block_attention_mask_4d(self, mask_3d: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
        attn = mask_3d.unsqueeze(1).to(dtype=dtype)
        return torch.where(
            attn > 0,
            torch.zeros((), device=attn.device, dtype=dtype),
            torch.full((), float("-inf"), device=attn.device, dtype=dtype),
        )


__all__ = ["SDARPipeline", "SDARPipelineOutput"]
