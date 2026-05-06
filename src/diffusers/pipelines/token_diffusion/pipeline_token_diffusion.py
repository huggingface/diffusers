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

from ...callbacks import MultiPipelineCallbacks, PipelineCallback
from ...utils import BaseOutput, logging, replace_example_docstring
from ..pipeline_utils import DiffusionPipeline, DiscreteDiffusionPipelineMixin


logger = logging.get_logger(__name__)

EXAMPLE_DOC_STRING = """
    Examples:
        ```python
        >>> import torch
        >>> from diffusers import TokenDiffusionPipeline, TokenDiffusionScheduler

        >>> model = ...  # Any masked language model returning logits over vocabulary
        >>> tokenizer = ...  # Corresponding tokenizer
        >>> scheduler = TokenDiffusionScheduler(vocab_size=32000, mask_token_id=32000)
        >>> pipe = TokenDiffusionPipeline(model=model, scheduler=scheduler, tokenizer=tokenizer)
        >>> output = pipe(batch_size=1, seq_len=128, num_inference_steps=64)
        >>> print(output.texts[0])
        ```
"""


@dataclass
class TokenDiffusionPipelineOutput(BaseOutput):
    """
    Output class for token diffusion pipelines.

    Args:
        sequences (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Sampled token IDs.
        texts (`list[str]`, *optional*):
            Decoded texts if a tokenizer was provided and `output_type="text"`.
    """

    sequences: torch.LongTensor
    texts: list[str] | None = None


class TokenDiffusionPipeline(DiffusionPipeline, DiscreteDiffusionPipelineMixin):
    """
    Generic token diffusion sampling pipeline.

    This pipeline is intended as a minimal, diffusers-native wrapper around:
    - a token denoiser model (e.g. `transformers.AutoModelForMaskedLM`-like, returning logits over vocab), and
    - a discrete token scheduler (e.g. `TokenDiffusionScheduler`) that implements `set_timesteps()` and `step()`.

    The pipeline supports multiple forward processes via the scheduler configuration (e.g. absorbing/mask, uniform).
    Conditioning (prefix/infill) is intentionally out of scope for the first version.
    """

    model: Any
    tokenizer: Any
    scheduler: Any

    _callback_tensor_inputs = ["input_ids", "logits"]

    def __init__(
        self,
        model: Any,
        scheduler: Any,
        tokenizer: Any | None = None,
    ):
        super().__init__()
        self.register_modules(model=model, scheduler=scheduler, tokenizer=tokenizer)

    @property
    def num_timesteps(self):
        return self._num_timesteps

    def prepare_latents(
        self,
        batch_size: int,
        seq_len: int,
        generator: torch.Generator | None = None,
        device: torch.device | None = None,
    ) -> torch.LongTensor:
        shape = torch.Size((batch_size, seq_len))
        return self.scheduler.sample_prior(shape, device=device, generator=generator)

    def check_inputs(
        self,
        batch_size: int,
        seq_len: int,
        num_inference_steps: int,
        output_type: str,
        callback_on_step_end: Callable | PipelineCallback | MultiPipelineCallbacks | None,
        callback_on_step_end_tensor_inputs: list[str] | None,
        infill_mask: torch.BoolTensor | None,
        prefix_ids: torch.LongTensor | None,
    ):
        # Generation parameter validation
        if batch_size <= 0:
            raise ValueError(f"`batch_size` must be > 0, got {batch_size}.")
        if seq_len <= 0:
            raise ValueError(f"`seq_len` must be > 0, got {seq_len}.")
        if num_inference_steps <= 0:
            raise ValueError(f"`num_inference_steps` must be > 0, got {num_inference_steps}.")
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

        # Mask / prefix validation
        if infill_mask is not None and infill_mask.shape != (batch_size, seq_len):
            raise ValueError(f"`infill_mask` must have shape {(batch_size, seq_len)}, got {tuple(infill_mask.shape)}.")
        if prefix_ids is not None:
            p = prefix_ids
            if p.ndim == 1:
                p = p.unsqueeze(0)
            if p.ndim == 2 and p.shape[1] > seq_len:
                raise ValueError(f"`prefix_ids` length {p.shape[1]} must be <= seq_len={seq_len}.")

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        batch_size: int = 1,
        seq_len: int = 64,
        num_inference_steps: int = 128,
        generator: torch.Generator | None = None,
        prefix_ids: torch.LongTensor | None = None,
        infill_mask: torch.BoolTensor | None = None,
        inject_start_token: bool = False,
        output_type: str = "text",
        return_dict: bool = True,
        callback_on_step_end: Callable[[int, int, dict], None]
        | PipelineCallback
        | MultiPipelineCallbacks
        | None = None,
        callback_on_step_end_tensor_inputs: list[str] | None = None,
        **model_kwargs,
    ) -> TokenDiffusionPipelineOutput | tuple[torch.LongTensor, list[str] | None]:
        """
        Args:
            batch_size: Number of sequences to generate.
            seq_len: Sequence length in tokens.
            num_inference_steps: Number of reverse diffusion steps.
            generator: Optional torch generator for determinism.
            prefix_ids: Optional prefix token IDs to keep fixed at the start of each sequence. Shape `[P]` or
                `[batch_size, P]`.
            infill_mask:
                Optional boolean mask of shape `[batch_size, seq_len]` indicating which positions are editable (`True`)
                vs fixed (`False`). Fixed positions are clamped to the initial values on every step.
            inject_start_token: If True, inject `bos_token_id` (or `cls_token_id`) into position 0 (if available).
            output_type (`str`, defaults to `"text"`):
                Output format. `"text"` decodes sequences into strings (requires a tokenizer). `"seq"` returns raw
                token ID sequences only.
            return_dict: If True, returns a `TokenDiffusionPipelineOutput`.
            callback_on_step_end: A function called after each denoising step with signature
                `callback_on_step_end(self, step: int, timestep: int, callback_kwargs: dict)`.
            callback_on_step_end_tensor_inputs: List of tensor keys to include in `callback_kwargs`.
            model_kwargs: Forward kwargs passed to `model(...)` (e.g. attention mask overrides).

        Examples:
        """
        # 1. Check inputs early
        if callback_on_step_end is not None and isinstance(
            callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)
        ):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs
        if callback_on_step_end_tensor_inputs is None:
            callback_on_step_end_tensor_inputs = ["input_ids"]

        self.check_inputs(
            batch_size=batch_size,
            seq_len=seq_len,
            num_inference_steps=num_inference_steps,
            output_type=output_type,
            callback_on_step_end=callback_on_step_end,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            infill_mask=infill_mask,
            prefix_ids=prefix_ids,
        )

        # 2. Prepare timesteps
        device = self._execution_device

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        self._num_timesteps = len(timesteps)

        # 3. Prepare latents
        input_ids = self.prepare_latents(batch_size, seq_len, generator=generator, device=device)

        # 4. Build fixed masks for prefix / infill conditioning
        fixed_mask = None
        fixed_values = None
        if infill_mask is not None:
            fixed_mask = (~infill_mask.to(device=device)).to(dtype=torch.bool)
            fixed_values = input_ids.clone()

        if prefix_ids is not None:
            prefix_ids = self._normalize_prefix_ids(prefix_ids, batch_size=batch_size, device=device)
            prefix_len = prefix_ids.shape[1]

            input_ids[:, :prefix_len] = prefix_ids
            if fixed_mask is None:
                fixed_mask = torch.zeros((batch_size, seq_len), device=device, dtype=torch.bool)
                fixed_values = input_ids.clone()
            fixed_mask[:, :prefix_len] = True
            fixed_values[:, :prefix_len] = prefix_ids

        start_token_id = self._resolve_start_token_id()
        if inject_start_token and start_token_id is not None:
            input_ids[:, 0] = start_token_id
            if fixed_mask is not None:
                fixed_mask[:, 0] = True
                fixed_values[:, 0] = start_token_id

        # 5. Denoising loop
        progress_bar = self.progress_bar(total=num_inference_steps)
        for step_idx, t in enumerate(timesteps):
            timestep = t.expand(batch_size)
            out = self.model(input_ids=input_ids, timesteps=timestep, return_dict=True, **model_kwargs)
            logits = getattr(out, "logits", None)
            if logits is None:
                # Fall back to tuple-style returns.
                logits = out[0]

            input_ids = self.scheduler.step(logits, t, input_ids, generator=generator, return_dict=True).prev_sample

            # Enforce fixed masks (prefix / infill conditioning)
            if fixed_mask is not None:
                input_ids = torch.where(fixed_mask, fixed_values, input_ids)

            if inject_start_token and start_token_id is not None:
                input_ids[:, 0] = start_token_id

            if callback_on_step_end is not None:
                callback_kwargs = {}
                for k in callback_on_step_end_tensor_inputs:
                    callback_kwargs[k] = locals()[k]
                callback_outputs = callback_on_step_end(self, step_idx, t, callback_kwargs)
                input_ids = callback_outputs.pop("input_ids", input_ids)

            progress_bar.update(1)
        progress_bar.close()

        # 6. Post-process output
        texts = None
        if output_type == "text" and getattr(self, "tokenizer", None) is not None:
            texts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)

        if not return_dict:
            return (input_ids, texts)
        return TokenDiffusionPipelineOutput(sequences=input_ids, texts=texts)


__all__ = ["TokenDiffusionPipeline", "TokenDiffusionPipelineOutput"]
