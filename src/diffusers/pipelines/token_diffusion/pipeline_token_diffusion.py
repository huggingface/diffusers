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
from typing import Any, List, Optional, Tuple, Union

import torch

from ...utils import BaseOutput
from ..pipeline_utils import DiffusionPipeline


@dataclass
class TokenDiffusionPipelineOutput(BaseOutput):
    """
    Output class for token diffusion pipelines.

    Args:
        sequences (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Sampled token IDs.
        texts (`List[str]`, *optional*):
            Decoded texts if a tokenizer was provided and `return_text=True`.
    """

    sequences: torch.LongTensor
    texts: Optional[List[str]] = None


class TokenDiffusionPipeline(DiffusionPipeline):
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

    def __init__(self, model: Any, scheduler: Any, tokenizer: Optional[Any] = None):
        super().__init__()
        self.register_modules(model=model, scheduler=scheduler, tokenizer=tokenizer)

    def _resolve_start_token_id(self) -> Optional[int]:
        tok = getattr(self, "tokenizer", None)
        if tok is None:
            return None
        for attr in ("bos_token_id", "cls_token_id"):
            token_id = getattr(tok, attr, None)
            if token_id is not None:
                return int(token_id)
        return None

    def _init_latents(
        self,
        batch_size: int,
        seq_len: int,
        *,
        generator: Optional[torch.Generator],
        device: torch.device,
    ) -> torch.LongTensor:
        # Prefer a scheduler-provided prior if available.
        if hasattr(self.scheduler, "forward_process") and getattr(self.scheduler, "forward_process") == "uniform":
            # Uniform prior over token IDs. Mirror scheduler's exclude-mask behavior.
            if getattr(self.scheduler, "exclude_mask_from_uniform", False) and hasattr(
                self.scheduler, "_sample_uniform_tokens"
            ):
                return self.scheduler._sample_uniform_tokens(
                    torch.Size((batch_size, seq_len)),
                    device=device,
                    dtype=torch.long,
                    generator=generator,
                )
            vocab_size = int(getattr(self.scheduler, "vocab_size", 0))
            if vocab_size <= 0:
                raise ValueError("Scheduler must define `vocab_size` for uniform prior sampling.")
            return torch.randint(
                0, vocab_size, (batch_size, seq_len), device=device, dtype=torch.long, generator=generator
            )

        mask_token_id = getattr(self.scheduler, "mask_token_id", None)
        if mask_token_id is None:
            raise ValueError("Scheduler must define `mask_token_id` for absorbing prior sampling.")
        return torch.full((batch_size, seq_len), int(mask_token_id), device=device, dtype=torch.long)

    def _normalize_prefix_ids(
        self, prefix_ids: torch.LongTensor, batch_size: int, device: torch.device
    ) -> torch.LongTensor:
        if prefix_ids.ndim == 1:
            prefix_ids = prefix_ids.unsqueeze(0)
        if prefix_ids.ndim != 2:
            raise ValueError(
                f"`prefix_ids` must have shape [prefix_len] or [batch, prefix_len], got {prefix_ids.shape}."
            )
        if prefix_ids.shape[0] not in (1, batch_size):
            raise ValueError(
                f"`prefix_ids` batch dim must be 1 or batch_size={batch_size}, got {prefix_ids.shape[0]}."
            )
        if prefix_ids.dtype != torch.long:
            raise ValueError(f"`prefix_ids` must be int64 token IDs, got dtype={prefix_ids.dtype}.")
        prefix_ids = prefix_ids.to(device=device)
        if prefix_ids.shape[0] == 1 and batch_size > 1:
            prefix_ids = prefix_ids.expand(batch_size, -1)
        return prefix_ids

    @torch.no_grad()
    def __call__(
        self,
        *,
        batch_size: int = 1,
        seq_len: int = 64,
        num_inference_steps: int = 128,
        generator: Optional[torch.Generator] = None,
        prefix_ids: Optional[torch.LongTensor] = None,
        infill_mask: Optional[torch.BoolTensor] = None,
        inject_start_token: bool = False,
        return_text: bool = True,
        return_dict: bool = True,
        **model_kwargs,
    ) -> Union[TokenDiffusionPipelineOutput, Tuple[torch.LongTensor, Optional[List[str]]]]:
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
            return_text: If True and tokenizer exists, also return decoded strings.
            return_dict: If True, returns a `TokenDiffusionPipelineOutput`.
            model_kwargs: Forward kwargs passed to `model(...)` (e.g. attention mask overrides).
        """
        device = self._execution_device

        self.scheduler.set_timesteps(num_inference_steps, device=device)

        input_ids = self._init_latents(batch_size, seq_len, generator=generator, device=device)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)

        fixed_mask = None
        fixed_values = None
        if infill_mask is not None:
            if infill_mask.shape != (batch_size, seq_len):
                raise ValueError(
                    f"`infill_mask` must have shape {(batch_size, seq_len)}, got {tuple(infill_mask.shape)}."
                )
            fixed_mask = (~infill_mask.to(device=device)).to(dtype=torch.bool)
            fixed_values = input_ids.clone()

        if prefix_ids is not None:
            prefix_ids = self._normalize_prefix_ids(prefix_ids, batch_size=batch_size, device=device)
            prefix_len = prefix_ids.shape[1]
            if prefix_len > seq_len:
                raise ValueError(f"`prefix_ids` length {prefix_len} must be <= seq_len={seq_len}.")

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

        for t in self.scheduler.timesteps:
            out = self.model(input_ids=input_ids, attention_mask=attention_mask, **model_kwargs)
            logits = getattr(out, "logits", None)
            if logits is None:
                # Fall back to tuple-style returns.
                logits = out[0]

            input_ids = self.scheduler.step(logits, t, input_ids, generator=generator, return_dict=True).prev_sample

            if fixed_mask is not None:
                input_ids = torch.where(fixed_mask, fixed_values, input_ids)

            if inject_start_token and start_token_id is not None:
                input_ids[:, 0] = start_token_id

        texts = None
        if return_text and getattr(self, "tokenizer", None) is not None:
            texts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)

        if not return_dict:
            return (input_ids, texts)
        return TokenDiffusionPipelineOutput(sequences=input_ids, texts=texts)


__all__ = ["TokenDiffusionPipeline", "TokenDiffusionPipelineOutput"]
