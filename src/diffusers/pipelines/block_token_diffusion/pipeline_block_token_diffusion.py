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
class BlockTokenDiffusionPipelineOutput(BaseOutput):
    sequences: torch.LongTensor
    texts: Optional[List[str]] = None


def _top_p_filtering(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    if top_p >= 1.0:
        return logits
    if not (0.0 < top_p <= 1.0):
        raise ValueError(f"`top_p` must be in (0, 1], got {top_p}.")

    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    cumulative_probs = sorted_probs.cumsum(dim=-1)

    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 0] = 0

    sorted_logits = sorted_logits.masked_fill(sorted_indices_to_remove, torch.finfo(sorted_logits.dtype).min)
    filtered = logits.scatter(-1, sorted_indices, sorted_logits)
    return filtered


class BlockTokenDiffusionPipeline(DiffusionPipeline):
    """
    Block-wise token diffusion sampling pipeline.

    Compared to `TokenDiffusionPipeline`, this pipeline updates the sequence in blocks. Only the current block's
    positions are allowed to change during the inner denoising loop.
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

    def _init_latents(
        self,
        batch_size: int,
        seq_len: int,
        *,
        generator: Optional[torch.Generator],
        device: torch.device,
    ) -> torch.LongTensor:
        mask_token_id = getattr(self.scheduler, "mask_token_id", None)
        if mask_token_id is None:
            raise ValueError("Scheduler must define `mask_token_id` for block diffusion sampling.")
        return torch.full((batch_size, seq_len), int(mask_token_id), device=device, dtype=torch.long)

    @torch.no_grad()
    def __call__(
        self,
        *,
        batch_size: int = 1,
        seq_len: int = 64,
        block_size: int = 32,
        num_inference_steps: int = 64,
        generator: Optional[torch.Generator] = None,
        prefix_ids: Optional[torch.LongTensor] = None,
        infill_mask: Optional[torch.BoolTensor] = None,
        inject_start_token: bool = False,
        top_p: float = 1.0,
        return_text: bool = True,
        return_dict: bool = True,
        **model_kwargs,
    ) -> Union[BlockTokenDiffusionPipelineOutput, Tuple[torch.LongTensor, Optional[List[str]]]]:
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
            if fixed_mask is None:
                fixed_mask = torch.zeros((batch_size, seq_len), device=device, dtype=torch.bool)
                fixed_values = input_ids.clone()
            fixed_mask[:, 0] = True
            fixed_values[:, 0] = start_token_id

        if block_size <= 0 or block_size > seq_len:
            raise ValueError(f"`block_size` must be in [1, seq_len], got block_size={block_size}, seq_len={seq_len}.")

        num_blocks = (seq_len + block_size - 1) // block_size
        for block_idx in range(num_blocks):
            start = block_idx * block_size
            end = min((block_idx + 1) * block_size, seq_len)

            block_mask = torch.zeros((batch_size, seq_len), device=device, dtype=torch.bool)
            block_mask[:, start:end] = True
            if fixed_mask is not None:
                block_mask = block_mask & (~fixed_mask)

            if not torch.any(block_mask):
                continue

            input_ids = torch.where(block_mask, int(self.scheduler.mask_token_id), input_ids)

            for t in self.scheduler.timesteps:
                out = self.model(input_ids=input_ids, attention_mask=attention_mask, **model_kwargs)
                logits = getattr(out, "logits", None)
                if logits is None:
                    logits = out[0]

                if top_p < 1.0:
                    logits_block = logits[block_mask].view(-1, logits.shape[-1])
                    logits_block = _top_p_filtering(logits_block, top_p=top_p)
                    logits = logits.clone()
                    logits[block_mask] = logits_block.view(-1, logits.shape[-1])

                input_ids = self.scheduler.step(
                    logits,
                    t,
                    input_ids,
                    generator=generator,
                    return_dict=True,
                    block_mask=block_mask,
                ).prev_sample

                if fixed_mask is not None:
                    input_ids = torch.where(fixed_mask, fixed_values, input_ids)

        texts = None
        if return_text and getattr(self, "tokenizer", None) is not None:
            texts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)

        if not return_dict:
            return (input_ids, texts)
        return BlockTokenDiffusionPipelineOutput(sequences=input_ids, texts=texts)


__all__ = ["BlockTokenDiffusionPipeline", "BlockTokenDiffusionPipelineOutput"]
