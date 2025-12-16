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

from typing import Optional, Tuple, Union

import torch

from .scheduling_token_diffusion import TokenDiffusionScheduler, TokenDiffusionSchedulerOutput


class BlockTokenDiffusionScheduler(TokenDiffusionScheduler):
    """
    A token diffusion scheduler that supports updating only a subset of positions (e.g. a block).

    This scheduler reuses the same alpha schedules and forward processes as `TokenDiffusionScheduler`, but allows
    callers to restrict noising/denoising to a boolean `block_mask` of shape `[batch, seq_len]`.
    """

    @classmethod
    def from_config(cls, config, **kwargs):
        # TokenDiffusionScheduler doesn't have compatibles; keep standard ConfigMixin behavior.
        return super().from_config(config, **kwargs)

    def add_noise(
        self,
        original_samples: torch.LongTensor,
        noise: Optional[torch.Tensor],
        timesteps: torch.LongTensor,
        block_mask: Optional[torch.BoolTensor] = None,
    ) -> torch.LongTensor:
        if block_mask is None:
            return super().add_noise(original_samples=original_samples, noise=noise, timesteps=timesteps)

        if block_mask.dtype != torch.bool:
            raise ValueError(f"`block_mask` must be boolean, got dtype={block_mask.dtype}.")
        if block_mask.shape != original_samples.shape:
            raise ValueError(
                f"`block_mask` must have shape {tuple(original_samples.shape)}, got {tuple(block_mask.shape)}."
            )

        noised = super().add_noise(original_samples=original_samples, noise=noise, timesteps=timesteps)
        return torch.where(block_mask.to(device=original_samples.device), noised, original_samples)

    def step(
        self,
        model_output: torch.Tensor,
        timestep: Union[int, torch.Tensor],
        sample: torch.LongTensor,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
        block_mask: Optional[torch.BoolTensor] = None,
    ) -> Union[TokenDiffusionSchedulerOutput, Tuple[torch.LongTensor]]:
        if block_mask is None:
            return super().step(
                model_output=model_output,
                timestep=timestep,
                sample=sample,
                generator=generator,
                return_dict=return_dict,
            )

        if block_mask.dtype != torch.bool:
            raise ValueError(f"`block_mask` must be boolean, got dtype={block_mask.dtype}.")
        if block_mask.shape != sample.shape:
            raise ValueError(f"`block_mask` must have shape {tuple(sample.shape)}, got {tuple(block_mask.shape)}.")

        out = super().step(
            model_output=model_output,
            timestep=timestep,
            sample=sample,
            generator=generator,
            return_dict=True,
        )
        prev = out.prev_sample
        prev = torch.where(block_mask.to(device=prev.device), prev, sample)

        if not return_dict:
            return (prev,)
        return TokenDiffusionSchedulerOutput(prev_sample=prev)


__all__ = ["BlockTokenDiffusionScheduler"]
