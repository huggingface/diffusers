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

import torch

from ..configuration_utils import ConfigMixin, register_to_config
from ..utils import BaseOutput
from .scheduling_utils import SchedulerMixin


@dataclass
class DFlashTokenDiffusionSchedulerOutput(BaseOutput):
    """
    Output class for DFlash-style speculative token scheduling.

    Args:
        prev_sample (`torch.LongTensor` of shape `(batch_size, block_size)`):
            The proposed block tokens from the draft model.
        accepted_length (`torch.LongTensor` of shape `(batch_size,)`):
            Number of consecutive accepted tokens from the block.
        next_token (`torch.LongTensor` of shape `(batch_size,)`):
            Next token sampled from the target posterior at the first rejection.
        posterior (`torch.LongTensor` of shape `(batch_size, block_size)`):
            Sampled tokens from the target posterior used for acceptance checks.
    """

    prev_sample: torch.LongTensor
    accepted_length: torch.LongTensor
    next_token: torch.LongTensor
    posterior: torch.LongTensor


class DFlashTokenDiffusionScheduler(SchedulerMixin, ConfigMixin):
    """
    Scheduler for DFlash-style block diffusion speculative decoding.

    This scheduler samples target posteriors and computes acceptance lengths for draft blocks.
    """

    order = 1

    @register_to_config
    def __init__(self):
        self.num_inference_steps = 1
        self.timesteps = torch.tensor([0], dtype=torch.long)

    def set_timesteps(self, num_inference_steps: int, device: str | torch.device | None = None) -> None:
        if num_inference_steps <= 0:
            raise ValueError(f"`num_inference_steps` must be > 0, got {num_inference_steps}.")
        self.num_inference_steps = int(num_inference_steps)
        self.timesteps = torch.arange(self.num_inference_steps - 1, -1, -1, device=device, dtype=torch.long)

    def sample(self, logits: torch.Tensor, temperature: float = 0.0) -> torch.LongTensor:
        if temperature < 1e-5:
            return torch.argmax(logits, dim=-1)
        bsz, seq_len, vocab_size = logits.shape
        flat = logits.view(-1, vocab_size) / float(temperature)
        probs = torch.softmax(flat, dim=-1)
        return torch.multinomial(probs, num_samples=1).view(bsz, seq_len)

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int | torch.Tensor,
        sample: torch.LongTensor,
        *,
        temperature: float = 0.0,
        return_dict: bool = True,
    ) -> (
        DFlashTokenDiffusionSchedulerOutput
        | tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor, torch.LongTensor]
    ):
        """
        Perform a single speculative decoding verification step.

        Args:
            model_output (`torch.Tensor` of shape `(batch_size, block_size, vocab_size)`):
                Raw logits from the target model for the current block.
            timestep (`int` or `torch.Tensor`):
                Current step index (unused for single-step DFlash, kept for interface compatibility).
            sample (`torch.LongTensor` of shape `(batch_size, block_size)`):
                Draft token IDs proposed by the draft model.
            temperature (`float`):
                Sampling temperature for the target posterior.
            return_dict (`bool`):
                Whether to return a `DFlashTokenDiffusionSchedulerOutput` or a tuple.
        """
        posterior = self.sample(model_output, temperature=temperature)
        if sample.shape[1] > 1:
            matches = sample[:, 1:] == posterior[:, :-1]
            accepted_length = matches.int().cumprod(dim=1).sum(dim=1)
        else:
            accepted_length = torch.zeros((sample.shape[0],), device=sample.device, dtype=torch.long)

        next_token = posterior.gather(1, accepted_length.unsqueeze(1)).squeeze(1)

        if not return_dict:
            return sample, accepted_length, next_token, posterior
        return DFlashTokenDiffusionSchedulerOutput(
            prev_sample=sample,
            accepted_length=accepted_length,
            next_token=next_token,
            posterior=posterior,
        )

    @staticmethod
    def cache_has_linear_attention(cache) -> bool:
        """
        Detect whether a `DynamicCache` contains any linear-attention layers (e.g. Qwen3.5's gated-delta-net layers).
        The spec-decoding loop needs this to know whether a partial-accept block requires snapshot/restore rather than
        a plain `.crop()` — transformers' `DynamicCache.crop()` silently no-ops on linear-attention layers, so rejected
        speculative tokens would otherwise permanently contaminate the recurrent state.

        Duck-typed on `recurrent_states`/`conv_states` attributes to avoid importing transformers.
        """
        for layer in getattr(cache, "layers", []):
            if hasattr(layer, "recurrent_states") and hasattr(layer, "conv_states"):
                return True
        return False

    @staticmethod
    def snapshot_cache(cache) -> list[dict]:
        """
        Clone the full per-layer cache state so a speculative target forward can be rolled back.

        Handles both full-attention `DynamicLayer` (keys/values) and linear-attention layers
        (conv_states/recurrent_states plus their init flags). Mirrors upstream DFlash's MLX `_GDNStateCapture`
        rollback, but via full-layer restore rather than kernel-level replay. Pair with `restore_cache()`; no-op if the
        caller only ever fully-accepts.
        """
        snapshots: list[dict] = []
        for layer in getattr(cache, "layers", []):
            snap: dict = {"cls": type(layer)}
            if hasattr(layer, "keys") and layer.keys is not None:
                snap["keys"] = layer.keys.clone()
                snap["values"] = layer.values.clone()
            if hasattr(layer, "recurrent_states"):
                snap["has_previous_state"] = bool(getattr(layer, "has_previous_state", False))
                snap["is_recurrent_states_initialized"] = bool(
                    getattr(layer, "is_recurrent_states_initialized", False)
                )
                snap["is_conv_states_initialized"] = bool(getattr(layer, "is_conv_states_initialized", False))
                snap["recurrent_states"] = (
                    layer.recurrent_states.clone() if getattr(layer, "recurrent_states", None) is not None else None
                )
                snap["conv_states"] = (
                    layer.conv_states.clone() if getattr(layer, "conv_states", None) is not None else None
                )
            snapshots.append(snap)
        return snapshots

    @staticmethod
    def restore_cache(cache, snapshots: list[dict]) -> None:
        """
        Restore a cache to the state captured by `snapshot_cache()`. After this call, the caller should re-advance the
        cache (e.g. by re-running the target model on just the accepted prefix) so both full- and linear-attention
        layers end up at the committed token count.
        """
        for layer, snap in zip(cache.layers, snapshots):
            if "keys" in snap:
                # DynamicLayer: reassign (shapes will have grown during the verify forward, so
                # in-place copy is not safe here).
                layer.keys = snap["keys"]
                layer.values = snap["values"]
            if "recurrent_states" in snap:
                # LinearAttentionLayer: in-place copy preserves any static-address assumption
                # (e.g. for cudagraph capture) on the live tensors.
                layer.has_previous_state = snap["has_previous_state"]
                layer.is_recurrent_states_initialized = snap["is_recurrent_states_initialized"]
                layer.is_conv_states_initialized = snap["is_conv_states_initialized"]
                if snap["recurrent_states"] is not None and getattr(layer, "recurrent_states", None) is not None:
                    layer.recurrent_states.copy_(snap["recurrent_states"])
                elif snap["recurrent_states"] is not None:
                    layer.recurrent_states = snap["recurrent_states"].clone()
                if snap["conv_states"] is not None and getattr(layer, "conv_states", None) is not None:
                    layer.conv_states.copy_(snap["conv_states"])
                elif snap["conv_states"] is not None:
                    layer.conv_states = snap["conv_states"].clone()

    @staticmethod
    def check_should_stop(
        output_ids: torch.LongTensor,
        stop_token_ids: list[int] | None,
        num_input_tokens: int,
    ) -> bool:
        """
        Check whether any stop token has been generated in the output sequence.

        Args:
            output_ids (`torch.LongTensor` of shape `(batch_size, seq_len)`):
                Current output token IDs including prompt and generated tokens.
            stop_token_ids (`list[int]` or `None`):
                Token IDs that signal generation should stop.
            num_input_tokens (`int`):
                Number of prompt tokens at the start of the sequence.

        Returns:
            `bool`: `True` if generation should stop, `False` otherwise.
        """
        if stop_token_ids is None:
            return False
        stop_tensor = torch.tensor(stop_token_ids, device=output_ids.device, dtype=torch.long)
        return torch.isin(output_ids[:, num_input_tokens:], stop_tensor).any().item()

    def add_noise(
        self,
        original_samples: torch.LongTensor,
        attention_mask: torch.LongTensor,
        *,
        prompt_length: int,
        block_size: int,
        mask_token_id: int,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.LongTensor, torch.BoolTensor]:
        """
        Apply the forward (noising) process for DFlash-style block diffusion training.

        For each block after the prompt, a random fraction of valid (non-padding) tokens are replaced with
        `mask_token_id`.

        Args:
            original_samples (`torch.LongTensor` of shape `(batch_size, seq_len)`):
                Clean token IDs.
            attention_mask (`torch.LongTensor` of shape `(batch_size, seq_len)`):
                Padding mask (1 for valid, 0 for padding).
            prompt_length (`int`):
                Number of leading prompt tokens to keep unmasked.
            block_size (`int`):
                Block size for masking.
            mask_token_id (`int`):
                Token ID to use for masked positions.
            generator (`torch.Generator`, *optional*):
                RNG for reproducibility.

        Returns:
            `tuple[torch.LongTensor, torch.BoolTensor]`:
                `(noisy, masked)` -- the noisy sequence and the boolean mask indicating which positions were masked.
        """
        batch_size, seq_len = original_samples.shape
        device = original_samples.device

        noisy = original_samples.clone()
        masked = torch.zeros_like(original_samples, dtype=torch.bool)

        valid = attention_mask.to(dtype=torch.bool)
        for block_start in range(prompt_length, seq_len, block_size):
            block_end = min(seq_len, block_start + block_size)
            seg_len = block_end - block_start
            if seg_len <= 0:
                continue

            p_mask = torch.rand((batch_size, 1), device=device, generator=generator)
            seg = torch.rand((batch_size, seg_len), device=device, generator=generator) < p_mask
            seg = seg & valid[:, block_start:block_end]

            masked[:, block_start:block_end] = seg

        noisy = torch.where(masked, torch.full_like(noisy, mask_token_id), noisy)
        return noisy, masked


__all__ = ["DFlashTokenDiffusionScheduler", "DFlashTokenDiffusionSchedulerOutput"]
