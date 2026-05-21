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
import torch.nn.functional as F

from ..configuration_utils import ConfigMixin, register_to_config
from ..utils import BaseOutput
from .scheduling_utils import SchedulerMixin


@dataclass
class IDLMBlockDiffusionSchedulerOutput(BaseOutput):
    """
    Output class for I-DLM block-N introspective strided decoding.

    Args:
        committed_tokens (`torch.LongTensor`):
            1-D tensor of tokens committed in this round. Always includes the round's `pending` token as the first
            element, followed by any accepted speculative tokens (left-to-right until the first reject).
        accepted_length (`int`):
            Number of speculative tokens accepted this round (`0 <= accepted_length <= num_input_specs`).
        next_pending (`int`):
            Next round's `pending` token. On full accept, sampled from the anchor logits at the last-spec position. On
            partial accept, resampled from `max(0, p - alpha*q)` at the first rejected spec.
        next_specs (`list[int]`):
            Next round's speculative tokens (length 0 on partial-accept → cold-start; length `num_masks` on full-accept
            → verify).
        next_draft_probs (`torch.Tensor` or `None`):
            Proposal (`q`) probabilities for `next_specs`, shape `[len(next_specs), vocab_size]` on full-accept (used
            for verification in the following round). `None` on partial-accept.
        was_full_accept (`bool`):
            `True` if all input specs were accepted and new specs were drawn, `False` if a reject occurred.
    """

    committed_tokens: torch.LongTensor
    accepted_length: int
    next_pending: int
    next_specs: list[int]
    next_draft_probs: torch.Tensor | None
    was_full_accept: bool


class IDLMBlockDiffusionScheduler(SchedulerMixin, ConfigMixin):
    """
    Scheduler for I-DLM (Introspective Diffusion Language Models) block-N decoding with speculative verification.

    Implements the *Introspective Strided Decoding* (ISD) step: given logits from a single model forward over
    `[pending, spec_0, ..., spec_{K-1}, MASK, MASK, ...]` (with Dream-style logit shift, i.e. `logits[i]` predicts the
    token at input position `i+1`), the scheduler:

    1. Verifies specs left-to-right via `min(1, p(x) / (alpha * q(x)))`. On the first reject, resamples from `max(0, p
       - alpha * q)` and discards the remaining specs.
    2. Samples the next batch of speculative tokens from the MASK-position anchor logits (full-accept only).
    3. Returns the committed tokens for this round + the `(next_pending, next_specs, next_draft_probs)` triple that the
       pipeline feeds into the next round.

    Stateless, pure-math — no cache or model I/O.
    """

    order = 1

    @register_to_config
    def __init__(
        self,
        gen_block_size: int = 3,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        verify_alpha: float = 1.0,
    ):
        self.num_inference_steps = 1
        self.timesteps = torch.tensor([0], dtype=torch.long)

    def set_timesteps(self, num_inference_steps: int, device: str | torch.device | None = None) -> None:
        if num_inference_steps <= 0:
            raise ValueError(f"`num_inference_steps` must be > 0, got {num_inference_steps}.")
        self.num_inference_steps = int(num_inference_steps)
        self.timesteps = torch.arange(self.num_inference_steps - 1, -1, -1, device=device, dtype=torch.long)

    @property
    def block_size(self) -> int:
        """Per-round input width = 1 + 2*(N-1) = 2N - 1 for gen_block_size N."""
        return 2 * int(self.config.gen_block_size) - 1

    @property
    def num_masks(self) -> int:
        """Number of MASK slots per round = N - 1."""
        return int(self.config.gen_block_size) - 1

    # ---- sampling helpers ---------------------------------------------------

    @staticmethod
    def _top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
        if k <= 0:
            return logits
        values, _ = torch.topk(logits, min(k, logits.shape[-1]))
        min_values = values[..., -1, None]
        return torch.where(logits < min_values, torch.full_like(logits, float("-inf")), logits)

    @staticmethod
    def _top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
        if p >= 1.0:
            return logits
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_mask = cum_probs > p
        sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
        sorted_mask[..., 0] = False
        mask_indices = torch.scatter(torch.zeros_like(logits, dtype=torch.bool), -1, sorted_indices, sorted_mask)
        return logits.masked_fill(mask_indices, float("-inf"))

    def sample(
        self,
        logits: torch.Tensor,
        *,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        return_probs: bool = False,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.LongTensor, torch.Tensor] | tuple[torch.LongTensor, torch.Tensor, torch.Tensor]:
        """
        Sample from logits of shape `[N, vocab]` with top-k/top-p. Returns `(token_ids [N], token_probs [N])`; if
        `return_probs` is set, also returns the full (post-filter) probability matrix `[N, vocab]` so the pipeline can
        stash it as `q` for next-round verification.
        """
        temperature = float(self.config.temperature if temperature is None else temperature)
        top_k = int(self.config.top_k if top_k is None else top_k)
        top_p = float(self.config.top_p if top_p is None else top_p)

        if logits.dim() != 2:
            logits = logits.reshape(-1, logits.shape[-1])

        if temperature < 1e-5:
            # Greedy: q becomes a one-hot at argmax. Use a softmax of raw logits so probs are well-defined
            # for later verification (avoids divide-by-zero when comparing p/q).
            token_ids = logits.argmax(dim=-1)
            probs = F.softmax(logits, dim=-1)
            token_probs = probs.gather(1, token_ids.unsqueeze(1)).squeeze(1)
            if return_probs:
                return token_ids, token_probs, probs
            return token_ids, token_probs

        scaled = logits / temperature
        scaled = self._top_k_filter(scaled, top_k)
        scaled = self._top_p_filter(scaled, top_p)
        probs = F.softmax(scaled, dim=-1)
        token_ids = torch.multinomial(probs, num_samples=1, generator=generator).squeeze(1)
        token_probs = probs.gather(1, token_ids.unsqueeze(1)).squeeze(1)
        if return_probs:
            return token_ids, token_probs, probs
        return token_ids, token_probs

    # ---- ISD verify step ----------------------------------------------------

    def verify_specs(
        self,
        anchor_logits: torch.Tensor,
        spec_tokens: list[int],
        draft_probs: torch.Tensor,
        *,
        alpha: float | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[int, int | None]:
        """
        Verify a list of speculative tokens left-to-right under the standard speculative-decoding criterion.

        Args:
            anchor_logits (`torch.Tensor` of shape `[K, vocab]`):
                Anchor distribution logits — `anchor_logits[i]` is the target/anchor distribution over the token at the
                position occupied by `spec_tokens[i]` (already shift-aligned by the pipeline).
            spec_tokens (`list[int]`):
                Proposed speculative tokens from the previous round (length `K`).
            draft_probs (`torch.Tensor` of shape `[K, vocab]`):
                Proposal (`q`) distributions the specs were sampled from.
            alpha (`float`, *optional*):
                Leniency factor for `min(1, p/(alpha*q))`. Defaults to `config.verify_alpha`.
            generator (`torch.Generator`, *optional*):
                RNG for the accept Bernoulli and resample multinomial.

        Returns:
            `(accepted_length, resample_token)`:
                * `accepted_length` — number of leading specs accepted (`0..K`).
                * `resample_token` — on reject, the token resampled from `max(0, p - alpha*q)`. `None` on full accept.
        """
        if alpha is None:
            alpha = float(self.config.verify_alpha)
        K = len(spec_tokens)
        if K == 0:
            return 0, None

        device = anchor_logits.device
        p_probs = F.softmax(anchor_logits.float(), dim=-1)  # [K, vocab]
        q_probs = draft_probs.to(device=device, dtype=p_probs.dtype)
        eps = 1e-20

        for i in range(K):
            tok = int(spec_tokens[i])
            p_i = float(p_probs[i, tok].item())
            q_i = float(q_probs[i, tok].item())
            ratio = p_i / max(alpha * q_i, eps)
            u = torch.rand(1, device=device, generator=generator).item()
            if u < min(1.0, ratio):
                continue  # accept
            # reject: resample from max(0, p - alpha*q)
            residual = (p_probs[i] - alpha * q_probs[i]).clamp_min(0.0)
            s = residual.sum()
            if s.item() <= eps:
                # Degenerate (p == alpha*q exactly): fall back to p.
                residual = p_probs[i]
                s = residual.sum()
            residual = residual / s
            resampled = int(torch.multinomial(residual, num_samples=1, generator=generator).item())
            return i, resampled

        return K, None

    # ---- ISD step: verify + assemble next-round inputs ---------------------

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int | torch.Tensor,
        pending: int,
        *,
        spec_tokens: list[int] | None,
        spec_draft_probs: torch.Tensor | None,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        verify_alpha: float | None = None,
        generator: torch.Generator | None = None,
        return_dict: bool = True,
    ) -> IDLMBlockDiffusionSchedulerOutput | tuple:
        """
        One ISD round.

        Args:
            model_output (`torch.Tensor` of shape `[1, L, vocab]`):
                Raw logits for the round's input `[pending, *spec_tokens, M, M, ...]` (length L). Dream logit shift is
                applied here: `logits[:, i, :]` is treated as the distribution for the token at input position `i+1`.
            timestep (`int` or `torch.Tensor`):
                Unused (single-step scheduler; kept for diffusers interface compatibility).
            pending (`int`):
                The current round's clean/committed token (input position 0).
            spec_tokens (`list[int]` or `None`):
                Specs carried over from the previous round. Empty/`None` triggers the cold-start path (no verify).
            spec_draft_probs (`torch.Tensor` or `None`):
                Proposal probabilities for `spec_tokens`, shape `[K, vocab_size]`. Required when `spec_tokens` is
                non-empty.
            temperature, top_k, top_p:
                Sampling overrides for new-spec sampling. Fall back to config defaults.
            verify_alpha (`float`, *optional*):
                Leniency factor for the accept criterion. Defaults to `config.verify_alpha`.
            generator (`torch.Generator`, *optional*):
                RNG used for verification and new-spec sampling.
            return_dict (`bool`):
                Whether to return `IDLMBlockDiffusionSchedulerOutput` or a raw tuple.
        """
        num_masks = self.num_masks
        # Apply the Dream shift inline: distribution predicting input position `j+1` is `logits[:, j, :]`.
        logits = model_output[0]  # [L, vocab]
        spec_tokens = list(spec_tokens) if spec_tokens else []
        K = len(spec_tokens)

        # --- verify ---
        if K > 0:
            anchor = logits[:K]  # [K, vocab] — each anchor[i] predicts position i+1 where spec_i sits
            if spec_draft_probs is None:
                raise ValueError("spec_draft_probs must be provided when spec_tokens is non-empty")
            accepted, resample = self.verify_specs(
                anchor, spec_tokens, spec_draft_probs, alpha=verify_alpha, generator=generator
            )
        else:
            accepted, resample = 0, None

        committed = [int(pending)] + [int(t) for t in spec_tokens[:accepted]]

        if K > 0 and accepted < K:
            # Partial accept: `resample` is the corrected token at the rejected position. It becomes the
            # next pending; no new specs are drawn (next round is cold-start).
            next_pending = int(resample)
            next_specs: list[int] = []
            next_draft_probs = None
            was_full_accept = False
        else:
            # Full accept (K=0 cold-start, or K>0 all-accepted). Under the Dream shift, logits[i] predicts
            # input position `i+1`. After committing positions 0..K, the next round's pending sits at
            # input position K+1 (the first MASK slot) → predicted by logits[K]. New specs for the
            # subsequent `num_masks` MASK slots (positions K+2 .. K+num_masks+1... wait,
            # 1 + K + num_masks positions total, 0-indexed up to K+num_masks) are predicted by
            # logits[K+1 .. K+num_masks].
            next_pending_ids, _ = self.sample(
                logits[K : K + 1],
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                generator=generator,
            )
            next_pending = int(next_pending_ids[0].item())

            spec_start = K + 1
            spec_logits = logits[spec_start : spec_start + num_masks]  # [num_masks, vocab] (may be shorter)
            if spec_logits.shape[0] > 0 and num_masks > 0:
                spec_ids, _, spec_q = self.sample(
                    spec_logits,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    return_probs=True,
                    generator=generator,
                )
                next_specs = [int(t.item()) for t in spec_ids]
                next_draft_probs = spec_q
            else:
                next_specs = []
                next_draft_probs = None
            was_full_accept = True

        committed_t = torch.tensor(committed, dtype=torch.long, device=logits.device)

        if not return_dict:
            return committed_t, accepted, next_pending, next_specs, next_draft_probs, was_full_accept
        return IDLMBlockDiffusionSchedulerOutput(
            committed_tokens=committed_t,
            accepted_length=accepted,
            next_pending=next_pending,
            next_specs=next_specs,
            next_draft_probs=next_draft_probs,
            was_full_accept=was_full_accept,
        )

    # ---- training-time noise ------------------------------------------------

    def add_noise(
        self,
        original_samples: torch.LongTensor,
        attention_mask: torch.LongTensor,
        *,
        prompt_length: int,
        mask_token_id: int,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.LongTensor, torch.LongTensor, torch.BoolTensor]:
        """
        I-DLM's *all-masked* training noising.

        The method's key difference from SDAR is that the model sees a fully-masked copy `x_t` concatenated with the
        clean copy `x_0` under strict causal attention. Here we return `noisy` (= fully-masked generation region),
        `original` (= clean for the CE-clean loss on shifted labels), and a boolean `noisy_mask` marking the masked
        positions (where CE-noisy loss is computed).

        Args:
            original_samples (`torch.LongTensor` of shape `(batch_size, seq_len)`):
                Clean token IDs.
            attention_mask (`torch.LongTensor` of shape `(batch_size, seq_len)`):
                Padding mask (1 for valid, 0 for padding).
            prompt_length (`int`):
                Number of prompt tokens to keep unmasked.
            mask_token_id (`int`):
                Token id for the MASK token.
            generator (`torch.Generator`, *optional*):
                Unused (deterministic all-masked regime); kept for interface parity.

        Returns:
            `(noisy, original, noisy_mask)` where positions in `noisy_mask` are where the mask was applied.
        """
        batch_size, seq_len = original_samples.shape
        noisy = original_samples.clone()
        noisy_mask = torch.zeros_like(original_samples, dtype=torch.bool)
        valid = attention_mask.to(dtype=torch.bool)
        if prompt_length < seq_len:
            region = torch.zeros_like(noisy_mask)
            region[:, prompt_length:] = True
            noisy_mask = region & valid
            noisy = torch.where(noisy_mask, torch.full_like(noisy, int(mask_token_id)), noisy)
        return noisy, original_samples.clone(), noisy_mask

    # ---- misc helpers -------------------------------------------------------

    @staticmethod
    def check_should_stop(
        sequences: torch.LongTensor,
        prompt_length: int,
        stop_token_ids: list[int] | None = None,
    ) -> bool:
        if stop_token_ids is None or len(stop_token_ids) == 0:
            return False
        stop_tensor = torch.tensor(stop_token_ids, device=sequences.device, dtype=torch.long)
        return torch.isin(sequences[:, prompt_length:], stop_tensor).any().item()


__all__ = ["IDLMBlockDiffusionScheduler", "IDLMBlockDiffusionSchedulerOutput"]
