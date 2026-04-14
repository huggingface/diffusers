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

import math
from dataclasses import dataclass

import torch

from ..configuration_utils import ConfigMixin, register_to_config
from ..utils import BaseOutput
from .scheduling_token_diffusion import _gumbel_argmax
from .scheduling_utils import SchedulerMixin


@dataclass
class BD3LMTokenDiffusionSchedulerOutput(BaseOutput):
    """
    Output class for BD3LM token diffusion scheduling.

    Args:
        prev_sample (`torch.LongTensor` of shape `(batch_size, seq_len)`):
            Updated token sequence after the current denoising step.
        p_x0_cache (`torch.Tensor` of shape `(batch_size, block_size, vocab_size)` or `None`):
            Cached clean-token probability distribution. When `None`, the model should be called again at the next
            step; when not `None`, the cached distribution can be reused.
    """

    prev_sample: torch.LongTensor
    p_x0_cache: torch.Tensor | None


class BD3LMTokenDiffusionScheduler(SchedulerMixin, ConfigMixin):
    """
    Scheduler for Block Discrete Denoising Diffusion Language Models (BD3LM).

    Implements the DDPM-style caching update from BD3LM, which iteratively denoises masked token sequences block by
    block. At each step the scheduler computes posterior transition probabilities q(x_s | x_t, x_0) and samples new
    tokens for currently masked positions while preserving already-unmasked tokens.

    Supports multiple noise schedules: loglinear, cosine, square, square_root, and log.
    """

    order = 1

    @register_to_config
    def __init__(
        self,
        block_size: int = 1024,
        num_inference_steps: int = 1024,
        noise_type: str = "loglinear",
        nucleus_p: float = 1.0,
        mask_token_id: int = 32000,
        eps: float = 1e-3,
    ):
        self.num_inference_steps = num_inference_steps
        self.timesteps: torch.Tensor | None = None
        self._dt: float | None = None

    # ------------------------------------------------------------------
    # Timestep management
    # ------------------------------------------------------------------

    def set_timesteps(self, num_inference_steps: int, device: str | torch.device | None = None) -> None:
        """
        Create linearly spaced timesteps from 1 to 0 (exclusive).

        Args:
            num_inference_steps (`int`):
                Number of denoising steps.
            device (`str` or `torch.device`, *optional*):
                Device for the timestep tensor.
        """
        if num_inference_steps <= 0:
            raise ValueError(f"`num_inference_steps` must be > 0, got {num_inference_steps}.")
        self.num_inference_steps = num_inference_steps
        self.timesteps = torch.linspace(1.0, 0.0, num_inference_steps, device=device)
        self._dt = 1.0 / num_inference_steps

    # ------------------------------------------------------------------
    # Noise schedule utilities
    # ------------------------------------------------------------------

    def _compute_move_chance(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute the probability that a token has been masked (move chance) at continuous time *t*.

        The move chance corresponds to ``1 - alpha(t)``, where ``alpha(t)`` is the probability that a token remains
        clean (i.e. unmasked). This is the same quantity called ``alpha`` in
        :class:`~diffusers.TokenDiffusionScheduler`, so ``move_chance = 1 - alpha(t)``.

        The move chance depends on the configured ``noise_type``:
        - **loglinear**: ``move_chance = t``
        - **cosine**: ``move_chance = 1 - (1 - eps) * cos(t * pi / 2)``
        - **square**: ``move_chance = t ** 2``
        - **square_root**: ``move_chance = t ** 0.5``
        - **log**: ``move_chance = log(1 + t) / log(2)``

        Args:
            t (`torch.Tensor`):
                Continuous timestep values in [0, 1].

        Returns:
            `torch.Tensor`: Move chance at each timestep value, same shape as *t*.
        """
        noise_type = self.config.noise_type
        eps = self.config.eps
        if noise_type == "loglinear":
            return t
        elif noise_type == "cosine":
            return 1.0 - (1.0 - eps) * torch.cos(t * math.pi / 2.0)
        elif noise_type == "square":
            return torch.clamp(t**2, min=eps)
        elif noise_type == "square_root":
            return torch.clamp(t**0.5, min=eps)
        elif noise_type == "log":
            return torch.log1p(t) / math.log(2.0)
        else:
            raise ValueError(
                f"Unknown noise_type '{noise_type}'. Must be one of: loglinear, cosine, square, square_root, log."
            )

    # ------------------------------------------------------------------
    # Nucleus (top-p) filtering
    # ------------------------------------------------------------------

    @staticmethod
    def _nucleus_filtering(probs: torch.Tensor, nucleus_p: float) -> torch.Tensor:
        """
        Apply nucleus (top-p) filtering to a probability distribution.

        Tokens outside the top-p cumulative probability mass are zeroed out and the distribution is renormalised.

        Args:
            probs (`torch.Tensor` of shape `(*, vocab_size)`):
                Token probability distributions (already softmaxed).
            nucleus_p (`float`):
                Cumulative probability threshold. Use 1.0 to disable filtering.

        Returns:
            `torch.Tensor`: Filtered and renormalised probability distributions.
        """
        if nucleus_p >= 1.0:
            return probs
        sorted_probs, sorted_indices = probs.sort(dim=-1, descending=True)
        cumulative_probs = sorted_probs.cumsum(dim=-1)
        nucleus_mask = cumulative_probs <= nucleus_p
        # Always keep at least the top-1 token
        nucleus_mask[..., 0] = True
        sorted_probs = sorted_probs * nucleus_mask
        # Scatter back to original order
        filtered = torch.zeros_like(probs)
        filtered.scatter_(-1, sorted_indices, sorted_probs)
        filtered = filtered / filtered.sum(dim=-1, keepdim=True)
        return filtered

    # ------------------------------------------------------------------
    # Core step
    # ------------------------------------------------------------------

    def step(
        self,
        model_output: torch.Tensor,
        timestep: float | torch.Tensor,
        sample: torch.LongTensor,
        *,
        mask_token_id: int | None = None,
        nucleus_p: float | None = None,
        generator: torch.Generator | None = None,
        return_dict: bool = True,
    ) -> BD3LMTokenDiffusionSchedulerOutput | tuple[torch.LongTensor, torch.Tensor | None]:
        """
        Perform a single DDPM caching denoising step.

        The method implements the BD3LM reverse-process update: given predicted clean-token logits from the model, it
        computes the posterior q(x_s | x_t, x_0), samples new tokens for masked positions, and copies through tokens
        that are already unmasked.

        Args:
            model_output (`torch.Tensor` of shape `(batch_size, seq_len, vocab_size)`):
                Raw logits from the model. Softmax and nucleus filtering are applied internally.
            timestep (`float` or `torch.Tensor`):
                Current continuous timestep *t* (in [0, 1], starting at 1 and decreasing toward 0).
            sample (`torch.LongTensor` of shape `(batch_size, seq_len)`):
                Current noisy token sequence. Masked positions contain ``mask_token_id``.
            mask_token_id (`int`, *optional*):
                Token ID used for masked positions. Defaults to the value from the scheduler config.
            nucleus_p (`float`, *optional*):
                Nucleus sampling threshold. Defaults to the value from the scheduler config.
            generator (`torch.Generator`, *optional*):
                Random number generator for reproducible sampling.
            return_dict (`bool`):
                Whether to return a [`BD3LMTokenDiffusionSchedulerOutput`] or a plain tuple.

        Returns:
            [`BD3LMTokenDiffusionSchedulerOutput`] or `tuple`:
                The denoised sample and the p_x0 cache (``None`` when the sample changed, meaning the cache is
                invalidated and the model must be called again at the next step).
        """
        if mask_token_id is None:
            mask_token_id = self.config.mask_token_id
        if nucleus_p is None:
            nucleus_p = self.config.nucleus_p

        block_size = self.config.block_size
        dt = self._dt if self._dt is not None else 1.0 / self.num_inference_steps

        # Ensure timestep is a tensor
        if not isinstance(timestep, torch.Tensor):
            t = torch.tensor([timestep], device=sample.device, dtype=torch.float64)
        else:
            t = timestep.to(dtype=torch.float64)
        if t.dim() == 0:
            t = t.unsqueeze(0)

        # ------------------------------------------------------------------
        # Compute move chances at t and s (next timestep)
        # ------------------------------------------------------------------
        # Try to get s from the timestep schedule for non-linspace support;
        # fall back to s = t - dt when the schedule is unavailable.
        s = None
        if self.timesteps is not None:
            t_val = t.item() if t.numel() == 1 else None
            if t_val is not None:
                matches = (self.timesteps - t_val).abs()
                idx = int(matches.argmin().item())
                if idx + 1 < len(self.timesteps):
                    s = self.timesteps[idx + 1].to(dtype=torch.float64, device=t.device).unsqueeze(0)
        if s is None:
            s = t - dt

        move_chance_t = self._compute_move_chance(t).to(dtype=torch.float64)
        move_chance_s = self._compute_move_chance(s).to(dtype=torch.float64)

        # Expand to (batch, 1) for broadcasting against (batch, seq_len)
        if move_chance_t.dim() == 1:
            move_chance_t = move_chance_t.unsqueeze(-1)
            move_chance_s = move_chance_s.unsqueeze(-1)

        # mask_prob: probability that a token stays masked at s given it was masked at t
        mask_prob = move_chance_s / move_chance_t  # (batch, 1)

        # ------------------------------------------------------------------
        # Apply subs parameterization and convert to p(x_0)
        # ------------------------------------------------------------------
        logits = model_output[:, -block_size:].to(dtype=torch.float64)

        # Subs parameterization: mask token gets -inf, then log_softmax normalizes.
        # For unmasked positions, the distribution is forced to be the identity.
        logits[..., mask_token_id] = -torch.inf
        log_probs = torch.log_softmax(logits, dim=-1)

        x_current_block = sample[:, -block_size:]
        unmasked = x_current_block != mask_token_id
        log_probs[unmasked] = -torch.inf
        log_probs[unmasked, x_current_block[unmasked]] = 0.0

        # Convert log-probs to probs and apply nucleus filtering
        p_x0 = log_probs.exp()
        p_x0 = self._nucleus_filtering(p_x0, nucleus_p)

        # ------------------------------------------------------------------
        # Compute posterior q(x_s | x_t, x_0) and sample
        # ------------------------------------------------------------------
        # For non-mask tokens: q_xs = p_x0 * (1 - mask_prob), q_xs[mask] = mask_prob
        q_xs = p_x0 * (1.0 - mask_prob.unsqueeze(-1))
        q_xs[..., mask_token_id] = mask_prob.squeeze(-1)

        # Gumbel-argmax categorical sampling
        x_block = _gumbel_argmax(torch.log(q_xs.clamp_min(1e-30)), generator=generator)

        # ------------------------------------------------------------------
        # Copy flag: preserve tokens that are already unmasked
        # ------------------------------------------------------------------
        x_current_block = sample[:, -block_size:]
        is_masked = (x_current_block == mask_token_id).to(dtype=x_block.dtype)
        x_block = (1 - is_masked) * x_current_block + is_masked * x_block

        # Assemble full sequence
        if sample.shape[-1] > block_size:
            prev_sample = torch.cat([sample[:, :-block_size], x_block], dim=-1)
        else:
            prev_sample = x_block

        # ------------------------------------------------------------------
        # Determine p_x0 cache validity
        # ------------------------------------------------------------------
        # If any token changed, invalidate the cache so the model is called again.
        if not torch.equal(prev_sample, sample):
            p_x0_cache = None
        else:
            p_x0_cache = p_x0

        if not return_dict:
            return prev_sample, p_x0_cache
        return BD3LMTokenDiffusionSchedulerOutput(
            prev_sample=prev_sample,
            p_x0_cache=p_x0_cache,
        )

    # ------------------------------------------------------------------
    # Forward (noising) process
    # ------------------------------------------------------------------

    def add_noise(
        self,
        original_samples: torch.LongTensor,
        timesteps: torch.Tensor,
        mask_token_id: int | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.LongTensor:
        """
        Apply the forward noising process: randomly mask tokens with probability determined by the noise schedule.

        Args:
            original_samples (`torch.LongTensor` of shape `(batch_size, seq_len)`):
                Clean token IDs.
            timesteps (`torch.Tensor` of shape `(batch_size,)` or `(batch_size, seq_len)`):
                Continuous timestep values in [0, 1] controlling the amount of noise.
            mask_token_id (`int`, *optional*):
                Token ID to use for masked positions. Defaults to the value from the scheduler config.
            generator (`torch.Generator`, *optional*):
                Random number generator for reproducibility.

        Returns:
            `torch.LongTensor`: Noisy token sequence with the same shape as *original_samples*.
        """
        if mask_token_id is None:
            mask_token_id = self.config.mask_token_id

        move_chance = self._compute_move_chance(timesteps)
        # Expand move_chance to match sample dimensions for broadcasting
        if move_chance.dim() == 1:
            move_chance = move_chance.unsqueeze(-1)  # (batch, 1)

        # Sample uniform noise and mask tokens where noise < move_chance
        uniform_noise = torch.rand(
            original_samples.shape,
            device=original_samples.device,
            dtype=move_chance.dtype,
            generator=generator,
        )
        mask = uniform_noise < move_chance
        noisy_samples = torch.where(mask, mask_token_id, original_samples)
        return noisy_samples

    # ------------------------------------------------------------------
    # Sigma computation (for model input)
    # ------------------------------------------------------------------

    def compute_sigma(self, t: float | torch.Tensor, batch_size: int = 1) -> torch.Tensor:
        """
        Compute the sigma value (noise level) for a given timestep.

        Sigma is derived from the noise schedule's move chance: ``sigma = -log(1 - move_chance)``, clamped at
        ``sigma_max = -log(eps)`` where ``eps`` is the configured epsilon value.

        Args:
            t (`float` or `torch.Tensor`):
                Continuous timestep value in [0, 1].
            batch_size (`int`):
                Batch size for expanding the result.

        Returns:
            `torch.Tensor`: Sigma values of shape ``(batch_size,)`` in float32.
        """
        if not isinstance(t, torch.Tensor):
            t = torch.tensor([t], dtype=torch.float64)
        t = t.to(dtype=torch.float64)
        if t.dim() == 0:
            t = t.unsqueeze(0)
        t = t.expand(batch_size)

        sigma_max = -torch.log(torch.tensor(self.config.eps, device=t.device, dtype=torch.float64))
        move_chance = self._compute_move_chance(t)
        sigma = torch.min(-torch.log(1.0 - move_chance), sigma_max)
        return sigma.float()

    # ------------------------------------------------------------------
    # Stopping criteria
    # ------------------------------------------------------------------

    @staticmethod
    def check_eos_finished(
        sequences: torch.LongTensor,
        prompt_length: int,
        eos_token_id: int,
        finished: torch.BoolTensor,
    ) -> torch.BoolTensor:
        """
        Update per-batch finished flags when EOS tokens appear in the generated portion.

        Args:
            sequences (`torch.LongTensor` of shape `(batch_size, seq_len)`):
                Full accumulated sequence including prompt.
            prompt_length (`int`):
                Number of prompt tokens at the start.
            eos_token_id (`int`):
                EOS token ID.
            finished (`torch.BoolTensor` of shape `(batch_size,)`):
                Current per-batch finished flags.

        Returns:
            `torch.BoolTensor`: Updated finished flags.
        """
        batch_size = sequences.shape[0]
        for b in range(batch_size):
            if finished[b]:
                continue
            generated = sequences[b, prompt_length:]
            if (generated == eos_token_id).any():
                finished[b] = True
        return finished

    @staticmethod
    def check_should_stop(sequences: torch.LongTensor, mask_token_id: int) -> bool:
        """
        Check whether all mask tokens have been resolved.

        Args:
            sequences (`torch.LongTensor`):
                Current token sequences.
            mask_token_id (`int`):
                Token ID used for masked positions.

        Returns:
            `bool`: `True` if no mask tokens remain in *sequences*.
        """
        return (sequences == mask_token_id).sum().item() == 0

    # ------------------------------------------------------------------
    # First-hitting sampler (Zheng et al., 2025)
    # ------------------------------------------------------------------

    @staticmethod
    def compute_first_hitting_timestep(
        t: float,
        num_masked: int,
        generator: torch.Generator | None = None,
    ) -> float:
        """
        Compute the next timestep using the first-hitting time sampler.

        Instead of uniform timestep spacing, this faster (but equivalent) sampler from Zheng et al. (2025) draws ``u ~
        Uniform(0, 1)`` and sets ``t_next = t * u^(1 / num_masked)``. This concentrates denoising effort on steps where
        fewer masked tokens remain.

        Args:
            t (`float`):
                Current continuous timestep in (0, 1].
            num_masked (`int`):
                Number of currently masked tokens in the block.
            generator (`torch.Generator`, *optional*):
                RNG for reproducibility.

        Returns:
            `float`: The next timestep value.
        """
        if num_masked <= 0:
            return 0.0
        u = torch.rand(1, generator=generator).item()
        return t * (u ** (1.0 / num_masked))

    # ------------------------------------------------------------------
    # Variable-length stopping
    # ------------------------------------------------------------------

    @staticmethod
    def check_variable_length_stop(
        sequences: torch.LongTensor,
        eos_token_id: int | None = None,
        entropy_threshold: float = 4.0,
        window_size: int = 256,
    ) -> tuple[bool, torch.LongTensor]:
        """
        Check whether variable-length generation should stop and optionally truncate the sequence.

        Implements the stopping criteria from BD3LM:

        1. **Entropy**: If the token entropy over the last ``window_size`` tokens falls below ``entropy_threshold``,
           stop and truncate to remove the low-entropy tail.
        2. **EOS**: If an EOS token appears in the generated portion (after the first one, which may be BOS), stop and
           truncate at the second EOS occurrence.

        Args:
            sequences (`torch.LongTensor` of shape `(batch_size, seq_len)`):
                Current accumulated token sequences.
            eos_token_id (`int`, *optional*):
                EOS token ID. If ``None``, the EOS criterion is skipped.
            entropy_threshold (`float`, defaults to `4.0`):
                Entropy threshold below which generation stops.
            window_size (`int`, defaults to `256`):
                Number of trailing tokens to compute entropy over.

        Returns:
            `tuple[bool, torch.LongTensor]`: ``(should_stop, sequences)`` where ``sequences`` may be truncated.
        """
        should_stop = False
        truncate_idx = None
        seq_len = sequences.shape[1]

        # Compute entropy of the last window_size tokens
        if seq_len > window_size:
            window = sequences[:, -window_size:]
            _, counts = torch.unique(window, return_counts=True, sorted=False)
            entropy = torch.special.entr(counts.float() / counts.sum()).sum().item()

            if entropy < entropy_threshold:
                should_stop = True
                truncate_idx = seq_len - window_size

        # Check for EOS (second occurrence means generation is done)
        if eos_token_id is not None:
            eos_positions = (sequences[0] == eos_token_id).nonzero(as_tuple=True)[0]
            if len(eos_positions) > 1:
                should_stop = True
                truncate_idx = min(int(eos_positions[1].item()) + 1, seq_len)

        if truncate_idx is not None:
            sequences = sequences[:, :truncate_idx]

        return should_stop, sequences


__all__ = ["BD3LMTokenDiffusionScheduler", "BD3LMTokenDiffusionSchedulerOutput"]
