# Copyright 2026 The HuggingFace Team. All rights reserved.
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
from typing import Optional, Union

import torch

from ..configuration_utils import ConfigMixin, register_to_config
from .scheduling_utils import DiscreteSchedulerOutput, SchedulerMixin


class DiscreteDDIMScheduler(SchedulerMixin, ConfigMixin):
    """
    Discrete DDIM scheduler for the uniform corruption process, following "Structured Denoising Diffusion Models in
    Discrete State-Spaces" (D3PM, https://huggingface.co/papers/2107.03006).

    On the linear schedule the survival probability of a clean token at time `t` is `alpha(t) = 1 - t`. One denoising
    step from time `t` to `s < t` samples every block position from the exact posterior `q(x_s | x_t, x0)`, which for
    the uniform kernel decomposes into three routes: jump to the predicted clean token `x0`, stay on the current token,
    or jump to a uniformly random token. Unlike masked diffusion, there is no mask token; uncommitted positions carry
    random tokens.

    An optional predictor-corrector mode follows "Uniform Diffusion Models Revisited: Leave-One-Out Denoiser and
    Absorbing State Reformulation" via the leave-one-out (LOO) denoiser (https://huggingface.co/papers/2605.22765).
    When `corrector_steps > 0`, the pipeline runs that many Gibbs corrector sweeps after each predictor step (see
    [`~DiscreteDDIMScheduler.step_correct`]), resampling the least-confident positions from the one-coordinate
    conditional `Cat(alpha_s * x0_loo + (1 - alpha_s) / K)` while holding the rest fixed, which leaves the marginal
    `p_s` invariant and improves generation at no training cost.

    Args:
        num_inference_steps (`int`, defaults to 32):
            The number of denoising steps, defining the linear time grid the posterior is evaluated on.
        temperature (`float`, defaults to 0.0):
            Sampling temperature applied to the logits when drawing the predicted clean tokens. `0.0` takes the argmax.
            The reported `sampled_probs` are always measured on the unscaled distribution, so confidence thresholds do
            not move with this value.
        corrector_steps (`int`, defaults to 0):
            Number of Gibbs corrector sweeps run after each predictor step. `0` recovers plain ancestral DDIM sampling.
        corrector_k (`int`, defaults to 1):
            Number of positions resampled per corrector sweep.
        corrector_selection (`str`, defaults to `"lowest_log_margin"`):
            How the resampled positions are chosen: `"lowest_log_margin"`, `"lowest_maxprob"`, `"lowest_current_prob"`,
            or `"random"`.
        corrector_selection_tau (`float`, defaults to 1.0):
            Temperature of the Gumbel-top-k position selection (lower is greedier).
    """

    order = 1

    @register_to_config
    def __init__(
        self,
        num_inference_steps: int = 32,
        temperature: float = 0.0,
        corrector_steps: int = 0,
        corrector_k: int = 1,
        corrector_selection: str = "lowest_log_margin",
        corrector_selection_tau: float = 1.0,
    ):
        self._step_index = None
        self._begin_index = None
        self.set_timesteps(num_inference_steps)

    @property
    def step_index(self):
        """
        The index counter for current timestep. It will increase 1 after each scheduler step.
        """
        return self._step_index

    @property
    def begin_index(self):
        """
        The index for the first timestep. It should be set from pipeline with `set_begin_index` method.
        """
        return self._begin_index

    # Copied from diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler.set_begin_index
    def set_begin_index(self, begin_index: int = 0):
        """
        Sets the begin index for the scheduler. This function should be run from pipeline before the inference.

        Args:
            begin_index (`int`, defaults to `0`):
                The begin index for the scheduler.
        """
        self._begin_index = begin_index

    def set_timesteps(self, num_inference_steps: int, device: str | torch.device | None = None) -> None:
        """
        Set the discrete timestep grid the posterior is evaluated on.

        Discrete diffusion parametrizes time as the corruption level: `t = 1` is fully noised, `t = 0` is clean, and
        `timesteps` decreases, matching both the discrete diffusion literature and the `sigmas` of the continuous
        schedulers. The survival probability of a clean token is `alpha = 1 - t`.

        `timesteps` runs from `1.0` down to `1 / num_inference_steps`, so that the step after the last one lands on the
        clean end `t = 0` where `alpha_s = 1` and the predicted clean tokens are committed deterministically.

        `timesteps` is the public loop variable — what a pipeline iterates and what a t-conditioned denoiser would
        consume. The grid arithmetic itself is derived from `step_index` rather than from these floats: `alpha_t =
        step_index / num_inference_steps`. That is not a micro-optimization but a correctness requirement, because `1 -
        i / n` in float32 is not exactly `(n - i) / n` for an `n` that is not a power of two, and the commit quota
        `ceil(alpha_s * block_length)` flips by one token when the product lands just above an integer (`n = 3`,
        `block_length = 15` gives 6 instead of 5). Integer indices keep every boundary exact; the continuous schedulers
        index `self.sigmas[self.step_index]` for the same reason.

        Args:
            num_inference_steps (`int`):
                The number of denoising steps.
            device (`str` or `torch.device`, *optional*):
                The device the timesteps should be moved to.
        """
        if num_inference_steps <= 0:
            raise ValueError(f"`num_inference_steps` must be > 0, got {num_inference_steps}.")
        self.num_inference_steps = num_inference_steps
        self.timesteps = (
            1.0 - torch.arange(num_inference_steps, device=device, dtype=torch.float32) / num_inference_steps
        )
        self._step_index = None
        self._begin_index = None

    # Copied from diffusers.schedulers.scheduling_flow_match_euler_discrete.FlowMatchEulerDiscreteScheduler.index_for_timestep
    def index_for_timestep(
        self,
        timestep: Union[float, torch.FloatTensor],
        schedule_timesteps: Optional[torch.FloatTensor] = None,
    ) -> int:
        """
        Get the index for the given timestep.

        Args:
            timestep (`float` or `torch.FloatTensor`):
                The timestep to find the index for.
            schedule_timesteps (`torch.FloatTensor`, *optional*):
                The schedule timesteps to validate against. If `None`, the scheduler's timesteps are used.

        Returns:
            `int`:
                The index of the timestep.
        """
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps

        indices = (schedule_timesteps == timestep).nonzero()

        # The sigma index that is taken for the **very** first `step`
        # is always the second index (or the last index if there is only 1)
        # This way we can ensure we don't accidentally skip a sigma in
        # case we start in the middle of the denoising schedule (e.g. for image-to-image)
        pos = 1 if len(indices) > 1 else 0

        return indices[pos].item()

    # Copied from diffusers.schedulers.scheduling_flow_match_euler_discrete.FlowMatchEulerDiscreteScheduler._init_step_index
    def _init_step_index(self, timestep: Union[float, torch.FloatTensor]) -> None:
        if self.begin_index is None:
            if isinstance(timestep, torch.Tensor):
                timestep = timestep.to(self.timesteps.device)
            self._step_index = self.index_for_timestep(timestep)
        else:
            self._step_index = self._begin_index

    @staticmethod
    def _sample_from_logits(
        logits: torch.Tensor,
        *,
        temperature: float,
        generator: torch.Generator | None,
    ) -> tuple[torch.LongTensor, torch.Tensor]:
        """
        Draw one token per position, returning the tokens and their probabilities.

        The draw is temperature-scaled; the returned probabilities are gathered from the softmax of `logits` *before*
        that scaling, so a confidence threshold does not move with `temperature`.
        """
        if temperature < 0:
            raise ValueError(f"`temperature` must be >= 0, got {temperature}.")

        vocab_size = logits.shape[-1]
        flat_logits = logits.reshape(-1, vocab_size)
        probs = torch.softmax(flat_logits.float(), dim=-1)

        if temperature == 0.0:
            token = flat_logits.argmax(dim=-1, keepdim=True)
        else:
            scaled_probs = torch.softmax(flat_logits.float() / temperature, dim=-1)
            token = torch.multinomial(scaled_probs, num_samples=1, generator=generator)

        token_prob = torch.gather(probs, -1, token)
        return token.view(*logits.shape[:-1]), token_prob.view(*logits.shape[:-1])

    @staticmethod
    def _to_loo_logits(logits: torch.Tensor, tokens: torch.LongTensor, alpha: float) -> torch.Tensor:
        """
        Convert plain-denoiser logits to the leave-one-out posterior for the uniform kernel.

        Subtracts `log(1 + K * alpha / (1 - alpha))` from the observed token's logit (eq. 13 of
        https://huggingface.co/papers/2605.22765); renormalization happens in the following softmax.
        """
        if alpha <= 0.0 or alpha >= 1.0:
            return logits
        delta = math.log1p(logits.shape[-1] * alpha / (1.0 - alpha))
        shifted = logits.clone()
        src = torch.full((*tokens.shape, 1), -delta, dtype=shifted.dtype, device=shifted.device)
        shifted.scatter_add_(-1, tokens.unsqueeze(-1), src)
        return shifted

    def step(
        self,
        model_output: torch.Tensor,
        timestep: float | torch.Tensor,
        sample: torch.LongTensor,
        *,
        generator: torch.Generator | None = None,
        return_dict: bool = True,
    ) -> DiscreteSchedulerOutput | tuple:
        """
        Sample the next block from the posterior `q(x_s | x_t, x0)` of the uniform corruption process.

        With `a = alpha_t / alpha_s` (survival probability from `s` to `t`) and `b = alpha_s`, the posterior mass of
        each route is

            clean: `b * (1 - a) / K + a * b * 1[x_t = x0]`, stay: `a * (1 - b) / K`, noise: `(1 - a) * (1 - b) / K`,

        so the last step (`b = 1`) deterministically commits the predicted clean tokens.

        Args:
            model_output (`torch.Tensor` of shape `(batch_size, block_length, vocab_size)`):
                Raw logits from the model for the current block.
            timestep (`float` or `torch.Tensor`):
                The current corruption level, one entry of [`~DiscreteDDIMScheduler.timesteps`].
            sample (`torch.LongTensor` of shape `(batch_size, block_length)`):
                Current block token IDs `x_t`.
            generator (`torch.Generator`, *optional*):
                RNG for sampling.
            return_dict (`bool`):
                Whether to return a [`DiscreteSchedulerOutput`] or a plain tuple.
        """
        if self.step_index is None:
            self._init_step_index(timestep)

        sampled_tokens, sampled_probs = self._sample_from_logits(
            model_output, temperature=float(self.config.temperature), generator=generator
        )

        vocab_size = model_output.shape[-1]
        num_steps = self.num_inference_steps
        # `alpha = 1 - t` is the survival probability of a clean token, so it increases towards the clean end and
        # reaches `alpha_s = 1` on the final step, committing the predicted clean tokens deterministically. Both
        # alphas come from the integer `step_index` rather than from `timestep`: see `set_timesteps` for why the
        # float form is not safe here.
        alpha_t = self.step_index / num_steps
        alpha_s = (self.step_index + 1) / num_steps
        survival = alpha_t / alpha_s

        same = (sample == sampled_tokens).float()
        clean_mass = alpha_s * (1 - survival) / vocab_size + survival * alpha_s * same
        stay_mass = survival * (1 - alpha_s) / vocab_size * torch.ones_like(same)
        noise_mass = (1 - survival) * (1 - alpha_s) / vocab_size * torch.ones_like(same)

        route_probs = torch.stack([clean_mass, stay_mass, noise_mass], dim=-1)
        route_probs = route_probs / route_probs.sum(dim=-1, keepdim=True)
        routes = torch.multinomial(route_probs.view(-1, 3), num_samples=1, generator=generator).view_as(sample)

        random_tokens = torch.randint(
            low=0, high=vocab_size, size=sample.shape, device=sample.device, generator=generator
        )
        prev_sample = torch.where(routes == 0, sampled_tokens, sample)
        prev_sample = torch.where(routes == 2, random_tokens, prev_sample)

        # The clean route is the one that adopts the predicted token; on the final step (`alpha_s = 1`) it is the
        # only route with mass, so every position commits.
        committed_mask = routes == 0

        self._step_index += 1

        if not return_dict:
            return prev_sample, sampled_tokens, sampled_probs, model_output, committed_mask, None
        return DiscreteSchedulerOutput(
            prev_sample=prev_sample,
            pred_original_sample=sampled_tokens,
            sampled_probs=sampled_probs,
            pred_logits=model_output,
            committed_mask=committed_mask,
        )

    def _select_positions(
        self, sample: torch.LongTensor, cond_log_probs: torch.Tensor, generator: torch.Generator | None
    ) -> torch.LongTensor:
        """Pick `corrector_k` positions per row to resample, least-confident first (Gumbel-top-k without replacement)."""
        selection = self.config.corrector_selection
        batch_size, seq_len = sample.shape
        k_eff = min(max(1, int(self.config.corrector_k)), seq_len)

        if selection == "random":
            scores = torch.rand(batch_size, seq_len, device=sample.device, generator=generator)
            return torch.topk(scores, k=k_eff, dim=-1).indices

        if selection == "lowest_maxprob":
            confidence = -cond_log_probs.max(dim=-1).values
        elif selection == "lowest_current_prob":
            confidence = -torch.gather(cond_log_probs, -1, sample.unsqueeze(-1)).squeeze(-1)
        elif selection == "lowest_log_margin":
            log_current = torch.gather(cond_log_probs, -1, sample.unsqueeze(-1)).squeeze(-1)
            alt = cond_log_probs.clone().scatter_(-1, sample.unsqueeze(-1), float("-inf"))
            confidence = -(log_current - alt.max(dim=-1).values)
        else:
            raise ValueError(f"Unknown `corrector_selection`: {selection!r}.")

        keys = confidence / float(self.config.corrector_selection_tau)
        u = torch.rand(keys.shape, device=keys.device, generator=generator).clamp_(1e-12, 1.0 - 1e-12)
        keys = keys + (-torch.log(-torch.log(u)))
        return torch.topk(keys, k=k_eff, dim=-1).indices

    def step_correct(
        self,
        model_output: torch.Tensor,
        timestep: float | torch.Tensor,
        sample: torch.LongTensor,
        *,
        generator: torch.Generator | None = None,
        return_dict: bool = True,
    ) -> DiscreteSchedulerOutput | tuple:
        """
        Run one Gibbs corrector sweep at the post-predictor time `s`, following the leave-one-out predictor-corrector
        of https://huggingface.co/papers/2605.22765.

        The model logits (recomputed on the current `sample`) are converted to the LOO denoiser, the one-coordinate
        conditional `p_s(x^l | x^{-l}) = Cat(alpha_s * x0_loo + (1 - alpha_s) / K)` is formed, the least-confident
        `corrector_k` positions are selected, and those positions are resampled while the rest are held fixed. The
        sweep preserves `p_s`, so it refines the sample without changing its marginal and needs no extra training.

        Args:
            model_output (`torch.Tensor` of shape `(batch_size, block_length, vocab_size)`):
                Raw logits from the model recomputed on the current (post-predictor) `sample`.
            timestep (`float` or `torch.Tensor`):
                The corruption level of the predictor step just completed; the corrector runs at the following grid
                point `s`. Resolved through [`~DiscreteDDIMScheduler.index_for_timestep`] rather than read off
                `step_index`, so the sweep does not depend on how many predictor steps have run.
            sample (`torch.LongTensor` of shape `(batch_size, block_length)`):
                Current block token IDs to refine.
            generator (`torch.Generator`, *optional*):
                RNG for sampling.
            return_dict (`bool`):
                Whether to return a [`DiscreteSchedulerOutput`] or a plain tuple.
        """
        # The corrector acts at the cleaner time `s` reached by the predictor, i.e. one grid point on.
        alpha_s = (self.index_for_timestep(timestep) + 1) / self.num_inference_steps
        vocab_size = model_output.shape[-1]

        # Match the reference corrector, which forms the conditional in float64 (the LOO correction reaches ~log(K)).
        loo_logits = self._to_loo_logits(model_output.double(), sample, alpha_s)
        loo_log_probs = torch.log_softmax(loo_logits, dim=-1)
        log_uniform = math.log1p(-alpha_s) - math.log(vocab_size)
        cond_log_probs = torch.logaddexp(
            math.log(alpha_s) + loo_log_probs, torch.full_like(loo_log_probs, log_uniform)
        )

        positions = self._select_positions(sample, cond_log_probs, generator)
        rows = torch.arange(sample.shape[0], device=sample.device).unsqueeze(-1).expand_as(positions)
        chosen_probs = cond_log_probs[rows, positions].exp()
        resampled = torch.multinomial(
            chosen_probs.reshape(-1, vocab_size), num_samples=1, generator=generator
        ).view_as(positions)

        prev_sample = sample.clone()
        prev_sample[rows, positions] = resampled

        # A corrector sweep has no separate `x0` prediction — it resamples coordinates of `p_s` in place — so the
        # refined tokens *are* the prediction. Both this and `sampled_probs` are reported at full sequence length,
        # measured under the one-coordinate conditional, so they satisfy the `DiscreteSchedulerOutput` shape
        # contract; the previous per-position variants were `(batch_size, corrector_k)` and disagreed with it.
        sampled_probs = torch.gather(cond_log_probs, -1, prev_sample.unsqueeze(-1)).squeeze(-1).exp()
        committed_mask = torch.zeros_like(sample, dtype=torch.bool)
        committed_mask[rows, positions] = True

        if not return_dict:
            return prev_sample, prev_sample, sampled_probs, model_output, committed_mask, None
        return DiscreteSchedulerOutput(
            prev_sample=prev_sample,
            pred_original_sample=prev_sample,
            sampled_probs=sampled_probs,
            pred_logits=model_output,
            committed_mask=committed_mask,
        )


__all__ = ["DiscreteDDIMScheduler"]
