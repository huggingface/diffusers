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

from ...callbacks import MultiPipelineCallbacks, PipelineCallback
from ...utils import BaseOutput, logging, replace_example_docstring
from ..pipeline_utils import DiffusionPipeline, DiscreteDiffusionPipelineMixin


logger = logging.get_logger(__name__)


EXAMPLE_DOC_STRING = """
    Examples:
        ```python
        >>> import torch
        >>> from transformers import AutoModelForMaskedLM, AutoTokenizer
        >>> from diffusers import BD3LMPipeline

        >>> model_id = "kuleshov-group/bd3lm-owt-block_size4"
        >>> model = AutoModelForMaskedLM.from_pretrained(model_id, trust_remote_code=True)
        >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
        >>> if tokenizer.mask_token_id is None:
        ...     tokenizer.add_special_tokens({"mask_token": "<|MASK|>"})

        >>> pipe = BD3LMPipeline(model=model, tokenizer=tokenizer)
        >>> pipe = pipe.to("cuda")
        >>> output = pipe(gen_length=256, num_inference_steps=64)
        >>> print(output.texts[0])
        ```
"""


def _sample_categorical(categorical_probs: torch.Tensor) -> torch.LongTensor:
    """Sample from categorical distributions using Gumbel-max trick."""
    gumbel_norm = 1e-10 - (torch.rand_like(categorical_probs) + 1e-10).log()
    return (categorical_probs / gumbel_norm).argmax(dim=-1)


def _nucleus_filter(
    p_x0: torch.Tensor,
    nucleus_p: float,
) -> torch.Tensor:
    """Apply nucleus (top-p) filtering to probability distributions.

    Args:
        p_x0: Probability tensor of shape `(batch, seq_len, vocab_size)`.
        nucleus_p: Cumulative probability threshold for nucleus sampling.

    Returns:
        Filtered and renormalised probability tensor of the same shape.
    """
    if nucleus_p >= 1.0:
        return p_x0

    sorted_probs, sorted_indices = torch.sort(p_x0, dim=-1, descending=True)
    cum_probs = torch.cumsum(sorted_probs, dim=-1)
    nucleus_mask = cum_probs <= nucleus_p
    # Always keep at least the top-1 token.
    nucleus_mask[..., 0] = True
    sorted_probs = sorted_probs * nucleus_mask

    # Scatter filtered probabilities back to original positions.
    filtered = torch.zeros_like(p_x0)
    filtered.scatter_(-1, sorted_indices, sorted_probs)
    filtered = filtered / filtered.sum(dim=-1, keepdim=True)
    return filtered


@dataclass
class BD3LMPipelineOutput(BaseOutput):
    """
    Output class for the BD3LM pipeline.

    Args:
        sequences (`torch.LongTensor` of shape `(batch_size, seq_len)`):
            Generated token ID sequences.
        texts (`list[str]` or `None`):
            Decoded text strings when `output_type="text"` and a tokenizer is available.
    """

    sequences: torch.LongTensor
    texts: list[str] | None = None


class BD3LMPipeline(DiffusionPipeline, DiscreteDiffusionPipelineMixin):
    r"""
    Pipeline for BD3LM (Block Discrete Denoising Diffusion Language Model) text generation via semi-autoregressive
    block diffusion.

    BD3LM generates text by autoregressively appending masked blocks and denoising each block via discrete DDPM
    updates. At each stride, a new block of mask tokens is appended and iteratively denoised using the model's
    predicted token probabilities.

    The model is expected to accept `(input_ids, timesteps, sample_mode)` and return logits of shape `[batch,
    block_length, vocab_size]` when in sample mode.
    """

    model: Any
    tokenizer: Any

    _callback_tensor_inputs = ["x_accum", "block_logits"]

    def __init__(
        self,
        model: Any,
        scheduler: Any,
        tokenizer: Any | None = None,
    ):
        super().__init__()
        self.register_modules(model=model, scheduler=scheduler, tokenizer=tokenizer)

        # Resolve mask token ID from model config or tokenizer.
        self.mask_token_id: int | None = None
        if hasattr(self.model, "config"):
            self.mask_token_id = getattr(self.model.config, "mask_index", None)
        if self.mask_token_id is None and self.tokenizer is not None:
            self.mask_token_id = getattr(self.tokenizer, "mask_token_id", None)

        self.eos_token_id = getattr(self.tokenizer, "eos_token_id", None) if self.tokenizer is not None else None

    @property
    def num_timesteps(self):
        return self._num_timesteps

    def check_inputs(
        self,
        prompt: str | list[str] | None,
        input_ids: torch.LongTensor | None,
        gen_length: int,
        block_length: int,
        num_inference_steps: int,
        nucleus_p: float,
        output_type: str,
        callback_on_step_end: Callable | PipelineCallback | MultiPipelineCallbacks | None,
        callback_on_step_end_tensor_inputs: list[str] | None,
    ):
        # Input source validation
        if prompt is None and input_ids is None:
            # No prompt provided -- unconditional generation starting from BOS.
            pass
        if input_ids is not None:
            if input_ids.ndim not in (1, 2):
                raise ValueError(f"`input_ids` must be 1D or 2D, got shape {tuple(input_ids.shape)}.")
            if input_ids.dtype != torch.long:
                raise ValueError(f"`input_ids` must be int64 token IDs, got dtype={input_ids.dtype}.")
        if prompt is not None and input_ids is None and self.tokenizer is None:
            raise ValueError("Tokenizer is required when `input_ids` is not provided.")

        # Generation parameter validation
        if gen_length <= 0:
            raise ValueError(f"`gen_length` must be > 0, got {gen_length}.")
        if block_length <= 0:
            raise ValueError(f"`block_length` must be > 0, got {block_length}.")
        if num_inference_steps <= 0:
            raise ValueError(f"`num_inference_steps` must be > 0, got {num_inference_steps}.")
        if not (0.0 < nucleus_p <= 1.0):
            raise ValueError(f"`nucleus_p` must be in (0, 1], got {nucleus_p}.")
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

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: str | list[str] | None = None,
        input_ids: torch.LongTensor | None = None,
        gen_length: int = 256,
        block_length: int | None = None,
        num_inference_steps: int = 64,
        nucleus_p: float = 1.0,
        mask_token_id: int | None = None,
        eos_token_id: int | None = None,
        eos_early_stop: bool = True,
        generator: torch.Generator | None = None,
        output_type: str = "text",
        return_dict: bool = True,
        callback_on_step_end: Callable[[int, int, dict], None]
        | PipelineCallback
        | MultiPipelineCallbacks
        | None = None,
        callback_on_step_end_tensor_inputs: list[str] | None = None,
    ) -> BD3LMPipelineOutput | tuple[torch.LongTensor, list[str] | None]:
        """
        Generate text with BD3LM semi-autoregressive block diffusion.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                Prompt text. When provided, it is tokenized and used as a prefix for generation.
            input_ids (`torch.LongTensor`, *optional*):
                Pre-tokenized input IDs. Takes precedence over `prompt`.
            gen_length (`int`, defaults to `256`):
                Total number of new tokens to generate.
            block_length (`int`, *optional*):
                Block size for diffusion. If not provided, reads `model.config.block_size`.
            num_inference_steps (`int`, defaults to `64`):
                Number of DDPM denoising steps per block.
            nucleus_p (`float`, defaults to `1.0`):
                Nucleus sampling probability threshold. Set to `1.0` to disable nucleus filtering.
            mask_token_id (`int`, *optional*):
                Mask token ID. Resolved from `model.config.mask_index` or tokenizer if not provided.
            eos_token_id (`int`, *optional*):
                EOS token ID for early stopping.
            eos_early_stop (`bool`, defaults to `True`):
                Whether to stop generation when EOS is produced.
            generator (`torch.Generator`, *optional*):
                RNG for reproducibility (currently unused; sampling uses Gumbel-max).
            output_type (`str`, defaults to `"text"`):
                Output format. `"text"` decodes sequences into strings. `"seq"` returns raw token IDs.
            return_dict (`bool`, defaults to `True`):
                Whether to return a [`BD3LMPipelineOutput`] instead of a tuple.
            callback_on_step_end (`Callable` or `PipelineCallback`, *optional*):
                Callback executed after each denoising step.
            callback_on_step_end_tensor_inputs (`List[str]`, *optional*):
                Tensor keys to pass to the callback. Allowed keys: `x_accum`, `block_logits`.

        Examples:
        """
        # 0. Resolve defaults
        if block_length is None:
            block_length = getattr(getattr(self.model, "config", None), "block_size", None)
            if block_length is None:
                raise ValueError("`block_length` must be provided or available as `model.config.block_size`.")

        if mask_token_id is None:
            mask_token_id = self.mask_token_id
        if mask_token_id is None:
            raise ValueError(
                "`mask_token_id` must be provided (or available via `model.config.mask_index` or tokenizer)."
            )

        if eos_token_id is None:
            eos_token_id = self.eos_token_id

        # 1. Handle callbacks
        if callback_on_step_end is not None and isinstance(
            callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)
        ):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs
        if callback_on_step_end_tensor_inputs is None:
            callback_on_step_end_tensor_inputs = ["x_accum"]

        self.check_inputs(
            prompt=prompt,
            input_ids=input_ids,
            gen_length=gen_length,
            block_length=block_length,
            num_inference_steps=num_inference_steps,
            nucleus_p=nucleus_p,
            output_type=output_type,
            callback_on_step_end=callback_on_step_end,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        )

        device = self._execution_device

        # 2. Prepare prompt IDs (prefix)
        if prompt is not None or input_ids is not None:
            prompt_ids = self._prepare_input_ids(
                prompt=prompt,
                messages=None,
                input_ids=input_ids,
                use_chat_template=False,
                add_generation_prompt=False,
                chat_template_kwargs=None,
            )
            if prompt_ids.ndim == 1:
                prompt_ids = prompt_ids.unsqueeze(0)
            prompt_ids = prompt_ids.to(device=device)
            batch_size = prompt_ids.shape[0]
        else:
            # Unconditional generation: start with BOS token.
            batch_size = 1
            bos_id = self._resolve_start_token_id()
            if bos_id is None:
                raise ValueError(
                    "No prompt provided and no BOS token found on the tokenizer. "
                    "Provide `prompt`, `input_ids`, or a tokenizer with a BOS token."
                )
            prompt_ids = torch.tensor([[bos_id]], device=device, dtype=torch.long)

        prompt_length = prompt_ids.shape[1]

        # 3. Compute number of strides (blocks to generate)
        num_strides = (gen_length + block_length - 1) // block_length

        self._num_timesteps = num_inference_steps * num_strides

        # 4. Log-linear noise schedule helpers (matching BD3LM's LogLinearNoise).
        #    The noise schedule gives move_chance = t (probability a token is masked at time t).
        #    sigma(t) = -log(1 - t), capped at sigma_max = -log(eps) for eps=1e-3.
        noise_eps = 1e-3
        sigma_max = -torch.log(torch.tensor(noise_eps, device=device, dtype=torch.float64))

        def _move_chance(t: torch.Tensor) -> torch.Tensor:
            """Compute masking probability at time t (log-linear schedule: move_chance = t)."""
            return t

        def _sigma_from_move_chance(p: torch.Tensor) -> torch.Tensor:
            """Convert move_chance to sigma, clamped at sigma_max."""
            return torch.min(-torch.log(1.0 - p), sigma_max)

        # 5. Semi-autoregressive block diffusion loop
        finished = torch.zeros((batch_size,), device=device, dtype=torch.bool)
        global_step = 0

        block_progress_bar_config = getattr(self, "_progress_bar_config", {}).copy()
        block_progress_bar_config["position"] = 0
        block_progress_bar_config["desc"] = "Blocks"

        x_accum: torch.LongTensor | None = None

        for stride_num in tqdm(range(num_strides), **block_progress_bar_config):
            # -- Extend x_accum with a new masked block --
            if stride_num == 0:
                # First block: prompt + masked block
                masked_block = torch.full((batch_size, block_length), mask_token_id, device=device, dtype=torch.long)
                x_accum = torch.cat([prompt_ids, masked_block], dim=1)
                # Set BOS token at position 0 if prompt is just BOS
                if prompt_length == 1:
                    bos_id = self._resolve_start_token_id()
                    if bos_id is not None:
                        x_accum[:, 0] = bos_id
            else:
                masked_block = torch.full((batch_size, block_length), mask_token_id, device=device, dtype=torch.long)
                x_accum = torch.cat([x_accum, masked_block], dim=1)

            # -- Determine the forward window indices --
            # The model sees the last (stride_num + 1) * block_length + prompt_length tokens,
            # but BD3LM in sample_mode only processes the current block.
            # We pass the full accumulated sequence as context.
            end_idx = prompt_length + (stride_num + 1) * block_length
            start_idx = 0  # Use full context
            fwd_indices = torch.arange(start_idx, end_idx, device=device)

            # -- DDPM denoising steps within this block --
            dt = 1.0 / num_inference_steps
            p_x0_cache = None
            timesteps = torch.linspace(1.0, 0.0, num_inference_steps, device=device, dtype=torch.float64)

            self.set_progress_bar_config(position=1, leave=False, desc=f"Block {stride_num} Denoising")
            progress_bar = self.progress_bar(total=num_inference_steps)

            for step_idx in range(num_inference_steps):
                # Check if any mask tokens remain in the current block
                current_block = x_accum[:, -block_length:]
                if (current_block != mask_token_id).all():
                    progress_bar.update(num_inference_steps - step_idx)
                    break

                t = timesteps[step_idx]

                # -- Compute move chances and sigma --
                t_tensor = t.unsqueeze(0).expand(batch_size).to(torch.float64)
                s_tensor = (t - dt).clamp(min=0.0).unsqueeze(0).expand(batch_size).to(torch.float64)

                move_chance_t = _move_chance(t_tensor)
                move_chance_s = _move_chance(s_tensor)
                sigma_t = _sigma_from_move_chance(move_chance_t)

                # mask_prob = move_chance_s / move_chance_t (probability token stays masked)
                mask_prob = (move_chance_s / move_chance_t).unsqueeze(-1)  # (batch, 1)

                # -- Get model predictions --
                x_window = x_accum[:, fwd_indices]
                # Only recompute logits if cache was invalidated
                if p_x0_cache is None:
                    # Pass only the current block to the model in sample_mode
                    model_input = x_window[:, -block_length:]
                    sigma_input = sigma_t.to(model_input.device).float()
                    model_output = self.model(
                        input_ids=model_input,
                        timesteps=sigma_input,
                        sample_mode=True,
                    )
                    logits = model_output.logits if hasattr(model_output, "logits") else model_output
                    logits = logits.to(torch.float64)

                    # Convert logits to probabilities
                    p_x0 = logits.exp()
                    p_x0 = _nucleus_filter(p_x0, nucleus_p)
                    p_x0_cache = p_x0

                # -- DDPM update: construct transition distribution --
                # q(x_s | x_t, x_0): with probability (1 - mask_prob) sample from p(x_0),
                # with probability mask_prob stay as mask token.
                q_xs = p_x0_cache * (1.0 - mask_prob)
                q_xs[:, :, mask_token_id] = mask_prob.squeeze(-1)

                # Sample from the transition distribution
                x_block_new = _sample_categorical(q_xs)

                # Preserve already-unmasked tokens (copy flag)
                current_block = x_accum[:, -block_length:]
                copy_flag = (current_block != mask_token_id).to(x_block_new.dtype)
                x_block_new = copy_flag * current_block + (1 - copy_flag) * x_block_new

                # Check if the block changed (for cache invalidation)
                if not torch.equal(x_block_new, current_block):
                    p_x0_cache = None  # Invalidate cache
                    x_accum = torch.cat([x_accum[:, :-block_length], x_block_new], dim=1)
                else:
                    # Block unchanged, keep cache for next step
                    pass

                # -- Callback --
                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    block_logits = logits if "block_logits" in callback_on_step_end_tensor_inputs else None
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, global_step, step_idx, callback_kwargs)
                    x_accum = callback_outputs.pop("x_accum", x_accum)

                global_step += 1
                progress_bar.update(1)

            progress_bar.close()

            # -- EOS early stopping --
            if eos_early_stop and eos_token_id is not None:
                for b in range(batch_size):
                    generated_so_far = x_accum[b, prompt_length:]
                    eos_positions = (generated_so_far == eos_token_id).nonzero(as_tuple=True)[0]
                    if len(eos_positions) > 0:
                        finished[b] = True
                if finished.all():
                    break

        # 6. Post-process output
        # Trim to prompt_length + gen_length
        total_generated = x_accum.shape[1] - prompt_length
        trim_length = min(total_generated, gen_length)
        sequences = x_accum[:, prompt_length : prompt_length + trim_length]

        # Truncate at first EOS if present
        if eos_token_id is not None and batch_size == 1:
            eos_positions = (sequences[0] == eos_token_id).nonzero(as_tuple=True)[0]
            if len(eos_positions) > 0:
                sequences = sequences[:, : int(eos_positions[0].item()) + 1]

        texts = None
        if output_type == "text" and self.tokenizer is not None:
            texts = self.tokenizer.batch_decode(sequences, skip_special_tokens=True)

        if not return_dict:
            return sequences.to(device=device), texts
        return BD3LMPipelineOutput(sequences=sequences.to(device=device), texts=texts)


__all__ = ["BD3LMPipeline", "BD3LMPipelineOutput"]
