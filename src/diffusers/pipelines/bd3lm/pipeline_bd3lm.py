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
from ...schedulers import BD3LMTokenDiffusionScheduler
from ...utils import BaseOutput, logging, replace_example_docstring
from ..pipeline_utils import DiffusionPipeline, DiscreteDiffusionPipelineMixin


logger = logging.get_logger(__name__)


EXAMPLE_DOC_STRING = """
    Examples:
        ```python
        >>> import torch
        >>> from transformers import AutoModelForMaskedLM, GPT2TokenizerFast
        >>> from diffusers import BD3LMPipeline, BD3LMTokenDiffusionScheduler

        >>> model_id = "kuleshov-group/bd3lm-owt-block_size4"
        >>> model = AutoModelForMaskedLM.from_pretrained(model_id, trust_remote_code=True, dtype=torch.bfloat16).cuda()
        >>> tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        >>> scheduler = BD3LMTokenDiffusionScheduler(
        ...     block_size=model.config.block_size,
        ...     mask_token_id=model.config.vocab_size,
        ... )

        >>> pipe = BD3LMPipeline(model=model, scheduler=scheduler, tokenizer=tokenizer)
        >>> output = pipe(gen_length=64, num_inference_steps=64)
        >>> print(output.texts[0])
        ```
"""


@dataclass
class BD3LMPipelineOutput(BaseOutput):
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
    scheduler: BD3LMTokenDiffusionScheduler
    tokenizer: Any

    _callback_tensor_inputs = ["x_accum"]

    def __init__(
        self,
        model: Any,
        scheduler: BD3LMTokenDiffusionScheduler,
        tokenizer: Any | None = None,
    ):
        super().__init__()
        self.register_modules(model=model, scheduler=scheduler, tokenizer=tokenizer)

        # Resolve mask token ID: model.config.mask_index > model.config.vocab_size > tokenizer
        self.mask_token_id: int | None = None
        if hasattr(self.model, "config"):
            self.mask_token_id = getattr(self.model.config, "mask_index", None)
            if self.mask_token_id is None:
                # BD3LM convention: mask_token_id = vocab_size (appended mask token)
                self.mask_token_id = getattr(self.model.config, "vocab_size", None)
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
        if input_ids is not None:
            if input_ids.ndim not in (1, 2):
                raise ValueError(f"`input_ids` must be 1D or 2D, got shape {tuple(input_ids.shape)}.")
            if input_ids.dtype != torch.long:
                raise ValueError(f"`input_ids` must be int64 token IDs, got dtype={input_ids.dtype}.")
        if prompt is not None and input_ids is None and self.tokenizer is None:
            raise ValueError("Tokenizer is required when `input_ids` is not provided.")

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
                Mask token ID. Resolved from model config or scheduler config if not provided.
            eos_token_id (`int`, *optional*):
                EOS token ID for early stopping.
            eos_early_stop (`bool`, defaults to `True`):
                Whether to stop generation when EOS is produced.
            generator (`torch.Generator`, *optional*):
                RNG for reproducibility.
            output_type (`str`, defaults to `"text"`):
                Output format. `"text"` decodes sequences into strings. `"seq"` returns raw token IDs.
            return_dict (`bool`, defaults to `True`):
                Whether to return a [`BD3LMPipelineOutput`] instead of a tuple.
            callback_on_step_end (`Callable` or `PipelineCallback`, *optional*):
                Callback executed after each denoising step.
            callback_on_step_end_tensor_inputs (`List[str]`, *optional*):
                Tensor keys to pass to the callback. Allowed keys: `x_accum`.

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
            mask_token_id = self.scheduler.config.mask_token_id
        if mask_token_id is None:
            raise ValueError("`mask_token_id` must be provided (or available via model config or scheduler config).")

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

        # 3. Set up scheduler timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        num_strides = (gen_length + block_length - 1) // block_length
        self._num_timesteps = num_inference_steps * num_strides

        # Compute sigma from move_chance for model input
        noise_eps = 1e-3
        sigma_max = -torch.log(torch.tensor(noise_eps, device=device, dtype=torch.float64))

        # 4. Semi-autoregressive block diffusion loop
        finished = torch.zeros((batch_size,), device=device, dtype=torch.bool)
        global_step = 0
        x_accum: torch.LongTensor = prompt_ids

        block_progress_bar_config = getattr(self, "_progress_bar_config", {}).copy()
        block_progress_bar_config["position"] = 0
        block_progress_bar_config["desc"] = "Blocks"

        for stride_num in tqdm(range(num_strides), **block_progress_bar_config):
            # Append a new masked block
            masked_block = torch.full((batch_size, block_length), mask_token_id, device=device, dtype=torch.long)
            x_accum = torch.cat([x_accum, masked_block], dim=1)

            # DDPM denoising steps within this block
            p_x0_cache = None

            self.set_progress_bar_config(position=1, leave=False, desc=f"Block {stride_num} Denoising")
            progress_bar = self.progress_bar(total=num_inference_steps)

            for step_idx in range(num_inference_steps):
                # Check if all mask tokens resolved
                if self.scheduler.check_should_stop(x_accum[:, -block_length:], mask_token_id):
                    progress_bar.update(num_inference_steps - step_idx)
                    break

                t = self.scheduler.timesteps[step_idx]

                # Get model predictions (only if cache was invalidated)
                if p_x0_cache is None:
                    # Compute sigma for the model
                    t_tensor = t.unsqueeze(0).expand(batch_size).to(torch.float64)
                    move_chance_t = self.scheduler._compute_move_chance(t_tensor)
                    sigma_t = torch.min(-torch.log(1.0 - move_chance_t), sigma_max).float()

                    model_input = x_accum[:, -block_length:]
                    model_output = self.model(
                        input_ids=model_input,
                        timesteps=sigma_t.to(model_input.device),
                        sample_mode=True,
                    )
                    logits = model_output.logits if hasattr(model_output, "logits") else model_output

                    # Scheduler step: DDPM update
                    scheduler_output = self.scheduler.step(
                        model_output=logits,
                        timestep=t,
                        sample=x_accum,
                        mask_token_id=mask_token_id,
                        nucleus_p=nucleus_p,
                        generator=generator,
                        return_dict=True,
                    )

                    x_accum = scheduler_output.prev_sample
                    p_x0_cache = scheduler_output.p_x0_cache
                else:
                    # Cache is valid — reuse p_x0. We still need to run the scheduler step
                    # but with the cached logits. Convert p_x0_cache back to logits.
                    logits = p_x0_cache.log()

                    scheduler_output = self.scheduler.step(
                        model_output=logits,
                        timestep=t,
                        sample=x_accum,
                        mask_token_id=mask_token_id,
                        nucleus_p=nucleus_p,
                        generator=generator,
                        return_dict=True,
                    )

                    x_accum = scheduler_output.prev_sample
                    p_x0_cache = scheduler_output.p_x0_cache

                # Callback
                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, global_step, step_idx, callback_kwargs)
                    x_accum = callback_outputs.pop("x_accum", x_accum)

                global_step += 1
                progress_bar.update(1)

            progress_bar.close()

            # EOS early stopping
            if eos_early_stop and eos_token_id is not None:
                for b in range(batch_size):
                    generated = x_accum[b, prompt_length:]
                    if (generated == eos_token_id).any():
                        finished[b] = True
                if finished.all():
                    break

        # 5. Post-process output
        total_generated = x_accum.shape[1] - prompt_length
        trim_length = min(total_generated, gen_length)
        sequences = x_accum[:, prompt_length : prompt_length + trim_length]

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
