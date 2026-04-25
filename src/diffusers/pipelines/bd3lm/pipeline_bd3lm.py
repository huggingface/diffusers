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
        ...     mask_token_id=model.config.vocab_size - 1,
        ... )

        >>> pipe = BD3LMPipeline(model=model, scheduler=scheduler, tokenizer=tokenizer)
        >>> output = pipe(gen_length=64, num_inference_steps=64, nucleus_p=0.9)
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

        # Resolve mask token ID: model.config.mask_index > vocab_size - 1 (BD3LM convention) > tokenizer
        self.mask_token_id: int | None = None
        if hasattr(self.model, "config"):
            self.mask_token_id = getattr(self.model.config, "mask_index", None)
            if self.mask_token_id is None:
                vocab_size = getattr(self.model.config, "vocab_size", None)
                if vocab_size is not None:
                    self.mask_token_id = vocab_size - 1
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
        first_hitting: bool = False,
        variable_length: bool = False,
        entropy_threshold: float = 4.0,
        context_size: int = 1024,
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
            first_hitting (`bool`, defaults to `False`):
                Use the first-hitting time sampler (Zheng et al., 2025) for faster denoising. Instead of uniform
                timestep spacing, concentrates steps where fewer masked tokens remain.
            variable_length (`bool`, defaults to `False`):
                Enable variable-length generation. Stops early when token entropy drops below ``entropy_threshold`` or
                when a second EOS token appears.
            entropy_threshold (`float`, defaults to `4.0`):
                Entropy threshold for variable-length stopping. Only used when ``variable_length=True``.
            context_size (`int`, defaults to `1024`):
                Maximum number of tokens to pass as context to the model (sliding window).
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
        # 1. Resolve defaults and check inputs early
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

        # 4. Semi-autoregressive block diffusion loop
        finished = torch.zeros((batch_size,), device=device, dtype=torch.bool)
        global_step = 0
        x_accum: torch.LongTensor = prompt_ids

        # Initialize KV cache for cross-block context
        if hasattr(self.model, "reset_kv_cache"):
            self.model.reset_kv_cache(eval_batch_size=batch_size)

        block_progress_bar_config = getattr(self, "_progress_bar_config", {}).copy()
        block_progress_bar_config["position"] = 0
        block_progress_bar_config["desc"] = "Blocks"

        for stride_num in tqdm(range(num_strides), **block_progress_bar_config):
            # Append a new masked block
            masked_block = torch.full((batch_size, block_length), mask_token_id, device=device, dtype=torch.long)
            x_accum = torch.cat([x_accum, masked_block], dim=1)

            # Compute the forward window indices (sliding window context)
            end_idx = x_accum.shape[1]
            start_idx = max(end_idx - context_size, 0)

            # DDPM denoising steps within this block
            p_x0_cache = None
            t = 1.0  # continuous timestep, starts at 1

            self.set_progress_bar_config(position=1, leave=False, desc=f"Block {stride_num} Denoising")
            progress_bar = self.progress_bar(total=num_inference_steps)

            for step_idx in range(num_inference_steps):
                # Check if all mask tokens in current block are resolved
                if self.scheduler.check_should_stop(x_accum[:, -block_length:], mask_token_id):
                    progress_bar.update(num_inference_steps - step_idx)
                    break

                # Compute timestep: first-hitting sampler or uniform schedule
                if first_hitting:
                    num_masked = (x_accum[:, -block_length:] == mask_token_id).sum(-1).item()
                    t = self.scheduler.compute_first_hitting_timestep(t, num_masked, generator=generator)
                else:
                    t = self.scheduler.timesteps[step_idx].item()

                # Get model predictions only when p_x0 cache is invalidated
                if p_x0_cache is None:
                    sigma_t = self.scheduler.compute_sigma(t, batch_size)

                    # Pass context window to model (sliding window), only use last block's logits
                    model_input = x_accum[:, start_idx:end_idx][:, -block_length:]
                    model_output = self.model(
                        input_ids=model_input,
                        timesteps=sigma_t.to(model_input.device),
                        sample_mode=True,
                    )
                    logits = model_output.logits if hasattr(model_output, "logits") else model_output
                else:
                    # Reuse cached p_x0 distribution (convert back to log-probs for scheduler)
                    logits = p_x0_cache.log()

                # Scheduler step: DDPM update with subs parameterization
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

            # Store denoised block's KVs into cache for cross-block context
            if hasattr(self.model, "reset_kv_cache"):
                denoised_block = x_accum[:, -block_length:]
                sigma_store = self.scheduler.compute_sigma(self.scheduler.timesteps[0], batch_size)
                self.model(
                    input_ids=denoised_block,
                    timesteps=sigma_store.to(denoised_block.device),
                    sample_mode=True,
                    store_kv=True,
                )

            # EOS early stopping (delegated to scheduler)
            if eos_early_stop and eos_token_id is not None:
                finished = self.scheduler.check_eos_finished(x_accum, prompt_length, eos_token_id, finished)
                if finished.all():
                    break

            # Variable-length stopping (entropy + EOS criteria, checked on generated portion only)
            generated_length = x_accum.shape[1] - prompt_length
            if variable_length and generated_length > 256:
                should_stop, trimmed = self.scheduler.check_variable_length_stop(
                    x_accum, eos_token_id=eos_token_id, entropy_threshold=entropy_threshold
                )
                x_accum = trimmed
                if should_stop:
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
