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
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import DynamicCache, StaticCache

from ...callbacks import MultiPipelineCallbacks, PipelineCallback
from ...schedulers import BlockRefinementScheduler
from ...utils import BaseOutput, logging, replace_example_docstring
from ..pipeline_utils import DiffusionPipeline


logger = logging.get_logger(__name__)


EXAMPLE_DOC_STRING = """
    Examples:
        ```python
        >>> import torch
        >>> from transformers import AutoProcessor, DiffusionGemmaForBlockDiffusion
        >>> from diffusers import BlockRefinementScheduler, DiffusionGemmaPipeline

        >>> model_id = "google/diffusiongemma-26B-A4B-it"
        >>> model = DiffusionGemmaForBlockDiffusion.from_pretrained(model_id, dtype=torch.bfloat16, device_map="auto")
        >>> processor = AutoProcessor.from_pretrained(model_id)
        >>> scheduler = BlockRefinementScheduler()

        >>> pipe = DiffusionGemmaPipeline(model=model, scheduler=scheduler, processor=processor)
        >>> output = pipe(prompt="Why is the sky blue?", gen_length=256)
        >>> print(output.texts[0])
        ```
"""


@dataclass
class DiffusionGemmaPipelineOutput(BaseOutput):
    sequences: torch.LongTensor
    texts: list[str] | None = None


class DiffusionGemmaPipeline(DiffusionPipeline):
    r"""
    Pipeline for DiffusionGemma block-diffusion text generation.

    DiffusionGemma is a block-diffusion encoder-decoder model: a causal encoder reads the clean prompt (and any
    previously generated blocks) into a KV cache, and a bidirectional decoder denoises a fixed-size "canvas" of
    `canvas_length` tokens by cross-attending to that cache. Generation alternates an outer autoregressive loop over
    canvases with an inner denoising loop, where each step samples candidate tokens, commits the most confident ones
    via [`BlockRefinementScheduler`] (uniform corruption mode, `mask_token_id=None`), and renoises the rest.

    The model is expected to be a `DiffusionGemmaForBlockDiffusion` instance exposing `forward(input_ids,
    decoder_input_ids=..., self_conditioning_logits=..., ...)` and returning logits of shape `[batch, canvas_length,
    vocab_size]` over the canvas.
    """

    model: Any
    scheduler: BlockRefinementScheduler
    processor: Any

    _callback_tensor_inputs = ["canvas", "logits"]

    def __init__(
        self,
        model: Any,
        scheduler: BlockRefinementScheduler,
        processor: Any | None = None,
    ):
        super().__init__()
        self.register_modules(model=model, scheduler=scheduler, processor=processor)
        text_config = model.config.get_text_config()
        self.canvas_length = model.config.canvas_length
        self.vocab_size = text_config.vocab_size
        tokenizer = getattr(processor, "tokenizer", processor)
        self.eos_token_id = getattr(tokenizer, "eos_token_id", None) if tokenizer is not None else None

    @property
    def num_timesteps(self):
        return self._num_timesteps

    # --- Prompt encoding ---

    def _prepare_input_ids(
        self,
        *,
        prompt: str | list[str] | None,
        messages: list[dict[str, str]] | None,
        input_ids: torch.LongTensor | None,
        attention_mask: torch.LongTensor | None,
        add_generation_prompt: bool,
        pixel_values: torch.FloatTensor | None = None,
        image_position_ids: torch.LongTensor | None = None,
        mm_token_type_ids: torch.LongTensor | None = None,
    ) -> tuple[torch.LongTensor, torch.LongTensor, dict[str, torch.Tensor]]:
        """Convert prompt/messages/input_ids to `(input_ids, attention_mask, multimodal_inputs)`, where
        `multimodal_inputs` holds any image tensors (`pixel_values`, `image_position_ids`, `mm_token_type_ids`) to
        forward to the encoder prefill."""
        multimodal_keys = ("pixel_values", "image_position_ids", "mm_token_type_ids")
        if input_ids is not None:
            if input_ids.ndim == 1:
                input_ids = input_ids.unsqueeze(0)
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids, dtype=torch.long)
            elif attention_mask.ndim == 1:
                attention_mask = attention_mask.unsqueeze(0)
            multimodal_inputs = {
                "pixel_values": pixel_values,
                "image_position_ids": image_position_ids,
                "mm_token_type_ids": mm_token_type_ids,
            }
            multimodal_inputs = {k: v for k, v in multimodal_inputs.items() if v is not None}
            return input_ids, attention_mask.to(dtype=torch.long), multimodal_inputs

        if self.processor is None:
            raise ValueError("`processor` is required when `input_ids` is not provided.")

        if messages is None:
            if isinstance(prompt, list):
                messages = [[{"role": "user", "content": p}] for p in prompt]
            else:
                messages = [{"role": "user", "content": prompt}]

        encoded = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        )
        ids = encoded["input_ids"]
        mask = encoded.get("attention_mask")
        if mask is None:
            mask = torch.ones_like(ids, dtype=torch.long)
        multimodal_inputs = {k: encoded[k] for k in multimodal_keys if k in encoded}
        return ids, mask.to(dtype=torch.long), multimodal_inputs

    def check_inputs(
        self,
        prompt: str | list[str] | None,
        messages: list[dict[str, str]] | None,
        input_ids: torch.LongTensor | None,
        gen_length: int,
        num_inference_steps: int,
        output_type: str,
        callback_on_step_end_tensor_inputs: list[str] | None,
    ):
        if prompt is None and messages is None and input_ids is None:
            raise ValueError("Provide one of `prompt`, `messages`, or `input_ids`.")
        if prompt is not None and messages is not None:
            raise ValueError("Provide either `prompt` or `messages`, not both.")
        if (prompt is not None or messages is not None) and input_ids is None and self.processor is None:
            raise ValueError("`processor` is required when `input_ids` is not provided.")
        if gen_length <= 0:
            raise ValueError(f"`gen_length` must be > 0, got {gen_length}.")
        if num_inference_steps <= 0:
            raise ValueError(f"`num_inference_steps` must be > 0, got {num_inference_steps}.")
        if output_type not in {"seq", "text"}:
            raise ValueError(f"`output_type` must be 'seq' or 'text', got {output_type!r}.")
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
        messages: list[dict[str, str]] | None = None,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        image_position_ids: torch.LongTensor | None = None,
        mm_token_type_ids: torch.LongTensor | None = None,
        add_generation_prompt: bool = True,
        gen_length: int = 256,
        num_inference_steps: int = 48,
        temperature: float = 0.0,
        top_p: float | None = None,
        top_k: int | None = None,
        threshold: float | None = None,
        editing_threshold: float | None = None,
        cache_implementation: str | None = None,
        eos_early_stop: bool = True,
        eos_token_id: int | None = None,
        generator: torch.Generator | None = None,
        output_type: str = "text",
        return_dict: bool = True,
        callback_on_step_end: Callable[[Any, int, int, dict], dict]
        | PipelineCallback
        | MultiPipelineCallbacks
        | None = None,
        callback_on_step_end_tensor_inputs: list[str] | None = None,
    ) -> DiffusionGemmaPipelineOutput | tuple[torch.LongTensor, list[str] | None]:
        """
        Generate text with block diffusion.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                Prompt text, wrapped in a chat template and tokenized by the processor.
            messages (`List[Dict[str, str]]`, *optional*):
                Chat messages to encode (e.g. `[{"role": "user", "content": "Hello"}]`). Takes precedence over
                `prompt`. Requires a processor with `apply_chat_template`.
            input_ids (`torch.LongTensor`, *optional*):
                Pre-tokenized prompt IDs. Takes precedence over `prompt` and `messages`.
            attention_mask (`torch.LongTensor`, *optional*):
                Per-token mask matching `input_ids`. Only used when `input_ids` is provided.
            pixel_values (`torch.FloatTensor`, *optional*):
                Image features for multimodal prompts, forwarded to the encoder prefill. When the prompt is built from
                `messages` with image content, the processor produces these (and `image_position_ids` /
                `mm_token_type_ids`) automatically; pass them explicitly only alongside pre-tokenized `input_ids`.
            image_position_ids (`torch.LongTensor`, *optional*):
                Patch position coordinates for `pixel_values`.
            mm_token_type_ids (`torch.LongTensor`, *optional*):
                Per-token modality ids marking image vs text positions for `pixel_values`.
            add_generation_prompt (`bool`, defaults to `True`):
                Whether to add the generation prompt when applying the chat template.
            gen_length (`int`, defaults to `256`):
                Number of tokens to generate, rounded up to a multiple of the model's `canvas_length`.
            num_inference_steps (`int`, defaults to `48`):
                Number of denoising steps per canvas.
            temperature (`float`, defaults to `0.0`):
                Sampling temperature. `0.0` is greedy.
            top_p (`float`, *optional*):
                Nucleus sampling cutoff.
            top_k (`int`, *optional*):
                Top-k sampling cutoff.
            threshold (`float`, *optional*):
                Confidence threshold for committing tokens. Defaults to the scheduler's configured value.
            editing_threshold (`float`, *optional*):
                Confidence threshold for re-editing already committed tokens. Defaults to the scheduler's value.
            cache_implementation (`str`, *optional*):
                Set to `"static"` to prefill the encoder once per block into a persistent `StaticCache` and run the
                decoder against it with fixed shapes, instead of re-encoding the full sequence on every step. The fixed
                shapes also let you compile the decoder, e.g. `pipe.model.model.decoder =
                torch.compile(pipe.model.model.decoder, fullgraph=True)`.
            eos_early_stop (`bool`, defaults to `True`):
                Whether to stop generating further canvases once every sequence has emitted EOS.
            eos_token_id (`int`, *optional*):
                EOS token ID for early stopping. Falls back to the processor's tokenizer.
            generator (`torch.Generator`, *optional*):
                RNG for sampling.
            output_type (`str`, defaults to `"text"`):
                `"text"` decodes sequences into strings (requires a processor); `"seq"` returns token IDs only.
            return_dict (`bool`, defaults to `True`):
                Whether to return a [`DiffusionGemmaPipelineOutput`] instead of a tuple.
            callback_on_step_end (`Callable` or `PipelineCallback`, *optional*):
                Callback run after each denoising step with signature `callback_on_step_end(self, step, timestep,
                callback_kwargs)`. Allowed tensor keys: `canvas`, `logits`.
            callback_on_step_end_tensor_inputs (`List[str]`, *optional*):
                Tensor keys to pass to the callback.

        Examples:

        Returns:
            [`~pipelines.diffusion_gemma.pipeline_diffusion_gemma.DiffusionGemmaPipelineOutput`] or `tuple`:
                The generated token IDs (`sequences`) and, for `output_type="text"`, the decoded `texts`.
        """
        if callback_on_step_end is not None and isinstance(
            callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)
        ):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs
        if callback_on_step_end_tensor_inputs is None:
            callback_on_step_end_tensor_inputs = ["canvas"]

        self.check_inputs(
            prompt=prompt,
            messages=messages,
            input_ids=input_ids,
            gen_length=gen_length,
            num_inference_steps=num_inference_steps,
            output_type=output_type,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        )

        prompt_ids, prompt_attention_mask, multimodal_inputs = self._prepare_input_ids(
            prompt=prompt,
            messages=messages,
            input_ids=input_ids,
            attention_mask=attention_mask,
            add_generation_prompt=add_generation_prompt,
            pixel_values=pixel_values,
            image_position_ids=image_position_ids,
            mm_token_type_ids=mm_token_type_ids,
        )

        device = self._execution_device
        prompt_ids = prompt_ids.to(device=device)
        prompt_attention_mask = prompt_attention_mask.to(device=device)
        multimodal_inputs = {k: v.to(device=device) for k, v in multimodal_inputs.items()}
        batch_size, prompt_length = prompt_ids.shape

        if eos_token_id is None:
            eos_token_id = self.eos_token_id

        canvas_length = self.canvas_length
        num_canvases = (gen_length + canvas_length - 1) // canvas_length
        self.scheduler.set_timesteps(num_inference_steps, device=device, block_length=canvas_length)
        self._num_timesteps = num_inference_steps * num_canvases

        cur_input_ids = prompt_ids
        cur_attention_mask = prompt_attention_mask
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        global_step = 0

        # Encode each block of context once into a reusable KV cache and run the decoder against it, rather than
        # re-encoding the whole sequence on every denoising step. The default `DynamicCache` grows with the context;
        # `cache_implementation="static"` uses a fixed-shape `StaticCache` so the decoder can be `torch.compile`-d.
        use_static_cache = cache_implementation == "static"
        text_config = self.model.config.get_text_config(decoder=True)
        max_cache_len = prompt_length + num_canvases * canvas_length
        if use_static_cache:
            past_key_values = StaticCache(config=text_config, max_cache_len=max_cache_len)
        else:
            past_key_values = DynamicCache(config=text_config)

        progress_bar = tqdm(range(num_canvases), **getattr(self, "_progress_bar_config", {}))
        for _ in progress_bar:
            cur_len = cur_input_ids.shape[1]
            decoder_position_ids = torch.arange(cur_len, cur_len + canvas_length, device=device).unsqueeze(0)

            # Encode the tokens not yet in the cache (the whole prompt on the first block, the last committed canvas
            # afterwards), so the decoder reuses the encoder KV cache instead of re-encoding the full sequence.
            cached_len = past_key_values.get_seq_length()
            self.model.model.encoder(
                input_ids=cur_input_ids[:, cached_len:],
                attention_mask=cur_attention_mask,
                past_key_values=past_key_values,
                position_ids=torch.arange(cached_len, cur_len, device=device).unsqueeze(0),
                # Image tensors are consumed by the prompt prefill only; later blocks encode text-only canvases.
                **(multimodal_inputs if cached_len == 0 else {}),
            )

            # Build the 4D decoder mask once per block (outside any compiled region). A static cache spans its full
            # buffer; a dynamic cache spans only the populated length.
            cache_buffer_len = max_cache_len if use_static_cache else cur_len
            decoder_attention_mask = torch.zeros(
                (batch_size, cache_buffer_len + canvas_length), dtype=torch.bool, device=device
            )
            decoder_attention_mask[:, :cur_len] = cur_attention_mask.bool()
            decoder_attention_mask[:, -canvas_length:] = True
            mask_mapping = self.model.model.decoder.create_diffusion_decoder_attention_mask(
                config=text_config,
                inputs_embeds=torch.empty((batch_size, canvas_length, 0), device=device),
                past_key_values=past_key_values,
                decoder_attention_mask=decoder_attention_mask,
            )

            # Start from a fully random canvas and denoise it; the scheduler resets its committed state at step 0.
            canvas = torch.randint(0, self.vocab_size, (batch_size, canvas_length), device=device, generator=generator)
            self_conditioning_logits = None

            for step_idx in range(num_inference_steps):
                logits = self.model(
                    decoder_input_ids=canvas,
                    past_key_values=past_key_values,
                    self_conditioning_logits=self_conditioning_logits,
                    decoder_attention_mask=mask_mapping,
                    decoder_position_ids=decoder_position_ids,
                ).logits
                self_conditioning_logits = logits

                scheduler_output = self.scheduler.step(
                    model_output=logits,
                    timestep=step_idx,
                    sample=canvas,
                    mask_token_id=None,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    threshold=threshold,
                    editing_threshold=editing_threshold,
                    generator=generator,
                    return_dict=True,
                )
                canvas = scheduler_output.prev_sample

                if callback_on_step_end is not None:
                    callback_kwargs = {k: locals()[k] for k in callback_on_step_end_tensor_inputs}
                    callback_outputs = callback_on_step_end(self, global_step, step_idx, callback_kwargs)
                    canvas = callback_outputs.pop("canvas", canvas)
                global_step += 1

            # Append the denoised canvas and extend the context for the next block.
            cur_input_ids = torch.cat([cur_input_ids, canvas], dim=-1)
            cur_attention_mask = F.pad(cur_attention_mask, (0, canvas_length), value=1)

            if eos_early_stop and eos_token_id is not None:
                finished = finished | (canvas == eos_token_id).any(dim=-1)
                if finished.all():
                    break

        progress_bar.close()

        sequences = cur_input_ids[:, prompt_length:]

        # Trim each row at its first EOS so post-EOS canvas tokens don't leak into the decoded text.
        decode_sequences: list[torch.LongTensor] | torch.LongTensor = sequences
        if eos_token_id is not None:
            decode_sequences = [
                seq[: int((seq == eos_token_id).nonzero(as_tuple=True)[0][0]) + 1]
                if (seq == eos_token_id).any()
                else seq
                for seq in sequences
            ]

        texts = None
        if output_type == "text" and self.processor is not None:
            texts = self.processor.batch_decode(decode_sequences, skip_special_tokens=True)

        if not return_dict:
            return sequences, texts
        return DiffusionGemmaPipelineOutput(sequences=sequences, texts=texts)


__all__ = ["DiffusionGemmaPipeline", "DiffusionGemmaPipelineOutput"]
