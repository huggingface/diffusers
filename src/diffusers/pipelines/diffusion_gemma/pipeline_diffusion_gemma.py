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

import inspect
from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import DynamicCache, StaticCache

from ...callbacks import MultiPipelineCallbacks, PipelineCallback
from ...schedulers import BlockRefinementScheduler, DiscreteDDIMScheduler, EntropyBoundScheduler
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
    scheduler: BlockRefinementScheduler | DiscreteDDIMScheduler | EntropyBoundScheduler
    processor: Any

    _callback_tensor_inputs = ["canvas", "logits"]

    def __init__(
        self,
        model: Any,
        scheduler: BlockRefinementScheduler | DiscreteDDIMScheduler | EntropyBoundScheduler,
        processor: Any,
    ):
        super().__init__()
        self.register_modules(model=model, scheduler=scheduler, processor=processor)
        tokenizer = getattr(processor, "tokenizer", processor)
        self.eos_token_id = getattr(tokenizer, "eos_token_id", None) if tokenizer is not None else None

    @property
    def num_timesteps(self):
        return self._num_timesteps

    # --- Prompt encoding ---

    def _prepare_inputs(
        self,
        *,
        prompt: str | list[str] | None,
        messages: list[dict] | None,
        image: Any | list[Any] | None,
        add_generation_prompt: bool,
    ) -> tuple[torch.LongTensor, torch.LongTensor, dict[str, torch.Tensor]]:
        """Tokenize a raw `prompt` (optionally with an `image`) or a raw `messages` conversation into
        `(input_ids, attention_mask, multimodal_inputs)`, where `multimodal_inputs` holds the image tensors the
        processor produced for the encoder prefill."""

        def build_content(text, img):
            if img is None:
                return text
            return [{"type": "image", "image": img}, {"type": "text", "text": text}]

        if messages is None:
            if isinstance(prompt, list):
                images = image if isinstance(image, list) else [image] * len(prompt)
                messages = [[{"role": "user", "content": build_content(p, im)}] for p, im in zip(prompt, images)]
            else:
                messages = [{"role": "user", "content": build_content(prompt, image)}]

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
        multimodal_keys = ("pixel_values", "image_position_ids", "mm_token_type_ids")
        multimodal_inputs = {k: encoded[k] for k in multimodal_keys if k in encoded}
        return ids, mask.to(dtype=torch.long), multimodal_inputs

    def check_inputs(
        self,
        prompt: str | list[str] | None,
        messages: list[dict] | None,
        gen_length: int,
        num_inference_steps: int,
        output_type: str,
        callback_on_step_end_tensor_inputs: list[str] | None,
    ):
        if output_type not in {"seq", "text"}:
            raise ValueError(f"`output_type` must be 'seq' or 'text', got {output_type!r}.")
        if gen_length <= 0:
            raise ValueError(f"`gen_length` must be > 0, got {gen_length}.")
        if num_inference_steps <= 0:
            raise ValueError(f"`num_inference_steps` must be > 0, got {num_inference_steps}.")
        if prompt is None and messages is None:
            raise ValueError("Provide either `prompt` or `messages`.")
        if prompt is not None and messages is not None:
            raise ValueError("Provide either `prompt` or `messages`, not both.")
        if self.processor is None:
            raise ValueError("`processor` is required to encode the prompt.")
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
        messages: list[dict] | None = None,
        image: Any | list[Any] | None = None,
        add_generation_prompt: bool = True,
        gen_length: int = 256,
        num_inference_steps: int = 48,
        temperature: float = 0.0,
        t_min: float | None = 0.4,
        t_max: float | None = 0.8,
        cache_implementation: str | None = None,
        eos_early_stop: bool = True,
        eos_token_id: int | None = None,
        stability_threshold: int = 1,
        confidence_threshold: float | None = 0.005,
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
                Prompt text, wrapped in a chat template and tokenized by the processor. Provide either this or
                `messages`.
            messages (`List[Dict]`, *optional*):
                A raw chat conversation to encode, e.g. `[{"role": "user", "content": "Hello"}]` or a multi-turn /
                multimodal conversation. Use this instead of `prompt` for anything beyond a single user turn.
            image (`PIL.Image.Image` or `List`, *optional*):
                Image(s) to pair with `prompt` for multimodal generation; the processor turns them into the model's
                image inputs. For richer layouts, put the image content directly in `messages`.
            add_generation_prompt (`bool`, defaults to `True`):
                Whether to add the generation prompt when applying the chat template.
            gen_length (`int`, defaults to `256`):
                Number of tokens to generate, rounded up to a multiple of the model's `canvas_length`.
            num_inference_steps (`int`, defaults to `48`):
                Number of denoising steps per canvas.
            temperature (`float`, defaults to `0.0`):
                Sampling temperature. `0.0` is greedy. Other sampling knobs (e.g. `top_k`, `threshold`) are scheduler
                config; set them on the scheduler, e.g. `pipe.scheduler =
                BlockRefinementScheduler.from_config(pipe.scheduler.config, top_k=...)`.
            t_min (`float`, *optional*, defaults to `0.4`):
                Temperature on the last denoising step. The temperature is annealed linearly from `t_max` down to
                `t_min` over the steps, matching the released checkpoint's sampler. Set both `t_min` and `t_max` to
                `None` to use a flat `temperature` instead.
            t_max (`float`, *optional*, defaults to `0.8`):
                Temperature on the first denoising step (see `t_min`).
            cache_implementation (`str`, *optional*):
                Set to `"static"` to prefill the encoder once per block into a persistent `StaticCache` and run the
                decoder against it with fixed shapes, instead of re-encoding the full sequence on every step. The fixed
                shapes also let you compile the decoder, e.g. `pipe.model.model.decoder =
                torch.compile(pipe.model.model.decoder, fullgraph=True)`.
            eos_early_stop (`bool`, defaults to `True`):
                Whether to stop generating further canvases once every sequence has emitted EOS.
            eos_token_id (`int`, *optional*):
                EOS token ID for early stopping. Falls back to the processor's tokenizer.
            stability_threshold (`int`, defaults to `1`):
                Number of consecutive steps the argmax prediction must be unchanged for a block to count as stable.
                Only used when `confidence_threshold` is set.
            confidence_threshold (`float`, *optional*, defaults to `0.005`):
                Leave a block's denoising loop early once every example is stable (see `stability_threshold`) and the
                mean per-token entropy of the prediction is below this value. Speeds up generation at matched quality;
                the default matches the released checkpoint. Set to `None` to always run all `num_inference_steps`.
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
            gen_length=gen_length,
            num_inference_steps=num_inference_steps,
            output_type=output_type,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        )

        prompt_ids, prompt_attention_mask, multimodal_inputs = self._prepare_inputs(
            prompt=prompt,
            messages=messages,
            image=image,
            add_generation_prompt=add_generation_prompt,
        )

        device = self._execution_device
        prompt_ids = prompt_ids.to(device=device)
        prompt_attention_mask = prompt_attention_mask.to(device=device)
        multimodal_inputs = {k: v.to(device=device) for k, v in multimodal_inputs.items()}
        batch_size, prompt_length = prompt_ids.shape

        if eos_token_id is None:
            eos_token_id = self.eos_token_id

        canvas_length = self.model.config.canvas_length
        num_canvases = (gen_length + canvas_length - 1) // canvas_length
        # `num_inference_steps` is the per-block budget of model forwards. With a corrector, fold its sweeps into that
        # budget (as in https://huggingface.co/papers/2605.22765) instead of adding them on top: the first
        # `corrected_steps` predictor steps each run `corrector_steps` extra forwards, so the total stays
        # `num_inference_steps` and the predictor-corrector costs the same as plain ancestral sampling.
        corrector_steps = getattr(self.scheduler.config, "corrector_steps", 0)
        if corrector_steps > 0:
            corrected_steps = (num_inference_steps - 1) // (1 + corrector_steps)
            predictor_steps = num_inference_steps - corrected_steps * corrector_steps
        else:
            corrected_steps = 0
            predictor_steps = num_inference_steps

        # Only `BlockRefinementScheduler` takes a per-call `block_length`; the DiscreteDDIM/EntropyBound schedulers do
        # not, so we pass scheduler-specific kwargs by signature.
        set_timesteps_kwargs = {"device": device}
        if "block_length" in inspect.signature(self.scheduler.set_timesteps).parameters:
            set_timesteps_kwargs["block_length"] = canvas_length
        self.scheduler.set_timesteps(predictor_steps, **set_timesteps_kwargs)
        step_param_names = set(inspect.signature(self.scheduler.step).parameters)
        self._num_timesteps = predictor_steps * num_canvases

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
            torch.compiler.cudagraph_mark_step_begin()
            self.model.model.encoder(
                input_ids=cur_input_ids[:, cached_len:],
                attention_mask=cur_attention_mask,
                past_key_values=past_key_values,
                position_ids=torch.arange(cached_len, cur_len, device=device).unsqueeze(0),
                # Image tensors are consumed by the prompt prefill only; later blocks encode text-only canvases.
                **(multimodal_inputs if cached_len == 0 else {}),
            )

            # Decoder attends bidirectionally over the populated cache (the live padding mask) plus the always-visible
            # canvas; the mask builder sizes this to the cache internally, including the static buffer for a StaticCache.
            decoder_attention_mask = torch.nn.functional.pad(cur_attention_mask.bool(), (0, canvas_length), value=True)
            mask_mapping = self.model.model.decoder.create_diffusion_decoder_attention_mask(
                config=self.model.config,
                inputs_embeds=torch.empty((batch_size, canvas_length, 0), device=device),
                past_key_values=past_key_values,
                decoder_attention_mask=decoder_attention_mask,
            )

            # Start from a fully random canvas and denoise it; the scheduler resets its committed state at step 0.
            canvas = torch.randint(
                0, text_config.vocab_size, (batch_size, canvas_length), device=device, generator=generator
            )
            self_conditioning_logits = None
            # Adaptive stopping history: the last `stability_threshold` argmax predictions of this block's canvas.
            argmax_history = torch.full(
                (max(stability_threshold, 1), batch_size, canvas_length), -1, dtype=torch.long, device=device
            )

            # Inner bar over the predictor steps of this canvas; the first `corrected_steps` also run corrector sweeps.
            step_bar = tqdm(
                range(predictor_steps), desc="denoising", leave=False, **getattr(self, "_progress_bar_config", {})
            )
            for step_idx in step_bar:
                if corrected_steps:
                    step_bar.set_description("denoising (corrector)" if step_idx < corrected_steps else "denoising")
                # Mark a fresh step and clone the logits so a cudagraph-compiled decoder (`mode="reduce-overhead"`)
                # does not overwrite the tensors that self-conditioning and the scheduler read next. Both are no-ops
                # when the decoder is not cudagraph-compiled.
                torch.compiler.cudagraph_mark_step_begin()
                logits = self.model(
                    decoder_input_ids=canvas,
                    past_key_values=past_key_values,
                    self_conditioning_logits=self_conditioning_logits,
                    decoder_attention_mask=mask_mapping,
                    decoder_position_ids=decoder_position_ids,
                ).logits.clone()

                # Anneal the temperature from t_max on the first step down to t_min on the last, like the released
                # checkpoint's sampler. Set both to None for a flat temperature.
                if t_min is not None and t_max is not None:
                    cur_step = predictor_steps - step_idx
                    step_temperature = t_min + (t_max - t_min) * cur_step / predictor_steps
                else:
                    step_temperature = temperature

                # Self-condition on the temperature-shaped logits the scheduler also samples from (reference sampler).
                self_conditioning_logits = logits if step_temperature == 0 else logits / step_temperature

                # Pass only the kwargs the chosen scheduler accepts, so any of the schedulers can drive the pipeline.
                # Per-scheduler sampling knobs (thresholds, top-k, ...) live on the scheduler config, not here.
                step_kwargs = {"mask_token_id": None, "temperature": step_temperature, "generator": generator}
                step_kwargs = {k: v for k, v in step_kwargs.items() if k in step_param_names}
                scheduler_output = self.scheduler.step(
                    model_output=logits, timestep=step_idx, sample=canvas, return_dict=True, **step_kwargs
                )
                canvas = scheduler_output.prev_sample

                # Predictor-corrector (https://huggingface.co/papers/2605.22765): a scheduler exposing `corrector_steps`
                # + `step_correct` refines the canvas with extra Gibbs sweeps on the first `corrected_steps` predictor
                # steps (the budget split computed above). Each sweep needs fresh logits on the updated canvas.
                if step_idx < corrected_steps:
                    for _ in range(corrector_steps):
                        torch.compiler.cudagraph_mark_step_begin()
                        corrector_logits = self.model(
                            decoder_input_ids=canvas,
                            past_key_values=past_key_values,
                            self_conditioning_logits=self_conditioning_logits,
                            decoder_attention_mask=mask_mapping,
                            decoder_position_ids=decoder_position_ids,
                        ).logits.clone()
                        canvas = self.scheduler.step_correct(
                            model_output=corrector_logits, timestep=step_idx, sample=canvas, generator=generator
                        ).prev_sample

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, global_step, step_idx, callback_kwargs)
                    canvas = callback_outputs.pop("canvas", canvas)
                global_step += 1

                # Adaptive stopping: leave this block early once every example's argmax prediction is stable across
                # `stability_threshold` steps and confident (mean per-token entropy below `confidence_threshold`).
                if confidence_threshold is not None:
                    argmax_canvas = logits.argmax(dim=-1)
                    stable = (argmax_history == argmax_canvas[None]).all(dim=-1).all(dim=0)
                    argmax_history = torch.roll(argmax_history, shifts=-1, dims=0)
                    argmax_history[-1] = argmax_canvas
                    confident = torch.distributions.Categorical(logits=logits.float()).entropy().mean(-1) < (
                        confidence_threshold
                    )
                    if bool((stable & confident).all()):
                        # Commit the converged prediction. Ancestral schedulers (e.g. DiscreteDDIM) only clean the
                        # canvas on their final step, so the in-progress canvas may still hold noise tokens; the
                        # denoiser argmax is the converged answer (and equals the canvas for commit-style schedulers).
                        canvas = argmax_canvas
                        break

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
