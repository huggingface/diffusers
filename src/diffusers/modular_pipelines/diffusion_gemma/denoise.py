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

import inspect

import torch
import torch.nn.functional as F

from ...schedulers import BlockRefinementScheduler
from ...utils import logging
from ...utils.import_utils import is_transformers_version
from ..modular_pipeline import LoopSequentialPipelineBlocks, ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam
from .modular_pipeline import DiffusionGemmaModularPipeline


if is_transformers_version("<", "5.11.0"):
    raise ImportError(
        "`DiffusionGemmaModularPipeline` requires `transformers>=5.11.0` for `DiffusionGemmaForBlockDiffusion`."
    )

from transformers import DiffusionGemmaForBlockDiffusion  # noqa: E402


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class DiffusionGemmaCanvasPrefillStep(ModularPipelineBlocks):
    model_name = "diffusion-gemma"

    @property
    def description(self) -> str:
        return (
            "Canvas step that encodes the tokens not yet in the KV cache (the whole prompt on the first canvas, "
            "the last committed canvas afterwards) and builds the decoder attention mask over the cache"
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("model", DiffusionGemmaForBlockDiffusion),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("multimodal_inputs", type_hint=dict),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "decoder_position_ids",
                type_hint=torch.LongTensor,
                description="Position IDs of the canvas tokens, continuing the running sequence.",
            ),
            OutputParam(
                "decoder_attention_mask_mapping",
                type_hint=object,
                description="The decoder attention mask mapping built over the populated cache plus the canvas.",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components: DiffusionGemmaModularPipeline, block_state, k: int):
        device = block_state.cur_input_ids.device
        canvas_length = block_state.canvas_length
        batch_size = block_state.cur_input_ids.shape[0]
        cur_len = block_state.cur_input_ids.shape[1]

        block_state.decoder_position_ids = torch.arange(cur_len, cur_len + canvas_length, device=device).unsqueeze(0)

        # Encode the tokens not yet in the cache so the decoder reuses the encoder KV cache instead of
        # re-encoding the full sequence.
        cached_len = block_state.past_key_values.get_seq_length()
        torch.compiler.cudagraph_mark_step_begin()
        components.model.model.encoder(
            input_ids=block_state.cur_input_ids[:, cached_len:],
            attention_mask=block_state.cur_attention_mask,
            past_key_values=block_state.past_key_values,
            position_ids=torch.arange(cached_len, cur_len, device=device).unsqueeze(0),
            # Image tensors are consumed by the prompt prefill only; later blocks encode text-only canvases.
            **(block_state.multimodal_inputs if cached_len == 0 and block_state.multimodal_inputs else {}),
        )

        # Decoder attends bidirectionally over the populated cache (the live padding mask) plus the always-visible
        # canvas; the mask builder sizes this to the cache internally, including the static buffer for a StaticCache.
        decoder_attention_mask = F.pad(block_state.cur_attention_mask.bool(), (0, canvas_length), value=True)
        block_state.decoder_attention_mask_mapping = (
            components.model.model.decoder.create_diffusion_decoder_attention_mask(
                config=components.model.config,
                inputs_embeds=torch.empty((batch_size, canvas_length, 0), device=device),
                past_key_values=block_state.past_key_values,
                decoder_attention_mask=decoder_attention_mask,
            )
        )

        return components, block_state


class DiffusionGemmaCanvasNoiseStep(ModularPipelineBlocks):
    model_name = "diffusion-gemma"

    @property
    def description(self) -> str:
        return "Canvas step that initializes the canvas with uniformly random tokens (the uniform corruption prior)"

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("model", DiffusionGemmaForBlockDiffusion),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("generator"),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "canvas",
                type_hint=torch.LongTensor,
                description="The noisy canvas of shape `(batch_size, canvas_length)` being denoised.",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components: DiffusionGemmaModularPipeline, block_state, k: int):
        device = block_state.cur_input_ids.device
        batch_size = block_state.cur_input_ids.shape[0]
        vocab_size = components.model.config.get_text_config(decoder=True).vocab_size

        # Start from a fully random canvas; the scheduler resets its committed state at step 0. `torch.randint`
        # requires the generator and the output device to match, so (as with `randn_tensor`) a CPU generator
        # samples on CPU and the result is moved to `device` afterwards.
        generator = block_state.generator
        rand_device = generator.device if generator is not None else device
        block_state.canvas = torch.randint(
            0, vocab_size, (batch_size, block_state.canvas_length), device=rand_device, generator=generator
        ).to(device)
        return components, block_state


class DiffusionGemmaCanvasDenoiseStep(ModularPipelineBlocks):
    model_name = "diffusion-gemma"

    @property
    def description(self) -> str:
        return (
            "Canvas step that runs the inner refinement loop: each step samples candidate tokens from the "
            "denoiser logits, commits the most confident ones via the scheduler, renoises the rest, and "
            "self-conditions the next step on the previous logits. The first `corrected_steps` predictor steps "
            "also run corrector sweeps, and adaptive stopping leaves the loop early once the prediction is "
            "stable and confident"
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("model", DiffusionGemmaForBlockDiffusion),
            ComponentSpec("scheduler", BlockRefinementScheduler),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                "temperature",
                type_hint=float,
                default=0.0,
                description="Sampling temperature (`0.0` is greedy). Other sampling knobs are scheduler config.",
            ),
            InputParam(
                "stability_threshold",
                type_hint=int,
                default=1,
                description="Consecutive steps the argmax prediction must be unchanged for a canvas to count as "
                "stable. Only used when `confidence_threshold` is set.",
            ),
            InputParam(
                "confidence_threshold",
                type_hint=float,
                default=0.005,
                description="Freeze each example once its prediction is stable and the mean per-token entropy is "
                "below this value, and leave the refinement loop once every example is frozen. Set to `None` to "
                "always run all steps.",
            ),
            InputParam.template("generator"),
        ]

    @torch.no_grad()
    def __call__(self, components: DiffusionGemmaModularPipeline, block_state, k: int):
        device = block_state.cur_input_ids.device
        batch_size, canvas_length = block_state.canvas.shape

        step_param_names = set(inspect.signature(components.scheduler.step).parameters)
        self_conditioning_logits = None
        finished_denoising = torch.zeros(batch_size, dtype=torch.bool, device=device)
        argmax_canvas = block_state.canvas
        # Adaptive stopping history: the last `stability_threshold` argmax predictions of this canvas.
        argmax_history = torch.full(
            (max(block_state.stability_threshold, 1), batch_size, canvas_length),
            -1,
            dtype=torch.long,
            device=device,
        )

        for step_idx in range(block_state.predictor_steps):
            # Mark a fresh step and clone the logits so a cudagraph-compiled decoder does not overwrite the
            # tensors that self-conditioning and the scheduler read next. Both are no-ops otherwise.
            torch.compiler.cudagraph_mark_step_begin()
            logits = components.model(
                decoder_input_ids=block_state.canvas,
                past_key_values=block_state.past_key_values,
                self_conditioning_logits=self_conditioning_logits,
                decoder_attention_mask=block_state.decoder_attention_mask_mapping,
                decoder_position_ids=block_state.decoder_position_ids,
            ).logits.clone()

            # Pass only the kwargs the chosen scheduler accepts, so any of the schedulers can drive the loop.
            step_kwargs = {
                "mask_token_id": None,
                "temperature": block_state.temperature,
                "generator": block_state.generator,
            }
            step_kwargs = {name: value for name, value in step_kwargs.items() if name in step_param_names}
            scheduler_output = components.scheduler.step(
                model_output=logits, timestep=step_idx, sample=block_state.canvas, return_dict=True, **step_kwargs
            )
            block_state.canvas = scheduler_output.prev_sample
            # Self-condition on the logits the scheduler sampled from: temperature-shaped for the reference
            # EntropyBound sampler, the raw denoiser logits for the others.
            pred_logits = scheduler_output.pred_logits
            self_conditioning_logits = pred_logits

            # Predictor-corrector (https://huggingface.co/papers/2605.22765): refine the canvas with extra Gibbs
            # sweeps on the first `corrected_steps` predictor steps. Each sweep needs fresh logits.
            if step_idx < block_state.corrected_steps:
                for _ in range(block_state.corrector_steps):
                    torch.compiler.cudagraph_mark_step_begin()
                    corrector_logits = components.model(
                        decoder_input_ids=block_state.canvas,
                        past_key_values=block_state.past_key_values,
                        self_conditioning_logits=self_conditioning_logits,
                        decoder_attention_mask=block_state.decoder_attention_mask_mapping,
                        decoder_position_ids=block_state.decoder_position_ids,
                    ).logits.clone()
                    block_state.canvas = components.scheduler.step_correct(
                        model_output=corrector_logits,
                        timestep=step_idx,
                        sample=block_state.canvas,
                        generator=block_state.generator,
                    ).prev_sample

            # Adaptive stopping: freeze each example once its scheduler-shaped prediction is stable across
            # `stability_threshold` steps and confident (mean per-token entropy below `confidence_threshold`),
            # then leave the canvas once every example is finished.
            if block_state.confidence_threshold is not None:
                next_argmax_canvas = pred_logits.argmax(dim=-1)
                next_argmax_canvas = torch.where(finished_denoising[:, None], argmax_canvas, next_argmax_canvas)
                stable = (argmax_history == next_argmax_canvas[None]).all(dim=-1).all(dim=0)
                argmax_history = torch.roll(argmax_history, shifts=-1, dims=0)
                argmax_history[-1] = next_argmax_canvas
                confident = torch.distributions.Categorical(logits=pred_logits.float()).entropy().mean(-1) < (
                    block_state.confidence_threshold
                )
                finished_denoising = finished_denoising | (stable & confident)
                argmax_canvas = next_argmax_canvas
                # Commit each converged prediction. Ancestral schedulers only clean the canvas on their final step,
                # so the in-progress canvas may still hold noise tokens; the denoiser argmax is the converged answer
                # (and equals the canvas for commit-style schedulers).
                block_state.canvas = torch.where(finished_denoising[:, None], argmax_canvas, block_state.canvas)
                if bool(finished_denoising.all()):
                    break

        return components, block_state


class DiffusionGemmaCanvasUpdateStep(ModularPipelineBlocks):
    model_name = "diffusion-gemma"

    @property
    def description(self) -> str:
        return "Canvas step that appends the denoised canvas to the running context and tracks EOS emission"

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("eos_token_id", type_hint=int),
        ]

    @torch.no_grad()
    def __call__(self, components: DiffusionGemmaModularPipeline, block_state, k: int):
        block_state.cur_input_ids = torch.cat([block_state.cur_input_ids, block_state.canvas], dim=-1)
        block_state.cur_attention_mask = F.pad(
            block_state.cur_attention_mask, (0, block_state.canvas.shape[1]), value=1
        )

        if block_state.eos_token_id is not None:
            block_state.finished = block_state.finished | (block_state.canvas == block_state.eos_token_id).any(dim=-1)

        return components, block_state


class DiffusionGemmaCanvasLoopWrapper(LoopSequentialPipelineBlocks):
    model_name = "diffusion-gemma"

    @property
    def description(self) -> str:
        return (
            "Loop that generates the text canvas by canvas: each iteration prefills the new context into the KV "
            "cache, initializes a random canvas, denoises it with the inner refinement loop, and appends it to "
            "the running sequence. Generation stops early once every sequence has emitted EOS"
        )

    @property
    def loop_inputs(self) -> list[InputParam]:
        return [
            InputParam("prompt_ids", required=True, type_hint=torch.LongTensor),
            InputParam("prompt_attention_mask", required=True, type_hint=torch.LongTensor),
            InputParam("num_canvases", required=True, type_hint=int),
            InputParam("canvas_length", required=True, type_hint=int),
            InputParam("predictor_steps", required=True, type_hint=int),
            InputParam("corrected_steps", required=True, type_hint=int),
            InputParam("corrector_steps", required=True, type_hint=int),
            InputParam("past_key_values", required=True, type_hint=object),
            InputParam("finished", required=True, type_hint=torch.Tensor),
            InputParam(
                "eos_early_stop",
                type_hint=bool,
                default=True,
                description="Whether to stop generating further canvases once every sequence has emitted EOS.",
            ),
        ]

    @property
    def loop_intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "sequences",
                type_hint=torch.LongTensor,
                description="The generated token IDs of shape `(batch_size, generated_length)`.",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components: DiffusionGemmaModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        device = components.model.device
        block_state.cur_input_ids = block_state.prompt_ids.to(device=device)
        block_state.cur_attention_mask = block_state.prompt_attention_mask.to(device=device)
        block_state.finished = block_state.finished.to(device=device)
        if getattr(block_state, "multimodal_inputs", None):
            block_state.multimodal_inputs = {
                name: value.to(device=device) for name, value in block_state.multimodal_inputs.items()
            }
        prompt_length = block_state.prompt_ids.shape[1]

        for k in range(block_state.num_canvases):
            components, block_state = self.loop_step(components, block_state, k=k)
            if (
                block_state.eos_early_stop
                and block_state.eos_token_id is not None
                and bool(block_state.finished.all())
            ):
                break

        block_state.sequences = block_state.cur_input_ids[:, prompt_length:]

        self.set_block_state(state, block_state)
        return components, state


class DiffusionGemmaDenoiseStep(DiffusionGemmaCanvasLoopWrapper):
    block_classes = [
        DiffusionGemmaCanvasPrefillStep,
        DiffusionGemmaCanvasNoiseStep,
        DiffusionGemmaCanvasDenoiseStep,
        DiffusionGemmaCanvasUpdateStep,
    ]
    block_names = ["prefill", "noise", "denoise", "update"]

    @property
    def description(self) -> str:
        return (
            "Canvas denoise step that iterates over canvases.\nAt each canvas: prefill -> noise -> denoise -> update."
        )
