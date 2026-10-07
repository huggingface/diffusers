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
from transformers import DynamicCache, ProcessorMixin, StaticCache

from ...schedulers import BlockRefinementScheduler
from ...utils import logging
from ...utils.import_utils import is_transformers_version
from ..modular_pipeline import ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam
from .modular_pipeline import DiffusionGemmaModularPipeline


if is_transformers_version("<", "5.11.0"):
    raise ImportError(
        "`DiffusionGemmaModularPipeline` requires `transformers>=5.11.0` for `DiffusionGemmaForBlockDiffusion`."
    )

from transformers import DiffusionGemmaForBlockDiffusion  # noqa: E402


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class DiffusionGemmaPrepareGenerationStep(ModularPipelineBlocks):
    model_name = "diffusion-gemma"

    @property
    def description(self) -> str:
        return (
            "Prepare step that sizes the generation into canvases, creates the encoder KV cache, and resolves "
            "the EOS token used for early stopping and trimming"
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("model", DiffusionGemmaForBlockDiffusion),
            ComponentSpec("processor", ProcessorMixin),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                "prompt_ids",
                required=True,
                type_hint=torch.LongTensor,
                description="Tokenized prompt of shape `(batch_size, prompt_length)`.",
            ),
            InputParam(
                "gen_length",
                type_hint=int,
                default=256,
                description="Number of tokens to generate, rounded up to a multiple of the model's `canvas_length`.",
            ),
            InputParam(
                "cache_implementation",
                type_hint=str,
                description='Set to `"static"` to use a fixed-shape `StaticCache` so the decoder can be compiled.',
            ),
            InputParam(
                "eos_token_id",
                type_hint=int,
                description="EOS token ID for early stopping. Falls back to the processor's tokenizer.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "canvas_length",
                type_hint=int,
                description="The model's canvas length, i.e. the number of tokens denoised per block.",
            ),
            OutputParam("num_canvases", type_hint=int, description="Number of canvases to generate."),
            OutputParam(
                "past_key_values",
                type_hint=object,
                description="The encoder KV cache reused across canvases and denoising steps.",
            ),
            OutputParam(
                "eos_token_id",
                type_hint=int,
                description="The resolved EOS token ID (user-provided or from the processor's tokenizer).",
            ),
            OutputParam(
                "finished",
                type_hint=torch.Tensor,
                description="Per-example flags marking sequences that already emitted EOS.",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components: DiffusionGemmaModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        if block_state.gen_length <= 0:
            raise ValueError(f"`gen_length` must be > 0, got {block_state.gen_length}.")

        canvas_length = components.model.config.canvas_length
        num_canvases = (block_state.gen_length + canvas_length - 1) // canvas_length

        batch_size, prompt_length = block_state.prompt_ids.shape
        text_config = components.model.config.get_text_config(decoder=True)
        max_cache_len = prompt_length + num_canvases * canvas_length
        if block_state.cache_implementation == "static":
            past_key_values = StaticCache(config=text_config, max_cache_len=max_cache_len)
        else:
            past_key_values = DynamicCache(config=text_config)

        eos_token_id = block_state.eos_token_id
        if eos_token_id is None:
            tokenizer = getattr(components.processor, "tokenizer", components.processor)
            eos_token_id = getattr(tokenizer, "eos_token_id", None)

        block_state.canvas_length = canvas_length
        block_state.num_canvases = num_canvases
        block_state.past_key_values = past_key_values
        block_state.eos_token_id = eos_token_id
        block_state.finished = torch.zeros(batch_size, dtype=torch.bool, device=block_state.prompt_ids.device)

        self.set_block_state(state, block_state)
        return components, state


class DiffusionGemmaSetTimestepsStep(ModularPipelineBlocks):
    model_name = "diffusion-gemma"

    @property
    def description(self) -> str:
        return (
            "Step that splits the per-canvas forward budget into predictor and corrector steps and configures "
            "the scheduler's refinement schedule"
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("scheduler", BlockRefinementScheduler),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                "num_inference_steps",
                type_hint=int,
                default=48,
                description="Number of denoising steps per canvas, i.e. the per-canvas budget of model forwards.",
            ),
            InputParam("canvas_length", required=True, type_hint=int),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("predictor_steps", type_hint=int, description="Predictor steps run per canvas."),
            OutputParam(
                "corrected_steps",
                type_hint=int,
                description="Number of leading predictor steps that also run corrector sweeps.",
            ),
            OutputParam(
                "corrector_steps",
                type_hint=int,
                description="Corrector sweeps run after each of the first `corrected_steps` predictor steps.",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components: DiffusionGemmaModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        num_inference_steps = block_state.num_inference_steps
        if num_inference_steps <= 0:
            raise ValueError(f"`num_inference_steps` must be > 0, got {num_inference_steps}.")

        # `num_inference_steps` is the per-block budget of model forwards. With a corrector, fold its sweeps into
        # that budget (as in https://huggingface.co/papers/2605.22765) instead of adding them on top: the first
        # `corrected_steps` predictor steps each run `corrector_steps` extra forwards, so the total stays
        # `num_inference_steps` and the predictor-corrector costs the same as plain ancestral sampling.
        corrector_steps = getattr(components.scheduler.config, "corrector_steps", 0)
        if corrector_steps > 0:
            corrected_steps = (num_inference_steps - 1) // (1 + corrector_steps)
            predictor_steps = num_inference_steps - corrected_steps * corrector_steps
        else:
            corrected_steps = 0
            predictor_steps = num_inference_steps

        # Only `BlockRefinementScheduler` takes a per-call `block_length`; the DiscreteDDIM/EntropyBound schedulers
        # do not, so we pass scheduler-specific kwargs by signature.
        set_timesteps_kwargs = {"device": None}
        if "block_length" in inspect.signature(components.scheduler.set_timesteps).parameters:
            set_timesteps_kwargs["block_length"] = block_state.canvas_length
        components.scheduler.set_timesteps(predictor_steps, **set_timesteps_kwargs)

        block_state.predictor_steps = predictor_steps
        block_state.corrected_steps = corrected_steps
        block_state.corrector_steps = corrector_steps

        self.set_block_state(state, block_state)
        return components, state
