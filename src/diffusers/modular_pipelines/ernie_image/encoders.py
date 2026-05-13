# Copyright 2025 Baidu ERNIE-Image Team and The HuggingFace Team. All rights reserved.
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

import json

import torch
from transformers import AutoTokenizer, Mistral3Model

from ...configuration_utils import FrozenDict
from ...guiders import ClassifierFreeGuidance
from ...utils import logging
from ...utils.import_utils import is_transformers_version
from ..modular_pipeline import ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam
from .modular_pipeline import ErnieImageModularPipeline


if is_transformers_version("<", "5.0.0"):
    raise ImportError("`ErnieImageModularPipeline` requires `transformers>=5.0.0` for `Ministral3ForCausalLM`.")

from transformers import Ministral3ForCausalLM  # noqa: E402


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class ErnieImagePromptEnhancerStep(ModularPipelineBlocks):
    model_name = "ernie-image"

    @property
    def description(self) -> str:
        return "Prompt enhancer step that rewrites the input prompt using a causal language model (PE)."

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("pe", Ministral3ForCausalLM),
            ComponentSpec("pe_tokenizer", AutoTokenizer),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                "prompt",
                required=True,
                type_hint=str,
                description="The prompt or prompts to guide image generation.",
            ),
            InputParam("height", type_hint=int, description="The height in pixels of the generated image."),
            InputParam("width", type_hint=int, description="The width in pixels of the generated image."),
            InputParam(
                "pe_system_prompt",
                type_hint=str,
                default=None,
                description="Optional system prompt passed to the prompt enhancer.",
            ),
            InputParam(
                "pe_temperature",
                type_hint=float,
                default=0.6,
                description="Sampling temperature used when generating with the prompt enhancer.",
            ),
            InputParam(
                "pe_top_p",
                type_hint=float,
                default=0.95,
                description="Nucleus sampling `top_p` used when generating with the prompt enhancer.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("prompt", type_hint=list, description="The prompt list after prompt-enhancer rewriting."),
            OutputParam("height", type_hint=int, description="The resolved image height in pixels."),
            OutputParam("width", type_hint=int, description="The resolved image width in pixels."),
        ]

    @staticmethod
    def _enhance_prompt(
        pe: Ministral3ForCausalLM,
        pe_tokenizer: AutoTokenizer,
        prompt: str,
        device: torch.device,
        width: int,
        height: int,
        system_prompt: str | None,
        temperature: float,
        top_p: float,
    ) -> str:
        user_content = json.dumps({"prompt": prompt, "width": width, "height": height}, ensure_ascii=False)
        messages = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_content})

        input_text = pe_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        inputs = pe_tokenizer(input_text, return_tensors="pt").to(device)
        output_ids = pe.generate(
            **inputs,
            max_new_tokens=pe_tokenizer.model_max_length,
            do_sample=temperature != 1.0 or top_p != 1.0,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=pe_tokenizer.pad_token_id,
            eos_token_id=pe_tokenizer.eos_token_id,
        )
        generated_ids = output_ids[0][inputs["input_ids"].shape[1] :]
        return pe_tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    @torch.no_grad()
    def __call__(self, components: ErnieImageModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        device = components._execution_device

        prompt = block_state.prompt
        if isinstance(prompt, str):
            prompt = [prompt]

        height = block_state.height or components.default_height
        width = block_state.width or components.default_width

        revised = [
            self._enhance_prompt(
                pe=components.pe,
                pe_tokenizer=components.pe_tokenizer,
                prompt=p,
                device=device,
                width=width,
                height=height,
                system_prompt=block_state.pe_system_prompt,
                temperature=block_state.pe_temperature,
                top_p=block_state.pe_top_p,
            )
            for p in prompt
        ]

        block_state.prompt = revised
        block_state.height = height
        block_state.width = width

        self.set_block_state(state, block_state)
        return components, state


class ErnieImageTextEncoderStep(ModularPipelineBlocks):
    model_name = "ernie-image"

    @property
    def description(self) -> str:
        return (
            "Text encoder step that encodes prompts into variable-length hidden states for the ErnieImage transformer."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("text_encoder", Mistral3Model),
            ComponentSpec("tokenizer", AutoTokenizer),
            ComponentSpec(
                "guider",
                ClassifierFreeGuidance,
                config=FrozenDict({"guidance_scale": 4.0}),
                default_creation_method="from_config",
            ),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("prompt", type_hint=str, description="The prompt or prompts to guide image generation."),
            InputParam(
                "negative_prompt",
                type_hint=str,
                description="The prompt or prompts to avoid during image generation.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "prompt_embeds",
                type_hint=list,
                kwargs_type="denoiser_input_fields",
                description="List of per-prompt text embeddings of shape (T, H).",
            ),
            OutputParam(
                "negative_prompt_embeds",
                type_hint=list,
                kwargs_type="denoiser_input_fields",
                description="List of per-prompt negative text embeddings for classifier-free guidance.",
            ),
        ]

    @staticmethod
    def _encode(
        text_encoder: Mistral3Model,
        tokenizer: AutoTokenizer,
        prompt: list[str],
        device: torch.device,
    ) -> list[torch.Tensor]:
        text_hiddens = []
        for p in prompt:
            ids = tokenizer(p, add_special_tokens=True, truncation=True, padding=False)["input_ids"]
            if len(ids) == 0:
                ids = [tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 0]
            input_ids = torch.tensor([ids], device=device)
            outputs = text_encoder(input_ids=input_ids, output_hidden_states=True)
            text_hiddens.append(outputs.hidden_states[-2][0])
        return text_hiddens

    @torch.no_grad()
    def __call__(self, components: ErnieImageModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        device = components._execution_device

        prompt = block_state.prompt
        if prompt is None:
            prompt = [""]
        if isinstance(prompt, str):
            prompt = [prompt]

        block_state.prompt_embeds = self._encode(
            text_encoder=components.text_encoder,
            tokenizer=components.tokenizer,
            prompt=prompt,
            device=device,
        )

        if components.requires_unconditional_embeds:
            negative_prompt = block_state.negative_prompt
            if negative_prompt is None:
                negative_prompt = ""
            if isinstance(negative_prompt, str):
                negative_prompt = [negative_prompt] * len(prompt)
            if len(negative_prompt) != len(prompt):
                raise ValueError(
                    f"`negative_prompt` must have the same length as `prompt` ({len(prompt)}), "
                    f"got {len(negative_prompt)}."
                )
            block_state.negative_prompt_embeds = self._encode(
                text_encoder=components.text_encoder,
                tokenizer=components.tokenizer,
                prompt=negative_prompt,
                device=device,
            )
        else:
            block_state.negative_prompt_embeds = None

        self.set_block_state(state, block_state)
        return components, state
