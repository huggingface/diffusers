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

import html
from typing import List, Optional, Union

import regex as re
import torch
from transformers import AutoTokenizer, UMT5EncoderModel

from ...configuration_utils import FrozenDict
from ...guiders import ClassifierFreeGuidance
from ...utils import is_ftfy_available, logging
from ..modular_pipeline import ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import ComponentSpec, ConfigSpec, InputParam, OutputParam
from .modular_pipeline import WanModularPipeline


if is_ftfy_available():
    import ftfy


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def prompt_clean(text):
    text = whitespace_clean(basic_clean(text))
    return text



def get_t5_prompt_embeds(
    text_encoder: UMT5EncoderModel,
    tokenizer: AutoTokenizer,
    prompt: Union[str, List[str]],
    max_sequence_length: int,
    device: torch.device,
):
    dtype = text_encoder.dtype
    prompt = [prompt] if isinstance(prompt, str) else prompt
    prompt = [prompt_clean(u) for u in prompt]

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
    seq_lens = mask.gt(0).sum(dim=1).long()
    prompt_embeds = text_encoder(text_input_ids.to(device), mask.to(device)).last_hidden_state
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
    prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
    prompt_embeds = torch.stack(
        [torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in prompt_embeds], dim=0
    )

    return prompt_embeds


class WanTextEncoderStep(ModularPipelineBlocks):
    model_name = "wan"

    @property
    def description(self) -> str:
        return "Text Encoder step that generate text_embeddings to guide the video generation"

    @property
    def expected_components(self) -> List[ComponentSpec]:
        return [
            ComponentSpec("text_encoder", UMT5EncoderModel),
            ComponentSpec("tokenizer", AutoTokenizer),
            ComponentSpec(
                "guider",
                ClassifierFreeGuidance,
                config=FrozenDict({"guidance_scale": 5.0}),
                default_creation_method="from_config",
            ),
        ]

    @property
    def inputs(self) -> List[InputParam]:
        return [
            InputParam("prompt"),
            InputParam("negative_prompt"),
            InputParam("max_sequence_length", default=512),
        ]

    @property
    def intermediate_outputs(self) -> List[OutputParam]:
        return [
            OutputParam(
                "prompt_embeds",
                type_hint=torch.Tensor,
                kwargs_type="denoiser_input_fields",
                description="text embeddings used to guide the image generation",
            ),
            OutputParam(
                "negative_prompt_embeds",
                type_hint=torch.Tensor,
                kwargs_type="denoiser_input_fields",
                description="negative text embeddings used to guide the image generation",
            ),
        ]

    @staticmethod
    def check_inputs(block_state):
        if block_state.prompt is not None and (
            not isinstance(block_state.prompt, str) and not isinstance(block_state.prompt, list)
        ):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(block_state.prompt)}")

    @staticmethod
    def encode_prompt(
        components,
        prompt: str,
        device: Optional[torch.device] = None,
        prepare_unconditional_embeds: bool = True,
        negative_prompt: Optional[str] = None,
        max_sequence_length: int = 512,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            prepare_unconditional_embeds (`bool`):
                whether to use prepare unconditional embeddings or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            max_sequence_length (`int`, defaults to `512`):
                The maximum number of text tokens to be used for the generation process.
        """
        device = device or components._execution_device
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt) if prompt is not None else prompt_embeds.shape[0]

        prompt_embeds = get_t5_prompt_embeds(
            text_encoder=components.text_encoder, 
            tokenizer=components.tokenizer, 
            prompt=prompt, 
            max_sequence_length=max_sequence_length,
            device=device,
        )

        if prepare_unconditional_embeds:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )

            negative_prompt_embeds = get_t5_prompt_embeds(
                text_encoder=components.text_encoder, 
                tokenizer=components.tokenizer, 
                prompt=negative_prompt, 
                max_sequence_length=max_sequence_length, 
                device=device,
            )

        return prompt_embeds, negative_prompt_embeds

    @torch.no_grad()
    def __call__(self, components: WanModularPipeline, state: PipelineState) -> PipelineState:
        # Get inputs and intermediates
        block_state = self.get_block_state(state)
        self.check_inputs(block_state)

        block_state.device = components._execution_device

        # Encode input prompt
        (
            block_state.prompt_embeds,
            block_state.negative_prompt_embeds,
        ) = self.encode_prompt(
            components=components,
            prompt=block_state.prompt,
            device=block_state.device,
            prepare_unconditional_embeds=components.requires_unconditional_embeds,
            negative_prompt=block_state.negative_prompt,
            max_sequence_length=block_state.max_sequence_length,
        )

        # Add outputs
        self.set_block_state(state, block_state)
        return components, state
