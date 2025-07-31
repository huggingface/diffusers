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
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

from ...loaders import FluxLoraLoaderMixin, TextualInversionLoaderMixin
from ...utils import USE_PEFT_BACKEND, is_ftfy_available, logging, scale_lora_layers, unscale_lora_layers
from ..modular_pipeline import PipelineBlock, PipelineState
from ..modular_pipeline_utils import ComponentSpec, ConfigSpec, InputParam, OutputParam
from .modular_pipeline import FluxModularPipeline


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


class FluxTextEncoderStep(PipelineBlock):
    model_name = "flux"

    @property
    def description(self) -> str:
        return "Text Encoder step that generate text_embeddings to guide the video generation"

    @property
    def expected_components(self) -> List[ComponentSpec]:
        return [
            ComponentSpec("text_encoder", CLIPTextModel),
            ComponentSpec("tokenizer", CLIPTokenizer),
            ComponentSpec("text_encoder_2", T5EncoderModel),
            ComponentSpec("tokenizer_2", T5TokenizerFast),
        ]

    @property
    def expected_configs(self) -> List[ConfigSpec]:
        return []

    @property
    def inputs(self) -> List[InputParam]:
        return [
            InputParam("prompt"),
            InputParam("prompt_2"),
            InputParam("joint_attention_kwargs"),
        ]

    @property
    def intermediate_outputs(self) -> List[OutputParam]:
        return [
            OutputParam(
                "prompt_embeds",
                type_hint=torch.Tensor,
                description="text embeddings used to guide the image generation",
            ),
            OutputParam(
                "pooled_prompt_embeds",
                type_hint=torch.Tensor,
                description="pooled text embeddings used to guide the image generation",
            ),
            OutputParam(
                "text_ids",
                type_hint=torch.Tensor,
                description="ids from the text sequence for RoPE",
            ),
        ]

    @staticmethod
    def check_inputs(block_state):
        for prompt in [block_state.prompt, block_state.prompt_2]:
            if prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
                raise ValueError(f"`prompt` or `prompt_2` has to be of type `str` or `list` but is {type(prompt)}")

    @staticmethod
    def _get_t5_prompt_embeds(
        components,
        prompt: Union[str, List[str]],
        num_images_per_prompt: int,
        max_sequence_length: int,
        device: torch.device,
    ):
        dtype = components.text_encoder_2.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        if isinstance(components, TextualInversionLoaderMixin):
            prompt = components.maybe_convert_prompt(prompt, components.tokenizer_2)

        text_inputs = components.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids

        untruncated_ids = components.tokenizer_2(prompt, padding="longest", return_tensors="pt").input_ids
        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = components.tokenizer_2.batch_decode(untruncated_ids[:, max_sequence_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because `max_sequence_length` is set to "
                f" {max_sequence_length} tokens: {removed_text}"
            )

        prompt_embeds = components.text_encoder_2(text_input_ids.to(device), output_hidden_states=False)[0]
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        _, seq_len, _ = prompt_embeds.shape

        # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        return prompt_embeds

    @staticmethod
    def _get_clip_prompt_embeds(
        components,
        prompt: Union[str, List[str]],
        num_images_per_prompt: int,
        device: torch.device,
    ):
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        if isinstance(components, TextualInversionLoaderMixin):
            prompt = components.maybe_convert_prompt(prompt, components.tokenizer)

        text_inputs = components.tokenizer(
            prompt,
            padding="max_length",
            max_length=components.tokenizer.model_max_length,
            truncation=True,
            return_overflowing_tokens=False,
            return_length=False,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
        tokenizer_max_length = components.tokenizer.model_max_length
        untruncated_ids = components.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids
        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = components.tokenizer.batch_decode(untruncated_ids[:, tokenizer_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {tokenizer_max_length} tokens: {removed_text}"
            )
        prompt_embeds = components.text_encoder(text_input_ids.to(device), output_hidden_states=False)

        # Use pooled output of CLIPTextModel
        prompt_embeds = prompt_embeds.pooler_output
        prompt_embeds = prompt_embeds.to(dtype=components.text_encoder.dtype, device=device)

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, -1)

        return prompt_embeds

    @staticmethod
    def encode_prompt(
        components,
        prompt: Union[str, List[str]],
        prompt_2: Union[str, List[str]],
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        max_sequence_length: int = 512,
        lora_scale: Optional[float] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to the `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                used in all text-encoders
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            lora_scale (`float`, *optional*):
                A lora scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
        """
        device = device or components._execution_device

        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(components, FluxLoraLoaderMixin):
            components._lora_scale = lora_scale

            # dynamically adjust the LoRA scale
            if components.text_encoder is not None and USE_PEFT_BACKEND:
                scale_lora_layers(components.text_encoder, lora_scale)
            if components.text_encoder_2 is not None and USE_PEFT_BACKEND:
                scale_lora_layers(components.text_encoder_2, lora_scale)

        prompt = [prompt] if isinstance(prompt, str) else prompt

        if prompt_embeds is None:
            prompt_2 = prompt_2 or prompt
            prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2

            # We only use the pooled prompt output from the CLIPTextModel
            pooled_prompt_embeds = FluxTextEncoderStep._get_clip_prompt_embeds(
                components,
                prompt=prompt,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
            )
            prompt_embeds = FluxTextEncoderStep._get_t5_prompt_embeds(
                components,
                prompt=prompt_2,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
            )

        if components.text_encoder is not None:
            if isinstance(components, FluxLoraLoaderMixin) and USE_PEFT_BACKEND:
                # Retrieve the original scale by scaling back the LoRA layers
                unscale_lora_layers(components.text_encoder, lora_scale)

        if components.text_encoder_2 is not None:
            if isinstance(components, FluxLoraLoaderMixin) and USE_PEFT_BACKEND:
                # Retrieve the original scale by scaling back the LoRA layers
                unscale_lora_layers(components.text_encoder_2, lora_scale)

        dtype = components.text_encoder.dtype if components.text_encoder is not None else torch.bfloat16
        text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(device=device, dtype=dtype)

        return prompt_embeds, pooled_prompt_embeds, text_ids

    @torch.no_grad()
    def __call__(self, components: FluxModularPipeline, state: PipelineState) -> PipelineState:
        # Get inputs and intermediates
        block_state = self.get_block_state(state)
        self.check_inputs(block_state)

        block_state.device = components._execution_device

        # Encode input prompt
        block_state.text_encoder_lora_scale = (
            block_state.joint_attention_kwargs.get("scale", None)
            if block_state.joint_attention_kwargs is not None
            else None
        )
        (block_state.prompt_embeds, block_state.pooled_prompt_embeds, block_state.text_ids) = self.encode_prompt(
            components,
            prompt=block_state.prompt,
            prompt_2=None,
            prompt_embeds=None,
            pooled_prompt_embeds=None,
            device=block_state.device,
            num_images_per_prompt=1,  # hardcoded for now.
            lora_scale=block_state.text_encoder_lora_scale,
        )

        # Add outputs
        self.set_block_state(state, block_state)
        return components, state
