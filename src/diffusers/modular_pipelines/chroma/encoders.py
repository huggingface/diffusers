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

import torch
from transformers import T5EncoderModel, T5TokenizerFast

from ...configuration_utils import FrozenDict
from ...guiders import ClassifierFreeGuidance
from ...loaders import FluxLoraLoaderMixin, TextualInversionLoaderMixin
from ...utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers
from ..modular_pipeline import ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam
from .modular_pipeline import ChromaModularPipeline


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def get_t5_prompt_embeds(
    components,
    prompt: str | list[str],
    max_sequence_length: int,
    device: torch.device,
):
    dtype = components.text_encoder.dtype
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if isinstance(components, TextualInversionLoaderMixin):
        prompt = components.maybe_convert_prompt(prompt, components.tokenizer)

    text_inputs = components.tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        return_length=False,
        return_overflowing_tokens=False,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    tokenizer_mask = text_inputs.attention_mask

    tokenizer_mask_device = tokenizer_mask.to(device)

    # unlike FLUX, Chroma uses the attention mask when generating the T5 embedding
    prompt_embeds = components.text_encoder(
        text_input_ids.to(device),
        output_hidden_states=False,
        attention_mask=tokenizer_mask_device,
    )[0]

    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    # for the text tokens, chroma requires that all except the first padding token are masked out during the forward
    # pass through the transformer
    seq_lengths = tokenizer_mask_device.sum(dim=1)
    mask_indices = torch.arange(tokenizer_mask_device.size(1), device=device).unsqueeze(0).expand(batch_size, -1)
    attention_mask = (mask_indices <= seq_lengths.unsqueeze(1)).to(dtype=dtype, device=device)

    return prompt_embeds, attention_mask


class ChromaTextEncoderStep(ModularPipelineBlocks):
    model_name = "chroma"

    @property
    def description(self) -> str:
        return (
            "Text Encoder step that generates T5 text embeddings and the attention masks Chroma uses to mask out "
            "padding tokens (keeping one padding token unmasked, as required by Chroma)"
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("text_encoder", T5EncoderModel),
            ComponentSpec("tokenizer", T5TokenizerFast),
            ComponentSpec(
                "guider",
                ClassifierFreeGuidance,
                config=FrozenDict({"guidance_scale": 5.0}),
                default_creation_method="from_config",
            ),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("prompt"),
            InputParam.template("negative_prompt"),
            InputParam.template("max_sequence_length"),
            InputParam(
                "joint_attention_kwargs",
                type_hint=dict,
                description="Additional kwargs for attention processors; `scale` is used as the text encoder LoRA scale.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam.template("prompt_embeds"),
            OutputParam.template("negative_prompt_embeds"),
            OutputParam(
                "prompt_attention_mask",
                type_hint=torch.Tensor,
                description="Attention mask for the prompt embeddings, with all padding tokens except the first one masked out.",
            ),
            OutputParam(
                "negative_prompt_attention_mask",
                type_hint=torch.Tensor,
                description="Attention mask for the negative prompt embeddings, with all padding tokens except the first one masked out.",
            ),
        ]

    @staticmethod
    def check_inputs(block_state):
        if block_state.prompt is not None and (
            not isinstance(block_state.prompt, str) and not isinstance(block_state.prompt, list)
        ):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(block_state.prompt)}")
        if block_state.max_sequence_length is not None and block_state.max_sequence_length > 512:
            raise ValueError(
                f"`max_sequence_length` cannot be greater than 512 but is {block_state.max_sequence_length}"
            )

    @staticmethod
    def encode_prompt(
        components,
        prompt: str | list[str],
        device: torch.device | None = None,
        prepare_unconditional_embeds: bool = True,
        negative_prompt: str | list[str] | None = None,
        max_sequence_length: int = 512,
        lora_scale: float | None = None,
    ):
        r"""
        Encodes the prompt into T5 hidden states and builds the Chroma attention masks.

        Args:
            prompt (`str` or `list[str]`):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            prepare_unconditional_embeds (`bool`):
                whether to prepare unconditional embeddings or not
            negative_prompt (`str` or `list[str]`, *optional*):
                The prompt or prompts not to guide the image generation.
            max_sequence_length (`int`, defaults to `512`):
                The maximum number of text tokens to be used for the generation process.
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

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        prompt_embeds, prompt_attention_mask = get_t5_prompt_embeds(
            components,
            prompt=prompt,
            max_sequence_length=max_sequence_length,
            device=device,
        )

        negative_prompt_embeds = None
        negative_prompt_attention_mask = None
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

            negative_prompt_embeds, negative_prompt_attention_mask = get_t5_prompt_embeds(
                components,
                prompt=negative_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
            )

        if components.text_encoder is not None:
            if isinstance(components, FluxLoraLoaderMixin) and USE_PEFT_BACKEND:
                # Retrieve the original scale by scaling back the LoRA layers
                unscale_lora_layers(components.text_encoder, lora_scale)

        return prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask

    @torch.no_grad()
    def __call__(self, components: ChromaModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        self.check_inputs(block_state)

        device = components._execution_device

        lora_scale = (
            block_state.joint_attention_kwargs.get("scale", None)
            if block_state.joint_attention_kwargs is not None
            else None
        )
        (
            block_state.prompt_embeds,
            block_state.prompt_attention_mask,
            block_state.negative_prompt_embeds,
            block_state.negative_prompt_attention_mask,
        ) = self.encode_prompt(
            components,
            prompt=block_state.prompt,
            device=device,
            prepare_unconditional_embeds=components.requires_unconditional_embeds,
            negative_prompt=block_state.negative_prompt,
            max_sequence_length=block_state.max_sequence_length,
            lora_scale=lora_scale,
        )

        self.set_block_state(state, block_state)
        return components, state
