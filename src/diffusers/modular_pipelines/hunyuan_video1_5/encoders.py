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

import torch
from transformers import ByT5Tokenizer, Qwen2_5_VLTextModel, Qwen2TokenizerFast, T5EncoderModel

from ...configuration_utils import FrozenDict
from ...guiders import ClassifierFreeGuidance
from ...utils import logging
from ..modular_pipeline import ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam
from .modular_pipeline import HunyuanVideo15ModularPipeline

from ...pipelines.hunyuan_video1_5.pipeline_hunyuan_video1_5 import (
    format_text_input,
    extract_glyph_texts,
)


logger = logging.get_logger(__name__)


class HunyuanVideo15TextEncoderStep(ModularPipelineBlocks):
    model_name = "hunyuan-video-1.5"

    @property
    def description(self) -> str:
        return "Dual text encoder step using Qwen2.5-VL (MLLM) and ByT5 (glyph text)"

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("text_encoder", Qwen2_5_VLTextModel),
            ComponentSpec("tokenizer", Qwen2TokenizerFast),
            ComponentSpec("text_encoder_2", T5EncoderModel),
            ComponentSpec("tokenizer_2", ByT5Tokenizer),
            ComponentSpec(
                "guider",
                ClassifierFreeGuidance,
                config=FrozenDict({"guidance_scale": 7.5}),
                default_creation_method="from_config",
            ),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("prompt"),
            InputParam("negative_prompt"),
            InputParam("prompt_embeds", type_hint=torch.Tensor),
            InputParam("prompt_embeds_mask", type_hint=torch.Tensor),
            InputParam("negative_prompt_embeds", type_hint=torch.Tensor),
            InputParam("negative_prompt_embeds_mask", type_hint=torch.Tensor),
            InputParam("prompt_embeds_2", type_hint=torch.Tensor),
            InputParam("prompt_embeds_mask_2", type_hint=torch.Tensor),
            InputParam("negative_prompt_embeds_2", type_hint=torch.Tensor),
            InputParam("negative_prompt_embeds_mask_2", type_hint=torch.Tensor),
            InputParam("num_videos_per_prompt", type_hint=int, default=1),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("prompt_embeds", type_hint=torch.Tensor, kwargs_type="denoiser_input_fields"),
            OutputParam("prompt_embeds_mask", type_hint=torch.Tensor, kwargs_type="denoiser_input_fields"),
            OutputParam("negative_prompt_embeds", type_hint=torch.Tensor, kwargs_type="denoiser_input_fields"),
            OutputParam("negative_prompt_embeds_mask", type_hint=torch.Tensor, kwargs_type="denoiser_input_fields"),
            OutputParam("prompt_embeds_2", type_hint=torch.Tensor, kwargs_type="denoiser_input_fields"),
            OutputParam("prompt_embeds_mask_2", type_hint=torch.Tensor, kwargs_type="denoiser_input_fields"),
            OutputParam("negative_prompt_embeds_2", type_hint=torch.Tensor, kwargs_type="denoiser_input_fields"),
            OutputParam("negative_prompt_embeds_mask_2", type_hint=torch.Tensor, kwargs_type="denoiser_input_fields"),
        ]

    @staticmethod
    def _get_mllm_prompt_embeds(
        text_encoder,
        tokenizer,
        prompt,
        device,
        tokenizer_max_length=1000,
        num_hidden_layers_to_skip=2,
        system_message="You are a helpful assistant. Describe the video by detailing the following aspects: "
        "1. The main content and theme of the video. "
        "2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects. "
        "3. Actions, events, behaviors temporal relationships, physical movement changes of the objects. "
        "4. background environment, light, style and atmosphere. "
        "5. camera angles, movements, and transitions used in the video.",
        crop_start=108,
    ):
        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt = format_text_input(prompt, system_message)

        text_inputs = tokenizer.apply_chat_template(
            prompt,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            padding="max_length",
            max_length=tokenizer_max_length + crop_start,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids.to(device=device)
        prompt_attention_mask = text_inputs.attention_mask.to(device=device)

        prompt_embeds = text_encoder(
            input_ids=text_input_ids,
            attention_mask=prompt_attention_mask,
            output_hidden_states=True,
        ).hidden_states[-(num_hidden_layers_to_skip + 1)]

        if crop_start is not None and crop_start > 0:
            prompt_embeds = prompt_embeds[:, crop_start:]
            prompt_attention_mask = prompt_attention_mask[:, crop_start:]

        return prompt_embeds, prompt_attention_mask

    @staticmethod
    def _get_byt5_prompt_embeds(tokenizer, text_encoder, prompt, device, tokenizer_max_length=256):
        prompt = [prompt] if isinstance(prompt, str) else prompt
        glyph_texts = [extract_glyph_texts(p) for p in prompt]

        prompt_embeds_list = []
        prompt_embeds_mask_list = []

        for glyph_text in glyph_texts:
            if glyph_text is None:
                glyph_text_embeds = torch.zeros(
                    (1, tokenizer_max_length, text_encoder.config.d_model), device=device, dtype=text_encoder.dtype
                )
                glyph_text_embeds_mask = torch.zeros((1, tokenizer_max_length), device=device, dtype=torch.int64)
            else:
                txt_tokens = tokenizer(
                    glyph_text,
                    padding="max_length",
                    max_length=tokenizer_max_length,
                    truncation=True,
                    add_special_tokens=True,
                    return_tensors="pt",
                ).to(device)

                glyph_text_embeds = text_encoder(
                    input_ids=txt_tokens.input_ids,
                    attention_mask=txt_tokens.attention_mask.float(),
                )[0].to(device=device)
                glyph_text_embeds_mask = txt_tokens.attention_mask.to(device=device)

            prompt_embeds_list.append(glyph_text_embeds)
            prompt_embeds_mask_list.append(glyph_text_embeds_mask)

        return torch.cat(prompt_embeds_list, dim=0), torch.cat(prompt_embeds_mask_list, dim=0)

    @staticmethod
    def encode_prompt(
        components,
        prompt,
        device=None,
        dtype=None,
        batch_size=1,
        num_videos_per_prompt=1,
        prompt_embeds=None,
        prompt_embeds_mask=None,
        prompt_embeds_2=None,
        prompt_embeds_mask_2=None,
    ):
        device = device or components._execution_device
        dtype = dtype or components.text_encoder.dtype

        if prompt is None:
            prompt = [""] * batch_size

        prompt = [prompt] if isinstance(prompt, str) else prompt

        if prompt_embeds is None:
            prompt_embeds, prompt_embeds_mask = HunyuanVideo15TextEncoderStep._get_mllm_prompt_embeds(
                tokenizer=components.tokenizer,
                text_encoder=components.text_encoder,
                prompt=prompt,
                device=device,
                tokenizer_max_length=components.tokenizer_max_length,
                system_message=components.system_message,
                crop_start=components.prompt_template_encode_start_idx,
            )

        if prompt_embeds_2 is None:
            prompt_embeds_2, prompt_embeds_mask_2 = HunyuanVideo15TextEncoderStep._get_byt5_prompt_embeds(
                tokenizer=components.tokenizer_2,
                text_encoder=components.text_encoder_2,
                prompt=prompt,
                device=device,
                tokenizer_max_length=components.tokenizer_2_max_length,
            )

        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1).view(
            batch_size * num_videos_per_prompt, seq_len, -1
        )
        prompt_embeds_mask = prompt_embeds_mask.repeat(1, num_videos_per_prompt, 1).view(
            batch_size * num_videos_per_prompt, seq_len
        )

        _, seq_len_2, _ = prompt_embeds_2.shape
        prompt_embeds_2 = prompt_embeds_2.repeat(1, num_videos_per_prompt, 1).view(
            batch_size * num_videos_per_prompt, seq_len_2, -1
        )
        prompt_embeds_mask_2 = prompt_embeds_mask_2.repeat(1, num_videos_per_prompt, 1).view(
            batch_size * num_videos_per_prompt, seq_len_2
        )

        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        prompt_embeds_mask = prompt_embeds_mask.to(dtype=dtype, device=device)
        prompt_embeds_2 = prompt_embeds_2.to(dtype=dtype, device=device)
        prompt_embeds_mask_2 = prompt_embeds_mask_2.to(dtype=dtype, device=device)

        return prompt_embeds, prompt_embeds_mask, prompt_embeds_2, prompt_embeds_mask_2

    @torch.no_grad()
    def __call__(self, components: HunyuanVideo15ModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        device = components._execution_device
        dtype = components.transformer.dtype

        prompt = block_state.prompt
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = getattr(block_state, "prompt_embeds", torch.empty(1)).shape[0]

        (
            block_state.prompt_embeds,
            block_state.prompt_embeds_mask,
            block_state.prompt_embeds_2,
            block_state.prompt_embeds_mask_2,
        ) = self.encode_prompt(
            components=components,
            prompt=prompt,
            device=device,
            dtype=dtype,
            batch_size=batch_size,
            num_videos_per_prompt=getattr(block_state, "num_videos_per_prompt", 1),
            prompt_embeds=getattr(block_state, "prompt_embeds", None),
            prompt_embeds_mask=getattr(block_state, "prompt_embeds_mask", None),
            prompt_embeds_2=getattr(block_state, "prompt_embeds_2", None),
            prompt_embeds_mask_2=getattr(block_state, "prompt_embeds_mask_2", None),
        )

        if components.guider._enabled and components.guider.num_conditions > 1:
            negative_prompt = block_state.negative_prompt
            (
                block_state.negative_prompt_embeds,
                block_state.negative_prompt_embeds_mask,
                block_state.negative_prompt_embeds_2,
                block_state.negative_prompt_embeds_mask_2,
            ) = self.encode_prompt(
                components=components,
                prompt=negative_prompt,
                device=device,
                dtype=dtype,
                batch_size=batch_size,
                num_videos_per_prompt=getattr(block_state, "num_videos_per_prompt", 1),
                prompt_embeds=getattr(block_state, "negative_prompt_embeds", None),
                prompt_embeds_mask=getattr(block_state, "negative_prompt_embeds_mask", None),
                prompt_embeds_2=getattr(block_state, "negative_prompt_embeds_2", None),
                prompt_embeds_mask_2=getattr(block_state, "negative_prompt_embeds_mask_2", None),
            )

        state.set("batch_size", batch_size)

        self.set_block_state(state, block_state)
        return components, state
