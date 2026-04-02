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
from ...pipelines.hunyuan_video1_5.pipeline_hunyuan_video1_5 import HunyuanVideo15Pipeline
from ...utils import logging
from ..modular_pipeline import ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam
from .modular_pipeline import HunyuanVideo15ModularPipeline


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

    # Copied from HunyuanVideo15Pipeline.encode_prompt
    @torch.no_grad()
    def __call__(self, components: HunyuanVideo15ModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        device = components._execution_device
        dtype = components.transformer.dtype

        prompt = block_state.prompt
        negative_prompt = block_state.negative_prompt
        num_videos_per_prompt = block_state.num_videos_per_prompt

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        elif getattr(block_state, "prompt_embeds", None) is not None:
            batch_size = block_state.prompt_embeds.shape[0]
        else:
            batch_size = 1

        # Encode positive prompt - copied from HunyuanVideo15Pipeline.encode_prompt
        prompt_embeds = getattr(block_state, "prompt_embeds", None)
        prompt_embeds_mask = getattr(block_state, "prompt_embeds_mask", None)
        prompt_embeds_2 = getattr(block_state, "prompt_embeds_2", None)
        prompt_embeds_mask_2 = getattr(block_state, "prompt_embeds_mask_2", None)

        if prompt is None:
            prompt = [""] * batch_size
        prompt = [prompt] if isinstance(prompt, str) else prompt

        if prompt_embeds is None:
            prompt_embeds, prompt_embeds_mask = HunyuanVideo15Pipeline._get_mllm_prompt_embeds(
                tokenizer=components.tokenizer,
                text_encoder=components.text_encoder,
                prompt=prompt,
                device=device,
                tokenizer_max_length=components.tokenizer_max_length,
                system_message=components.system_message,
                crop_start=components.prompt_template_encode_start_idx,
            )

        if prompt_embeds_2 is None:
            prompt_embeds_2, prompt_embeds_mask_2 = HunyuanVideo15Pipeline._get_byt5_prompt_embeds(
                tokenizer=components.tokenizer_2,
                text_encoder=components.text_encoder_2,
                prompt=prompt,
                device=device,
                tokenizer_max_length=components.tokenizer_2_max_length,
            )

        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1).view(batch_size * num_videos_per_prompt, seq_len, -1)
        prompt_embeds_mask = prompt_embeds_mask.repeat(1, num_videos_per_prompt, 1).view(batch_size * num_videos_per_prompt, seq_len)

        _, seq_len_2, _ = prompt_embeds_2.shape
        prompt_embeds_2 = prompt_embeds_2.repeat(1, num_videos_per_prompt, 1).view(batch_size * num_videos_per_prompt, seq_len_2, -1)
        prompt_embeds_mask_2 = prompt_embeds_mask_2.repeat(1, num_videos_per_prompt, 1).view(batch_size * num_videos_per_prompt, seq_len_2)

        block_state.prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        block_state.prompt_embeds_mask = prompt_embeds_mask.to(dtype=dtype, device=device)
        block_state.prompt_embeds_2 = prompt_embeds_2.to(dtype=dtype, device=device)
        block_state.prompt_embeds_mask_2 = prompt_embeds_mask_2.to(dtype=dtype, device=device)

        # Encode negative prompt if guider needs it
        if components.requires_unconditional_embeds:
            neg_prompt_embeds = getattr(block_state, "negative_prompt_embeds", None)
            neg_prompt_embeds_mask = getattr(block_state, "negative_prompt_embeds_mask", None)
            neg_prompt_embeds_2 = getattr(block_state, "negative_prompt_embeds_2", None)
            neg_prompt_embeds_mask_2 = getattr(block_state, "negative_prompt_embeds_mask_2", None)

            neg_prompt = negative_prompt
            if neg_prompt is None:
                neg_prompt = [""] * batch_size
            neg_prompt = [neg_prompt] if isinstance(neg_prompt, str) else neg_prompt

            if neg_prompt_embeds is None:
                neg_prompt_embeds, neg_prompt_embeds_mask = HunyuanVideo15Pipeline._get_mllm_prompt_embeds(
                    tokenizer=components.tokenizer,
                    text_encoder=components.text_encoder,
                    prompt=neg_prompt,
                    device=device,
                    tokenizer_max_length=components.tokenizer_max_length,
                    system_message=components.system_message,
                    crop_start=components.prompt_template_encode_start_idx,
                )

            if neg_prompt_embeds_2 is None:
                neg_prompt_embeds_2, neg_prompt_embeds_mask_2 = HunyuanVideo15Pipeline._get_byt5_prompt_embeds(
                    tokenizer=components.tokenizer_2,
                    text_encoder=components.text_encoder_2,
                    prompt=neg_prompt,
                    device=device,
                    tokenizer_max_length=components.tokenizer_2_max_length,
                )

            _, seq_len, _ = neg_prompt_embeds.shape
            neg_prompt_embeds = neg_prompt_embeds.repeat(1, num_videos_per_prompt, 1).view(batch_size * num_videos_per_prompt, seq_len, -1)
            neg_prompt_embeds_mask = neg_prompt_embeds_mask.repeat(1, num_videos_per_prompt, 1).view(batch_size * num_videos_per_prompt, seq_len)

            _, seq_len_2, _ = neg_prompt_embeds_2.shape
            neg_prompt_embeds_2 = neg_prompt_embeds_2.repeat(1, num_videos_per_prompt, 1).view(batch_size * num_videos_per_prompt, seq_len_2, -1)
            neg_prompt_embeds_mask_2 = neg_prompt_embeds_mask_2.repeat(1, num_videos_per_prompt, 1).view(batch_size * num_videos_per_prompt, seq_len_2)

            block_state.negative_prompt_embeds = neg_prompt_embeds.to(dtype=dtype, device=device)
            block_state.negative_prompt_embeds_mask = neg_prompt_embeds_mask.to(dtype=dtype, device=device)
            block_state.negative_prompt_embeds_2 = neg_prompt_embeds_2.to(dtype=dtype, device=device)
            block_state.negative_prompt_embeds_mask_2 = neg_prompt_embeds_mask_2.to(dtype=dtype, device=device)

        # Pass batch_size downstream
        state.set("batch_size", batch_size)

        self.set_block_state(state, block_state)
        return components, state
