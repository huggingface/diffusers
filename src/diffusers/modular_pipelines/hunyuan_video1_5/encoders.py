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
from ...pipelines.hunyuan_video1_5.pipeline_hunyuan_video1_5 import (
    HunyuanVideo15Pipeline,
    format_text_input,
    extract_glyph_texts,
)
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

        # Encode positive prompt (reuse pipeline's encode_prompt verbatim)
        (
            block_state.prompt_embeds,
            block_state.prompt_embeds_mask,
            block_state.prompt_embeds_2,
            block_state.prompt_embeds_mask_2,
        ) = HunyuanVideo15Pipeline.encode_prompt(
            components,
            prompt=prompt,
            device=device,
            dtype=dtype,
            batch_size=batch_size,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=getattr(block_state, "prompt_embeds", None),
            prompt_embeds_mask=getattr(block_state, "prompt_embeds_mask", None),
            prompt_embeds_2=getattr(block_state, "prompt_embeds_2", None),
            prompt_embeds_mask_2=getattr(block_state, "prompt_embeds_mask_2", None),
        )

        # Encode negative prompt if guider needs it
        if components.requires_unconditional_embeds:
            (
                block_state.negative_prompt_embeds,
                block_state.negative_prompt_embeds_mask,
                block_state.negative_prompt_embeds_2,
                block_state.negative_prompt_embeds_mask_2,
            ) = HunyuanVideo15Pipeline.encode_prompt(
                components,
                prompt=negative_prompt,
                device=device,
                dtype=dtype,
                batch_size=batch_size,
                num_videos_per_prompt=num_videos_per_prompt,
                prompt_embeds=getattr(block_state, "negative_prompt_embeds", None),
                prompt_embeds_mask=getattr(block_state, "negative_prompt_embeds_mask", None),
                prompt_embeds_2=getattr(block_state, "negative_prompt_embeds_2", None),
                prompt_embeds_mask_2=getattr(block_state, "negative_prompt_embeds_mask_2", None),
            )

        # Pass batch_size downstream
        state.set("batch_size", batch_size)

        self.set_block_state(state, block_state)
        return components, state
