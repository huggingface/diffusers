# Copyright 2026 The Echo-WM and HuggingFace Teams. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import math

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from ..ltx2.encoders import LTX2TextConnectors, LTX2VaeEncoderStep, _get_gemma_prompt_embeds
from ..modular_pipeline import ModularPipelineBlocks, PipelineState, SequentialPipelineBlocks
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam
from .utils import apply_image_conditioning_crf, resolve_default_image_crf


def _preprocess_echo_wm_images(images, height: int, width: int, crf: int, device: torch.device) -> torch.Tensor:
    """Match the reference's torch bilinear resize and center crop before VAE encoding."""
    images = images if isinstance(images, list) else [images]
    processed = []
    for image in images:
        if not isinstance(image, PIL.Image.Image):
            raise ValueError(
                f"Echo-WM image preprocessing requires PIL images, got {type(image)}. "
                "Pass a PIL image or a preprocessed tensor."
            )
        array = np.array(image.convert("RGB"))
        if crf != 0:
            array = apply_image_conditioning_crf(array, crf)
        tensor = torch.tensor(array, dtype=torch.float32, device=device).permute(2, 0, 1).unsqueeze(0)
        source_height, source_width = tensor.shape[-2:]
        scale = max(height / source_height, width / source_width)
        resized_height = math.ceil(source_height * scale)
        resized_width = math.ceil(source_width * scale)
        tensor = F.interpolate(tensor, size=(resized_height, resized_width), mode="bilinear", align_corners=False)
        crop_top = (resized_height - height) // 2
        crop_left = (resized_width - width) // 2
        tensor = tensor[:, :, crop_top : crop_top + height, crop_left : crop_left + width]
        processed.append(tensor / 127.5 - 1.0)
    return torch.cat(processed)


class EchoWMVaeEncoderStep(LTX2VaeEncoderStep):
    """Encode images using the reference's tensor resize instead of PIL resizing."""

    model_name = "echo-wm"

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        if not isinstance(block_state.image, torch.Tensor):
            crf = (
                block_state.image_crf
                if block_state.image_crf is not None
                else resolve_default_image_crf(components.text_encoder)
            )
            image = _preprocess_echo_wm_images(
                block_state.image,
                block_state.height,
                block_state.width,
                crf,
                components._execution_device,
            )
            state.set("image", image)
        return super().__call__(components, state)


class EchoWMFlashTextEncoderStep(ModularPipelineBlocks):
    """Encode the positive prompt used by the guidance-distilled Echo-WM Flash checkpoint."""

    model_name = "echo-wm-flash"

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("text_encoder", PreTrainedModel),
            ComponentSpec("tokenizer", PreTrainedTokenizerBase),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("prompt", required=True),
            InputParam.template("max_sequence_length", default=1024),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "prompt_embeds",
                type_hint=torch.Tensor,
                description="Packed per-layer Gemma hidden states for the prompt.",
            ),
            OutputParam(
                "prompt_attention_mask",
                type_hint=torch.Tensor,
                description="Binary attention mask for `prompt_embeds`.",
            ),
            OutputParam(
                "batch_size",
                type_hint=int,
                description="The number of prompts being denoised before per-prompt expansion.",
            ),
            OutputParam("dtype", type_hint=torch.dtype, description="The dtype of the prompt embeddings."),
        ]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        if not isinstance(block_state.prompt, (str, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(block_state.prompt)}")

        prompt = [block_state.prompt] if isinstance(block_state.prompt, str) else block_state.prompt
        block_state.prompt_embeds, block_state.prompt_attention_mask = _get_gemma_prompt_embeds(
            components,
            prompt,
            block_state.max_sequence_length,
            components._execution_device,
            components.text_encoder.dtype,
        )
        block_state.batch_size = block_state.prompt_embeds.shape[0]
        block_state.dtype = block_state.prompt_embeds.dtype
        self.set_block_state(state, block_state)
        return components, state


class EchoWMFlashTextConnectorStep(ModularPipelineBlocks):
    """Project positive prompt features into Echo-WM Flash's video and audio conditioning spaces."""

    model_name = "echo-wm-flash"

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("connectors", LTX2TextConnectors),
            ComponentSpec("tokenizer", PreTrainedTokenizerBase),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("prompt_embeds", type_hint=torch.Tensor, required=True),
            InputParam("prompt_attention_mask", type_hint=torch.Tensor, required=True),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "connector_prompt_embeds", type_hint=torch.Tensor, description="Video-branch text conditioning."
            ),
            OutputParam(
                "connector_audio_prompt_embeds",
                type_hint=torch.Tensor,
                description="Audio-branch text conditioning.",
            ),
            OutputParam("connector_attention_mask", type_hint=torch.Tensor, description="Binary text attention mask."),
        ]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        (
            block_state.connector_prompt_embeds,
            block_state.connector_audio_prompt_embeds,
            block_state.connector_attention_mask,
        ) = components.connectors(
            block_state.prompt_embeds,
            block_state.prompt_attention_mask,
            padding_side=components.tokenizer.padding_side,
        )
        self.set_block_state(state, block_state)
        return components, state


# auto_docstring
class EchoWMFlashTextConditioningStep(SequentialPipelineBlocks):
    """
    Components:
          text_encoder (`PreTrainedModel`)
          tokenizer (`PreTrainedTokenizerBase`)
          connectors (`LTX2TextConnectors`)

      Inputs:
          prompt (`str`):
              The prompt or prompts to guide image generation.
          max_sequence_length (`int`, *optional*, defaults to 1024):
              Maximum sequence length for prompt encoding.

      Outputs:
          prompt_embeds (`Tensor`):
              Packed per-layer Gemma hidden states for the prompt.
          prompt_attention_mask (`Tensor`):
              Binary attention mask for `prompt_embeds`.
          batch_size (`int`):
              The number of prompts being denoised before per-prompt expansion.
          dtype (`dtype`):
              The dtype of the prompt embeddings.
          connector_prompt_embeds (`Tensor`):
              Video-branch text conditioning.
          connector_audio_prompt_embeds (`Tensor`):
              Audio-branch text conditioning.
          connector_attention_mask (`Tensor`):
              Binary text attention mask.
    """

    model_name = "echo-wm-flash"
    block_classes = [EchoWMFlashTextEncoderStep, EchoWMFlashTextConnectorStep]
    block_names = ["text_encoder", "connectors"]
