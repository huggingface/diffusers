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

import re

import torch
from transformers import (
    ByT5Tokenizer,
    Qwen2_5_VLTextModel,
    Qwen2TokenizerFast,
    SiglipImageProcessor,
    SiglipVisionModel,
    T5EncoderModel,
)

from ...configuration_utils import FrozenDict
from ...guiders import ClassifierFreeGuidance
from ...models import AutoencoderKLHunyuanVideo15
from ...pipelines.hunyuan_video1_5.image_processor import HunyuanVideo15ImageProcessor
from ...utils import logging
from ..modular_pipeline import ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam
from .modular_pipeline import HunyuanVideo15ModularPipeline


logger = logging.get_logger(__name__)


def format_text_input(prompt, system_message):
    return [
        [{"role": "system", "content": system_message}, {"role": "user", "content": p if p else " "}] for p in prompt
    ]


def extract_glyph_texts(prompt):
    pattern = r"\"(.*?)\"|\"(.*?)\""
    matches = re.findall(pattern, prompt)
    result = [match[0] or match[1] for match in matches]
    result = list(dict.fromkeys(result)) if len(result) > 1 else result
    if result:
        formatted_result = ". ".join([f'Text "{text}"' for text in result]) + ". "
    else:
        formatted_result = None
    return formatted_result


def _get_mllm_prompt_embeds(
    text_encoder,
    tokenizer,
    prompt,
    device,
    tokenizer_max_length=1000,
    num_hidden_layers_to_skip=2,
    # fmt: off
    system_message="You are a helpful assistant. Describe the video by detailing the following aspects: \
    1. The main content and theme of the video. \
    2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects. \
    3. Actions, events, behaviors temporal relationships, physical movement changes of the objects. \
    4. background environment, light, style and atmosphere. \
    5. camera angles, movements, and transitions used in the video.",
    # fmt: on
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
            )[0]
            glyph_text_embeds = glyph_text_embeds.to(device=device)
            glyph_text_embeds_mask = txt_tokens.attention_mask.to(device=device)

        prompt_embeds_list.append(glyph_text_embeds)
        prompt_embeds_mask_list.append(glyph_text_embeds_mask)

    return torch.cat(prompt_embeds_list, dim=0), torch.cat(prompt_embeds_mask_list, dim=0)


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
            InputParam.template("prompt", required=False),
            InputParam.template("negative_prompt"),
            InputParam.template("num_images_per_prompt", name="num_videos_per_prompt"),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam.template("prompt_embeds"),
            OutputParam.template("prompt_embeds_mask"),
            OutputParam.template("negative_prompt_embeds"),
            OutputParam.template("negative_prompt_embeds_mask"),
            OutputParam(
                "prompt_embeds_2",
                type_hint=torch.Tensor,
                kwargs_type="denoiser_input_fields",
                description="ByT5 glyph-text embeddings used as a second conditioning stream for the transformer.",
            ),
            OutputParam(
                "prompt_embeds_mask_2",
                type_hint=torch.Tensor,
                kwargs_type="denoiser_input_fields",
                description="Attention mask for the ByT5 glyph-text embeddings.",
            ),
            OutputParam(
                "negative_prompt_embeds_2",
                type_hint=torch.Tensor,
                kwargs_type="denoiser_input_fields",
                description="ByT5 glyph-text negative embeddings for classifier-free guidance.",
            ),
            OutputParam(
                "negative_prompt_embeds_mask_2",
                type_hint=torch.Tensor,
                kwargs_type="denoiser_input_fields",
                description="Attention mask for the ByT5 glyph-text negative embeddings.",
            ),
        ]

    @staticmethod
    def encode_prompt(
        components,
        prompt,
        device=None,
        dtype=None,
        batch_size=1,
        num_videos_per_prompt=1,
    ):
        device = device or components._execution_device
        dtype = dtype or components.text_encoder.dtype

        if prompt is None:
            prompt = [""] * batch_size
        prompt = [prompt] if isinstance(prompt, str) else prompt

        prompt_embeds, prompt_embeds_mask = _get_mllm_prompt_embeds(
            tokenizer=components.tokenizer,
            text_encoder=components.text_encoder,
            prompt=prompt,
            device=device,
            tokenizer_max_length=components.tokenizer_max_length,
            system_message=components.system_message,
            crop_start=components.prompt_template_encode_start_idx,
        )

        prompt_embeds_2, prompt_embeds_mask_2 = _get_byt5_prompt_embeds(
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
        negative_prompt = block_state.negative_prompt
        num_videos_per_prompt = block_state.num_videos_per_prompt

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = 1

        (
            block_state.prompt_embeds,
            block_state.prompt_embeds_mask,
            block_state.prompt_embeds_2,
            block_state.prompt_embeds_mask_2,
        ) = self.encode_prompt(
            components,
            prompt=prompt,
            device=device,
            dtype=dtype,
            batch_size=batch_size,
            num_videos_per_prompt=num_videos_per_prompt,
        )

        if components.requires_unconditional_embeds:
            (
                block_state.negative_prompt_embeds,
                block_state.negative_prompt_embeds_mask,
                block_state.negative_prompt_embeds_2,
                block_state.negative_prompt_embeds_mask_2,
            ) = self.encode_prompt(
                components,
                prompt=negative_prompt,
                device=device,
                dtype=dtype,
                batch_size=batch_size,
                num_videos_per_prompt=num_videos_per_prompt,
            )

        state.set("batch_size", batch_size)

        self.set_block_state(state, block_state)
        return components, state


def retrieve_latents(
    encoder_output: torch.Tensor, generator: torch.Generator | None = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


class HunyuanVideo15VaeEncoderStep(ModularPipelineBlocks):
    model_name = "hunyuan-video-1.5"

    @property
    def description(self) -> str:
        return "VAE Encoder step that encodes an input image into latent space for image-to-video generation"

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("vae", AutoencoderKLHunyuanVideo15),
            ComponentSpec(
                "video_processor",
                HunyuanVideo15ImageProcessor,
                config=FrozenDict({"vae_scale_factor": 16}),
                default_creation_method="from_config",
            ),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("image", required=True),
            InputParam.template("height"),
            InputParam.template("width"),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "image_latents",
                type_hint=torch.Tensor,
                description="Encoded image latents from the VAE encoder",
            ),
            OutputParam("height", type_hint=int, description="Target height resolved from image"),
            OutputParam("width", type_hint=int, description="Target width resolved from image"),
        ]

    @torch.no_grad()
    def __call__(self, components: HunyuanVideo15ModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        device = components._execution_device

        image = block_state.image
        height = block_state.height
        width = block_state.width
        if height is None or width is None:
            height, width = components.video_processor.calculate_default_height_width(
                height=image.size[1], width=image.size[0], target_size=components.target_size
            )
        image = components.video_processor.resize(image, height=height, width=width, resize_mode="crop")

        vae_dtype = components.vae.dtype
        image_tensor = components.video_processor.preprocess(image, height=height, width=width).to(
            device=device, dtype=vae_dtype
        )
        image_tensor = image_tensor.unsqueeze(2)
        image_latents = retrieve_latents(components.vae.encode(image_tensor), sample_mode="argmax")
        image_latents = image_latents * components.vae.config.scaling_factor

        block_state.image_latents = image_latents
        block_state.height = height
        block_state.width = width
        state.set("image", image)

        self.set_block_state(state, block_state)
        return components, state


class HunyuanVideo15ImageEncoderStep(ModularPipelineBlocks):
    model_name = "hunyuan-video-1.5"

    @property
    def description(self) -> str:
        return "Siglip image encoder step that produces image_embeds for image-to-video generation"

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("image_encoder", SiglipVisionModel),
            ComponentSpec("feature_extractor", SiglipImageProcessor),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("image", required=True),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "image_embeds",
                type_hint=torch.Tensor,
                description="Image embeddings from the Siglip vision encoder",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components: HunyuanVideo15ModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        device = components._execution_device

        image_encoder_dtype = next(components.image_encoder.parameters()).dtype
        image_inputs = components.feature_extractor.preprocess(
            images=block_state.image, do_resize=True, return_tensors="pt", do_convert_rgb=True
        )
        image_inputs = image_inputs.to(device=device, dtype=image_encoder_dtype)
        image_embeds = components.image_encoder(**image_inputs).last_hidden_state

        block_state.image_embeds = image_embeds

        self.set_block_state(state, block_state)
        return components, state
