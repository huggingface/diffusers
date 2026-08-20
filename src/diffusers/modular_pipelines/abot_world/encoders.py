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

import numpy as np
import PIL.Image
import torch
from PIL import ImageOps
from transformers import AutoTokenizer, UMT5EncoderModel

from ...models import AutoencoderKLWan
from ...utils import logging
from ..modular_pipeline import ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def encode_image_to_latent(vae: AutoencoderKLWan, image: PIL.Image.Image, device, dtype) -> torch.Tensor:
    """Preprocess a PIL image to `[-1, 1]` and VAE-encode it to a normalized latent `[1, C, 1, h, w]`."""
    pixels = torch.from_numpy(np.array(image, dtype=np.float32))
    pixels = pixels.to(device=device, dtype=dtype) * (2 / 255) - 1
    pixels = pixels.permute(2, 0, 1)[None, :, None]  # [1, C, 1, H, W]

    latent = vae.encode(pixels).latent_dist.mode().float()
    latents_mean = torch.tensor(vae.config.latents_mean, device=device).view(1, -1, 1, 1, 1)
    latents_std = torch.tensor(vae.config.latents_std, device=device).view(1, -1, 1, 1, 1)
    return (latent - latents_mean) / latents_std


class ABotWorldTextEncoderStep(ModularPipelineBlocks):
    model_name = "abot-world"

    @property
    def description(self) -> str:
        return "Text encoder step that encodes the prompt with umt5-xxl; masked positions are zeroed."

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("text_encoder", UMT5EncoderModel),
            ComponentSpec("tokenizer", AutoTokenizer),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("prompt", required=True, type_hint=str, description="The text prompt describing the world"),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam.template("prompt_embeds"),
        ]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        device = components._execution_device

        text_inputs = components.tokenizer(
            [block_state.prompt],
            padding="max_length",
            max_length=512,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_ids = text_inputs.input_ids.to(device)
        mask = text_inputs.attention_mask.to(device)
        seq_lens = mask.gt(0).sum(dim=1).long()

        prompt_embeds = components.text_encoder(input_ids, mask).last_hidden_state
        prompt_embeds = prompt_embeds.to(components.text_encoder.dtype)
        prompt_embeds = torch.stack(
            [torch.cat([u[:v], u.new_zeros(u.size(0) - v, u.size(1))]) for u, v in zip(prompt_embeds, seq_lens)]
        )
        block_state.prompt_embeds = prompt_embeds

        self.set_block_state(state, block_state)
        return components, state


class ABotWorldImageEncoderStep(ModularPipelineBlocks):
    model_name = "abot-world"

    @property
    def description(self) -> str:
        return (
            "Image encoder step that fits the input image to the target resolution (cover + center-crop) and "
            "VAE-encodes it into `first_frame_latents`. The rollout loop pins this clean latent as frame 0 of the "
            "first block, which is how the generated world starts from the input image."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("vae", AutoencoderKLWan),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("image", required=True, type_hint=PIL.Image.Image, description="The starting frame"),
            InputParam("height", type_hint=int, default=704, description="Height of the generated video in pixels"),
            InputParam("width", type_hint=int, default=1280, description="Width of the generated video in pixels"),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "first_frame_latents",
                type_hint=torch.Tensor,
                description="Normalized VAE latent of the starting frame `[B, C, 1, h, w]`",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        device = components._execution_device

        image = ImageOps.fit(
            block_state.image.convert("RGB"),
            (block_state.width, block_state.height),
            method=PIL.Image.LANCZOS,
            centering=(0.5, 0.5),
        )
        block_state.first_frame_latents = encode_image_to_latent(components.vae, image, device, components.vae.dtype)

        self.set_block_state(state, block_state)
        return components, state


class ABotWorldRefImagesEncoderStep(ModularPipelineBlocks):
    model_name = "abot-world"

    @property
    def description(self) -> str:
        return (
            "Reference encoder step that VAE-encodes the character reference views (e.g. head/left/right/front/back "
            "at 512x512) into `reference_latents`. The transformer pins these tokens at the head of its K/V cache, "
            "so every generated frame attends to them — this is what keeps the character consistent over an "
            "unbounded rollout."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("vae", AutoencoderKLWan),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                "reference_images",
                required=True,
                type_hint=list[PIL.Image.Image],
                description="The character reference views; each is resized to `reference_resolution`",
            ),
            InputParam(
                "reference_resolution",
                type_hint=int,
                default=512,
                description="Side length the reference views are resized to before encoding",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "reference_latents",
                type_hint=torch.Tensor,
                description="Normalized VAE latents of the reference views `[B, K, C, 1, h, w]`",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        device = components._execution_device

        size = (block_state.reference_resolution, block_state.reference_resolution)
        latents = [
            encode_image_to_latent(components.vae, img.convert("RGB").resize(size), device, components.vae.dtype)
            for img in block_state.reference_images
        ]
        block_state.reference_latents = torch.stack(latents, dim=1)  # [B, K, C, 1, h, w]

        self.set_block_state(state, block_state)
        return components, state
