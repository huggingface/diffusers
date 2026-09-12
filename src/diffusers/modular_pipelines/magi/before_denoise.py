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

import math

import torch

from ...models import AutoencoderKLMagi, MagiTransformer3DModel
from ...models.magi_conditioning import MagiTextConditioningModel
from ...utils.torch_utils import randn_tensor
from ..modular_pipeline import ModularPipelineBlocks
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam


class MagiPrepareLatentsStep(ModularPipelineBlocks):
    model_name = "magi"

    @property
    def description(self):
        return "Prepare FP32 text-to-video noise and expand text features for the requested video batch."

    @property
    def expected_components(self):
        return [
            ComponentSpec("transformer", MagiTransformer3DModel),
            ComponentSpec("vae", AutoencoderKLMagi),
            ComponentSpec("text_conditioning", MagiTextConditioningModel),
        ]

    @property
    def inputs(self):
        return [
            InputParam(
                "text_embeds", required=True, type_hint=torch.Tensor, description="Per-prompt T5 hidden states."
            ),
            InputParam(
                "text_attention_mask", required=True, type_hint=torch.Tensor, description="Per-prompt T5 keep-mask."
            ),
            InputParam("height", default=720, type_hint=int, description="Video height in pixels."),
            InputParam("width", default=720, type_hint=int, description="Video width in pixels."),
            InputParam(
                "num_frames",
                default=96,
                type_hint=int,
                description="Requested video frames; generation rounds up to full chunks.",
            ),
            InputParam("chunk_width", default=6, type_hint=int, description="Latent frames per chunk."),
            InputParam.template("num_images_per_prompt"),
            InputParam.template("generator"),
            InputParam(
                "latents",
                default=None,
                type_hint=torch.Tensor,
                description="Optional initial FP32 noise for all generated chunks.",
            ),
            InputParam(
                "prefix_latents",
                default=None,
                type_hint=torch.Tensor,
                description="Not supported by this text-to-video preparation block.",
            ),
        ]

    @property
    def intermediate_outputs(self):
        return [
            OutputParam(name, type_hint=torch.Tensor, description=description)
            for name, description in [
                ("latents", "Initial FP32 noise."),
                ("prompt_embeds", "HQ/duration conditioned chunk text features."),
                ("prompt_attention_mask", "Conditional text keep-mask."),
                ("negative_prompt_embeds", "Learned null text features."),
                ("negative_prompt_attention_mask", "Null text keep-mask."),
            ]
        ]

    @torch.no_grad()
    def __call__(self, components, state):
        block_state = self.get_block_state(state)
        if block_state.prefix_latents is not None:
            raise ValueError(
                "This workflow is text-to-video; use MagiDenoiseStep for prepared full-chunk prefix latents."
            )
        for name in ("height", "width", "num_frames", "chunk_width", "num_images_per_prompt"):
            value = getattr(block_state, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        spatial, temporal = components.vae.spatial_compression_ratio, components.vae.temporal_compression_ratio
        if block_state.height % spatial or block_state.width % spatial or block_state.num_frames % temporal:
            raise ValueError("Video dimensions must be divisible by the VAE compression ratios.")
        if components.vae.config.latent_channels != components.transformer.config.in_channels:
            raise ValueError("VAE latent channels must match the Transformer input channels.")
        num_chunks = math.ceil(block_state.num_frames // temporal / block_state.chunk_width)
        count = block_state.num_images_per_prompt
        device = components._execution_device
        text = block_state.text_embeds.repeat_interleave(count, dim=0).to(device=device, dtype=torch.float32)
        mask = block_state.text_attention_mask.repeat_interleave(count, dim=0).to(device)
        conditioning = components.text_conditioning(text, mask, num_chunks=num_chunks)
        block_state.prompt_embeds = conditioning.sample
        block_state.prompt_attention_mask = conditioning.attention_mask
        block_state.negative_prompt_embeds = conditioning.negative_prompt_embeds
        block_state.negative_prompt_attention_mask = conditioning.negative_prompt_attention_mask
        shape = (
            text.shape[0],
            components.transformer.config.in_channels,
            num_chunks * block_state.chunk_width,
            block_state.height // spatial,
            block_state.width // spatial,
        )
        if block_state.latents is None:
            block_state.latents = randn_tensor(
                shape, generator=block_state.generator, device=device, dtype=torch.float32
            )
        elif tuple(block_state.latents.shape) != shape:
            raise ValueError(f"Initial latents must have shape {shape}.")
        else:
            block_state.latents = block_state.latents.to(device=device, dtype=torch.float32)
        self.set_block_state(state, block_state)
        return components, state


class MagiPrepareConditionedLatentsStep(MagiPrepareLatentsStep):
    model_name = "magi"

    @property
    def description(self):
        return "Prepare prefix-aware FP32 noise and duration conditioning for newly generated chunks."

    @property
    def expected_components(self):
        return [
            ComponentSpec("transformer", MagiTransformer3DModel),
            ComponentSpec("vae", AutoencoderKLMagi),
            ComponentSpec("text_conditioning", MagiTextConditioningModel),
        ]

    @property
    def inputs(self):
        return [param for param in super().inputs if param.name not in ("prefix_latents", "num_frames")] + [
            InputParam(
                "conditioning_latents",
                required=True,
                type_hint=torch.Tensor,
                description="Per-prompt scaled VAE prefix.",
            ),
            InputParam(
                "num_frames",
                default=96,
                type_hint=int,
                description="Requested new frames; prefix plus new frames rounds up to full latent chunks.",
            ),
        ]

    @property
    def intermediate_outputs(self):
        return [
            OutputParam(name, type_hint=torch.Tensor, description=description)
            for name, description in [
                ("latents", "Initial FP32 noise."),
                ("conditioning_latents", "Scaled prefix expanded to the generated video batch."),
                ("prompt_embeds", "HQ/duration conditioned chunk text features."),
                ("prompt_attention_mask", "Conditional text keep-mask."),
                ("negative_prompt_embeds", "Learned null text features."),
                ("negative_prompt_attention_mask", "Null text keep-mask."),
            ]
        ]

    @torch.no_grad()
    def __call__(self, components, state):
        block_state = self.get_block_state(state)
        for name in ("height", "width", "num_frames", "chunk_width", "num_images_per_prompt"):
            value = getattr(block_state, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        spatial, temporal = components.vae.spatial_compression_ratio, components.vae.temporal_compression_ratio
        if block_state.height % spatial or block_state.width % spatial or block_state.num_frames % temporal:
            raise ValueError("Video dimensions must be divisible by the VAE compression ratios.")
        if components.vae.config.latent_channels != components.transformer.config.in_channels:
            raise ValueError("VAE latent channels must match the Transformer input channels.")
        prefix = block_state.conditioning_latents
        expected = (block_state.height // spatial, block_state.width // spatial)
        if (
            not isinstance(prefix, torch.Tensor)
            or prefix.ndim != 5
            or prefix.shape[0] not in (1, block_state.text_embeds.shape[0])
            or prefix.shape[1] != components.transformer.config.in_channels
            or prefix.shape[2] < 1
            or tuple(prefix.shape[3:]) != expected
            or not prefix.is_floating_point()
            or not prefix.isfinite().all()
        ):
            raise ValueError(
                "conditioning_latents must match the prompt batch, latent channels, and requested dimensions."
            )
        prefix_chunks = prefix.shape[2] // block_state.chunk_width
        num_chunks = math.ceil((block_state.num_frames // temporal + prefix.shape[2]) / block_state.chunk_width)
        count = block_state.num_images_per_prompt
        device = components._execution_device
        text = block_state.text_embeds.repeat_interleave(count, dim=0).to(device=device, dtype=torch.float32)
        mask = block_state.text_attention_mask.repeat_interleave(count, dim=0).to(device)
        conditioning = components.text_conditioning(text, mask, num_chunks=num_chunks - prefix_chunks)
        block_state.conditioning_latents = (
            prefix.expand(block_state.text_embeds.shape[0], -1, -1, -1, -1)
            .repeat_interleave(count, dim=0)
            .to(device=device, dtype=torch.float32)
        )
        block_state.prompt_embeds = conditioning.sample
        block_state.prompt_attention_mask = conditioning.attention_mask
        block_state.negative_prompt_embeds = conditioning.negative_prompt_embeds
        block_state.negative_prompt_attention_mask = conditioning.negative_prompt_attention_mask
        if prefix_chunks:
            # Clean-prefix conditional slots are never evaluated; use valid null captions for the shared validator.
            block_state.prompt_embeds = torch.cat(
                [
                    conditioning.negative_prompt_embeds[:, None].expand(-1, prefix_chunks, -1, -1),
                    block_state.prompt_embeds,
                ],
                dim=1,
            )
            block_state.prompt_attention_mask = torch.cat(
                [
                    conditioning.negative_prompt_attention_mask[:, None].expand(-1, prefix_chunks, -1),
                    block_state.prompt_attention_mask,
                ],
                dim=1,
            )
        shape = (
            text.shape[0],
            components.transformer.config.in_channels,
            num_chunks * block_state.chunk_width,
            block_state.height // spatial,
            block_state.width // spatial,
        )
        if block_state.latents is None:
            block_state.latents = randn_tensor(
                shape, generator=block_state.generator, device=device, dtype=torch.float32
            )
        elif tuple(block_state.latents.shape) != shape:
            raise ValueError(f"Initial latents must have shape {shape}.")
        else:
            block_state.latents = block_state.latents.to(device=device, dtype=torch.float32)
        self.set_block_state(state, block_state)
        return components, state
