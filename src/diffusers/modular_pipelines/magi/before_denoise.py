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
from numbers import Real

import torch

from ...models import AutoencoderKLMagi, MagiTextConditioningModel, MagiTransformer3DModel
from ...utils.torch_utils import randn_tensor
from ..modular_pipeline import ModularPipelineBlocks
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam


def _validate_component_compatibility(
    transformer, text_conditioning, height, width, chunk_width, spatial_compression_ratio
):
    patch_t, patch_h, patch_w = transformer.config.patch_size
    if (
        chunk_width % patch_t
        or (height // spatial_compression_ratio) % patch_h
        or (width // spatial_compression_ratio) % patch_w
    ):
        raise ValueError("Latent dimensions and chunk width must be divisible by the Transformer patch size.")
    if (
        text_conditioning.config.caption_channels != transformer.config.caption_channels
        or text_conditioning.config.caption_max_length != transformer.config.caption_max_length
    ):
        raise ValueError("Text conditioning dimensions must match the Transformer caption dimensions.")


def prepare_magi_latents(
    text_conditioning,
    text_embeds,
    text_attention_mask,
    num_images_per_prompt,
    num_chunks,
    shape,
    latents,
    generator,
    device,
):
    text = text_embeds.repeat_interleave(num_images_per_prompt, dim=0).to(device=device, dtype=torch.float32)
    mask = text_attention_mask.repeat_interleave(num_images_per_prompt, dim=0).to(device)
    conditioning = text_conditioning(text, mask, num_chunks=num_chunks)
    if latents is None:
        latents = randn_tensor(shape, generator=generator, device=device, dtype=torch.float32)
    else:
        if not isinstance(latents, torch.Tensor) or tuple(latents.shape) != shape:
            raise ValueError(f"Initial latents must have shape {shape}.")
        if not latents.is_floating_point() or not latents.isfinite().all():
            raise ValueError("Initial latents must be a finite floating-point tensor.")
        latents = latents.to(device=device, dtype=torch.float32)
    return latents, conditioning


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
        for name in ("height", "width", "num_frames", "chunk_width", "num_images_per_prompt"):
            value = getattr(block_state, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        spatial, temporal = components.vae.spatial_compression_ratio, components.vae.temporal_compression_ratio
        if block_state.height % spatial or block_state.width % spatial or block_state.num_frames % temporal:
            raise ValueError("Video dimensions must be divisible by the VAE compression ratios.")
        if components.vae.config.latent_channels != components.transformer.config.in_channels:
            raise ValueError("VAE latent channels must match the Transformer input channels.")
        _validate_component_compatibility(
            components.transformer,
            components.text_conditioning,
            block_state.height,
            block_state.width,
            block_state.chunk_width,
            spatial,
        )
        num_chunks = math.ceil(block_state.num_frames // temporal / block_state.chunk_width)
        count = block_state.num_images_per_prompt
        device = components._execution_device
        shape = (
            block_state.text_embeds.shape[0] * count,
            components.transformer.config.in_channels,
            num_chunks * block_state.chunk_width,
            block_state.height // spatial,
            block_state.width // spatial,
        )
        block_state.latents, conditioning = prepare_magi_latents(
            components.text_conditioning,
            block_state.text_embeds,
            block_state.text_attention_mask,
            count,
            num_chunks,
            shape,
            block_state.latents,
            block_state.generator,
            device,
        )
        block_state.prompt_embeds = conditioning.sample
        block_state.prompt_attention_mask = conditioning.attention_mask
        block_state.negative_prompt_embeds = conditioning.negative_prompt_embeds
        block_state.negative_prompt_attention_mask = conditioning.negative_prompt_attention_mask
        self.set_block_state(state, block_state)
        return components, state


class MagiPrepareConditionedLatentsStep(MagiPrepareLatentsStep):
    model_name = "magi"

    @property
    def description(self):
        return "Prepare prefix-aware FP32 noise and duration conditioning for newly generated chunks."

    @property
    def inputs(self):
        return [param for param in super().inputs if param.name != "num_frames"] + [
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
        return super().intermediate_outputs + [
            OutputParam(
                "conditioning_latents",
                type_hint=torch.Tensor,
                description="Scaled prefix expanded to the generated video batch.",
            )
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
        _validate_component_compatibility(
            components.transformer,
            components.text_conditioning,
            block_state.height,
            block_state.width,
            block_state.chunk_width,
            spatial,
        )
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
        shape = (
            block_state.text_embeds.shape[0] * count,
            components.transformer.config.in_channels,
            num_chunks * block_state.chunk_width,
            block_state.height // spatial,
            block_state.width // spatial,
        )
        block_state.latents, conditioning = prepare_magi_latents(
            components.text_conditioning,
            block_state.text_embeds,
            block_state.text_attention_mask,
            count,
            num_chunks - prefix_chunks,
            shape,
            block_state.latents,
            block_state.generator,
            device,
        )
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
        self.set_block_state(state, block_state)
        return components, state


class MagiPrepareDenoiseStep(ModularPipelineBlocks):
    model_name = "magi"

    @property
    def description(self):
        return "Prepare the MAGI base-model chunk schedule and optional full-chunk clean prefix."

    @property
    def expected_components(self):
        return [ComponentSpec("transformer", MagiTransformer3DModel)]

    @property
    def inputs(self):
        return [
            InputParam(
                "latents",
                type_hint=torch.Tensor,
                description="FP32 latent state shaped (batch, channels, frames, height, width), including prefix slots.",
                required=True,
            ),
            InputParam(
                "prompt_embeds",
                type_hint=torch.Tensor,
                description="Prepared conditional text features, shared across chunks or provided per chunk.",
                required=True,
            ),
            InputParam(
                "prompt_attention_mask",
                type_hint=torch.Tensor,
                description="Boolean keep-mask matching the conditional text features.",
                required=True,
            ),
            InputParam(
                "negative_prompt_embeds",
                type_hint=torch.Tensor,
                description="Learned null-caption features shaped (batch, length, caption_channels), shared across chunks.",
                required=True,
            ),
            InputParam(
                "negative_prompt_attention_mask",
                type_hint=torch.Tensor,
                description="Boolean keep-mask for the learned null-caption features.",
                required=True,
            ),
            InputParam(
                "prefix_latents",
                type_hint=torch.Tensor,
                description="Optional full-chunk clean prefix; replaces the leading latent slots and remains unchanged.",
                default=None,
            ),
            InputParam(
                "num_inference_steps",
                type_hint=int,
                description="Number of Euler updates per generated chunk.",
                default=64,
            ),
            InputParam("chunk_width", type_hint=int, description="Number of latent frames in each chunk.", default=6),
            InputParam(
                "window_size",
                type_hint=int,
                description="Maximum number of simultaneously denoised chunks.",
                default=4,
            ),
            InputParam(
                "noise2clean_kvrange",
                type_hint=tuple,
                description="Positive attention-window lengths in chunks, from early to late denoising stages.",
                default=(5, 4, 3, 2),
            ),
            InputParam(
                "clean_chunk_kvrange",
                type_hint=int,
                description="Attention-window length used when recomputing clean chunks; `-1` uses the final noise-to-clean range.",
                default=1,
            ),
            InputParam(
                "clean_t",
                type_hint=float,
                description="Model evaluation time for clean-prefix cache extraction.",
                default=0.9999,
            ),
            InputParam(
                "attention_kwargs",
                type_hint=dict,
                description="Optional keyword arguments passed to Transformer attention.",
                default=None,
            ),
            InputParam(
                "cache_device",
                type_hint=str,
                description="Optional device for clean-prefix KV storage; use cpu to offload between layer evaluations.",
                default=None,
            ),
        ]

    @property
    def intermediate_outputs(self):
        return [
            OutputParam(
                "num_chunks",
                type_hint=int,
                description="Total number of latent chunks, including the supplied prefix.",
            ),
            OutputParam("prefix_chunks", type_hint=int, description="Number of supplied full-chunk prefix chunks."),
            OutputParam("chunk_tokens", type_hint=int, description="Number of Transformer tokens per latent chunk."),
            OutputParam(
                "steps_per_stage",
                type_hint=int,
                description="Number of iterations before the active chunk window moves forward.",
            ),
            OutputParam(
                "num_window_steps", type_hint=int, description="Total number of asynchronous window iterations."
            ),
            OutputParam(
                "clean_kv_cache",
                type_hint=tuple,
                description="Per-layer clean-prefix key/value tensors, or None before any prefix is cached; excludes the final generated chunk.",
            ),
            OutputParam(
                "completed_chunks",
                type_hint=list,
                description="Indices of supplied prefix chunks and finalized generated chunks.",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components, state):
        block_state = self.get_block_state(state)
        config = components.transformer.config
        if config.distilled:
            raise ValueError("MagiDenoiseStep supports base models only, not distilled models.")
        for name in ("chunk_width", "window_size", "num_inference_steps"):
            value = getattr(block_state, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if (
            not isinstance(block_state.noise2clean_kvrange, (tuple, list))
            or not block_state.noise2clean_kvrange
            or any(
                not isinstance(value, int) or isinstance(value, bool) or value <= 0
                for value in block_state.noise2clean_kvrange
            )
        ):
            raise ValueError("noise2clean_kvrange must be a nonempty sequence of positive integers.")
        if (
            not isinstance(block_state.clean_chunk_kvrange, int)
            or isinstance(block_state.clean_chunk_kvrange, bool)
            or (block_state.clean_chunk_kvrange != -1 and block_state.clean_chunk_kvrange <= 0)
        ):
            raise ValueError("clean_chunk_kvrange must be a positive integer or -1.")
        if block_state.clean_chunk_kvrange == -1:
            block_state.clean_chunk_kvrange = block_state.noise2clean_kvrange[-1]
        if block_state.num_inference_steps % block_state.window_size or block_state.num_inference_steps % len(
            block_state.noise2clean_kvrange
        ):
            raise ValueError("num_inference_steps must be divisible by window_size and the number of KV ranges.")
        if (
            not isinstance(block_state.clean_t, Real)
            or isinstance(block_state.clean_t, bool)
            or not math.isfinite(block_state.clean_t)
            or not 0 <= block_state.clean_t <= 1
        ):
            raise ValueError("clean_t must be a finite number in [0, 1].")
        if (
            not isinstance(block_state.latents, torch.Tensor)
            or not block_state.latents.is_floating_point()
            or block_state.latents.ndim != 5
            or min(block_state.latents.shape) <= 0
            or block_state.latents.shape[1] != config.in_channels
        ):
            raise ValueError(
                "latents must be a nonempty floating-point tensor with shape "
                "(batch, in_channels, frames, height, width)."
            )
        batch, channels, frames, height, width = block_state.latents.shape
        pt, ph, pw = config.patch_size
        if frames % block_state.chunk_width or block_state.chunk_width % pt or height % ph or width % pw:
            raise ValueError("Latents must contain full chunks and be divisible by the Transformer patch size.")
        block_state.num_chunks = frames // block_state.chunk_width
        for name, mask_name, per_chunk in [
            ("prompt_embeds", "prompt_attention_mask", True),
            ("negative_prompt_embeds", "negative_prompt_attention_mask", False),
        ]:
            features, mask = getattr(block_state, name), getattr(block_state, mask_name)
            valid_shape = (features.ndim == 3 and features.shape[0] == batch) or (
                per_chunk and features.ndim == 4 and features.shape[:2] == (batch, block_state.num_chunks)
            )
            if not valid_shape or features.shape[-1] != config.caption_channels or features.shape[-2] <= 0:
                raise ValueError(f"{name} must match the batch, caption channels, and optional chunk count.")
            if mask.shape != features.shape[:-1] or mask.dtype != torch.bool or not mask.any(dim=-1).all():
                raise ValueError(f"{mask_name} must be a boolean keep-mask with at least one valid token per caption.")
            if features.device != block_state.latents.device or mask.device != block_state.latents.device:
                raise ValueError("All latents, text features, and masks must be on the same device.")
        if block_state.prompt_embeds.shape[-2] != block_state.negative_prompt_embeds.shape[-2]:
            raise ValueError("Conditional and null text features must have the same padded length.")
        block_state.latents = block_state.latents.float().clone()
        if block_state.prompt_embeds.ndim == 3:
            block_state.prompt_embeds = block_state.prompt_embeds[:, None].expand(-1, block_state.num_chunks, -1, -1)
            block_state.prompt_attention_mask = block_state.prompt_attention_mask[:, None].expand(
                -1, block_state.num_chunks, -1
            )
        block_state.prefix_chunks = 0
        block_state.clean_kv_cache = None
        block_state.chunk_tokens = (block_state.chunk_width // pt) * (height // ph) * (width // pw)
        if block_state.prefix_latents is not None:
            prefix = block_state.prefix_latents
            if (
                not isinstance(prefix, torch.Tensor)
                or not prefix.is_floating_point()
                or prefix.ndim != 5
                or prefix.shape[:2] != (batch, channels)
                or prefix.shape[3:] != (height, width)
                or prefix.device != block_state.latents.device
            ):
                raise ValueError(
                    "prefix_latents must be a floating-point tensor matching the latent batch, channels, "
                    "spatial dimensions, and device."
                )
            if prefix.shape[2] <= 0 or prefix.shape[2] % block_state.chunk_width or prefix.shape[2] >= frames:
                raise ValueError("prefix_latents must contain full chunks and leave at least one chunk to generate.")
            block_state.prefix_chunks = prefix.shape[2] // block_state.chunk_width
            block_state.latents[:, :, : prefix.shape[2]] = prefix.float()
            block_state.clean_kv_cache = components.transformer(
                hidden_states=block_state.latents[:, :, : prefix.shape[2]],
                encoder_hidden_states=block_state.negative_prompt_embeds,
                encoder_attention_mask=block_state.negative_prompt_attention_mask,
                caption_dropout_mask=torch.ones(1, device=block_state.latents.device, dtype=torch.bool),
                timestep=block_state.latents.new_full((batch, block_state.prefix_chunks), block_state.clean_t),
                kv_ranges=tuple(
                    (
                        max(0, index + 1 - block_state.clean_chunk_kvrange) * block_state.chunk_tokens,
                        (index + 1) * block_state.chunk_tokens,
                    )
                    for index in range(block_state.prefix_chunks)
                ),
                use_cache=True,
                cache_device=block_state.cache_device,
                attention_kwargs=block_state.attention_kwargs,
            ).kv_cache
        block_state.steps_per_stage = block_state.num_inference_steps // block_state.window_size
        block_state.num_window_steps = block_state.steps_per_stage * (
            block_state.num_chunks + block_state.window_size - 1 - block_state.prefix_chunks
        )
        block_state.completed_chunks = list(range(block_state.prefix_chunks))
        self.set_block_state(state, block_state)
        return components, state


class MagiPreparePrefixStep(ModularPipelineBlocks):
    model_name = "magi"

    @property
    def description(self):
        return "Select complete prefix chunks for clean-cache initialization."

    @property
    def inputs(self):
        return [
            InputParam(
                "latents",
                type_hint=torch.Tensor,
                description="FP32 latent state shaped (batch, channels, frames, height, width), including prefix slots.",
                required=True,
            ),
            InputParam("chunk_width", type_hint=int, description="Number of latent frames in each chunk.", default=6),
            InputParam(
                "conditioning_latents",
                required=True,
                type_hint=torch.Tensor,
                description="Scaled prefix matching the generated latent batch and device.",
            ),
        ]

    @property
    def intermediate_outputs(self):
        return [
            OutputParam(
                "prefix_latents",
                type_hint=torch.Tensor,
                description="Optional full-chunk clean prefix; replaces the leading latent slots and remains unchanged.",
            )
        ]

    def __call__(self, components, state):
        block_state = self.get_block_state(state)
        prefix = block_state.conditioning_latents
        if (
            not isinstance(block_state.chunk_width, int)
            or isinstance(block_state.chunk_width, bool)
            or block_state.chunk_width <= 0
        ):
            raise ValueError("chunk_width must be a positive integer.")
        if (
            not isinstance(block_state.latents, torch.Tensor)
            or not block_state.latents.is_floating_point()
            or block_state.latents.ndim != 5
            or min(block_state.latents.shape) <= 0
        ):
            raise ValueError(
                "latents must be a nonempty floating-point tensor with shape (batch, channels, frames, height, width)."
            )
        if (
            not isinstance(prefix, torch.Tensor)
            or prefix.ndim != 5
            or prefix.shape[:2] != block_state.latents.shape[:2]
            or prefix.shape[3:] != block_state.latents.shape[3:]
            or not 0 < prefix.shape[2] < block_state.latents.shape[2]
            or prefix.device != block_state.latents.device
            or not prefix.is_floating_point()
        ):
            raise ValueError("conditioning_latents must be a prefix matching the latent batch, shape, and device.")
        length = prefix.shape[2] // block_state.chunk_width * block_state.chunk_width
        block_state.prefix_latents = prefix[:, :, :length] if length else None
        self.set_block_state(state, block_state)
        return components, state
