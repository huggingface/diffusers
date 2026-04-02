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

import inspect

import numpy as np
import torch

from ...models import HunyuanVideo15Transformer3DModel
from ...schedulers import FlowMatchEulerDiscreteScheduler
from ...utils import logging
from ...utils.torch_utils import randn_tensor
from ..modular_pipeline import ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam
from .modular_pipeline import HunyuanVideo15ModularPipeline


logger = logging.get_logger(__name__)


def retrieve_timesteps(
    scheduler,
    num_inference_steps=None,
    device=None,
    timesteps=None,
    sigmas=None,
    **kwargs,
):
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed.")
    if timesteps is not None:
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom sigmas."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class HunyuanVideo15TextInputStep(ModularPipelineBlocks):
    model_name = "hunyuan-video-1.5"

    @property
    def description(self) -> str:
        return "Input processing step that determines batch_size and dtype"

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("transformer", HunyuanVideo15Transformer3DModel),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("num_videos_per_prompt", default=1),
            InputParam("prompt_embeds", required=True, type_hint=torch.Tensor),
            InputParam("batch_size", type_hint=int),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("batch_size", type_hint=int),
            OutputParam("dtype", type_hint=torch.dtype),
        ]

    @torch.no_grad()
    def __call__(self, components: HunyuanVideo15ModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        block_state.batch_size = getattr(block_state, "batch_size", None) or block_state.prompt_embeds.shape[0]
        block_state.dtype = components.transformer.dtype

        self.set_block_state(state, block_state)
        return components, state


class HunyuanVideo15SetTimestepsStep(ModularPipelineBlocks):
    model_name = "hunyuan-video-1.5"

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("scheduler", FlowMatchEulerDiscreteScheduler),
        ]

    @property
    def description(self) -> str:
        return "Step that sets the scheduler's timesteps for inference"

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("num_inference_steps", default=50),
            InputParam("sigmas"),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("timesteps", type_hint=torch.Tensor),
            OutputParam("num_inference_steps", type_hint=int),
        ]

    @torch.no_grad()
    def __call__(self, components: HunyuanVideo15ModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        device = components._execution_device

        sigmas = block_state.sigmas
        if sigmas is None:
            sigmas = np.linspace(1.0, 0.0, block_state.num_inference_steps + 1)[:-1]

        block_state.timesteps, block_state.num_inference_steps = retrieve_timesteps(
            components.scheduler,
            block_state.num_inference_steps,
            device,
            sigmas=sigmas,
        )

        self.set_block_state(state, block_state)
        return components, state


class HunyuanVideo15PrepareLatentsStep(ModularPipelineBlocks):
    model_name = "hunyuan-video-1.5"

    @property
    def description(self) -> str:
        return "Prepare latents step for text-to-video generation"

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("height", type_hint=int),
            InputParam("width", type_hint=int),
            InputParam("num_frames", type_hint=int, default=121),
            InputParam("latents", type_hint=torch.Tensor | None),
            InputParam("num_videos_per_prompt", type_hint=int, default=1),
            InputParam("generator"),
            InputParam("batch_size", required=True, type_hint=int),
            InputParam("dtype", type_hint=torch.dtype),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("latents", type_hint=torch.Tensor),
            OutputParam("cond_latents_concat", type_hint=torch.Tensor),
            OutputParam("mask_concat", type_hint=torch.Tensor),
            OutputParam("image_embeds", type_hint=torch.Tensor),
        ]

    @torch.no_grad()
    def __call__(self, components: HunyuanVideo15ModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        device = components._execution_device
        dtype = block_state.dtype

        batch_size = block_state.batch_size * block_state.num_videos_per_prompt
        height = block_state.height or components.default_height
        width = block_state.width or components.default_width
        num_frames = block_state.num_frames

        num_channels_latents = components.num_channels_latents
        latent_height = height // components.vae_spatial_compression_ratio
        latent_width = width // components.vae_spatial_compression_ratio
        latent_num_frames = (num_frames - 1) // components.vae_temporal_compression_ratio + 1

        if block_state.latents is not None:
            block_state.latents = block_state.latents.to(device=device, dtype=dtype)
        else:
            shape = (batch_size, num_channels_latents, latent_num_frames, latent_height, latent_width)
            block_state.latents = randn_tensor(shape, generator=block_state.generator, device=device, dtype=dtype)

        # T2V: zero cond_latents and mask
        b, c, f, h, w = block_state.latents.shape
        block_state.cond_latents_concat = torch.zeros(b, c, f, h, w, dtype=dtype, device=device)
        block_state.mask_concat = torch.zeros(b, 1, f, h, w, dtype=dtype, device=device)

        # T2V: zero image_embeds
        block_state.image_embeds = torch.zeros(
            block_state.batch_size,
            components.vision_num_semantic_tokens,
            components.vision_states_dim,
            dtype=dtype,
            device=device,
        )

        self.set_block_state(state, block_state)
        return components, state


def retrieve_latents(encoder_output, generator=None, sample_mode="sample"):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    raise AttributeError("Could not access latents of provided encoder_output")


class HunyuanVideo15Image2VideoPrepareLatentsStep(ModularPipelineBlocks):
    model_name = "hunyuan-video-1.5"

    @property
    def description(self) -> str:
        return "Prepare latents step for image-to-video: encodes the first frame and creates conditioning mask"

    @property
    def expected_components(self) -> list[ComponentSpec]:
        from ...models import AutoencoderKLHunyuanVideo15
        from transformers import SiglipVisionModel, SiglipImageProcessor
        return [
            ComponentSpec("vae", AutoencoderKLHunyuanVideo15),
            ComponentSpec("image_encoder", SiglipVisionModel),
            ComponentSpec("feature_extractor", SiglipImageProcessor),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("image", required=True),
            InputParam("height", type_hint=int),
            InputParam("width", type_hint=int),
            InputParam("num_frames", type_hint=int, default=121),
            InputParam("latents", type_hint=torch.Tensor | None),
            InputParam("num_videos_per_prompt", type_hint=int, default=1),
            InputParam("generator"),
            InputParam("batch_size", required=True, type_hint=int),
            InputParam("dtype", type_hint=torch.dtype),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("latents", type_hint=torch.Tensor),
            OutputParam("cond_latents_concat", type_hint=torch.Tensor),
            OutputParam("mask_concat", type_hint=torch.Tensor),
            OutputParam("image_embeds", type_hint=torch.Tensor),
        ]

    @torch.no_grad()
    def __call__(self, components: HunyuanVideo15ModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        device = components._execution_device
        dtype = block_state.dtype

        batch_size = block_state.batch_size * block_state.num_videos_per_prompt
        num_frames = block_state.num_frames

        # Resize/crop image to target resolution first (determines latent dims)
        image = block_state.image
        from ...pipelines.hunyuan_video1_5.image_processor import HunyuanVideo15ImageProcessor
        video_processor = HunyuanVideo15ImageProcessor(vae_scale_factor=components.vae_spatial_compression_ratio)
        height, width = video_processor.calculate_default_height_width(
            height=image.size[1], width=image.size[0], target_size=components.target_size
        )
        image = video_processor.resize(image, height=height, width=width, resize_mode="crop")

        num_channels_latents = components.num_channels_latents
        latent_height = height // components.vae_spatial_compression_ratio
        latent_width = width // components.vae_spatial_compression_ratio
        latent_num_frames = (num_frames - 1) // components.vae_temporal_compression_ratio + 1

        if block_state.latents is not None:
            block_state.latents = block_state.latents.to(device=device, dtype=dtype)
        else:
            shape = (batch_size, num_channels_latents, latent_num_frames, latent_height, latent_width)
            block_state.latents = randn_tensor(shape, generator=block_state.generator, device=device, dtype=dtype)

        # Encode image for Siglip embeddings
        image_encoder_dtype = next(components.image_encoder.parameters()).dtype
        image_inputs = components.feature_extractor.preprocess(
            images=image, do_resize=True, return_tensors="pt", do_convert_rgb=True
        )
        image_inputs = image_inputs.to(device=device, dtype=image_encoder_dtype)
        image_embeds = components.image_encoder(**image_inputs).last_hidden_state
        image_embeds = image_embeds.repeat(batch_size, 1, 1)
        block_state.image_embeds = image_embeds.to(device=device, dtype=dtype)

        # Encode image for VAE conditioning latents
        vae_dtype = components.vae.dtype
        image_tensor = video_processor.preprocess(image, height=height, width=width).to(device, dtype=vae_dtype)
        image_tensor = image_tensor.unsqueeze(2)
        image_latents = retrieve_latents(components.vae.encode(image_tensor), sample_mode="argmax")
        image_latents = image_latents * components.vae.config.scaling_factor

        b, c, f, h, w = block_state.latents.shape
        latent_condition = image_latents.repeat(batch_size, 1, f, 1, 1)
        latent_condition[:, :, 1:, :, :] = 0
        block_state.cond_latents_concat = latent_condition.to(device=device, dtype=dtype)

        latent_mask = torch.zeros(b, 1, f, h, w, dtype=dtype, device=device)
        latent_mask[:, :, 0, :, :] = 1.0
        block_state.mask_concat = latent_mask

        self.set_block_state(state, block_state)
        return components, state
