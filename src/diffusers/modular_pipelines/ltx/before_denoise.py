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

from ...models import LTXVideoTransformer3DModel
from ...pipelines.ltx.pipeline_ltx import LTXPipeline
from ...schedulers import FlowMatchEulerDiscreteScheduler
from ...utils import logging
from ...utils.torch_utils import randn_tensor
from ..modular_pipeline import ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam
from .modular_pipeline import LTXModularPipeline


logger = logging.get_logger(__name__)


def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: int | None = None,
    device: str | torch.device | None = None,
    timesteps: list[int] | None = None,
    sigmas: list[float] | None = None,
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


class LTXTextInputStep(ModularPipelineBlocks):
    model_name = "ltx"

    @property
    def description(self) -> str:
        return (
            "Input processing step that:\n"
            "  1. Determines `batch_size` and `dtype` based on `prompt_embeds`\n"
            "  2. Adjusts input tensor shapes based on `batch_size` and `num_videos_per_prompt`"
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("transformer", LTXVideoTransformer3DModel),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("num_videos_per_prompt", default=1),
            InputParam("guidance_scale", type_hint=float, default=3.0),
            InputParam("prompt_embeds", required=True, type_hint=torch.Tensor),
            InputParam("prompt_attention_mask", type_hint=torch.Tensor),
            InputParam("negative_prompt_embeds", type_hint=torch.Tensor),
            InputParam("negative_prompt_attention_mask", type_hint=torch.Tensor),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("batch_size", type_hint=int),
            OutputParam("dtype", type_hint=torch.dtype),
        ]

    @torch.no_grad()
    def __call__(self, components: LTXModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        # Set guidance_scale on guider so CFG is configured correctly
        guidance_scale = getattr(block_state, "guidance_scale", 3.0)
        if hasattr(components, "guider") and components.guider is not None:
            components.guider.guidance_scale = guidance_scale

        block_state.batch_size = block_state.prompt_embeds.shape[0]
        block_state.dtype = block_state.prompt_embeds.dtype
        num_videos = block_state.num_videos_per_prompt

        # Repeat prompt_embeds for num_videos_per_prompt
        _, seq_len, _ = block_state.prompt_embeds.shape
        block_state.prompt_embeds = block_state.prompt_embeds.repeat(1, num_videos, 1)
        block_state.prompt_embeds = block_state.prompt_embeds.view(
            block_state.batch_size * num_videos, seq_len, -1
        )

        if block_state.prompt_attention_mask is not None:
            block_state.prompt_attention_mask = block_state.prompt_attention_mask.repeat(num_videos, 1)

        if block_state.negative_prompt_embeds is not None:
            _, seq_len, _ = block_state.negative_prompt_embeds.shape
            block_state.negative_prompt_embeds = block_state.negative_prompt_embeds.repeat(1, num_videos, 1)
            block_state.negative_prompt_embeds = block_state.negative_prompt_embeds.view(
                block_state.batch_size * num_videos, seq_len, -1
            )

        if block_state.negative_prompt_attention_mask is not None:
            block_state.negative_prompt_attention_mask = block_state.negative_prompt_attention_mask.repeat(num_videos, 1)

        self.set_block_state(state, block_state)
        return components, state


class LTXSetTimestepsStep(ModularPipelineBlocks):
    model_name = "ltx"

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
            InputParam("timesteps"),
            InputParam("sigmas"),
            InputParam("height", type_hint=int, default=512),
            InputParam("width", type_hint=int, default=704),
            InputParam("num_frames", type_hint=int, default=161),
            InputParam("frame_rate", type_hint=int, default=25),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("timesteps", type_hint=torch.Tensor),
            OutputParam("num_inference_steps", type_hint=int),
            OutputParam("rope_interpolation_scale", type_hint=tuple),
        ]

    @torch.no_grad()
    def __call__(self, components: LTXModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        device = components._execution_device

        height = block_state.height
        width = block_state.width
        num_frames = block_state.num_frames
        frame_rate = block_state.frame_rate

        latent_num_frames = (num_frames - 1) // components.vae_temporal_compression_ratio + 1
        latent_height = height // components.vae_spatial_compression_ratio
        latent_width = width // components.vae_spatial_compression_ratio
        video_sequence_length = latent_num_frames * latent_height * latent_width

        custom_timesteps = block_state.timesteps
        sigmas = block_state.sigmas

        if custom_timesteps is not None:
            # User provided custom timesteps, don't compute sigmas
            block_state.timesteps, block_state.num_inference_steps = retrieve_timesteps(
                components.scheduler,
                block_state.num_inference_steps,
                device,
                custom_timesteps,
            )
        else:
            if sigmas is None:
                sigmas = np.linspace(1.0, 1 / block_state.num_inference_steps, block_state.num_inference_steps)

            mu = calculate_shift(
                video_sequence_length,
                components.scheduler.config.get("base_image_seq_len", 256),
                components.scheduler.config.get("max_image_seq_len", 4096),
                components.scheduler.config.get("base_shift", 0.5),
                components.scheduler.config.get("max_shift", 1.15),
            )

            block_state.timesteps, block_state.num_inference_steps = retrieve_timesteps(
                components.scheduler,
                block_state.num_inference_steps,
                device,
                sigmas=sigmas,
                mu=mu,
            )

        block_state.rope_interpolation_scale = (
            components.vae_temporal_compression_ratio / frame_rate,
            components.vae_spatial_compression_ratio,
            components.vae_spatial_compression_ratio,
        )

        self.set_block_state(state, block_state)
        return components, state


class LTXPrepareLatentsStep(ModularPipelineBlocks):
    model_name = "ltx"

    @property
    def description(self) -> str:
        return "Prepare latents step that prepares the latents for the text-to-video generation process"

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("height", type_hint=int, default=512),
            InputParam("width", type_hint=int, default=704),
            InputParam("num_frames", type_hint=int, default=161),
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
        ]

    @torch.no_grad()
    def __call__(self, components: LTXModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        device = components._execution_device

        batch_size = block_state.batch_size * block_state.num_videos_per_prompt
        num_channels_latents = components.transformer.config.in_channels

        if block_state.latents is not None:
            block_state.latents = block_state.latents.to(device=device, dtype=torch.float32)
        else:
            height = block_state.height // components.vae_spatial_compression_ratio
            width = block_state.width // components.vae_spatial_compression_ratio
            num_frames = (block_state.num_frames - 1) // components.vae_temporal_compression_ratio + 1

            shape = (batch_size, num_channels_latents, num_frames, height, width)
            block_state.latents = randn_tensor(shape, generator=block_state.generator, device=device, dtype=torch.float32)
            block_state.latents = LTXPipeline._pack_latents(
                block_state.latents,
                components.transformer_spatial_patch_size,
                components.transformer_temporal_patch_size,
            )

        self.set_block_state(state, block_state)
        return components, state


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.retrieve_latents
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


class LTXImage2VideoPrepareLatentsStep(ModularPipelineBlocks):
    model_name = "ltx"

    @property
    def description(self) -> str:
        return "Prepare latents step for image-to-video: encodes the first frame and creates a conditioning mask"

    @property
    def expected_components(self) -> list[ComponentSpec]:
        from ...models import AutoencoderKLLTXVideo
        return [
            ComponentSpec("vae", AutoencoderKLLTXVideo),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("image", required=True),
            InputParam("height", type_hint=int, default=512),
            InputParam("width", type_hint=int, default=704),
            InputParam("num_frames", type_hint=int, default=161),
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
            OutputParam("conditioning_mask", type_hint=torch.Tensor),
        ]

    @torch.no_grad()
    def __call__(self, components: LTXModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        device = components._execution_device

        batch_size = block_state.batch_size * block_state.num_videos_per_prompt
        num_channels_latents = components.transformer.config.in_channels

        height = block_state.height // components.vae_spatial_compression_ratio
        width = block_state.width // components.vae_spatial_compression_ratio
        num_frames = (block_state.num_frames - 1) // components.vae_temporal_compression_ratio + 1

        shape = (batch_size, num_channels_latents, num_frames, height, width)
        mask_shape = (batch_size, 1, num_frames, height, width)

        if block_state.latents is not None:
            conditioning_mask = block_state.latents.new_zeros(mask_shape)
            conditioning_mask[:, :, 0] = 1.0
            conditioning_mask = LTXPipeline._pack_latents(
                conditioning_mask,
                components.transformer_spatial_patch_size,
                components.transformer_temporal_patch_size,
            ).squeeze(-1)
            block_state.latents = block_state.latents.to(device=device, dtype=torch.float32)
            block_state.conditioning_mask = conditioning_mask
            self.set_block_state(state, block_state)
            return components, state

        image = block_state.image
        if not isinstance(image, torch.Tensor):
            from ...video_processor import VideoProcessor
            processor = VideoProcessor(vae_scale_factor=components.vae_spatial_compression_ratio)
            image = processor.preprocess(image, height=block_state.height, width=block_state.width)
            image = image.to(device=device, dtype=torch.float32)

        vae_dtype = components.vae.dtype

        if isinstance(block_state.generator, list):
            init_latents = [
                retrieve_latents(components.vae.encode(image[i].unsqueeze(0).unsqueeze(2).to(vae_dtype)), block_state.generator[i])
                for i in range(batch_size)
            ]
        else:
            init_latents = [
                retrieve_latents(components.vae.encode(img.unsqueeze(0).unsqueeze(2).to(vae_dtype)), block_state.generator)
                for img in image
            ]

        init_latents = torch.cat(init_latents, dim=0).to(torch.float32)
        init_latents = LTXPipeline._normalize_latents(
            init_latents, components.vae.latents_mean, components.vae.latents_std
        )
        init_latents = init_latents.repeat(1, 1, num_frames, 1, 1)

        actual_mask_shape = (init_latents.shape[0], 1, init_latents.shape[2], init_latents.shape[3], init_latents.shape[4])
        conditioning_mask = torch.zeros(actual_mask_shape, device=device, dtype=torch.float32)
        conditioning_mask[:, :, 0] = 1.0

        noise = randn_tensor(init_latents.shape, generator=block_state.generator, device=device, dtype=torch.float32)
        latents = init_latents * conditioning_mask + noise * (1 - conditioning_mask)

        conditioning_mask = LTXPipeline._pack_latents(
            conditioning_mask,
            components.transformer_spatial_patch_size,
            components.transformer_temporal_patch_size,
        ).squeeze(-1)
        latents = LTXPipeline._pack_latents(
            latents,
            components.transformer_spatial_patch_size,
            components.transformer_temporal_patch_size,
        )

        block_state.latents = latents
        block_state.conditioning_mask = conditioning_mask

        self.set_block_state(state, block_state)
        return components, state
