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

from typing import Any

import torch

from ...configuration_utils import FrozenDict
from ...models import AutoencoderKLLTXVideo
from ...utils import deprecate, logging
from ...utils.torch_utils import randn_tensor
from ...video_processor import VideoProcessor
from ..modular_pipeline import ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam
from .modular_pipeline import LTXVideoPachifier


logger = logging.get_logger(__name__)


def _denormalize_latents(
    latents: torch.Tensor, latents_mean: torch.Tensor, latents_std: torch.Tensor, scaling_factor: float = 1.0
) -> torch.Tensor:
    # Denormalize latents across the channel dimension [B, C, F, H, W]
    latents_mean = latents_mean.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
    latents_std = latents_std.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
    latents = latents * latents_std / scaling_factor + latents_mean
    return latents


class LTXUnpackLatentsStep(ModularPipelineBlocks):
    model_name = "ltx"

    @property
    def description(self) -> str:
        return (
            "Unpacks the denoised latents from the transformer's token layout back into the `[B, C, F, H, W]` form "
            "the VAE takes (still normalized). Closes the core denoise group, so the blocks that follow take the same "
            "form the VAE encoder produces and need no geometry inputs."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec(
                "pachifier",
                LTXVideoPachifier,
                config=FrozenDict({"patch_size": 1, "patch_size_t": 1}),
                default_creation_method="from_config",
            ),
        ]

    @property
    def inputs(self) -> list[tuple[str, Any]]:
        return [
            InputParam(
                "latents",
                required=True,
                type_hint=torch.Tensor,
                description="The denoised latents from the denoising step, packed, of shape `[B, S, C]`.",
            ),
            InputParam.template("height", default=512),
            InputParam.template("width", default=704),
            InputParam("num_frames", type_hint=int, default=161),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "latents",
                type_hint=torch.Tensor,
                description="The denoised latents of shape `[B, C, F, H, W]` (normalized, not packed).",
            )
        ]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        latent_num_frames = (block_state.num_frames - 1) // components.vae_temporal_compression_ratio + 1
        latent_height = block_state.height // components.vae_spatial_compression_ratio
        latent_width = block_state.width // components.vae_spatial_compression_ratio
        block_state.latents = components.pachifier.unpack_latents(
            block_state.latents, latent_num_frames, latent_height, latent_width
        )
        self.set_block_state(state, block_state)
        return components, state


class LTXVaeDecoderStep(ModularPipelineBlocks):
    model_name = "ltx"

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("vae", AutoencoderKLLTXVideo),
            ComponentSpec(
                "video_processor",
                VideoProcessor,
                config=FrozenDict({"vae_scale_factor": 32}),
                default_creation_method="from_config",
            ),
            ComponentSpec(
                "pachifier",
                LTXVideoPachifier,
                config=FrozenDict({"patch_size": 1, "patch_size_t": 1}),
                default_creation_method="from_config",
            ),
        ]

    @property
    def description(self) -> str:
        return "Step that decodes the denoised latents into videos"

    @property
    def inputs(self) -> list[tuple[str, Any]]:
        return [
            InputParam.template("latents", required=True),
            InputParam.template("output_type", default="np"),
            InputParam("decode_timestep", default=0.0),
            InputParam("decode_noise_scale", default=None),
            InputParam.template("generator"),
            InputParam.template("batch_size"),
            InputParam.template("dtype", required=True),
            # Only read on the deprecated path that still accepts packed `[B, S, C]` latents.
            InputParam.template("height", default=512),
            InputParam.template("width", default=704),
            InputParam("num_frames", type_hint=int, default=161),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [OutputParam.template("videos")]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        vae = components.vae

        latents = block_state.latents
        if latents.ndim == 3:
            deprecate(
                "packed latents",
                "1.0.0",
                "Passing packed latents of shape `[B, S, C]` to the decode step is deprecated; the denoise group "
                "now unpacks them. Pass latents of shape `[B, C, F, H, W]` instead.",
            )
            latent_num_frames = (block_state.num_frames - 1) // components.vae_temporal_compression_ratio + 1
            latent_height = block_state.height // components.vae_spatial_compression_ratio
            latent_width = block_state.width // components.vae_spatial_compression_ratio
            latents = components.pachifier.unpack_latents(latents, latent_num_frames, latent_height, latent_width)

        latents = _denormalize_latents(latents, vae.latents_mean, vae.latents_std, vae.config.scaling_factor)
        latents = latents.to(block_state.dtype)

        if not vae.config.timestep_conditioning:
            timestep = None
        else:
            device = latents.device
            batch_size = block_state.batch_size
            decode_timestep = block_state.decode_timestep
            decode_noise_scale = block_state.decode_noise_scale

            noise = randn_tensor(latents.shape, generator=block_state.generator, device=device, dtype=latents.dtype)
            if not isinstance(decode_timestep, list):
                decode_timestep = [decode_timestep] * batch_size
            if decode_noise_scale is None:
                decode_noise_scale = decode_timestep
            elif not isinstance(decode_noise_scale, list):
                decode_noise_scale = [decode_noise_scale] * batch_size

            timestep = torch.tensor(decode_timestep, device=device, dtype=latents.dtype)
            decode_noise_scale = torch.tensor(decode_noise_scale, device=device, dtype=latents.dtype)[
                :, None, None, None, None
            ]
            latents = (1 - decode_noise_scale) * latents + decode_noise_scale * noise

        latents = latents.to(vae.dtype)
        video = vae.decode(latents, timestep, return_dict=False)[0]
        block_state.videos = components.video_processor.postprocess_video(video, output_type=block_state.output_type)

        self.set_block_state(state, block_state)
        return components, state
