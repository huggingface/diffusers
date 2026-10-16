# Copyright 2026 Black Forest Labs and The HuggingFace Team. All rights reserved.
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

from __future__ import annotations

from typing import Any, Union

import numpy as np
import PIL
import torch

from ...configuration_utils import FrozenDict
from ...models import AutoencoderKLFlux2
from ...utils import logging
from ..modular_pipeline import ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam
from .image_processor import Flux3ImageProcessor
from .modular_pipeline import Flux3ModularPipeline


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class Flux3UnpackLatentsStep(ModularPipelineBlocks):
    model_name = "flux3"

    @property
    def description(self) -> str:
        return "Step that unpacks the image latent tokens from the denoising step back onto their 2D grid."

    @property
    def inputs(self) -> list[tuple[str, Any]]:
        return [
            InputParam(
                "latents",
                required=True,
                type_hint=torch.Tensor,
                description="The denoised image latent tokens from the denoising step. Shape: (B, seq_len, C)",
            ),
            # no default: the size is already baked into the latents, every workflow sets these upstream
            InputParam(
                "height",
                type_hint=int,
                description="The height in pixels of the generated image; must match the size the latents were prepared at",
            ),
            InputParam(
                "width",
                type_hint=int,
                description="The width in pixels of the generated image; must match the size the latents were prepared at",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "latents",
                type_hint=torch.Tensor,
                description="The denoised latents, unpacked to (B, C, height, width).",
            )
        ]

    @torch.no_grad()
    def __call__(self, components: Flux3ModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        latent_height = block_state.height // (components.vae_scale_factor * 2)
        latent_width = block_state.width // (components.vae_scale_factor * 2)
        if latent_height * latent_width != block_state.latents.shape[1]:
            raise ValueError(
                f"`latents` has {block_state.latents.shape[1]} tokens, which does not match "
                f"height x width = {block_state.height} x {block_state.width} ({latent_height * latent_width} tokens)"
            )

        # unpack: (batch_size, latent_height * latent_width, num_channels) -> (batch_size, num_channels, latent_height, latent_width)
        batch_size, _, num_channels = block_state.latents.shape
        block_state.latents = block_state.latents.view(batch_size, latent_height, latent_width, num_channels).permute(
            0, 3, 1, 2
        )

        self.set_block_state(state, block_state)
        return components, state


class Flux3DecodeStep(ModularPipelineBlocks):
    model_name = "flux3"

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("vae", AutoencoderKLFlux2),
            ComponentSpec(
                "image_processor",
                Flux3ImageProcessor,
                config=FrozenDict({"vae_scale_factor": 16, "vae_latent_channels": 32}),
                default_creation_method="from_config",
            ),
        ]

    @property
    def description(self) -> str:
        return (
            "Step that decodes the denoised latents into images using the FLUX.3 image VAE (the FLUX.2 packed "
            "autoencoder) with batch norm denormalization."
        )

    @property
    def inputs(self) -> list[tuple[str, Any]]:
        return [
            InputParam.template("output_type"),
            InputParam(
                "latents",
                required=True,
                type_hint=torch.Tensor,
                description="The denoised latents from the denoising step",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "images",
                type_hint=Union[list[PIL.Image.Image], torch.Tensor, np.ndarray],
                description="The generated images, can be a list of PIL.Image.Image, torch.Tensor or a numpy array",
            )
        ]

    @staticmethod
    def _unpatchify_latents(latents):
        """Convert patchified latents back to regular format."""
        batch_size, num_channels_latents, height, width = latents.shape
        latents = latents.reshape(batch_size, num_channels_latents // (2 * 2), 2, 2, height, width)
        latents = latents.permute(0, 1, 4, 2, 5, 3)
        latents = latents.reshape(batch_size, num_channels_latents // (2 * 2), height * 2, width * 2)
        return latents

    @torch.no_grad()
    def __call__(self, components: Flux3ModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        vae = components.vae

        latents = block_state.latents

        latents_bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
        latents_bn_std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps).to(
            latents.device, latents.dtype
        )
        latents = latents * latents_bn_std + latents_bn_mean

        latents = self._unpatchify_latents(latents)

        block_state.images = vae.decode(latents.to(vae.dtype), return_dict=False)[0]
        block_state.images = components.image_processor.postprocess(
            block_state.images, output_type=block_state.output_type
        )

        self.set_block_state(state, block_state)
        return components, state
