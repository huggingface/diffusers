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

from typing import Any, List, Tuple, Union

import numpy as np
import PIL
import torch

from ...configuration_utils import FrozenDict
from ...models import AutoencoderKLFlux2
from ...pipelines.flux2.image_processor import Flux2ImageProcessor
from ...utils import logging
from ..modular_pipeline import ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class Flux2DecodeStep(ModularPipelineBlocks):
    model_name = "flux2"

    @property
    def expected_components(self) -> List[ComponentSpec]:
        return [
            ComponentSpec("vae", AutoencoderKLFlux2),
            ComponentSpec(
                "image_processor",
                Flux2ImageProcessor,
                config=FrozenDict({"vae_scale_factor": 16, "vae_latent_channels": 32}),
                default_creation_method="from_config",
            ),
        ]

    @property
    def description(self) -> str:
        return "Step that decodes the denoised latents into images using Flux2 VAE with batch norm denormalization"

    @property
    def inputs(self) -> List[Tuple[str, Any]]:
        return [
            InputParam("output_type", default="pil"),
            InputParam(
                "latents",
                required=True,
                type_hint=torch.Tensor,
                description="The denoised latents from the denoising step",
            ),
            InputParam(
                "latent_ids",
                required=True,
                type_hint=torch.Tensor,
                description="Position IDs for the latents, used for unpacking",
            ),
        ]

    @property
    def intermediate_outputs(self) -> List[str]:
        return [
            OutputParam(
                "images",
                type_hint=Union[List[PIL.Image.Image], torch.Tensor, np.ndarray],
                description="The generated images, can be a list of PIL.Image.Image, torch.Tensor or a numpy array",
            )
        ]

    @staticmethod
    def _unpack_latents_with_ids(x: torch.Tensor, x_ids: torch.Tensor) -> torch.Tensor:
        """
        Unpack latents using position IDs to scatter tokens into place.

        Args:
            x: Packed latents tensor of shape (B, seq_len, C)
            x_ids: Position IDs tensor of shape (B, seq_len, 4) with (T, H, W, L) coordinates

        Returns:
            Unpacked latents tensor of shape (B, C, H, W)
        """
        x_list = []
        for data, pos in zip(x, x_ids):
            _, ch = data.shape  # noqa: F841
            h_ids = pos[:, 1].to(torch.int64)
            w_ids = pos[:, 2].to(torch.int64)

            h = torch.max(h_ids) + 1
            w = torch.max(w_ids) + 1

            flat_ids = h_ids * w + w_ids

            out = torch.zeros((h * w, ch), device=data.device, dtype=data.dtype)
            out.scatter_(0, flat_ids.unsqueeze(1).expand(-1, ch), data)

            out = out.view(h, w, ch).permute(2, 0, 1)
            x_list.append(out)

        return torch.stack(x_list, dim=0)

    @staticmethod
    def _unpatchify_latents(latents):
        """Convert patchified latents back to regular format."""
        batch_size, num_channels_latents, height, width = latents.shape
        latents = latents.reshape(batch_size, num_channels_latents // (2 * 2), 2, 2, height, width)
        latents = latents.permute(0, 1, 4, 2, 5, 3)
        latents = latents.reshape(batch_size, num_channels_latents // (2 * 2), height * 2, width * 2)
        return latents

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        vae = components.vae

        if block_state.output_type == "latent":
            block_state.images = block_state.latents
        else:
            latents = block_state.latents
            latent_ids = block_state.latent_ids

            latents = self._unpack_latents_with_ids(latents, latent_ids)

            latents_bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
            latents_bn_std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps).to(
                latents.device, latents.dtype
            )
            latents = latents * latents_bn_std + latents_bn_mean

            latents = self._unpatchify_latents(latents)

            block_state.images = vae.decode(latents, return_dict=False)[0]
            block_state.images = components.image_processor.postprocess(
                block_state.images, output_type=block_state.output_type
            )

        self.set_block_state(state, block_state)
        return components, state
