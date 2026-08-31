# Copyright 2026 Ideogram AI and The HuggingFace Team. All rights reserved.
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


import torch

from ...configuration_utils import FrozenDict
from ...image_processor import VaeImageProcessor
from ...models import AutoencoderKLFlux2
from ...utils import logging
from ..modular_pipeline import ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam
from .modular_pipeline import Ideogram4ModularPipeline


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# auto_docstring
class Ideogram4DecodeStep(ModularPipelineBlocks):
    """
    Step that decodes the unpatchified (B, ae_channels, H, W) latents into images: de-normalizes with the VAE
    batch-norm statistics and decodes through the VAE.

      Components:
          vae (`AutoencoderKLFlux2`) image_processor (`VaeImageProcessor`)

      Inputs:
          output_type (`str`, *optional*, defaults to pil):
              Output format: 'pil', 'np', 'pt'.
          latents (`Tensor`):
              The unpatchified (B, ae_channels, H, W) latents to decode, from the after-denoise step.

      Outputs:
          images (`list`):
              Generated images.
    """

    model_name = "ideogram4"

    @property
    def description(self) -> str:
        return (
            "Step that decodes the unpatchified (B, ae_channels, H, W) latents into images: de-normalizes with the "
            "VAE batch-norm statistics and decodes through the VAE."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("vae", AutoencoderKLFlux2),
            ComponentSpec(
                "image_processor",
                VaeImageProcessor,
                config=FrozenDict({"vae_scale_factor": 16}),
                default_creation_method="from_config",
            ),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("output_type", default="pil"),
            InputParam(
                name="latents",
                required=True,
                type_hint=torch.Tensor,
                description="The unpatchified (B, ae_channels, H, W) latents to decode, from the after-denoise step.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [OutputParam.template("images")]

    @torch.no_grad()
    def __call__(self, components: Ideogram4ModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        z = block_state.latents
        patch = components.patch_size
        ae_channels = z.shape[1]
        grid_h, grid_w = z.shape[2] // patch, z.shape[3] // patch

        # VAE bn stores per-channel statistics over the packed channels, laid out as (patch_row, patch_col,
        # ae_channel). Reshape them into an (ae_channels, patch, patch) tile and repeat across the grid so the
        # denormalization on the unpatchified latents matches the packed-space statistics.
        bn_mean = components.vae.bn.running_mean.view(patch, patch, ae_channels).permute(2, 0, 1)
        bn_std = torch.sqrt(components.vae.bn.running_var + components.vae.config.batch_norm_eps)
        bn_std = bn_std.view(patch, patch, ae_channels).permute(2, 0, 1)
        bn_mean = bn_mean.repeat(1, grid_h, grid_w).to(device=z.device, dtype=z.dtype)
        bn_std = bn_std.repeat(1, grid_h, grid_w).to(device=z.device, dtype=z.dtype)
        z = z * bn_std + bn_mean

        decoded = components.vae.decode(z.to(components.vae.dtype), return_dict=False)[0]
        block_state.images = components.image_processor.postprocess(
            decoded.float(), output_type=block_state.output_type
        )

        self.set_block_state(state, block_state)
        return components, state
