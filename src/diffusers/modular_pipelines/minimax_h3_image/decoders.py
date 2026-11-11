# Copyright 2026 The MiniMax and HuggingFace Teams. All rights reserved.
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
from ...models import AutoencoderKLMiniMaxH3Image
from ...utils import logging
from ..modular_pipeline import ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam
from .modular_pipeline import MiniMaxH3ImageModularPipeline


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class MiniMaxH3ImageAfterDenoiseStep(ModularPipelineBlocks):
    model_name = "minimax-h3-image"

    @property
    def description(self) -> str:
        return (
            "Turns the denoised rows of the packed sequence back into latents: drops the reference rows the loop "
            "never wrote, then unpacks the generated rows into `(1, latent_channels, latent_height, latent_width)`. "
            "The decoder then takes latents from any source, and popping it leaves latents in hand."
        )

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                name="latents",
                type_hint=torch.Tensor,
                required=True,
                description="The denoised image rows of the packed sequence, reference rows first.",
            ),
            InputParam(
                name="num_reference_rows",
                type_hint=int,
                default=0,
                description="How many leading image rows are reference rows and are dropped here.",
            ),
            InputParam(
                name="latent_height", type_hint=int, required=True, description="Height of the generated latents."
            ),
            InputParam(
                name="latent_width", type_hint=int, required=True, description="Width of the generated latents."
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "latents",
                type_hint=torch.Tensor,
                description="The generated latents, of shape `(1, latent_channels, latent_height, latent_width)`.",
            ),
        ]

    @torch.no_grad()
    def __call__(
        self, components: MiniMaxH3ImageModularPipeline, state: PipelineState
    ) -> tuple[MiniMaxH3ImageModularPipeline, PipelineState]:
        block_state = self.get_block_state(state)
        _, patch_h, patch_w = components.patch_size
        channels = components.vae_latent_channels

        # The inverse of the patchify in the prepare-latents step: rows are row-major over the patch grid.
        rows = block_state.latents[block_state.num_reference_rows :]
        rows = rows.reshape(
            -1, block_state.latent_height // patch_h, block_state.latent_width // patch_w, channels, patch_h, patch_w
        )
        rows = rows.permute(0, 3, 1, 4, 2, 5)
        block_state.latents = rows.reshape(
            -1, channels, block_state.latent_height, block_state.latent_width
        ).contiguous()

        self.set_block_state(state, block_state)
        return components, state


class MiniMaxH3ImageDecodeStep(ModularPipelineBlocks):
    model_name = "minimax-h3-image"

    @property
    def description(self) -> str:
        return (
            "Denormalizes the generated latents and decodes them into an image. The VAE produces ImageNet-normalized "
            "RGB that is reverted here. An image generated on a larger canvas than requested is resized back to "
            "`output_height x output_width`."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("vae", AutoencoderKLMiniMaxH3Image),
            ComponentSpec(
                "image_processor",
                VaeImageProcessor,
                # The VAE's ImageNet normalization is reverted by this block, not by the processor.
                config=FrozenDict({"vae_scale_factor": 16, "do_normalize": False}),
                default_creation_method="from_config",
            ),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                name="latents",
                type_hint=torch.Tensor,
                required=True,
                description="The generated latents, of shape `(1, latent_channels, latent_height, latent_width)`.",
            ),
            InputParam(
                name="output_height",
                type_hint=int,
                description="Height of the returned image in pixels. Defaults to the height the latents decode to.",
            ),
            InputParam(
                name="output_width",
                type_hint=int,
                description="Width of the returned image in pixels. Defaults to the width the latents decode to.",
            ),
            InputParam.template("output_type"),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [OutputParam.template("images")]

    @torch.no_grad()
    def __call__(
        self, components: MiniMaxH3ImageModularPipeline, state: PipelineState
    ) -> tuple[MiniMaxH3ImageModularPipeline, PipelineState]:
        block_state = self.get_block_state(state)
        device = components._execution_device

        latents_mean = torch.tensor(components.vae.config.latents_mean, device=device).view(1, -1, 1, 1)
        latents_std = torch.tensor(components.vae.config.latents_std, device=device).view(1, -1, 1, 1)
        latents = block_state.latents.to(device) * latents_std + latents_mean
        image = components.vae.decode(latents, return_dict=False)[0]

        pixel_mean = torch.tensor(components.pixel_mean, device=device).view(1, -1, 1, 1)
        pixel_std = torch.tensor(components.pixel_std, device=device).view(1, -1, 1, 1)
        image = (image.float() * pixel_std + pixel_mean).clamp(0, 1)

        # The released pipeline quantizes its images to 8 bits by truncation, where the image processor rounds, and
        # resizes the 8-bit image. Quantizing here makes the processor's rounding reproduce the same bytes.
        image = (image * 255).clamp(0, 255).to(torch.uint8).float() / 255

        output_height = block_state.output_height or image.shape[-2]
        output_width = block_state.output_width or image.shape[-1]
        if (output_height, output_width) != tuple(image.shape[-2:]):
            image = components.image_processor.postprocess(image, output_type="pil")
            image = [components.image_processor.resize(item, output_height, output_width) for item in image]
            image = components.image_processor.numpy_to_pt(components.image_processor.pil_to_numpy(image))
        block_state.images = components.image_processor.postprocess(image, output_type=block_state.output_type)

        self.set_block_state(state, block_state)
        return components, state
