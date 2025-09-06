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
from ...image_processor import VaeImageProcessor
from ...models import AutoencoderKL
from ...models.attention_processor import AttnProcessor2_0, XFormersAttnProcessor
from ...utils import logging
from ..modular_pipeline import (
    ModularPipelineBlocks,
    PipelineState,
)
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class StableDiffusionXLDecodeStep(ModularPipelineBlocks):
    model_name = "stable-diffusion-xl"

    @property
    def expected_components(self) -> List[ComponentSpec]:
        return [
            ComponentSpec("vae", AutoencoderKL),
            ComponentSpec(
                "image_processor",
                VaeImageProcessor,
                config=FrozenDict({"vae_scale_factor": 8}),
                default_creation_method="from_config",
            ),
        ]

    @property
    def description(self) -> str:
        return "Step that decodes the denoised latents into images"

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
        ]

    @property
    def intermediate_outputs(self) -> List[str]:
        return [
            OutputParam(
                "images",
                type_hint=Union[List[PIL.Image.Image], List[torch.Tensor], List[np.array]],
                description="The generated images, can be a PIL.Image.Image, torch.Tensor or a numpy array",
            )
        ]

    @staticmethod
    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_upscale.StableDiffusionUpscalePipeline.upcast_vae with self->components
    def upcast_vae(components):
        dtype = components.vae.dtype
        components.vae.to(dtype=torch.float32)
        use_torch_2_0_or_xformers = isinstance(
            components.vae.decoder.mid_block.attentions[0].processor,
            (
                AttnProcessor2_0,
                XFormersAttnProcessor,
            ),
        )
        # if xformers or torch_2_0 is used attention block does not need
        # to be in float32 which can save lots of memory
        if use_torch_2_0_or_xformers:
            components.vae.post_quant_conv.to(dtype)
            components.vae.decoder.conv_in.to(dtype)
            components.vae.decoder.mid_block.to(dtype)

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        if not block_state.output_type == "latent":
            latents = block_state.latents
            # make sure the VAE is in float32 mode, as it overflows in float16
            block_state.needs_upcasting = components.vae.dtype == torch.float16 and components.vae.config.force_upcast

            if block_state.needs_upcasting:
                self.upcast_vae(components)
                latents = latents.to(next(iter(components.vae.post_quant_conv.parameters())).dtype)
            elif latents.dtype != components.vae.dtype:
                if torch.backends.mps.is_available():
                    # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                    components.vae = components.vae.to(latents.dtype)

            # unscale/denormalize the latents
            # denormalize with the mean and std if available and not None
            block_state.has_latents_mean = (
                hasattr(components.vae.config, "latents_mean") and components.vae.config.latents_mean is not None
            )
            block_state.has_latents_std = (
                hasattr(components.vae.config, "latents_std") and components.vae.config.latents_std is not None
            )
            if block_state.has_latents_mean and block_state.has_latents_std:
                block_state.latents_mean = (
                    torch.tensor(components.vae.config.latents_mean).view(1, 4, 1, 1).to(latents.device, latents.dtype)
                )
                block_state.latents_std = (
                    torch.tensor(components.vae.config.latents_std).view(1, 4, 1, 1).to(latents.device, latents.dtype)
                )
                latents = (
                    latents * block_state.latents_std / components.vae.config.scaling_factor + block_state.latents_mean
                )
            else:
                latents = latents / components.vae.config.scaling_factor

            block_state.images = components.vae.decode(latents, return_dict=False)[0]

            # cast back to fp16 if needed
            if block_state.needs_upcasting:
                components.vae.to(dtype=torch.float16)
        else:
            block_state.images = block_state.latents

        # apply watermark if available
        if hasattr(components, "watermark") and components.watermark is not None:
            block_state.images = components.watermark.apply_watermark(block_state.images)

        block_state.images = components.image_processor.postprocess(
            block_state.images, output_type=block_state.output_type
        )

        self.set_block_state(state, block_state)

        return components, state


class StableDiffusionXLInpaintOverlayMaskStep(ModularPipelineBlocks):
    model_name = "stable-diffusion-xl"

    @property
    def description(self) -> str:
        return (
            "A post-processing step that overlays the mask on the image (inpainting task only).\n"
            + "only needed when you are using the `padding_mask_crop` option when pre-processing the image and mask"
        )

    @property
    def expected_components(self) -> List[ComponentSpec]:
        return [
            ComponentSpec(
                "image_processor",
                VaeImageProcessor,
                config=FrozenDict({"vae_scale_factor": 8}),
                default_creation_method="from_config",
            ),
        ]

    @property
    def inputs(self) -> List[Tuple[str, Any]]:
        return [
            InputParam("image"),
            InputParam("mask_image"),
            InputParam("padding_mask_crop"),
            InputParam(
                "images",
                type_hint=Union[List[PIL.Image.Image], List[torch.Tensor], List[np.array]],
                description="The generated images from the decode step",
            ),
            InputParam(
                "crops_coords",
                type_hint=Tuple[int, int],
                description="The crop coordinates to use for preprocess/postprocess the image and mask, for inpainting task only. Can be generated in vae_encode step.",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        if block_state.padding_mask_crop is not None and block_state.crops_coords is not None:
            block_state.images = [
                components.image_processor.apply_overlay(
                    block_state.mask_image, block_state.image, i, block_state.crops_coords
                )
                for i in block_state.images
            ]

        self.set_block_state(state, block_state)

        return components, state
