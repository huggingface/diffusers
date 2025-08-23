# Copyright 2025 Qwen-Image Team and The HuggingFace Team. All rights reserved.
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

from typing import List, Union
import torch
import numpy as np
import PIL

from ..modular_pipeline import ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam
from ...models import AutoencoderKLQwenImage
from ...configuration_utils import FrozenDict
from ...utils import logging
from ...image_processor import VaeImageProcessor

from .modular_pipeline import QwenImageModularPipeline

logger = logging.get_logger(__name__)


def unpack_latents(latents, height, width, vae_scale_factor):
    batch_size, num_patches, channels = latents.shape

    # VAE applies 8x compression on images but we must also account for packing which requires
    # latent height and width to be divisible by 2.
    height = 2 * (int(height) // (vae_scale_factor * 2))
    width = 2 * (int(width) // (vae_scale_factor * 2))

    latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)

    latents = latents.reshape(batch_size, channels // (2 * 2), 1, height, width)

    return latents


class QwenImageDecodeStep(ModularPipelineBlocks):
    model_name = "qwenimage"

    @property
    def description(self) -> str:
        return "Step that decodes the latents to images"
    
    @property
    def expected_components(self) -> List[ComponentSpec]:
        return [
            ComponentSpec("vae", AutoencoderKLQwenImage),
            ComponentSpec(
                "image_processor",
                VaeImageProcessor,
                config=FrozenDict({"vae_scale_factor": 16}),
                default_creation_method="from_config"
            ),
        ]
    
    @property
    def inputs(self) -> List[InputParam]:
        return [
            InputParam(name="height"),
            InputParam(name="width"),
            InputParam(name="latents", required=True, type_hint=torch.Tensor, description="The latents to decode, can be generated in the denoise step"),
            InputParam(name="output_type", default="pil", type_hint=str, description="The type of the output images, can be 'pil', 'np', 'pt'"),
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
    def check_inputs(output_type):
        if output_type not in ["pil", "np", "pt"]:
            raise ValueError(f"Invalid output_type: {output_type}")
    
    @torch.no_grad()
    def __call__(self, components: QwenImageModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        self.check_inputs(block_state.output_type)

        height = block_state.height or components.default_height
        width = block_state.width or components.default_width

        # YiYi Notes: remove support for output_type = "latents', we can just skip decode/encode step in modular
        block_state.latents = unpack_latents(block_state.latents, height, width, components.vae_scale_factor)
        block_state.latents = block_state.latents.to(components.vae.dtype)

        latents_mean = (
            torch.tensor(components.vae.config.latents_mean)
            .view(1, components.vae.config.z_dim, 1, 1, 1)
            .to(block_state.latents.device, block_state.latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(components.vae.config.latents_std).view(1, components.vae.config.z_dim, 1, 1, 1).to(
            block_state.latents.device, block_state.latents.dtype
        )
        block_state.latents = block_state.latents / latents_std + latents_mean
        block_state.images = components.vae.decode(block_state.latents, return_dict=False)[0][:, :, 0]
        block_state.images = components.image_processor.postprocess(block_state.images, output_type=block_state.output_type)

        self.set_block_state(state, block_state)
        return components, state
        