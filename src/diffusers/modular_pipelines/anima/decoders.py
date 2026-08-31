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

import numpy as np
import PIL
import torch

from ...configuration_utils import FrozenDict
from ...image_processor import VaeImageProcessor
from ...models import AutoencoderKLQwenImage
from ..modular_pipeline import ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam
from .modular_pipeline import AnimaModularPipeline


class AnimaVaeDecoderStep(ModularPipelineBlocks):
    model_name = "anima"

    @property
    def description(self) -> str:
        return "Step that decodes Anima latents into image tensors."

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("vae", AutoencoderKLQwenImage)]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("latents", required=True, type_hint=torch.Tensor, description="Denoised Anima latents."),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [OutputParam.template("images", note="tensor output of the VAE decoder")]

    @torch.no_grad()
    def __call__(self, components: AnimaModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        latents = block_state.latents.to(components.vae.dtype)
        latents_mean = (
            torch.tensor(components.vae.config.latents_mean)
            .view(1, components.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(components.vae.config.latents_std).view(
            1, components.vae.config.z_dim, 1, 1, 1
        ).to(latents.device, latents.dtype)
        latents = latents / latents_std + latents_mean

        block_state.images = components.vae.decode(latents, return_dict=False)[0][:, :, 0]

        self.set_block_state(state, block_state)
        return components, state


class AnimaProcessImagesOutputStep(ModularPipelineBlocks):
    model_name = "anima"

    @property
    def description(self) -> str:
        return "Postprocess decoded Anima image tensors."

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec(
                "image_processor",
                VaeImageProcessor,
                config=FrozenDict({"vae_scale_factor": 8}),
                default_creation_method="from_config",
            ),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("images", required=True, type_hint=torch.Tensor, description="Decoded Anima image tensors."),
            InputParam.template("output_type"),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "images",
                type_hint=list[PIL.Image.Image] | np.ndarray | torch.Tensor,
                description="Generated images.",
            )
        ]

    @staticmethod
    def check_inputs(output_type):
        if output_type not in ["pil", "np", "pt"]:
            raise ValueError(f"Invalid output_type: {output_type}")

    @torch.no_grad()
    def __call__(self, components: AnimaModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        self.check_inputs(block_state.output_type)

        block_state.images = components.image_processor.postprocess(
            image=block_state.images,
            output_type=block_state.output_type,
        )

        self.set_block_state(state, block_state)
        return components, state
