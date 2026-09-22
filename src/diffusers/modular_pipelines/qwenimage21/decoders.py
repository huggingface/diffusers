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

import torch

from ...configuration_utils import FrozenDict
from ...image_processor import VaeImageProcessor
from ...models import AutoencoderKLQwenImage21
from ..modular_pipeline import ModularPipelineBlocks
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam


class QwenImage21DecodeStep(ModularPipelineBlocks):
    model_name = "qwenimage21"

    @property
    def description(self):
        return "Unpack and decode QwenImage21 target tokens to RGBA images."

    @property
    def expected_components(self):
        return [
            ComponentSpec("vae", AutoencoderKLQwenImage21),
            ComponentSpec(
                "image_processor",
                VaeImageProcessor,
                config=FrozenDict({"vae_scale_factor": 16}),
                default_creation_method="from_config",
            ),
        ]

    @property
    def inputs(self):
        return [
            InputParam.template("latents", required=True),
            InputParam.template("height", required=True),
            InputParam.template("width", required=True),
            InputParam.template("output_type", default="pil"),
        ]

    @property
    def intermediate_outputs(self):
        return [OutputParam.template("images")]

    @torch.no_grad()
    def __call__(self, components, state):
        block_state = self.get_block_state(state)
        latents = block_state.latents
        if block_state.output_type == "latent":
            block_state.images = latents
        else:
            latents = (
                latents.transpose(1, 2)
                .reshape(latents.shape[0], latents.shape[-1], 1, block_state.height // 16, block_state.width // 16)
                .to(components.vae.dtype)
            )
            mean = latents.new_tensor(components.vae.config.latents_mean).view(1, -1, 1, 1, 1)
            std = latents.new_tensor(components.vae.config.latents_std).view(1, -1, 1, 1, 1)
            images = components.vae.decode(latents * std + mean, return_dict=False)[0][:, :, 0]
            block_state.images = components.image_processor.postprocess(images, output_type=block_state.output_type)
        self.set_block_state(state, block_state)
        return components, state
