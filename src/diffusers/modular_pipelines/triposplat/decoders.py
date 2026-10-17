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
import torch

from ...models.autoencoders.autoencoder_triposplat import TripoSplatGaussianDecoder
from ..modular_pipeline import ModularPipeline, ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam


class TripoSplatGaussianDecodeStep(ModularPipelineBlocks):
    model_name = "triposplat"

    @property
    def expected_components(self):
        return [ComponentSpec("decoder", TripoSplatGaussianDecoder)]

    @property
    def description(self):
        return "Decode denoised latents into Gaussian splats at the requested densities."

    @property
    def inputs(self):
        return [
            InputParam.template("latents", required=True),
            InputParam.template("generator", type_hint=torch.Generator | list[torch.Generator]),
            InputParam(
                "num_gaussians",
                default=262144,
                type_hint=int | list[int],
                description="Gaussian counts between 32768 and 262144.",
            ),
            InputParam(
                "decoder_generator",
                type_hint=torch.Generator | list,
                description="Random generator or per-sample generators for octree sampling.",
            ),
            InputParam("output_type", default="pt", type_hint=str, description="Output format: 'pt' or 'np'."),
        ]

    @property
    def intermediate_outputs(self):
        return [
            OutputParam(
                "gaussians",
                type_hint=torch.Tensor | np.ndarray | list,
                description="Gaussian parameters in xyz, SH color, scale, wxyz rotation, opacity order.",
            )
        ]

    @torch.no_grad()
    def __call__(self, components: ModularPipeline, state: PipelineState) -> tuple[ModularPipeline, PipelineState]:
        block_state = self.get_block_state(state)
        if block_state.output_type not in ("pt", "np"):
            raise ValueError("output_type must be 'pt' or 'np'; remove the decode block to obtain latents.")
        counts = (
            block_state.num_gaussians if isinstance(block_state.num_gaussians, list) else [block_state.num_gaussians]
        )
        if not counts or any(not isinstance(count, int) or not 32768 <= count <= 262144 for count in counts):
            raise ValueError(
                "num_gaussians must be an integer or a nonempty list of integers between 32768 and 262144."
            )
        gaussians_per_point = components.decoder.config.gaussians_per_point
        counts = [round(count / gaussians_per_point) * gaussians_per_point for count in counts]
        generator = block_state.generator if block_state.decoder_generator is None else block_state.decoder_generator
        gaussians = [components.decoder(block_state.latents, count, generator=generator).sample for count in counts]
        if block_state.output_type == "np":
            gaussians = [item.cpu().numpy() for item in gaussians]
        block_state.gaussians = gaussians if isinstance(block_state.num_gaussians, list) else gaussians[0]
        self.set_block_state(state, block_state)
        return components, state
