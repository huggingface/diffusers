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
from ...models import AutoencoderKLMagi
from ...video_processor import VideoProcessor
from ..modular_pipeline import ModularPipelineBlocks
from ..modular_pipeline_utils import ComponentSpec, ConfigSpec, InputParam, OutputParam


class MagiVaeDecoderStep(ModularPipelineBlocks):
    model_name = "magi"

    @property
    def description(self):
        return "Decode each generated chunk independently, then assemble the output video."

    @property
    def expected_components(self):
        return [
            ComponentSpec("vae", AutoencoderKLMagi),
            ComponentSpec(
                "video_processor",
                VideoProcessor,
                config=FrozenDict({"vae_scale_factor": 8}),
                default_creation_method="from_config",
            ),
        ]

    @property
    def expected_configs(self):
        return [ConfigSpec("latent_scaling_factor", 0.18215)]

    @property
    def inputs(self):
        return [
            InputParam.template("latents", required=True),
            InputParam("chunk_width", default=6, type_hint=int, description="Latent frames per generated chunk."),
            InputParam(
                "output_type", default="np", type_hint=str, description="Output format: pt, np, pil, or latent."
            ),
        ]

    @property
    def intermediate_outputs(self):
        return [OutputParam.template("videos")]

    @torch.no_grad()
    def __call__(self, components, state):
        block_state = self.get_block_state(state)
        if block_state.output_type == "latent":
            block_state.videos = block_state.latents
        else:
            if block_state.output_type not in ("pt", "np", "pil"):
                raise ValueError("output_type must be pt, np, pil, or latent.")
            if not isinstance(block_state.chunk_width, int) or block_state.chunk_width < 1:
                raise ValueError("chunk_width must be a positive integer.")
            if components.config.latent_scaling_factor <= 0:
                raise ValueError("latent_scaling_factor must be positive.")
            chunks = []
            for chunk in block_state.latents.split(block_state.chunk_width, dim=2):
                chunk = (chunk.float() / components.config.latent_scaling_factor).to(components.vae.dtype)
                num_frames = chunk.shape[2] * components.vae.temporal_compression_ratio
                with torch.autocast(
                    device_type=chunk.device.type,
                    dtype=chunk.dtype,
                    enabled=chunk.dtype in (torch.float16, torch.bfloat16),
                ):
                    chunks.append(components.vae.decode(chunk, num_frames=num_frames, return_dict=False)[0])
            video = torch.cat(chunks, dim=2).float()
            block_state.videos = components.video_processor.postprocess_video(
                video, output_type=block_state.output_type
            )
        self.set_block_state(state, block_state)
        return components, state


class MagiPrefixVaeDecoderStep(MagiVaeDecoderStep):
    model_name = "magi"

    @property
    def description(self):
        return "Trim prefix latents before per-chunk decoding, retaining the first four frames for a one-frame prefix."

    @property
    def inputs(self):
        return super().inputs + [
            InputParam(
                "conditioning_latents",
                required=True,
                type_hint=torch.Tensor,
                description="Scaled input prefix; determines the latent frames omitted from the output.",
            )
        ]

    @torch.no_grad()
    def __call__(self, components, state):
        block_state = self.get_block_state(state)
        prefix_length = block_state.conditioning_latents.shape[2]
        if not 0 < prefix_length < block_state.latents.shape[2]:
            raise ValueError("The prefix must leave at least one generated latent frame.")
        output_start = 0 if prefix_length == 1 else prefix_length
        if block_state.output_type == "latent":
            block_state.videos = block_state.latents[:, :, output_start:]
        else:
            if block_state.output_type not in ("pt", "np", "pil"):
                raise ValueError("output_type must be pt, np, pil, or latent.")
            if not isinstance(block_state.chunk_width, int) or block_state.chunk_width < 1:
                raise ValueError("chunk_width must be a positive integer.")
            if components.config.latent_scaling_factor <= 0:
                raise ValueError("latent_scaling_factor must be positive.")
            chunks = []
            for start in range(0, block_state.latents.shape[2], block_state.chunk_width):
                end = start + block_state.chunk_width
                if end <= output_start:
                    continue
                chunk = block_state.latents[:, :, max(start, output_start) : end]
                chunk = (chunk.float() / components.config.latent_scaling_factor).to(components.vae.dtype)
                with torch.autocast(
                    device_type=chunk.device.type,
                    dtype=chunk.dtype,
                    enabled=chunk.dtype in (torch.float16, torch.bfloat16),
                ):
                    chunks.append(components.vae.decode(chunk, return_dict=False)[0])
            video = torch.cat(chunks, dim=2).float()
            block_state.videos = components.video_processor.postprocess_video(
                video, output_type=block_state.output_type
            )
        self.set_block_state(state, block_state)
        return components, state
