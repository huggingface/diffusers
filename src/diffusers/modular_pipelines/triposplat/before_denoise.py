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

from ...models.transformers.transformer_triposplat import TripoSplatTransformer3DModel
from ...schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from ...utils.torch_utils import randn_tensor
from ..modular_pipeline import ModularPipeline, ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam


class TripoSplatImageInputStep(ModularPipelineBlocks):
    model_name = "triposplat"

    @property
    def description(self):
        return "Expand per-image conditioning for the requested number of objects."

    @property
    def inputs(self):
        return [
            InputParam(
                "encoder_hidden_states", required=True, type_hint=torch.Tensor, description="DINOv3 image features."
            ),
            InputParam(
                "image_latents", required=True, type_hint=torch.Tensor, description="Packed VAE image latents."
            ),
            InputParam.template("num_images_per_prompt", default=1),
        ]

    @property
    def intermediate_outputs(self):
        return [
            OutputParam("encoder_hidden_states", type_hint=torch.Tensor, description="Expanded DINOv3 conditioning."),
            OutputParam("image_latents", type_hint=torch.Tensor, description="Expanded VAE conditioning."),
        ]

    @torch.no_grad()
    def __call__(self, components: ModularPipeline, state: PipelineState) -> tuple[ModularPipeline, PipelineState]:
        block_state = self.get_block_state(state)
        if not isinstance(block_state.num_images_per_prompt, int) or block_state.num_images_per_prompt < 1:
            raise ValueError("num_images_per_prompt must be a positive integer.")
        if block_state.encoder_hidden_states.shape[:2] != block_state.image_latents.shape[:2]:
            raise ValueError("The DINO and VAE features must have matching batch and token dimensions.")
        block_state.encoder_hidden_states = block_state.encoder_hidden_states.repeat_interleave(
            block_state.num_images_per_prompt, 0
        )
        block_state.image_latents = block_state.image_latents.repeat_interleave(block_state.num_images_per_prompt, 0)
        self.set_block_state(state, block_state)
        return components, state


class TripoSplatSetTimestepsStep(ModularPipelineBlocks):
    model_name = "triposplat"

    @property
    def expected_components(self):
        return [ComponentSpec("scheduler", FlowMatchEulerDiscreteScheduler)]

    @property
    def description(self):
        return "Configure the shifted Euler schedule."

    @property
    def inputs(self):
        return [InputParam.template("num_inference_steps", default=20)]

    @property
    def intermediate_outputs(self):
        return [OutputParam("timesteps", type_hint=torch.Tensor, description="Shifted Euler timesteps.")]

    @torch.no_grad()
    def __call__(self, components: ModularPipeline, state: PipelineState) -> tuple[ModularPipeline, PipelineState]:
        block_state = self.get_block_state(state)
        sigmas = np.linspace(1.0, 0.0, block_state.num_inference_steps + 1)[:-1]
        components.scheduler.set_timesteps(sigmas=sigmas, device=components._execution_device)
        block_state.timesteps = components.scheduler.timesteps
        self.set_block_state(state, block_state)
        return components, state


class TripoSplatPrepareLatentsStep(ModularPipelineBlocks):
    model_name = "triposplat"

    @property
    def expected_components(self):
        return [ComponentSpec("transformer", TripoSplatTransformer3DModel)]

    @property
    def description(self):
        return "Prepare Gaussian and camera noise."

    @property
    def inputs(self):
        return [
            InputParam(
                "encoder_hidden_states", required=True, type_hint=torch.Tensor, description="Expanded image features."
            ),
            InputParam.template("generator", type_hint=torch.Generator | list[torch.Generator]),
            InputParam.template("latents"),
            InputParam(
                "camera_latents", type_hint=torch.Tensor, description="Initial camera noise, with shape (batch, 1, 5)."
            ),
        ]

    @property
    def intermediate_outputs(self):
        return [
            OutputParam("latents", type_hint=torch.Tensor, description="Initial Gaussian noise."),
            OutputParam("camera_latents", type_hint=torch.Tensor, description="Initial camera noise."),
        ]

    @torch.no_grad()
    def __call__(self, components: ModularPipeline, state: PipelineState) -> tuple[ModularPipeline, PipelineState]:
        block_state = self.get_block_state(state)
        device = components._execution_device
        batch_size = block_state.encoder_hidden_states.shape[0]
        if isinstance(block_state.generator, list) and len(block_state.generator) != batch_size:
            raise ValueError("Pass one generator per generated sample.")
        shape = (batch_size, components.transformer.config.q_token_length, components.transformer.config.in_channels)
        camera_shape = (batch_size, 1, components.transformer.config.cam_channels)
        if block_state.latents is None:
            block_state.latents = randn_tensor(
                shape, generator=block_state.generator, device=device, dtype=torch.float32
            )
        elif tuple(block_state.latents.shape) != shape:
            raise ValueError(f"Expected latents with shape {shape}, got {tuple(block_state.latents.shape)}.")
        if block_state.camera_latents is None:
            block_state.camera_latents = randn_tensor(
                camera_shape, generator=block_state.generator, device=device, dtype=torch.float32
            )
        elif tuple(block_state.camera_latents.shape) != camera_shape:
            raise ValueError(
                f"Expected camera_latents with shape {camera_shape}, got {tuple(block_state.camera_latents.shape)}."
            )
        block_state.latents = block_state.latents.to(device=device, dtype=torch.float32)
        block_state.camera_latents = block_state.camera_latents.to(device=device, dtype=torch.float32)
        self.set_block_state(state, block_state)
        return components, state
