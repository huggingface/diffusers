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
from ...guiders.triposplat_classifier_free_guidance import TripoSplatClassifierFreeGuidance
from ...models.transformers.transformer_triposplat import TripoSplatTransformer3DModel
from ...schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from ..modular_pipeline import (
    BlockState,
    LoopSequentialPipelineBlocks,
    ModularPipeline,
    ModularPipelineBlocks,
    PipelineState,
)
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam


class TripoSplatLoopDenoiser(ModularPipelineBlocks):
    model_name = "triposplat"

    @property
    def expected_components(self):
        return [
            ComponentSpec("transformer", TripoSplatTransformer3DModel),
            ComponentSpec(
                "guider",
                TripoSplatClassifierFreeGuidance,
                config=FrozenDict({"guidance_scale": 3.0}),
                default_creation_method="from_config",
            ),
        ]

    @property
    def description(self):
        return "Predict guided Gaussian and camera velocities."

    @property
    def inputs(self):
        return [
            InputParam.template("latents", required=True),
            InputParam("camera_latents", required=True, type_hint=torch.Tensor, description="Current camera latents."),
            InputParam(
                "encoder_hidden_states",
                required=True,
                type_hint=torch.Tensor,
                description="DINOv3 image conditioning.",
            ),
            InputParam(
                "image_latents", required=True, type_hint=torch.Tensor, description="Packed VAE image conditioning."
            ),
            InputParam.template("num_inference_steps", default=20),
        ]

    @property
    def intermediate_outputs(self):
        return [
            OutputParam("noise_pred", type_hint=torch.Tensor, description="Guided Gaussian velocity."),
            OutputParam("camera_pred", type_hint=torch.Tensor, description="Guided camera velocity."),
        ]

    @torch.no_grad()
    def __call__(
        self, components: ModularPipeline, block_state: BlockState, i: int, t: torch.Tensor
    ) -> tuple[ModularPipeline, BlockState]:
        components.guider.set_state(step=i, num_inference_steps=block_state.num_inference_steps, timestep=t)
        guider_state = components.guider.prepare_inputs(
            {
                "encoder_hidden_states": (
                    block_state.encoder_hidden_states,
                    torch.zeros_like(block_state.encoder_hidden_states),
                ),
                "image_latents": (block_state.image_latents, torch.zeros_like(block_state.image_latents)),
            }
        )
        for condition in guider_state:
            components.guider.prepare_models(components.transformer)
            noise_pred, camera_pred = components.transformer(
                block_state.latents,
                t.expand(block_state.latents.shape[0]),
                condition.encoder_hidden_states.to(block_state.latents.device),
                condition.image_latents.to(block_state.latents.device),
                block_state.camera_latents,
                return_dict=False,
            )
            condition.noise_pred = torch.cat([noise_pred.flatten(1), camera_pred.flatten(1)], dim=1)
            components.guider.cleanup_models(components.transformer)
        prediction = components.guider(guider_state)[0]
        latent_size = block_state.latents[0].numel()
        block_state.noise_pred = prediction[:, :latent_size].reshape_as(block_state.latents)
        block_state.camera_pred = prediction[:, latent_size:].reshape_as(block_state.camera_latents)
        return components, block_state


class TripoSplatLoopSchedulerStep(ModularPipelineBlocks):
    model_name = "triposplat"

    @property
    def expected_components(self):
        return [ComponentSpec("scheduler", FlowMatchEulerDiscreteScheduler)]

    @property
    def description(self):
        return "Advance Gaussian and camera latents with one Euler step."

    @property
    def inputs(self):
        return [
            InputParam.template("latents", required=True),
            InputParam("camera_latents", required=True, type_hint=torch.Tensor, description="Current camera latents."),
            InputParam("noise_pred", required=True, type_hint=torch.Tensor, description="Gaussian velocity."),
            InputParam("camera_pred", required=True, type_hint=torch.Tensor, description="Camera velocity."),
        ]

    @property
    def intermediate_outputs(self):
        return [
            OutputParam.template("latents"),
            OutputParam("camera_latents", type_hint=torch.Tensor, description="Updated camera latents."),
        ]

    @torch.no_grad()
    def __call__(
        self, components: ModularPipeline, block_state: BlockState, i: int, t: torch.Tensor
    ) -> tuple[ModularPipeline, BlockState]:
        latent_size = block_state.latents[0].numel()
        prediction = torch.cat([block_state.noise_pred.flatten(1), block_state.camera_pred.flatten(1)], dim=1)
        sample = torch.cat([block_state.latents.flatten(1), block_state.camera_latents.flatten(1)], dim=1)
        sample = components.scheduler.step(prediction.float(), t, sample, return_dict=False)[0]
        block_state.latents = sample[:, :latent_size].reshape_as(block_state.latents)
        block_state.camera_latents = sample[:, latent_size:].reshape_as(block_state.camera_latents)
        return components, block_state


# auto_docstring
class TripoSplatDenoiseStep(LoopSequentialPipelineBlocks):
    """
    Denoise Gaussian and camera latents over the configured schedule.

      Components:
          transformer (`TripoSplatTransformer3DModel`) guider (`TripoSplatClassifierFreeGuidance`) scheduler
          (`FlowMatchEulerDiscreteScheduler`)

      Inputs:
          timesteps (`Tensor`):
              Euler timesteps.
          num_inference_steps (`int`, *optional*, defaults to 20):
              The number of denoising steps.
          latents (`Tensor`):
              Pre-generated noisy latents for image generation.
          camera_latents (`Tensor`):
              Current camera latents.
          encoder_hidden_states (`Tensor`):
              DINOv3 image conditioning.
          image_latents (`Tensor`):
              Packed VAE image conditioning.

      Outputs:
          latents (`Tensor`):
              Denoised latents.
          camera_latents (`Tensor`):
              Updated camera latents.
    """

    model_name = "triposplat"
    block_classes = [TripoSplatLoopDenoiser, TripoSplatLoopSchedulerStep]
    block_names = ["denoiser", "scheduler"]

    @property
    def description(self):
        return "Denoise Gaussian and camera latents over the configured schedule."

    @property
    def loop_inputs(self):
        return [
            InputParam("timesteps", required=True, type_hint=torch.Tensor, description="Euler timesteps."),
            InputParam.template("num_inference_steps", default=20),
        ]

    @torch.no_grad()
    def __call__(self, components: ModularPipeline, state: PipelineState) -> tuple[ModularPipeline, PipelineState]:
        block_state = self.get_block_state(state)
        with self.progress_bar(total=block_state.num_inference_steps) as progress_bar:
            for i, t in enumerate(block_state.timesteps):
                components, block_state = self.loop_step(components, block_state, i=i, t=t)
                progress_bar.update()
        self.set_block_state(state, block_state)
        return components, state
