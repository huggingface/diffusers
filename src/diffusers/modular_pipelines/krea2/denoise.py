# Copyright 2026 Krea AI and The HuggingFace Team. All rights reserved.
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

from ...models import Krea2Transformer2DModel
from ...schedulers import FlowMatchEulerDiscreteScheduler
from ..modular_pipeline import BlockState, LoopSequentialPipelineBlocks, ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam
from .modular_pipeline import Krea2ModularPipeline


class Krea2LoopBeforeDenoiser(ModularPipelineBlocks):
    model_name = "krea2"

    @property
    def inputs(self) -> list[InputParam]:
        return [InputParam("latents", required=True, type_hint=torch.Tensor)]

    @torch.no_grad()
    def __call__(self, components: Krea2ModularPipeline, block_state: BlockState, i: int, t: torch.Tensor):
        block_state.timestep = (
            (t / components.scheduler.config.num_train_timesteps)
            .expand(block_state.latents.shape[0])
            .to(block_state.latents.dtype)
        )
        return components, block_state


class Krea2LoopDenoiser(ModularPipelineBlocks):
    model_name = "krea2"

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("transformer", Krea2Transformer2DModel)]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("attention_kwargs"),
            InputParam.template("denoiser_input_fields"),
            InputParam("position_ids", required=True, type_hint=torch.Tensor),
            InputParam("guidance_scale", type_hint=float, default=4.5),
        ]

    @torch.no_grad()
    def __call__(self, components: Krea2ModularPipeline, block_state: BlockState, i: int, t: torch.Tensor):
        noise_pred = components.transformer(
            hidden_states=block_state.latents,
            encoder_hidden_states=block_state.prompt_embeds,
            timestep=block_state.timestep,
            position_ids=block_state.position_ids,
            encoder_attention_mask=block_state.prompt_embeds_mask,
            attention_kwargs=block_state.attention_kwargs,
            return_dict=False,
        )[0]

        if block_state.guidance_scale > 0:
            if block_state.negative_prompt_embeds is None or block_state.negative_prompt_embeds_mask is None:
                raise ValueError("Krea 2 classifier-free guidance requires negative prompt embeddings.")
            neg_noise_pred = components.transformer(
                hidden_states=block_state.latents,
                encoder_hidden_states=block_state.negative_prompt_embeds,
                timestep=block_state.timestep,
                position_ids=block_state.position_ids,
                encoder_attention_mask=block_state.negative_prompt_embeds_mask,
                attention_kwargs=block_state.attention_kwargs,
                return_dict=False,
            )[0]
            noise_pred = noise_pred + block_state.guidance_scale * (noise_pred - neg_noise_pred)

        block_state.noise_pred = noise_pred
        return components, block_state


class Krea2LoopAfterDenoiser(ModularPipelineBlocks):
    model_name = "krea2"

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("scheduler", FlowMatchEulerDiscreteScheduler)]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [OutputParam.template("latents")]

    @torch.no_grad()
    def __call__(self, components: Krea2ModularPipeline, block_state: BlockState, i: int, t: torch.Tensor):
        latents_dtype = block_state.latents.dtype
        block_state.latents = components.scheduler.step(
            block_state.noise_pred,
            t,
            block_state.latents,
            return_dict=False,
        )[0]

        if block_state.latents.dtype != latents_dtype:
            if torch.backends.mps.is_available():
                block_state.latents = block_state.latents.to(latents_dtype)

        return components, block_state


class Krea2DenoiseLoopWrapper(LoopSequentialPipelineBlocks):
    model_name = "krea2"

    @property
    def loop_expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("scheduler", FlowMatchEulerDiscreteScheduler)]

    @property
    def loop_inputs(self) -> list[InputParam]:
        return [
            InputParam("timesteps", required=True, type_hint=torch.Tensor),
            InputParam.template("num_inference_steps", required=True),
        ]

    @torch.no_grad()
    def __call__(self, components: Krea2ModularPipeline, state: PipelineState):
        block_state = self.get_block_state(state)
        block_state.num_warmup_steps = max(
            len(block_state.timesteps) - block_state.num_inference_steps * components.scheduler.order, 0
        )

        with self.progress_bar(total=block_state.num_inference_steps) as progress_bar:
            for i, t in enumerate(block_state.timesteps):
                components, block_state = self.loop_step(components, block_state, i=i, t=t)
                if i == len(block_state.timesteps) - 1 or (
                    (i + 1) > block_state.num_warmup_steps and (i + 1) % components.scheduler.order == 0
                ):
                    progress_bar.update()

        self.set_block_state(state, block_state)
        return components, state


class Krea2DenoiseStep(Krea2DenoiseLoopWrapper):
    """
    Denoise Krea 2 latents using the transformer and scheduler.
    """

    model_name = "krea2"

    block_classes = [
        Krea2LoopBeforeDenoiser,
        Krea2LoopDenoiser,
        Krea2LoopAfterDenoiser,
    ]
    block_names = ["before_denoiser", "denoiser", "after_denoiser"]
