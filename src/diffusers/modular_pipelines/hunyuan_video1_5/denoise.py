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

from typing import Any

import torch

from ...configuration_utils import FrozenDict
from ...guiders import ClassifierFreeGuidance
from ...models import HunyuanVideo15Transformer3DModel
from ...schedulers import FlowMatchEulerDiscreteScheduler
from ...utils import logging
from ..modular_pipeline import (
    BlockState,
    LoopSequentialPipelineBlocks,
    ModularPipelineBlocks,
    PipelineState,
)
from ..modular_pipeline_utils import ComponentSpec, InputParam
from .modular_pipeline import HunyuanVideo15ModularPipeline


logger = logging.get_logger(__name__)


class HunyuanVideo15LoopBeforeDenoiser(ModularPipelineBlocks):
    model_name = "hunyuan-video-1.5"

    @property
    def description(self) -> str:
        return "Step within the denoising loop that prepares the latent input"

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("latents", required=True, type_hint=torch.Tensor),
            InputParam("cond_latents_concat", required=True, type_hint=torch.Tensor),
            InputParam("mask_concat", required=True, type_hint=torch.Tensor),
            InputParam("dtype", required=True, type_hint=torch.dtype),
        ]

    @torch.no_grad()
    def __call__(self, components: HunyuanVideo15ModularPipeline, block_state: BlockState, i: int, t: torch.Tensor):
        block_state.latent_model_input = torch.cat(
            [block_state.latents, block_state.cond_latents_concat, block_state.mask_concat], dim=1
        )
        return components, block_state


class HunyuanVideo15LoopDenoiser(ModularPipelineBlocks):
    model_name = "hunyuan-video-1.5"

    def __init__(self, guider_input_fields=None):
        if guider_input_fields is None:
            guider_input_fields = {
                "encoder_hidden_states": ("prompt_embeds", "negative_prompt_embeds"),
                "encoder_attention_mask": ("prompt_embeds_mask", "negative_prompt_embeds_mask"),
                "encoder_hidden_states_2": ("prompt_embeds_2", "negative_prompt_embeds_2"),
                "encoder_attention_mask_2": ("prompt_embeds_mask_2", "negative_prompt_embeds_mask_2"),
            }
        if not isinstance(guider_input_fields, dict):
            raise ValueError(f"guider_input_fields must be a dictionary but is {type(guider_input_fields)}")
        self._guider_input_fields = guider_input_fields
        super().__init__()

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec(
                "guider",
                ClassifierFreeGuidance,
                config=FrozenDict({"guidance_scale": 1.0}),
                default_creation_method="from_config",
            ),
            ComponentSpec("transformer", HunyuanVideo15Transformer3DModel),
        ]

    @property
    def description(self) -> str:
        return "Step within the denoising loop that denoises the latents with guidance"

    @property
    def inputs(self) -> list[InputParam]:
        inputs = [
            InputParam("attention_kwargs"),
            InputParam("num_inference_steps", required=True, type_hint=int),
            InputParam("image_embeds", type_hint=torch.Tensor),
        ]
        for value in self._guider_input_fields.values():
            if isinstance(value, tuple):
                inputs.append(InputParam(name=value[0], required=True, type_hint=torch.Tensor))
                for neg_name in value[1:]:
                    inputs.append(InputParam(name=neg_name, type_hint=torch.Tensor))
            else:
                inputs.append(InputParam(name=value, required=True, type_hint=torch.Tensor))
        return inputs

    @torch.no_grad()
    def __call__(
        self, components: HunyuanVideo15ModularPipeline, block_state: BlockState, i: int, t: torch.Tensor
    ) -> PipelineState:
        components.guider.set_state(step=i, num_inference_steps=block_state.num_inference_steps, timestep=t)
        guider_state = components.guider.prepare_inputs_from_block_state(block_state, self._guider_input_fields)

        timestep = t.expand(block_state.latent_model_input.shape[0]).to(block_state.latent_model_input.dtype)

        for guider_state_batch in guider_state:
            components.guider.prepare_models(components.transformer)
            cond_kwargs = guider_state_batch.as_dict()
            cond_kwargs = {
                k: v for k, v in cond_kwargs.items() if k in self._guider_input_fields.keys()
            }

            context_name = getattr(guider_state_batch, components.guider._identifier_key, None)
            with components.transformer.cache_context(context_name):
                guider_state_batch.noise_pred = components.transformer(
                    hidden_states=block_state.latent_model_input,
                    image_embeds=block_state.image_embeds,
                    timestep=timestep,
                    attention_kwargs=block_state.attention_kwargs,
                    return_dict=False,
                    **cond_kwargs,
                )[0]
            components.guider.cleanup_models(components.transformer)

        block_state.noise_pred = components.guider(guider_state)[0]

        return components, block_state


class HunyuanVideo15LoopAfterDenoiser(ModularPipelineBlocks):
    model_name = "hunyuan-video-1.5"

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("scheduler", FlowMatchEulerDiscreteScheduler),
        ]

    @property
    def description(self) -> str:
        return "Step within the denoising loop that updates the latents"

    @torch.no_grad()
    def __call__(self, components: HunyuanVideo15ModularPipeline, block_state: BlockState, i: int, t: torch.Tensor):
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


class HunyuanVideo15DenoiseLoopWrapper(LoopSequentialPipelineBlocks):
    model_name = "hunyuan-video-1.5"

    @property
    def description(self) -> str:
        return "Pipeline block that iteratively denoises the latents over timesteps"

    @property
    def loop_expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("scheduler", FlowMatchEulerDiscreteScheduler),
            ComponentSpec("transformer", HunyuanVideo15Transformer3DModel),
        ]

    @property
    def loop_inputs(self) -> list[InputParam]:
        return [
            InputParam("timesteps", required=True, type_hint=torch.Tensor),
            InputParam("num_inference_steps", required=True, type_hint=int),
        ]

    @torch.no_grad()
    def __call__(self, components: HunyuanVideo15ModularPipeline, state: PipelineState) -> PipelineState:
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


class HunyuanVideo15DenoiseStep(HunyuanVideo15DenoiseLoopWrapper):
    block_classes = [
        HunyuanVideo15LoopBeforeDenoiser,
        HunyuanVideo15LoopDenoiser(),
        HunyuanVideo15LoopAfterDenoiser,
    ]
    block_names = ["before_denoiser", "denoiser", "after_denoiser"]

    @property
    def description(self) -> str:
        return (
            "Denoise step that iteratively denoises the latents.\n"
            "At each iteration, it runs blocks defined in `sub_blocks` sequentially:\n"
            " - `HunyuanVideo15LoopBeforeDenoiser`\n"
            " - `HunyuanVideo15LoopDenoiser`\n"
            " - `HunyuanVideo15LoopAfterDenoiser`\n"
            "This block supports text-to-video tasks."
        )


class HunyuanVideo15Image2VideoLoopDenoiser(ModularPipelineBlocks):
    model_name = "hunyuan-video-1.5"

    def __init__(self, guider_input_fields=None):
        if guider_input_fields is None:
            guider_input_fields = {
                "encoder_hidden_states": ("prompt_embeds", "negative_prompt_embeds"),
                "encoder_attention_mask": ("prompt_embeds_mask", "negative_prompt_embeds_mask"),
                "encoder_hidden_states_2": ("prompt_embeds_2", "negative_prompt_embeds_2"),
                "encoder_attention_mask_2": ("prompt_embeds_mask_2", "negative_prompt_embeds_mask_2"),
            }
        if not isinstance(guider_input_fields, dict):
            raise ValueError(f"guider_input_fields must be a dictionary but is {type(guider_input_fields)}")
        self._guider_input_fields = guider_input_fields
        super().__init__()

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec(
                "guider",
                ClassifierFreeGuidance,
                config=FrozenDict({"guidance_scale": 1.0}),
                default_creation_method="from_config",
            ),
            ComponentSpec("transformer", HunyuanVideo15Transformer3DModel),
        ]

    @property
    def description(self) -> str:
        return "I2V denoiser with MeanFlow timestep_r support"

    @property
    def inputs(self) -> list[InputParam]:
        inputs = [
            InputParam("attention_kwargs"),
            InputParam("num_inference_steps", required=True, type_hint=int),
            InputParam("image_embeds", type_hint=torch.Tensor),
            InputParam("timesteps", required=True, type_hint=torch.Tensor),
        ]
        for value in self._guider_input_fields.values():
            if isinstance(value, tuple):
                inputs.append(InputParam(name=value[0], required=True, type_hint=torch.Tensor))
                for neg_name in value[1:]:
                    inputs.append(InputParam(name=neg_name, type_hint=torch.Tensor))
            else:
                inputs.append(InputParam(name=value, required=True, type_hint=torch.Tensor))
        return inputs

    @torch.no_grad()
    def __call__(
        self, components: HunyuanVideo15ModularPipeline, block_state: BlockState, i: int, t: torch.Tensor
    ) -> PipelineState:
        components.guider.set_state(step=i, num_inference_steps=block_state.num_inference_steps, timestep=t)
        guider_state = components.guider.prepare_inputs_from_block_state(block_state, self._guider_input_fields)

        timestep = t.expand(block_state.latent_model_input.shape[0]).to(block_state.latent_model_input.dtype)

        # MeanFlow: compute timestep_r
        timestep_r = None
        if components.transformer.config.use_meanflow:
            if i == len(block_state.timesteps) - 1:
                timestep_r = torch.tensor([0.0], device=timestep.device)
            else:
                timestep_r = block_state.timesteps[i + 1]
            timestep_r = timestep_r.expand(block_state.latents.shape[0]).to(block_state.latents.dtype)

        for guider_state_batch in guider_state:
            components.guider.prepare_models(components.transformer)
            cond_kwargs = guider_state_batch.as_dict()
            cond_kwargs = {
                k: v for k, v in cond_kwargs.items() if k in self._guider_input_fields.keys()
            }

            context_name = getattr(guider_state_batch, components.guider._identifier_key, None)
            with components.transformer.cache_context(context_name):
                guider_state_batch.noise_pred = components.transformer(
                    hidden_states=block_state.latent_model_input,
                    image_embeds=block_state.image_embeds,
                    timestep=timestep,
                    timestep_r=timestep_r,
                    attention_kwargs=block_state.attention_kwargs,
                    return_dict=False,
                    **cond_kwargs,
                )[0]
            components.guider.cleanup_models(components.transformer)

        block_state.noise_pred = components.guider(guider_state)[0]

        return components, block_state


class HunyuanVideo15Image2VideoDenoiseStep(HunyuanVideo15DenoiseLoopWrapper):
    block_classes = [
        HunyuanVideo15LoopBeforeDenoiser,
        HunyuanVideo15Image2VideoLoopDenoiser(),
        HunyuanVideo15LoopAfterDenoiser,
    ]
    block_names = ["before_denoiser", "denoiser", "after_denoiser"]

    @property
    def description(self) -> str:
        return (
            "Denoise step for image-to-video with MeanFlow support.\n"
            "At each iteration, it runs blocks defined in `sub_blocks` sequentially:\n"
            " - `HunyuanVideo15LoopBeforeDenoiser`\n"
            " - `HunyuanVideo15Image2VideoLoopDenoiser`\n"
            " - `HunyuanVideo15LoopAfterDenoiser`"
        )
