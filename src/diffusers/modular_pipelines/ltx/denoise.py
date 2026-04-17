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
from ...models import LTXVideoTransformer3DModel
from ...schedulers import FlowMatchEulerDiscreteScheduler
from ..modular_pipeline import (
    BlockState,
    LoopSequentialPipelineBlocks,
    ModularPipelineBlocks,
    PipelineState,
)
from ..modular_pipeline_utils import ComponentSpec, InputParam
from .modular_pipeline import LTXModularPipeline, LTXVideoPachifier


class LTXLoopBeforeDenoiser(ModularPipelineBlocks):
    model_name = "ltx"

    @property
    def description(self) -> str:
        return (
            "Step within the denoising loop that prepares the latent input for the denoiser. "
            "This block should be used to compose the `sub_blocks` attribute of a `LoopSequentialPipelineBlocks` "
            "object (e.g. `LTXDenoiseLoopWrapper`)"
        )

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("latents", required=True),
            InputParam.template("dtype", required=True),
        ]

    @torch.no_grad()
    def __call__(self, components: LTXModularPipeline, block_state: BlockState, i: int, t: torch.Tensor):
        block_state.latent_model_input = block_state.latents.to(block_state.dtype)
        return components, block_state


class LTXLoopDenoiser(ModularPipelineBlocks):
    model_name = "ltx"

    def __init__(
        self,
        guider_input_fields: dict[str, Any] | None = None,
    ):
        if guider_input_fields is None:
            guider_input_fields = {
                "encoder_hidden_states": ("prompt_embeds", "negative_prompt_embeds"),
                "encoder_attention_mask": ("prompt_attention_mask", "negative_prompt_attention_mask"),
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
                config=FrozenDict({"guidance_scale": 3.0}),
                default_creation_method="from_config",
            ),
            ComponentSpec("transformer", LTXVideoTransformer3DModel),
        ]

    @property
    def description(self) -> str:
        return (
            "Step within the denoising loop that denoises the latents with guidance. "
            "This block should be used to compose the `sub_blocks` attribute of a `LoopSequentialPipelineBlocks` "
            "object (e.g. `LTXDenoiseLoopWrapper`)"
        )

    @property
    def inputs(self) -> list[tuple[str, Any]]:
        inputs = [
            InputParam.template("attention_kwargs"),
            InputParam.template("num_inference_steps", required=True),
            InputParam("rope_interpolation_scale", type_hint=tuple),
            InputParam.template("height"),
            InputParam.template("width"),
            InputParam("num_frames", type_hint=int),
        ]
        guider_input_names = []
        for value in self._guider_input_fields.values():
            if isinstance(value, tuple):
                guider_input_names.extend(value)
            else:
                guider_input_names.append(value)

        for name in guider_input_names:
            inputs.append(InputParam(name=name, required=True, type_hint=torch.Tensor))
        return inputs

    @torch.no_grad()
    def __call__(
        self, components: LTXModularPipeline, block_state: BlockState, i: int, t: torch.Tensor
    ) -> PipelineState:
        components.guider.set_state(step=i, num_inference_steps=block_state.num_inference_steps, timestep=t)

        latent_num_frames = (block_state.num_frames - 1) // components.vae_temporal_compression_ratio + 1
        latent_height = block_state.height // components.vae_spatial_compression_ratio
        latent_width = block_state.width // components.vae_spatial_compression_ratio

        guider_state = components.guider.prepare_inputs_from_block_state(block_state, self._guider_input_fields)

        for guider_state_batch in guider_state:
            components.guider.prepare_models(components.transformer)
            cond_kwargs = guider_state_batch.as_dict()
            cond_kwargs = {
                k: v.to(block_state.dtype) if isinstance(v, torch.Tensor) else v
                for k, v in cond_kwargs.items()
                if k in self._guider_input_fields.keys()
            }

            context_name = getattr(guider_state_batch, components.guider._identifier_key, None)
            with components.transformer.cache_context(context_name):
                guider_state_batch.noise_pred = components.transformer(
                    hidden_states=block_state.latent_model_input,
                    timestep=t.expand(block_state.latent_model_input.shape[0]).to(block_state.dtype),
                    num_frames=latent_num_frames,
                    height=latent_height,
                    width=latent_width,
                    rope_interpolation_scale=block_state.rope_interpolation_scale,
                    attention_kwargs=block_state.attention_kwargs,
                    return_dict=False,
                    **cond_kwargs,
                )[0]
            components.guider.cleanup_models(components.transformer)

        block_state.noise_pred = components.guider(guider_state)[0]

        return components, block_state


class LTXLoopAfterDenoiser(ModularPipelineBlocks):
    model_name = "ltx"

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("scheduler", FlowMatchEulerDiscreteScheduler),
        ]

    @property
    def description(self) -> str:
        return (
            "Step within the denoising loop that updates the latents. "
            "This block should be used to compose the `sub_blocks` attribute of a `LoopSequentialPipelineBlocks` "
            "object (e.g. `LTXDenoiseLoopWrapper`)"
        )

    @torch.no_grad()
    def __call__(self, components: LTXModularPipeline, block_state: BlockState, i: int, t: torch.Tensor):
        latents_dtype = block_state.latents.dtype
        block_state.latents = components.scheduler.step(
            block_state.noise_pred,
            t,
            block_state.latents,
            return_dict=False,
        )[0]

        if block_state.latents.dtype != latents_dtype:
            block_state.latents = block_state.latents.to(latents_dtype)

        return components, block_state


class LTXDenoiseLoopWrapper(LoopSequentialPipelineBlocks):
    model_name = "ltx"

    @property
    def description(self) -> str:
        return (
            "Pipeline block that iteratively denoises the latents over `timesteps`. "
            "The specific steps within each iteration can be customized with `sub_blocks` attributes"
        )

    @property
    def loop_expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("scheduler", FlowMatchEulerDiscreteScheduler),
            ComponentSpec("transformer", LTXVideoTransformer3DModel),
        ]

    @property
    def loop_inputs(self) -> list[InputParam]:
        return [
            InputParam.template("timesteps", required=True),
            InputParam.template("num_inference_steps", required=True),
        ]

    @torch.no_grad()
    def __call__(self, components: LTXModularPipeline, state: PipelineState) -> PipelineState:
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


class LTXDenoiseStep(LTXDenoiseLoopWrapper):
    block_classes = [
        LTXLoopBeforeDenoiser,
        LTXLoopDenoiser(
            guider_input_fields={
                "encoder_hidden_states": ("prompt_embeds", "negative_prompt_embeds"),
                "encoder_attention_mask": ("prompt_attention_mask", "negative_prompt_attention_mask"),
            }
        ),
        LTXLoopAfterDenoiser,
    ]
    block_names = ["before_denoiser", "denoiser", "after_denoiser"]

    @property
    def description(self) -> str:
        return (
            "Denoise step that iteratively denoises the latents.\n"
            "Its loop logic is defined in `LTXDenoiseLoopWrapper.__call__` method.\n"
            "At each iteration, it runs blocks defined in `sub_blocks` sequentially:\n"
            " - `LTXLoopBeforeDenoiser`\n"
            " - `LTXLoopDenoiser`\n"
            " - `LTXLoopAfterDenoiser`\n"
            "This block supports text-to-video tasks."
        )


class LTXImage2VideoLoopBeforeDenoiser(ModularPipelineBlocks):
    model_name = "ltx"

    @property
    def description(self) -> str:
        return (
            "Step within the i2v denoising loop that prepares the latent input and modulates "
            "the timestep with the conditioning mask."
        )

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("latents", required=True),
            InputParam("conditioning_mask", required=True, type_hint=torch.Tensor),
            InputParam.template("dtype", required=True),
        ]

    @torch.no_grad()
    def __call__(self, components: LTXModularPipeline, block_state: BlockState, i: int, t: torch.Tensor):
        block_state.latent_model_input = block_state.latents.to(block_state.dtype)
        block_state.timestep_adjusted = t.expand(block_state.latent_model_input.shape[0]).unsqueeze(-1) * (
            1 - block_state.conditioning_mask
        )
        return components, block_state


class LTXImage2VideoLoopDenoiser(ModularPipelineBlocks):
    model_name = "ltx"

    def __init__(
        self,
        guider_input_fields: dict[str, Any] | None = None,
    ):
        if guider_input_fields is None:
            guider_input_fields = {
                "encoder_hidden_states": ("prompt_embeds", "negative_prompt_embeds"),
                "encoder_attention_mask": ("prompt_attention_mask", "negative_prompt_attention_mask"),
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
                config=FrozenDict({"guidance_scale": 3.0}),
                default_creation_method="from_config",
            ),
            ComponentSpec("transformer", LTXVideoTransformer3DModel),
        ]

    @property
    def description(self) -> str:
        return (
            "Step within the i2v denoising loop that denoises the latents with guidance "
            "using timestep modulated by the conditioning mask."
        )

    @property
    def inputs(self) -> list[tuple[str, Any]]:
        inputs = [
            InputParam.template("attention_kwargs"),
            InputParam.template("num_inference_steps", required=True),
            InputParam("rope_interpolation_scale", type_hint=tuple),
            InputParam.template("height"),
            InputParam.template("width"),
            InputParam("num_frames", type_hint=int),
        ]
        guider_input_names = []
        for value in self._guider_input_fields.values():
            if isinstance(value, tuple):
                guider_input_names.extend(value)
            else:
                guider_input_names.append(value)
        for name in guider_input_names:
            inputs.append(InputParam(name=name, required=True, type_hint=torch.Tensor))
        return inputs

    @torch.no_grad()
    def __call__(
        self, components: LTXModularPipeline, block_state: BlockState, i: int, t: torch.Tensor
    ) -> PipelineState:
        components.guider.set_state(step=i, num_inference_steps=block_state.num_inference_steps, timestep=t)

        latent_num_frames = (block_state.num_frames - 1) // components.vae_temporal_compression_ratio + 1
        latent_height = block_state.height // components.vae_spatial_compression_ratio
        latent_width = block_state.width // components.vae_spatial_compression_ratio

        guider_state = components.guider.prepare_inputs_from_block_state(block_state, self._guider_input_fields)

        for guider_state_batch in guider_state:
            components.guider.prepare_models(components.transformer)
            cond_kwargs = guider_state_batch.as_dict()
            cond_kwargs = {
                k: v.to(block_state.dtype) if isinstance(v, torch.Tensor) else v
                for k, v in cond_kwargs.items()
                if k in self._guider_input_fields.keys()
            }

            context_name = getattr(guider_state_batch, components.guider._identifier_key, None)
            with components.transformer.cache_context(context_name):
                guider_state_batch.noise_pred = components.transformer(
                    hidden_states=block_state.latent_model_input,
                    timestep=block_state.timestep_adjusted,
                    num_frames=latent_num_frames,
                    height=latent_height,
                    width=latent_width,
                    rope_interpolation_scale=block_state.rope_interpolation_scale,
                    attention_kwargs=block_state.attention_kwargs,
                    return_dict=False,
                    **cond_kwargs,
                )[0]
            components.guider.cleanup_models(components.transformer)

        block_state.noise_pred = components.guider(guider_state)[0]

        return components, block_state


class LTXImage2VideoLoopAfterDenoiser(ModularPipelineBlocks):
    model_name = "ltx"

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("scheduler", FlowMatchEulerDiscreteScheduler),
            ComponentSpec(
                "pachifier",
                LTXVideoPachifier,
                config=FrozenDict({"patch_size": 1, "patch_size_t": 1}),
                default_creation_method="from_config",
            ),
        ]

    @property
    def description(self) -> str:
        return (
            "Step within the i2v denoising loop that updates the latents, "
            "applying the scheduler step only to frames after the first (conditioned) frame."
        )

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("height"),
            InputParam.template("width"),
            InputParam("num_frames", type_hint=int),
        ]

    @torch.no_grad()
    def __call__(self, components: LTXModularPipeline, block_state: BlockState, i: int, t: torch.Tensor):
        latent_num_frames = (block_state.num_frames - 1) // components.vae_temporal_compression_ratio + 1
        latent_height = block_state.height // components.vae_spatial_compression_ratio
        latent_width = block_state.width // components.vae_spatial_compression_ratio

        noise_pred = components.pachifier.unpack_latents(
            block_state.noise_pred, latent_num_frames, latent_height, latent_width
        )
        latents = components.pachifier.unpack_latents(
            block_state.latents, latent_num_frames, latent_height, latent_width
        )

        noise_pred = noise_pred[:, :, 1:]
        noise_latents = latents[:, :, 1:]
        pred_latents = components.scheduler.step(noise_pred, t, noise_latents, return_dict=False)[0]

        latents = torch.cat([latents[:, :, :1], pred_latents], dim=2)
        block_state.latents = components.pachifier.pack_latents(latents)

        return components, block_state


class LTXImage2VideoDenoiseStep(LTXDenoiseLoopWrapper):
    block_classes = [
        LTXImage2VideoLoopBeforeDenoiser,
        LTXImage2VideoLoopDenoiser(
            guider_input_fields={
                "encoder_hidden_states": ("prompt_embeds", "negative_prompt_embeds"),
                "encoder_attention_mask": ("prompt_attention_mask", "negative_prompt_attention_mask"),
            }
        ),
        LTXImage2VideoLoopAfterDenoiser,
    ]
    block_names = ["before_denoiser", "denoiser", "after_denoiser"]

    @property
    def description(self) -> str:
        return (
            "Denoise step for image-to-video that iteratively denoises the latents.\n"
            "The first frame is kept fixed via a conditioning mask.\n"
            "At each iteration, it runs blocks defined in `sub_blocks` sequentially:\n"
            " - `LTXImage2VideoLoopBeforeDenoiser`\n"
            " - `LTXImage2VideoLoopDenoiser`\n"
            " - `LTXImage2VideoLoopAfterDenoiser`"
        )
