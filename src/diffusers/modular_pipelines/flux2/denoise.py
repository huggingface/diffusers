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
from ...models import Flux2Transformer2DModel
from ...schedulers import FlowMatchEulerDiscreteScheduler
from ...utils import is_torch_xla_available, logging
from ..modular_pipeline import (
    BlockState,
    LoopSequentialPipelineBlocks,
    ModularPipelineBlocks,
    PipelineState,
)
from ..modular_pipeline_utils import ComponentSpec, ConfigSpec, InputParam, OutputParam
from .modular_pipeline import Flux2KleinModularPipeline, Flux2ModularPipeline


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class Flux2LoopDenoiser(ModularPipelineBlocks):
    model_name = "flux2"

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("transformer", Flux2Transformer2DModel)]

    @property
    def description(self) -> str:
        return (
            "Step within the denoising loop that denoises the latents for Flux2. "
            "This block should be used to compose the `sub_blocks` attribute of a `LoopSequentialPipelineBlocks` "
            "object (e.g. `Flux2DenoiseLoopWrapper`)"
        )

    @property
    def inputs(self) -> list[tuple[str, Any]]:
        return [
            InputParam("joint_attention_kwargs"),
            InputParam(
                "latents",
                required=True,
                type_hint=torch.Tensor,
                description="The latents to denoise. Shape: (B, seq_len, C)",
            ),
            InputParam(
                "image_latents",
                type_hint=torch.Tensor,
                description="Packed image latents for conditioning. Shape: (B, img_seq_len, C)",
            ),
            InputParam(
                "image_latent_ids",
                type_hint=torch.Tensor,
                description="Position IDs for image latents. Shape: (B, img_seq_len, 4)",
            ),
            InputParam(
                "guidance",
                required=True,
                type_hint=torch.Tensor,
                description="Guidance scale as a tensor",
            ),
            InputParam(
                "prompt_embeds",
                required=True,
                type_hint=torch.Tensor,
                description="Text embeddings from Mistral3",
            ),
            InputParam(
                "txt_ids",
                required=True,
                type_hint=torch.Tensor,
                description="4D position IDs for text tokens (T, H, W, L)",
            ),
            InputParam(
                "latent_ids",
                required=True,
                type_hint=torch.Tensor,
                description="4D position IDs for latent tokens (T, H, W, L)",
            ),
        ]

    @torch.no_grad()
    def __call__(
        self, components: Flux2ModularPipeline, block_state: BlockState, i: int, t: torch.Tensor
    ) -> PipelineState:
        latents = block_state.latents
        latent_model_input = latents.to(components.transformer.dtype)
        img_ids = block_state.latent_ids

        image_latents = getattr(block_state, "image_latents", None)
        if image_latents is not None:
            latent_model_input = torch.cat([latents, image_latents], dim=1).to(components.transformer.dtype)
            image_latent_ids = block_state.image_latent_ids
            img_ids = torch.cat([img_ids, image_latent_ids], dim=1)

        timestep = t.expand(latents.shape[0]).to(latents.dtype)

        noise_pred = components.transformer(
            hidden_states=latent_model_input,
            timestep=timestep / 1000,
            guidance=block_state.guidance,
            encoder_hidden_states=block_state.prompt_embeds,
            txt_ids=block_state.txt_ids,
            img_ids=img_ids,
            joint_attention_kwargs=block_state.joint_attention_kwargs,
            return_dict=False,
        )[0]

        noise_pred = noise_pred[:, : latents.size(1)]
        block_state.noise_pred = noise_pred

        return components, block_state


# same as Flux2LoopDenoiser but guidance=None
class Flux2KleinLoopDenoiser(ModularPipelineBlocks):
    model_name = "flux2-klein"

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("transformer", Flux2Transformer2DModel)]

    @property
    def description(self) -> str:
        return (
            "Step within the denoising loop that denoises the latents for Flux2. "
            "This block should be used to compose the `sub_blocks` attribute of a `LoopSequentialPipelineBlocks` "
            "object (e.g. `Flux2DenoiseLoopWrapper`)"
        )

    @property
    def inputs(self) -> list[tuple[str, Any]]:
        return [
            InputParam("joint_attention_kwargs"),
            InputParam(
                "latents",
                required=True,
                type_hint=torch.Tensor,
                description="The latents to denoise. Shape: (B, seq_len, C)",
            ),
            InputParam(
                "image_latents",
                type_hint=torch.Tensor,
                description="Packed image latents for conditioning. Shape: (B, img_seq_len, C)",
            ),
            InputParam(
                "image_latent_ids",
                type_hint=torch.Tensor,
                description="Position IDs for image latents. Shape: (B, img_seq_len, 4)",
            ),
            InputParam(
                "prompt_embeds",
                required=True,
                type_hint=torch.Tensor,
                description="Text embeddings from Qwen3",
            ),
            InputParam(
                "txt_ids",
                required=True,
                type_hint=torch.Tensor,
                description="4D position IDs for text tokens (T, H, W, L)",
            ),
            InputParam(
                "latent_ids",
                required=True,
                type_hint=torch.Tensor,
                description="4D position IDs for latent tokens (T, H, W, L)",
            ),
        ]

    @torch.no_grad()
    def __call__(
        self, components: Flux2KleinModularPipeline, block_state: BlockState, i: int, t: torch.Tensor
    ) -> PipelineState:
        latents = block_state.latents
        latent_model_input = latents.to(components.transformer.dtype)
        img_ids = block_state.latent_ids

        image_latents = getattr(block_state, "image_latents", None)
        if image_latents is not None:
            latent_model_input = torch.cat([latents, image_latents], dim=1).to(components.transformer.dtype)
            image_latent_ids = block_state.image_latent_ids
            img_ids = torch.cat([img_ids, image_latent_ids], dim=1)

        timestep = t.expand(latents.shape[0]).to(latents.dtype)

        noise_pred = components.transformer(
            hidden_states=latent_model_input,
            timestep=timestep / 1000,
            guidance=None,
            encoder_hidden_states=block_state.prompt_embeds,
            txt_ids=block_state.txt_ids,
            img_ids=img_ids,
            joint_attention_kwargs=block_state.joint_attention_kwargs,
            return_dict=False,
        )[0]

        noise_pred = noise_pred[:, : latents.size(1)]
        block_state.noise_pred = noise_pred

        return components, block_state


# support CFG for Flux2-Klein base model
class Flux2KleinBaseLoopDenoiser(ModularPipelineBlocks):
    model_name = "flux2-klein"

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("transformer", Flux2Transformer2DModel),
            ComponentSpec(
                "guider",
                ClassifierFreeGuidance,
                config=FrozenDict({"guidance_scale": 4.0}),
                default_creation_method="from_config",
            ),
        ]

    @property
    def expected_configs(self) -> list[ConfigSpec]:
        return [
            ConfigSpec(name="is_distilled", default=False),
        ]

    @property
    def description(self) -> str:
        return (
            "Step within the denoising loop that denoises the latents for Flux2. "
            "This block should be used to compose the `sub_blocks` attribute of a `LoopSequentialPipelineBlocks` "
            "object (e.g. `Flux2DenoiseLoopWrapper`)"
        )

    @property
    def inputs(self) -> list[tuple[str, Any]]:
        return [
            InputParam("joint_attention_kwargs"),
            InputParam(
                "latents",
                required=True,
                type_hint=torch.Tensor,
                description="The latents to denoise. Shape: (B, seq_len, C)",
            ),
            InputParam(
                "image_latents",
                type_hint=torch.Tensor,
                description="Packed image latents for conditioning. Shape: (B, img_seq_len, C)",
            ),
            InputParam(
                "image_latent_ids",
                type_hint=torch.Tensor,
                description="Position IDs for image latents. Shape: (B, img_seq_len, 4)",
            ),
            InputParam(
                "prompt_embeds",
                required=True,
                type_hint=torch.Tensor,
                description="Text embeddings from Qwen3",
            ),
            InputParam(
                "negative_prompt_embeds",
                required=False,
                type_hint=torch.Tensor,
                description="Negative text embeddings from Qwen3",
            ),
            InputParam(
                "txt_ids",
                required=True,
                type_hint=torch.Tensor,
                description="4D position IDs for text tokens (T, H, W, L)",
            ),
            InputParam(
                "negative_txt_ids",
                required=False,
                type_hint=torch.Tensor,
                description="4D position IDs for negative text tokens (T, H, W, L)",
            ),
            InputParam(
                "latent_ids",
                required=True,
                type_hint=torch.Tensor,
                description="4D position IDs for latent tokens (T, H, W, L)",
            ),
        ]

    @torch.no_grad()
    def __call__(
        self, components: Flux2KleinModularPipeline, block_state: BlockState, i: int, t: torch.Tensor
    ) -> PipelineState:
        latents = block_state.latents
        latent_model_input = latents.to(components.transformer.dtype)
        img_ids = block_state.latent_ids

        image_latents = getattr(block_state, "image_latents", None)
        if image_latents is not None:
            latent_model_input = torch.cat([latents, image_latents], dim=1).to(components.transformer.dtype)
            image_latent_ids = block_state.image_latent_ids
            img_ids = torch.cat([img_ids, image_latent_ids], dim=1)

        timestep = t.expand(latents.shape[0]).to(latents.dtype)

        guider_inputs = {
            "encoder_hidden_states": (
                getattr(block_state, "prompt_embeds", None),
                getattr(block_state, "negative_prompt_embeds", None),
            ),
            "txt_ids": (
                getattr(block_state, "txt_ids", None),
                getattr(block_state, "negative_txt_ids", None),
            ),
        }

        components.guider.set_state(step=i, num_inference_steps=block_state.num_inference_steps, timestep=t)
        guider_state = components.guider.prepare_inputs(guider_inputs)

        for guider_state_batch in guider_state:
            components.guider.prepare_models(components.transformer)
            cond_kwargs = {input_name: getattr(guider_state_batch, input_name) for input_name in guider_inputs.keys()}

            noise_pred = components.transformer(
                hidden_states=latent_model_input,
                timestep=timestep / 1000,
                guidance=None,
                img_ids=img_ids,
                joint_attention_kwargs=block_state.joint_attention_kwargs,
                return_dict=False,
                **cond_kwargs,
            )[0]
            guider_state_batch.noise_pred = noise_pred[:, : latents.size(1)]
            components.guider.cleanup_models(components.transformer)

        # perform guidance
        block_state.noise_pred = components.guider(guider_state)[0]

        return components, block_state


class Flux2LoopAfterDenoiser(ModularPipelineBlocks):
    model_name = "flux2"

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("scheduler", FlowMatchEulerDiscreteScheduler)]

    @property
    def description(self) -> str:
        return (
            "Step within the denoising loop that updates the latents after denoising. "
            "This block should be used to compose the `sub_blocks` attribute of a `LoopSequentialPipelineBlocks` "
            "object (e.g. `Flux2DenoiseLoopWrapper`)"
        )

    @property
    def inputs(self) -> list[tuple[str, Any]]:
        return []

    @property
    def intermediate_inputs(self) -> list[str]:
        return [InputParam("generator")]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [OutputParam("latents", type_hint=torch.Tensor, description="The denoised latents")]

    @torch.no_grad()
    def __call__(self, components: Flux2ModularPipeline, block_state: BlockState, i: int, t: torch.Tensor):
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


class Flux2DenoiseLoopWrapper(LoopSequentialPipelineBlocks):
    model_name = "flux2"

    @property
    def description(self) -> str:
        return (
            "Pipeline block that iteratively denoises the latents over `timesteps`. "
            "The specific steps within each iteration can be customized with `sub_blocks` attribute"
        )

    @property
    def loop_expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("scheduler", FlowMatchEulerDiscreteScheduler),
            ComponentSpec("transformer", Flux2Transformer2DModel),
        ]

    @property
    def loop_inputs(self) -> list[InputParam]:
        return [
            InputParam(
                "timesteps",
                required=True,
                type_hint=torch.Tensor,
                description="The timesteps to use for the denoising process.",
            ),
            InputParam(
                "num_inference_steps",
                required=True,
                type_hint=int,
                description="The number of inference steps to use for the denoising process.",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components: Flux2ModularPipeline, state: PipelineState) -> PipelineState:
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

                if XLA_AVAILABLE:
                    xm.mark_step()

        self.set_block_state(state, block_state)
        return components, state


class Flux2DenoiseStep(Flux2DenoiseLoopWrapper):
    block_classes = [Flux2LoopDenoiser, Flux2LoopAfterDenoiser]
    block_names = ["denoiser", "after_denoiser"]

    @property
    def description(self) -> str:
        return (
            "Denoise step that iteratively denoises the latents for Flux2. \n"
            "Its loop logic is defined in `Flux2DenoiseLoopWrapper.__call__` method \n"
            "At each iteration, it runs blocks defined in `sub_blocks` sequentially:\n"
            " - `Flux2LoopDenoiser`\n"
            " - `Flux2LoopAfterDenoiser`\n"
            "This block supports both text-to-image and image-conditioned generation."
        )


class Flux2KleinDenoiseStep(Flux2DenoiseLoopWrapper):
    block_classes = [Flux2KleinLoopDenoiser, Flux2LoopAfterDenoiser]
    block_names = ["denoiser", "after_denoiser"]

    @property
    def description(self) -> str:
        return (
            "Denoise step that iteratively denoises the latents for Flux2. \n"
            "Its loop logic is defined in `Flux2DenoiseLoopWrapper.__call__` method \n"
            "At each iteration, it runs blocks defined in `sub_blocks` sequentially:\n"
            " - `Flux2KleinLoopDenoiser`\n"
            " - `Flux2LoopAfterDenoiser`\n"
            "This block supports both text-to-image and image-conditioned generation."
        )


class Flux2KleinBaseDenoiseStep(Flux2DenoiseLoopWrapper):
    block_classes = [Flux2KleinBaseLoopDenoiser, Flux2LoopAfterDenoiser]
    block_names = ["denoiser", "after_denoiser"]

    @property
    def description(self) -> str:
        return (
            "Denoise step that iteratively denoises the latents for Flux2. \n"
            "Its loop logic is defined in `Flux2DenoiseLoopWrapper.__call__` method \n"
            "At each iteration, it runs blocks defined in `sub_blocks` sequentially:\n"
            " - `Flux2KleinBaseLoopDenoiser`\n"
            " - `Flux2LoopAfterDenoiser`\n"
            "This block supports both text-to-image and image-conditioned generation."
        )
