# Copyright 2026 Black Forest Labs and The HuggingFace Team. All rights reserved.
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

import inspect
from typing import Any

import torch

from ...configuration_utils import FrozenDict
from ...guiders import ClassifierFreeGuidance
from ...models import Flux3Transformer2DModel
from ...schedulers import FlowMatchEulerDiscreteScheduler
from ...utils import is_torch_xla_available, logging
from ..modular_pipeline import (
    BlockState,
    LoopSequentialPipelineBlocks,
    ModularPipelineBlocks,
    PipelineState,
)
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam
from .modular_pipeline import Flux3ModularPipeline


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class Flux3LoopDenoiser(ModularPipelineBlocks):
    model_name = "flux3"

    def __init__(self, guidance_scale: float = 3.5):
        self.guidance_scale = guidance_scale
        super().__init__()

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("transformer", Flux3Transformer2DModel),
            ComponentSpec(
                "guider",
                ClassifierFreeGuidance,
                config=FrozenDict({"guidance_scale": self.guidance_scale}),
                default_creation_method="from_config",
            ),
        ]

    @property
    def description(self) -> str:
        return (
            "Step within the denoising loop that predicts the velocity of the image tokens. The static conditioning "
            "fields (`img_ids`, the reference image tokens and their ids) arrive via `denoiser_input_fields`; "
            "everything matching the transformer signature is forwarded. "
            "This block should be used to compose the `sub_blocks` attribute of a `LoopSequentialPipelineBlocks` "
            "object (e.g. `Flux3DenoiseStep`)"
        )

    @property
    def inputs(self) -> list[tuple[str, Any]]:
        return [
            InputParam(
                "latents",
                required=True,
                type_hint=torch.Tensor,
                description="The image latent tokens to denoise. Shape: (B, seq_len, C)",
            ),
            InputParam.template("denoiser_input_fields"),
            InputParam.template("attention_kwargs", kwargs_type="denoiser_input_fields"),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [OutputParam("noise_pred", type_hint=torch.Tensor, description="The predicted velocity for this step")]

    @torch.no_grad()
    def __call__(
        self, components: Flux3ModularPipeline, block_state: BlockState, i: int, t: torch.Tensor
    ) -> PipelineState:
        denoiser_input_fields = dict(block_state.denoiser_input_fields)
        guider_inputs = {
            "encoder_hidden_states": (
                denoiser_input_fields.pop("prompt_embeds", None),
                denoiser_input_fields.pop("negative_prompt_embeds", None),
            ),
            "txt_ids": (
                denoiser_input_fields.pop("txt_ids", None),
                denoiser_input_fields.pop("negative_txt_ids", None),
            ),
        }

        # the remaining collected fields that match the forward signature go straight to the transformer (`img_ids`,
        # and the `image_cond_*` fields of image-to-image generation)
        transformer_args = set(inspect.signature(components.transformer.forward).parameters.keys())
        transformer_kwargs = {}
        for field_name in list(denoiser_input_fields):
            if field_name in transformer_args:
                transformer_kwargs[field_name] = denoiser_input_fields.pop(field_name)
        if denoiser_input_fields and i == 0:
            logger.warning(f"`denoiser_input_fields` not passed to the transformer: {sorted(denoiser_input_fields)}")

        latents = block_state.latents.to(components.transformer.dtype)
        # the flow timestep in [0, 1] (scheduler timesteps are sigmas * 1000); the transformer applies it to every
        # image token while the reference image tokens and the text context stay at 0
        timestep = (t / 1000).expand(latents.shape[0]).to(latents.dtype)

        components.guider.set_state(step=i, num_inference_steps=block_state.num_inference_steps, timestep=t)
        guider_state = components.guider.prepare_inputs(guider_inputs)

        for guider_state_batch in guider_state:
            components.guider.prepare_models(components.transformer)
            cond_kwargs = {input_name: getattr(guider_state_batch, input_name) for input_name in guider_inputs.keys()}

            guider_state_batch.noise_pred = components.transformer(
                hidden_states=latents,
                timestep=timestep,
                return_dict=False,
                **transformer_kwargs,
                **cond_kwargs,
            )[0]
            components.guider.cleanup_models(components.transformer)

        block_state.noise_pred = components.guider(guider_state)[0]

        return components, block_state


class Flux3DistilledLoopDenoiser(ModularPipelineBlocks):
    model_name = "flux3-distilled"

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("transformer", Flux3Transformer2DModel)]

    @property
    def description(self) -> str:
        return (
            "Step within the denoising loop that predicts the velocity of the image tokens with a step-distilled "
            "(flow-map) FLUX.3 transformer: no classifier-free guidance, and each step also passes the transformer the "
            "timestep it jumps to. The static conditioning fields (`prompt_embeds`, `txt_ids`, `img_ids`, the "
            "reference image tokens and their ids) arrive via `denoiser_input_fields`. "
            "This block should be used to compose the `sub_blocks` attribute of a `LoopSequentialPipelineBlocks` "
            "object (e.g. `Flux3DistilledDenoiseStep`)"
        )

    @property
    def inputs(self) -> list[tuple[str, Any]]:
        return [
            InputParam(
                "latents",
                required=True,
                type_hint=torch.Tensor,
                description="The image latent tokens to denoise. Shape: (B, seq_len, C)",
            ),
            InputParam.template("denoiser_input_fields"),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [OutputParam("noise_pred", type_hint=torch.Tensor, description="The predicted velocity for this step")]

    @torch.no_grad()
    def __call__(
        self, components: Flux3ModularPipeline, block_state: BlockState, i: int, t: torch.Tensor
    ) -> PipelineState:
        denoiser_input_fields = dict(block_state.denoiser_input_fields)
        prompt_embeds = denoiser_input_fields.pop("prompt_embeds")

        # the remaining collected fields that match the forward signature go straight to the transformer (`img_ids`,
        # `txt_ids`, and the `image_cond_*` fields of image-to-image generation)
        transformer_args = set(inspect.signature(components.transformer.forward).parameters.keys())
        transformer_kwargs = {}
        for field_name in list(denoiser_input_fields):
            if field_name in transformer_args:
                transformer_kwargs[field_name] = denoiser_input_fields.pop(field_name)
        if denoiser_input_fields and i == 0:
            logger.warning(f"`denoiser_input_fields` not passed to the transformer: {sorted(denoiser_input_fields)}")

        latents = block_state.latents.to(components.transformer.dtype)
        batch_size = latents.shape[0]
        # the flow timestep in [0, 1] (scheduler timesteps are sigmas * 1000), and the one this step jumps to: the next
        # scheduler timestep, 0 after the last
        timesteps = block_state.timesteps
        next_t = timesteps[i + 1] if i + 1 < len(timesteps) else torch.zeros_like(t)
        timestep = (t / 1000).expand(batch_size).to(latents.dtype)
        target_timestep = (next_t / 1000).expand(batch_size).to(latents.dtype)

        block_state.noise_pred = components.transformer(
            hidden_states=latents,
            encoder_hidden_states=prompt_embeds,
            timestep=timestep,
            target_timestep=target_timestep,
            return_dict=False,
            **transformer_kwargs,
        )[0]

        return components, block_state


class Flux3LoopAfterDenoiser(ModularPipelineBlocks):
    model_name = "flux3"

    @property
    def expected_components(self) -> list[ComponentSpec]:
        # config-less: the scheduler config is owned by `Flux3SetTimestepsStep`, whose spec is combined first
        return [ComponentSpec("scheduler", FlowMatchEulerDiscreteScheduler, default_creation_method="from_config")]

    @property
    def description(self) -> str:
        return (
            "Step within the denoising loop that updates the latents after denoising. "
            "This block should be used to compose the `sub_blocks` attribute of a `LoopSequentialPipelineBlocks` "
            "object (e.g. `Flux3DenoiseLoopWrapper`)"
        )

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [OutputParam("latents", type_hint=torch.Tensor, description="The denoised latents")]

    @torch.no_grad()
    def __call__(
        self, components: Flux3ModularPipeline, block_state: BlockState, i: int, t: torch.Tensor
    ) -> tuple[Flux3ModularPipeline, BlockState]:
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


class Flux3DenoiseLoopWrapper(LoopSequentialPipelineBlocks):
    model_name = "flux3"

    @property
    def description(self) -> str:
        return (
            "Pipeline block that iteratively denoises the latents over `timesteps`. "
            "The specific steps within each iteration can be customized with `sub_blocks` attribute"
        )

    @property
    def loop_expected_components(self) -> list[ComponentSpec]:
        return [
            # config-less: the scheduler config is owned by `Flux3SetTimestepsStep`, whose spec is combined first
            ComponentSpec("scheduler", FlowMatchEulerDiscreteScheduler, default_creation_method="from_config"),
            ComponentSpec("transformer", Flux3Transformer2DModel),
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
    def __call__(self, components: Flux3ModularPipeline, state: PipelineState) -> PipelineState:
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


# auto_docstring
class Flux3DenoiseStep(Flux3DenoiseLoopWrapper):
    """
    Denoise step that iteratively denoises the latents for FLUX.3.
      Its loop logic is defined in `Flux3DenoiseLoopWrapper.__call__` method At each iteration, it runs blocks defined
      in `sub_blocks` sequentially:
       - `Flux3LoopDenoiser`
       - `Flux3LoopAfterDenoiser`
      This block supports text-to-image and image-to-image generation.

      Components:
          transformer (`Flux3Transformer2DModel`) guider (`ClassifierFreeGuidance`) scheduler
          (`FlowMatchEulerDiscreteScheduler`)

      Inputs:
          timesteps (`Tensor`):
              The timesteps to use for the denoising process.
          num_inference_steps (`int`):
              The number of inference steps to use for the denoising process.
          latents (`Tensor`):
              The image latent tokens to denoise. Shape: (B, seq_len, C)
          **denoiser_input_fields (`None`, *optional*):
              conditional model inputs for the denoiser: e.g. prompt_embeds, negative_prompt_embeds, etc.
          attention_kwargs (`dict`, *optional*):
              Additional kwargs for attention processors.

      Outputs:
          latents (`Tensor`):
              The denoised latents
    """

    block_classes = [Flux3LoopDenoiser(), Flux3LoopAfterDenoiser()]
    block_names = ["denoiser", "after_denoiser"]

    @property
    def description(self) -> str:
        return (
            "Denoise step that iteratively denoises the latents for FLUX.3. \n"
            "Its loop logic is defined in `Flux3DenoiseLoopWrapper.__call__` method \n"
            "At each iteration, it runs blocks defined in `sub_blocks` sequentially:\n"
            " - `Flux3LoopDenoiser`\n"
            " - `Flux3LoopAfterDenoiser`\n"
            "This block supports text-to-image and image-to-image generation."
        )


# auto_docstring
class Flux3DistilledDenoiseStep(Flux3DenoiseLoopWrapper):
    """
    Denoise step that iteratively denoises the latents for the step-distilled (flow-map) FLUX.3 model.
      Its loop logic is defined in `Flux3DenoiseLoopWrapper.__call__` method At each iteration, it runs blocks defined
      in `sub_blocks` sequentially:
       - `Flux3DistilledLoopDenoiser`
       - `Flux3LoopAfterDenoiser`
      This block supports text-to-image and image-to-image generation, without classifier-free guidance.

      Components:
          transformer (`Flux3Transformer2DModel`) scheduler (`FlowMatchEulerDiscreteScheduler`)

      Inputs:
          timesteps (`Tensor`):
              The timesteps to use for the denoising process.
          num_inference_steps (`int`):
              The number of inference steps to use for the denoising process.
          latents (`Tensor`):
              The image latent tokens to denoise. Shape: (B, seq_len, C)
          **denoiser_input_fields (`None`, *optional*):
              conditional model inputs for the denoiser: e.g. prompt_embeds, negative_prompt_embeds, etc.

      Outputs:
          latents (`Tensor`):
              The denoised latents
    """

    model_name = "flux3-distilled"
    """
    Denoise step that iteratively denoises the latents for the step-distilled (flow-map) FLUX.3 model.
      Its loop logic is defined in `Flux3DenoiseLoopWrapper.__call__` method At each iteration, it runs blocks defined
      in `sub_blocks` sequentially:
       - `Flux3DistilledLoopDenoiser`
       - `Flux3LoopAfterDenoiser`
      This block supports text-to-image and image-to-image generation, without classifier-free guidance.

      Components:
          transformer (`Flux3Transformer2DModel`) scheduler (`FlowMatchEulerDiscreteScheduler`)

      Inputs:
          timesteps (`Tensor`):
              The timesteps to use for the denoising process.
          num_inference_steps (`int`):
              The number of inference steps to use for the denoising process.
          latents (`Tensor`):
              The image latent tokens to denoise. Shape: (B, seq_len, C)
          **denoiser_input_fields (`None`, *optional*):
              conditional model inputs for the denoiser: e.g. prompt_embeds, negative_prompt_embeds, etc.

      Outputs:
          latents (`Tensor`):
              The denoised latents
    """

    block_classes = [Flux3DistilledLoopDenoiser(), Flux3LoopAfterDenoiser()]
    block_names = ["denoiser", "after_denoiser"]

    @property
    def description(self) -> str:
        return (
            "Denoise step that iteratively denoises the latents for the step-distilled (flow-map) FLUX.3 model. \n"
            "Its loop logic is defined in `Flux3DenoiseLoopWrapper.__call__` method \n"
            "At each iteration, it runs blocks defined in `sub_blocks` sequentially:\n"
            " - `Flux3DistilledLoopDenoiser`\n"
            " - `Flux3LoopAfterDenoiser`\n"
            "This block supports text-to-image and image-to-image generation, without classifier-free guidance."
        )
