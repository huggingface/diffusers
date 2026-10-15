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

import inspect

import torch

from ...configuration_utils import FrozenDict
from ...guiders import ClassifierFreeGuidance
from ...models import QwenImage21Transformer2DModel
from ...models.transformers.transformer_qwenimage21 import QwenImage21KVCache
from ...schedulers import FlowMatchEulerDiscreteScheduler
from ...utils import is_torch_xla_available, logging
from ..modular_pipeline import BlockState, LoopSequentialPipelineBlocks, ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam
from .modular_pipeline import QwenImage21ModularPipeline


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def joint_vision_mask(image_pad_mask: torch.Tensor, num_target_tokens: int) -> torch.Tensor:
    # The transformer expands every vision slot into a 2x2 group of latent tokens, so the target image takes one
    # slot per four latents, appended after the prompt's vision positions.
    target_slots = num_target_tokens // 4
    return torch.cat([image_pad_mask, image_pad_mask.new_ones(image_pad_mask.shape[0], target_slots)], dim=1)


# ====================
# 1. LOOP STEPS (run at each denoising step)
# ====================


# loop step: before denoiser
class QwenImage21LoopBeforeDenoiser(ModularPipelineBlocks):
    model_name = "qwenimage21"

    @property
    def description(self) -> str:
        return (
            "step within the denoising loop that prepares the latent input for the denoiser. "
            "This block should be used to compose the `sub_blocks` attribute of a `LoopSequentialPipelineBlocks` "
            "object (e.g. `QwenImage21DenoiseLoopWrapper`)"
        )

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                name="latents",
                required=True,
                type_hint=torch.Tensor,
                description="The initial latents to use for the denoising process. Can be generated in prepare_latent step.",
            ),
        ]

    @torch.no_grad()
    def __call__(
        self, components: QwenImage21ModularPipeline, block_state: BlockState, i: int, t: torch.Tensor
    ) -> tuple[QwenImage21ModularPipeline, BlockState]:
        block_state.latent_model_input = block_state.latents
        block_state.timestep = t.expand(block_state.latents.shape[0]).to(block_state.latents.dtype)
        return components, block_state


class QwenImage21ImageConditionedLoopBeforeDenoiser(ModularPipelineBlocks):
    model_name = "qwenimage21"

    @property
    def description(self) -> str:
        return (
            "step within the denoising loop that prepends the condition image tokens to the target latents. "
            "This block should be used to compose the `sub_blocks` attribute of a `LoopSequentialPipelineBlocks` "
            "object (e.g. `QwenImage21DenoiseLoopWrapper`)"
        )

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                name="latents",
                required=True,
                type_hint=torch.Tensor,
                description="The initial latents to use for the denoising process. Can be generated in prepare_latent step.",
            ),
            InputParam(
                name="image_latents",
                required=True,
                type_hint=torch.Tensor,
                description="Packed condition image latents. Can be generated in the additional inputs step.",
            ),
        ]

    @torch.no_grad()
    def __call__(
        self, components: QwenImage21ModularPipeline, block_state: BlockState, i: int, t: torch.Tensor
    ) -> tuple[QwenImage21ModularPipeline, BlockState]:
        # Condition images come first in the joint sequence, the target image last.
        block_state.latent_model_input = torch.cat([block_state.image_latents, block_state.latents], dim=1)
        block_state.timestep = t.expand(block_state.latents.shape[0]).to(block_state.latents.dtype)
        return components, block_state


# loop step: denoiser
class QwenImage21LoopDenoiser(ModularPipelineBlocks):
    model_name = "qwenimage21"

    @property
    def description(self) -> str:
        return (
            "step within the denoising loop that denoise the latent input for the denoiser. "
            "This block should be used to compose the `sub_blocks` attribute of a `LoopSequentialPipelineBlocks` "
            "object (e.g. `QwenImage21DenoiseLoopWrapper`)"
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec(
                "guider",
                ClassifierFreeGuidance,
                config=FrozenDict({"guidance_scale": 1.0}),
                default_creation_method="from_config",
            ),
            ComponentSpec("transformer", QwenImage21Transformer2DModel),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("attention_kwargs"),
            InputParam.template("denoiser_input_fields"),
            InputParam(
                name="img_shapes",
                required=True,
                type_hint=list[list[tuple[int, int, int]]],
                description="Per-sample latent grids, condition images first and the target image last. Can be generated in prepare_rope_inputs step.",
            ),
            InputParam(
                name="image_pad_mask",
                required=True,
                kwargs_type="denoiser_input_fields",
                type_hint=torch.Tensor,
                description="Bool mask marking the vision positions of the prompt embeddings. Can be generated in the text input step.",
            ),
            InputParam(
                name="negative_image_pad_mask",
                kwargs_type="denoiser_input_fields",
                type_hint=torch.Tensor,
                description="Bool mask marking the vision positions of the negative prompt embeddings. Can be generated in the text input step.",
            ),
        ]

    @torch.no_grad()
    def __call__(
        self, components: QwenImage21ModularPipeline, block_state: BlockState, i: int, t: torch.Tensor
    ) -> tuple[QwenImage21ModularPipeline, BlockState]:
        guider_inputs = {
            "encoder_hidden_states": (
                getattr(block_state, "prompt_embeds", None),
                getattr(block_state, "negative_prompt_embeds", None),
            ),
            "encoder_hidden_states_mask": (
                getattr(block_state, "prompt_embeds_mask", None),
                getattr(block_state, "negative_prompt_embeds_mask", None),
            ),
            "img_mask": (
                joint_vision_mask(block_state.image_pad_mask, block_state.latents.shape[1]),
                joint_vision_mask(block_state.negative_image_pad_mask, block_state.latents.shape[1])
                if block_state.negative_image_pad_mask is not None
                else None,
            ),
        }

        transformer_args = set(inspect.signature(components.transformer.forward).parameters.keys())
        additional_cond_kwargs = {}
        for field_name, field_value in block_state.denoiser_input_fields.items():
            if field_name in transformer_args and field_name not in guider_inputs:
                additional_cond_kwargs[field_name] = field_value
        block_state.additional_cond_kwargs.update(additional_cond_kwargs)

        components.guider.set_state(step=i, num_inference_steps=block_state.num_inference_steps, timestep=t)
        guider_state = components.guider.prepare_inputs(guider_inputs)

        num_target_tokens = block_state.latents.shape[1]
        for guider_state_batch in guider_state:
            components.guider.prepare_models(components.transformer)
            cond_kwargs = {input_name: getattr(guider_state_batch, input_name) for input_name in guider_inputs.keys()}
            context_name = getattr(guider_state_batch, components.guider._identifier_key)

            # Text and condition-image keys and values are step-independent under `causal_condition`, so the first
            # call of each guidance branch prefills its cache and the later calls only recompute the target tokens.
            kv_cache, kv_cache_mode = None, None
            if block_state.cache_enabled:
                kv_cache = block_state.kv_caches.get(context_name)
                kv_cache_mode = "cached" if kv_cache is not None else "extract"
                if kv_cache is None:
                    kv_cache = QwenImage21KVCache(len(components.transformer.transformer_blocks))
                    block_state.kv_caches[context_name] = kv_cache

            with components.transformer.cache_context(context_name):
                noise_pred = components.transformer(
                    hidden_states=block_state.latent_model_input,
                    timestep=block_state.timestep / 1000,
                    attention_kwargs=block_state.attention_kwargs,
                    kv_cache=kv_cache,
                    kv_cache_mode=kv_cache_mode,
                    return_dict=False,
                    **cond_kwargs,
                    **block_state.additional_cond_kwargs,
                )[0]
            guider_state_batch.noise_pred = noise_pred[:, -num_target_tokens:]
            components.guider.cleanup_models(components.transformer)

        block_state.noise_pred = components.guider(guider_state)[0]

        return components, block_state


# loop step: denoiser (Turbo, no guidance)
class QwenImage21TurboLoopDenoiser(ModularPipelineBlocks):
    model_name = "qwenimage21-turbo"

    @property
    def description(self) -> str:
        return (
            "step within the denoising loop that denoise the latent input for the denoiser without guidance. "
            "This block should be used to compose the `sub_blocks` attribute of a `LoopSequentialPipelineBlocks` "
            "object (e.g. `QwenImage21DenoiseLoopWrapper`)"
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("transformer", QwenImage21Transformer2DModel)]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("attention_kwargs"),
            InputParam.template("denoiser_input_fields"),
            InputParam(
                name="img_shapes",
                required=True,
                type_hint=list[list[tuple[int, int, int]]],
                description="Per-sample latent grids, condition images first and the target image last. Can be generated in prepare_rope_inputs step.",
            ),
            InputParam(
                name="image_pad_mask",
                required=True,
                kwargs_type="denoiser_input_fields",
                type_hint=torch.Tensor,
                description="Bool mask marking the vision positions of the prompt embeddings. Can be generated in the text input step.",
            ),
        ]

    @torch.no_grad()
    def __call__(
        self, components: QwenImage21ModularPipeline, block_state: BlockState, i: int, t: torch.Tensor
    ) -> tuple[QwenImage21ModularPipeline, BlockState]:
        transformer_args = set(inspect.signature(components.transformer.forward).parameters.keys())
        additional_cond_kwargs = {}
        for field_name, field_value in block_state.denoiser_input_fields.items():
            if field_name in transformer_args:
                additional_cond_kwargs[field_name] = field_value
        block_state.additional_cond_kwargs.update(additional_cond_kwargs)

        num_target_tokens = block_state.latents.shape[1]
        # Text and condition-image keys and values are step-independent under `causal_condition`, so the first
        # call prefills the cache and the later calls only recompute the target tokens.
        kv_cache, kv_cache_mode = None, None
        if block_state.cache_enabled:
            kv_cache = block_state.kv_caches.get("cond")
            kv_cache_mode = "cached" if kv_cache is not None else "extract"
            if kv_cache is None:
                kv_cache = QwenImage21KVCache(len(components.transformer.transformer_blocks))
                block_state.kv_caches["cond"] = kv_cache

        with components.transformer.cache_context("cond"):
            noise_pred = components.transformer(
                hidden_states=block_state.latent_model_input,
                timestep=block_state.timestep / 1000,
                encoder_hidden_states=block_state.prompt_embeds,
                encoder_hidden_states_mask=getattr(block_state, "prompt_embeds_mask", None),
                img_mask=joint_vision_mask(block_state.image_pad_mask, num_target_tokens),
                attention_kwargs=block_state.attention_kwargs,
                kv_cache=kv_cache,
                kv_cache_mode=kv_cache_mode,
                return_dict=False,
                **block_state.additional_cond_kwargs,
            )[0]
        block_state.noise_pred = noise_pred[:, -num_target_tokens:]

        return components, block_state


# loop step: after denoiser
class QwenImage21LoopAfterDenoiser(ModularPipelineBlocks):
    model_name = "qwenimage21"

    @property
    def description(self) -> str:
        return (
            "step within the denoising loop that updates the latents. "
            "This block should be used to compose the `sub_blocks` attribute of a `LoopSequentialPipelineBlocks` "
            "object (e.g. `QwenImage21DenoiseLoopWrapper`)"
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("scheduler", FlowMatchEulerDiscreteScheduler),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam.template("latents"),
        ]

    @torch.no_grad()
    def __call__(
        self, components: QwenImage21ModularPipeline, block_state: BlockState, i: int, t: torch.Tensor
    ) -> tuple[QwenImage21ModularPipeline, BlockState]:
        latents_dtype = block_state.latents.dtype
        block_state.latents = components.scheduler.step(
            block_state.noise_pred,
            t,
            block_state.latents,
            return_dict=False,
        )[0]

        if block_state.latents.dtype != latents_dtype:
            if torch.backends.mps.is_available():
                # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                block_state.latents = block_state.latents.to(latents_dtype)

        return components, block_state


# ====================
# 2. LOOP WRAPPER (the denoising loop)
# ====================


class QwenImage21DenoiseLoopWrapper(LoopSequentialPipelineBlocks):
    model_name = "qwenimage21"

    @property
    def description(self) -> str:
        return (
            "Pipeline block that iteratively denoise the latents over `timesteps`. "
            "The specific steps with each iteration can be customized with `sub_blocks` attributes"
        )

    @property
    def loop_expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("scheduler", FlowMatchEulerDiscreteScheduler),
            ComponentSpec("transformer", QwenImage21Transformer2DModel),
        ]

    @property
    def loop_inputs(self) -> list[InputParam]:
        return [
            InputParam(
                name="timesteps",
                required=True,
                type_hint=torch.Tensor,
                description="The timesteps to use for the denoising process. Can be generated in set_timesteps step.",
            ),
            InputParam.template("num_inference_steps", required=True),
            InputParam(
                name="use_kv_cache",
                type_hint=bool,
                default=True,
                description=(
                    "Cache the text and condition-image keys and values after the first step. Valid because "
                    "`causal_condition` modulates those tokens from `t = 0`, making their activations step-independent. "
                    "Toggling it does not reproduce the same image bit-for-bit in reduced precision."
                ),
            ),
        ]

    @torch.no_grad()
    def __call__(
        self, components: QwenImage21ModularPipeline, state: PipelineState
    ) -> tuple[QwenImage21ModularPipeline, PipelineState]:
        block_state = self.get_block_state(state)

        block_state.num_warmup_steps = max(
            len(block_state.timesteps) - block_state.num_inference_steps * components.scheduler.order, 0
        )
        block_state.additional_cond_kwargs = {}
        # One prefix cache per guidance branch, filled on that branch's first call and discarded with the loop.
        block_state.cache_enabled = block_state.use_kv_cache and components.transformer.config.causal_condition
        block_state.kv_caches = {}

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


# ====================
# 3. DENOISE STEPS: compose the denoising loop with loop wrapper + loop steps
# ====================


# auto_docstring
class QwenImage21DenoiseStep(QwenImage21DenoiseLoopWrapper):
    """
    Denoise step that iteratively denoise the latents.
      Its loop logic is defined in `QwenImage21DenoiseLoopWrapper.__call__` method At each iteration, it runs blocks
      defined in `sub_blocks` sequencially:
       - `QwenImage21LoopBeforeDenoiser`
       - `QwenImage21LoopDenoiser`
       - `QwenImage21LoopAfterDenoiser`
      This block supports text-to-image generation.

      Components:
          guider (`ClassifierFreeGuidance`) transformer (`QwenImage21Transformer2DModel`) scheduler
          (`FlowMatchEulerDiscreteScheduler`)

      Inputs:
          timesteps (`Tensor`):
              The timesteps to use for the denoising process. Can be generated in set_timesteps step.
          num_inference_steps (`int`):
              The number of denoising steps.
          use_kv_cache (`bool`, *optional*, defaults to True):
              Cache the text and condition-image keys and values after the first step. Valid because `causal_condition`
              modulates those tokens from `t = 0`, making their activations step-independent. Toggling it does not
              reproduce the same image bit-for-bit in reduced precision.
          latents (`Tensor`):
              The initial latents to use for the denoising process. Can be generated in prepare_latent step.
          attention_kwargs (`dict`, *optional*):
              Additional kwargs for attention processors.
          **denoiser_input_fields (`None`, *optional*):
              conditional model inputs for the denoiser: e.g. prompt_embeds, negative_prompt_embeds, etc.
          img_shapes (`list`):
              Per-sample latent grids, condition images first and the target image last. Can be generated in
              prepare_rope_inputs step.
          image_pad_mask (`Tensor`):
              Bool mask marking the vision positions of the prompt embeddings. Can be generated in the text input step.
          negative_image_pad_mask (`Tensor`, *optional*):
              Bool mask marking the vision positions of the negative prompt embeddings. Can be generated in the text
              input step.

      Outputs:
          latents (`Tensor`):
              Denoised latents.
    """

    model_name = "qwenimage21"
    block_classes = [
        QwenImage21LoopBeforeDenoiser,
        QwenImage21LoopDenoiser,
        QwenImage21LoopAfterDenoiser,
    ]
    block_names = ["before_denoiser", "denoiser", "after_denoiser"]

    @property
    def description(self) -> str:
        return (
            "Denoise step that iteratively denoise the latents. \n"
            "Its loop logic is defined in `QwenImage21DenoiseLoopWrapper.__call__` method \n"
            "At each iteration, it runs blocks defined in `sub_blocks` sequencially:\n"
            " - `QwenImage21LoopBeforeDenoiser`\n"
            " - `QwenImage21LoopDenoiser`\n"
            " - `QwenImage21LoopAfterDenoiser`\n"
            "This block supports text-to-image generation."
        )


# auto_docstring
class QwenImage21ImageConditionedDenoiseStep(QwenImage21DenoiseLoopWrapper):
    """
    Denoise step that iteratively denoise the latents.
      Its loop logic is defined in `QwenImage21DenoiseLoopWrapper.__call__` method At each iteration, it runs blocks
      defined in `sub_blocks` sequencially:
       - `QwenImage21ImageConditionedLoopBeforeDenoiser`
       - `QwenImage21LoopDenoiser`
       - `QwenImage21LoopAfterDenoiser`
      This block supports image-conditioned generation.

      Components:
          guider (`ClassifierFreeGuidance`) transformer (`QwenImage21Transformer2DModel`) scheduler
          (`FlowMatchEulerDiscreteScheduler`)

      Inputs:
          timesteps (`Tensor`):
              The timesteps to use for the denoising process. Can be generated in set_timesteps step.
          num_inference_steps (`int`):
              The number of denoising steps.
          use_kv_cache (`bool`, *optional*, defaults to True):
              Cache the text and condition-image keys and values after the first step. Valid because `causal_condition`
              modulates those tokens from `t = 0`, making their activations step-independent. Toggling it does not
              reproduce the same image bit-for-bit in reduced precision.
          latents (`Tensor`):
              The initial latents to use for the denoising process. Can be generated in prepare_latent step.
          image_latents (`Tensor`):
              Packed condition image latents. Can be generated in the additional inputs step.
          attention_kwargs (`dict`, *optional*):
              Additional kwargs for attention processors.
          **denoiser_input_fields (`None`, *optional*):
              conditional model inputs for the denoiser: e.g. prompt_embeds, negative_prompt_embeds, etc.
          img_shapes (`list`):
              Per-sample latent grids, condition images first and the target image last. Can be generated in
              prepare_rope_inputs step.
          image_pad_mask (`Tensor`):
              Bool mask marking the vision positions of the prompt embeddings. Can be generated in the text input step.
          negative_image_pad_mask (`Tensor`, *optional*):
              Bool mask marking the vision positions of the negative prompt embeddings. Can be generated in the text
              input step.

      Outputs:
          latents (`Tensor`):
              Denoised latents.
    """

    model_name = "qwenimage21"
    block_classes = [
        QwenImage21ImageConditionedLoopBeforeDenoiser,
        QwenImage21LoopDenoiser,
        QwenImage21LoopAfterDenoiser,
    ]
    block_names = ["before_denoiser", "denoiser", "after_denoiser"]

    @property
    def description(self) -> str:
        return (
            "Denoise step that iteratively denoise the latents. \n"
            "Its loop logic is defined in `QwenImage21DenoiseLoopWrapper.__call__` method \n"
            "At each iteration, it runs blocks defined in `sub_blocks` sequencially:\n"
            " - `QwenImage21ImageConditionedLoopBeforeDenoiser`\n"
            " - `QwenImage21LoopDenoiser`\n"
            " - `QwenImage21LoopAfterDenoiser`\n"
            "This block supports image-conditioned generation."
        )


# auto_docstring
class QwenImage21TurboDenoiseStep(QwenImage21DenoiseLoopWrapper):
    """
    Denoise step that iteratively denoise the latents.
      Its loop logic is defined in `QwenImage21DenoiseLoopWrapper.__call__` method At each iteration, it runs blocks
      defined in `sub_blocks` sequencially:
       - `QwenImage21LoopBeforeDenoiser`
       - `QwenImage21TurboLoopDenoiser`
       - `QwenImage21LoopAfterDenoiser`
      This block supports text-to-image generation with the Turbo checkpoint.

      Components:
          transformer (`QwenImage21Transformer2DModel`) scheduler (`FlowMatchEulerDiscreteScheduler`)

      Inputs:
          timesteps (`Tensor`):
              The timesteps to use for the denoising process. Can be generated in set_timesteps step.
          num_inference_steps (`int`):
              The number of denoising steps.
          use_kv_cache (`bool`, *optional*, defaults to True):
              Cache the text and condition-image keys and values after the first step. Valid because `causal_condition`
              modulates those tokens from `t = 0`, making their activations step-independent. Toggling it does not
              reproduce the same image bit-for-bit in reduced precision.
          latents (`Tensor`):
              The initial latents to use for the denoising process. Can be generated in prepare_latent step.
          attention_kwargs (`dict`, *optional*):
              Additional kwargs for attention processors.
          **denoiser_input_fields (`None`, *optional*):
              conditional model inputs for the denoiser: e.g. prompt_embeds, negative_prompt_embeds, etc.
          img_shapes (`list`):
              Per-sample latent grids, condition images first and the target image last. Can be generated in
              prepare_rope_inputs step.
          image_pad_mask (`Tensor`):
              Bool mask marking the vision positions of the prompt embeddings. Can be generated in the text input step.

      Outputs:
          latents (`Tensor`):
              Denoised latents.
    """

    model_name = "qwenimage21-turbo"
    block_classes = [
        QwenImage21LoopBeforeDenoiser,
        QwenImage21TurboLoopDenoiser,
        QwenImage21LoopAfterDenoiser,
    ]
    block_names = ["before_denoiser", "denoiser", "after_denoiser"]

    @property
    def description(self) -> str:
        return (
            "Denoise step that iteratively denoise the latents. \n"
            "Its loop logic is defined in `QwenImage21DenoiseLoopWrapper.__call__` method \n"
            "At each iteration, it runs blocks defined in `sub_blocks` sequencially:\n"
            " - `QwenImage21LoopBeforeDenoiser`\n"
            " - `QwenImage21TurboLoopDenoiser`\n"
            " - `QwenImage21LoopAfterDenoiser`\n"
            "This block supports text-to-image generation with the Turbo checkpoint."
        )


# auto_docstring
class QwenImage21TurboImageConditionedDenoiseStep(QwenImage21DenoiseLoopWrapper):
    """
    Denoise step that iteratively denoise the latents.
      Its loop logic is defined in `QwenImage21DenoiseLoopWrapper.__call__` method At each iteration, it runs blocks
      defined in `sub_blocks` sequencially:
       - `QwenImage21ImageConditionedLoopBeforeDenoiser`
       - `QwenImage21TurboLoopDenoiser`
       - `QwenImage21LoopAfterDenoiser`
      This block supports image-conditioned generation with the Turbo checkpoint.

      Components:
          transformer (`QwenImage21Transformer2DModel`) scheduler (`FlowMatchEulerDiscreteScheduler`)

      Inputs:
          timesteps (`Tensor`):
              The timesteps to use for the denoising process. Can be generated in set_timesteps step.
          num_inference_steps (`int`):
              The number of denoising steps.
          use_kv_cache (`bool`, *optional*, defaults to True):
              Cache the text and condition-image keys and values after the first step. Valid because `causal_condition`
              modulates those tokens from `t = 0`, making their activations step-independent. Toggling it does not
              reproduce the same image bit-for-bit in reduced precision.
          latents (`Tensor`):
              The initial latents to use for the denoising process. Can be generated in prepare_latent step.
          image_latents (`Tensor`):
              Packed condition image latents. Can be generated in the additional inputs step.
          attention_kwargs (`dict`, *optional*):
              Additional kwargs for attention processors.
          **denoiser_input_fields (`None`, *optional*):
              conditional model inputs for the denoiser: e.g. prompt_embeds, negative_prompt_embeds, etc.
          img_shapes (`list`):
              Per-sample latent grids, condition images first and the target image last. Can be generated in
              prepare_rope_inputs step.
          image_pad_mask (`Tensor`):
              Bool mask marking the vision positions of the prompt embeddings. Can be generated in the text input step.

      Outputs:
          latents (`Tensor`):
              Denoised latents.
    """

    model_name = "qwenimage21-turbo"
    block_classes = [
        QwenImage21ImageConditionedLoopBeforeDenoiser,
        QwenImage21TurboLoopDenoiser,
        QwenImage21LoopAfterDenoiser,
    ]
    block_names = ["before_denoiser", "denoiser", "after_denoiser"]

    @property
    def description(self) -> str:
        return (
            "Denoise step that iteratively denoise the latents. \n"
            "Its loop logic is defined in `QwenImage21DenoiseLoopWrapper.__call__` method \n"
            "At each iteration, it runs blocks defined in `sub_blocks` sequencially:\n"
            " - `QwenImage21ImageConditionedLoopBeforeDenoiser`\n"
            " - `QwenImage21TurboLoopDenoiser`\n"
            " - `QwenImage21LoopAfterDenoiser`\n"
            "This block supports image-conditioned generation with the Turbo checkpoint."
        )
