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
from ...guiders import ClassifierFreeGuidance
from ...models import QwenImage21Transformer2DModel
from ...models.transformers.transformer_qwenimage21 import QwenImage21KVCache
from ...schedulers import FlowMatchEulerDiscreteScheduler
from ..modular_pipeline import LoopSequentialPipelineBlocks, ModularPipelineBlocks
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam


class QwenImage21LoopDenoiser(ModularPipelineBlocks):
    model_name = "qwenimage21"

    @property
    def description(self):
        return "Predict target flow with separate condition and guidance KV histories."

    @property
    def expected_components(self):
        return [
            ComponentSpec("transformer", QwenImage21Transformer2DModel),
            ComponentSpec(
                "guider",
                ClassifierFreeGuidance,
                config=FrozenDict({"guidance_scale": 1.0}),
                default_creation_method="from_config",
            ),
        ]

    @property
    def inputs(self):
        return [
            InputParam.template("latents", required=True),
            InputParam.template("prompt_embeds", required=True),
            InputParam.template("negative_prompt_embeds", required=False),
            InputParam.template("prompt_embeds_mask", required=False),
            InputParam.template("negative_prompt_embeds_mask", required=False),
            InputParam.template("attention_kwargs"),
            InputParam(
                "condition_latents", type_hint=torch.Tensor, description="Condition image tokens preceding the target."
            ),
            InputParam("img_shapes", required=True, type_hint=list, description="Condition and target grid shapes."),
            InputParam("img_mask", required=True, type_hint=torch.Tensor, description="Joint positive vision mask."),
            InputParam("negative_img_mask", type_hint=torch.Tensor, description="Joint negative vision mask."),
        ]

    @property
    def intermediate_outputs(self):
        return [OutputParam("noise_pred", type_hint=torch.Tensor, description="Guided target flow prediction.")]

    @torch.no_grad()
    def __call__(self, components, block_state, i, t):
        guider_inputs = {
            "encoder_hidden_states": (block_state.prompt_embeds, block_state.negative_prompt_embeds),
            "encoder_hidden_states_mask": (block_state.prompt_embeds_mask, block_state.negative_prompt_embeds_mask),
            "img_mask": (block_state.img_mask, block_state.negative_img_mask),
        }
        components.guider.set_state(step=i, num_inference_steps=block_state.num_inference_steps, timestep=t)
        batches = components.guider.prepare_inputs(guider_inputs)
        latent_input = block_state.latents
        if block_state.condition_latents is not None:
            latent_input = torch.cat([block_state.condition_latents, latent_input], dim=1)
        timestep = t.expand(block_state.latents.shape[0]).to(block_state.latents.dtype) / 1000
        for batch in batches:
            components.guider.prepare_models(components.transformer)
            context = getattr(batch, components.guider._identifier_key)
            cache, mode = None, None
            if block_state.cache_enabled:
                if context not in block_state.kv_caches:
                    block_state.kv_caches[context] = QwenImage21KVCache(len(components.transformer.transformer_blocks))
                    mode = "extract"
                else:
                    mode = "cached"
                cache = block_state.kv_caches[context]
            try:
                with components.transformer.cache_context(context):
                    batch.noise_pred = components.transformer(
                        hidden_states=latent_input,
                        timestep=timestep,
                        img_shapes=block_state.img_shapes,
                        attention_kwargs=block_state.attention_kwargs,
                        kv_cache=cache,
                        kv_cache_mode=mode,
                        return_dict=False,
                        **{name: getattr(batch, name) for name in guider_inputs},
                    )[0][:, -block_state.latents.shape[1] :]
            finally:
                components.guider.cleanup_models(components.transformer)
        block_state.noise_pred = components.guider(batches)[0]
        return components, block_state


class QwenImage21LoopStep(ModularPipelineBlocks):
    model_name = "qwenimage21"

    @property
    def description(self):
        return "Advance the flow-matching scheduler."

    @property
    def expected_components(self):
        return [ComponentSpec("scheduler", FlowMatchEulerDiscreteScheduler)]

    @property
    def inputs(self):
        return [
            InputParam.template("latents", required=True),
            InputParam("noise_pred", required=True, type_hint=torch.Tensor, description="Predicted target flow."),
        ]

    @property
    def intermediate_outputs(self):
        return [OutputParam.template("latents")]

    @torch.no_grad()
    def __call__(self, components, block_state, i, t):
        dtype = block_state.latents.dtype
        block_state.latents = components.scheduler.step(
            block_state.noise_pred, t, block_state.latents, return_dict=False
        )[0]
        if torch.backends.mps.is_available():
            block_state.latents = block_state.latents.to(dtype)
        return components, block_state


class QwenImage21LoopInpaintStep(ModularPipelineBlocks):
    model_name = "qwenimage21"

    @property
    def description(self):
        return "Restore unmasked source tokens at the next step's noise level."

    @property
    def expected_components(self):
        return [ComponentSpec("scheduler", FlowMatchEulerDiscreteScheduler)]

    @property
    def inputs(self):
        return [
            InputParam.template("latents", required=True),
            InputParam("source_latents", required=True, type_hint=torch.Tensor, description="Clean source tokens."),
            InputParam("initial_noise", required=True, type_hint=torch.Tensor, description="Initial target noise."),
            InputParam("mask", required=True, type_hint=torch.Tensor, description="Latent repaint mask."),
        ]

    @property
    def intermediate_outputs(self):
        return [OutputParam.template("latents")]

    @torch.no_grad()
    def __call__(self, components, block_state, i, t):
        source = block_state.source_latents
        if i + 1 < len(block_state.timesteps):
            source = components.scheduler.scale_noise(
                source, block_state.timesteps[i + 1 : i + 2], block_state.initial_noise
            )
        block_state.latents = (1 - block_state.mask) * source + block_state.mask * block_state.latents
        return components, block_state


# auto_docstring
class QwenImage21DenoiseStep(LoopSequentialPipelineBlocks):
    """
    Denoise target tokens with per-call causal-condition KV caches.

      Components:
          transformer (`QwenImage21Transformer2DModel`) guider (`ClassifierFreeGuidance`) scheduler
          (`FlowMatchEulerDiscreteScheduler`)

      Inputs:
          timesteps (`Tensor`):
              Denoising timesteps.
          num_inference_steps (`int`, *optional*, defaults to 40):
              The number of denoising steps.
          use_kv_cache (`bool`, *optional*, defaults to True):
              Cache step-independent text and condition-image keys and values.
          latents (`Tensor`):
              Pre-generated noisy latents for image generation.
          prompt_embeds (`Tensor`):
              text embeddings used to guide the image generation. Can be generated from text_encoder step.
          negative_prompt_embeds (`Tensor`, *optional*):
              negative text embeddings used to guide the image generation. Can be generated from text_encoder step.
          prompt_embeds_mask (`Tensor`, *optional*):
              mask for the text embeddings. Can be generated from text_encoder step.
          negative_prompt_embeds_mask (`Tensor`, *optional*):
              mask for the negative text embeddings. Can be generated from text_encoder step.
          attention_kwargs (`dict`, *optional*):
              Additional kwargs for attention processors.
          condition_latents (`Tensor`, *optional*):
              Condition image tokens preceding the target.
          img_shapes (`list`):
              Condition and target grid shapes.
          img_mask (`Tensor`):
              Joint positive vision mask.
          negative_img_mask (`Tensor`, *optional*):
              Joint negative vision mask.

      Outputs:
          latents (`Tensor`):
              Denoised latents.
    """

    model_name = "qwenimage21"
    block_classes = [QwenImage21LoopDenoiser, QwenImage21LoopStep]
    block_names = ["denoiser", "scheduler"]

    @property
    def description(self):
        return "Denoise target tokens with per-call causal-condition KV caches."

    @property
    def loop_inputs(self):
        return [
            InputParam("timesteps", required=True, type_hint=torch.Tensor, description="Denoising timesteps."),
            InputParam.template("num_inference_steps", default=40),
            InputParam(
                "use_kv_cache",
                default=True,
                type_hint=bool,
                description="Cache step-independent text and condition-image keys and values.",
            ),
        ]

    @property
    def loop_intermediate_outputs(self):
        return [
            OutputParam("kv_caches", type_hint=dict, description="Temporary KV caches, cleared after generation."),
            OutputParam("cache_enabled", type_hint=bool, description="Whether causal-condition caching is active."),
        ]

    @torch.no_grad()
    def __call__(self, components, state):
        block_state = self.get_block_state(state)
        block_state.kv_caches = {}
        block_state.cache_enabled = block_state.use_kv_cache and components.transformer.config.causal_condition
        try:
            with self.progress_bar(total=len(block_state.timesteps)) as progress_bar:
                for i, t in enumerate(block_state.timesteps):
                    components, block_state = self.loop_step(components, block_state, i=i, t=t)
                    progress_bar.update()
        finally:
            block_state.kv_caches.clear()
            components.guider.set_state(step=0, num_inference_steps=None, timestep=None)
        self.set_block_state(state, block_state)
        return components, state


# auto_docstring
class QwenImage21InpaintDenoiseStep(QwenImage21DenoiseStep):
    """
    Denoise the masked target while preserving source tokens outside the mask.

      Components:
          transformer (`QwenImage21Transformer2DModel`) guider (`ClassifierFreeGuidance`) scheduler
          (`FlowMatchEulerDiscreteScheduler`)

      Inputs:
          timesteps (`Tensor`):
              Denoising timesteps.
          num_inference_steps (`int`, *optional*, defaults to 40):
              The number of denoising steps.
          use_kv_cache (`bool`, *optional*, defaults to True):
              Cache step-independent text and condition-image keys and values.
          latents (`Tensor`):
              Pre-generated noisy latents for image generation.
          prompt_embeds (`Tensor`):
              text embeddings used to guide the image generation. Can be generated from text_encoder step.
          negative_prompt_embeds (`Tensor`, *optional*):
              negative text embeddings used to guide the image generation. Can be generated from text_encoder step.
          prompt_embeds_mask (`Tensor`, *optional*):
              mask for the text embeddings. Can be generated from text_encoder step.
          negative_prompt_embeds_mask (`Tensor`, *optional*):
              mask for the negative text embeddings. Can be generated from text_encoder step.
          attention_kwargs (`dict`, *optional*):
              Additional kwargs for attention processors.
          condition_latents (`Tensor`, *optional*):
              Condition image tokens preceding the target.
          img_shapes (`list`):
              Condition and target grid shapes.
          img_mask (`Tensor`):
              Joint positive vision mask.
          negative_img_mask (`Tensor`, *optional*):
              Joint negative vision mask.
          source_latents (`Tensor`):
              Clean source tokens.
          initial_noise (`Tensor`):
              Initial target noise.
          mask (`Tensor`):
              Latent repaint mask.

      Outputs:
          latents (`Tensor`):
              Denoised latents.
    """

    block_classes = [QwenImage21LoopDenoiser, QwenImage21LoopStep, QwenImage21LoopInpaintStep]
    block_names = ["denoiser", "scheduler", "blend"]

    @property
    def description(self):
        return "Denoise the masked target while preserving source tokens outside the mask."
