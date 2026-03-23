from typing import Any

import torch

from ...models.transformers import SD3Transformer2DModel
from ...schedulers import FlowMatchEulerDiscreteScheduler
from ...utils import logging
from ..modular_pipeline import (
    BlockState,
    LoopSequentialPipelineBlocks,
    ModularPipelineBlocks,
    PipelineState,
)
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam
from .modular_pipeline import SD3ModularPipeline

logger = logging.get_logger(__name__)


class SD3LoopDenoiser(ModularPipelineBlocks):
    model_name = "stable-diffusion-3"

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("transformer", SD3Transformer2DModel)]

    @property
    def description(self) -> str:
        return "Step within the denoising loop that denoise the latents."

    @property
    def inputs(self) -> list[tuple[str, Any]]:
        return[
            InputParam("joint_attention_kwargs"),
            InputParam("latents", required=True, type_hint=torch.Tensor),
            InputParam("prompt_embeds", required=True, type_hint=torch.Tensor),
            InputParam("pooled_prompt_embeds", required=True, type_hint=torch.Tensor),
            InputParam("do_classifier_free_guidance", type_hint=bool),
            InputParam("guidance_scale", default=7.0),
            InputParam("skip_guidance_layers", type_hint=list),
            InputParam("skip_layer_guidance_scale", default=2.8),
            InputParam("skip_layer_guidance_stop", default=0.2),
            InputParam("skip_layer_guidance_start", default=0.01),
            InputParam("original_prompt_embeds", type_hint=torch.Tensor),
            InputParam("original_pooled_prompt_embeds", type_hint=torch.Tensor),
            InputParam("num_inference_steps", type_hint=int),
        ]

    @torch.no_grad()
    def __call__(
        self, components: SD3ModularPipeline, block_state: BlockState, i: int, t: torch.Tensor
    ) -> PipelineState:
        latent_model_input = torch.cat([block_state.latents] * 2) if block_state.do_classifier_free_guidance else block_state.latents
        timestep = t.expand(latent_model_input.shape[0])

        noise_pred = components.transformer(
            hidden_states=latent_model_input,
            timestep=timestep,
            encoder_hidden_states=block_state.prompt_embeds,
            pooled_projections=block_state.pooled_prompt_embeds,
            joint_attention_kwargs=getattr(block_state, "joint_attention_kwargs", None),
            return_dict=False,
        )[0]

        if block_state.do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + block_state.guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            should_skip_layers = (
                getattr(block_state, "skip_guidance_layers", None) is not None
                and i > getattr(block_state, "num_inference_steps", 50) * getattr(block_state, "skip_layer_guidance_start", 0.01)
                and i < getattr(block_state, "num_inference_steps", 50) * getattr(block_state, "skip_layer_guidance_stop", 0.2)
            )

            if should_skip_layers:
                timestep_skip = t.expand(block_state.latents.shape[0])
                noise_pred_skip_layers = components.transformer(
                    hidden_states=block_state.latents,
                    timestep=timestep_skip,
                    encoder_hidden_states=block_state.original_prompt_embeds,
                    pooled_projections=block_state.original_pooled_prompt_embeds,
                    joint_attention_kwargs=getattr(block_state, "joint_attention_kwargs", None),
                    return_dict=False,
                    skip_layers=block_state.skip_guidance_layers,
                )[0]
                noise_pred = noise_pred + (noise_pred_text - noise_pred_skip_layers) * getattr(block_state, "skip_layer_guidance_scale", 2.8)

        block_state.noise_pred = noise_pred
        return components, block_state


class SD3LoopAfterDenoiser(ModularPipelineBlocks):
    model_name = "stable-diffusion-3"

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("scheduler", FlowMatchEulerDiscreteScheduler)]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return[OutputParam("latents", type_hint=torch.Tensor)]

    @torch.no_grad()
    def __call__(self, components: SD3ModularPipeline, block_state: BlockState, i: int, t: torch.Tensor):
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


class SD3DenoiseLoopWrapper(LoopSequentialPipelineBlocks):
    model_name = "stable-diffusion-3"

    @property
    def loop_expected_components(self) -> list[ComponentSpec]:
        return[
            ComponentSpec("scheduler", FlowMatchEulerDiscreteScheduler),
            ComponentSpec("transformer", SD3Transformer2DModel),
        ]

    @property
    def loop_inputs(self) -> list[InputParam]:
        return[
            InputParam("timesteps", required=True, type_hint=torch.Tensor),
            InputParam("num_inference_steps", required=True, type_hint=int),
        ]

    @torch.no_grad()
    def __call__(self, components: SD3ModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        block_state.num_warmup_steps = max(len(block_state.timesteps) - block_state.num_inference_steps * components.scheduler.order, 0)
        
        with self.progress_bar(total=block_state.num_inference_steps) as progress_bar:
            for i, t in enumerate(block_state.timesteps):
                components, block_state = self.loop_step(components, block_state, i=i, t=t)
                if i == len(block_state.timesteps) - 1 or ((i + 1) > block_state.num_warmup_steps and (i + 1) % components.scheduler.order == 0):
                    progress_bar.update()

        self.set_block_state(state, block_state)
        return components, state


class SD3DenoiseStep(SD3DenoiseLoopWrapper):
    block_classes = [SD3LoopDenoiser, SD3LoopAfterDenoiser]
    block_names = ["denoiser", "after_denoiser"]