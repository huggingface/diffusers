# Copyright 2025 Qwen-Image Team and The HuggingFace Team. All rights reserved.
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

from typing import List, Optional, Union, Tuple, Dict

import numpy as np
import torch

from ...utils import logging

from ..modular_pipeline import ModularPipelineBlocks, PipelineState, BlockState, LoopSequentialPipelineBlocks
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam
from .modular_pipeline import QwenImageModularPipeline

from ...guiders import ClassifierFreeGuidance
from ...configuration_utils import FrozenDict
from ...models import QwenImageTransformer2DModel
from ...schedulers import FlowMatchEulerDiscreteScheduler

logger = logging.get_logger(__name__)


class QwenImageLoopBeforeDenoiser(ModularPipelineBlocks):
    model_name = "qwenimage"

    @property
    def description(self) -> str:
        return (
            "step within the denoising loop that prepares the latent input for the denoiser. "
            "This block should be used to compose the `sub_blocks` attribute of a `LoopSequentialPipelineBlocks` "
            "object (e.g. `QwenImageDenoiseLoopWrapper`)"
        )
    
    @property
    def inputs(self) -> List[InputParam]:
        return [
            InputParam("latents", required=True, type_hint=torch.Tensor, description="The initial latents to use for the denoising process. Can be generated in prepare_latent step."),
        ]

    @torch.no_grad()
    def __call__(self, components: QwenImageModularPipeline, block_state: BlockState, i: int, t: torch.Tensor):
        # one timestep
        block_state.timestep = t.expand(block_state.latents.shape[0]).to(block_state.latents.dtype)
        return components, block_state


class QwenImageLoopDenoiser(ModularPipelineBlocks):
    model_name = "qwenimage"

    @property
    def description(self) -> str:
        return (
            "step within the denoising loop that denoise the latent input for the denoiser. "
            "This block should be used to compose the `sub_blocks` attribute of a `LoopSequentialPipelineBlocks` "
            "object (e.g. `QwenImageDenoiseLoopWrapper`)"
        )
    
    @property
    def expected_components(self) -> List[ComponentSpec]:
        return [
            ComponentSpec(
                "guider",
                ClassifierFreeGuidance,
                config=FrozenDict({"guidance_scale": 4.0}),
                default_creation_method="from_config",
            ),
            ComponentSpec("transformer", QwenImageTransformer2DModel),
        ]

    @property
    def inputs(self) -> List[InputParam]:
        return [
            InputParam("attention_kwargs"),
            InputParam("latents", required=True, type_hint=torch.Tensor, description="The latents to use for the denoising process. Can be generated in prepare_latents step."),
            InputParam("num_inference_steps", required=True, type_hint=int, description="The number of inference steps to use for the denoising process. Can be generated in set_timesteps step."),
            InputParam(kwargs_type="guider_input_fields", description="All coditional model inputs that need to be prepared with guider: e.g. prompt_embeds, negative_prompt_embeds, etc."),
            InputParam("img_shapes", required=True, type_hint=List[Tuple[int, int]], description="The shape of the image latents for RoPE calculation. Can be generated in prepare_additional_inputs step."),
        ]
    
    @torch.no_grad()
    def __call__(self, components: QwenImageModularPipeline, block_state: BlockState, i: int, t: torch.Tensor):
        
        guider_input_fields = {
            "encoder_hidden_states": ("prompt_embeds", "negative_prompt_embeds"),
            "encoder_hidden_states_mask": ("prompt_embeds_mask", "negative_prompt_embeds_mask"),
            "txt_seq_lens": ("txt_seq_lens", "negative_txt_seq_lens"),
        }

        components.guider.set_state(step=i, num_inference_steps=block_state.num_inference_steps, timestep=t)
        guider_state = components.guider.prepare_inputs(block_state, guider_input_fields)

        for guider_state_batch in guider_state:
            components.guider.prepare_models(components.transformer)
            cond_kwargs = guider_state_batch.as_dict()
            cond_kwargs = {k: v for k, v in cond_kwargs.items() if k in guider_input_fields}

            # YiYi TODO: add cache context
            guider_state_batch.noise_pred = components.transformer(
                hidden_states=block_state.latents,
                timestep=block_state.timestep / 1000,
                img_shapes=block_state.img_shapes,
                attention_kwargs=block_state.attention_kwargs,
                return_dict=False,
                **cond_kwargs,
            )[0]

            components.guider.cleanup_models(components.transformer)
        
        guider_output = components.guider(guider_state)
        
        # apply guidance rescale
        pred_cond_norm = torch.norm(guider_output.pred_cond, dim=-1, keepdim=True)
        pred_norm = torch.norm(guider_output.pred, dim=-1, keepdim=True)
        block_state.noise_pred = guider_output.pred * (pred_cond_norm / pred_norm)
        

        return components, block_state

            
            


class QwenImageLoopAfterDenoiser(ModularPipelineBlocks):
    model_name = "qwenimage"

    @property
    def description(self) -> str:
        return (
            "step within the denoising loop that updates the latents. "
            "This block should be used to compose the `sub_blocks` attribute of a `LoopSequentialPipelineBlocks` "
            "object (e.g. `QwenImageDenoiseLoopWrapper`)"
        )
    
    @property
    def expected_components(self) -> List[ComponentSpec]:
        return [
            ComponentSpec("scheduler", FlowMatchEulerDiscreteScheduler),
        ]
    
    @property
    def intermediate_outputs(self) -> List[OutputParam]:
        return [
            OutputParam("latents", type_hint=torch.Tensor, description="The denoised latents."),
        ]

    @torch.no_grad()
    def __call__(self, components: QwenImageModularPipeline, block_state: BlockState, i: int, t: torch.Tensor):

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
        




class QwenImageDenoiseLoopWrapper(LoopSequentialPipelineBlocks):
    model_name = "qwenimage"

    @property
    def description(self) -> str:
        return (
            "Pipeline block that iteratively denoise the latents over `timesteps`. "
            "The specific steps with each iteration can be customized with `sub_blocks` attributes"
        )
    
    @property
    def loop_expected_components(self) -> List[ComponentSpec]:
        return [
            ComponentSpec("scheduler", FlowMatchEulerDiscreteScheduler),
        ]
    
    @property
    def loop_inputs(self) -> List[InputParam]:
        return [
            InputParam("timesteps", required=True, type_hint=torch.Tensor, description="The timesteps to use for the denoising process. Can be generated in set_timesteps step."),
            InputParam("num_inference_steps", required=True, type_hint=int, description="The number of inference steps to use for the denoising process. Can be generated in set_timesteps step."),
        ]

    @torch.no_grad()
    def __call__(self, components: QwenImageModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        block_state.num_warmup_steps = max(len(block_state.timesteps) - block_state.num_inference_steps * components.scheduler.order, 0)

        with self.progress_bar(total=block_state.num_inference_steps) as progress_bar:
            for i, t in enumerate(block_state.timesteps):
                components, block_state = self.loop_step(components, block_state, i=i, t=t)
                if i == len(block_state.timesteps) - 1 or (
                    (i + 1) > block_state.num_warmup_steps and (i + 1) % components.scheduler.order == 0
                ):
                    progress_bar.update()

        self.set_block_state(state, block_state)

        return components, state


# composing the denoising loops
class QwenImageDenoiseStep(QwenImageDenoiseLoopWrapper):
    block_classes = [
        QwenImageLoopBeforeDenoiser,
        QwenImageLoopDenoiser,
        QwenImageLoopAfterDenoiser,
    ]
    block_names = ["before_denoiser", "denoiser", "after_denoiser"]

    @property
    def description(self) -> str:
        return (
            "Denoise step that iteratively denoise the latents. \n"
            "Its loop logic is defined in `QwenImageDenoiseLoopWrapper.__call__` method \n"
            "At each iteration, it runs blocks defined in `sub_blocks` sequencially:\n"
            " - `QwenImageLoopBeforeDenoiser`\n"
            " - `QwenImageLoopDenoiser`\n"
            " - `QwenImageLoopAfterDenoiser`\n"
            "This block supports text2img tasks."
        )