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

from typing import Any, List, Tuple

import torch

from ...models import FluxTransformer2DModel
from ...schedulers import FlowMatchEulerDiscreteScheduler
from ...utils import logging
from ..modular_pipeline import (
    BlockState,
    LoopSequentialPipelineBlocks,
    ModularPipelineBlocks,
    PipelineState,
)
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam
from .modular_pipeline import FluxModularPipeline


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class FluxLoopDenoiser(ModularPipelineBlocks):
    model_name = "flux"

    @property
    def expected_components(self) -> List[ComponentSpec]:
        return [ComponentSpec("transformer", FluxTransformer2DModel)]

    @property
    def description(self) -> str:
        return (
            "Step within the denoising loop that denoise the latents. "
            "This block should be used to compose the `sub_blocks` attribute of a `LoopSequentialPipelineBlocks` "
            "object (e.g. `FluxDenoiseLoopWrapper`)"
        )

    @property
    def inputs(self) -> List[Tuple[str, Any]]:
        return [
            InputParam("joint_attention_kwargs"),
            InputParam(
                "latents",
                required=True,
                type_hint=torch.Tensor,
                description="The initial latents to use for the denoising process. Can be generated in prepare_latent step.",
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
                description="Prompt embeddings",
            ),
            InputParam(
                "pooled_prompt_embeds",
                required=True,
                type_hint=torch.Tensor,
                description="Pooled prompt embeddings",
            ),
            InputParam(
                "txt_ids",
                required=True,
                type_hint=torch.Tensor,
                description="IDs computed from text sequence needed for RoPE",
            ),
            InputParam(
                "img_ids",
                required=True,
                type_hint=torch.Tensor,
                description="IDs computed from image sequence needed for RoPE",
            ),
        ]

    @torch.no_grad()
    def __call__(
        self, components: FluxModularPipeline, block_state: BlockState, i: int, t: torch.Tensor
    ) -> PipelineState:
        noise_pred = components.transformer(
            hidden_states=block_state.latents,
            timestep=t.flatten() / 1000,
            guidance=block_state.guidance,
            encoder_hidden_states=block_state.prompt_embeds,
            pooled_projections=block_state.pooled_prompt_embeds,
            joint_attention_kwargs=block_state.joint_attention_kwargs,
            txt_ids=block_state.txt_ids,
            img_ids=block_state.img_ids,
            return_dict=False,
        )[0]
        block_state.noise_pred = noise_pred

        return components, block_state


class FluxLoopAfterDenoiser(ModularPipelineBlocks):
    model_name = "flux"

    @property
    def expected_components(self) -> List[ComponentSpec]:
        return [ComponentSpec("scheduler", FlowMatchEulerDiscreteScheduler)]

    @property
    def description(self) -> str:
        return (
            "step within the denoising loop that update the latents. "
            "This block should be used to compose the `sub_blocks` attribute of a `LoopSequentialPipelineBlocks` "
            "object (e.g. `FluxDenoiseLoopWrapper`)"
        )

    @property
    def inputs(self) -> List[Tuple[str, Any]]:
        return []

    @property
    def intermediate_inputs(self) -> List[str]:
        return [InputParam("generator")]

    @property
    def intermediate_outputs(self) -> List[OutputParam]:
        return [OutputParam("latents", type_hint=torch.Tensor, description="The denoised latents")]

    @torch.no_grad()
    def __call__(self, components: FluxModularPipeline, block_state: BlockState, i: int, t: torch.Tensor):
        # Perform scheduler step using the predicted output
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


class FluxDenoiseLoopWrapper(LoopSequentialPipelineBlocks):
    model_name = "flux"

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
            ComponentSpec("transformer", FluxTransformer2DModel),
        ]

    @property
    def loop_inputs(self) -> List[InputParam]:
        return [
            InputParam(
                "timesteps",
                required=True,
                type_hint=torch.Tensor,
                description="The timesteps to use for the denoising process. Can be generated in set_timesteps step.",
            ),
            InputParam(
                "num_inference_steps",
                required=True,
                type_hint=int,
                description="The number of inference steps to use for the denoising process. Can be generated in set_timesteps step.",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components: FluxModularPipeline, state: PipelineState) -> PipelineState:
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


class FluxDenoiseStep(FluxDenoiseLoopWrapper):
    block_classes = [FluxLoopDenoiser, FluxLoopAfterDenoiser]
    block_names = ["denoiser", "after_denoiser"]

    @property
    def description(self) -> str:
        return (
            "Denoise step that iteratively denoise the latents. \n"
            "Its loop logic is defined in `FluxDenoiseLoopWrapper.__call__` method \n"
            "At each iteration, it runs blocks defined in `sub_blocks` sequentially:\n"
            " - `FluxLoopDenoiser`\n"
            " - `FluxLoopAfterDenoiser`\n"
            "This block supports both text2image and img2img tasks."
        )
