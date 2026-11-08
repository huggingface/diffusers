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

from typing import Any

import torch

from ...configuration_utils import FrozenDict
from ...guiders import ClassifierFreeGuidance
from ...models import ChromaTransformer2DModel
from ...schedulers import FlowMatchEulerDiscreteScheduler
from ...utils import logging
from ..modular_pipeline import (
    BlockState,
    LoopSequentialPipelineBlocks,
    ModularPipelineBlocks,
    PipelineState,
)
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam
from .modular_pipeline import ChromaModularPipeline


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class ChromaLoopDenoiser(ModularPipelineBlocks):
    model_name = "chroma"

    def __init__(
        self,
        guider_input_fields: dict[str, Any] | None = None,
    ):
        """Initialize a denoiser block that calls the denoiser model with guidance. This block is used in Chroma.

        Args:
            guider_input_fields: A dictionary that maps each argument expected by the denoiser model
                (for example, "encoder_hidden_states") to data stored on 'block_state'. The value can be either:

                - A tuple of strings. For instance, {"encoder_hidden_states": ("prompt_embeds",
                  "negative_prompt_embeds")} tells the guider to read `block_state.prompt_embeds` and
                  `block_state.negative_prompt_embeds` and pass them as the conditional and unconditional batches of
                  'encoder_hidden_states'.
                - A string. For example, {"encoder_hidden_image": "image_embeds"} makes the guider forward
                  `block_state.image_embeds` for both conditional and unconditional batches.
        """
        if guider_input_fields is None:
            guider_input_fields = {
                "encoder_hidden_states": ("prompt_embeds", "negative_prompt_embeds"),
                "attention_mask": ("attention_mask", "negative_attention_mask"),
            }
        if not isinstance(guider_input_fields, dict):
            raise ValueError(f"`guider_input_fields` must be a dictionary but is {type(guider_input_fields)}")
        self._guider_input_fields = guider_input_fields
        super().__init__()

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec(
                "guider",
                ClassifierFreeGuidance,
                config=FrozenDict({"guidance_scale": 5.0}),
                default_creation_method="from_config",
            ),
            ComponentSpec("transformer", ChromaTransformer2DModel),
        ]

    @property
    def description(self) -> str:
        return (
            "Step within the denoising loop that denoise the latents with guidance. "
            "This block should be used to compose the `sub_blocks` attribute of a `LoopSequentialPipelineBlocks` "
            "object (e.g. `ChromaDenoiseLoopWrapper`)"
        )

    @property
    def inputs(self) -> list[InputParam]:
        inputs = [
            InputParam(
                "latents",
                required=True,
                type_hint=torch.Tensor,
                description="The initial latents to use for the denoising process. Can be generated in prepare_latents step.",
            ),
            InputParam(
                "num_inference_steps",
                required=True,
                type_hint=int,
                description="The number of inference steps to use for the denoising process. Can be generated in set_timesteps step.",
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
                description="IDs computed from latent sequence needed for RoPE",
            ),
            InputParam(
                "joint_attention_kwargs",
                type_hint=dict,
                description="Additional kwargs passed along to the attention processors.",
            ),
            InputParam.template("denoiser_input_fields"),
        ]

        guider_input_names = []
        uncond_guider_input_names = []
        for value in self._guider_input_fields.values():
            if isinstance(value, tuple):
                guider_input_names.append(value[0])
                uncond_guider_input_names.append(value[1])
            else:
                guider_input_names.append(value)

        for name in guider_input_names:
            inputs.append(InputParam(name=name, required=True))
        for name in uncond_guider_input_names:
            inputs.append(InputParam(name=name))
        return inputs

    @torch.no_grad()
    def __call__(
        self, components: ChromaModularPipeline, block_state: BlockState, i: int, t: torch.Tensor
    ) -> PipelineState:
        components.guider.set_state(step=i, num_inference_steps=block_state.num_inference_steps, timestep=t)
        guider_state = components.guider.prepare_inputs_from_block_state(block_state, self._guider_input_fields)

        latents = block_state.latents
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timestep = t.expand(latents.shape[0]).to(latents.dtype)

        for guider_state_batch in guider_state:
            components.guider.prepare_models(components.transformer)
            cond_kwargs = {key: getattr(guider_state_batch, key) for key in self._guider_input_fields.keys()}

            guider_state_batch.noise_pred = components.transformer(
                hidden_states=latents,
                timestep=timestep / 1000,
                txt_ids=block_state.txt_ids,
                img_ids=block_state.img_ids,
                joint_attention_kwargs=block_state.joint_attention_kwargs,
                return_dict=False,
                **cond_kwargs,
            )[0]
            components.guider.cleanup_models(components.transformer)

        block_state.noise_pred = components.guider(guider_state)[0]

        return components, block_state


class ChromaLoopAfterDenoiser(ModularPipelineBlocks):
    model_name = "chroma"

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("scheduler", FlowMatchEulerDiscreteScheduler)]

    @property
    def description(self) -> str:
        return (
            "step within the denoising loop that update the latents. "
            "This block should be used to compose the `sub_blocks` attribute of a `LoopSequentialPipelineBlocks` "
            "object (e.g. `ChromaDenoiseLoopWrapper`)"
        )

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [OutputParam("latents", type_hint=torch.Tensor, description="The denoised latents")]

    @torch.no_grad()
    def __call__(self, components: ChromaModularPipeline, block_state: BlockState, i: int, t: torch.Tensor):
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


class ChromaDenoiseLoopWrapper(LoopSequentialPipelineBlocks):
    model_name = "chroma"

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
            ComponentSpec("transformer", ChromaTransformer2DModel),
        ]

    @property
    def loop_inputs(self) -> list[InputParam]:
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
    def __call__(self, components: ChromaModularPipeline, state: PipelineState) -> PipelineState:
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


class ChromaDenoiseStep(ChromaDenoiseLoopWrapper):
    block_classes = [
        ChromaLoopDenoiser(
            guider_input_fields={
                "encoder_hidden_states": ("prompt_embeds", "negative_prompt_embeds"),
                "attention_mask": ("attention_mask", "negative_attention_mask"),
            }
        ),
        ChromaLoopAfterDenoiser,
    ]
    block_names = ["denoiser", "after_denoiser"]

    @property
    def description(self) -> str:
        return (
            "Denoise step that iteratively denoise the latents. \n"
            "Its loop logic is defined in `ChromaDenoiseLoopWrapper.__call__` method \n"
            "At each iteration, it runs blocks defined in `sub_blocks` sequentially:\n"
            " - `ChromaLoopDenoiser`\n"
            " - `ChromaLoopAfterDenoiser`\n"
            "This block supports the text2image task."
        )
