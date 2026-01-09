# Copyright 2025 Alibaba Z-Image Team and The HuggingFace Team. All rights reserved.
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

from typing import Any, Dict, List, Tuple

import torch

from ...configuration_utils import FrozenDict
from ...guiders import ClassifierFreeGuidance
from ...models import ZImageTransformer2DModel
from ...schedulers import FlowMatchEulerDiscreteScheduler
from ...utils import logging
from ..modular_pipeline import (
    BlockState,
    LoopSequentialPipelineBlocks,
    ModularPipelineBlocks,
    PipelineState,
)
from ..modular_pipeline_utils import ComponentSpec, InputParam
from .modular_pipeline import ZImageModularPipeline


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class ZImageLoopBeforeDenoiser(ModularPipelineBlocks):
    model_name = "z-image"

    @property
    def description(self) -> str:
        return (
            "step within the denoising loop that prepares the latent input for the denoiser. "
            "This block should be used to compose the `sub_blocks` attribute of a `LoopSequentialPipelineBlocks` "
            "object (e.g. `ZImageDenoiseLoopWrapper`)"
        )

    @property
    def inputs(self) -> List[InputParam]:
        return [
            InputParam(
                "latents",
                required=True,
                type_hint=torch.Tensor,
                description="The initial latents to use for the denoising process. Can be generated in prepare_latent step.",
            ),
            InputParam(
                "dtype",
                required=True,
                type_hint=torch.dtype,
                description="The dtype of the model inputs. Can be generated in input step.",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components: ZImageModularPipeline, block_state: BlockState, i: int, t: torch.Tensor):
        latents = block_state.latents.unsqueeze(2).to(
            block_state.dtype
        )  # [batch_size, num_channels, 1, height, width]
        block_state.latent_model_input = list(latents.unbind(dim=0))  # list of [num_channels, 1, height, width]

        timestep = t.expand(latents.shape[0]).to(block_state.dtype)
        timestep = (1000 - timestep) / 1000
        block_state.timestep = timestep
        return components, block_state


class ZImageLoopDenoiser(ModularPipelineBlocks):
    model_name = "z-image"

    def __init__(
        self,
        guider_input_fields: Dict[str, Any] = {"cap_feats": ("prompt_embeds", "negative_prompt_embeds")},
    ):
        """Initialize a denoiser block that calls the denoiser model. This block is used in Z-Image.

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
        if not isinstance(guider_input_fields, dict):
            raise ValueError(f"guider_input_fields must be a dictionary but is {type(guider_input_fields)}")
        self._guider_input_fields = guider_input_fields
        super().__init__()

    @property
    def expected_components(self) -> List[ComponentSpec]:
        return [
            ComponentSpec(
                "guider",
                ClassifierFreeGuidance,
                config=FrozenDict({"guidance_scale": 5.0, "enabled": False}),
                default_creation_method="from_config",
            ),
            ComponentSpec("transformer", ZImageTransformer2DModel),
        ]

    @property
    def description(self) -> str:
        return (
            "Step within the denoising loop that denoise the latents with guidance. "
            "This block should be used to compose the `sub_blocks` attribute of a `LoopSequentialPipelineBlocks` "
            "object (e.g. `ZImageDenoiseLoopWrapper`)"
        )

    @property
    def inputs(self) -> List[Tuple[str, Any]]:
        inputs = [
            InputParam(
                "num_inference_steps",
                required=True,
                type_hint=int,
                description="The number of inference steps to use for the denoising process. Can be generated in set_timesteps step.",
            ),
            InputParam(
                kwargs_type="denoiser_input_fields",
                description="conditional model inputs for the denoiser: e.g. prompt_embeds, negative_prompt_embeds, etc.",
            ),
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
        self, components: ZImageModularPipeline, block_state: BlockState, i: int, t: torch.Tensor
    ) -> PipelineState:
        components.guider.set_state(step=i, num_inference_steps=block_state.num_inference_steps, timestep=t)

        # The guider splits model inputs into separate batches for conditional/unconditional predictions.
        # For CFG with guider_inputs = {"encoder_hidden_states": (prompt_embeds, negative_prompt_embeds)}:
        # you will get a guider_state with two batches:
        #   guider_state = [
        #       {"encoder_hidden_states": prompt_embeds, "__guidance_identifier__": "pred_cond"},      # conditional batch
        #       {"encoder_hidden_states": negative_prompt_embeds, "__guidance_identifier__": "pred_uncond"},  # unconditional batch
        #   ]
        # Other guidance methods may return 1 batch (no guidance) or 3+ batches (e.g., PAG, APG).
        guider_state = components.guider.prepare_inputs_from_block_state(block_state, self._guider_input_fields)

        # run the denoiser for each guidance batch
        for guider_state_batch in guider_state:
            components.guider.prepare_models(components.transformer)
            cond_kwargs = guider_state_batch.as_dict()

            def _convert_dtype(v, dtype):
                if isinstance(v, torch.Tensor):
                    return v.to(dtype)
                elif isinstance(v, list):
                    return [_convert_dtype(t, dtype) for t in v]
                return v

            cond_kwargs = {
                k: _convert_dtype(v, block_state.dtype)
                for k, v in cond_kwargs.items()
                if k in self._guider_input_fields.keys()
            }

            # Predict the noise residual
            # store the noise_pred in guider_state_batch so that we can apply guidance across all batches
            model_out_list = components.transformer(
                x=block_state.latent_model_input,
                t=block_state.timestep,
                return_dict=False,
                **cond_kwargs,
            )[0]
            noise_pred = torch.stack(model_out_list, dim=0).squeeze(2)
            guider_state_batch.noise_pred = -noise_pred
            components.guider.cleanup_models(components.transformer)

        # Perform guidance
        block_state.noise_pred = components.guider(guider_state)[0]

        return components, block_state


class ZImageLoopAfterDenoiser(ModularPipelineBlocks):
    model_name = "z-image"

    @property
    def expected_components(self) -> List[ComponentSpec]:
        return [
            ComponentSpec("scheduler", FlowMatchEulerDiscreteScheduler),
        ]

    @property
    def description(self) -> str:
        return (
            "step within the denoising loop that update the latents. "
            "This block should be used to compose the `sub_blocks` attribute of a `LoopSequentialPipelineBlocks` "
            "object (e.g. `ZImageDenoiseLoopWrapper`)"
        )

    @torch.no_grad()
    def __call__(self, components: ZImageModularPipeline, block_state: BlockState, i: int, t: torch.Tensor):
        # Perform scheduler step using the predicted output
        latents_dtype = block_state.latents.dtype
        block_state.latents = components.scheduler.step(
            block_state.noise_pred.float(),
            t,
            block_state.latents.float(),
            return_dict=False,
        )[0]

        if block_state.latents.dtype != latents_dtype:
            block_state.latents = block_state.latents.to(latents_dtype)

        return components, block_state


class ZImageDenoiseLoopWrapper(LoopSequentialPipelineBlocks):
    model_name = "z-image"

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
    def __call__(self, components: ZImageModularPipeline, state: PipelineState) -> PipelineState:
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


class ZImageDenoiseStep(ZImageDenoiseLoopWrapper):
    block_classes = [
        ZImageLoopBeforeDenoiser,
        ZImageLoopDenoiser(
            guider_input_fields={
                "cap_feats": ("prompt_embeds", "negative_prompt_embeds"),
            }
        ),
        ZImageLoopAfterDenoiser,
    ]
    block_names = ["before_denoiser", "denoiser", "after_denoiser"]

    @property
    def description(self) -> str:
        return (
            "Denoise step that iteratively denoise the latents. \n"
            "Its loop logic is defined in `ZImageDenoiseLoopWrapper.__call__` method \n"
            "At each iteration, it runs blocks defined in `sub_blocks` sequentially:\n"
            " - `ZImageLoopBeforeDenoiser`\n"
            " - `ZImageLoopDenoiser`\n"
            " - `ZImageLoopAfterDenoiser`\n"
            "This block supports text-to-image and image-to-image tasks for Z-Image."
        )
