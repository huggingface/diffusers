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

from typing import List, Tuple

import torch

from ...configuration_utils import FrozenDict
from ...guiders import ClassifierFreeGuidance
from ...models import QwenImageControlNetModel, QwenImageTransformer2DModel
from ...schedulers import FlowMatchEulerDiscreteScheduler
from ...utils import logging
from ..modular_pipeline import BlockState, LoopSequentialPipelineBlocks, ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam
from .modular_pipeline import QwenImageModularPipeline


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
            InputParam(
                "latents",
                required=True,
                type_hint=torch.Tensor,
                description="The initial latents to use for the denoising process. Can be generated in prepare_latent step.",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components: QwenImageModularPipeline, block_state: BlockState, i: int, t: torch.Tensor):
        # one timestep
        block_state.timestep = t.expand(block_state.latents.shape[0]).to(block_state.latents.dtype)
        block_state.latent_model_input = block_state.latents
        return components, block_state


class QwenImageEditLoopBeforeDenoiser(ModularPipelineBlocks):
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
            InputParam(
                "latents",
                required=True,
                type_hint=torch.Tensor,
                description="The initial latents to use for the denoising process. Can be generated in prepare_latent step.",
            ),
            InputParam(
                "image_latents",
                required=True,
                type_hint=torch.Tensor,
                description="The initial image latents to use for the denoising process. Can be encoded in vae_encoder step and packed in prepare_image_latents step.",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components: QwenImageModularPipeline, block_state: BlockState, i: int, t: torch.Tensor):
        # one timestep

        block_state.latent_model_input = torch.cat([block_state.latents, block_state.image_latents], dim=1)
        block_state.timestep = t.expand(block_state.latents.shape[0]).to(block_state.latents.dtype)
        return components, block_state


class QwenImageLoopBeforeDenoiserControlNet(ModularPipelineBlocks):
    model_name = "qwenimage"

    @property
    def expected_components(self) -> List[ComponentSpec]:
        return [
            ComponentSpec(
                "guider",
                ClassifierFreeGuidance,
                config=FrozenDict({"guidance_scale": 4.0}),
                default_creation_method="from_config",
            ),
            ComponentSpec("controlnet", QwenImageControlNetModel),
        ]

    @property
    def description(self) -> str:
        return (
            "step within the denoising loop that runs the controlnet before the denoiser. "
            "This block should be used to compose the `sub_blocks` attribute of a `LoopSequentialPipelineBlocks` "
            "object (e.g. `QwenImageDenoiseLoopWrapper`)"
        )

    @property
    def inputs(self) -> List[InputParam]:
        return [
            InputParam(
                "control_image_latents",
                required=True,
                type_hint=torch.Tensor,
                description="The control image to use for the denoising process. Can be generated in prepare_controlnet_inputs step.",
            ),
            InputParam(
                "controlnet_conditioning_scale",
                type_hint=float,
                description="The controlnet conditioning scale value to use for the denoising process. Can be generated in prepare_controlnet_inputs step.",
            ),
            InputParam(
                "controlnet_keep",
                required=True,
                type_hint=List[float],
                description="The controlnet keep values to use for the denoising process. Can be generated in prepare_controlnet_inputs step.",
            ),
            InputParam(
                "num_inference_steps",
                required=True,
                type_hint=int,
                description="The number of inference steps to use for the denoising process. Can be generated in set_timesteps step.",
            ),
            InputParam(
                kwargs_type="denoiser_input_fields",
                description=(
                    "All conditional model inputs for the denoiser. "
                    "It should contain prompt_embeds/negative_prompt_embeds, txt_seq_lens/negative_txt_seq_lens."
                ),
            ),
        ]

    @torch.no_grad()
    def __call__(self, components: QwenImageModularPipeline, block_state: BlockState, i: int, t: int):
        # cond_scale for the timestep (controlnet input)
        if isinstance(block_state.controlnet_keep[i], list):
            block_state.cond_scale = [
                c * s for c, s in zip(block_state.controlnet_conditioning_scale, block_state.controlnet_keep[i])
            ]
        else:
            controlnet_cond_scale = block_state.controlnet_conditioning_scale
            if isinstance(controlnet_cond_scale, list):
                controlnet_cond_scale = controlnet_cond_scale[0]
            block_state.cond_scale = controlnet_cond_scale * block_state.controlnet_keep[i]

        # run controlnet for the guidance batch
        controlnet_block_samples = components.controlnet(
            hidden_states=block_state.latent_model_input,
            controlnet_cond=block_state.control_image_latents,
            conditioning_scale=block_state.cond_scale,
            timestep=block_state.timestep / 1000,
            img_shapes=block_state.img_shapes,
            encoder_hidden_states=block_state.prompt_embeds,
            encoder_hidden_states_mask=block_state.prompt_embeds_mask,
            txt_seq_lens=block_state.txt_seq_lens,
            return_dict=False,
        )

        block_state.additional_cond_kwargs["controlnet_block_samples"] = controlnet_block_samples

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
            InputParam(
                "latents",
                required=True,
                type_hint=torch.Tensor,
                description="The latents to use for the denoising process. Can be generated in prepare_latents step.",
            ),
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
            InputParam(
                "img_shapes",
                required=True,
                type_hint=List[Tuple[int, int]],
                description="The shape of the image latents for RoPE calculation. Can be generated in prepare_additional_inputs step.",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components: QwenImageModularPipeline, block_state: BlockState, i: int, t: torch.Tensor):
        guider_inputs = {
            "encoder_hidden_states": (
                getattr(block_state, "prompt_embeds", None),
                getattr(block_state, "negative_prompt_embeds", None),
            ),
            "encoder_hidden_states_mask": (
                getattr(block_state, "prompt_embeds_mask", None),
                getattr(block_state, "negative_prompt_embeds_mask", None),
            ),
            "txt_seq_lens": (
                getattr(block_state, "txt_seq_lens", None),
                getattr(block_state, "negative_txt_seq_lens", None),
            ),
        }

        components.guider.set_state(step=i, num_inference_steps=block_state.num_inference_steps, timestep=t)
        guider_state = components.guider.prepare_inputs(guider_inputs)

        for guider_state_batch in guider_state:
            components.guider.prepare_models(components.transformer)
            cond_kwargs = {input_name: getattr(guider_state_batch, input_name) for input_name in guider_inputs.keys()}

            # YiYi TODO: add cache context
            guider_state_batch.noise_pred = components.transformer(
                hidden_states=block_state.latent_model_input,
                timestep=block_state.timestep / 1000,
                img_shapes=block_state.img_shapes,
                attention_kwargs=block_state.attention_kwargs,
                return_dict=False,
                **cond_kwargs,
                **block_state.additional_cond_kwargs,
            )[0]

            components.guider.cleanup_models(components.transformer)

        guider_output = components.guider(guider_state)

        # apply guidance rescale
        pred_cond_norm = torch.norm(guider_output.pred_cond, dim=-1, keepdim=True)
        pred_norm = torch.norm(guider_output.pred, dim=-1, keepdim=True)
        block_state.noise_pred = guider_output.pred * (pred_cond_norm / pred_norm)

        return components, block_state


class QwenImageEditLoopDenoiser(ModularPipelineBlocks):
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
            InputParam(
                "latents",
                required=True,
                type_hint=torch.Tensor,
                description="The latents to use for the denoising process. Can be generated in prepare_latents step.",
            ),
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
            InputParam(
                "img_shapes",
                required=True,
                type_hint=List[Tuple[int, int]],
                description="The shape of the image latents for RoPE calculation. Can be generated in prepare_additional_inputs step.",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components: QwenImageModularPipeline, block_state: BlockState, i: int, t: torch.Tensor):
        guider_inputs = {
            "encoder_hidden_states": (
                getattr(block_state, "prompt_embeds", None),
                getattr(block_state, "negative_prompt_embeds", None),
            ),
            "encoder_hidden_states_mask": (
                getattr(block_state, "prompt_embeds_mask", None),
                getattr(block_state, "negative_prompt_embeds_mask", None),
            ),
            "txt_seq_lens": (
                getattr(block_state, "txt_seq_lens", None),
                getattr(block_state, "negative_txt_seq_lens", None),
            ),
        }

        components.guider.set_state(step=i, num_inference_steps=block_state.num_inference_steps, timestep=t)
        guider_state = components.guider.prepare_inputs(guider_inputs)

        for guider_state_batch in guider_state:
            components.guider.prepare_models(components.transformer)
            cond_kwargs = {input_name: getattr(guider_state_batch, input_name) for input_name in guider_inputs.keys()}

            # YiYi TODO: add cache context
            guider_state_batch.noise_pred = components.transformer(
                hidden_states=block_state.latent_model_input,
                timestep=block_state.timestep / 1000,
                img_shapes=block_state.img_shapes,
                attention_kwargs=block_state.attention_kwargs,
                return_dict=False,
                **cond_kwargs,
                **block_state.additional_cond_kwargs,
            )[0]

            components.guider.cleanup_models(components.transformer)

        guider_output = components.guider(guider_state)

        pred = guider_output.pred[:, : block_state.latents.size(1)]
        pred_cond = guider_output.pred_cond[:, : block_state.latents.size(1)]

        # apply guidance rescale
        pred_cond_norm = torch.norm(pred_cond, dim=-1, keepdim=True)
        pred_norm = torch.norm(pred, dim=-1, keepdim=True)
        block_state.noise_pred = pred * (pred_cond_norm / pred_norm)

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


class QwenImageLoopAfterDenoiserInpaint(ModularPipelineBlocks):
    model_name = "qwenimage"

    @property
    def description(self) -> str:
        return (
            "step within the denoising loop that updates the latents using mask and image_latents for inpainting. "
            "This block should be used to compose the `sub_blocks` attribute of a `LoopSequentialPipelineBlocks` "
            "object (e.g. `QwenImageDenoiseLoopWrapper`)"
        )

    @property
    def inputs(self) -> List[InputParam]:
        return [
            InputParam(
                "mask",
                required=True,
                type_hint=torch.Tensor,
                description="The mask to use for the inpainting process. Can be generated in inpaint prepare latents step.",
            ),
            InputParam(
                "image_latents",
                required=True,
                type_hint=torch.Tensor,
                description="The image latents to use for the inpainting process. Can be generated in inpaint prepare latents step.",
            ),
            InputParam(
                "initial_noise",
                required=True,
                type_hint=torch.Tensor,
                description="The initial noise to use for the inpainting process. Can be generated in inpaint prepare latents step.",
            ),
            InputParam(
                "timesteps",
                required=True,
                type_hint=torch.Tensor,
                description="The timesteps to use for the denoising process. Can be generated in set_timesteps step.",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components: QwenImageModularPipeline, block_state: BlockState, i: int, t: torch.Tensor):
        block_state.init_latents_proper = block_state.image_latents
        if i < len(block_state.timesteps) - 1:
            block_state.noise_timestep = block_state.timesteps[i + 1]
            block_state.init_latents_proper = components.scheduler.scale_noise(
                block_state.init_latents_proper, torch.tensor([block_state.noise_timestep]), block_state.initial_noise
            )

        block_state.latents = (
            1 - block_state.mask
        ) * block_state.init_latents_proper + block_state.mask * block_state.latents

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
    def __call__(self, components: QwenImageModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        block_state.num_warmup_steps = max(
            len(block_state.timesteps) - block_state.num_inference_steps * components.scheduler.order, 0
        )

        block_state.additional_cond_kwargs = {}

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
            "This block supports text2image and image2image tasks for QwenImage."
        )


# composing the inpainting denoising loops
class QwenImageInpaintDenoiseStep(QwenImageDenoiseLoopWrapper):
    block_classes = [
        QwenImageLoopBeforeDenoiser,
        QwenImageLoopDenoiser,
        QwenImageLoopAfterDenoiser,
        QwenImageLoopAfterDenoiserInpaint,
    ]
    block_names = ["before_denoiser", "denoiser", "after_denoiser", "after_denoiser_inpaint"]

    @property
    def description(self) -> str:
        return (
            "Denoise step that iteratively denoise the latents. \n"
            "Its loop logic is defined in `QwenImageDenoiseLoopWrapper.__call__` method \n"
            "At each iteration, it runs blocks defined in `sub_blocks` sequencially:\n"
            " - `QwenImageLoopBeforeDenoiser`\n"
            " - `QwenImageLoopDenoiser`\n"
            " - `QwenImageLoopAfterDenoiser`\n"
            " - `QwenImageLoopAfterDenoiserInpaint`\n"
            "This block supports inpainting tasks for QwenImage."
        )


# composing the controlnet denoising loops
class QwenImageControlNetDenoiseStep(QwenImageDenoiseLoopWrapper):
    block_classes = [
        QwenImageLoopBeforeDenoiser,
        QwenImageLoopBeforeDenoiserControlNet,
        QwenImageLoopDenoiser,
        QwenImageLoopAfterDenoiser,
    ]
    block_names = ["before_denoiser", "before_denoiser_controlnet", "denoiser", "after_denoiser"]

    @property
    def description(self) -> str:
        return (
            "Denoise step that iteratively denoise the latents. \n"
            "Its loop logic is defined in `QwenImageDenoiseLoopWrapper.__call__` method \n"
            "At each iteration, it runs blocks defined in `sub_blocks` sequencially:\n"
            " - `QwenImageLoopBeforeDenoiser`\n"
            " - `QwenImageLoopBeforeDenoiserControlNet`\n"
            " - `QwenImageLoopDenoiser`\n"
            " - `QwenImageLoopAfterDenoiser`\n"
            "This block supports text2img/img2img tasks with controlnet for QwenImage."
        )


# composing the controlnet denoising loops
class QwenImageInpaintControlNetDenoiseStep(QwenImageDenoiseLoopWrapper):
    block_classes = [
        QwenImageLoopBeforeDenoiser,
        QwenImageLoopBeforeDenoiserControlNet,
        QwenImageLoopDenoiser,
        QwenImageLoopAfterDenoiser,
        QwenImageLoopAfterDenoiserInpaint,
    ]
    block_names = [
        "before_denoiser",
        "before_denoiser_controlnet",
        "denoiser",
        "after_denoiser",
        "after_denoiser_inpaint",
    ]

    @property
    def description(self) -> str:
        return (
            "Denoise step that iteratively denoise the latents. \n"
            "Its loop logic is defined in `QwenImageDenoiseLoopWrapper.__call__` method \n"
            "At each iteration, it runs blocks defined in `sub_blocks` sequencially:\n"
            " - `QwenImageLoopBeforeDenoiser`\n"
            " - `QwenImageLoopBeforeDenoiserControlNet`\n"
            " - `QwenImageLoopDenoiser`\n"
            " - `QwenImageLoopAfterDenoiser`\n"
            " - `QwenImageLoopAfterDenoiserInpaint`\n"
            "This block supports inpainting tasks with controlnet for QwenImage."
        )


# composing the denoising loops
class QwenImageEditDenoiseStep(QwenImageDenoiseLoopWrapper):
    block_classes = [
        QwenImageEditLoopBeforeDenoiser,
        QwenImageEditLoopDenoiser,
        QwenImageLoopAfterDenoiser,
    ]
    block_names = ["before_denoiser", "denoiser", "after_denoiser"]

    @property
    def description(self) -> str:
        return (
            "Denoise step that iteratively denoise the latents. \n"
            "Its loop logic is defined in `QwenImageDenoiseLoopWrapper.__call__` method \n"
            "At each iteration, it runs blocks defined in `sub_blocks` sequencially:\n"
            " - `QwenImageEditLoopBeforeDenoiser`\n"
            " - `QwenImageEditLoopDenoiser`\n"
            " - `QwenImageLoopAfterDenoiser`\n"
            "This block supports QwenImage Edit."
        )


class QwenImageEditInpaintDenoiseStep(QwenImageDenoiseLoopWrapper):
    block_classes = [
        QwenImageEditLoopBeforeDenoiser,
        QwenImageEditLoopDenoiser,
        QwenImageLoopAfterDenoiser,
        QwenImageLoopAfterDenoiserInpaint,
    ]
    block_names = ["before_denoiser", "denoiser", "after_denoiser", "after_denoiser_inpaint"]

    @property
    def description(self) -> str:
        return (
            "Denoise step that iteratively denoise the latents. \n"
            "Its loop logic is defined in `QwenImageDenoiseLoopWrapper.__call__` method \n"
            "At each iteration, it runs blocks defined in `sub_blocks` sequencially:\n"
            " - `QwenImageEditLoopBeforeDenoiser`\n"
            " - `QwenImageEditLoopDenoiser`\n"
            " - `QwenImageLoopAfterDenoiser`\n"
            " - `QwenImageLoopAfterDenoiserInpaint`\n"
            "This block supports inpainting tasks for QwenImage Edit."
        )
