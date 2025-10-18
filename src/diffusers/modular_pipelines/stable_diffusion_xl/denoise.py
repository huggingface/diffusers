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

import inspect
from typing import Any, List, Optional, Tuple

import torch

from ...configuration_utils import FrozenDict
from ...guiders import ClassifierFreeGuidance
from ...models import ControlNetModel, UNet2DConditionModel
from ...schedulers import EulerDiscreteScheduler
from ...utils import logging
from ..modular_pipeline import (
    BlockState,
    LoopSequentialPipelineBlocks,
    ModularPipelineBlocks,
    PipelineState,
)
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam
from .modular_pipeline import StableDiffusionXLModularPipeline


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# YiYi experimenting composible denoise loop
# loop step (1): prepare latent input for denoiser
class StableDiffusionXLLoopBeforeDenoiser(ModularPipelineBlocks):
    model_name = "stable-diffusion-xl"

    @property
    def expected_components(self) -> List[ComponentSpec]:
        return [
            ComponentSpec("scheduler", EulerDiscreteScheduler),
        ]

    @property
    def description(self) -> str:
        return (
            "step within the denoising loop that prepare the latent input for the denoiser. "
            "This block should be used to compose the `sub_blocks` attribute of a `LoopSequentialPipelineBlocks` "
            "object (e.g. `StableDiffusionXLDenoiseLoopWrapper`)"
        )

    @property
    def inputs(self) -> List[str]:
        return [
            InputParam(
                "latents",
                required=True,
                type_hint=torch.Tensor,
                description="The initial latents to use for the denoising process. Can be generated in prepare_latent step.",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components: StableDiffusionXLModularPipeline, block_state: BlockState, i: int, t: int):
        block_state.scaled_latents = components.scheduler.scale_model_input(block_state.latents, t)

        return components, block_state


# loop step (1): prepare latent input for denoiser (with inpainting)
class StableDiffusionXLInpaintLoopBeforeDenoiser(ModularPipelineBlocks):
    model_name = "stable-diffusion-xl"

    @property
    def expected_components(self) -> List[ComponentSpec]:
        return [
            ComponentSpec("scheduler", EulerDiscreteScheduler),
            ComponentSpec("unet", UNet2DConditionModel),
        ]

    @property
    def description(self) -> str:
        return (
            "step within the denoising loop that prepare the latent input for the denoiser (for inpainting workflow only). "
            "This block should be used to compose the `sub_blocks` attribute of a `LoopSequentialPipelineBlocks` object"
        )

    @property
    def inputs(self) -> List[str]:
        return [
            InputParam(
                "latents",
                required=True,
                type_hint=torch.Tensor,
                description="The initial latents to use for the denoising process. Can be generated in prepare_latent step.",
            ),
            InputParam(
                "mask",
                type_hint=Optional[torch.Tensor],
                description="The mask to use for the denoising process, for inpainting task only. Can be generated in vae_encode or prepare_latent step.",
            ),
            InputParam(
                "masked_image_latents",
                type_hint=Optional[torch.Tensor],
                description="The masked image latents to use for the denoising process, for inpainting task only. Can be generated in vae_encode or prepare_latent step.",
            ),
        ]

    @staticmethod
    def check_inputs(components, block_state):
        num_channels_unet = components.num_channels_unet
        if num_channels_unet == 9:
            # default case for stable-diffusion-v1-5/stable-diffusion-inpainting
            if block_state.mask is None or block_state.masked_image_latents is None:
                raise ValueError("mask and masked_image_latents must be provided for inpainting-specific Unet")
            num_channels_latents = block_state.latents.shape[1]
            num_channels_mask = block_state.mask.shape[1]
            num_channels_masked_image = block_state.masked_image_latents.shape[1]
            if num_channels_latents + num_channels_mask + num_channels_masked_image != num_channels_unet:
                raise ValueError(
                    f"Incorrect configuration settings! The config of `components.unet`: {components.unet.config} expects"
                    f" {components.unet.config.in_channels} but received `num_channels_latents`: {num_channels_latents} +"
                    f" `num_channels_mask`: {num_channels_mask} + `num_channels_masked_image`: {num_channels_masked_image}"
                    f" = {num_channels_latents + num_channels_masked_image + num_channels_mask}. Please verify the config of"
                    " `components.unet` or your `mask_image` or `image` input."
                )

    @torch.no_grad()
    def __call__(self, components: StableDiffusionXLModularPipeline, block_state: BlockState, i: int, t: int):
        self.check_inputs(components, block_state)

        block_state.scaled_latents = components.scheduler.scale_model_input(block_state.latents, t)
        if components.num_channels_unet == 9:
            block_state.scaled_latents = torch.cat(
                [block_state.scaled_latents, block_state.mask, block_state.masked_image_latents], dim=1
            )

        return components, block_state


# loop step (2): denoise the latents with guidance
class StableDiffusionXLLoopDenoiser(ModularPipelineBlocks):
    model_name = "stable-diffusion-xl"

    @property
    def expected_components(self) -> List[ComponentSpec]:
        return [
            ComponentSpec(
                "guider",
                ClassifierFreeGuidance,
                config=FrozenDict({"guidance_scale": 7.5}),
                default_creation_method="from_config",
            ),
            ComponentSpec("unet", UNet2DConditionModel),
        ]

    @property
    def description(self) -> str:
        return (
            "Step within the denoising loop that denoise the latents with guidance. "
            "This block should be used to compose the `sub_blocks` attribute of a `LoopSequentialPipelineBlocks` "
            "object (e.g. `StableDiffusionXLDenoiseLoopWrapper`)"
        )

    @property
    def inputs(self) -> List[Tuple[str, Any]]:
        return [
            InputParam("cross_attention_kwargs"),
            InputParam(
                "num_inference_steps",
                required=True,
                type_hint=int,
                description="The number of inference steps to use for the denoising process. Can be generated in set_timesteps step.",
            ),
            InputParam(
                "timestep_cond",
                type_hint=Optional[torch.Tensor],
                description="The guidance scale embedding to use for Latent Consistency Models(LCMs). Can be generated in prepare_additional_conditioning step.",
            ),
            InputParam(
                kwargs_type="denoiser_input_fields",
                description=(
                    "All conditional model inputs that need to be prepared with guider. "
                    "It should contain prompt_embeds/negative_prompt_embeds, "
                    "add_time_ids/negative_add_time_ids, "
                    "pooled_prompt_embeds/negative_pooled_prompt_embeds, "
                    "and ip_adapter_embeds/negative_ip_adapter_embeds (optional)."
                    "please add `kwargs_type=denoiser_input_fields` to their parameter spec (`OutputParam`) when they are created and added to the pipeline state"
                ),
            ),
        ]

    @torch.no_grad()
    def __call__(
        self, components: StableDiffusionXLModularPipeline, block_state: BlockState, i: int, t: int
    ) -> PipelineState:
        #  Map the keys we'll see on each `guider_state_batch` (e.g. guider_state_batch.prompt_embeds)
        #  to the corresponding (cond, uncond) fields on block_state. (e.g. block_state.prompt_embeds, block_state.negative_prompt_embeds)
        guider_input_fields = {
            "prompt_embeds": ("prompt_embeds", "negative_prompt_embeds"),
            "time_ids": ("add_time_ids", "negative_add_time_ids"),
            "text_embeds": ("pooled_prompt_embeds", "negative_pooled_prompt_embeds"),
            "image_embeds": ("ip_adapter_embeds", "negative_ip_adapter_embeds"),
        }

        components.guider.set_state(step=i, num_inference_steps=block_state.num_inference_steps, timestep=t)

        # Prepare mini‐batches according to guidance method and `guider_input_fields`
        # Each guider_state_batch will have .prompt_embeds, .time_ids, text_embeds, image_embeds.
        # e.g. for CFG, we prepare two batches: one for uncond, one for cond
        # for first batch, guider_state_batch.prompt_embeds correspond to block_state.prompt_embeds
        # for second batch, guider_state_batch.prompt_embeds correspond to block_state.negative_prompt_embeds
        guider_state = components.guider.prepare_inputs(block_state, guider_input_fields)

        # run the denoiser for each guidance batch
        for guider_state_batch in guider_state:
            components.guider.prepare_models(components.unet)
            cond_kwargs = guider_state_batch.as_dict()
            cond_kwargs = {k: v for k, v in cond_kwargs.items() if k in guider_input_fields}
            prompt_embeds = cond_kwargs.pop("prompt_embeds")

            # Predict the noise residual
            # store the noise_pred in guider_state_batch so that we can apply guidance across all batches
            guider_state_batch.noise_pred = components.unet(
                block_state.scaled_latents,
                t,
                encoder_hidden_states=prompt_embeds,
                timestep_cond=block_state.timestep_cond,
                cross_attention_kwargs=block_state.cross_attention_kwargs,
                added_cond_kwargs=cond_kwargs,
                return_dict=False,
            )[0]
            components.guider.cleanup_models(components.unet)

        # Perform guidance
        block_state.noise_pred = components.guider(guider_state)[0]

        return components, block_state


# loop step (2): denoise the latents with guidance (with controlnet)
class StableDiffusionXLControlNetLoopDenoiser(ModularPipelineBlocks):
    model_name = "stable-diffusion-xl"

    @property
    def expected_components(self) -> List[ComponentSpec]:
        return [
            ComponentSpec(
                "guider",
                ClassifierFreeGuidance,
                config=FrozenDict({"guidance_scale": 7.5}),
                default_creation_method="from_config",
            ),
            ComponentSpec("unet", UNet2DConditionModel),
            ComponentSpec("controlnet", ControlNetModel),
        ]

    @property
    def description(self) -> str:
        return (
            "step within the denoising loop that denoise the latents with guidance (with controlnet). "
            "This block should be used to compose the `sub_blocks` attribute of a `LoopSequentialPipelineBlocks` "
            "object (e.g. `StableDiffusionXLDenoiseLoopWrapper`)"
        )

    @property
    def inputs(self) -> List[Tuple[str, Any]]:
        return [
            InputParam("cross_attention_kwargs"),
            InputParam(
                "controlnet_cond",
                required=True,
                type_hint=torch.Tensor,
                description="The control image to use for the denoising process. Can be generated in prepare_controlnet_inputs step.",
            ),
            InputParam(
                "conditioning_scale",
                type_hint=float,
                description="The controlnet conditioning scale value to use for the denoising process. Can be generated in prepare_controlnet_inputs step.",
            ),
            InputParam(
                "guess_mode",
                required=True,
                type_hint=bool,
                description="The guess mode value to use for the denoising process. Can be generated in prepare_controlnet_inputs step.",
            ),
            InputParam(
                "controlnet_keep",
                required=True,
                type_hint=List[float],
                description="The controlnet keep values to use for the denoising process. Can be generated in prepare_controlnet_inputs step.",
            ),
            InputParam(
                "timestep_cond",
                type_hint=Optional[torch.Tensor],
                description="The guidance scale embedding to use for Latent Consistency Models(LCMs), can be generated by prepare_additional_conditioning step",
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
                    "All conditional model inputs that need to be prepared with guider. "
                    "It should contain prompt_embeds/negative_prompt_embeds, "
                    "add_time_ids/negative_add_time_ids, "
                    "pooled_prompt_embeds/negative_pooled_prompt_embeds, "
                    "and ip_adapter_embeds/negative_ip_adapter_embeds (optional)."
                    "please add `kwargs_type=denoiser_input_fields` to their parameter spec (`OutputParam`) when they are created and added to the pipeline state"
                ),
            ),
            InputParam(
                kwargs_type="controlnet_kwargs",
                description=(
                    "additional kwargs for controlnet (e.g. control_type_idx and control_type from the controlnet union input step )"
                    "please add `kwargs_type=controlnet_kwargs` to their parameter spec (`OutputParam`) when they are created and added to the pipeline state"
                ),
            ),
        ]

    @staticmethod
    def prepare_extra_kwargs(func, exclude_kwargs=[], **kwargs):
        accepted_kwargs = set(inspect.signature(func).parameters.keys())
        extra_kwargs = {}
        for key, value in kwargs.items():
            if key in accepted_kwargs and key not in exclude_kwargs:
                extra_kwargs[key] = value

        return extra_kwargs

    @torch.no_grad()
    def __call__(self, components: StableDiffusionXLModularPipeline, block_state: BlockState, i: int, t: int):
        extra_controlnet_kwargs = self.prepare_extra_kwargs(
            components.controlnet.forward, **block_state.controlnet_kwargs
        )

        #  Map the keys we'll see on each `guider_state_batch` (e.g. guider_state_batch.prompt_embeds)
        #  to the corresponding (cond, uncond) fields on block_state. (e.g. block_state.prompt_embeds, block_state.negative_prompt_embeds)
        guider_input_fields = {
            "prompt_embeds": ("prompt_embeds", "negative_prompt_embeds"),
            "time_ids": ("add_time_ids", "negative_add_time_ids"),
            "text_embeds": ("pooled_prompt_embeds", "negative_pooled_prompt_embeds"),
            "image_embeds": ("ip_adapter_embeds", "negative_ip_adapter_embeds"),
        }

        # cond_scale for the timestep (controlnet input)
        if isinstance(block_state.controlnet_keep[i], list):
            block_state.cond_scale = [
                c * s for c, s in zip(block_state.conditioning_scale, block_state.controlnet_keep[i])
            ]
        else:
            controlnet_cond_scale = block_state.conditioning_scale
            if isinstance(controlnet_cond_scale, list):
                controlnet_cond_scale = controlnet_cond_scale[0]
            block_state.cond_scale = controlnet_cond_scale * block_state.controlnet_keep[i]

        # default controlnet output/unet input for guess mode + conditional path
        block_state.down_block_res_samples_zeros = None
        block_state.mid_block_res_sample_zeros = None

        # guided denoiser step
        components.guider.set_state(step=i, num_inference_steps=block_state.num_inference_steps, timestep=t)

        # Prepare mini‐batches according to guidance method and `guider_input_fields`
        # Each guider_state_batch will have .prompt_embeds, .time_ids, text_embeds, image_embeds.
        # e.g. for CFG, we prepare two batches: one for uncond, one for cond
        # for first batch, guider_state_batch.prompt_embeds correspond to block_state.prompt_embeds
        # for second batch, guider_state_batch.prompt_embeds correspond to block_state.negative_prompt_embeds
        guider_state = components.guider.prepare_inputs(block_state, guider_input_fields)

        # run the denoiser for each guidance batch
        for guider_state_batch in guider_state:
            components.guider.prepare_models(components.unet)

            # Prepare additional conditionings
            added_cond_kwargs = {
                "text_embeds": guider_state_batch.text_embeds,
                "time_ids": guider_state_batch.time_ids,
            }
            if hasattr(guider_state_batch, "image_embeds") and guider_state_batch.image_embeds is not None:
                added_cond_kwargs["image_embeds"] = guider_state_batch.image_embeds

            # Prepare controlnet additional conditionings
            controlnet_added_cond_kwargs = {
                "text_embeds": guider_state_batch.text_embeds,
                "time_ids": guider_state_batch.time_ids,
            }
            # run controlnet for the guidance batch
            if block_state.guess_mode and not components.guider.is_conditional:
                # guider always run uncond batch first, so these tensors should be set already
                down_block_res_samples = block_state.down_block_res_samples_zeros
                mid_block_res_sample = block_state.mid_block_res_sample_zeros
            else:
                down_block_res_samples, mid_block_res_sample = components.controlnet(
                    block_state.scaled_latents,
                    t,
                    encoder_hidden_states=guider_state_batch.prompt_embeds,
                    controlnet_cond=block_state.controlnet_cond,
                    conditioning_scale=block_state.cond_scale,
                    guess_mode=block_state.guess_mode,
                    added_cond_kwargs=controlnet_added_cond_kwargs,
                    return_dict=False,
                    **extra_controlnet_kwargs,
                )

                # assign it to block_state so it will be available for the uncond guidance batch
                if block_state.down_block_res_samples_zeros is None:
                    block_state.down_block_res_samples_zeros = [torch.zeros_like(d) for d in down_block_res_samples]
                if block_state.mid_block_res_sample_zeros is None:
                    block_state.mid_block_res_sample_zeros = torch.zeros_like(mid_block_res_sample)

            # Predict the noise
            # store the noise_pred in guider_state_batch so we can apply guidance across all batches
            guider_state_batch.noise_pred = components.unet(
                block_state.scaled_latents,
                t,
                encoder_hidden_states=guider_state_batch.prompt_embeds,
                timestep_cond=block_state.timestep_cond,
                cross_attention_kwargs=block_state.cross_attention_kwargs,
                added_cond_kwargs=added_cond_kwargs,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
                return_dict=False,
            )[0]
            components.guider.cleanup_models(components.unet)

        # Perform guidance
        block_state.noise_pred = components.guider(guider_state)[0]

        return components, block_state


# loop step (3): scheduler step to update latents
class StableDiffusionXLLoopAfterDenoiser(ModularPipelineBlocks):
    model_name = "stable-diffusion-xl"

    @property
    def expected_components(self) -> List[ComponentSpec]:
        return [
            ComponentSpec("scheduler", EulerDiscreteScheduler),
        ]

    @property
    def description(self) -> str:
        return (
            "step within the denoising loop that update the latents. "
            "This block should be used to compose the `sub_blocks` attribute of a `LoopSequentialPipelineBlocks` "
            "object (e.g. `StableDiffusionXLDenoiseLoopWrapper`)"
        )

    @property
    def inputs(self) -> List[Tuple[str, Any]]:
        return [
            InputParam("eta", default=0.0),
            InputParam("generator"),
        ]

    @property
    def intermediate_outputs(self) -> List[OutputParam]:
        return [OutputParam("latents", type_hint=torch.Tensor, description="The denoised latents")]

    # YiYi TODO: move this out of here
    @staticmethod
    def prepare_extra_kwargs(func, exclude_kwargs=[], **kwargs):
        accepted_kwargs = set(inspect.signature(func).parameters.keys())
        extra_kwargs = {}
        for key, value in kwargs.items():
            if key in accepted_kwargs and key not in exclude_kwargs:
                extra_kwargs[key] = value

        return extra_kwargs

    @torch.no_grad()
    def __call__(self, components: StableDiffusionXLModularPipeline, block_state: BlockState, i: int, t: int):
        # Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        block_state.extra_step_kwargs = self.prepare_extra_kwargs(
            components.scheduler.step, generator=block_state.generator, eta=block_state.eta
        )

        # Perform scheduler step using the predicted output
        block_state.latents_dtype = block_state.latents.dtype
        block_state.latents = components.scheduler.step(
            block_state.noise_pred,
            t,
            block_state.latents,
            **block_state.extra_step_kwargs,
            return_dict=False,
        )[0]

        if block_state.latents.dtype != block_state.latents_dtype:
            if torch.backends.mps.is_available():
                # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                block_state.latents = block_state.latents.to(block_state.latents_dtype)

        return components, block_state


# loop step (3): scheduler step to update latents (with inpainting)
class StableDiffusionXLInpaintLoopAfterDenoiser(ModularPipelineBlocks):
    model_name = "stable-diffusion-xl"

    @property
    def expected_components(self) -> List[ComponentSpec]:
        return [
            ComponentSpec("scheduler", EulerDiscreteScheduler),
            ComponentSpec("unet", UNet2DConditionModel),
        ]

    @property
    def description(self) -> str:
        return (
            "step within the denoising loop that update the latents (for inpainting workflow only). "
            "This block should be used to compose the `sub_blocks` attribute of a `LoopSequentialPipelineBlocks` "
            "object (e.g. `StableDiffusionXLDenoiseLoopWrapper`)"
        )

    @property
    def inputs(self) -> List[Tuple[str, Any]]:
        return [
            InputParam("eta", default=0.0),
            InputParam("generator"),
            InputParam(
                "timesteps",
                required=True,
                type_hint=torch.Tensor,
                description="The timesteps to use for the denoising process. Can be generated in set_timesteps step.",
            ),
            InputParam(
                "mask",
                type_hint=Optional[torch.Tensor],
                description="The mask to use for the denoising process, for inpainting task only. Can be generated in vae_encode or prepare_latent step.",
            ),
            InputParam(
                "noise",
                type_hint=Optional[torch.Tensor],
                description="The noise added to the image latents, for inpainting task only. Can be generated in prepare_latent step.",
            ),
            InputParam(
                "image_latents",
                type_hint=Optional[torch.Tensor],
                description="The image latents to use for the denoising process, for inpainting/image-to-image task only. Can be generated in vae_encode or prepare_latent step.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> List[OutputParam]:
        return [OutputParam("latents", type_hint=torch.Tensor, description="The denoised latents")]

    @staticmethod
    def prepare_extra_kwargs(func, exclude_kwargs=[], **kwargs):
        accepted_kwargs = set(inspect.signature(func).parameters.keys())
        extra_kwargs = {}
        for key, value in kwargs.items():
            if key in accepted_kwargs and key not in exclude_kwargs:
                extra_kwargs[key] = value

        return extra_kwargs

    def check_inputs(self, components, block_state):
        if components.num_channels_unet == 4:
            if block_state.image_latents is None:
                raise ValueError(f"image_latents is required for this step {self.__class__.__name__}")
            if block_state.mask is None:
                raise ValueError(f"mask is required for this step {self.__class__.__name__}")
            if block_state.noise is None:
                raise ValueError(f"noise is required for this step {self.__class__.__name__}")

    @torch.no_grad()
    def __call__(self, components: StableDiffusionXLModularPipeline, block_state: BlockState, i: int, t: int):
        self.check_inputs(components, block_state)

        # Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        block_state.extra_step_kwargs = self.prepare_extra_kwargs(
            components.scheduler.step, generator=block_state.generator, eta=block_state.eta
        )

        # Perform scheduler step using the predicted output
        block_state.latents_dtype = block_state.latents.dtype
        block_state.latents = components.scheduler.step(
            block_state.noise_pred,
            t,
            block_state.latents,
            **block_state.extra_step_kwargs,
            return_dict=False,
        )[0]

        if block_state.latents.dtype != block_state.latents_dtype:
            if torch.backends.mps.is_available():
                # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                block_state.latents = block_state.latents.to(block_state.latents_dtype)

        # adjust latent for inpainting
        if components.num_channels_unet == 4:
            block_state.init_latents_proper = block_state.image_latents
            if i < len(block_state.timesteps) - 1:
                block_state.noise_timestep = block_state.timesteps[i + 1]
                block_state.init_latents_proper = components.scheduler.add_noise(
                    block_state.init_latents_proper, block_state.noise, torch.tensor([block_state.noise_timestep])
                )

            block_state.latents = (
                1 - block_state.mask
            ) * block_state.init_latents_proper + block_state.mask * block_state.latents

        return components, block_state


# the loop wrapper that iterates over the timesteps
class StableDiffusionXLDenoiseLoopWrapper(LoopSequentialPipelineBlocks):
    model_name = "stable-diffusion-xl"

    @property
    def description(self) -> str:
        return (
            "Pipeline block that iteratively denoise the latents over `timesteps`. "
            "The specific steps with each iteration can be customized with `sub_blocks` attributes"
        )

    @property
    def loop_expected_components(self) -> List[ComponentSpec]:
        return [
            ComponentSpec(
                "guider",
                ClassifierFreeGuidance,
                config=FrozenDict({"guidance_scale": 7.5}),
                default_creation_method="from_config",
            ),
            ComponentSpec("scheduler", EulerDiscreteScheduler),
            ComponentSpec("unet", UNet2DConditionModel),
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
    def __call__(self, components: StableDiffusionXLModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        block_state.disable_guidance = True if components.unet.config.time_cond_proj_dim is not None else False
        if block_state.disable_guidance:
            components.guider.disable()
        else:
            components.guider.enable()

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


# composing the denoising loops
class StableDiffusionXLDenoiseStep(StableDiffusionXLDenoiseLoopWrapper):
    block_classes = [
        StableDiffusionXLLoopBeforeDenoiser,
        StableDiffusionXLLoopDenoiser,
        StableDiffusionXLLoopAfterDenoiser,
    ]
    block_names = ["before_denoiser", "denoiser", "after_denoiser"]

    @property
    def description(self) -> str:
        return (
            "Denoise step that iteratively denoise the latents. \n"
            "Its loop logic is defined in `StableDiffusionXLDenoiseLoopWrapper.__call__` method \n"
            "At each iteration, it runs blocks defined in `sub_blocks` sequentially:\n"
            " - `StableDiffusionXLLoopBeforeDenoiser`\n"
            " - `StableDiffusionXLLoopDenoiser`\n"
            " - `StableDiffusionXLLoopAfterDenoiser`\n"
            "This block supports both text2img and img2img tasks."
        )


# control_cond
class StableDiffusionXLControlNetDenoiseStep(StableDiffusionXLDenoiseLoopWrapper):
    block_classes = [
        StableDiffusionXLLoopBeforeDenoiser,
        StableDiffusionXLControlNetLoopDenoiser,
        StableDiffusionXLLoopAfterDenoiser,
    ]
    block_names = ["before_denoiser", "denoiser", "after_denoiser"]

    @property
    def description(self) -> str:
        return (
            "Denoise step that iteratively denoise the latents with controlnet. \n"
            "Its loop logic is defined in  `StableDiffusionXLDenoiseLoopWrapper.__call__` method \n"
            "At each iteration, it runs blocks defined in `sub_blocks` sequentially:\n"
            " - `StableDiffusionXLLoopBeforeDenoiser`\n"
            " - `StableDiffusionXLControlNetLoopDenoiser`\n"
            " - `StableDiffusionXLLoopAfterDenoiser`\n"
            "This block supports using controlnet for both text2img and img2img tasks."
        )


# mask
class StableDiffusionXLInpaintDenoiseStep(StableDiffusionXLDenoiseLoopWrapper):
    block_classes = [
        StableDiffusionXLInpaintLoopBeforeDenoiser,
        StableDiffusionXLLoopDenoiser,
        StableDiffusionXLInpaintLoopAfterDenoiser,
    ]
    block_names = ["before_denoiser", "denoiser", "after_denoiser"]

    @property
    def description(self) -> str:
        return (
            "Denoise step that iteratively denoise the latents(for inpainting task only). \n"
            "Its loop logic is defined in `StableDiffusionXLDenoiseLoopWrapper.__call__` method \n"
            "At each iteration, it runs blocks defined in `sub_blocks` sequentially:\n"
            " - `StableDiffusionXLInpaintLoopBeforeDenoiser`\n"
            " - `StableDiffusionXLLoopDenoiser`\n"
            " - `StableDiffusionXLInpaintLoopAfterDenoiser`\n"
            "This block onlysupports inpainting tasks."
        )


# control_cond + mask
class StableDiffusionXLInpaintControlNetDenoiseStep(StableDiffusionXLDenoiseLoopWrapper):
    block_classes = [
        StableDiffusionXLInpaintLoopBeforeDenoiser,
        StableDiffusionXLControlNetLoopDenoiser,
        StableDiffusionXLInpaintLoopAfterDenoiser,
    ]
    block_names = ["before_denoiser", "denoiser", "after_denoiser"]

    @property
    def description(self) -> str:
        return (
            "Denoise step that iteratively denoise the latents(for inpainting task only) with controlnet. \n"
            "Its loop logic is defined in `StableDiffusionXLDenoiseLoopWrapper.__call__` method \n"
            "At each iteration, it runs blocks defined in `sub_blocks` sequentially:\n"
            " - `StableDiffusionXLInpaintLoopBeforeDenoiser`\n"
            " - `StableDiffusionXLControlNetLoopDenoiser`\n"
            " - `StableDiffusionXLInpaintLoopAfterDenoiser`\n"
            "This block only supports using controlnet for inpainting tasks."
        )
