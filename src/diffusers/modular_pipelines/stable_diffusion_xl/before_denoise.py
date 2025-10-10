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
from typing import Any, List, Optional, Tuple, Union

import PIL
import torch

from ...configuration_utils import FrozenDict
from ...guiders import ClassifierFreeGuidance
from ...image_processor import VaeImageProcessor
from ...models import AutoencoderKL, ControlNetModel, ControlNetUnionModel, UNet2DConditionModel
from ...models.controlnets.multicontrolnet import MultiControlNetModel
from ...schedulers import EulerDiscreteScheduler
from ...utils import logging
from ...utils.torch_utils import randn_tensor, unwrap_module
from ..modular_pipeline import (
    ModularPipelineBlocks,
    PipelineState,
)
from ..modular_pipeline_utils import ComponentSpec, ConfigSpec, InputParam, OutputParam
from .modular_pipeline import StableDiffusionXLModularPipeline


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# TODO(yiyi, aryan): We need another step before text encoder to set the `num_inference_steps` attribute for guider so that
# things like when to do guidance and how many conditions to be prepared can be determined. Currently, this is done by
# always assuming you want to do guidance in the Guiders. So, negative embeddings are prepared regardless of what the
# configuration of guider is.


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.retrieve_latents
def retrieve_latents(
    encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


def prepare_latents_img2img(
    vae, scheduler, image, timestep, batch_size, num_images_per_prompt, dtype, device, generator=None, add_noise=True
):
    if not isinstance(image, (torch.Tensor, PIL.Image.Image, list)):
        raise ValueError(f"`image` has to be of type `torch.Tensor`, `PIL.Image.Image` or list but is {type(image)}")

    image = image.to(device=device, dtype=dtype)

    batch_size = batch_size * num_images_per_prompt

    if image.shape[1] == 4:
        init_latents = image

    else:
        latents_mean = latents_std = None
        if hasattr(vae.config, "latents_mean") and vae.config.latents_mean is not None:
            latents_mean = torch.tensor(vae.config.latents_mean).view(1, 4, 1, 1)
        if hasattr(vae.config, "latents_std") and vae.config.latents_std is not None:
            latents_std = torch.tensor(vae.config.latents_std).view(1, 4, 1, 1)
        # make sure the VAE is in float32 mode, as it overflows in float16
        if vae.config.force_upcast:
            image = image.float()
            vae.to(dtype=torch.float32)

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        elif isinstance(generator, list):
            if image.shape[0] < batch_size and batch_size % image.shape[0] == 0:
                image = torch.cat([image] * (batch_size // image.shape[0]), dim=0)
            elif image.shape[0] < batch_size and batch_size % image.shape[0] != 0:
                raise ValueError(
                    f"Cannot duplicate `image` of batch size {image.shape[0]} to effective batch_size {batch_size} "
                )

            init_latents = [
                retrieve_latents(vae.encode(image[i : i + 1]), generator=generator[i]) for i in range(batch_size)
            ]
            init_latents = torch.cat(init_latents, dim=0)
        else:
            init_latents = retrieve_latents(vae.encode(image), generator=generator)

        if vae.config.force_upcast:
            vae.to(dtype)

        init_latents = init_latents.to(dtype)
        if latents_mean is not None and latents_std is not None:
            latents_mean = latents_mean.to(device=device, dtype=dtype)
            latents_std = latents_std.to(device=device, dtype=dtype)
            init_latents = (init_latents - latents_mean) * vae.config.scaling_factor / latents_std
        else:
            init_latents = vae.config.scaling_factor * init_latents

    if batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] == 0:
        # expand init_latents for batch_size
        additional_image_per_prompt = batch_size // init_latents.shape[0]
        init_latents = torch.cat([init_latents] * additional_image_per_prompt, dim=0)
    elif batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] != 0:
        raise ValueError(
            f"Cannot duplicate `image` of batch size {init_latents.shape[0]} to {batch_size} text prompts."
        )
    else:
        init_latents = torch.cat([init_latents], dim=0)

    if add_noise:
        shape = init_latents.shape
        noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        # get latents
        init_latents = scheduler.add_noise(init_latents, noise, timestep)

    latents = init_latents

    return latents


class StableDiffusionXLInputStep(ModularPipelineBlocks):
    model_name = "stable-diffusion-xl"

    @property
    def description(self) -> str:
        return (
            "Input processing step that:\n"
            "  1. Determines `batch_size` and `dtype` based on `prompt_embeds`\n"
            "  2. Adjusts input tensor shapes based on `batch_size` (number of prompts) and `num_images_per_prompt`\n\n"
            "All input tensors are expected to have either batch_size=1 or match the batch_size\n"
            "of prompt_embeds. The tensors will be duplicated across the batch dimension to\n"
            "have a final batch_size of batch_size * num_images_per_prompt."
        )

    @property
    def inputs(self) -> List[InputParam]:
        return [
            InputParam("num_images_per_prompt", default=1),
            InputParam(
                "prompt_embeds",
                required=True,
                type_hint=torch.Tensor,
                description="Pre-generated text embeddings. Can be generated from text_encoder step.",
            ),
            InputParam(
                "negative_prompt_embeds",
                type_hint=torch.Tensor,
                description="Pre-generated negative text embeddings. Can be generated from text_encoder step.",
            ),
            InputParam(
                "pooled_prompt_embeds",
                required=True,
                type_hint=torch.Tensor,
                description="Pre-generated pooled text embeddings. Can be generated from text_encoder step.",
            ),
            InputParam(
                "negative_pooled_prompt_embeds",
                description="Pre-generated negative pooled text embeddings. Can be generated from text_encoder step.",
            ),
            InputParam(
                "ip_adapter_embeds",
                type_hint=List[torch.Tensor],
                description="Pre-generated image embeddings for IP-Adapter. Can be generated from ip_adapter step.",
            ),
            InputParam(
                "negative_ip_adapter_embeds",
                type_hint=List[torch.Tensor],
                description="Pre-generated negative image embeddings for IP-Adapter. Can be generated from ip_adapter step.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> List[str]:
        return [
            OutputParam(
                "batch_size",
                type_hint=int,
                description="Number of prompts, the final batch size of model inputs should be batch_size * num_images_per_prompt",
            ),
            OutputParam(
                "dtype",
                type_hint=torch.dtype,
                description="Data type of model tensor inputs (determined by `prompt_embeds`)",
            ),
            OutputParam(
                "prompt_embeds",
                type_hint=torch.Tensor,
                kwargs_type="denoiser_input_fields",  # already in intermedites state but declare here again for denoiser_input_fields
                description="text embeddings used to guide the image generation",
            ),
            OutputParam(
                "negative_prompt_embeds",
                type_hint=torch.Tensor,
                kwargs_type="denoiser_input_fields",  # already in intermedites state but declare here again for denoiser_input_fields
                description="negative text embeddings used to guide the image generation",
            ),
            OutputParam(
                "pooled_prompt_embeds",
                type_hint=torch.Tensor,
                kwargs_type="denoiser_input_fields",  # already in intermedites state but declare here again for denoiser_input_fields
                description="pooled text embeddings used to guide the image generation",
            ),
            OutputParam(
                "negative_pooled_prompt_embeds",
                type_hint=torch.Tensor,
                kwargs_type="denoiser_input_fields",  # already in intermedites state but declare here again for denoiser_input_fields
                description="negative pooled text embeddings used to guide the image generation",
            ),
            OutputParam(
                "ip_adapter_embeds",
                type_hint=List[torch.Tensor],
                kwargs_type="denoiser_input_fields",  # already in intermedites state but declare here again for denoiser_input_fields
                description="image embeddings for IP-Adapter",
            ),
            OutputParam(
                "negative_ip_adapter_embeds",
                type_hint=List[torch.Tensor],
                kwargs_type="denoiser_input_fields",  # already in intermedites state but declare here again for denoiser_input_fields
                description="negative image embeddings for IP-Adapter",
            ),
        ]

    def check_inputs(self, components, block_state):
        if block_state.prompt_embeds is not None and block_state.negative_prompt_embeds is not None:
            if block_state.prompt_embeds.shape != block_state.negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {block_state.prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {block_state.negative_prompt_embeds.shape}."
                )

        if block_state.prompt_embeds is not None and block_state.pooled_prompt_embeds is None:
            raise ValueError(
                "If `prompt_embeds` are provided, `pooled_prompt_embeds` also have to be passed. Make sure to generate `pooled_prompt_embeds` from the same text encoder that was used to generate `prompt_embeds`."
            )

        if block_state.negative_prompt_embeds is not None and block_state.negative_pooled_prompt_embeds is None:
            raise ValueError(
                "If `negative_prompt_embeds` are provided, `negative_pooled_prompt_embeds` also have to be passed. Make sure to generate `negative_pooled_prompt_embeds` from the same text encoder that was used to generate `negative_prompt_embeds`."
            )

        if block_state.ip_adapter_embeds is not None and not isinstance(block_state.ip_adapter_embeds, list):
            raise ValueError("`ip_adapter_embeds` must be a list")

        if block_state.negative_ip_adapter_embeds is not None and not isinstance(
            block_state.negative_ip_adapter_embeds, list
        ):
            raise ValueError("`negative_ip_adapter_embeds` must be a list")

        if block_state.ip_adapter_embeds is not None and block_state.negative_ip_adapter_embeds is not None:
            for i, ip_adapter_embed in enumerate(block_state.ip_adapter_embeds):
                if ip_adapter_embed.shape != block_state.negative_ip_adapter_embeds[i].shape:
                    raise ValueError(
                        "`ip_adapter_embeds` and `negative_ip_adapter_embeds` must have the same shape when passed directly, but"
                        f" got: `ip_adapter_embeds` {ip_adapter_embed.shape} != `negative_ip_adapter_embeds`"
                        f" {block_state.negative_ip_adapter_embeds[i].shape}."
                    )

    @torch.no_grad()
    def __call__(self, components: StableDiffusionXLModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        self.check_inputs(components, block_state)

        block_state.batch_size = block_state.prompt_embeds.shape[0]
        block_state.dtype = block_state.prompt_embeds.dtype

        _, seq_len, _ = block_state.prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        block_state.prompt_embeds = block_state.prompt_embeds.repeat(1, block_state.num_images_per_prompt, 1)
        block_state.prompt_embeds = block_state.prompt_embeds.view(
            block_state.batch_size * block_state.num_images_per_prompt, seq_len, -1
        )

        if block_state.negative_prompt_embeds is not None:
            _, seq_len, _ = block_state.negative_prompt_embeds.shape
            block_state.negative_prompt_embeds = block_state.negative_prompt_embeds.repeat(
                1, block_state.num_images_per_prompt, 1
            )
            block_state.negative_prompt_embeds = block_state.negative_prompt_embeds.view(
                block_state.batch_size * block_state.num_images_per_prompt, seq_len, -1
            )

        block_state.pooled_prompt_embeds = block_state.pooled_prompt_embeds.repeat(
            1, block_state.num_images_per_prompt, 1
        )
        block_state.pooled_prompt_embeds = block_state.pooled_prompt_embeds.view(
            block_state.batch_size * block_state.num_images_per_prompt, -1
        )

        if block_state.negative_pooled_prompt_embeds is not None:
            block_state.negative_pooled_prompt_embeds = block_state.negative_pooled_prompt_embeds.repeat(
                1, block_state.num_images_per_prompt, 1
            )
            block_state.negative_pooled_prompt_embeds = block_state.negative_pooled_prompt_embeds.view(
                block_state.batch_size * block_state.num_images_per_prompt, -1
            )

        if block_state.ip_adapter_embeds is not None:
            for i, ip_adapter_embed in enumerate(block_state.ip_adapter_embeds):
                block_state.ip_adapter_embeds[i] = torch.cat(
                    [ip_adapter_embed] * block_state.num_images_per_prompt, dim=0
                )

        if block_state.negative_ip_adapter_embeds is not None:
            for i, negative_ip_adapter_embed in enumerate(block_state.negative_ip_adapter_embeds):
                block_state.negative_ip_adapter_embeds[i] = torch.cat(
                    [negative_ip_adapter_embed] * block_state.num_images_per_prompt, dim=0
                )

        self.set_block_state(state, block_state)

        return components, state


class StableDiffusionXLImg2ImgSetTimestepsStep(ModularPipelineBlocks):
    model_name = "stable-diffusion-xl"

    @property
    def expected_components(self) -> List[ComponentSpec]:
        return [
            ComponentSpec("scheduler", EulerDiscreteScheduler),
        ]

    @property
    def description(self) -> str:
        return (
            "Step that sets the timesteps for the scheduler and determines the initial noise level (latent_timestep) for image-to-image/inpainting generation.\n"
            + "The latent_timestep is calculated from the `strength` parameter - higher strength means starting from a noisier version of the input image."
        )

    @property
    def inputs(self) -> List[InputParam]:
        return [
            InputParam("num_inference_steps", default=50),
            InputParam("timesteps"),
            InputParam("sigmas"),
            InputParam("denoising_end"),
            InputParam("strength", default=0.3),
            InputParam("denoising_start"),
            # YiYi TODO: do we need num_images_per_prompt here?
            InputParam("num_images_per_prompt", default=1),
            InputParam(
                "batch_size",
                required=True,
                type_hint=int,
                description="Number of prompts, the final batch size of model inputs should be batch_size * num_images_per_prompt",
            ),
        ]

    @property
    def intermediate_outputs(self) -> List[str]:
        return [
            OutputParam("timesteps", type_hint=torch.Tensor, description="The timesteps to use for inference"),
            OutputParam(
                "num_inference_steps",
                type_hint=int,
                description="The number of denoising steps to perform at inference time",
            ),
            OutputParam(
                "latent_timestep",
                type_hint=torch.Tensor,
                description="The timestep that represents the initial noise level for image-to-image generation",
            ),
        ]

    @staticmethod
    # Copied from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_img2img.StableDiffusionXLImg2ImgPipeline.get_timesteps with self->components
    def get_timesteps(components, num_inference_steps, strength, device, denoising_start=None):
        # get the original timestep using init_timestep
        if denoising_start is None:
            init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
            t_start = max(num_inference_steps - init_timestep, 0)

            timesteps = components.scheduler.timesteps[t_start * components.scheduler.order :]
            if hasattr(components.scheduler, "set_begin_index"):
                components.scheduler.set_begin_index(t_start * components.scheduler.order)

            return timesteps, num_inference_steps - t_start

        else:
            # Strength is irrelevant if we directly request a timestep to start at;
            # that is, strength is determined by the denoising_start instead.
            discrete_timestep_cutoff = int(
                round(
                    components.scheduler.config.num_train_timesteps
                    - (denoising_start * components.scheduler.config.num_train_timesteps)
                )
            )

            num_inference_steps = (components.scheduler.timesteps < discrete_timestep_cutoff).sum().item()
            if components.scheduler.order == 2 and num_inference_steps % 2 == 0:
                # if the scheduler is a 2nd order scheduler we might have to do +1
                # because `num_inference_steps` might be even given that every timestep
                # (except the highest one) is duplicated. If `num_inference_steps` is even it would
                # mean that we cut the timesteps in the middle of the denoising step
                # (between 1st and 2nd derivative) which leads to incorrect results. By adding 1
                # we ensure that the denoising process always ends after the 2nd derivate step of the scheduler
                num_inference_steps = num_inference_steps + 1

            # because t_n+1 >= t_n, we slice the timesteps starting from the end
            t_start = len(components.scheduler.timesteps) - num_inference_steps
            timesteps = components.scheduler.timesteps[t_start:]
            if hasattr(components.scheduler, "set_begin_index"):
                components.scheduler.set_begin_index(t_start)
            return timesteps, num_inference_steps

    @torch.no_grad()
    def __call__(self, components: StableDiffusionXLModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        block_state.device = components._execution_device

        block_state.timesteps, block_state.num_inference_steps = retrieve_timesteps(
            components.scheduler,
            block_state.num_inference_steps,
            block_state.device,
            block_state.timesteps,
            block_state.sigmas,
        )

        def denoising_value_valid(dnv):
            return isinstance(dnv, float) and 0 < dnv < 1

        block_state.timesteps, block_state.num_inference_steps = self.get_timesteps(
            components,
            block_state.num_inference_steps,
            block_state.strength,
            block_state.device,
            denoising_start=block_state.denoising_start
            if denoising_value_valid(block_state.denoising_start)
            else None,
        )
        block_state.latent_timestep = block_state.timesteps[:1].repeat(
            block_state.batch_size * block_state.num_images_per_prompt
        )

        if (
            block_state.denoising_end is not None
            and isinstance(block_state.denoising_end, float)
            and block_state.denoising_end > 0
            and block_state.denoising_end < 1
        ):
            block_state.discrete_timestep_cutoff = int(
                round(
                    components.scheduler.config.num_train_timesteps
                    - (block_state.denoising_end * components.scheduler.config.num_train_timesteps)
                )
            )
            block_state.num_inference_steps = len(
                list(filter(lambda ts: ts >= block_state.discrete_timestep_cutoff, block_state.timesteps))
            )
            block_state.timesteps = block_state.timesteps[: block_state.num_inference_steps]

        self.set_block_state(state, block_state)

        return components, state


class StableDiffusionXLSetTimestepsStep(ModularPipelineBlocks):
    model_name = "stable-diffusion-xl"

    @property
    def expected_components(self) -> List[ComponentSpec]:
        return [
            ComponentSpec("scheduler", EulerDiscreteScheduler),
        ]

    @property
    def description(self) -> str:
        return "Step that sets the scheduler's timesteps for inference"

    @property
    def inputs(self) -> List[InputParam]:
        return [
            InputParam("num_inference_steps", default=50),
            InputParam("timesteps"),
            InputParam("sigmas"),
            InputParam("denoising_end"),
        ]

    @property
    def intermediate_outputs(self) -> List[OutputParam]:
        return [
            OutputParam("timesteps", type_hint=torch.Tensor, description="The timesteps to use for inference"),
            OutputParam(
                "num_inference_steps",
                type_hint=int,
                description="The number of denoising steps to perform at inference time",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components: StableDiffusionXLModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        block_state.device = components._execution_device

        block_state.timesteps, block_state.num_inference_steps = retrieve_timesteps(
            components.scheduler,
            block_state.num_inference_steps,
            block_state.device,
            block_state.timesteps,
            block_state.sigmas,
        )

        if (
            block_state.denoising_end is not None
            and isinstance(block_state.denoising_end, float)
            and block_state.denoising_end > 0
            and block_state.denoising_end < 1
        ):
            block_state.discrete_timestep_cutoff = int(
                round(
                    components.scheduler.config.num_train_timesteps
                    - (block_state.denoising_end * components.scheduler.config.num_train_timesteps)
                )
            )
            block_state.num_inference_steps = len(
                list(filter(lambda ts: ts >= block_state.discrete_timestep_cutoff, block_state.timesteps))
            )
            block_state.timesteps = block_state.timesteps[: block_state.num_inference_steps]

        self.set_block_state(state, block_state)
        return components, state


class StableDiffusionXLInpaintPrepareLatentsStep(ModularPipelineBlocks):
    model_name = "stable-diffusion-xl"

    @property
    def expected_components(self) -> List[ComponentSpec]:
        return [
            ComponentSpec("scheduler", EulerDiscreteScheduler),
        ]

    @property
    def description(self) -> str:
        return "Step that prepares the latents for the inpainting process"

    @property
    def inputs(self) -> List[Tuple[str, Any]]:
        return [
            InputParam("latents"),
            InputParam("num_images_per_prompt", default=1),
            InputParam("denoising_start"),
            InputParam(
                "strength",
                default=0.9999,
                description="Conceptually, indicates how much to transform the reference `image` (the masked portion of image for inpainting). Must be between 0 and 1. `image` "
                "will be used as a starting point, adding more noise to it the larger the `strength`. The number of "
                "denoising steps depends on the amount of noise initially added. When `strength` is 1, added noise will "
                "be maximum and the denoising process will run for the full number of iterations specified in "
                "`num_inference_steps`. A value of 1, therefore, essentially ignores `image`. Note that in the case of "
                "`denoising_start` being declared as an integer, the value of `strength` will be ignored.",
            ),
            InputParam("generator"),
            InputParam(
                "batch_size",
                required=True,
                type_hint=int,
                description="Number of prompts, the final batch size of model inputs should be batch_size * num_images_per_prompt. Can be generated in input step.",
            ),
            InputParam(
                "latent_timestep",
                required=True,
                type_hint=torch.Tensor,
                description="The timestep that represents the initial noise level for image-to-image/inpainting generation. Can be generated in set_timesteps step.",
            ),
            InputParam(
                "image_latents",
                required=True,
                type_hint=torch.Tensor,
                description="The latents representing the reference image for image-to-image/inpainting generation. Can be generated in vae_encode step.",
            ),
            InputParam(
                "mask",
                required=True,
                type_hint=torch.Tensor,
                description="The mask for the inpainting generation. Can be generated in vae_encode step.",
            ),
            InputParam(
                "masked_image_latents",
                type_hint=torch.Tensor,
                description="The masked image latents for the inpainting generation (only for inpainting-specific unet). Can be generated in vae_encode step.",
            ),
            InputParam("dtype", type_hint=torch.dtype, description="The dtype of the model inputs"),
        ]

    @property
    def intermediate_outputs(self) -> List[str]:
        return [
            OutputParam(
                "latents", type_hint=torch.Tensor, description="The initial latents to use for the denoising process"
            ),
            OutputParam(
                "noise",
                type_hint=torch.Tensor,
                description="The noise added to the image latents, used for inpainting generation",
            ),
        ]

    # Modified from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_inpaint.StableDiffusionXLInpaintPipeline._encode_vae_image with self->components
    # YiYi TODO: update the _encode_vae_image so that we can use #Coped from
    @staticmethod
    def _encode_vae_image(components, image: torch.Tensor, generator: torch.Generator):
        latents_mean = latents_std = None
        if hasattr(components.vae.config, "latents_mean") and components.vae.config.latents_mean is not None:
            latents_mean = torch.tensor(components.vae.config.latents_mean).view(1, 4, 1, 1)
        if hasattr(components.vae.config, "latents_std") and components.vae.config.latents_std is not None:
            latents_std = torch.tensor(components.vae.config.latents_std).view(1, 4, 1, 1)

        dtype = image.dtype
        if components.vae.config.force_upcast:
            image = image.float()
            components.vae.to(dtype=torch.float32)

        if isinstance(generator, list):
            image_latents = [
                retrieve_latents(components.vae.encode(image[i : i + 1]), generator=generator[i])
                for i in range(image.shape[0])
            ]
            image_latents = torch.cat(image_latents, dim=0)
        else:
            image_latents = retrieve_latents(components.vae.encode(image), generator=generator)

        if components.vae.config.force_upcast:
            components.vae.to(dtype)

        image_latents = image_latents.to(dtype)
        if latents_mean is not None and latents_std is not None:
            latents_mean = latents_mean.to(device=image_latents.device, dtype=dtype)
            latents_std = latents_std.to(device=image_latents.device, dtype=dtype)
            image_latents = (image_latents - latents_mean) * components.vae.config.scaling_factor / latents_std
        else:
            image_latents = components.vae.config.scaling_factor * image_latents

        return image_latents

    # Modified from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_inpaint.StableDiffusionXLInpaintPipeline.prepare_latents adding components as first argument
    def prepare_latents_inpaint(
        self,
        components,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
        image=None,
        timestep=None,
        is_strength_max=True,
        add_noise=True,
    ):
        shape = (
            batch_size,
            num_channels_latents,
            int(height) // components.vae_scale_factor,
            int(width) // components.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if (image is None or timestep is None) and not is_strength_max:
            raise ValueError(
                "Since strength < 1. initial latents are to be initialised as a combination of Image + Noise."
                "However, either the image or the noise timestep has not been provided."
            )

        if image.shape[1] == 4:
            image_latents = image.to(device=device, dtype=dtype)
            image_latents = image_latents.repeat(batch_size // image_latents.shape[0], 1, 1, 1)
        elif latents is None and not is_strength_max:
            image = image.to(device=device, dtype=dtype)
            image_latents = self._encode_vae_image(components, image=image, generator=generator)
            image_latents = image_latents.repeat(batch_size // image_latents.shape[0], 1, 1, 1)

        if latents is None and add_noise:
            noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            # if strength is 1. then initialise the latents to noise, else initial to image + noise
            latents = noise if is_strength_max else components.scheduler.add_noise(image_latents, noise, timestep)
            # if pure noise then scale the initial latents by the  Scheduler's init sigma
            latents = latents * components.scheduler.init_noise_sigma if is_strength_max else latents
        elif add_noise:
            noise = latents.to(device)
            latents = noise * components.scheduler.init_noise_sigma
        else:
            noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            latents = image_latents.to(device)

        outputs = (latents, noise, image_latents)

        return outputs

    # modified from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_inpaint.StableDiffusionXLInpaintPipeline.prepare_mask_latents
    # do not accept do_classifier_free_guidance
    def prepare_mask_latents(
        self, components, mask, masked_image, batch_size, height, width, dtype, device, generator
    ):
        # resize the mask to latents shape as we concatenate the mask to the latents
        # we do that before converting to dtype to avoid breaking in case we're using cpu_offload
        # and half precision
        mask = torch.nn.functional.interpolate(
            mask, size=(height // components.vae_scale_factor, width // components.vae_scale_factor)
        )
        mask = mask.to(device=device, dtype=dtype)

        # duplicate mask and masked_image_latents for each generation per prompt, using mps friendly method
        if mask.shape[0] < batch_size:
            if not batch_size % mask.shape[0] == 0:
                raise ValueError(
                    "The passed mask and the required batch size don't match. Masks are supposed to be duplicated to"
                    f" a total batch size of {batch_size}, but {mask.shape[0]} masks were passed. Make sure the number"
                    " of masks that you pass is divisible by the total requested batch size."
                )
            mask = mask.repeat(batch_size // mask.shape[0], 1, 1, 1)

        if masked_image is not None and masked_image.shape[1] == 4:
            masked_image_latents = masked_image
        else:
            masked_image_latents = None

        if masked_image is not None:
            if masked_image_latents is None:
                masked_image = masked_image.to(device=device, dtype=dtype)
                masked_image_latents = self._encode_vae_image(components, masked_image, generator=generator)

            if masked_image_latents.shape[0] < batch_size:
                if not batch_size % masked_image_latents.shape[0] == 0:
                    raise ValueError(
                        "The passed images and the required batch size don't match. Images are supposed to be duplicated"
                        f" to a total batch size of {batch_size}, but {masked_image_latents.shape[0]} images were passed."
                        " Make sure the number of images that you pass is divisible by the total requested batch size."
                    )
                masked_image_latents = masked_image_latents.repeat(
                    batch_size // masked_image_latents.shape[0], 1, 1, 1
                )

            # aligning device to prevent device errors when concating it with the latent model input
            masked_image_latents = masked_image_latents.to(device=device, dtype=dtype)

        return mask, masked_image_latents

    @torch.no_grad()
    def __call__(self, components: StableDiffusionXLModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        block_state.dtype = block_state.dtype if block_state.dtype is not None else components.vae.dtype
        block_state.device = components._execution_device

        block_state.is_strength_max = block_state.strength == 1.0

        # for non-inpainting specific unet, we do not need masked_image_latents
        if hasattr(components, "unet") and components.unet is not None:
            if components.unet.config.in_channels == 4:
                block_state.masked_image_latents = None

        block_state.add_noise = True if block_state.denoising_start is None else False

        block_state.height = block_state.image_latents.shape[-2] * components.vae_scale_factor
        block_state.width = block_state.image_latents.shape[-1] * components.vae_scale_factor

        block_state.latents, block_state.noise, block_state.image_latents = self.prepare_latents_inpaint(
            components,
            block_state.batch_size * block_state.num_images_per_prompt,
            components.num_channels_latents,
            block_state.height,
            block_state.width,
            block_state.dtype,
            block_state.device,
            block_state.generator,
            block_state.latents,
            image=block_state.image_latents,
            timestep=block_state.latent_timestep,
            is_strength_max=block_state.is_strength_max,
            add_noise=block_state.add_noise,
        )

        # 7. Prepare mask latent variables
        block_state.mask, block_state.masked_image_latents = self.prepare_mask_latents(
            components,
            block_state.mask,
            block_state.masked_image_latents,
            block_state.batch_size * block_state.num_images_per_prompt,
            block_state.height,
            block_state.width,
            block_state.dtype,
            block_state.device,
            block_state.generator,
        )

        self.set_block_state(state, block_state)

        return components, state


class StableDiffusionXLImg2ImgPrepareLatentsStep(ModularPipelineBlocks):
    model_name = "stable-diffusion-xl"

    @property
    def expected_components(self) -> List[ComponentSpec]:
        return [
            ComponentSpec("vae", AutoencoderKL),
            ComponentSpec("scheduler", EulerDiscreteScheduler),
        ]

    @property
    def description(self) -> str:
        return "Step that prepares the latents for the image-to-image generation process"

    @property
    def inputs(self) -> List[Tuple[str, Any]]:
        return [
            InputParam("latents"),
            InputParam("num_images_per_prompt", default=1),
            InputParam("denoising_start"),
            InputParam("generator"),
            InputParam(
                "latent_timestep",
                required=True,
                type_hint=torch.Tensor,
                description="The timestep that represents the initial noise level for image-to-image/inpainting generation. Can be generated in set_timesteps step.",
            ),
            InputParam(
                "image_latents",
                required=True,
                type_hint=torch.Tensor,
                description="The latents representing the reference image for image-to-image/inpainting generation. Can be generated in vae_encode step.",
            ),
            InputParam(
                "batch_size",
                required=True,
                type_hint=int,
                description="Number of prompts, the final batch size of model inputs should be batch_size * num_images_per_prompt. Can be generated in input step.",
            ),
            InputParam("dtype", required=True, type_hint=torch.dtype, description="The dtype of the model inputs"),
        ]

    @property
    def intermediate_outputs(self) -> List[OutputParam]:
        return [
            OutputParam(
                "latents", type_hint=torch.Tensor, description="The initial latents to use for the denoising process"
            )
        ]

    @torch.no_grad()
    def __call__(self, components: StableDiffusionXLModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        block_state.dtype = block_state.dtype if block_state.dtype is not None else components.vae.dtype
        block_state.device = components._execution_device
        block_state.add_noise = True if block_state.denoising_start is None else False
        if block_state.latents is None:
            block_state.latents = prepare_latents_img2img(
                components.vae,
                components.scheduler,
                block_state.image_latents,
                block_state.latent_timestep,
                block_state.batch_size,
                block_state.num_images_per_prompt,
                block_state.dtype,
                block_state.device,
                block_state.generator,
                block_state.add_noise,
            )

        self.set_block_state(state, block_state)

        return components, state


class StableDiffusionXLPrepareLatentsStep(ModularPipelineBlocks):
    model_name = "stable-diffusion-xl"

    @property
    def expected_components(self) -> List[ComponentSpec]:
        return [
            ComponentSpec("scheduler", EulerDiscreteScheduler),
            ComponentSpec("vae", AutoencoderKL),
        ]

    @property
    def description(self) -> str:
        return "Prepare latents step that prepares the latents for the text-to-image generation process"

    @property
    def inputs(self) -> List[InputParam]:
        return [
            InputParam("height"),
            InputParam("width"),
            InputParam("latents"),
            InputParam("num_images_per_prompt", default=1),
            InputParam("generator"),
            InputParam(
                "batch_size",
                required=True,
                type_hint=int,
                description="Number of prompts, the final batch size of model inputs should be batch_size * num_images_per_prompt. Can be generated in input step.",
            ),
            InputParam("dtype", type_hint=torch.dtype, description="The dtype of the model inputs"),
        ]

    @property
    def intermediate_outputs(self) -> List[OutputParam]:
        return [
            OutputParam(
                "latents", type_hint=torch.Tensor, description="The initial latents to use for the denoising process"
            )
        ]

    @staticmethod
    def check_inputs(components, block_state):
        if (
            block_state.height is not None
            and block_state.height % components.vae_scale_factor != 0
            or block_state.width is not None
            and block_state.width % components.vae_scale_factor != 0
        ):
            raise ValueError(
                f"`height` and `width` have to be divisible by {components.vae_scale_factor} but are {block_state.height} and {block_state.width}."
            )

    @staticmethod
    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents with self->comp
    def prepare_latents(comp, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        shape = (
            batch_size,
            num_channels_latents,
            int(height) // comp.vae_scale_factor,
            int(width) // comp.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * comp.scheduler.init_noise_sigma
        return latents

    @torch.no_grad()
    def __call__(self, components: StableDiffusionXLModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        if block_state.dtype is None:
            block_state.dtype = components.vae.dtype

        block_state.device = components._execution_device

        self.check_inputs(components, block_state)

        block_state.height = block_state.height or components.default_sample_size * components.vae_scale_factor
        block_state.width = block_state.width or components.default_sample_size * components.vae_scale_factor
        block_state.num_channels_latents = components.num_channels_latents
        block_state.latents = self.prepare_latents(
            components,
            block_state.batch_size * block_state.num_images_per_prompt,
            block_state.num_channels_latents,
            block_state.height,
            block_state.width,
            block_state.dtype,
            block_state.device,
            block_state.generator,
            block_state.latents,
        )

        self.set_block_state(state, block_state)

        return components, state


class StableDiffusionXLImg2ImgPrepareAdditionalConditioningStep(ModularPipelineBlocks):
    model_name = "stable-diffusion-xl"

    @property
    def expected_configs(self) -> List[ConfigSpec]:
        return [
            ConfigSpec("requires_aesthetics_score", False),
        ]

    @property
    def expected_components(self) -> List[ComponentSpec]:
        return [
            ComponentSpec("unet", UNet2DConditionModel),
            ComponentSpec(
                "guider",
                ClassifierFreeGuidance,
                config=FrozenDict({"guidance_scale": 7.5}),
                default_creation_method="from_config",
            ),
        ]

    @property
    def description(self) -> str:
        return "Step that prepares the additional conditioning for the image-to-image/inpainting generation process"

    @property
    def inputs(self) -> List[Tuple[str, Any]]:
        return [
            InputParam("original_size"),
            InputParam("target_size"),
            InputParam("negative_original_size"),
            InputParam("negative_target_size"),
            InputParam("crops_coords_top_left", default=(0, 0)),
            InputParam("negative_crops_coords_top_left", default=(0, 0)),
            InputParam("num_images_per_prompt", default=1),
            InputParam("aesthetic_score", default=6.0),
            InputParam("negative_aesthetic_score", default=2.0),
            InputParam(
                "latents",
                required=True,
                type_hint=torch.Tensor,
                description="The initial latents to use for the denoising process. Can be generated in prepare_latent step.",
            ),
            InputParam(
                "pooled_prompt_embeds",
                required=True,
                type_hint=torch.Tensor,
                description="The pooled prompt embeddings to use for the denoising process (used to determine shapes and dtypes for other additional conditioning inputs). Can be generated in text_encoder step.",
            ),
            InputParam(
                "batch_size",
                required=True,
                type_hint=int,
                description="Number of prompts, the final batch size of model inputs should be batch_size * num_images_per_prompt. Can be generated in input step.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> List[OutputParam]:
        return [
            OutputParam(
                "add_time_ids",
                type_hint=torch.Tensor,
                kwargs_type="denoiser_input_fields",
                description="The time ids to condition the denoising process",
            ),
            OutputParam(
                "negative_add_time_ids",
                type_hint=torch.Tensor,
                kwargs_type="denoiser_input_fields",
                description="The negative time ids to condition the denoising process",
            ),
            OutputParam("timestep_cond", type_hint=torch.Tensor, description="The timestep cond to use for LCM"),
        ]

    @staticmethod
    # Copied from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_img2img.StableDiffusionXLImg2ImgPipeline._get_add_time_ids with self->components
    def _get_add_time_ids(
        components,
        original_size,
        crops_coords_top_left,
        target_size,
        aesthetic_score,
        negative_aesthetic_score,
        negative_original_size,
        negative_crops_coords_top_left,
        negative_target_size,
        dtype,
        text_encoder_projection_dim=None,
    ):
        if components.config.requires_aesthetics_score:
            add_time_ids = list(original_size + crops_coords_top_left + (aesthetic_score,))
            add_neg_time_ids = list(
                negative_original_size + negative_crops_coords_top_left + (negative_aesthetic_score,)
            )
        else:
            add_time_ids = list(original_size + crops_coords_top_left + target_size)
            add_neg_time_ids = list(negative_original_size + crops_coords_top_left + negative_target_size)

        passed_add_embed_dim = (
            components.unet.config.addition_time_embed_dim * len(add_time_ids) + text_encoder_projection_dim
        )
        expected_add_embed_dim = components.unet.add_embedding.linear_1.in_features

        if (
            expected_add_embed_dim > passed_add_embed_dim
            and (expected_add_embed_dim - passed_add_embed_dim) == components.unet.config.addition_time_embed_dim
        ):
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. Please make sure to enable `requires_aesthetics_score` with `pipe.register_to_config(requires_aesthetics_score=True)` to make sure `aesthetic_score` {aesthetic_score} and `negative_aesthetic_score` {negative_aesthetic_score} is correctly used by the model."
            )
        elif (
            expected_add_embed_dim < passed_add_embed_dim
            and (passed_add_embed_dim - expected_add_embed_dim) == components.unet.config.addition_time_embed_dim
        ):
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. Please make sure to disable `requires_aesthetics_score` with `pipe.register_to_config(requires_aesthetics_score=False)` to make sure `target_size` {target_size} is correctly used by the model."
            )
        elif expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        add_neg_time_ids = torch.tensor([add_neg_time_ids], dtype=dtype)

        return add_time_ids, add_neg_time_ids

    # Copied from diffusers.pipelines.latent_consistency_models.pipeline_latent_consistency_text2img.LatentConsistencyModelPipeline.get_guidance_scale_embedding
    def get_guidance_scale_embedding(
        self, w: torch.Tensor, embedding_dim: int = 512, dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """
        See https://github.com/google-research/vdm/blob/dc27b98a554f65cdc654b800da5aa1846545d41b/model_vdm.py#L298

        Args:
            w (`torch.Tensor`):
                Generate embedding vectors with a specified guidance scale to subsequently enrich timestep embeddings.
            embedding_dim (`int`, *optional*, defaults to 512):
                Dimension of the embeddings to generate.
            dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
                Data type of the generated embeddings.

        Returns:
            `torch.Tensor`: Embedding vectors with shape `(len(w), embedding_dim)`.
        """
        assert len(w.shape) == 1
        w = w * 1000.0

        half_dim = embedding_dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
        emb = w.to(dtype)[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:  # zero pad
            emb = torch.nn.functional.pad(emb, (0, 1))
        assert emb.shape == (w.shape[0], embedding_dim)
        return emb

    @torch.no_grad()
    def __call__(self, components: StableDiffusionXLModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        block_state.device = components._execution_device

        block_state.vae_scale_factor = components.vae_scale_factor

        block_state.height, block_state.width = block_state.latents.shape[-2:]
        block_state.height = block_state.height * block_state.vae_scale_factor
        block_state.width = block_state.width * block_state.vae_scale_factor

        block_state.original_size = block_state.original_size or (block_state.height, block_state.width)
        block_state.target_size = block_state.target_size or (block_state.height, block_state.width)

        block_state.text_encoder_projection_dim = int(block_state.pooled_prompt_embeds.shape[-1])

        if block_state.negative_original_size is None:
            block_state.negative_original_size = block_state.original_size
        if block_state.negative_target_size is None:
            block_state.negative_target_size = block_state.target_size

        block_state.add_time_ids, block_state.negative_add_time_ids = self._get_add_time_ids(
            components,
            block_state.original_size,
            block_state.crops_coords_top_left,
            block_state.target_size,
            block_state.aesthetic_score,
            block_state.negative_aesthetic_score,
            block_state.negative_original_size,
            block_state.negative_crops_coords_top_left,
            block_state.negative_target_size,
            dtype=block_state.pooled_prompt_embeds.dtype,
            text_encoder_projection_dim=block_state.text_encoder_projection_dim,
        )
        block_state.add_time_ids = block_state.add_time_ids.repeat(
            block_state.batch_size * block_state.num_images_per_prompt, 1
        ).to(device=block_state.device)
        block_state.negative_add_time_ids = block_state.negative_add_time_ids.repeat(
            block_state.batch_size * block_state.num_images_per_prompt, 1
        ).to(device=block_state.device)

        # Optionally get Guidance Scale Embedding for LCM
        block_state.timestep_cond = None
        if (
            hasattr(components, "unet")
            and components.unet is not None
            and components.unet.config.time_cond_proj_dim is not None
        ):
            # TODO(yiyi, aryan): Ideally, this should be `embedded_guidance_scale` instead of pulling from guider. Guider scales should be different from this!
            block_state.guidance_scale_tensor = torch.tensor(components.guider.guidance_scale - 1).repeat(
                block_state.batch_size * block_state.num_images_per_prompt
            )
            block_state.timestep_cond = self.get_guidance_scale_embedding(
                block_state.guidance_scale_tensor, embedding_dim=components.unet.config.time_cond_proj_dim
            ).to(device=block_state.device, dtype=block_state.latents.dtype)

        self.set_block_state(state, block_state)
        return components, state


class StableDiffusionXLPrepareAdditionalConditioningStep(ModularPipelineBlocks):
    model_name = "stable-diffusion-xl"

    @property
    def description(self) -> str:
        return "Step that prepares the additional conditioning for the text-to-image generation process"

    @property
    def expected_components(self) -> List[ComponentSpec]:
        return [
            ComponentSpec("unet", UNet2DConditionModel),
            ComponentSpec(
                "guider",
                ClassifierFreeGuidance,
                config=FrozenDict({"guidance_scale": 7.5}),
                default_creation_method="from_config",
            ),
        ]

    @property
    def inputs(self) -> List[Tuple[str, Any]]:
        return [
            InputParam("original_size"),
            InputParam("target_size"),
            InputParam("negative_original_size"),
            InputParam("negative_target_size"),
            InputParam("crops_coords_top_left", default=(0, 0)),
            InputParam("negative_crops_coords_top_left", default=(0, 0)),
            InputParam("num_images_per_prompt", default=1),
            InputParam(
                "latents",
                required=True,
                type_hint=torch.Tensor,
                description="The initial latents to use for the denoising process. Can be generated in prepare_latent step.",
            ),
            InputParam(
                "pooled_prompt_embeds",
                required=True,
                type_hint=torch.Tensor,
                description="The pooled prompt embeddings to use for the denoising process (used to determine shapes and dtypes for other additional conditioning inputs). Can be generated in text_encoder step.",
            ),
            InputParam(
                "batch_size",
                required=True,
                type_hint=int,
                description="Number of prompts, the final batch size of model inputs should be batch_size * num_images_per_prompt. Can be generated in input step.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> List[OutputParam]:
        return [
            OutputParam(
                "add_time_ids",
                type_hint=torch.Tensor,
                kwargs_type="denoiser_input_fields",
                description="The time ids to condition the denoising process",
            ),
            OutputParam(
                "negative_add_time_ids",
                type_hint=torch.Tensor,
                kwargs_type="denoiser_input_fields",
                description="The negative time ids to condition the denoising process",
            ),
            OutputParam("timestep_cond", type_hint=torch.Tensor, description="The timestep cond to use for LCM"),
        ]

    @staticmethod
    # Copied from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl.StableDiffusionXLPipeline._get_add_time_ids with self->components
    def _get_add_time_ids(
        components, original_size, crops_coords_top_left, target_size, dtype, text_encoder_projection_dim=None
    ):
        add_time_ids = list(original_size + crops_coords_top_left + target_size)

        passed_add_embed_dim = (
            components.unet.config.addition_time_embed_dim * len(add_time_ids) + text_encoder_projection_dim
        )
        expected_add_embed_dim = components.unet.add_embedding.linear_1.in_features

        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        return add_time_ids

    # Copied from diffusers.pipelines.latent_consistency_models.pipeline_latent_consistency_text2img.LatentConsistencyModelPipeline.get_guidance_scale_embedding
    def get_guidance_scale_embedding(
        self, w: torch.Tensor, embedding_dim: int = 512, dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """
        See https://github.com/google-research/vdm/blob/dc27b98a554f65cdc654b800da5aa1846545d41b/model_vdm.py#L298

        Args:
            w (`torch.Tensor`):
                Generate embedding vectors with a specified guidance scale to subsequently enrich timestep embeddings.
            embedding_dim (`int`, *optional*, defaults to 512):
                Dimension of the embeddings to generate.
            dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
                Data type of the generated embeddings.

        Returns:
            `torch.Tensor`: Embedding vectors with shape `(len(w), embedding_dim)`.
        """
        assert len(w.shape) == 1
        w = w * 1000.0

        half_dim = embedding_dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
        emb = w.to(dtype)[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:  # zero pad
            emb = torch.nn.functional.pad(emb, (0, 1))
        assert emb.shape == (w.shape[0], embedding_dim)
        return emb

    @torch.no_grad()
    def __call__(self, components: StableDiffusionXLModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        block_state.device = components._execution_device

        block_state.height, block_state.width = block_state.latents.shape[-2:]
        block_state.height = block_state.height * components.vae_scale_factor
        block_state.width = block_state.width * components.vae_scale_factor

        block_state.original_size = block_state.original_size or (block_state.height, block_state.width)
        block_state.target_size = block_state.target_size or (block_state.height, block_state.width)

        block_state.text_encoder_projection_dim = int(block_state.pooled_prompt_embeds.shape[-1])

        block_state.add_time_ids = self._get_add_time_ids(
            components,
            block_state.original_size,
            block_state.crops_coords_top_left,
            block_state.target_size,
            block_state.pooled_prompt_embeds.dtype,
            text_encoder_projection_dim=block_state.text_encoder_projection_dim,
        )
        if block_state.negative_original_size is not None and block_state.negative_target_size is not None:
            block_state.negative_add_time_ids = self._get_add_time_ids(
                components,
                block_state.negative_original_size,
                block_state.negative_crops_coords_top_left,
                block_state.negative_target_size,
                block_state.pooled_prompt_embeds.dtype,
                text_encoder_projection_dim=block_state.text_encoder_projection_dim,
            )
        else:
            block_state.negative_add_time_ids = block_state.add_time_ids

        block_state.add_time_ids = block_state.add_time_ids.repeat(
            block_state.batch_size * block_state.num_images_per_prompt, 1
        ).to(device=block_state.device)
        block_state.negative_add_time_ids = block_state.negative_add_time_ids.repeat(
            block_state.batch_size * block_state.num_images_per_prompt, 1
        ).to(device=block_state.device)

        # Optionally get Guidance Scale Embedding for LCM
        block_state.timestep_cond = None
        if (
            hasattr(components, "unet")
            and components.unet is not None
            and components.unet.config.time_cond_proj_dim is not None
        ):
            # TODO(yiyi, aryan): Ideally, this should be `embedded_guidance_scale` instead of pulling from guider. Guider scales should be different from this!
            block_state.guidance_scale_tensor = torch.tensor(components.guider.guidance_scale - 1).repeat(
                block_state.batch_size * block_state.num_images_per_prompt
            )
            block_state.timestep_cond = self.get_guidance_scale_embedding(
                block_state.guidance_scale_tensor, embedding_dim=components.unet.config.time_cond_proj_dim
            ).to(device=block_state.device, dtype=block_state.latents.dtype)

        self.set_block_state(state, block_state)
        return components, state


class StableDiffusionXLControlNetInputStep(ModularPipelineBlocks):
    model_name = "stable-diffusion-xl"

    @property
    def expected_components(self) -> List[ComponentSpec]:
        return [
            ComponentSpec("controlnet", ControlNetModel),
            ComponentSpec(
                "control_image_processor",
                VaeImageProcessor,
                config=FrozenDict({"do_convert_rgb": True, "do_normalize": False}),
                default_creation_method="from_config",
            ),
        ]

    @property
    def description(self) -> str:
        return "step that prepare inputs for controlnet"

    @property
    def inputs(self) -> List[Tuple[str, Any]]:
        return [
            InputParam("control_image", required=True),
            InputParam("control_guidance_start", default=0.0),
            InputParam("control_guidance_end", default=1.0),
            InputParam("controlnet_conditioning_scale", default=1.0),
            InputParam("guess_mode", default=False),
            InputParam("num_images_per_prompt", default=1),
            InputParam(
                "latents",
                required=True,
                type_hint=torch.Tensor,
                description="The initial latents to use for the denoising process. Can be generated in prepare_latent step.",
            ),
            InputParam(
                "batch_size",
                required=True,
                type_hint=int,
                description="Number of prompts, the final batch size of model inputs should be batch_size * num_images_per_prompt. Can be generated in input step.",
            ),
            InputParam(
                "timesteps",
                required=True,
                type_hint=torch.Tensor,
                description="The timesteps to use for the denoising process. Can be generated in set_timesteps step.",
            ),
            InputParam(
                "crops_coords",
                type_hint=Optional[Tuple[int]],
                description="The crop coordinates to use for preprocess/postprocess the image and mask, for inpainting task only. Can be generated in vae_encode step.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> List[OutputParam]:
        return [
            OutputParam("controlnet_cond", type_hint=torch.Tensor, description="The processed control image"),
            OutputParam(
                "control_guidance_start", type_hint=List[float], description="The controlnet guidance start values"
            ),
            OutputParam(
                "control_guidance_end", type_hint=List[float], description="The controlnet guidance end values"
            ),
            OutputParam(
                "conditioning_scale", type_hint=List[float], description="The controlnet conditioning scale values"
            ),
            OutputParam("guess_mode", type_hint=bool, description="Whether guess mode is used"),
            OutputParam("controlnet_keep", type_hint=List[float], description="The controlnet keep values"),
        ]

    # Modified from diffusers.pipelines.controlnet.pipeline_controlnet_sd_xl.StableDiffusionXLControlNetPipeline.prepare_image
    # 1. return image without apply any guidance
    # 2. add crops_coords and resize_mode to preprocess()
    @staticmethod
    def prepare_control_image(
        components,
        image,
        width,
        height,
        batch_size,
        num_images_per_prompt,
        device,
        dtype,
        crops_coords=None,
    ):
        if crops_coords is not None:
            image = components.control_image_processor.preprocess(
                image, height=height, width=width, crops_coords=crops_coords, resize_mode="fill"
            ).to(dtype=torch.float32)
        else:
            image = components.control_image_processor.preprocess(image, height=height, width=width).to(
                dtype=torch.float32
            )

        image_batch_size = image.shape[0]
        if image_batch_size == 1:
            repeat_by = batch_size
        else:
            # image batch size is the same as prompt batch size
            repeat_by = num_images_per_prompt

        image = image.repeat_interleave(repeat_by, dim=0)
        image = image.to(device=device, dtype=dtype)
        return image

    @torch.no_grad()
    def __call__(self, components: StableDiffusionXLModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        # (1) prepare controlnet inputs
        block_state.device = components._execution_device
        block_state.height, block_state.width = block_state.latents.shape[-2:]
        block_state.height = block_state.height * components.vae_scale_factor
        block_state.width = block_state.width * components.vae_scale_factor

        controlnet = unwrap_module(components.controlnet)

        # (1.1)
        # control_guidance_start/control_guidance_end (align format)
        if not isinstance(block_state.control_guidance_start, list) and isinstance(
            block_state.control_guidance_end, list
        ):
            block_state.control_guidance_start = len(block_state.control_guidance_end) * [
                block_state.control_guidance_start
            ]
        elif not isinstance(block_state.control_guidance_end, list) and isinstance(
            block_state.control_guidance_start, list
        ):
            block_state.control_guidance_end = len(block_state.control_guidance_start) * [
                block_state.control_guidance_end
            ]
        elif not isinstance(block_state.control_guidance_start, list) and not isinstance(
            block_state.control_guidance_end, list
        ):
            mult = len(controlnet.nets) if isinstance(controlnet, MultiControlNetModel) else 1
            block_state.control_guidance_start, block_state.control_guidance_end = (
                mult * [block_state.control_guidance_start],
                mult * [block_state.control_guidance_end],
            )

        # (1.2)
        # controlnet_conditioning_scale (align format)
        if isinstance(controlnet, MultiControlNetModel) and isinstance(
            block_state.controlnet_conditioning_scale, float
        ):
            block_state.controlnet_conditioning_scale = [block_state.controlnet_conditioning_scale] * len(
                controlnet.nets
            )

        # (1.3)
        # global_pool_conditions
        block_state.global_pool_conditions = (
            controlnet.config.global_pool_conditions
            if isinstance(controlnet, ControlNetModel)
            else controlnet.nets[0].config.global_pool_conditions
        )
        # (1.4)
        # guess_mode
        block_state.guess_mode = block_state.guess_mode or block_state.global_pool_conditions

        # (1.5)
        # control_image
        if isinstance(controlnet, ControlNetModel):
            block_state.control_image = self.prepare_control_image(
                components,
                image=block_state.control_image,
                width=block_state.width,
                height=block_state.height,
                batch_size=block_state.batch_size * block_state.num_images_per_prompt,
                num_images_per_prompt=block_state.num_images_per_prompt,
                device=block_state.device,
                dtype=controlnet.dtype,
                crops_coords=block_state.crops_coords,
            )
        elif isinstance(controlnet, MultiControlNetModel):
            control_images = []

            for control_image_ in block_state.control_image:
                control_image = self.prepare_control_image(
                    components,
                    image=control_image_,
                    width=block_state.width,
                    height=block_state.height,
                    batch_size=block_state.batch_size * block_state.num_images_per_prompt,
                    num_images_per_prompt=block_state.num_images_per_prompt,
                    device=block_state.device,
                    dtype=controlnet.dtype,
                    crops_coords=block_state.crops_coords,
                )

                control_images.append(control_image)

            block_state.control_image = control_images
        else:
            assert False

        # (1.6)
        # controlnet_keep
        block_state.controlnet_keep = []
        for i in range(len(block_state.timesteps)):
            keeps = [
                1.0 - float(i / len(block_state.timesteps) < s or (i + 1) / len(block_state.timesteps) > e)
                for s, e in zip(block_state.control_guidance_start, block_state.control_guidance_end)
            ]
            block_state.controlnet_keep.append(keeps[0] if isinstance(controlnet, ControlNetModel) else keeps)

        block_state.controlnet_cond = block_state.control_image
        block_state.conditioning_scale = block_state.controlnet_conditioning_scale

        self.set_block_state(state, block_state)

        return components, state


class StableDiffusionXLControlNetUnionInputStep(ModularPipelineBlocks):
    model_name = "stable-diffusion-xl"

    @property
    def expected_components(self) -> List[ComponentSpec]:
        return [
            ComponentSpec("controlnet", ControlNetUnionModel),
            ComponentSpec(
                "control_image_processor",
                VaeImageProcessor,
                config=FrozenDict({"do_convert_rgb": True, "do_normalize": False}),
                default_creation_method="from_config",
            ),
        ]

    @property
    def description(self) -> str:
        return "step that prepares inputs for the ControlNetUnion model"

    @property
    def inputs(self) -> List[Tuple[str, Any]]:
        return [
            InputParam("control_image", required=True),
            InputParam("control_mode", required=True),
            InputParam("control_guidance_start", default=0.0),
            InputParam("control_guidance_end", default=1.0),
            InputParam("controlnet_conditioning_scale", default=1.0),
            InputParam("guess_mode", default=False),
            InputParam("num_images_per_prompt", default=1),
            InputParam(
                "latents",
                required=True,
                type_hint=torch.Tensor,
                description="The initial latents to use for the denoising process. Used to determine the shape of the control images. Can be generated in prepare_latent step.",
            ),
            InputParam(
                "batch_size",
                required=True,
                type_hint=int,
                description="Number of prompts, the final batch size of model inputs should be batch_size * num_images_per_prompt. Can be generated in input step.",
            ),
            InputParam(
                "dtype",
                required=True,
                type_hint=torch.dtype,
                description="The dtype of model tensor inputs. Can be generated in input step.",
            ),
            InputParam(
                "timesteps",
                required=True,
                type_hint=torch.Tensor,
                description="The timesteps to use for the denoising process. Needed to determine `controlnet_keep`. Can be generated in set_timesteps step.",
            ),
            InputParam(
                "crops_coords",
                type_hint=Optional[Tuple[int]],
                description="The crop coordinates to use for preprocess/postprocess the image and mask, for inpainting task only. Can be generated in vae_encode step.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> List[OutputParam]:
        return [
            OutputParam("controlnet_cond", type_hint=List[torch.Tensor], description="The processed control images"),
            OutputParam(
                "control_type_idx",
                type_hint=List[int],
                description="The control mode indices",
                kwargs_type="controlnet_kwargs",
            ),
            OutputParam(
                "control_type",
                type_hint=torch.Tensor,
                description="The control type tensor that specifies which control type is active",
                kwargs_type="controlnet_kwargs",
            ),
            OutputParam("control_guidance_start", type_hint=float, description="The controlnet guidance start value"),
            OutputParam("control_guidance_end", type_hint=float, description="The controlnet guidance end value"),
            OutputParam(
                "conditioning_scale", type_hint=List[float], description="The controlnet conditioning scale values"
            ),
            OutputParam("guess_mode", type_hint=bool, description="Whether guess mode is used"),
            OutputParam("controlnet_keep", type_hint=List[float], description="The controlnet keep values"),
        ]

    # Modified from diffusers.pipelines.controlnet.pipeline_controlnet_sd_xl.StableDiffusionXLControlNetPipeline.prepare_image
    # 1. return image without apply any guidance
    # 2. add crops_coords and resize_mode to preprocess()
    @staticmethod
    def prepare_control_image(
        components,
        image,
        width,
        height,
        batch_size,
        num_images_per_prompt,
        device,
        dtype,
        crops_coords=None,
    ):
        if crops_coords is not None:
            image = components.control_image_processor.preprocess(
                image, height=height, width=width, crops_coords=crops_coords, resize_mode="fill"
            ).to(dtype=torch.float32)
        else:
            image = components.control_image_processor.preprocess(image, height=height, width=width).to(
                dtype=torch.float32
            )

        image_batch_size = image.shape[0]
        if image_batch_size == 1:
            repeat_by = batch_size
        else:
            # image batch size is the same as prompt batch size
            repeat_by = num_images_per_prompt

        image = image.repeat_interleave(repeat_by, dim=0)
        image = image.to(device=device, dtype=dtype)
        return image

    @torch.no_grad()
    def __call__(self, components: StableDiffusionXLModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        controlnet = unwrap_module(components.controlnet)

        device = components._execution_device
        dtype = block_state.dtype or components.controlnet.dtype

        block_state.height, block_state.width = block_state.latents.shape[-2:]
        block_state.height = block_state.height * components.vae_scale_factor
        block_state.width = block_state.width * components.vae_scale_factor

        # control_guidance_start/control_guidance_end (align format)
        if not isinstance(block_state.control_guidance_start, list) and isinstance(
            block_state.control_guidance_end, list
        ):
            block_state.control_guidance_start = len(block_state.control_guidance_end) * [
                block_state.control_guidance_start
            ]
        elif not isinstance(block_state.control_guidance_end, list) and isinstance(
            block_state.control_guidance_start, list
        ):
            block_state.control_guidance_end = len(block_state.control_guidance_start) * [
                block_state.control_guidance_end
            ]

        # guess_mode
        block_state.global_pool_conditions = controlnet.config.global_pool_conditions
        block_state.guess_mode = block_state.guess_mode or block_state.global_pool_conditions

        # control_image
        if not isinstance(block_state.control_image, list):
            block_state.control_image = [block_state.control_image]
        # control_mode
        if not isinstance(block_state.control_mode, list):
            block_state.control_mode = [block_state.control_mode]

        if len(block_state.control_image) != len(block_state.control_mode):
            raise ValueError("Expected len(control_image) == len(control_type)")

        # control_type
        block_state.num_control_type = controlnet.config.num_control_type
        block_state.control_type = [0 for _ in range(block_state.num_control_type)]
        for control_idx in block_state.control_mode:
            block_state.control_type[control_idx] = 1
        block_state.control_type = torch.Tensor(block_state.control_type)

        block_state.control_type = block_state.control_type.reshape(1, -1).to(device, dtype=block_state.dtype)
        repeat_by = block_state.batch_size * block_state.num_images_per_prompt // block_state.control_type.shape[0]
        block_state.control_type = block_state.control_type.repeat_interleave(repeat_by, dim=0)

        # prepare control_image
        for idx, _ in enumerate(block_state.control_image):
            block_state.control_image[idx] = self.prepare_control_image(
                components,
                image=block_state.control_image[idx],
                width=block_state.width,
                height=block_state.height,
                batch_size=block_state.batch_size * block_state.num_images_per_prompt,
                num_images_per_prompt=block_state.num_images_per_prompt,
                device=device,
                dtype=dtype,
                crops_coords=block_state.crops_coords,
            )
            block_state.height, block_state.width = block_state.control_image[idx].shape[-2:]

        # controlnet_keep
        block_state.controlnet_keep = []
        for i in range(len(block_state.timesteps)):
            block_state.controlnet_keep.append(
                1.0
                - float(
                    i / len(block_state.timesteps) < block_state.control_guidance_start
                    or (i + 1) / len(block_state.timesteps) > block_state.control_guidance_end
                )
            )
        block_state.control_type_idx = block_state.control_mode
        block_state.controlnet_cond = block_state.control_image
        block_state.conditioning_scale = block_state.controlnet_conditioning_scale

        self.set_block_state(state, block_state)

        return components, state
