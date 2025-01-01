# Copyright 2024 The HuggingFace Team. All rights reserved.
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

from ...guider import CFGGuider
from ...image_processor import VaeImageProcessor
from ...loaders import StableDiffusionXLLoraLoaderMixin, TextualInversionLoaderMixin
from ...models import ControlNetModel, ImageProjection
from ...models.attention_processor import AttnProcessor2_0, XFormersAttnProcessor
from ...models.lora import adjust_lora_scale_text_encoder
from ...utils import (
    USE_PEFT_BACKEND,
    logging,
    scale_lora_layers,
    unscale_lora_layers,
)
from ...utils.torch_utils import is_compiled_module, randn_tensor
from ..controlnet.multicontrolnet import MultiControlNetModel
from ..modular_pipeline import (
    AutoPipelineBlocks,
    ModularPipeline,
    PipelineBlock,
    PipelineState,
    SequentialPipelineBlocks,
)
from ..pipeline_utils import DiffusionPipeline, StableDiffusionMixin
from .pipeline_output import (
    StableDiffusionXLPipelineOutput,
)


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


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


class StableDiffusionXLInputStep(PipelineBlock):
    @property
    def inputs(self) -> List[Tuple[str, Any]]:
        return [
            ("prompt", None),
            ("prompt_embeds", None),
        ]

    @property
    def intermediates_outputs(self) -> List[str]:
        return ["batch_size"]

    @torch.no_grad()
    def __call__(self, pipeline, state: PipelineState) -> PipelineState:
        prompt = state.get_input("prompt")
        prompt_embeds = state.get_input("prompt_embeds")

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        state.add_intermediate("batch_size", batch_size)

        return pipeline, state


class StableDiffusionXLTextEncoderStep(PipelineBlock):
    expected_components = ["text_encoder", "text_encoder_2", "tokenizer", "tokenizer_2"]
    expected_configs = ["force_zeros_for_empty_prompt"]

    @property
    def inputs(self) -> List[Tuple[str, Any]]:
        return [
            ("prompt", None),
            ("prompt_2", None),
            ("negative_prompt", None),
            ("negative_prompt_2", None),
            ("cross_attention_kwargs", None),
            ("prompt_embeds", None),
            ("negative_prompt_embeds", None),
            ("pooled_prompt_embeds", None),
            ("negative_pooled_prompt_embeds", None),
            ("num_images_per_prompt", 1),
            ("guidance_scale", 5.0),
            ("clip_skip", None),
        ]

    @property
    def intermediates_outputs(self) -> List[str]:
        return [
            "prompt_embeds",
            "negative_prompt_embeds",
            "pooled_prompt_embeds",
            "negative_pooled_prompt_embeds",
        ]

    def __init__(self):
        super().__init__()
        self.configs["force_zeros_for_empty_prompt"] = True
        self.components["text_encoder"] = None
        self.components["text_encoder_2"] = None
        self.components["tokenizer"] = None
        self.components["tokenizer_2"] = None

    @staticmethod
    def check_inputs(
        pipeline,
        prompt,
        prompt_2,
        negative_prompt=None,
        negative_prompt_2=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        pooled_prompt_embeds=None,
        negative_pooled_prompt_embeds=None,
    ):
        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt_2 is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt_2`: {prompt_2} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
        elif prompt_2 is not None and (not isinstance(prompt_2, str) and not isinstance(prompt_2, list)):
            raise ValueError(f"`prompt_2` has to be of type `str` or `list` but is {type(prompt_2)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )
        elif negative_prompt_2 is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt_2`: {negative_prompt_2} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

        if prompt_embeds is not None and pooled_prompt_embeds is None:
            raise ValueError(
                "If `prompt_embeds` are provided, `pooled_prompt_embeds` also have to be passed. Make sure to generate `pooled_prompt_embeds` from the same text encoder that was used to generate `prompt_embeds`."
            )

        if negative_prompt_embeds is not None and negative_pooled_prompt_embeds is None:
            raise ValueError(
                "If `negative_prompt_embeds` are provided, `negative_pooled_prompt_embeds` also have to be passed. Make sure to generate `negative_pooled_prompt_embeds` from the same text encoder that was used to generate `negative_prompt_embeds`."
            )

    @torch.no_grad()
    def __call__(self, pipeline, state: PipelineState) -> PipelineState:
        # Get inputs
        prompt = state.get_input("prompt")
        prompt_2 = state.get_input("prompt_2")
        negative_prompt = state.get_input("negative_prompt")
        negative_prompt_2 = state.get_input("negative_prompt_2")
        cross_attention_kwargs = state.get_input("cross_attention_kwargs")
        prompt_embeds = state.get_input("prompt_embeds")
        negative_prompt_embeds = state.get_input("negative_prompt_embeds")
        pooled_prompt_embeds = state.get_input("pooled_prompt_embeds")
        negative_pooled_prompt_embeds = state.get_input("negative_pooled_prompt_embeds")
        num_images_per_prompt = state.get_input("num_images_per_prompt")
        guidance_scale = state.get_input("guidance_scale")
        clip_skip = state.get_input("clip_skip")

        do_classifier_free_guidance = guidance_scale > 1.0
        device = pipeline._execution_device

        self.check_inputs(
            pipeline,
            prompt,
            prompt_2,
            negative_prompt,
            negative_prompt_2,
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        )

        # Encode input prompt
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = pipeline.encode_prompt(
            prompt,
            prompt_2,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
            clip_skip=clip_skip,
        )
        # Add outputs
        state.add_intermediate("prompt_embeds", prompt_embeds)
        state.add_intermediate("negative_prompt_embeds", negative_prompt_embeds)
        state.add_intermediate("pooled_prompt_embeds", pooled_prompt_embeds)
        state.add_intermediate("negative_pooled_prompt_embeds", negative_pooled_prompt_embeds)
        return pipeline, state


class StableDiffusionXLVAEEncoderStep(PipelineBlock):
    expected_components = ["vae"]

    @property
    def inputs(self) -> List[Tuple[str, Any]]:
        return [
            ("image", None),
            ("generator", None),
            ("height", None),
            ("width", None),
            ("device", None),
            ("dtype", None),
        ]

    @property
    def intermediates_inputs(self) -> List[str]:
        return ["batch_size"]

    @property
    def intermediates_outputs(self) -> List[str]:
        return ["image_latents"]

    def __init__(self):
        super().__init__()
        self.components["vae"] = None
        self.auxiliaries["image_processor"] = VaeImageProcessor()

    @torch.no_grad()
    def __call__(self, pipeline, state: PipelineState) -> PipelineState:
        image = state.get_input("image")
        generator = state.get_input("generator")
        height = state.get_input("height")
        width = state.get_input("width")
        device = state.get_input("device")
        dtype = state.get_input("dtype")

        batch_size = state.get_intermediate("batch_size")

        if device is None:
            device = pipeline._execution_device
        if dtype is None:
            dtype = pipeline.vae.dtype

        image = pipeline.image_processor.preprocess(image, height=height, width=width)
        image = image.to(device=device, dtype=dtype)

        latents_mean = latents_std = None
        if hasattr(pipeline.vae.config, "latents_mean") and pipeline.vae.config.latents_mean is not None:
            latents_mean = torch.tensor(pipeline.vae.config.latents_mean).view(1, 4, 1, 1)
        if hasattr(pipeline.vae.config, "latents_std") and pipeline.vae.config.latents_std is not None:
            latents_std = torch.tensor(pipeline.vae.config.latents_std).view(1, 4, 1, 1)

        # make sure the VAE is in float32 mode, as it overflows in float16
        if pipeline.vae.config.force_upcast:
            image = image.float()
            pipeline.vae.to(dtype=torch.float32)

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
                retrieve_latents(pipeline.vae.encode(image[i : i + 1]), generator=generator[i])
                for i in range(batch_size)
            ]
            init_latents = torch.cat(init_latents, dim=0)
        else:
            init_latents = retrieve_latents(pipeline.vae.encode(image), generator=generator)

        if pipeline.vae.config.force_upcast:
            pipeline.vae.to(dtype)

        init_latents = init_latents.to(dtype)
        if latents_mean is not None and latents_std is not None:
            latents_mean = latents_mean.to(device=device, dtype=dtype)
            latents_std = latents_std.to(device=device, dtype=dtype)
            init_latents = (init_latents - latents_mean) * pipeline.vae.config.scaling_factor / latents_std
        else:
            init_latents = pipeline.vae.config.scaling_factor * init_latents

        state.add_intermediate("image_latents", init_latents)

        return pipeline, state


class StableDiffusionXLImg2ImgSetTimestepsStep(PipelineBlock):
    expected_components = ["scheduler"]

    @property
    def inputs(self) -> List[Tuple[str, Any]]:
        return [
            ("num_inference_steps", 50),
            ("timesteps", None),
            ("sigmas", None),
            ("denoising_end", None),
            ("strength", 0.3),
            ("denoising_start", None),
            ("num_images_per_prompt", 1),
            ("device", None),
        ]

    @property
    def intermediates_inputs(self) -> List[str]:
        return ["batch_size"]

    @property
    def intermediates_outputs(self) -> List[str]:
        return ["timesteps", "num_inference_steps", "latent_timestep"]

    def __init__(self):
        super().__init__()
        self.components["scheduler"] = None

    @torch.no_grad()
    def __call__(self, pipeline, state: PipelineState) -> PipelineState:
        num_inference_steps = state.get_input("num_inference_steps")
        timesteps = state.get_input("timesteps")
        sigmas = state.get_input("sigmas")
        denoising_end = state.get_input("denoising_end")
        device = state.get_input("device")

        # image to image only
        strength = state.get_input("strength")
        denoising_start = state.get_input("denoising_start")
        num_images_per_prompt = state.get_input("num_images_per_prompt")

        # image to image only
        batch_size = state.get_intermediate("batch_size")

        if device is None:
            device = pipeline._execution_device

        timesteps, num_inference_steps = retrieve_timesteps(
            pipeline.scheduler, num_inference_steps, device, timesteps, sigmas
        )

        def denoising_value_valid(dnv):
            return isinstance(dnv, float) and 0 < dnv < 1

        timesteps, num_inference_steps = pipeline.get_timesteps(
            num_inference_steps,
            strength,
            device,
            denoising_start=denoising_start if denoising_value_valid(denoising_start) else None,
        )
        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)

        if denoising_end is not None and isinstance(denoising_end, float) and denoising_end > 0 and denoising_end < 1:
            discrete_timestep_cutoff = int(
                round(
                    pipeline.scheduler.config.num_train_timesteps
                    - (denoising_end * pipeline.scheduler.config.num_train_timesteps)
                )
            )
            num_inference_steps = len(list(filter(lambda ts: ts >= discrete_timestep_cutoff, timesteps)))
            timesteps = timesteps[:num_inference_steps]

        state.add_intermediate("timesteps", timesteps)
        state.add_intermediate("num_inference_steps", num_inference_steps)
        state.add_intermediate("latent_timestep", latent_timestep)

        return pipeline, state


class StableDiffusionXLSetTimestepsStep(PipelineBlock):
    expected_components = ["scheduler"]

    @property
    def inputs(self) -> List[Tuple[str, Any]]:
        return [
            ("num_inference_steps", 50),
            ("timesteps", None),
            ("sigmas", None),
            ("denoising_end", None),
            ("device", None),
        ]

    @property
    def intermediates_outputs(self) -> List[str]:
        return ["timesteps", "num_inference_steps"]

    def __init__(self):
        super().__init__()
        self.components["scheduler"] = None

    @torch.no_grad()
    def __call__(self, pipeline, state: PipelineState) -> PipelineState:
        num_inference_steps = state.get_input("num_inference_steps")
        timesteps = state.get_input("timesteps")
        sigmas = state.get_input("sigmas")
        denoising_end = state.get_input("denoising_end")
        device = state.get_input("device")

        if device is None:
            device = pipeline._execution_device

        timesteps, num_inference_steps = retrieve_timesteps(
            pipeline.scheduler, num_inference_steps, device, timesteps, sigmas
        )

        if denoising_end is not None and isinstance(denoising_end, float) and denoising_end > 0 and denoising_end < 1:
            discrete_timestep_cutoff = int(
                round(
                    pipeline.scheduler.config.num_train_timesteps
                    - (denoising_end * pipeline.scheduler.config.num_train_timesteps)
                )
            )
            num_inference_steps = len(list(filter(lambda ts: ts >= discrete_timestep_cutoff, timesteps)))
            timesteps = timesteps[:num_inference_steps]

        state.add_intermediate("timesteps", timesteps)
        state.add_intermediate("num_inference_steps", num_inference_steps)

        return pipeline, state


class StableDiffusionXLImg2ImgPrepareLatentsStep(PipelineBlock):
    expected_components = ["vae", "scheduler"]

    @property
    def inputs(self) -> List[Tuple[str, Any]]:
        return [
            ("height", None),
            ("width", None),
            ("generator", None),
            ("latents", None),
            ("num_images_per_prompt", 1),
            ("device", None),
            ("dtype", None),
            ("image", None),
            ("denoising_start", None),
        ]

    @property
    def intermediates_inputs(self) -> List[str]:
        return ["batch_size", "latent_timestep", "prompt_embeds"]

    @property
    def intermediates_outputs(self) -> List[str]:
        return ["latents"]

    def __init__(self):
        super().__init__()
        self.auxiliaries["image_processor"] = VaeImageProcessor()
        self.components["vae"] = None
        self.components["scheduler"] = None

    @torch.no_grad()
    def __call__(self, pipeline: DiffusionPipeline, state: PipelineState) -> PipelineState:
        latents = state.get_input("latents")
        num_images_per_prompt = state.get_input("num_images_per_prompt")
        generator = state.get_input("generator")
        device = state.get_input("device")
        dtype = state.get_input("dtype")

        # image to image only
        image = state.get_input("image")
        denoising_start = state.get_input("denoising_start")

        batch_size = state.get_intermediate("batch_size")
        prompt_embeds = state.get_intermediate("prompt_embeds")
        # image to image only
        latent_timestep = state.get_intermediate("latent_timestep")

        if dtype is None and prompt_embeds is not None:
            dtype = prompt_embeds.dtype
        elif dtype is None:
            dtype = pipeline.vae.dtype

        if device is None:
            device = pipeline._execution_device

        image = pipeline.image_processor.preprocess(image)
        add_noise = True if denoising_start is None else False
        if latents is None:
            latents = pipeline.prepare_latents_img2img(
                image,
                latent_timestep,
                batch_size,
                num_images_per_prompt,
                dtype,
                device,
                generator,
                add_noise,
            )

        state.add_intermediate("latents", latents)

        return pipeline, state


class StableDiffusionXLPrepareLatentsStep(PipelineBlock):
    expected_components = ["scheduler"]

    @property
    def inputs(self) -> List[Tuple[str, Any]]:
        return [
            ("height", None),
            ("width", None),
            ("generator", None),
            ("latents", None),
            ("num_images_per_prompt", 1),
            ("device", None),
            ("dtype", None),
        ]

    @property
    def intermediates_inputs(self) -> List[str]:
        return ["batch_size", "prompt_embeds"]

    @property
    def intermediates_outputs(self) -> List[str]:
        return ["latents"]

    def __init__(self):
        super().__init__()
        self.components["scheduler"] = None

    @staticmethod
    def check_inputs(pipeline, height, width):
        if (
            height is not None
            and height % pipeline.vae_scale_factor != 0
            or width is not None
            and width % pipeline.vae_scale_factor != 0
        ):
            raise ValueError(
                f"`height` and `width` have to be divisible by {pipeline.vae_scale_factor} but are {height} and {width}."
            )

    @torch.no_grad()
    def __call__(self, pipeline: DiffusionPipeline, state: PipelineState) -> PipelineState:
        latents = state.get_input("latents")
        num_images_per_prompt = state.get_input("num_images_per_prompt")
        generator = state.get_input("generator")
        device = state.get_input("device")
        dtype = state.get_input("dtype")

        # text to image only
        height = state.get_input("height")
        width = state.get_input("width")

        batch_size = state.get_intermediate("batch_size")
        prompt_embeds = state.get_intermediate("prompt_embeds")

        if dtype is None:
            dtype = prompt_embeds.dtype

        if device is None:
            device = pipeline._execution_device

        self.check_inputs(pipeline, height, width)

        height = height or pipeline.default_sample_size * pipeline.vae_scale_factor
        width = width or pipeline.default_sample_size * pipeline.vae_scale_factor
        num_channels_latents = pipeline.num_channels_latents
        latents = pipeline.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            dtype,
            device,
            generator,
            latents,
        )

        state.add_intermediate("latents", latents)

        return pipeline, state


class StableDiffusionXLImg2ImgPrepareAdditionalConditioningStep(PipelineBlock):
    expected_configs = ["requires_aesthetics_score"]

    @property
    def inputs(self) -> List[Tuple[str, Any]]:
        return [
            ("original_size", None),
            ("target_size", None),
            ("negative_original_size", None),
            ("negative_target_size", None),
            ("crops_coords_top_left", (0, 0)),
            ("negative_crops_coords_top_left", (0, 0)),
            ("num_images_per_prompt", 1),
            ("guidance_scale", 5.0),
            ("aesthetic_score", 6.0),
            ("negative_aesthetic_score", 2.0),
            ("device", None),
        ]

    @property
    def intermediates_inputs(self) -> List[str]:
        return ["latents", "batch_size", "pooled_prompt_embeds"]

    @property
    def intermediates_outputs(self) -> List[str]:
        return ["add_time_ids", "negative_add_time_ids", "timestep_cond"]

    def __init__(self):
        super().__init__()
        self.configs["requires_aesthetics_score"] = False

    @torch.no_grad()
    def __call__(self, pipeline: DiffusionPipeline, state: PipelineState) -> PipelineState:
        original_size = state.get_input("original_size")
        target_size = state.get_input("target_size")
        negative_original_size = state.get_input("negative_original_size")
        negative_target_size = state.get_input("negative_target_size")
        crops_coords_top_left = state.get_input("crops_coords_top_left")
        negative_crops_coords_top_left = state.get_input("negative_crops_coords_top_left")
        num_images_per_prompt = state.get_input("num_images_per_prompt")
        guidance_scale = state.get_input("guidance_scale")
        device = state.get_input("device")

        # image to image only
        aesthetic_score = state.get_input("aesthetic_score")
        negative_aesthetic_score = state.get_input("negative_aesthetic_score")

        latents = state.get_intermediate("latents")
        batch_size = state.get_intermediate("batch_size")
        pooled_prompt_embeds = state.get_intermediate("pooled_prompt_embeds")

        if device is None:
            device = pipeline._execution_device

        height, width = latents.shape[-2:]
        height = height * pipeline.vae_scale_factor
        width = width * pipeline.vae_scale_factor

        original_size = original_size or (height, width)
        target_size = target_size or (height, width)

        if hasattr(pipeline, "text_encoder_2") and pipeline.text_encoder_2 is not None:
            text_encoder_projection_dim = pipeline.text_encoder_2.config.projection_dim
        else:
            text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])

        if negative_original_size is None:
            negative_original_size = original_size
        if negative_target_size is None:
            negative_target_size = target_size

        add_time_ids, negative_add_time_ids = pipeline._get_add_time_ids_img2img(
            original_size,
            crops_coords_top_left,
            target_size,
            aesthetic_score,
            negative_aesthetic_score,
            negative_original_size,
            negative_crops_coords_top_left,
            negative_target_size,
            dtype=pooled_prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )
        add_time_ids = add_time_ids.repeat(batch_size * num_images_per_prompt, 1).to(device=device)
        negative_add_time_ids = negative_add_time_ids.repeat(batch_size * num_images_per_prompt, 1).to(device=device)

        # Optionally get Guidance Scale Embedding for LCM
        timestep_cond = None
        if (
            hasattr(pipeline, "unet")
            and pipeline.unet is not None
            and pipeline.unet.config.time_cond_proj_dim is not None
        ):
            guidance_scale_tensor = torch.tensor(guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = pipeline.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=pipeline.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        state.add_intermediate("add_time_ids", add_time_ids)
        state.add_intermediate("negative_add_time_ids", negative_add_time_ids)
        state.add_intermediate("timestep_cond", timestep_cond)
        return pipeline, state


class StableDiffusionXLPrepareAdditionalConditioningStep(PipelineBlock):
    @property
    def inputs(self) -> List[Tuple[str, Any]]:
        return [
            ("original_size", None),
            ("target_size", None),
            ("negative_original_size", None),
            ("negative_target_size", None),
            ("crops_coords_top_left", (0, 0)),
            ("negative_crops_coords_top_left", (0, 0)),
            ("num_images_per_prompt", 1),
            ("guidance_scale", 5.0),
            ("device", None),
        ]

    @property
    def intermediates_inputs(self) -> List[str]:
        return ["latents", "batch_size", "pooled_prompt_embeds"]

    @property
    def intermediates_outputs(self) -> List[str]:
        return ["add_time_ids", "negative_add_time_ids", "timestep_cond"]

    @torch.no_grad()
    def __call__(self, pipeline: DiffusionPipeline, state: PipelineState) -> PipelineState:
        original_size = state.get_input("original_size")
        target_size = state.get_input("target_size")
        negative_original_size = state.get_input("negative_original_size")
        negative_target_size = state.get_input("negative_target_size")
        crops_coords_top_left = state.get_input("crops_coords_top_left")
        negative_crops_coords_top_left = state.get_input("negative_crops_coords_top_left")
        num_images_per_prompt = state.get_input("num_images_per_prompt")
        guidance_scale = state.get_input("guidance_scale")
        device = state.get_input("device")

        latents = state.get_intermediate("latents")
        batch_size = state.get_intermediate("batch_size")
        pooled_prompt_embeds = state.get_intermediate("pooled_prompt_embeds")

        if device is None:
            device = pipeline._execution_device

        height, width = latents.shape[-2:]
        height = height * pipeline.vae_scale_factor
        width = width * pipeline.vae_scale_factor

        original_size = original_size or (height, width)
        target_size = target_size or (height, width)

        if hasattr(pipeline, "text_encoder_2") and pipeline.text_encoder_2 is not None:
            text_encoder_projection_dim = pipeline.text_encoder_2.config.projection_dim
        else:
            text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])

        add_time_ids = pipeline._get_add_time_ids(
            original_size,
            crops_coords_top_left,
            target_size,
            pooled_prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )
        add_time_ids = add_time_ids.repeat(batch_size * num_images_per_prompt, 1).to(device=device)

        if negative_original_size is not None and negative_target_size is not None:
            negative_add_time_ids = pipeline._get_add_time_ids(
                negative_original_size,
                negative_crops_coords_top_left,
                negative_target_size,
                pooled_prompt_embeds.dtype,
                text_encoder_projection_dim=text_encoder_projection_dim,
            )
        else:
            negative_add_time_ids = add_time_ids
        negative_add_time_ids = negative_add_time_ids.repeat(batch_size * num_images_per_prompt, 1).to(device=device)

        # Optionally get Guidance Scale Embedding for LCM
        timestep_cond = None
        if (
            hasattr(pipeline, "unet")
            and pipeline.unet is not None
            and pipeline.unet.config.time_cond_proj_dim is not None
        ):
            guidance_scale_tensor = torch.tensor(guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = pipeline.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=pipeline.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        state.add_intermediate("add_time_ids", add_time_ids)
        state.add_intermediate("negative_add_time_ids", negative_add_time_ids)
        state.add_intermediate("timestep_cond", timestep_cond)
        return pipeline, state


class StableDiffusionXLDenoiseStep(PipelineBlock):
    expected_components = ["unet", "scheduler", "guider"]

    @property
    def inputs(self) -> List[Tuple[str, Any]]:
        return [
            ("guidance_scale", 5.0),
            ("guidance_rescale", 0.0),
            ("cross_attention_kwargs", None),
            ("generator", None),
            ("eta", 0.0),
            ("guider_kwargs", None),
        ]

    @property
    def intermediates_inputs(self) -> List[str]:
        return [
            "latents",
            "timesteps",
            "num_inference_steps",
            "pooled_prompt_embeds",
            "negative_pooled_prompt_embeds",
            "add_time_ids",
            "negative_add_time_ids",
            "timestep_cond",
            "prompt_embeds",
            "negative_prompt_embeds",
        ]

    @property
    def intermediates_outputs(self) -> List[str]:
        return ["latents"]

    def __init__(self):
        super().__init__()
        self.components["guider"] = CFGGuider()
        self.components["scheduler"] = None
        self.components["unet"] = None

    @torch.no_grad()
    def __call__(self, pipeline, state: PipelineState) -> PipelineState:
        guidance_scale = state.get_input("guidance_scale")
        guidance_rescale = state.get_input("guidance_rescale")

        cross_attention_kwargs = state.get_input("cross_attention_kwargs")
        generator = state.get_input("generator")
        eta = state.get_input("eta")
        guider_kwargs = state.get_input("guider_kwargs")

        batch_size = state.get_intermediate("batch_size")
        prompt_embeds = state.get_intermediate("prompt_embeds")
        negative_prompt_embeds = state.get_intermediate("negative_prompt_embeds")
        pooled_prompt_embeds = state.get_intermediate("pooled_prompt_embeds")
        negative_pooled_prompt_embeds = state.get_intermediate("negative_pooled_prompt_embeds")
        add_time_ids = state.get_intermediate("add_time_ids")
        negative_add_time_ids = state.get_intermediate("negative_add_time_ids")

        timestep_cond = state.get_intermediate("timestep_cond")
        latents = state.get_intermediate("latents")

        timesteps = state.get_intermediate("timesteps")
        num_inference_steps = state.get_intermediate("num_inference_steps")
        disable_guidance = True if pipeline.unet.config.time_cond_proj_dim is not None else False

        # adding default guider arguments: do_classifier_free_guidance, guidance_scale, guidance_rescale
        guider_kwargs = guider_kwargs or {}
        guider_kwargs = {
            **guider_kwargs,
            "disable_guidance": disable_guidance,
            "guidance_scale": guidance_scale,
            "guidance_rescale": guidance_rescale,
            "batch_size": batch_size,
        }

        pipeline.guider.set_guider(pipeline, guider_kwargs)
        # Prepare conditional inputs using the guider
        prompt_embeds = pipeline.guider.prepare_input(
            prompt_embeds,
            negative_prompt_embeds,
        )
        add_time_ids = pipeline.guider.prepare_input(
            add_time_ids,
            negative_add_time_ids,
        )
        pooled_prompt_embeds = pipeline.guider.prepare_input(
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        )

        added_cond_kwargs = {
            "text_embeds": pooled_prompt_embeds,
            "time_ids": add_time_ids,
        }

        # Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = pipeline.prepare_extra_step_kwargs(generator, eta)
        num_warmup_steps = max(len(timesteps) - num_inference_steps * pipeline.scheduler.order, 0)

        with pipeline.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = pipeline.guider.prepare_input(latents, latents)
                latent_model_input = pipeline.scheduler.scale_model_input(latent_model_input, t)
                # predict the noise residual
                noise_pred = pipeline.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs=cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]
                # perform guidance
                noise_pred = pipeline.guider.apply_guidance(
                    noise_pred,
                    timestep=t,
                    latents=latents,
                )
                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = pipeline.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % pipeline.scheduler.order == 0):
                    progress_bar.update()

        pipeline.guider.reset_guider(pipeline)
        state.add_intermediate("latents", latents)

        return pipeline, state


class StableDiffusionXLControlNetDenoiseStep(PipelineBlock):
    expected_components = ["unet", "controlnet", "scheduler", "guider", "controlnet_guider"]

    @property
    def inputs(self) -> List[Tuple[str, Any]]:
        return [
            ("control_image", None),
            ("control_guidance_start", 0.0),
            ("control_guidance_end", 1.0),
            ("controlnet_conditioning_scale", 1.0),
            ("guess_mode", False),
            ("num_images_per_prompt", 1),
            ("guidance_scale", 5.0),
            ("guidance_rescale", 0.0),
            ("cross_attention_kwargs", None),
            ("generator", None),
            ("eta", 0.0),
            ("guider_kwargs", None),
        ]

    @property
    def intermediates_inputs(self) -> List[str]:
        return [
            "latents",
            "batch_size",
            "timesteps",
            "num_inference_steps",
            "prompt_embeds",
            "negative_prompt_embeds",
            "add_time_ids",
            "negative_add_time_ids",
            "pooled_prompt_embeds",
            "negative_pooled_prompt_embeds",
            "timestep_cond",
        ]

    @property
    def intermediates_outputs(self) -> List[str]:
        return ["latents"]

    def __init__(self):
        super().__init__()
        self.components["guider"] = CFGGuider()
        self.components["controlnet_guider"] = CFGGuider()
        self.components["scheduler"] = None
        self.components["unet"] = None
        self.components["controlnet"] = None
        control_image_processor = VaeImageProcessor(do_convert_rgb=True, do_normalize=False)
        self.auxiliaries["control_image_processor"] = control_image_processor

    @torch.no_grad()
    def __call__(self, pipeline, state: PipelineState) -> PipelineState:
        guidance_scale = state.get_input("guidance_scale")
        guidance_rescale = state.get_input("guidance_rescale")
        cross_attention_kwargs = state.get_input("cross_attention_kwargs")
        guider_kwargs = state.get_input("guider_kwargs")
        generator = state.get_input("generator")
        eta = state.get_input("eta")
        num_images_per_prompt = state.get_input("num_images_per_prompt")
        # controlnet-specific inputs
        control_image = state.get_input("control_image")
        control_guidance_start = state.get_input("control_guidance_start")
        control_guidance_end = state.get_input("control_guidance_end")
        controlnet_conditioning_scale = state.get_input("controlnet_conditioning_scale")
        guess_mode = state.get_input("guess_mode")

        batch_size = state.get_intermediate("batch_size")
        latents = state.get_intermediate("latents")
        timesteps = state.get_intermediate("timesteps")
        num_inference_steps = state.get_intermediate("num_inference_steps")

        prompt_embeds = state.get_intermediate("prompt_embeds")
        negative_prompt_embeds = state.get_intermediate("negative_prompt_embeds")
        pooled_prompt_embeds = state.get_intermediate("pooled_prompt_embeds")
        negative_pooled_prompt_embeds = state.get_intermediate("negative_pooled_prompt_embeds")
        add_time_ids = state.get_intermediate("add_time_ids")
        negative_add_time_ids = state.get_intermediate("negative_add_time_ids")

        timestep_cond = state.get_intermediate("timestep_cond")

        device = pipeline._execution_device

        height, width = latents.shape[-2:]
        height = height * pipeline.vae_scale_factor
        width = width * pipeline.vae_scale_factor

        # prepare controlnet inputs
        controlnet = pipeline.controlnet._orig_mod if is_compiled_module(pipeline.controlnet) else pipeline.controlnet

        # align format for control guidance
        if not isinstance(control_guidance_start, list) and isinstance(control_guidance_end, list):
            control_guidance_start = len(control_guidance_end) * [control_guidance_start]
        elif not isinstance(control_guidance_end, list) and isinstance(control_guidance_start, list):
            control_guidance_end = len(control_guidance_start) * [control_guidance_end]
        elif not isinstance(control_guidance_start, list) and not isinstance(control_guidance_end, list):
            mult = len(controlnet.nets) if isinstance(controlnet, MultiControlNetModel) else 1
            control_guidance_start, control_guidance_end = (
                mult * [control_guidance_start],
                mult * [control_guidance_end],
            )

        if isinstance(controlnet, MultiControlNetModel) and isinstance(controlnet_conditioning_scale, float):
            controlnet_conditioning_scale = [controlnet_conditioning_scale] * len(controlnet.nets)

        global_pool_conditions = (
            controlnet.config.global_pool_conditions
            if isinstance(controlnet, ControlNetModel)
            else controlnet.nets[0].config.global_pool_conditions
        )
        guess_mode = guess_mode or global_pool_conditions

        # 4. Prepare image
        if isinstance(controlnet, ControlNetModel):
            control_image = pipeline.prepare_control_image(
                image=control_image,
                width=width,
                height=height,
                batch_size=batch_size * num_images_per_prompt,
                num_images_per_prompt=num_images_per_prompt,
                device=device,
                dtype=controlnet.dtype,
            )
        elif isinstance(controlnet, MultiControlNetModel):
            control_images = []

            for control_image_ in control_image:
                control_image = pipeline.prepare_control_image(
                    image=control_image_,
                    width=width,
                    height=height,
                    batch_size=batch_size * num_images_per_prompt,
                    num_images_per_prompt=num_images_per_prompt,
                    device=device,
                    dtype=controlnet.dtype,
                )

                control_images.append(control_image)

            control_image = control_images
        else:
            assert False

        # 7.1 Create tensor stating which controlnets to keep
        controlnet_keep = []
        for i in range(len(timesteps)):
            keeps = [
                1.0 - float(i / len(timesteps) < s or (i + 1) / len(timesteps) > e)
                for s, e in zip(control_guidance_start, control_guidance_end)
            ]
            controlnet_keep.append(keeps[0] if isinstance(controlnet, ControlNetModel) else keeps)

        # Prepare conditional inputs for unet using the guider
        # adding default guider arguments: disable_guidance, guidance_scale, guidance_rescale
        disable_guidance = True if pipeline.unet.config.time_cond_proj_dim is not None else False
        guider_kwargs = guider_kwargs or {}
        guider_kwargs = {
            **guider_kwargs,
            "disable_guidance": disable_guidance,
            "guidance_scale": guidance_scale,
            "guidance_rescale": guidance_rescale,
            "batch_size": batch_size,
        }
        pipeline.guider.set_guider(pipeline, guider_kwargs)
        prompt_embeds = pipeline.guider.prepare_input(
            prompt_embeds,
            negative_prompt_embeds,
        )
        add_time_ids = pipeline.guider.prepare_input(
            add_time_ids,
            negative_add_time_ids,
        )
        pooled_prompt_embeds = pipeline.guider.prepare_input(
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        )

        added_cond_kwargs = {
            "text_embeds": pooled_prompt_embeds,
            "time_ids": add_time_ids,
        }

        # Prepare conditional inputs for controlnet using the guider
        controlnet_disable_guidance = True if disable_guidance or guess_mode else False
        controlnet_guider_kwargs = guider_kwargs or {}
        controlnet_guider_kwargs = {
            **controlnet_guider_kwargs,
            "disable_guidance": controlnet_disable_guidance,
            "guidance_scale": guidance_scale,
            "guidance_rescale": guidance_rescale,
            "batch_size": batch_size,
        }
        pipeline.controlnet_guider.set_guider(pipeline, controlnet_guider_kwargs)
        controlnet_prompt_embeds = pipeline.controlnet_guider.prepare_input(prompt_embeds)
        controlnet_added_cond_kwargs = {
            "text_embeds": pipeline.controlnet_guider.prepare_input(pooled_prompt_embeds),
            "time_ids": pipeline.controlnet_guider.prepare_input(add_time_ids),
        }
        # controlnet-specific inputs: control_image
        control_image = pipeline.controlnet_guider.prepare_input(control_image, control_image)

        # Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = pipeline.prepare_extra_step_kwargs(generator, eta)
        num_warmup_steps = max(len(timesteps) - num_inference_steps * pipeline.scheduler.order, 0)

        with pipeline.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # prepare latents for unet using the guider
                latent_model_input = pipeline.guider.prepare_input(latents, latents)

                # prepare latents for controlnet using the guider
                control_model_input = pipeline.controlnet_guider.prepare_input(latents, latents)

                if isinstance(controlnet_keep[i], list):
                    cond_scale = [c * s for c, s in zip(controlnet_conditioning_scale, controlnet_keep[i])]
                else:
                    controlnet_cond_scale = controlnet_conditioning_scale
                    if isinstance(controlnet_cond_scale, list):
                        controlnet_cond_scale = controlnet_cond_scale[0]
                    cond_scale = controlnet_cond_scale * controlnet_keep[i]

                down_block_res_samples, mid_block_res_sample = pipeline.controlnet(
                    pipeline.scheduler.scale_model_input(control_model_input, t),
                    t,
                    encoder_hidden_states=controlnet_prompt_embeds,
                    controlnet_cond=control_image,
                    conditioning_scale=cond_scale,
                    guess_mode=guess_mode,
                    added_cond_kwargs=controlnet_added_cond_kwargs,
                    return_dict=False,
                )

                # when we apply guidance for unet, but not for controlnet:
                # add 0 to the unconditional batch
                down_block_res_samples = pipeline.guider.prepare_input(
                    down_block_res_samples, [torch.zeros_like(d) for d in down_block_res_samples]
                )
                mid_block_res_sample = pipeline.guider.prepare_input(
                    mid_block_res_sample, torch.zeros_like(mid_block_res_sample)
                )

                noise_pred = pipeline.unet(
                    pipeline.scheduler.scale_model_input(latent_model_input, t),
                    t,
                    encoder_hidden_states=prompt_embeds,
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs=cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                    return_dict=False,
                )[0]
                # perform guidance
                noise_pred = pipeline.guider.apply_guidance(noise_pred, timestep=t, latents=latents)
                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = pipeline.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % pipeline.scheduler.order == 0):
                    progress_bar.update()

        pipeline.guider.reset_guider(pipeline)
        pipeline.controlnet_guider.reset_guider(pipeline)
        state.add_intermediate("latents", latents)

        return pipeline, state


class StableDiffusionXLDecodeLatentsStep(PipelineBlock):
    expected_components = ["vae"]

    @property
    def inputs(self) -> List[Tuple[str, Any]]:
        return [
            ("output_type", "pil"),
            ("return_dict", True),
        ]

    @property
    def intermediates_inputs(self) -> List[str]:
        return ["latents"]

    @property
    def intermediates_outputs(self) -> List[str]:
        return ["images"]

    def __init__(self):
        super().__init__()
        self.components["vae"] = None
        self.auxiliaries["image_processor"] = VaeImageProcessor(vae_scale_factor=8)

    @torch.no_grad()
    def __call__(self, pipeline, state: PipelineState) -> PipelineState:
        output_type = state.get_input("output_type")
        return_dict = state.get_input("return_dict")

        latents = state.get_intermediate("latents")

        if not output_type == "latent":
            # make sure the VAE is in float32 mode, as it overflows in float16
            needs_upcasting = pipeline.vae.dtype == torch.float16 and pipeline.vae.config.force_upcast

            if needs_upcasting:
                pipeline.upcast_vae()
                latents = latents.to(next(iter(pipeline.vae.post_quant_conv.parameters())).dtype)
            elif latents.dtype != pipeline.vae.dtype:
                if torch.backends.mps.is_available():
                    # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                    pipeline.vae = pipeline.vae.to(latents.dtype)

            # unscale/denormalize the latents
            # denormalize with the mean and std if available and not None
            has_latents_mean = (
                hasattr(pipeline.vae.config, "latents_mean") and pipeline.vae.config.latents_mean is not None
            )
            has_latents_std = (
                hasattr(pipeline.vae.config, "latents_std") and pipeline.vae.config.latents_std is not None
            )
            if has_latents_mean and has_latents_std:
                latents_mean = (
                    torch.tensor(pipeline.vae.config.latents_mean).view(1, 4, 1, 1).to(latents.device, latents.dtype)
                )
                latents_std = (
                    torch.tensor(self.vae.config.latents_std).view(1, 4, 1, 1).to(latents.device, latents.dtype)
                )
                latents = latents * latents_std / pipeline.vae.config.scaling_factor + latents_mean
            else:
                latents = latents / pipeline.vae.config.scaling_factor

            image = pipeline.vae.decode(latents, return_dict=False)[0]

            # cast back to fp16 if needed
            if needs_upcasting:
                pipeline.vae.to(dtype=torch.float16)
        else:
            image = latents

        # apply watermark if available
        if hasattr(pipeline, "watermark") and pipeline.watermark is not None:
            image = pipeline.watermark.apply_watermark(image)

        image = pipeline.image_processor.postprocess(image, output_type=output_type)

        if not return_dict:
            output = (image,)
        else:
            output = StableDiffusionXLPipelineOutput(images=image)

        state.add_intermediate("images", image)
        state.add_output("images", output)

        return pipeline, state


class StableDiffusionXLAutoSetTimestepsStep(AutoPipelineBlocks):
    block_classes = [StableDiffusionXLSetTimestepsStep, StableDiffusionXLImg2ImgSetTimestepsStep]
    block_prefixes = ["", "img2img"]
    block_trigger_inputs = [None, "image"]


class StableDiffusionXLAutoPrepareLatentsStep(AutoPipelineBlocks):
    block_classes = [StableDiffusionXLPrepareLatentsStep, StableDiffusionXLImg2ImgPrepareLatentsStep]
    block_prefixes = ["", "img2img"]
    block_trigger_inputs = [None, "image"]


class StableDiffusionXLAutoPrepareAdditionalConditioningStep(AutoPipelineBlocks):
    block_classes = [
        StableDiffusionXLPrepareAdditionalConditioningStep,
        StableDiffusionXLImg2ImgPrepareAdditionalConditioningStep,
    ]
    block_prefixes = ["", "img2img"]
    block_trigger_inputs = [None, "image"]


class StableDiffusionXLAutoDenoiseStep(AutoPipelineBlocks):
    block_classes = [StableDiffusionXLDenoiseStep, StableDiffusionXLControlNetDenoiseStep]
    block_prefixes = ["", "controlnet"]
    block_trigger_inputs = [None, "control_image"]


class StableDiffusionXLAllSteps(SequentialPipelineBlocks):
    block_classes = [
        StableDiffusionXLInputStep,
        StableDiffusionXLTextEncoderStep,
        StableDiffusionXLAutoSetTimestepsStep,
        StableDiffusionXLAutoPrepareLatentsStep,
        StableDiffusionXLAutoPrepareAdditionalConditioningStep,
        StableDiffusionXLAutoDenoiseStep,
        StableDiffusionXLDecodeLatentsStep,
    ]
    block_prefixes = [
        "input",
        "text_encoder",
        "set_timesteps",
        "prepare_latents",
        "prepare_add_cond",
        "denoise",
        "decode_latents",
    ]


class StableDiffusionXLModularPipeline(
    ModularPipeline,
    StableDiffusionMixin,
    TextualInversionLoaderMixin,
    StableDiffusionXLLoraLoaderMixin,
):
    @property
    def default_sample_size(self):
        default_sample_size = 128
        if hasattr(self, "unet") and self.unet is not None:
            default_sample_size = self.unet.config.sample_size
        return default_sample_size

    @property
    def vae_scale_factor(self):
        vae_scale_factor = 8
        if hasattr(self, "vae") and self.vae is not None:
            vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        return vae_scale_factor

    @property
    def num_channels_latents(self):
        num_channels_latents = 4
        if hasattr(self, "unet") and self.unet is not None:
            num_channels_latents = self.unet.config.in_channels
        return num_channels_latents

    # Copied from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl.StableDiffusionXLPipeline._get_add_time_ids
    def _get_add_time_ids(
        self, original_size, crops_coords_top_left, target_size, dtype, text_encoder_projection_dim=None
    ):
        add_time_ids = list(original_size + crops_coords_top_left + target_size)

        passed_add_embed_dim = (
            self.unet.config.addition_time_embed_dim * len(add_time_ids) + text_encoder_projection_dim
        )
        expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features

        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        return add_time_ids

    # Copied from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_img2img.StableDiffusionXLImg2ImgPipeline._get_add_time_ids
    def _get_add_time_ids_img2img(
        self,
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
        if self.config.requires_aesthetics_score:
            add_time_ids = list(original_size + crops_coords_top_left + (aesthetic_score,))
            add_neg_time_ids = list(
                negative_original_size + negative_crops_coords_top_left + (negative_aesthetic_score,)
            )
        else:
            add_time_ids = list(original_size + crops_coords_top_left + target_size)
            add_neg_time_ids = list(negative_original_size + crops_coords_top_left + negative_target_size)

        passed_add_embed_dim = (
            self.unet.config.addition_time_embed_dim * len(add_time_ids) + text_encoder_projection_dim
        )
        expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features

        if (
            expected_add_embed_dim > passed_add_embed_dim
            and (expected_add_embed_dim - passed_add_embed_dim) == self.unet.config.addition_time_embed_dim
        ):
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. Please make sure to enable `requires_aesthetics_score` with `pipe.register_to_config(requires_aesthetics_score=True)` to make sure `aesthetic_score` {aesthetic_score} and `negative_aesthetic_score` {negative_aesthetic_score} is correctly used by the model."
            )
        elif (
            expected_add_embed_dim < passed_add_embed_dim
            and (passed_add_embed_dim - expected_add_embed_dim) == self.unet.config.addition_time_embed_dim
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

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.encode_image
    def encode_image(self, image, device, num_images_per_prompt, output_hidden_states=None):
        dtype = next(self.image_encoder.parameters()).dtype

        if not isinstance(image, torch.Tensor):
            image = self.feature_extractor(image, return_tensors="pt").pixel_values

        image = image.to(device=device, dtype=dtype)
        if output_hidden_states:
            image_enc_hidden_states = self.image_encoder(image, output_hidden_states=True).hidden_states[-2]
            image_enc_hidden_states = image_enc_hidden_states.repeat_interleave(num_images_per_prompt, dim=0)
            uncond_image_enc_hidden_states = self.image_encoder(
                torch.zeros_like(image), output_hidden_states=True
            ).hidden_states[-2]
            uncond_image_enc_hidden_states = uncond_image_enc_hidden_states.repeat_interleave(
                num_images_per_prompt, dim=0
            )
            return image_enc_hidden_states, uncond_image_enc_hidden_states
        else:
            image_embeds = self.image_encoder(image).image_embeds
            image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0)
            uncond_image_embeds = torch.zeros_like(image_embeds)

            return image_embeds, uncond_image_embeds

    # Modified from diffusers.pipelines.controlnet.pipeline_controlnet_sd_xl.StableDiffusionXLControlNetPipeline.prepare_image
    # return image without apply any guidance
    def prepare_control_image(
        self,
        image,
        width,
        height,
        batch_size,
        num_images_per_prompt,
        device,
        dtype,
    ):
        image = self.control_image_processor.preprocess(image, height=height, width=width).to(dtype=torch.float32)
        image_batch_size = image.shape[0]

        if image_batch_size == 1:
            repeat_by = batch_size
        else:
            # image batch size is the same as prompt batch size
            repeat_by = num_images_per_prompt

        image = image.repeat_interleave(repeat_by, dim=0)

        image = image.to(device=device, dtype=dtype)

        return image

    # Copied from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl.StableDiffusionXLPipeline.encode_prompt
    def encode_prompt(
        self,
        prompt: str,
        prompt_2: Optional[str] = None,
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        do_classifier_free_guidance: bool = True,
        negative_prompt: Optional[str] = None,
        negative_prompt_2: Optional[str] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
        lora_scale: Optional[float] = None,
        clip_skip: Optional[int] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to the `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                used in both text-encoders
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            negative_prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and
                `text_encoder_2`. If not defined, `negative_prompt` is used in both text-encoders
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            pooled_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            negative_pooled_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
                input argument.
            lora_scale (`float`, *optional*):
                A lora scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
        """
        device = device or self._execution_device

        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(self, StableDiffusionXLLoraLoaderMixin):
            self._lora_scale = lora_scale

            # dynamically adjust the LoRA scale
            if self.text_encoder is not None:
                if not USE_PEFT_BACKEND:
                    adjust_lora_scale_text_encoder(self.text_encoder, lora_scale)
                else:
                    scale_lora_layers(self.text_encoder, lora_scale)

            if self.text_encoder_2 is not None:
                if not USE_PEFT_BACKEND:
                    adjust_lora_scale_text_encoder(self.text_encoder_2, lora_scale)
                else:
                    scale_lora_layers(self.text_encoder_2, lora_scale)

        prompt = [prompt] if isinstance(prompt, str) else prompt

        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # Define tokenizers and text encoders
        tokenizers = [self.tokenizer, self.tokenizer_2] if self.tokenizer is not None else [self.tokenizer_2]
        text_encoders = (
            [self.text_encoder, self.text_encoder_2] if self.text_encoder is not None else [self.text_encoder_2]
        )

        if prompt_embeds is None:
            prompt_2 = prompt_2 or prompt
            prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2

            # textual inversion: process multi-vector tokens if necessary
            prompt_embeds_list = []
            prompts = [prompt, prompt_2]
            for prompt, tokenizer, text_encoder in zip(prompts, tokenizers, text_encoders):
                if isinstance(self, TextualInversionLoaderMixin):
                    prompt = self.maybe_convert_prompt(prompt, tokenizer)

                text_inputs = tokenizer(
                    prompt,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )

                text_input_ids = text_inputs.input_ids
                untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

                if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                    text_input_ids, untruncated_ids
                ):
                    removed_text = tokenizer.batch_decode(untruncated_ids[:, tokenizer.model_max_length - 1 : -1])
                    logger.warning(
                        "The following part of your input was truncated because CLIP can only handle sequences up to"
                        f" {tokenizer.model_max_length} tokens: {removed_text}"
                    )

                prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)

                # We are only ALWAYS interested in the pooled output of the final text encoder
                pooled_prompt_embeds = prompt_embeds[0]
                if clip_skip is None:
                    prompt_embeds = prompt_embeds.hidden_states[-2]
                else:
                    # "2" because SDXL always indexes from the penultimate layer.
                    prompt_embeds = prompt_embeds.hidden_states[-(clip_skip + 2)]

                prompt_embeds_list.append(prompt_embeds)

            prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)

        # get unconditional embeddings for classifier free guidance
        zero_out_negative_prompt = negative_prompt is None and self.config.force_zeros_for_empty_prompt
        if do_classifier_free_guidance and negative_prompt_embeds is None and zero_out_negative_prompt:
            negative_prompt_embeds = torch.zeros_like(prompt_embeds)
            negative_pooled_prompt_embeds = torch.zeros_like(pooled_prompt_embeds)
        elif do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt_2 = negative_prompt_2 or negative_prompt

            # normalize str to list
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
            negative_prompt_2 = (
                batch_size * [negative_prompt_2] if isinstance(negative_prompt_2, str) else negative_prompt_2
            )

            uncond_tokens: List[str]
            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = [negative_prompt, negative_prompt_2]

            negative_prompt_embeds_list = []
            for negative_prompt, tokenizer, text_encoder in zip(uncond_tokens, tokenizers, text_encoders):
                if isinstance(self, TextualInversionLoaderMixin):
                    negative_prompt = self.maybe_convert_prompt(negative_prompt, tokenizer)

                max_length = prompt_embeds.shape[1]
                uncond_input = tokenizer(
                    negative_prompt,
                    padding="max_length",
                    max_length=max_length,
                    truncation=True,
                    return_tensors="pt",
                )

                negative_prompt_embeds = text_encoder(
                    uncond_input.input_ids.to(device),
                    output_hidden_states=True,
                )
                # We are only ALWAYS interested in the pooled output of the final text encoder
                negative_pooled_prompt_embeds = negative_prompt_embeds[0]
                negative_prompt_embeds = negative_prompt_embeds.hidden_states[-2]

                negative_prompt_embeds_list.append(negative_prompt_embeds)

            negative_prompt_embeds = torch.concat(negative_prompt_embeds_list, dim=-1)

        if self.text_encoder_2 is not None:
            prompt_embeds = prompt_embeds.to(dtype=self.text_encoder_2.dtype, device=device)
        else:
            prompt_embeds = prompt_embeds.to(dtype=self.unet.dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            if self.text_encoder_2 is not None:
                negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.text_encoder_2.dtype, device=device)
            else:
                negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.unet.dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt).view(
            bs_embed * num_images_per_prompt, -1
        )
        if do_classifier_free_guidance:
            negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.repeat(1, num_images_per_prompt).view(
                bs_embed * num_images_per_prompt, -1
            )

        if self.text_encoder is not None:
            if isinstance(self, StableDiffusionXLLoraLoaderMixin) and USE_PEFT_BACKEND:
                # Retrieve the original scale by scaling back the LoRA layers
                unscale_lora_layers(self.text_encoder, lora_scale)

        if self.text_encoder_2 is not None:
            if isinstance(self, StableDiffusionXLLoraLoaderMixin) and USE_PEFT_BACKEND:
                # Retrieve the original scale by scaling back the LoRA layers
                unscale_lora_layers(self.text_encoder_2, lora_scale)

        return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_ip_adapter_image_embeds
    def prepare_ip_adapter_image_embeds(
        self, ip_adapter_image, ip_adapter_image_embeds, device, num_images_per_prompt, do_classifier_free_guidance
    ):
        image_embeds = []
        if do_classifier_free_guidance:
            negative_image_embeds = []
        if ip_adapter_image_embeds is None:
            if not isinstance(ip_adapter_image, list):
                ip_adapter_image = [ip_adapter_image]

            if len(ip_adapter_image) != len(self.unet.encoder_hid_proj.image_projection_layers):
                raise ValueError(
                    f"`ip_adapter_image` must have same length as the number of IP Adapters. Got {len(ip_adapter_image)} images and {len(self.unet.encoder_hid_proj.image_projection_layers)} IP Adapters."
                )

            for single_ip_adapter_image, image_proj_layer in zip(
                ip_adapter_image, self.unet.encoder_hid_proj.image_projection_layers
            ):
                output_hidden_state = not isinstance(image_proj_layer, ImageProjection)
                single_image_embeds, single_negative_image_embeds = self.encode_image(
                    single_ip_adapter_image, device, 1, output_hidden_state
                )

                image_embeds.append(single_image_embeds[None, :])
                if do_classifier_free_guidance:
                    negative_image_embeds.append(single_negative_image_embeds[None, :])
        else:
            for single_image_embeds in ip_adapter_image_embeds:
                if do_classifier_free_guidance:
                    single_negative_image_embeds, single_image_embeds = single_image_embeds.chunk(2)
                    negative_image_embeds.append(single_negative_image_embeds)
                image_embeds.append(single_image_embeds)

        ip_adapter_image_embeds = []
        for i, single_image_embeds in enumerate(image_embeds):
            single_image_embeds = torch.cat([single_image_embeds] * num_images_per_prompt, dim=0)
            if do_classifier_free_guidance:
                single_negative_image_embeds = torch.cat([negative_image_embeds[i]] * num_images_per_prompt, dim=0)
                single_image_embeds = torch.cat([single_negative_image_embeds, single_image_embeds], dim=0)

            single_image_embeds = single_image_embeds.to(device=device)
            ip_adapter_image_embeds.append(single_image_embeds)

        return ip_adapter_image_embeds

    # Copied from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_img2img.StableDiffusionXLImg2ImgPipeline.get_timesteps
    def get_timesteps(self, num_inference_steps, strength, device, denoising_start=None):
        # get the original timestep using init_timestep
        if denoising_start is None:
            init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
            t_start = max(num_inference_steps - init_timestep, 0)

            timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]
            if hasattr(self.scheduler, "set_begin_index"):
                self.scheduler.set_begin_index(t_start * self.scheduler.order)

            return timesteps, num_inference_steps - t_start

        else:
            # Strength is irrelevant if we directly request a timestep to start at;
            # that is, strength is determined by the denoising_start instead.
            discrete_timestep_cutoff = int(
                round(
                    self.scheduler.config.num_train_timesteps
                    - (denoising_start * self.scheduler.config.num_train_timesteps)
                )
            )

            num_inference_steps = (self.scheduler.timesteps < discrete_timestep_cutoff).sum().item()
            if self.scheduler.order == 2 and num_inference_steps % 2 == 0:
                # if the scheduler is a 2nd order scheduler we might have to do +1
                # because `num_inference_steps` might be even given that every timestep
                # (except the highest one) is duplicated. If `num_inference_steps` is even it would
                # mean that we cut the timesteps in the middle of the denoising step
                # (between 1st and 2nd derivative) which leads to incorrect results. By adding 1
                # we ensure that the denoising process always ends after the 2nd derivate step of the scheduler
                num_inference_steps = num_inference_steps + 1

            # because t_n+1 >= t_n, we slice the timesteps starting from the end
            t_start = len(self.scheduler.timesteps) - num_inference_steps
            timesteps = self.scheduler.timesteps[t_start:]
            if hasattr(self.scheduler, "set_begin_index"):
                self.scheduler.set_begin_index(t_start)
            return timesteps, num_inference_steps

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents
    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        shape = (
            batch_size,
            num_channels_latents,
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
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
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    # Copied from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_img2img.StableDiffusionXLImg2ImgPipeline.prepare_latents
    def prepare_latents_img2img(
        self, image, timestep, batch_size, num_images_per_prompt, dtype, device, generator=None, add_noise=True
    ):
        if not isinstance(image, (torch.Tensor, PIL.Image.Image, list)):
            raise ValueError(
                f"`image` has to be of type `torch.Tensor`, `PIL.Image.Image` or list but is {type(image)}"
            )

        # Offload text encoder if `enable_model_cpu_offload` was enabled
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.text_encoder_2.to("cpu")
            torch.cuda.empty_cache()

        image = image.to(device=device, dtype=dtype)

        batch_size = batch_size * num_images_per_prompt

        if image.shape[1] == 4:
            init_latents = image

        else:
            latents_mean = latents_std = None
            if hasattr(self.vae.config, "latents_mean") and self.vae.config.latents_mean is not None:
                latents_mean = torch.tensor(self.vae.config.latents_mean).view(1, 4, 1, 1)
            if hasattr(self.vae.config, "latents_std") and self.vae.config.latents_std is not None:
                latents_std = torch.tensor(self.vae.config.latents_std).view(1, 4, 1, 1)
            # make sure the VAE is in float32 mode, as it overflows in float16
            if self.vae.config.force_upcast:
                image = image.float()
                self.vae.to(dtype=torch.float32)

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
                    retrieve_latents(self.vae.encode(image[i : i + 1]), generator=generator[i])
                    for i in range(batch_size)
                ]
                init_latents = torch.cat(init_latents, dim=0)
            else:
                init_latents = retrieve_latents(self.vae.encode(image), generator=generator)

            if self.vae.config.force_upcast:
                self.vae.to(dtype)

            init_latents = init_latents.to(dtype)
            if latents_mean is not None and latents_std is not None:
                latents_mean = latents_mean.to(device=device, dtype=dtype)
                latents_std = latents_std.to(device=device, dtype=dtype)
                init_latents = (init_latents - latents_mean) * self.vae.config.scaling_factor / latents_std
            else:
                init_latents = self.vae.config.scaling_factor * init_latents

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
            init_latents = self.scheduler.add_noise(init_latents, noise, timestep)

        latents = init_latents

        return latents

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_upscale.StableDiffusionUpscalePipeline.upcast_vae
    def upcast_vae(self):
        dtype = self.vae.dtype
        self.vae.to(dtype=torch.float32)
        use_torch_2_0_or_xformers = isinstance(
            self.vae.decoder.mid_block.attentions[0].processor,
            (
                AttnProcessor2_0,
                XFormersAttnProcessor,
            ),
        )
        # if xformers or torch_2_0 is used attention block does not need
        # to be in float32 which can save lots of memory
        if use_torch_2_0_or_xformers:
            self.vae.post_quant_conv.to(dtype)
            self.vae.decoder.conv_in.to(dtype)
            self.vae.decoder.mid_block.to(dtype)

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
