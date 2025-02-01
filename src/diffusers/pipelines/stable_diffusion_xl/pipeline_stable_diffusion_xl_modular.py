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
from typing import Any, List, Optional, Tuple, Union, Dict

import PIL
import torch
from collections import OrderedDict

from ...guider import CFGGuider
from ...image_processor import VaeImageProcessor, PipelineImageInput
from ...loaders import StableDiffusionXLLoraLoaderMixin, TextualInversionLoaderMixin, ModularIPAdapterMixin
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
    InputParam,
    OutputParam,
    SequentialPipelineBlocks,
)
from ..pipeline_utils import DiffusionPipeline, StableDiffusionMixin
from .pipeline_output import (
    StableDiffusionXLPipelineOutput,
)

import numpy as np

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



class StableDiffusionXLLoraStep(PipelineBlock):
    expected_components = ["text_encoder", "text_encoder_2", "unet"]
    model_name = "stable-diffusion-xl"

    @property
    def description(self) -> str:
        return (
            "Lora step that handles all the lora related tasks: load/unload lora weights into unet and text encoders, manage lora adapters etc"
            " See [StableDiffusionXLLoraLoaderMixin](https://huggingface.co/docs/diffusers/api/loaders/lora#diffusers.loaders.StableDiffusionXLLoraLoaderMixin)"
            " for more details"
        )

    
    @property
    def inputs(self) -> List[InputParam]:
        return []

    @property
    def intermediates_inputs(self) -> List[InputParam]:
        return []
    
    @property
    def intermediates_outputs(self) -> List[OutputParam]:
        return []
    
    def __init__(self):
        super().__init__()
        self.components["text_encoder"] = None
        self.components["text_encoder_2"] = None
        self.components["unet"] = None
    
    @torch.no_grad()
    def __call__(self, pipeline, state: PipelineState) -> PipelineState:
        raise EnvironmentError("StableDiffusionXLLoraStep is desgined to be used to load lora weights, __call__ is not implemented")


class StableDiffusionXLIPAdapterStep(PipelineBlock):
    expected_components = ["image_encoder", "feature_extractor", "unet"]
    model_name = "stable-diffusion-xl"

    
    @property
    def description(self) -> str:
        return (
            "IP Adapter step that handles all the ip adapter related tasks: Load/unload ip adapter weights into unet, prepare ip adapter image embeddings, etc"
            " See [ModularIPAdapterMixin](https://huggingface.co/docs/diffusers/api/loaders/ip_adapter#diffusers.loaders.ModularIPAdapterMixin)"
            " for more details"
        )

    @property
    def inputs(self) -> List[InputParam]:
        return [
            InputParam(
                "ip_adapter_image", 
                required=True, 
                type_hint=PipelineImageInput, 
                description="The image(s) to be used as ip adapter"
            ), 
            InputParam(
                "guidance_scale", 
                default=5.0, 
                description="Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598). Guidance scale is enabled by setting `guidance_scale > 1`."
            ),
        ]

    @property
    def intermediates_inputs(self) -> List[InputParam]:
        return []

    @property
    def intermediates_outputs(self) -> List[OutputParam]:
        return [
            OutputParam("ip_adapter_embeds", type_hint=torch.Tensor, description="IP adapter image embeddings"),
            OutputParam("negative_ip_adapter_embeds", type_hint=torch.Tensor, description="Negative IP adapter image embeddings")
        ]
    
    def __init__(self):
        super().__init__()
        self.components["image_encoder"] = None
        self.components["feature_extractor"] = None
        self.components["unet"] = None
    

    @torch.no_grad()
    def __call__(self, pipeline, state: PipelineState) -> PipelineState:
        data = self.get_block_state(state)

        data.do_classifier_free_guidance = data.guidance_scale > 1.0
        data.device = pipeline._execution_device

        data.ip_adapter_embeds = pipeline.prepare_ip_adapter_image_embeds(
            ip_adapter_image=data.ip_adapter_image,
            ip_adapter_image_embeds=None,
            device=data.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=data.do_classifier_free_guidance,
        )
        if data.do_classifier_free_guidance:
            data.negative_ip_adapter_embeds = []
            for i, image_embeds in enumerate(data.ip_adapter_embeds):
                negative_image_embeds, image_embeds = image_embeds.chunk(2)
                data.negative_ip_adapter_embeds.append(negative_image_embeds)
                data.ip_adapter_embeds[i] = image_embeds

        self.add_block_state(state, data)
        return pipeline, state


class StableDiffusionXLTextEncoderStep(PipelineBlock):
    expected_components = ["text_encoder", "text_encoder_2", "tokenizer", "tokenizer_2"]
    expected_configs = ["force_zeros_for_empty_prompt"]
    model_name = "stable-diffusion-xl"

    @property
    def description(self) -> str:
        return(
            "Text Encoder step that generate text_embeddings to guide the image generation"
        )


    @property
    def inputs(self) -> List[InputParam]:
        return [
            InputParam(
                name="prompt",
                type_hint=Union[str, List[str]],
                description="The prompt or prompts to guide the image generation.",
            ),
            InputParam(
                name="prompt_2",
                type_hint=Union[str, List[str]],
                description="The prompt or prompts to be sent to the `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is used in both text-encoders",
            ),
            InputParam(
                name="negative_prompt",
                type_hint=Union[str, List[str]],
                description="The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).",
            ),
            InputParam(
                name="negative_prompt_2",
                type_hint=Union[str, List[str]],
                description="The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and `text_encoder_2`. If not defined, `negative_prompt` is used in both text-encoders",
            ),
            InputParam(
                name="cross_attention_kwargs",
                type_hint=Optional[dict],
                description="A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under `self.processor` in [diffusers.models.attention_processor]",
            ),
            InputParam(
                name="guidance_scale",
                type_hint=float,
                default=5.0,
                description="Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598). `guidance_scale` is defined as `w` of equation 2. of [Imagen Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`, usually at the expense of lower image quality.",
            ),
            InputParam(
                name="clip_skip",
                type_hint=Optional[int],
            ),
        ]

    
    @property
    def intermediates_inputs(self) -> List[InputParam]:
        return []

    @property
    def intermediates_outputs(self) -> List[OutputParam]:
        return [
            OutputParam("prompt_embeds", type_hint=torch.Tensor, description="text embeddings used to guide the image generation"),
            OutputParam("negative_prompt_embeds", type_hint=torch.Tensor, description="negative text embeddings used to guide the image generation"),
            OutputParam("pooled_prompt_embeds", type_hint=torch.Tensor, description="pooled text embeddings used to guide the image generation"),
            OutputParam("negative_pooled_prompt_embeds", type_hint=torch.Tensor, description="negative pooled text embeddings used to guide the image generation"),
        ]

    def __init__(self):
        super().__init__()
        self.configs["force_zeros_for_empty_prompt"] = True
        self.components["text_encoder"] = None
        self.components["text_encoder_2"] = None
        self.components["tokenizer"] = None
        self.components["tokenizer_2"] = None

    def check_inputs(self, pipeline, data):

        if data.prompt is not None and (not isinstance(data.prompt, str) and not isinstance(data.prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(data.prompt)}")
        elif data.prompt_2 is not None and (not isinstance(data.prompt_2, str) and not isinstance(data.prompt_2, list)):
            raise ValueError(f"`prompt_2` has to be of type `str` or `list` but is {type(data.prompt_2)}")


    @torch.no_grad()
    def __call__(self, pipeline, state: PipelineState) -> PipelineState:
        # Get inputs and intermediates
        data = self.get_block_state(state)
        self.check_inputs(pipeline, data)

        data.do_classifier_free_guidance = data.guidance_scale > 1.0
        data.device = pipeline._execution_device


        # Encode input prompt
        data.text_encoder_lora_scale = (
            data.cross_attention_kwargs.get("scale", None) if data.cross_attention_kwargs is not None else None
        )
        (
            data.prompt_embeds,
            data.negative_prompt_embeds,
            data.pooled_prompt_embeds,
            data.negative_pooled_prompt_embeds,
        ) = pipeline.encode_prompt(
            data.prompt,
            data.prompt_2,
            data.device,
            1,
            data.do_classifier_free_guidance,
            data.negative_prompt,
            data.negative_prompt_2,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            pooled_prompt_embeds=None,
            negative_pooled_prompt_embeds=None,
            lora_scale=data.text_encoder_lora_scale,
            clip_skip=data.clip_skip,
        )
        # Add outputs
        self.add_block_state(state, data)
        return pipeline, state


class StableDiffusionXLVaeEncoderStep(PipelineBlock):
    expected_components = ["vae"]
    model_name = "stable-diffusion-xl"

    
    @property
    def description(self) -> str:
        return (
            "Vae Encoder step that encode the input image into a latent representation"
        )

    @property
    def inputs(self) -> List[InputParam]:
        return [
            InputParam(
                name="image", 
                type_hint=PipelineImageInput, 
                required=True, 
                description="The image(s) to modify with the pipeline, for img2img or inpainting task. When using for inpainting task, parts of the image will be masked out with `mask_image` and repainted according to `prompt`."
            ),
            InputParam(
                name="generator", 
                type_hint=Optional[Union[torch.Generator, List[torch.Generator]]], 
                description="One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)"
                            "to make generation deterministic."
            ),
            InputParam(
                name="height", 
                type_hint=Optional[int], 
                description="The height in pixels of the generated image. This is set to 1024 by default for the best results. "
                            "Anything below 512 pixels won't work well for [stabilityai/stable-diffusion-xl-base-1.0]"
                            "(https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) and checkpoints that are not "
                            "specifically fine-tuned on low resolutions.",
            ),
            InputParam(
                name="width", 
                type_hint=Optional[int], 
                description="The width in pixels of the generated image. This is set to 1024 by default for the best results. "
                            "Anything below 512 pixels won't work well for [stabilityai/stable-diffusion-xl-base-1.0]"
                            "(https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) and checkpoints that are not "
                            "specifically fine-tuned on low resolutions.",
            ),
        ]

    @property
    def intermediates_inputs(self) -> List[InputParam]:
        return [
            InputParam("dtype", type_hint=torch.dtype, description="Data type of model tensor inputs"), 
            InputParam("preprocess_kwargs", type_hint=Optional[dict], description="A kwargs dictionary that if specified is passed along to the `ImageProcessor` as defined under `self.image_processor` in [diffusers.image_processor.VaeImageProcessor]")]

    @property
    def intermediates_outputs(self) -> List[OutputParam]:
        return [OutputParam("image_latents", type_hint=torch.Tensor, description="The latents representing the reference image for image-to-image/inpainting generation")]

    def __init__(self):
        super().__init__()
        self.components["vae"] = None
        self.auxiliaries["image_processor"] = VaeImageProcessor()

    @torch.no_grad()
    def __call__(self, pipeline, state: PipelineState) -> PipelineState:
        data = self.get_block_state(state)
        data.preprocess_kwargs = data.preprocess_kwargs or {}
        data.device = pipeline._execution_device
        data.dtype = data.dtype if data.dtype is not None else pipeline.vae.dtype
        
        data.image = pipeline.image_processor.preprocess(data.image, height=data.height, width=data.width, **data.preprocess_kwargs)
        data.image = data.image.to(device=data.device, dtype=data.dtype)

        data.batch_size = data.image.shape[0]

        # if generator is a list, make sure the length of it matches the length of images (both should be batch_size)
        if isinstance(data.generator, list) and len(data.generator) != data.batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(data.generator)}, but requested an effective batch"
                f" size of {data.batch_size}. Make sure the batch size matches the length of the generators."
            )


        data.image_latents = pipeline._encode_vae_image(image=data.image, generator=data.generator)

        self.add_block_state(state, data)

        return pipeline, state


class StableDiffusionXLInpaintVaeEncoderStep(PipelineBlock):
    expected_components = ["vae"]
    model_name = "stable-diffusion-xl"

    @property
    def description(self) -> str:
        return (
            "Vae encoder step that prepares the image and mask for the inpainting process"
        )

    @property
    def inputs(self) -> List[InputParam]:
        return [
            InputParam(
                "height", 
                type_hint=Optional[int], 
                description="The height in pixels of the generated image. This is set to 1024 by default for the best results. "
                           "Anything below 512 pixels won't work well for [stabilityai/stable-diffusion-xl-base-1.0]"
                           "(https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) and checkpoints that are not "
                           "specifically fine-tuned on low resolutions.",
            ),
            InputParam(
                "width", 
                type_hint=Optional[int], 
                description="The width in pixels of the generated image. This is set to 1024 by default for the best results. "
                           "Anything below 512 pixels won't work well for [stabilityai/stable-diffusion-xl-base-1.0]"
                           "(https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) and checkpoints that are not "
                           "specifically fine-tuned on low resolutions.",
            ),
            InputParam(
                "generator", 
                type_hint=Optional[Union[torch.Generator, List[torch.Generator]]], 
                description="One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html) "
                           "to make generation deterministic."
            ),
            InputParam(
                "image", 
                required=True, 
                type_hint=PipelineImageInput, 
                description="The image(s) to modify with the pipeline, for img2img or inpainting task. When using for inpainting task, parts of the image will be masked out with `mask_image` and repainted according to `prompt`."
            ),
            InputParam(
                "mask_image", 
                required=True, 
                type_hint=PipelineImageInput, 
                description="`Image`, or tensor representing an image batch, to mask `image`. White pixels in the mask will be "
                           "repainted, while black pixels will be preserved. If `mask_image` is a PIL image, it will be converted "
                           "to a single channel (luminance) before use. If it's a tensor, it should contain one color channel (L) "
                           "instead of 3, so the expected shape would be `(B, H, W, 1)`."
            ),
            InputParam(
                "padding_mask_crop", 
                type_hint=Optional[Tuple[int, int]], 
                description="The size of margin in the crop to be applied to the image and masking. If `None`, no crop is applied to "
                           "image and mask_image. If `padding_mask_crop` is not `None`, it will first find a rectangular region "
                           "with the same aspect ratio of the image and contains all masked area, and then expand that area based "
                           "on `padding_mask_crop`. The image and mask_image will then be cropped based on the expanded area before "
                           "resizing to the original image size for inpainting. This is useful when the masked area is small while "
                           "the image is large and contain information irrelevant for inpainting, such as background."
            ),
        ]

    @property
    def intermediates_inputs(self) -> List[InputParam]:
        return [InputParam("dtype", type_hint=torch.dtype, description="The dtype of the model inputs")]

    @property
    def intermediates_outputs(self) -> List[OutputParam]:
        return [OutputParam("image_latents", type_hint=torch.Tensor, description="The latents representation of the input image"), 
                OutputParam("mask", type_hint=torch.Tensor, description="The mask to use for the inpainting process"), 
                OutputParam("masked_image_latents", type_hint=torch.Tensor, description="The masked image latents to use for the inpainting process (only for inpainting-specifid unet)"), 
                OutputParam("crops_coords", type_hint=Optional[Tuple[int, int]], description="The crop coordinates to use for the preprocess/postprocess of the image and mask")]
    
    def __init__(self):
        super().__init__()
        self.auxiliaries["image_processor"] = VaeImageProcessor()
        self.auxiliaries["mask_processor"] = VaeImageProcessor(do_normalize=False, do_binarize=True, do_convert_grayscale=True)
        self.components["vae"] = None

    @torch.no_grad()
    def __call__(self, pipeline: DiffusionPipeline, state: PipelineState) -> PipelineState:

        data = self.get_block_state(state)

        data.dtype = data.dtype if data.dtype is not None else pipeline.vae.dtype
        data.device = pipeline._execution_device

        if data.padding_mask_crop is not None:
            data.crops_coords = pipeline.mask_processor.get_crop_region(data.mask_image, data.width, data.height, pad=data.padding_mask_crop)
            data.resize_mode = "fill"
        else:
            data.crops_coords = None
            data.resize_mode = "default"
        
        data.image = pipeline.image_processor.preprocess(data.image, height=data.height, width=data.width, crops_coords=data.crops_coords, resize_mode=data.resize_mode)
        data.image = data.image.to(dtype=torch.float32)

        data.mask = pipeline.mask_processor.preprocess(data.mask_image, height=data.height, width=data.width, resize_mode=data.resize_mode, crops_coords=data.crops_coords)
        data.masked_image = data.image * (data.mask < 0.5)

        data.batch_size = data.image.shape[0]
        data.image = data.image.to(device=data.device, dtype=data.dtype)
        data.image_latents = pipeline._encode_vae_image(image=data.image, generator=data.generator)

        # 7. Prepare mask latent variables
        data.mask, data.masked_image_latents = pipeline.prepare_mask_latents(
            data.mask,
            data.masked_image,
            data.batch_size,
            data.height,
            data.width,
            data.dtype,
            data.device,
            data.generator,
        )

        self.add_block_state(state, data)


        return pipeline, state


class StableDiffusionXLInputStep(PipelineBlock):
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
            InputParam(
                name="num_images_per_prompt",
                type_hint=int,
                default=1,
                description="The number of images to generate per prompt.",
            ),
        ]

    @property
    def intermediates_inputs(self) -> List[str]:
        return [
            InputParam("prompt_embeds", required=True, type_hint=torch.Tensor, description="Pre-generated text embeddings. Can be generated from text_encoder step."),
            InputParam("negative_prompt_embeds", type_hint=torch.Tensor, description="Pre-generated negative text embeddings. Can be generated from text_encoder step."),
            InputParam("pooled_prompt_embeds", required=True, type_hint=torch.Tensor, description="Pre-generated pooled text embeddings. Can be generated from text_encoder step."),
            InputParam("negative_pooled_prompt_embeds", description="Pre-generated negative pooled text embeddings. Can be generated from text_encoder step."),
            InputParam("ip_adapter_embeds", type_hint=List[torch.Tensor], description="Pre-generated image embeddings for IP-Adapter. Can be generated from ip_adapter step."),
            InputParam("negative_ip_adapter_embeds", type_hint=List[torch.Tensor], description="Pre-generated negative image embeddings for IP-Adapter. Can be generated from ip_adapter step."),
        ]
    
    @property
    def intermediates_outputs(self) -> List[str]:
        return [
            OutputParam("batch_size", type_hint=int, description="Number of prompts, the final batch size of model inputs should be batch_size * num_images_per_prompt"),
            OutputParam("dtype", type_hint=torch.dtype, description="Data type of model tensor inputs (determined by `prompt_embeds`)"),
            OutputParam("prompt_embeds", type_hint=torch.Tensor, description="text embeddings used to guide the image generation"),
            OutputParam("negative_prompt_embeds", type_hint=torch.Tensor, description="negative text embeddings used to guide the image generation"),
            OutputParam("pooled_prompt_embeds", type_hint=torch.Tensor, description="pooled text embeddings used to guide the image generation"),
            OutputParam("negative_pooled_prompt_embeds", type_hint=torch.Tensor, description="negative pooled text embeddings used to guide the image generation"),
            OutputParam("ip_adapter_embeds", type_hint=List[torch.Tensor], description="image embeddings for IP-Adapter"),
            OutputParam("negative_ip_adapter_embeds", type_hint=List[torch.Tensor], description="negative image embeddings for IP-Adapter"),
        ]
    
    def check_inputs(self, pipeline, data):

        if data.prompt_embeds is not None and data.negative_prompt_embeds is not None:
            if data.prompt_embeds.shape != data.negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {data.prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {data.negative_prompt_embeds.shape}."
                )

        if data.prompt_embeds is not None and data.pooled_prompt_embeds is None:
            raise ValueError(
                "If `prompt_embeds` are provided, `pooled_prompt_embeds` also have to be passed. Make sure to generate `pooled_prompt_embeds` from the same text encoder that was used to generate `prompt_embeds`."
            )

        if data.negative_prompt_embeds is not None and data.negative_pooled_prompt_embeds is None:
            raise ValueError(
                "If `negative_prompt_embeds` are provided, `negative_pooled_prompt_embeds` also have to be passed. Make sure to generate `negative_pooled_prompt_embeds` from the same text encoder that was used to generate `negative_prompt_embeds`."
            )
        
        if data.ip_adapter_embeds is not None and not isinstance(data.ip_adapter_embeds, list):
            raise ValueError("`ip_adapter_embeds` must be a list")
        
        if data.negative_ip_adapter_embeds is not None and not isinstance(data.negative_ip_adapter_embeds, list):
            raise ValueError("`negative_ip_adapter_embeds` must be a list")
        
        if data.ip_adapter_embeds is not None and data.negative_ip_adapter_embeds is not None:
            for i, ip_adapter_embed in enumerate(data.ip_adapter_embeds):
                if ip_adapter_embed.shape != data.negative_ip_adapter_embeds[i].shape:
                    raise ValueError(
                        "`ip_adapter_embeds` and `negative_ip_adapter_embeds` must have the same shape when passed directly, but"
                        f" got: `ip_adapter_embeds` {ip_adapter_embed.shape} != `negative_ip_adapter_embeds`"
                        f" {data.negative_ip_adapter_embeds[i].shape}."
                    )

    @torch.no_grad()
    def __call__(self, pipeline, state: PipelineState) -> PipelineState:
        data = self.get_block_state(state)
        self.check_inputs(pipeline, data)

        data.batch_size = data.prompt_embeds.shape[0]
        data.dtype = data.prompt_embeds.dtype

        _, seq_len, _ = data.prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        data.prompt_embeds = data.prompt_embeds.repeat(1, data.num_images_per_prompt, 1)
        data.prompt_embeds = data.prompt_embeds.view(data.batch_size * data.num_images_per_prompt, seq_len, -1)
    
        if data.negative_prompt_embeds is not None:
            _, seq_len, _ = data.negative_prompt_embeds.shape
            data.negative_prompt_embeds = data.negative_prompt_embeds.repeat(1, data.num_images_per_prompt, 1)
            data.negative_prompt_embeds = data.negative_prompt_embeds.view(data.batch_size * data.num_images_per_prompt, seq_len, -1)
        
        data.pooled_prompt_embeds = data.pooled_prompt_embeds.repeat(1, data.num_images_per_prompt, 1)
        data.pooled_prompt_embeds = data.pooled_prompt_embeds.view(data.batch_size * data.num_images_per_prompt, -1)
        
        if data.negative_pooled_prompt_embeds is not None:
            data.negative_pooled_prompt_embeds = data.negative_pooled_prompt_embeds.repeat(1, data.num_images_per_prompt, 1)
            data.negative_pooled_prompt_embeds = data.negative_pooled_prompt_embeds.view(data.batch_size * data.num_images_per_prompt, -1)
        
        if data.ip_adapter_embeds is not None:
            for i, ip_adapter_embed in enumerate(data.ip_adapter_embeds):
                data.ip_adapter_embeds[i] = torch.cat([ip_adapter_embed] * data.num_images_per_prompt, dim=0)

        if data.negative_ip_adapter_embeds is not None:
            for i, negative_ip_adapter_embed in enumerate(data.negative_ip_adapter_embeds):
                data.negative_ip_adapter_embeds[i] = torch.cat([negative_ip_adapter_embed] * data.num_images_per_prompt, dim=0)
        
        self.add_block_state(state, data)

        return pipeline, state


class StableDiffusionXLImg2ImgSetTimestepsStep(PipelineBlock):
    expected_components = ["scheduler"]
    model_name = "stable-diffusion-xl"

    @property
    def description(self) -> str:
        return (
            "Step that sets the timesteps for the scheduler and determines the initial noise level (latent_timestep) for image-to-image/inpainting generation."
            "The latent_timestep is calculated from the `strength` parameter - higher strength means starting from a noisier version of the input image."
        )

    @property
    def inputs(self) -> List[InputParam]:
        return [
            InputParam(
                "num_inference_steps", 
                default=50, 
                type_hint=int, 
                description="The number of denoising steps. More denoising steps usually lead to a higher quality image at the"
                " expense of slower inference."
            ),
            InputParam(
                "timesteps", 
                type_hint=Optional[torch.Tensor], 
                description="Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed will be used. Must be in descending order."
            ),
            InputParam(
                "sigmas", 
                type_hint=Optional[torch.Tensor], 
                description="Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed will be used."
            ),
            InputParam(
                "denoising_end", 
                type_hint=Optional[float], 
                description="When specified, determines the fraction (between 0.0 and 1.0) of the total denoising process to be completed before it is intentionally prematurely terminated. As a result, the returned sample will still retain a substantial amount of noise as determined by the discrete timesteps selected by the scheduler. The denoising_end parameter should ideally be utilized when this pipeline forms a part of a 'Mixture of Denoisers' multi-pipeline setup."
            ),
            InputParam(
                "strength",
                default=0.3,
                type_hint=float,
                description="Conceptually, indicates how much to transform the reference `image` (the masked portion of image for inpainting). Must be between 0 and 1. `image` "
                "will be used as a starting point, adding more noise to it the larger the `strength`. The number of "
                "denoising steps depends on the amount of noise initially added. When `strength` is 1, added noise will "
                "be maximum and the denoising process will run for the full number of iterations specified in "
                "`num_inference_steps`. A value of 1, therefore, essentially ignores `image`. Note that in the case of "
                "`denoising_start` being declared as an integer, the value of `strength` will be ignored."
            ),
            InputParam(
                "denoising_start",
                type_hint=Optional[float],
                description="The denoising start value to use for the scheduler. Determines the starting point of the denoising process."
            ),
            InputParam(
                "num_images_per_prompt",
                default=1,
                type_hint=int,
                description="The number of images to generate per prompt. Defaults to 1."
            ),
        ]

    @property
    def intermediates_inputs(self) -> List[str]:
        return [
            InputParam("batch_size", required=True, type_hint=int, description="Number of prompts, the final batch size of model inputs should be batch_size * num_images_per_prompt"), 
        ]

    @property
    def intermediates_outputs(self) -> List[str]:
        return [
            OutputParam("timesteps", type_hint=torch.Tensor, description="The timesteps to use for inference"), 
            OutputParam("num_inference_steps", type_hint=int, description="The number of denoising steps to perform at inference time"), 
            OutputParam("latent_timestep", type_hint=torch.Tensor, description="The timestep that represents the initial noise level for image-to-image generation")
        ]

    def __init__(self):
        super().__init__()
        self.components["scheduler"] = None

    @torch.no_grad()
    def __call__(self, pipeline, state: PipelineState) -> PipelineState:
        data = self.get_block_state(state)

        data.device = pipeline._execution_device

        data.timesteps, data.num_inference_steps = retrieve_timesteps(
            pipeline.scheduler, data.num_inference_steps, data.device, data.timesteps, data.sigmas
        )

        def denoising_value_valid(dnv):
            return isinstance(dnv, float) and 0 < dnv < 1

        data.timesteps, data.num_inference_steps = pipeline.get_timesteps(
            data.num_inference_steps,
            data.strength,
            data.device,
            denoising_start=data.denoising_start if denoising_value_valid(data.denoising_start) else None,
        )
        data.latent_timestep = data.timesteps[:1].repeat(data.batch_size * data.num_images_per_prompt)

        if data.denoising_end is not None and isinstance(data.denoising_end, float) and data.denoising_end > 0 and data.denoising_end < 1:
            data.discrete_timestep_cutoff = int(
                round(
                    pipeline.scheduler.config.num_train_timesteps
                    - (data.denoising_end * pipeline.scheduler.config.num_train_timesteps)
                )
            )
            data.num_inference_steps = len(list(filter(lambda ts: ts >= data.discrete_timestep_cutoff, data.timesteps)))
            data.timesteps = data.timesteps[:data.num_inference_steps]

        self.add_block_state(state, data)

        return pipeline, state


class StableDiffusionXLSetTimestepsStep(PipelineBlock):
    expected_components = ["scheduler"]
    model_name = "stable-diffusion-xl"
    
    @property
    def description(self) -> str:
        return (
            "Step that sets the scheduler's timesteps for inference"
        )

    @property
    def inputs(self) -> List[InputParam]:
        return [
            InputParam(
                "num_inference_steps", 
                default=50, 
                type_hint=int, 
                description="The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference."
            ),
            InputParam(
                "timesteps", 
                type_hint=Optional[torch.Tensor], 
                description="Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed will be used. Must be in descending order."
            ),
            InputParam(
                "sigmas", 
                type_hint=Optional[torch.Tensor], 
                description="Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed will be used."
            ),
            InputParam(
                "denoising_end", 
                type_hint=Optional[float], 
                description="When specified, determines the fraction (between 0.0 and 1.0) of the total denoising process to be completed before it is intentionally prematurely terminated. As a result, the returned sample will still retain a substantial amount of noise as determined by the discrete timesteps selected by the scheduler. The denoising_end parameter should ideally be utilized when this pipeline forms a part of a 'Mixture of Denoisers' multi-pipeline setup."
            ),
        ]

    @property
    def intermediates_outputs(self) -> List[OutputParam]:
        return [OutputParam("timesteps", type_hint=torch.Tensor, description="The timesteps to use for inference"), 
                OutputParam("num_inference_steps", type_hint=int, description="The number of denoising steps to perform at inference time")]

    @property
    def intermediates_inputs(self) -> List[InputParam]:
        return []
    
    def __init__(self):
        super().__init__()
        self.components["scheduler"] = None

    @torch.no_grad()
    def __call__(self, pipeline, state: PipelineState) -> PipelineState:
        data = self.get_block_state(state)

        data.device = pipeline._execution_device

        data.timesteps, data.num_inference_steps = retrieve_timesteps(
            pipeline.scheduler, data.num_inference_steps, data.device, data.timesteps, data.sigmas
        )

        if data.denoising_end is not None and isinstance(data.denoising_end, float) and data.denoising_end > 0 and data.denoising_end < 1:
            data.discrete_timestep_cutoff = int(
                round(
                    pipeline.scheduler.config.num_train_timesteps
                    - (data.denoising_end * pipeline.scheduler.config.num_train_timesteps)
                )
            )
            data.num_inference_steps = len(list(filter(lambda ts: ts >= data.discrete_timestep_cutoff, data.timesteps)))
            data.timesteps = data.timesteps[:data.num_inference_steps]

        self.add_block_state(state, data)
        return pipeline, state


class StableDiffusionXLInpaintPrepareLatentsStep(PipelineBlock):
    expected_components = ["scheduler"]
    model_name = "stable-diffusion-xl"

    @property
    def description(self) -> str:
        return (
            "Step that prepares the latents for the inpainting process"
        )

    @property
    def inputs(self) -> List[Tuple[str, Any]]:
        return [
            InputParam(
                "generator", 
                type_hint=Optional[Union[torch.Generator, List[torch.Generator]]], 
                description="One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html) "
                           "to make generation deterministic."),
            InputParam(
                "latents", 
                type_hint=Optional[torch.Tensor], 
                description="Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image generation. Can be used to tweak the same generation with different prompts. If not provided, a latents tensor will ge generated by sampling using the supplied random `generator`."
            ),
            InputParam(
                "num_images_per_prompt", 
                default=1, 
                type_hint=int, 
                description="The number of images to generate per prompt"
            ),
            InputParam(
                "denoising_start",
                type_hint=Optional[float],
                description="When specified, indicates the fraction (between 0.0 and 1.0) of the total denoising process to be bypassed before it is initiated. The initial part of the denoising process is skipped and it is assumed that the passed `image` is a partly denoised image. Note that when this is specified, strength will be ignored. Useful for 'Mixture of Denoisers' multi-pipeline setups."
            ),
            InputParam(
                "strength", 
                default=0.9999, 
                type_hint=float, 
                description="Conceptually, indicates how much to transform the reference `image` (the masked portion of image for inpainting). Must be between 0 and 1. `image` "
                "will be used as a starting point, adding more noise to it the larger the `strength`. The number of "
                "denoising steps depends on the amount of noise initially added. When `strength` is 1, added noise will "
                "be maximum and the denoising process will run for the full number of iterations specified in "
                "`num_inference_steps`. A value of 1, therefore, essentially ignores `image`. Note that in the case of "
                "`denoising_start` being declared as an integer, the value of `strength` will be ignored."
            ),
        ]

    @property
    def intermediates_inputs(self) -> List[str]:
        return [
            InputParam(
                "batch_size", 
                required=True, 
                type_hint=int, 
                description="Number of prompts, the final batch size of model inputs should be batch_size * num_images_per_prompt. Can be generated in input step."
            ), 
            InputParam(
                "latent_timestep", 
                required=True, 
                type_hint=torch.Tensor, 
                description="The timestep that represents the initial noise level for image-to-image/inpainting generation. Can be generated in set_timesteps step."
            ), 
            InputParam(
                "image_latents", 
                required=True, 
                type_hint=torch.Tensor, 
                description="The latents representing the reference image for image-to-image/inpainting generation. Can be generated in vae_encode step."
            ), 
            InputParam(
                "mask", 
                required=True, 
                type_hint=torch.Tensor, 
                description="The mask for the inpainting generation. Can be generated in vae_encode step."
            ), 
            InputParam(
                "masked_image_latents", 
                type_hint=torch.Tensor, 
                description="The masked image latents for the inpainting generation (only for inpainting-specific unet). Can be generated in vae_encode step."
            ),
            InputParam(
                "dtype", 
                type_hint=torch.dtype, 
                description="The dtype of the model inputs"
            )
        ]

    @property
    def intermediates_outputs(self) -> List[str]:
        return [OutputParam("latents", type_hint=torch.Tensor, description="The initial latents to use for the denoising process"), 
                OutputParam("mask", type_hint=torch.Tensor, description="The mask to use for inpainting generation"), 
                OutputParam("masked_image_latents", type_hint=torch.Tensor, description="The masked image latents to use for the inpainting generation (only for inpainting-specific unet)"), 
                OutputParam("noise", type_hint=torch.Tensor, description="The noise added to the image latents, used for inpainting generation")]

    def __init__(self):
        super().__init__()
        self.components["scheduler"] = None

    @torch.no_grad()
    def __call__(self, pipeline: DiffusionPipeline, state: PipelineState) -> PipelineState:
        data = self.get_block_state(state)

        data.dtype = data.dtype if data.dtype is not None else pipeline.vae.dtype
        data.device = pipeline._execution_device
        
        data.is_strength_max = data.strength == 1.0

        # for non-inpainting specific unet, we do not need masked_image_latents
        if hasattr(pipeline,"unet") and pipeline.unet is not None:
            if pipeline.unet.config.in_channels == 4:
                data.masked_image_latents = None

        data.add_noise = True if data.denoising_start is None else False

        data.height = data.image_latents.shape[-2] * pipeline.vae_scale_factor
        data.width = data.image_latents.shape[-1] * pipeline.vae_scale_factor

        data.latents, data.noise = pipeline.prepare_latents_inpaint(
            data.batch_size * data.num_images_per_prompt,
            pipeline.num_channels_latents,
            data.height,
            data.width,
            data.dtype,
            data.device,
            data.generator,
            data.latents,
            image=data.image_latents,
            timestep=data.latent_timestep,
            is_strength_max=data.is_strength_max,
            add_noise=data.add_noise,
            return_noise=True,
            return_image_latents=False,
        )

        # 7. Prepare mask latent variables
        data.mask, data.masked_image_latents = pipeline.prepare_mask_latents(
            data.mask,
            data.masked_image_latents,
            data.batch_size * data.num_images_per_prompt,
            data.height,
            data.width,
            data.dtype,
            data.device,
            data.generator,
        )

        self.add_block_state(state, data)

        return pipeline, state


class StableDiffusionXLImg2ImgPrepareLatentsStep(PipelineBlock):
    expected_components = ["vae", "scheduler"]
    model_name = "stable-diffusion-xl"

    @property
    def description(self) -> str:
        return (
            "Step that prepares the latents for the image-to-image generation process"
        )

    @property
    def inputs(self) -> List[Tuple[str, Any]]:
        return [
            InputParam(
                "generator", 
                type_hint=Optional[Union[torch.Generator, List[torch.Generator]]], 
                description="One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html) "
                           "to make generation deterministic."
            ),
            InputParam(
                "latents", 
                type_hint=Optional[torch.Tensor], 
                description="Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image generation. Can be used to tweak the same generation with different prompts. If not provided, a latents tensor will ge generated by sampling using the supplied random `generator`."
            ),
            InputParam(
                "num_images_per_prompt", 
                default=1, 
                type_hint=int, 
                description="The number of images to generate per prompt"
            ),
            InputParam(
                "denoising_start", 
                type_hint=Optional[float], 
                description="When specified, indicates the fraction (between 0.0 and 1.0) of the total denoising process to be bypassed before it is initiated. The initial part of the denoising process is skipped and it is assumed that the passed `image` is a partly denoised image. Note that when this is specified, strength will be ignored. Useful for 'Mixture of Denoisers' multi-pipeline setups."
            ),
        ]

    @property
    def intermediates_inputs(self) -> List[InputParam]:
        return [
            InputParam("latent_timestep", required=True, type_hint=torch.Tensor, description="The timestep that represents the initial noise level for image-to-image/inpainting generation. Can be generated in set_timesteps step."), 
            InputParam("image_latents", required=True, type_hint=torch.Tensor, description="The latents representing the reference image for image-to-image/inpainting generation. Can be generated in vae_encode step."), 
            InputParam("batch_size", required=True, type_hint=int, description="Number of prompts, the final batch size of model inputs should be batch_size * num_images_per_prompt. Can be generated in input step."), 
            InputParam("dtype", required=True, type_hint=torch.dtype, description="The dtype of the model inputs")]

    @property
    def intermediates_outputs(self) -> List[OutputParam]:
        return [OutputParam("latents", type_hint=torch.Tensor, description="The initial latents to use for the denoising process")]

    def __init__(self):
        super().__init__()
        self.components["scheduler"] = None

    @torch.no_grad()
    def __call__(self, pipeline: DiffusionPipeline, state: PipelineState) -> PipelineState:
        data = self.get_block_state(state)

        data.dtype = data.dtype if data.dtype is not None else pipeline.vae.dtype
        data.device = pipeline._execution_device
        data.add_noise = True if data.denoising_start is None else False
        if data.latents is None:
            data.latents = pipeline.prepare_latents_img2img(
                data.image_latents,
                data.latent_timestep,
                data.batch_size,
                data.num_images_per_prompt,
                data.dtype,
                data.device,
                data.generator,
                data.add_noise,
            )

        self.add_block_state(state, data)

        return pipeline, state


class StableDiffusionXLPrepareLatentsStep(PipelineBlock):
    expected_components = ["scheduler"]
    model_name = "stable-diffusion-xl"

    @property
    def description(self) -> str:
        return (
            "Prepare latents step that prepares the latents for the text-to-image generation process"
        )

    @property
    def inputs(self) -> List[InputParam]:
        return [
            InputParam(
                "height", 
                type_hint=Optional[int], 
                description="The height in pixels of the generated image. This is set to 1024 by default for the best results. "
                           "Anything below 512 pixels won't work well for [stabilityai/stable-diffusion-xl-base-1.0]"
                           "(https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) and checkpoints that are not "
                           "specifically fine-tuned on low resolutions."),
            InputParam(
                "width", 
                type_hint=Optional[int], 
                description="The width in pixels of the generated image. This is set to 1024 by default for the best results. "
                           "Anything below 512 pixels won't work well for [stabilityai/stable-diffusion-xl-base-1.0]"
                           "(https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) and checkpoints that are not "
                           "specifically fine-tuned on low resolutions."),
            InputParam(
                "generator", 
                type_hint=Optional[Union[torch.Generator, List[torch.Generator]]], 
                description="One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html) "
                           "to make generation deterministic."
            ),
            InputParam(
                "latents", 
                type_hint=Optional[torch.Tensor], 
                description="Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image generation. Can be used to tweak the same generation with different prompts. If not provided, a latents tensor will ge generated by sampling using the supplied random `generator`."
            ),
            InputParam(
                "num_images_per_prompt", 
                default=1, 
                type_hint=int, 
                description="The number of images to generate per prompt"
            ),
        ]

    @property
    def intermediates_inputs(self) -> List[InputParam]:
        return [
            InputParam(
                "batch_size", 
                required=True, 
                type_hint=int, 
                description="Number of prompts, the final batch size of model inputs should be batch_size * num_images_per_prompt. Can be generated in input step."
            ), 
            InputParam(
                "dtype", 
                type_hint=torch.dtype, 
                description="The dtype of the model inputs"
            )
        ]

    @property
    def intermediates_outputs(self) -> List[OutputParam]:
        return [
            OutputParam(
                "latents", 
                type_hint=torch.Tensor, 
                description="The initial latents to use for the denoising process"
            )
        ]

    def __init__(self):
        super().__init__()
        self.components["scheduler"] = None

    @staticmethod
    def check_inputs(pipeline, data):
        if (
            data.height is not None
            and data.height % pipeline.vae_scale_factor != 0
            or data.width is not None
            and data.width % pipeline.vae_scale_factor != 0
        ):
            raise ValueError(
                f"`height` and `width` have to be divisible by {pipeline.vae_scale_factor} but are {data.height} and {data.width}."
            )

    @torch.no_grad()
    def __call__(self, pipeline: DiffusionPipeline, state: PipelineState) -> PipelineState:
        data = self.get_block_state(state)

        if data.dtype is None:
            data.dtype = pipeline.vae.dtype

        data.device = pipeline._execution_device

        self.check_inputs(pipeline, data)

        data.height = data.height or pipeline.default_sample_size * pipeline.vae_scale_factor
        data.width = data.width or pipeline.default_sample_size * pipeline.vae_scale_factor
        data.num_channels_latents = pipeline.num_channels_latents
        data.latents = pipeline.prepare_latents(
            data.batch_size * data.num_images_per_prompt,
            data.num_channels_latents,
            data.height,
            data.width,
            data.dtype,
            data.device,
            data.generator,
            data.latents,
        )

        self.add_block_state(state, data)

        return pipeline, state


class StableDiffusionXLImg2ImgPrepareAdditionalConditioningStep(PipelineBlock):
    expected_configs = ["requires_aesthetics_score"]
    model_name = "stable-diffusion-xl"

    @property
    def description(self) -> str:
        return (
            "Step that prepares the additional conditioning for the image-to-image/inpainting generation process"
        )

    @property
    def inputs(self) -> List[Tuple[str, Any]]:
        return [
            InputParam(
                "original_size", 
                type_hint=Optional[Tuple[int]], 
                description="If `original_size` is not the same as `target_size` the image will appear to be down- or upsampled. "
                           "`original_size` defaults to `(height, width)` if not specified. Part of SDXL's micro-conditioning as "
                           "explained in section 2.2 of https://huggingface.co/papers/2307.01952"
            ),
            InputParam(
                "target_size", 
                type_hint=Optional[Tuple[int]], 
                description="For most cases, `target_size` should be set to the desired height and width of the generated image. If "
                           "not specified it will default to `(height, width)`. Part of SDXL's micro-conditioning as explained in "
                           "section 2.2 of https://huggingface.co/papers/2307.01952"
            ),
            InputParam(
                "negative_original_size", 
                type_hint=Optional[Tuple[int]], 
                description="To negatively condition the generation process based on a specific image resolution. Part of SDXL's "
                           "micro-conditioning as explained in section 2.2 of https://huggingface.co/papers/2307.01952"
            ),
            InputParam(
                "negative_target_size", 
                type_hint=Optional[Tuple[int]], 
                description="To negatively condition the generation process based on a target image resolution. It should be as same "
                           "as the `target_size` for most cases. Part of SDXL's micro-conditioning as explained in section 2.2 of "
                           "https://huggingface.co/papers/2307.01952"
            ),
            InputParam(
                "crops_coords_top_left", 
                default=(0, 0), 
                type_hint=Tuple[int], 
                description="`crops_coords_top_left` can be used to generate an image that appears to be \"cropped\" from the position "
                           "`crops_coords_top_left` to (0, 0). Part of SDXL's micro-conditioning"
            ),
            InputParam(
                "negative_crops_coords_top_left", 
                default=(0, 0), 
                type_hint=Tuple[int], 
                description="To negatively condition the generation process based on a specific crop coordinates. Part of SDXL's "
                            "micro-conditioning"
            ),
            InputParam(
                "num_images_per_prompt", 
                default=1, 
                type_hint=int, 
                description="The number of images to generate per prompt."
            ),
            InputParam(
                "guidance_scale", 
                default=5.0, 
                type_hint=float, 
                description="Guidance scale as defined in Classifier-Free Diffusion Guidance. `guidance_scale` is defined as `w` of equation 2. "
                           "Higher guidance scale encourages to generate images that are closely linked to the text `prompt`, "
                           "usually at the expense of lower image quality."
            ),
            InputParam(
                "aesthetic_score", 
                default=6.0, 
                type_hint=float,
                description="Used to simulate an aesthetic score of the generated image by influencing the positive text condition. "
                           "Part of SDXL's micro-conditioning as explained in section 2.2 of https://huggingface.co/papers/2307.01952"
            ),
            InputParam(
                "negative_aesthetic_score", 
                default=2.0, 
                type_hint=float, 
                description="Part of SDXL's micro-conditioning as explained in section 2.2 of https://huggingface.co/papers/2307.01952. "
                           "Can be used to simulate an aesthetic score of the generated image by influencing the negative text condition."
            ),
        ]

    @property
    def intermediates_inputs(self) -> List[InputParam]:
        return [
            InputParam("latents", required=True, type_hint=torch.Tensor, description="The initial latents to use for the denoising process. Can be generated in prepare_latent step."), 
            InputParam("pooled_prompt_embeds", required=True, type_hint=torch.Tensor, description="The pooled prompt embeddings to use for the denoising process (used to determine shapes and dtypes for other additional conditioning inputs). Can be generated in text_encoder step."),
            InputParam("batch_size", required=True, type_hint=int, description="Number of prompts, the final batch size of model inputs should be batch_size * num_images_per_prompt. Can be generated in input step."),
        ]

    @property
    def intermediates_outputs(self) -> List[OutputParam]:
        return [OutputParam("add_time_ids", type_hint=torch.Tensor, description="The time ids to condition the denoising process"), 
                OutputParam("negative_add_time_ids", type_hint=torch.Tensor, description="The negative time ids to condition the denoising process"), 
                OutputParam("timestep_cond", type_hint=torch.Tensor, description="The timestep cond to use for LCM")]

    def __init__(self):
        super().__init__()
        self.configs["requires_aesthetics_score"] = False

    @torch.no_grad()
    def __call__(self, pipeline: DiffusionPipeline, state: PipelineState) -> PipelineState:
        data = self.get_block_state(state)
        data.device = pipeline._execution_device

        data.vae_scale_factor = pipeline.vae_scale_factor

        data.height, data.width = data.latents.shape[-2:]
        data.height = data.height * data.vae_scale_factor
        data.width = data.width * data.vae_scale_factor

        data.original_size = data.original_size or (data.height, data.width)
        data.target_size = data.target_size or (data.height, data.width)

        data.text_encoder_projection_dim = int(data.pooled_prompt_embeds.shape[-1])

        if data.negative_original_size is None:
            data.negative_original_size = data.original_size
        if data.negative_target_size is None:
            data.negative_target_size = data.target_size

        data.add_time_ids, data.negative_add_time_ids = pipeline._get_add_time_ids_img2img(
            data.original_size,
            data.crops_coords_top_left,
            data.target_size,
            data.aesthetic_score,
            data.negative_aesthetic_score,
            data.negative_original_size,
            data.negative_crops_coords_top_left,
            data.negative_target_size,
            dtype=data.pooled_prompt_embeds.dtype,
            text_encoder_projection_dim=data.text_encoder_projection_dim,
        )
        data.add_time_ids = data.add_time_ids.repeat(data.batch_size * data.num_images_per_prompt, 1).to(device=data.device)
        data.negative_add_time_ids = data.negative_add_time_ids.repeat(data.batch_size * data.num_images_per_prompt, 1).to(device=data.device)

        # Optionally get Guidance Scale Embedding for LCM
        data.timestep_cond = None
        if (
            hasattr(pipeline, "unet")
            and pipeline.unet is not None
            and pipeline.unet.config.time_cond_proj_dim is not None
        ):
            data.guidance_scale_tensor = torch.tensor(data.guidance_scale - 1).repeat(data.batch_size * data.num_images_per_prompt)
            data.timestep_cond = pipeline.get_guidance_scale_embedding(
                data.guidance_scale_tensor, embedding_dim=pipeline.unet.config.time_cond_proj_dim
            ).to(device=data.device, dtype=data.latents.dtype)

        self.add_block_state(state, data)
        return pipeline, state


class StableDiffusionXLPrepareAdditionalConditioningStep(PipelineBlock):
    model_name = "stable-diffusion-xl"

    @property
    def description(self) -> str:
        return (
            "Step that prepares the additional conditioning for the text-to-image generation process"
        )

    @property
    def inputs(self) -> List[Tuple[str, Any]]:
        return [
            InputParam(
                "original_size",
                type_hint=Tuple[int, int],
                default=(1024, 1024),
                description="The original size (height, width) of the image that conditions the generation process. If different from target_size, the image will appear to be down- or upsampled. Part of SDXL's micro-conditioning as explained in section 2.2 of https://huggingface.co/papers/2307.01952"
            ),
            InputParam(
                "target_size",
                type_hint=Tuple[int, int],
                default=(1024, 1024),
                description="The target size (height, width) of the generated image. For most cases, this should be set to the desired output dimensions. Part of SDXL's micro-conditioning as explained in section 2.2 of https://huggingface.co/papers/2307.01952"
            ),
            InputParam(
                "negative_original_size",
                type_hint=Tuple[int, int],
                default=(1024, 1024),
                description="The negative original size to condition against during generation. Part of SDXL's micro-conditioning as explained in section 2.2 of https://huggingface.co/papers/2307.01952. See: https://github.com/huggingface/diffusers/issues/4208"
            ),
            InputParam(
                "negative_target_size",
                type_hint=Tuple[int, int],
                default=(1024, 1024),
                description="The negative target size to condition against during generation. Should typically match target_size. Part of SDXL's micro-conditioning as explained in section 2.2 of https://huggingface.co/papers/2307.01952. See: https://github.com/huggingface/diffusers/issues/4208"
            ),
            InputParam(
                "crops_coords_top_left", 
                default=(0, 0), 
                type_hint=Tuple[int, int], 
                description="The top-left coordinates (x, y) used to condition the generation process. Setting this to (0, 0) typically produces well-centered images. Part of SDXL's micro-conditioning as explained in section 2.2 of https://huggingface.co/papers/2307.01952"
            ),
            InputParam(
                "negative_crops_coords_top_left", 
                default=(0, 0), 
                type_hint=Tuple[int, int], 
                description="The top-left coordinates (x, y) used to negatively condition the generation process. Part of SDXL's micro-conditioning as explained in section 2.2 of https://huggingface.co/papers/2307.01952. For more information, see: https://github.com/huggingface/diffusers/issues/4208"
            ),
            InputParam(
                "num_images_per_prompt", 
                default=1, 
                type_hint=int, 
                description="The number of images to generate per prompt"
            ),
            InputParam(
                "guidance_scale", 
                default=5.0, 
                type_hint=float, 
                description="Guidance scale as defined in Classifier-Free Diffusion Guidance. `guidance_scale` is defined as `w` of equation 2. "
                           "Higher guidance scale encourages to generate images that are closely linked to the text `prompt`, "
                      "usually at the expense of lower image quality."),
        ]

    @property
    def intermediates_inputs(self) -> List[InputParam]:
        return [
            InputParam(
                "latents", 
                required=True, 
                type_hint=torch.Tensor, 
                description="The initial latents to use for the denoising process. Can be generated in prepare_latent step."
            ), 
            InputParam(
                "pooled_prompt_embeds", 
                required=True, 
                type_hint=torch.Tensor, 
                description="The pooled prompt embeddings to use for the denoising process (used to determine shapes and dtypes for other additional conditioning inputs). Can be generated in text_encoder step."
            ),
            InputParam(
                "batch_size", 
                required=True, 
                type_hint=int, 
                description="Number of prompts, the final batch size of model inputs should be batch_size * num_images_per_prompt. Can be generated in input step."
            ),
        ]

    @property
    def intermediates_outputs(self) -> List[OutputParam]:
        return [OutputParam("add_time_ids", type_hint=torch.Tensor, description="The time ids to condition the denoising process"), 
                OutputParam("negative_add_time_ids", type_hint=torch.Tensor, description="The negative time ids to condition the denoising process"), 
                OutputParam("timestep_cond", type_hint=torch.Tensor, description="The timestep cond to use for LCM")]

    @torch.no_grad()
    def __call__(self, pipeline: DiffusionPipeline, state: PipelineState) -> PipelineState:
        data = self.get_block_state(state)
        data.device = pipeline._execution_device

        data.height, data.width = data.latents.shape[-2:]
        data.height = data.height * pipeline.vae_scale_factor
        data.width = data.width * pipeline.vae_scale_factor

        data.original_size = data.original_size or (data.height, data.width)
        data.target_size = data.target_size or (data.height, data.width)

        data.text_encoder_projection_dim = int(data.pooled_prompt_embeds.shape[-1])

        data.add_time_ids = pipeline._get_add_time_ids(
            data.original_size,
            data.crops_coords_top_left,
            data.target_size,
            data.pooled_prompt_embeds.dtype,
            text_encoder_projection_dim=data.text_encoder_projection_dim,
        )
        if data.negative_original_size is not None and data.negative_target_size is not None:
            data.negative_add_time_ids = pipeline._get_add_time_ids(
                data.negative_original_size,
                data.negative_crops_coords_top_left,
                data.negative_target_size,
                data.pooled_prompt_embeds.dtype,
                text_encoder_projection_dim=data.text_encoder_projection_dim,
            )
        else:
            data.negative_add_time_ids = data.add_time_ids

        data.add_time_ids = data.add_time_ids.repeat(data.batch_size * data.num_images_per_prompt, 1).to(device=data.device)
        data.negative_add_time_ids = data.negative_add_time_ids.repeat(data.batch_size * data.num_images_per_prompt, 1).to(device=data.device)

        # Optionally get Guidance Scale Embedding for LCM
        data.timestep_cond = None
        if (
            hasattr(pipeline, "unet")
            and pipeline.unet is not None
            and pipeline.unet.config.time_cond_proj_dim is not None
        ):
            data.guidance_scale_tensor = torch.tensor(data.guidance_scale - 1).repeat(data.batch_size * data.num_images_per_prompt)
            data.timestep_cond = pipeline.get_guidance_scale_embedding(
                data.guidance_scale_tensor, embedding_dim=pipeline.unet.config.time_cond_proj_dim
            ).to(device=data.device, dtype=data.latents.dtype)

        self.add_block_state(state, data)
        return pipeline, state


class StableDiffusionXLDenoiseStep(PipelineBlock):
    expected_components = ["unet", "scheduler", "guider"]
    model_name = "stable-diffusion-xl"

    @property
    def description(self) -> str:
        return (
            "Step that iteratively denoise the latents for the text-to-image/image-to-image/inpainting generation process"
        )

    @property
    def inputs(self) -> List[Tuple[str, Any]]:
        return [
            InputParam(
                "guidance_scale", 
                type_hint=float,
                default=5.0,
                description="Guidance scale as defined in Classifier-Free Diffusion Guidance. Higher values encourage images closely linked to the text prompt, potentially at the expense of image quality. Enabled when > 1."
            ),
            InputParam(
                "guidance_rescale",
                type_hint=float,
                default=0.0,
                description="Guidance rescale factor (φ) to fix overexposure when using zero terminal SNR, as proposed in 'Common Diffusion Noise Schedules and Sample Steps are Flawed'."
            ),
            InputParam(
                "cross_attention_kwargs",
                type_hint=Optional[Dict[str, Any]],
                default=None,
                description="Optional kwargs dictionary passed to the AttentionProcessor."
            ),
            InputParam(
                "generator",
                type_hint=Optional[Union[torch.Generator, List[torch.Generator]]],
                description="One or a list of torch generator(s) to make generation deterministic."
            ),
            InputParam(
                "eta",
                type_hint=float,
                default=0.0,
                description="Parameter η in the DDIM paper. Only applies to DDIMScheduler, ignored for others."
            ),
            InputParam(
                "guider_kwargs",
                type_hint=Optional[Dict[str, Any]],
                default=None,
                description="Optional kwargs dictionary passed to the Guider."
            ),
            InputParam(
                "num_images_per_prompt",
                type_hint=int,
                default=1,
                description="The number of images to generate per prompt."
            ),
        ]

    @property
    def intermediates_inputs(self) -> List[str]:
        return [
            InputParam(
                "latents", 
                required=True, 
                type_hint=torch.Tensor, 
                description="The initial latents to use for the denoising process. Can be generated in prepare_latent step."
            ),
            InputParam(
                "batch_size", 
                required=True, 
                type_hint=int, 
                description="Number of prompts, the final batch size of model inputs should be batch_size * num_images_per_prompt. Can be generated in input step."
            ),
            InputParam(
                "timesteps", 
                required=True, 
                type_hint=torch.Tensor, 
                description="The timesteps to use for the denoising process. Can be generated in set_timesteps step."
            ),
            InputParam(
                "num_inference_steps", 
                required=True, 
                type_hint=int, 
                description="The number of inference steps to use for the denoising process. Can be generated in set_timesteps step."
            ),
            InputParam(
                "pooled_prompt_embeds", 
                required=True, 
                type_hint=torch.Tensor, 
                description="The pooled prompt embeddings to use to condition the denoising process. Can be generated in text_encoder step."
            ),
            InputParam(
                "negative_pooled_prompt_embeds", 
                type_hint=Optional[torch.Tensor], 
                description="The negative pooled prompt embeddings to use to condition the denoising process. Can be generated in text_encoder step.    "
            ),
            InputParam(
                "add_time_ids", 
                required=True, 
                type_hint=torch.Tensor, 
                description="The time ids to use as additional conditioning for the denoising process. Can be generated in prepare_additional_conditioning step."
            ),
            InputParam(
                "negative_add_time_ids", 
                type_hint=Optional[torch.Tensor], 
                description="The negative time ids to use as additional conditioning for the denoising process. Can be generated in prepare_additional_conditioning step."
            ),
            InputParam(
                "prompt_embeds", 
                required=True, 
                type_hint=torch.Tensor, 
                description="The prompt embeddings to use to condition the denoising process. Can be generated in text_encoder step."
            ),
            InputParam(
                "negative_prompt_embeds", 
                type_hint=Optional[torch.Tensor], 
                description="The negative prompt embeddings to use to condition the denoising process. Can be generated in text_encoder step.   "
            ),
            InputParam(
                "timestep_cond", 
                type_hint=Optional[torch.Tensor], 
                description="The guidance scale embedding to use for Latent Consistency Models(LCMs). Can be generated in prepare_additional_conditioning step."
            ),
            InputParam(
                "mask", 
                type_hint=Optional[torch.Tensor], 
                description="The mask to use for the denoising process, for inpainting task only. Can be generated in vae_encode or prepare_latent step."
            ),
            InputParam(
                "masked_image_latents", 
                type_hint=Optional[torch.Tensor], 
                description="The masked image latents to use for the denoising process, for inpainting task only. Can be generated in vae_encode or prepare_latent step."
            ),
            InputParam(
                "noise", 
                type_hint=Optional[torch.Tensor], 
                description="The noise added to the image latents, for inpainting task only. Can be generated in prepare_latent step."
            ),
            InputParam(
                "image_latents", 
                type_hint=Optional[torch.Tensor], 
                description="The image latents to use for the denoising process, for inpainting/image-to-image task only. Can be generated in vae_encode or prepare_latent step."
            ),
            InputParam(
                "ip_adapter_embeds", 
                type_hint=Optional[torch.Tensor], 
                description="The ip adapter embeddings to use to condition the denoising process, need to have ip adapter model loaded. Can be generated in ip_adapter step."
            ),
            InputParam(
                "negative_ip_adapter_embeds", 
                type_hint=Optional[torch.Tensor], 
                description="The negative ip adapter embeddings to use to condition the denoising process, need to have ip adapter model loaded. Can be generated in ip_adapter step."
            ),
        ]

    @property
    def intermediates_outputs(self) -> List[OutputParam]:
        return [OutputParam("latents", type_hint=torch.Tensor, description="The denoised latents")]

    def __init__(self):
        super().__init__()
        self.components["guider"] = CFGGuider()
        self.components["scheduler"] = None
        self.components["unet"] = None

    def check_inputs(self, pipeline, data):

        num_channels_unet = pipeline.unet.config.in_channels
        if num_channels_unet == 9:
            # default case for runwayml/stable-diffusion-inpainting
            if data.mask is None or data.masked_image_latents is None:
                raise ValueError("mask and masked_image_latents must be provided for inpainting-specific Unet")
            num_channels_latents = data.latents.shape[1]
            num_channels_mask = data.mask.shape[1]
            num_channels_masked_image = data.masked_image_latents.shape[1]
            if num_channels_latents + num_channels_mask + num_channels_masked_image != num_channels_unet:
                raise ValueError(
                    f"Incorrect configuration settings! The config of `pipeline.unet`: {pipeline.unet.config} expects"
                    f" {pipeline.unet.config.in_channels} but received `num_channels_latents`: {num_channels_latents} +"
                    f" `num_channels_mask`: {num_channels_mask} + `num_channels_masked_image`: {num_channels_masked_image}"
                    f" = {num_channels_latents+num_channels_masked_image+num_channels_mask}. Please verify the config of"
                    " `pipeline.unet` or your `mask_image` or `image` input."
                )

    
    @torch.no_grad()
    def __call__(self, pipeline, state: PipelineState) -> PipelineState:

        data = self.get_block_state(state)
        self.check_inputs(pipeline, data)

        data.num_channels_unet = pipeline.unet.config.in_channels
        data.disable_guidance = True if pipeline.unet.config.time_cond_proj_dim is not None else False

        # adding default guider arguments: do_classifier_free_guidance, guidance_scale, guidance_rescale
        data.guider_kwargs = data.guider_kwargs or {}
        data.guider_kwargs = {
            **data.guider_kwargs,
            "disable_guidance": data.disable_guidance,
            "guidance_scale": data.guidance_scale,
            "guidance_rescale": data.guidance_rescale,
            "batch_size": data.batch_size * data.num_images_per_prompt,
        }

        pipeline.guider.set_guider(pipeline, data.guider_kwargs)
        # Prepare conditional inputs using the guider
        data.prompt_embeds = pipeline.guider.prepare_input(
            data.prompt_embeds,
            data.negative_prompt_embeds,
        )
        data.add_time_ids = pipeline.guider.prepare_input(
            data.add_time_ids,
            data.negative_add_time_ids,
        )
        data.pooled_prompt_embeds = pipeline.guider.prepare_input(
            data.pooled_prompt_embeds,
            data.negative_pooled_prompt_embeds,
        )

        if data.num_channels_unet == 9:
            data.mask = pipeline.guider.prepare_input(data.mask, data.mask)
            data.masked_image_latents = pipeline.guider.prepare_input(data.masked_image_latents, data.masked_image_latents)

        data.added_cond_kwargs = {
            "text_embeds": data.pooled_prompt_embeds,
            "time_ids": data.add_time_ids,
        }

        if data.ip_adapter_embeds is not None:
            data.ip_adapter_embeds = pipeline.guider.prepare_input(data.ip_adapter_embeds, data.negative_ip_adapter_embeds)
            data.added_cond_kwargs["image_embeds"] = data.ip_adapter_embeds

        # Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        data.extra_step_kwargs = pipeline.prepare_extra_step_kwargs(data.generator, data.eta)
        data.num_warmup_steps = max(len(data.timesteps) - data.num_inference_steps * pipeline.scheduler.order, 0)

        with pipeline.progress_bar(total=data.num_inference_steps) as progress_bar:
            for i, t in enumerate(data.timesteps):
                # expand the latents if we are doing classifier free guidance
                data.latent_model_input = pipeline.guider.prepare_input(data.latents, data.latents)
                data.latent_model_input = pipeline.scheduler.scale_model_input(data.latent_model_input, t)

                # inpainting
                if data.num_channels_unet == 9:
                    data.latent_model_input = torch.cat([data.latent_model_input, data.mask, data.masked_image_latents], dim=1)

                # predict the noise residual
                data.noise_pred = pipeline.unet(
                    data.latent_model_input,
                    t,
                    encoder_hidden_states=data.prompt_embeds,
                    timestep_cond=data.timestep_cond,
                    cross_attention_kwargs=data.cross_attention_kwargs,
                    added_cond_kwargs=data.added_cond_kwargs,
                    return_dict=False,
                )[0]
                # perform guidance
                data.noise_pred = pipeline.guider.apply_guidance(
                    data.noise_pred,
                    timestep=t,
                    latents=data.latents,
                )
                # compute the previous noisy sample x_t -> x_t-1
                data.latents_dtype = data.latents.dtype
                data.latents = pipeline.scheduler.step(data.noise_pred, t, data.latents, **data.extra_step_kwargs, return_dict=False)[0]
                if data.latents.dtype != data.latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        data.latents = data.latents.to(data.latents_dtype)
                
                if data.num_channels_unet == 4 and data.mask is not None and data.image_latents is not None:
                    data.init_latents_proper = data.image_latents
                    if i < len(data.timesteps) - 1:
                        data.noise_timestep = data.timesteps[i + 1]
                        data.init_latents_proper = pipeline.scheduler.add_noise(
                            data.init_latents_proper, data.noise, torch.tensor([data.noise_timestep])
                        )

                    data.latents = (1 - data.mask) * data.init_latents_proper + data.mask * data.latents

                if i == len(data.timesteps) - 1 or ((i + 1) > data.num_warmup_steps and (i + 1) % pipeline.scheduler.order == 0):
                    progress_bar.update()

        pipeline.guider.reset_guider(pipeline)
        self.add_block_state(state, data)

        return pipeline, state


class StableDiffusionXLControlNetDenoiseStep(PipelineBlock):
    expected_components = ["unet", "controlnet", "scheduler", "guider", "controlnet_guider"]
    model_name = "stable-diffusion-xl"

    @property
    def description(self) -> str:
        return "step that iteratively denoise the latents for the text-to-image/image-to-image/inpainting generation process. Using ControlNet to condition the denoising process"

    @property
    def inputs(self) -> List[Tuple[str, Any]]:
        return [
            InputParam(
                "control_image",
                required=True,
                type_hint=PipelineImageInput,
                description="The ControlNet input condition to provide guidance to the unet for generation. If passed as torch.Tensor, it is used as-is. PIL.Image.Image inputs are accepted and default to image dimensions. For multiple ControlNets, pass images as a list for proper batching."
            ),
            InputParam(
                "control_guidance_start",
                default=0.0,
                type_hint=Union[float, List[float]],
                description="The percentage of total steps at which the ControlNet starts applying."
            ),
            InputParam(
                "control_guidance_end",
                default=1.0,
                type_hint=Union[float, List[float]],
                description="The percentage of total steps at which the ControlNet stops applying."
            ),
            InputParam(
                "controlnet_conditioning_scale",
                default=1.0,
                type_hint=Union[float, List[float]],
                description="Scale factor for ControlNet outputs before adding to unet residual. For multiple ControlNets, can be set as a list of scales."
            ),
            InputParam(
                "guess_mode",
                default=False,
                type_hint=bool,
                description="Enables ControlNet encoder to recognize input image content without prompts. Recommended guidance_scale: 3.0-5.0."
            ),
            InputParam(
                "num_images_per_prompt",
                default=1,
                type_hint=int,
                description="The number of images to generate per prompt."
            ),
            InputParam(
                "guidance_scale",
                default=5.0,
                type_hint=float,
                description="Guidance scale as defined in Classifier-Free Diffusion Guidance. Higher values encourage images closely linked to the text prompt, potentially at the expense of image quality. Enabled when > 1."
            ),
            InputParam(
                "guidance_rescale",
                default=0.0,
                type_hint=float,
                description="Guidance rescale factor (φ) to fix overexposure when using zero terminal SNR, as proposed in 'Common Diffusion Noise Schedules and Sample Steps are Flawed'."
            ),
            InputParam(
                "cross_attention_kwargs",
                default=None,
                type_hint=Optional[Dict[str, Any]],
                description="Optional kwargs dictionary passed to the AttentionProcessor."
            ),
            InputParam(
                "generator",
                default=None,
                type_hint=Optional[Union[torch.Generator, List[torch.Generator]]],
                description="One or a list of torch generator(s) to make generation deterministic."
            ),
            InputParam(
                "eta",
                default=0.0,
                type_hint=float,
                description="Parameter η in the DDIM paper. Only applies to DDIMScheduler, ignored for others."
            ),
            InputParam(
                "guider_kwargs",
                default=None,
                type_hint=Optional[Dict[str, Any]],
                description="Optional kwargs dictionary passed to the Guider."
            ),
        ]

    @property
    def intermediates_inputs(self) -> List[str]:
        return [
            InputParam(
                "latents", 
                required=True, 
                type_hint=torch.Tensor, 
                description="The initial latents to use for the denoising process. Can be generated in prepare_latent step."
            ),
            InputParam(
                "batch_size", 
                required=True, 
                type_hint=int, 
                description="Number of prompts, the final batch size of model inputs should be batch_size * num_images_per_prompt. Can be generated in input step."
            ),
            InputParam(
                "timesteps", 
                required=True, 
                type_hint=torch.Tensor, 
                description="The timesteps to use for the denoising process. Can be generated in set_timesteps step."
            ),
            InputParam(
                "num_inference_steps", 
                required=True, 
                type_hint=int, 
                description="The number of inference steps to use for the denoising process. Can be generated in set_timesteps step."
            ),
            InputParam(
                "prompt_embeds", 
                required=True, 
                type_hint=torch.Tensor, 
                description="The prompt embeddings used to condition the denoising process. Can be generated in text_encoder step."
            ),
            InputParam(
                "negative_prompt_embeds", 
                type_hint=Optional[torch.Tensor], 
                description="The negative prompt embeddings used to condition the denoising process. Can be generated in text_encoder step."
            ),
            InputParam(
                "add_time_ids", 
                required=True, 
                type_hint=torch.Tensor, 
                description="The time ids used to condition the denoising process. Can be generated in parepare_additional_conditioning step."
            ),
            InputParam(
                "negative_add_time_ids", 
                type_hint=Optional[torch.Tensor], 
                description="The negative time ids used to condition the denoising process. Can be generated in parepare_additional_conditioning step."
            ),
            InputParam(
                "pooled_prompt_embeds", 
                required=True, 
                type_hint=torch.Tensor, 
                description="The pooled prompt embeddings used to condition the denoising process. Can be generated in text_encoder step."
            ),
            InputParam(
                "negative_pooled_prompt_embeds", 
                type_hint=Optional[torch.Tensor], 
                description="The negative pooled prompt embeddings to use to condition the denoising process. Can be generated in text_encoder step."
            ),
            InputParam(
                "timestep_cond", 
                type_hint=Optional[torch.Tensor], 
                description="The guidance scale embedding to use for Latent Consistency Models(LCMs), can be generated by prepare_additional_conditioning step"
            ),
            InputParam(
                "mask", 
                type_hint=Optional[torch.Tensor], 
                description="The mask to use for the denoising process, for inpainting task only. Can be generated in vae_encode or prepare_latent step."
            ),
            InputParam(
                "masked_image_latents", 
                type_hint=Optional[torch.Tensor], 
                description="The masked image latents to use for the denoising process, for inpainting task only. Can be generated in vae_encode or prepare_latent step."
            ),
            InputParam(
                "noise", 
                type_hint=Optional[torch.Tensor], 
                description="The noise added to the image latents, for inpainting task only. Can be generated in prepare_latent step."
            ),
            InputParam(
                "image_latents", 
                type_hint=Optional[torch.Tensor], 
                description="The image latents to use for the denoising process, for inpainting/image-to-image task only. Can be generated in vae_encode or prepare_latent step."
            ),
            InputParam(
                "crops_coords", 
                type_hint=Optional[Tuple[int]], 
                description="The crop coordinates to use for preprocess/postprocess the image and mask, for inpainting task only. Can be generated in vae_encode step."
            ),
            InputParam(
                "ip_adapter_embeds", 
                type_hint=Optional[torch.Tensor], 
                description="The ip adapter embeddings to use to condition the denoising process, need to have ip adapter model loaded. Can be generated in ip_adapter step."
            ),
            InputParam(
                "negative_ip_adapter_embeds", 
                type_hint=Optional[torch.Tensor], 
                description="The negative ip adapter embeddings to use to condition the denoising process, need to have ip adapter model loaded. Can be generated in ip_adapter step."
            ),
        ]

    @property
    def intermediates_outputs(self) -> List[OutputParam]:
        return [OutputParam("latents", type_hint=torch.Tensor, description="The denoised latents")]

    def __init__(self):
        super().__init__()
        self.components["guider"] = CFGGuider()
        self.components["controlnet_guider"] = CFGGuider()
        self.components["scheduler"] = None
        self.components["unet"] = None
        self.components["controlnet"] = None
        control_image_processor = VaeImageProcessor(do_convert_rgb=True, do_normalize=False)
        self.auxiliaries["control_image_processor"] = control_image_processor

    def check_inputs(self, pipeline, data):

        num_channels_unet = pipeline.unet.config.in_channels
        if num_channels_unet == 9:
            # default case for runwayml/stable-diffusion-inpainting
            if data.mask is None or data.masked_image_latents is None:
                raise ValueError("mask and masked_image_latents must be provided for inpainting-specific Unet")
            num_channels_latents = data.latents.shape[1]
            num_channels_mask = data.mask.shape[1]
            num_channels_masked_image = data.masked_image_latents.shape[1]
            if num_channels_latents + num_channels_mask + num_channels_masked_image != num_channels_unet:
                raise ValueError(
                    f"Incorrect configuration settings! The config of `pipeline.unet`: {pipeline.unet.config} expects"
                    f" {pipeline.unet.config.in_channels} but received `num_channels_latents`: {num_channels_latents} +"
                    f" `num_channels_mask`: {num_channels_mask} + `num_channels_masked_image`: {num_channels_masked_image}"
                    f" = {num_channels_latents+num_channels_masked_image+num_channels_mask}. Please verify the config of"
                    " `pipeline.unet` or your `mask_image` or `image` input."
                )

    @torch.no_grad()
    def __call__(self, pipeline, state: PipelineState) -> PipelineState:
        
        data = self.get_block_state(state)
        self.check_inputs(pipeline, data)
        
        data.num_channels_unet = pipeline.unet.config.in_channels
        
        # (1) prepare controlnet inputs
        
        data.device = pipeline._execution_device

        data.height, data.width = data.latents.shape[-2:]
        data.height = data.height * pipeline.vae_scale_factor
        data.width = data.width * pipeline.vae_scale_factor

        controlnet = pipeline.controlnet._orig_mod if is_compiled_module(pipeline.controlnet) else pipeline.controlnet

        # (1.1)
        # control_guidance_start/control_guidance_end (align format)
        if not isinstance(data.control_guidance_start, list) and isinstance(data.control_guidance_end, list):
            data.control_guidance_start = len(data.control_guidance_end) * [data.control_guidance_start]
        elif not isinstance(data.control_guidance_end, list) and isinstance(data.control_guidance_start, list):
            data.control_guidance_end = len(data.control_guidance_start) * [data.control_guidance_end]
        elif not isinstance(data.control_guidance_start, list) and not isinstance(data.control_guidance_end, list):
            mult = len(controlnet.nets) if isinstance(controlnet, MultiControlNetModel) else 1
            data.control_guidance_start, data.control_guidance_end = (
                mult * [data.control_guidance_start],
                mult * [data.control_guidance_end],
            )

        # (1.2)
        # controlnet_conditioning_scale (align format)
        if isinstance(controlnet, MultiControlNetModel) and isinstance(data.controlnet_conditioning_scale, float):
            data.controlnet_conditioning_scale = [data.controlnet_conditioning_scale] * len(controlnet.nets)

        # (1.3)
        # global_pool_conditions    
        data.global_pool_conditions = (
            controlnet.config.global_pool_conditions
            if isinstance(controlnet, ControlNetModel)
            else controlnet.nets[0].config.global_pool_conditions
        )
        # (1.4)
        # guess_mode    
        data.guess_mode = data.guess_mode or data.global_pool_conditions

        # (1.5)
        # control_image
        if isinstance(controlnet, ControlNetModel):
            data.control_image = pipeline.prepare_control_image(
                image=data.control_image,
                width=data.width,
                height=data.height,
                batch_size=data.batch_size * data.num_images_per_prompt,
                num_images_per_prompt=data.num_images_per_prompt,
                device=data.device,
                dtype=controlnet.dtype,
                crops_coords=data.crops_coords,
            )
        elif isinstance(controlnet, MultiControlNetModel):
            control_images = []

            for control_image_ in data.control_image:
                control_image = pipeline.prepare_control_image(
                    image=control_image_,
                    width=data.width,
                    height=data.height,
                    batch_size=data.batch_size * data.num_images_per_prompt,
                    num_images_per_prompt=data.num_images_per_prompt,
                    device=data.device,
                    dtype=controlnet.dtype,
                    crops_coords=data.crops_coords,
                )

                control_images.append(control_image)

            data.control_image = control_images
        else:
            assert False

        # (1.6)
        # controlnet_keep
        data.controlnet_keep = []
        for i in range(len(data.timesteps)):
            keeps = [
                1.0 - float(i / len(data.timesteps) < s or (i + 1) / len(data.timesteps) > e)
                for s, e in zip(data.control_guidance_start, data.control_guidance_end)
            ]
            data.controlnet_keep.append(keeps[0] if isinstance(controlnet, ControlNetModel) else keeps)

        # (2) Prepare conditional inputs for unet using the guider
        # adding default guider arguments: disable_guidance, guidance_scale, guidance_rescale
        data.disable_guidance = True if pipeline.unet.config.time_cond_proj_dim is not None else False
        data.guider_kwargs = data.guider_kwargs or {}
        data.guider_kwargs = {
            **data.guider_kwargs,
            "disable_guidance": data.disable_guidance,
            "guidance_scale": data.guidance_scale,
            "guidance_rescale": data.guidance_rescale,
            "batch_size": data.batch_size * data.num_images_per_prompt,
        }
        pipeline.guider.set_guider(pipeline, data.guider_kwargs)
        data.prompt_embeds = pipeline.guider.prepare_input(
            data.prompt_embeds,
            data.negative_prompt_embeds,
        )
        data.add_time_ids = pipeline.guider.prepare_input(
            data.add_time_ids,
            data.negative_add_time_ids,
        )
        data.pooled_prompt_embeds = pipeline.guider.prepare_input(
            data.pooled_prompt_embeds,
            data.negative_pooled_prompt_embeds,
        )
        if data.num_channels_unet == 9:
            data.mask = pipeline.guider.prepare_input(data.mask, data.mask)
            data.masked_image_latents = pipeline.guider.prepare_input(data.masked_image_latents, data.masked_image_latents)

        data.added_cond_kwargs = {
            "text_embeds": data.pooled_prompt_embeds,
            "time_ids": data.add_time_ids,
        }

        if data.ip_adapter_embeds is not None:  
            data.ip_adapter_embeds = pipeline.guider.prepare_input(data.ip_adapter_embeds, data.negative_ip_adapter_embeds)
            data.added_cond_kwargs["image_embeds"] = data.ip_adapter_embeds

        # (3) Prepare conditional inputs for controlnet using the guider
        data.controlnet_disable_guidance = True if data.disable_guidance or data.guess_mode else False
        data.controlnet_guider_kwargs = data.guider_kwargs or {}
        data.controlnet_guider_kwargs = {
            **data.controlnet_guider_kwargs,
            "disable_guidance": data.controlnet_disable_guidance,
            "guidance_scale": data.guidance_scale,
            "guidance_rescale": data.guidance_rescale,
            "batch_size": data.batch_size * data.num_images_per_prompt,
        }
        pipeline.controlnet_guider.set_guider(pipeline, data.controlnet_guider_kwargs)
        data.controlnet_prompt_embeds = pipeline.controlnet_guider.prepare_input(data.prompt_embeds)
        data.controlnet_added_cond_kwargs = {
            "text_embeds": pipeline.controlnet_guider.prepare_input(data.pooled_prompt_embeds),
            "time_ids": pipeline.controlnet_guider.prepare_input(data.add_time_ids),
        }
        data.control_image = pipeline.controlnet_guider.prepare_input(data.control_image, data.control_image)

        # (4) Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        data.extra_step_kwargs = pipeline.prepare_extra_step_kwargs(data.generator, data.eta)
        data.num_warmup_steps = max(len(data.timesteps) - data.num_inference_steps * pipeline.scheduler.order, 0)

        # (5) Denoise loop
        with pipeline.progress_bar(total=data.num_inference_steps) as progress_bar:
            for i, t in enumerate(data.timesteps):
                # prepare latents for unet using the guider
                data.latent_model_input = pipeline.guider.prepare_input(data.latents, data.latents)

                # prepare latents for controlnet using the guider
                data.control_model_input = pipeline.controlnet_guider.prepare_input(data.latents, data.latents)

                if isinstance(data.controlnet_keep[i], list):
                    data.cond_scale = [c * s for c, s in zip(data.controlnet_conditioning_scale, data.controlnet_keep[i])]
                else:
                    data.controlnet_cond_scale = data.controlnet_conditioning_scale
                    if isinstance(data.controlnet_cond_scale, list):
                        data.controlnet_cond_scale = data.controlnet_cond_scale[0]
                    data.cond_scale = data.controlnet_cond_scale * data.controlnet_keep[i]

                data.down_block_res_samples, data.mid_block_res_sample = pipeline.controlnet(
                    pipeline.scheduler.scale_model_input(data.control_model_input, t),
                    t,
                    encoder_hidden_states=data.controlnet_prompt_embeds,
                    controlnet_cond=data.control_image,
                    conditioning_scale=data.cond_scale,
                    guess_mode=data.guess_mode,
                    added_cond_kwargs=data.controlnet_added_cond_kwargs,
                    return_dict=False,
                )

                # when we apply guidance for unet, but not for controlnet:
                # add 0 to the unconditional batch
                data.down_block_res_samples = pipeline.guider.prepare_input(
                    data.down_block_res_samples, [torch.zeros_like(d) for d in data.down_block_res_samples]
                )
                data.mid_block_res_sample = pipeline.guider.prepare_input(
                    data.mid_block_res_sample, torch.zeros_like(data.mid_block_res_sample)
                )

                data.latent_model_input = pipeline.scheduler.scale_model_input(data.latent_model_input, t)
                if data.num_channels_unet == 9:
                    data.latent_model_input = torch.cat([data.latent_model_input, data.mask, data.masked_image_latents], dim=1)

                data.noise_pred = pipeline.unet(
                    data.latent_model_input,
                    t,
                    encoder_hidden_states=data.prompt_embeds,
                    timestep_cond=data.timestep_cond,
                    cross_attention_kwargs=data.cross_attention_kwargs,
                    added_cond_kwargs=data.added_cond_kwargs,
                    down_block_additional_residuals=data.down_block_res_samples,
                    mid_block_additional_residual=data.mid_block_res_sample,
                    return_dict=False,
                )[0]
                # perform guidance
                data.noise_pred = pipeline.guider.apply_guidance(data.noise_pred, timestep=t, latents=data.latents)
                # compute the previous noisy sample x_t -> x_t-1
                data.latents_dtype = data.latents.dtype
                data.latents = pipeline.scheduler.step(data.noise_pred, t, data.latents, **data.extra_step_kwargs, return_dict=False)[0]
                if data.latents.dtype != data.latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        data.latents = data.latents.to(data.latents_dtype)

                
                if data.num_channels_unet == 4 and data.mask is not None and data.image_latents is not None:
                    data.init_latents_proper = data.image_latents
                    if i < len(data.timesteps) - 1:
                        data.noise_timestep = data.timesteps[i + 1]
                        data.init_latents_proper = pipeline.scheduler.add_noise(
                            data.init_latents_proper, data.noise, torch.tensor([data.noise_timestep])
                        )

                    data.latents = (1 - data.mask) * data.init_latents_proper + data.mask * data.latents

                if i == len(data.timesteps) - 1 or ((i + 1) > data.num_warmup_steps and (i + 1) % pipeline.scheduler.order == 0):
                    progress_bar.update()

        pipeline.guider.reset_guider(pipeline)
        pipeline.controlnet_guider.reset_guider(pipeline)
        
        self.add_block_state(state, data)

        return pipeline, state


class StableDiffusionXLControlNetUnionDenoiseStep(PipelineBlock):
    expected_components = ["unet", "controlnet", "scheduler", "guider", "controlnet_guider"]
    model_name = "stable-diffusion-xl"

    @property
    def description(self) -> str:
        return " The denoising step for the controlnet union model, works for inpainting, image-to-image, and text-to-image tasks"
    @property
    def inputs(self) -> List[Tuple[str, Any]]:
        return [
            InputParam(
                "control_image", 
                required=True,
                type_hint=PipelineImageInput,
                description="The ControlNet input condition to provide guidance to the unet for generation. If passed as torch.Tensor, it is used as-is. PIL.Image.Image inputs are accepted and default to image dimensions. For multiple ControlNets, pass images as a list for proper batching."),
            InputParam(
                "control_guidance_start",
                default=0.0,
                type_hint=Union[float, List[float]],
                description="The percentage of total steps at which the ControlNet starts applying."),
            InputParam(
                "control_guidance_end",
                default=1.0,
                type_hint=Union[float, List[float]],
                description="The percentage of total steps at which the ControlNet stops applying."),
            InputParam(
                "control_mode", 
                required=True, 
                type_hint=List[int], 
                description="The control mode for union controlnet, 0 for openpose, 1 for depth, 2 for hed/pidi/scribble/ted, 3 for canny/lineart/anime_lineart/mlsd, 4 for normal and 5 for segment"
            ),
            InputParam(
                "controlnet_conditioning_scale",
                default=1.0,
                type_hint=Union[float, List[float]],
                description="Scale factor for ControlNet outputs before adding to unet residual. For multiple ControlNets, can be set as a list of scales."
            ),
            InputParam(
                "guess_mode",
                default=False,
                type_hint=bool,
                description="Enables ControlNet encoder to recognize input image content without prompts. Recommended guidance_scale: 3.0-5.0."
            ),
            InputParam(
                "num_images_per_prompt",
                default=1,
                type_hint=int,
                description="The number of images to generate per prompt."
            ),
            InputParam(
                "guidance_scale",
                default=5.0,
                type_hint=float,
                description="Guidance scale as defined in Classifier-Free Diffusion Guidance. Higher values encourage images closely linked to the text prompt, potentially at the expense of image quality. Enabled when > 1."),
            InputParam(
                "guidance_rescale",
                default=0.0,
                type_hint=float,
                description="Guidance rescale factor (φ) to fix overexposure when using zero terminal SNR, as proposed in 'Common Diffusion Noise Schedules and Sample Steps are Flawed'."),
            InputParam(
                "cross_attention_kwargs",
                default=None,
                type_hint=Optional[Dict[str, Any]],
                description="Optional kwargs dictionary passed to the AttentionProcessor."),
            InputParam(
                "generator",
                default=None,
                type_hint=Optional[Union[torch.Generator, List[torch.Generator]]],
                description="One or a list of torch generator(s) to make generation deterministic."),
            InputParam(
                "eta",
                default=0.0,
                type_hint=float,
                description="Parameter η in the DDIM paper. Only applies to DDIMScheduler, ignored for others."),
            InputParam(
                "guider_kwargs",
                default=None,
                type_hint=Optional[Dict[str, Any]],
                description="Optional kwargs dictionary passed to the Guider."),
        ]

    @property
    def intermediates_inputs(self) -> List[str]:
        return [
            InputParam(
                "latents", 
                required=True,
                type_hint=torch.Tensor,
                description="The initial latents to use for the denoising process. Can be generated in prepare_latent step."
            ),
            InputParam(
                "batch_size", 
                required=True,
                type_hint=int,
                description="Number of prompts, the final batch size of model inputs should be batch_size * num_images_per_prompt. Can be generated in input step."
            ),
            InputParam(
                "timesteps", 
                required=True,
                type_hint=torch.Tensor,
                description="The timesteps to use for the denoising process. Can be generated in set_timesteps step."
            ),
            InputParam(
                "num_inference_steps", 
                required=True,
                type_hint=int,
                description="The number of inference steps to use for the denoising process. Can be generated in set_timesteps step."
            ),
            InputParam(
                "prompt_embeds", 
                required=True,
                type_hint=torch.Tensor,
                description="The prompt embeddings used to condition the denoising process. Can be generated in text_encoder step."
            ),
            InputParam(
                "negative_prompt_embeds",
                type_hint=Optional[torch.Tensor],
                description="The negative prompt embeddings used to condition the denoising process. Can be generated in text_encoder step. See: https://github.com/huggingface/diffusers/issues/4208"
            ),
            InputParam(
                "add_time_ids", 
                required=True,
                type_hint=torch.Tensor,
                description="The time ids used to condition the denoising process. Can be generated in prepare_additional_conditioning step."
            ),
            InputParam(
                "negative_add_time_ids",
                type_hint=Optional[torch.Tensor],
                description="The negative time ids used to condition the denoising process. Can be generated in prepare_additional_conditioning step.   "
            ),
            InputParam(
                "pooled_prompt_embeds", 
                required=True,
                type_hint=torch.Tensor,
                description="The pooled prompt embeddings used to condition the denoising process. Can be generated in text_encoder step."
            ),
            InputParam(
                "negative_pooled_prompt_embeds",
                type_hint=Optional[torch.Tensor],
                description="The negative pooled prompt embeddings to use to condition the denoising process. Can be generated in text_encoder step. See: https://github.com/huggingface/diffusers/issues/4208"
            ),
            InputParam(
                "timestep_cond",
                type_hint=Optional[torch.Tensor],
                description="The guidance scale embedding to use for Latent Consistency Models(LCMs). Can be generated in prepare_additional_conditioning step."
            ),
            InputParam(
                "mask",
                type_hint=Optional[torch.Tensor],
                description="The mask to use for the denoising process, for inpainting task only. Can be generated in vae_encode or prepare_latent step."
            ),
            InputParam(
                "masked_image_latents",
                type_hint=Optional[torch.Tensor],
                description="The masked image latents to use for the denoising process, for inpainting task only. Can be generated in vae_encode or prepare_latent step."
            ),
            InputParam(
                "noise",
                type_hint=Optional[torch.Tensor],
                description="The noise added to the image latents, for inpainting task only. Can be generated in prepare_latent step."
            ),
            InputParam(
                "image_latents",
                type_hint=Optional[torch.Tensor],
                description="The image latents to use for the denoising process, for inpainting/image-to-image task only. Can be generated in vae_encode or prepare_latent step."
            ),
            InputParam(
                "crops_coords",
                type_hint=Optional[Tuple[int]],
                description="The crop coordinates to use for preprocess/postprocess the image and mask, for inpainting task only. Can be generated in vae_encode or prepare_latent step."
            ),
            InputParam(
                "ip_adapter_embeds",
                type_hint=Optional[torch.Tensor],
                description="The ip adapter embeddings to use to condition the denoising process, need to have ip adapter model loaded. Can be generated in ip_adapter step."
            ),
            InputParam(
                "negative_ip_adapter_embeds",
                type_hint=Optional[torch.Tensor],
                description="The negative ip adapter embeddings to use to condition the denoising process, need to have ip adapter model loaded. Can be generated in ip_adapter step."
            ),
        ]

    @property
    def intermediates_outputs(self) -> List[str]:
        return [OutputParam("latents", type_hint=torch.Tensor, description="The denoised latents")]

    def __init__(self):
        super().__init__()
        self.components["guider"] = CFGGuider()
        self.components["controlnet_guider"] = CFGGuider()
        self.components["scheduler"] = None
        self.components["unet"] = None
        self.components["controlnet"] = None
        control_image_processor = VaeImageProcessor(do_convert_rgb=True, do_normalize=False)
        self.auxiliaries["control_image_processor"] = control_image_processor

    def check_inputs(self, pipeline, data):

        num_channels_unet = pipeline.unet.config.in_channels
        if num_channels_unet == 9:
            # default case for runwayml/stable-diffusion-inpainting
            if data.mask is None or data.masked_image_latents is None:
                raise ValueError("mask and masked_image_latents must be provided for inpainting-specific Unet")
            num_channels_latents = data.latents.shape[1]
            num_channels_mask = data.mask.shape[1]
            num_channels_masked_image = data.masked_image_latents.shape[1]
            if num_channels_latents + num_channels_mask + num_channels_masked_image != num_channels_unet:
                raise ValueError(
                    f"Incorrect configuration settings! The config of `pipeline.unet`: {pipeline.unet.config} expects"
                    f" {pipeline.unet.config.in_channels} but received `num_channels_latents`: {num_channels_latents} +"
                    f" `num_channels_mask`: {num_channels_mask} + `num_channels_masked_image`: {num_channels_masked_image}"
                    f" = {num_channels_latents+num_channels_masked_image+num_channels_mask}. Please verify the config of"
                    " `pipeline.unet` or your `mask_image` or `image` input."
                )

    @torch.no_grad()
    def __call__(self, pipeline, state: PipelineState) -> PipelineState:
        data = self.get_block_state(state)
        self.check_inputs(pipeline, data)

        data.num_channels_unet = pipeline.unet.config.in_channels

        # (1) prepare controlnet inputs
        data.device = pipeline._execution_device    
        data.height, data.width = data.latents.shape[-2:]
        data.height = data.height * pipeline.vae_scale_factor
        data.width = data.width * pipeline.vae_scale_factor

        controlnet = pipeline.controlnet._orig_mod if is_compiled_module(pipeline.controlnet) else pipeline.controlnet

        # (1.1)
        # control guidance
        if not isinstance(data.control_guidance_start, list) and isinstance(data.control_guidance_end, list):
            data.control_guidance_start = len(data.control_guidance_end) * [data.control_guidance_start]
        elif not isinstance(data.control_guidance_end, list) and isinstance(data.control_guidance_start, list):
            data.control_guidance_end = len(data.control_guidance_start) * [data.control_guidance_end]

        # (1.2)
        # global_pool_conditions & guess_mode
        data.global_pool_conditions = controlnet.config.global_pool_conditions
        data.guess_mode = data.guess_mode or data.global_pool_conditions

        # (1.3)
        # control_type  
        data.num_control_type = controlnet.config.num_control_type

        # (1.4)
        # control_type
        if not isinstance(data.control_image, list):
            data.control_image = [data.control_image]

        if not isinstance(data.control_mode, list):
            data.control_mode = [data.control_mode]

        if len(data.control_image) != len(data.control_mode):
            raise ValueError("Expected len(control_image) == len(control_type)")

        data.control_type = [0 for _ in range(data.num_control_type)]
        for control_idx in data.control_mode:
            data.control_type[control_idx] = 1

        data.control_type = torch.Tensor(data.control_type)

        # (1.5)
        # prepare control_image
        for idx, _ in enumerate(data.control_image):
            data.control_image[idx] = pipeline.prepare_control_image(
                image=data.control_image[idx],
                width=data.width,
                height=data.height,
                batch_size=data.batch_size * data.num_images_per_prompt,
                num_images_per_prompt=data.num_images_per_prompt,
                device=data.device,
                dtype=controlnet.dtype,
                crops_coords=data.crops_coords,
            )
            data.height, data.width = data.control_image[idx].shape[-2:]

        
        # (1.6)
        # controlnet_keep
        data.controlnet_keep = []
        for i in range(len(data.timesteps)):
            data.controlnet_keep.append(
                1.0
                - float(i / len(data.timesteps) < data.control_guidance_start or (i + 1) / len(data.timesteps) > data.control_guidance_end)
            )

        # (2) Prepare conditional inputs for unet using the guider
        # adding default guider arguments: disable_guidance, guidance_scale, guidance_rescale
        data.disable_guidance = True if pipeline.unet.config.time_cond_proj_dim is not None else False
        data.guider_kwargs = data.guider_kwargs or {}
        data.guider_kwargs = {
            **data.guider_kwargs,
            "disable_guidance": data.disable_guidance,
            "guidance_scale": data.guidance_scale,
            "guidance_rescale": data.guidance_rescale,
            "batch_size": data.batch_size * data.num_images_per_prompt,
        }
        pipeline.guider.set_guider(pipeline, data.guider_kwargs)
        data.prompt_embeds = pipeline.guider.prepare_input(
            data.prompt_embeds,
            data.negative_prompt_embeds,
        )
        data.add_time_ids = pipeline.guider.prepare_input(
            data.add_time_ids,
            data.negative_add_time_ids,
        )
        data.pooled_prompt_embeds = pipeline.guider.prepare_input(
            data.pooled_prompt_embeds,
            data.negative_pooled_prompt_embeds,
        )

        if data.num_channels_unet == 9:
            data.mask = pipeline.guider.prepare_input(data.mask, data.mask)
            data.masked_image_latents = pipeline.guider.prepare_input(data.masked_image_latents, data.masked_image_latents)

        data.added_cond_kwargs = {
            "text_embeds": data.pooled_prompt_embeds,
            "time_ids": data.add_time_ids,
        }

        if data.ip_adapter_embeds is not None:  
            data.ip_adapter_embeds = pipeline.guider.prepare_input(data.ip_adapter_embeds, data.negative_ip_adapter_embeds)
            data.added_cond_kwargs["image_embeds"] = data.ip_adapter_embeds

        # (3) Prepare conditional inputs for controlnet using the guider
        data.controlnet_disable_guidance = True if data.disable_guidance or data.guess_mode else False
        data.controlnet_guider_kwargs = data.guider_kwargs or {}
        data.controlnet_guider_kwargs = {
            **data.controlnet_guider_kwargs,
            "disable_guidance": data.controlnet_disable_guidance,
            "guidance_scale": data.guidance_scale,
            "guidance_rescale": data.guidance_rescale,
            "batch_size": data.batch_size * data.num_images_per_prompt,
        }
        pipeline.controlnet_guider.set_guider(pipeline, data.controlnet_guider_kwargs)
        data.controlnet_prompt_embeds = pipeline.controlnet_guider.prepare_input(data.prompt_embeds)
        data.controlnet_added_cond_kwargs = {
            "text_embeds": pipeline.controlnet_guider.prepare_input(data.pooled_prompt_embeds),
            "time_ids": pipeline.controlnet_guider.prepare_input(data.add_time_ids),
        }
        for idx, _ in enumerate(data.control_image):
            data.control_image[idx] = pipeline.controlnet_guider.prepare_input(data.control_image[idx], data.control_image[idx])

        data.control_type = (
            data.control_type.reshape(1, -1)
            .to(data.device, dtype=data.prompt_embeds.dtype)
        )
        repeat_by = data.batch_size * data.num_images_per_prompt // data.control_type.shape[0]
        data.control_type = data.control_type.repeat_interleave(repeat_by, dim=0)
        data.control_type = pipeline.controlnet_guider.prepare_input(data.control_type, data.control_type)

        # (4) Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        data.extra_step_kwargs = pipeline.prepare_extra_step_kwargs(data.generator, data.eta)
        data.num_warmup_steps = max(len(data.timesteps) - data.num_inference_steps * pipeline.scheduler.order, 0)


        with pipeline.progress_bar(total=data.num_inference_steps) as progress_bar:
            for i, t in enumerate(data.timesteps):
                # prepare latents for unet using the guider
                data.latent_model_input = pipeline.guider.prepare_input(data.latents, data.latents)

                # prepare latents for controlnet using the guider
                data.control_model_input = pipeline.controlnet_guider.prepare_input(data.latents, data.latents)

                if isinstance(data.controlnet_keep[i], list):
                    data.cond_scale = [c * s for c, s in zip(data.controlnet_conditioning_scale, data.controlnet_keep[i])]
                else:
                    data.controlnet_cond_scale = data.controlnet_conditioning_scale
                    if isinstance(data.controlnet_cond_scale, list):
                        data.controlnet_cond_scale = data.controlnet_cond_scale[0]
                    data.cond_scale = data.controlnet_cond_scale * data.controlnet_keep[i]

                data.down_block_res_samples, data.mid_block_res_sample = pipeline.controlnet(
                    pipeline.scheduler.scale_model_input(data.control_model_input, t),
                    t,
                    encoder_hidden_states=data.controlnet_prompt_embeds,
                    controlnet_cond=data.control_image,
                    control_type=data.control_type,
                    control_type_idx=data.control_mode,
                    conditioning_scale=data.cond_scale,
                    guess_mode=data.guess_mode,
                    added_cond_kwargs=data.controlnet_added_cond_kwargs,
                    return_dict=False,
                )

                # when we apply guidance for unet, but not for controlnet:
                # add 0 to the unconditional batch
                data.down_block_res_samples = pipeline.guider.prepare_input(
                    data.down_block_res_samples, [torch.zeros_like(d) for d in data.down_block_res_samples]
                )
                data.mid_block_res_sample = pipeline.guider.prepare_input(
                    data.mid_block_res_sample, torch.zeros_like(data.mid_block_res_sample)
                )

                data.latent_model_input = pipeline.scheduler.scale_model_input(data.latent_model_input, t)
                if data.num_channels_unet == 9:
                    data.latent_model_input = torch.cat([data.latent_model_input, data.mask, data.masked_image_latents], dim=1)

                data.noise_pred = pipeline.unet(
                    data.latent_model_input,
                    t,
                    encoder_hidden_states=data.prompt_embeds,
                    timestep_cond=data.timestep_cond,
                    cross_attention_kwargs=data.cross_attention_kwargs,
                    added_cond_kwargs=data.added_cond_kwargs,
                    down_block_additional_residuals=data.down_block_res_samples,
                    mid_block_additional_residual=data.mid_block_res_sample,
                    return_dict=False,
                )[0]
                # perform guidance
                data.noise_pred = pipeline.guider.apply_guidance(data.noise_pred, timestep=t, latents=data.latents)
                # compute the previous noisy sample x_t -> x_t-1
                data.latents_dtype = data.latents.dtype
                data.latents = pipeline.scheduler.step(data.noise_pred, t, data.latents, **data.extra_step_kwargs, return_dict=False)[0]
                if data.latents.dtype != data.latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        data.latents = data.latents.to(data.latents_dtype)

                if data.num_channels_unet == 9 and data.mask is not None and data.image_latents is not None:
                    data.init_latents_proper = data.image_latents
                    if i < len(data.timesteps) - 1:
                        data.noise_timestep = data.timesteps[i + 1]
                        data.init_latents_proper = pipeline.scheduler.add_noise(
                            data.init_latents_proper, data.noise, torch.tensor([data.noise_timestep])
                        )

                    data.latents = (1 - data.mask) * data.init_latents_proper + data.mask * data.latents

                if i == len(data.timesteps) - 1 or ((i + 1) > data.num_warmup_steps and (i + 1) % pipeline.scheduler.order == 0):
                    progress_bar.update()

        pipeline.guider.reset_guider(pipeline)
        pipeline.controlnet_guider.reset_guider(pipeline)
        
        self.add_block_state(state, data)

        return pipeline, state


class StableDiffusionXLDecodeLatentsStep(PipelineBlock):
    expected_components = ["vae"]
    model_name = "stable-diffusion-xl"

    @property
    def description(self) -> str:
        return "Step that decodes the denoised latents into images"

    @property
    def inputs(self) -> List[Tuple[str, Any]]:
        return [
            InputParam(
                "output_type",
                type_hint=str,
                default="pil",
                description="The output format of the generated image. Choose between PIL (PIL.Image.Image), torch.Tensor or np.array."
            ),
        ]

    @property
    def intermediates_inputs(self) -> List[str]:
        return [InputParam("latents", required=True, type_hint=torch.Tensor, description="The denoised latents from the denoising step")]

    @property
    def intermediates_outputs(self) -> List[str]:
        return [OutputParam("images", type_hint=Union[List[PIL.Image.Image], List[torch.Tensor], List[np.array]], description="The generated images, can be a PIL.Image.Image, torch.Tensor or a numpy array")]

    def __init__(self):
        super().__init__()
        self.components["vae"] = None
        self.auxiliaries["image_processor"] = VaeImageProcessor(vae_scale_factor=8)

    @torch.no_grad()
    def __call__(self, pipeline, state: PipelineState) -> PipelineState:
        data = self.get_block_state(state)

        if not data.output_type == "latent":
            # make sure the VAE is in float32 mode, as it overflows in float16
            data.needs_upcasting = pipeline.vae.dtype == torch.float16 and pipeline.vae.config.force_upcast

            if data.needs_upcasting:
                pipeline.upcast_vae()
                data.latents = data.latents.to(next(iter(pipeline.vae.post_quant_conv.parameters())).dtype)
            elif data.latents.dtype != pipeline.vae.dtype:
                if torch.backends.mps.is_available():
                    # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                    pipeline.vae = pipeline.vae.to(data.latents.dtype)

            # unscale/denormalize the latents
            # denormalize with the mean and std if available and not None
            data.has_latents_mean = (
                hasattr(pipeline.vae.config, "latents_mean") and pipeline.vae.config.latents_mean is not None
            )
            data.has_latents_std = (
                hasattr(pipeline.vae.config, "latents_std") and pipeline.vae.config.latents_std is not None
            )
            if data.has_latents_mean and data.has_latents_std:
                data.latents_mean = (
                    torch.tensor(pipeline.vae.config.latents_mean).view(1, 4, 1, 1).to(data.latents.device, data.latents.dtype)
                )
                data.latents_std = (
                    torch.tensor(pipeline.vae.config.latents_std).view(1, 4, 1, 1).to(data.latents.device, data.latents.dtype)
                )
                data.latents = data.latents * data.latents_std / pipeline.vae.config.scaling_factor + data.latents_mean
            else:
                data.latents = data.latents / pipeline.vae.config.scaling_factor

            data.images = pipeline.vae.decode(data.latents, return_dict=False)[0]

            # cast back to fp16 if needed
            if data.needs_upcasting:
                pipeline.vae.to(dtype=torch.float16)
        else:
            data.images = data.latents

        # apply watermark if available
        if hasattr(pipeline, "watermark") and pipeline.watermark is not None:
            data.images = pipeline.watermark.apply_watermark(data.images)

        data.images = pipeline.image_processor.postprocess(data.images, output_type=data.output_type)

        self.add_block_state(state, data)

        return pipeline, state


class StableDiffusionXLInpaintOverlayMaskStep(PipelineBlock):
    model_name = "stable-diffusion-xl"

    @property
    def description(self) -> str:
        return "A post-processing step that overlays the mask on the image (inpainting task only)" + \
               "only needed when you are using the `padding_mask_crop` option when pre-processing the image and mask"

    @property
    def inputs(self) -> List[Tuple[str, Any]]:
        return [
            InputParam(
                "image", 
                type_hint=PipelineImageInput,
                required=True, 
                description="The image(s) to modify with the pipeline, for img2img or inpainting task. When using for inpainting task, parts of the image will be masked out with `mask_image` and repainted according to `prompt`."
            ),
            InputParam(
                "mask_image", 
                type_hint=PipelineImageInput,
                required=True, 
                description="The mask image(s) to use for inpainting, white pixels in the mask will be repainted, while black pixels will be preserved. If mask_image is a PIL image, it will be converted to a single channel (luminance) before use. If it's a tensor, it should contain one color channel (L) instead of 3, so the expected shape would be (B, H, W, 1). Must be a `PIL.Image.Image`"
            ),
            InputParam(
                "padding_mask_crop", 
                type_hint=Optional[Tuple[int, int]],
                default=None, 
                description="The size of margin in the crop to be applied to the image and masking. If `None`, no crop is applied. If set, it will find a rectangular region with the same aspect ratio as the image that contains all masked areas, then expand that area by this margin. The image and mask_image are cropped to this expanded area before resizing to the original size for inpainting. Useful when the masked area is small in a large image with irrelevant background information."
            ),
        ]
    
    @property
    def intermediates_inputs(self) -> List[str]:
        return [
            InputParam("images", required=True, type_hint=Union[List[PIL.Image.Image], List[torch.Tensor], List[np.array]], description="The generated images from the decode step"),
            InputParam("crops_coords", required=True, type_hint=Tuple[int, int], description="The crop coordinates to use for preprocess/postprocess the image and mask, for inpainting task only. Can be generated in vae_encode step.")
        ]

    @property
    def intermediates_outputs(self) -> List[str]:
        return [OutputParam("images", type_hint=Union[List[PIL.Image.Image], List[torch.Tensor], List[np.array]], description="The generated images with the mask overlayed")]

    @torch.no_grad()
    def __call__(self, pipeline, state: PipelineState) -> PipelineState:
        data = self.get_block_state(state)

        if data.padding_mask_crop is not None and data.crops_coords is not None:
            data.images = [pipeline.image_processor.apply_overlay(data.mask_image, data.image, i, data.crops_coords) for i in data.images]

        self.add_block_state(state, data)

        return pipeline, state


class StableDiffusionXLOutputStep(PipelineBlock):
    model_name = "stable-diffusion-xl"

    @property
    def description(self) -> str:
        return "final step to return a [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] or a plain tuple."

    @property
    def inputs(self) -> List[Tuple[str, Any]]:
        return [(InputParam("return_dict", type_hint=bool, default=True, description="Whether or not to return a StableDiffusionXLPipelineOutput instead of a plain tuple."))] 

    @property
    def intermediates_inputs(self) -> List[str]:
        return [InputParam("images", required=True, type_hint=Union[List[PIL.Image.Image], List[torch.Tensor], List[np.array]], description="The generated images from the decode step.")]
    
    @property
    def intermediates_outputs(self) -> List[str]:
        return [OutputParam("images", description="The final images output, can be a tuple or a `StableDiffusionXLPipelineOutput`")]
    
    
    @torch.no_grad()
    def __call__(self, pipeline, state: PipelineState) -> PipelineState:
        data = self.get_block_state(state)

        if not data.return_dict:
            data.images = (data.images,)
        else:
            data.images = StableDiffusionXLPipelineOutput(images=data.images)
        self.add_block_state(state, data)
        return pipeline, state


# Encode
class StableDiffusionXLAutoVaeEncoderStep(AutoPipelineBlocks): 
    block_classes = [StableDiffusionXLInpaintVaeEncoderStep, StableDiffusionXLVaeEncoderStep]
    block_names = ["inpaint", "img2img"]
    block_trigger_inputs = ["mask_image", "image"]

    @property
    def description(self):
        return "Vae encoder step that encode the image inputs into their latent representations.\n" + \
               "This is an auto pipeline block that works for both inpainting and img2img tasks.\n" + \
               " - `StableDiffusionXLInpaintVaeEncoderStep` (inpaint) is used when both `mask_image` and `image` are provided.\n" + \
               " - `StableDiffusionXLVaeEncoderStep` (img2img) is used when only `image` is provided."


# Before denoise
class StableDiffusionXLBeforeDenoiseStep(SequentialPipelineBlocks):
    block_classes = [StableDiffusionXLInputStep, StableDiffusionXLSetTimestepsStep, StableDiffusionXLPrepareLatentsStep, StableDiffusionXLPrepareAdditionalConditioningStep]
    block_names = ["input", "set_timesteps", "prepare_latents", "prepare_add_cond"]

    @property
    def description(self):
        return "Before denoise step that prepare the inputs for the denoise step.\n" + \
               "This is a sequential pipeline blocks:\n" + \
               " - `StableDiffusionXLInputStep` is used to adjust the batch size of the model inputs\n" + \
               " - `StableDiffusionXLSetTimestepsStep` is used to set the timesteps\n" + \
               " - `StableDiffusionXLPrepareLatentsStep` is used to prepare the latents\n" + \
               " - `StableDiffusionXLPrepareAdditionalConditioningStep` is used to prepare the additional conditioning"

class StableDiffusionXLImg2ImgBeforeDenoiseStep(SequentialPipelineBlocks):
    block_classes = [StableDiffusionXLInputStep, StableDiffusionXLImg2ImgSetTimestepsStep, StableDiffusionXLImg2ImgPrepareLatentsStep, StableDiffusionXLImg2ImgPrepareAdditionalConditioningStep]
    block_names = ["input", "set_timesteps", "prepare_latents", "prepare_add_cond"]

    @property
    def description(self):
        return "Before denoise step that prepare the inputs for the denoise step for img2img task.\n" + \
               "This is a sequential pipeline blocks:\n" + \
               " - `StableDiffusionXLInputStep` is used to adjust the batch size of the model inputs\n" + \
               " - `StableDiffusionXLImg2ImgSetTimestepsStep` is used to set the timesteps\n" + \
               " - `StableDiffusionXLImg2ImgPrepareLatentsStep` is used to prepare the latents\n" + \
               " - `StableDiffusionXLImg2ImgPrepareAdditionalConditioningStep` is used to prepare the additional conditioning"

class StableDiffusionXLInpaintBeforeDenoiseStep(SequentialPipelineBlocks):
    block_classes = [StableDiffusionXLInputStep, StableDiffusionXLImg2ImgSetTimestepsStep, StableDiffusionXLInpaintPrepareLatentsStep, StableDiffusionXLImg2ImgPrepareAdditionalConditioningStep]
    block_names = ["input", "set_timesteps", "prepare_latents", "prepare_add_cond"]

    @property
    def description(self):
        return "Before denoise step that prepare the inputs for the denoise step for inpainting task.\n" + \
               "This is a sequential pipeline blocks:\n" + \
               " - `StableDiffusionXLInputStep` is used to adjust the batch size of the model inputs\n" + \
               " - `StableDiffusionXLImg2ImgSetTimestepsStep` is used to set the timesteps\n" + \
               " - `StableDiffusionXLInpaintPrepareLatentsStep` is used to prepare the latents\n" + \
               " - `StableDiffusionXLImg2ImgPrepareAdditionalConditioningStep` is used to prepare the additional conditioning"


class StableDiffusionXLAutoBeforeDenoiseStep(AutoPipelineBlocks):
    block_classes = [StableDiffusionXLInpaintBeforeDenoiseStep, StableDiffusionXLImg2ImgBeforeDenoiseStep, StableDiffusionXLBeforeDenoiseStep]
    block_names = ["inpaint", "img2img", "text2img"]
    block_trigger_inputs = ["mask", "image_latents", None]

    @property
    def description(self):
        return "Before denoise step that prepare the inputs for the denoise step.\n" + \
               "This is an auto pipeline block that works for text2img, img2img and inpainting tasks.\n" + \
               " - `StableDiffusionXLInpaintBeforeDenoiseStep` (inpaint) is used when both `mask` and `image_latents` are provided.\n" + \
               " - `StableDiffusionXLImg2ImgBeforeDenoiseStep` (img2img) is used when only `image_latents` is provided.\n" + \
               " - `StableDiffusionXLBeforeDenoiseStep` (text2img) is used when both `image_latents` and `mask` are not provided."


# Denoise
class StableDiffusionXLAutoDenoiseStep(AutoPipelineBlocks):
    block_classes = [StableDiffusionXLControlNetUnionDenoiseStep, StableDiffusionXLControlNetDenoiseStep, StableDiffusionXLDenoiseStep]
    block_names = ["controlnet_union", "controlnet", "unet"]
    block_trigger_inputs = ["control_mode", "control_image", None]

    @property
    def description(self):
        return "Denoise step that denoise the latents.\n" + \
               "This is an auto pipeline block that works for controlnet, controlnet_union and no controlnet.\n" + \
               " - `StableDiffusionXLControlNetUnionDenoiseStep` (controlnet_union) is used when both `control_mode` and `control_image` are provided.\n" + \
               " - `StableDiffusionXLControlNetDenoiseStep` (controlnet) is used when `control_image` is provided.\n" + \
               " - `StableDiffusionXLDenoiseStep` (unet only) is used when both `control_mode` and `control_image` are not provided."

# After denoise

class StableDiffusionXLDecodeStep(SequentialPipelineBlocks):
    block_classes = [StableDiffusionXLDecodeLatentsStep, StableDiffusionXLOutputStep]
    block_names = ["decode", "output"]

    @property
    def description(self):
        return "Decode step that decode the denoised latents into images outputs.\n" + \
               "This is a sequential pipeline blocks:\n" + \
               " - `StableDiffusionXLDecodeLatentsStep` is used to decode the denoised latents into images\n" + \
               " - `StableDiffusionXLOutputStep` is used to return a [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] or a plain tuple."


class StableDiffusionXLInpaintDecodeStep(SequentialPipelineBlocks):
    block_classes = [StableDiffusionXLDecodeLatentsStep, StableDiffusionXLInpaintOverlayMaskStep, StableDiffusionXLOutputStep]
    block_names = ["decode", "mask_overlay", "output"]

    @property
    def description(self):
        return "Inpaint decode step that decode the denoised latents into images outputs.\n" + \
               "This is a sequential pipeline blocks:\n" + \
               " - `StableDiffusionXLDecodeLatentsStep` is used to decode the denoised latents into images\n" + \
               " - `StableDiffusionXLInpaintOverlayMaskStep` is used to overlay the mask on the image\n" + \
               " - `StableDiffusionXLOutputStep` is used to return a [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] or a plain tuple."



class StableDiffusionXLAutoDecodeStep(AutoPipelineBlocks):
    block_classes = [StableDiffusionXLInpaintDecodeStep, StableDiffusionXLDecodeStep]
    block_names = ["inpaint", "non-inpaint"]
    block_trigger_inputs = ["padding_mask_crop", None]

    @property
    def description(self):
        return "Decode step that decode the denoised latents into images outputs.\n" + \
               "This is an auto pipeline block that works for inpainting and non-inpainting tasks.\n" + \
               " - `StableDiffusionXLInpaintDecodeStep` (inpaint) is used when `padding_mask_crop` is provided.\n" + \
               " - `StableDiffusionXLDecodeStep` (non-inpaint) is used when `padding_mask_crop` is not provided."


# block mapping 
TEXT2IMAGE_BLOCKS = OrderedDict([
    ("text_encoder", StableDiffusionXLTextEncoderStep),
    ("input", StableDiffusionXLInputStep),
    ("set_timesteps", StableDiffusionXLSetTimestepsStep),
    ("prepare_latents", StableDiffusionXLPrepareLatentsStep),
    ("prepare_add_cond", StableDiffusionXLPrepareAdditionalConditioningStep),
    ("denoise", StableDiffusionXLDenoiseStep),
    ("decode", StableDiffusionXLDecodeStep)
])

IMAGE2IMAGE_BLOCKS = OrderedDict([
    ("text_encoder", StableDiffusionXLTextEncoderStep),
    ("image_encoder", StableDiffusionXLVaeEncoderStep),
    ("input", StableDiffusionXLInputStep),
    ("set_timesteps", StableDiffusionXLImg2ImgSetTimestepsStep),
    ("prepare_latents", StableDiffusionXLImg2ImgPrepareLatentsStep),
    ("prepare_add_cond", StableDiffusionXLImg2ImgPrepareAdditionalConditioningStep),
    ("denoise", StableDiffusionXLDenoiseStep),
    ("decode", StableDiffusionXLDecodeStep)
])

INPAINT_BLOCKS = OrderedDict([
    ("text_encoder", StableDiffusionXLTextEncoderStep),
    ("image_encoder", StableDiffusionXLInpaintVaeEncoderStep),
    ("input", StableDiffusionXLInputStep),
    ("set_timesteps", StableDiffusionXLImg2ImgSetTimestepsStep),
    ("prepare_latents", StableDiffusionXLInpaintPrepareLatentsStep),
    ("prepare_add_cond", StableDiffusionXLImg2ImgPrepareAdditionalConditioningStep),
    ("denoise", StableDiffusionXLDenoiseStep),
    ("decode", StableDiffusionXLInpaintDecodeStep)
])

CONTROLNET_BLOCKS = OrderedDict([
    ("denoise", StableDiffusionXLControlNetDenoiseStep),
])

CONTROLNET_UNION_BLOCKS = OrderedDict([
    ("denoise", StableDiffusionXLControlNetUnionDenoiseStep),
])

AUTO_BLOCKS = OrderedDict([
    ("text_encoder", StableDiffusionXLTextEncoderStep),
    ("image_encoder", StableDiffusionXLAutoVaeEncoderStep),
    ("before_denoise", StableDiffusionXLAutoBeforeDenoiseStep),
    ("denoise", StableDiffusionXLAutoDenoiseStep),
    ("decode", StableDiffusionXLAutoDecodeStep)
])

AUTO_CORE_BLOCKS = OrderedDict([
    ("before_denoise", StableDiffusionXLAutoBeforeDenoiseStep),
    ("denoise", StableDiffusionXLAutoDenoiseStep),
])


SDXL_SUPPORTED_BLOCKS = {
    "text2img": TEXT2IMAGE_BLOCKS,
    "img2img": IMAGE2IMAGE_BLOCKS,
    "inpaint": INPAINT_BLOCKS,
    "controlnet": CONTROLNET_BLOCKS,
    "controlnet_union": CONTROLNET_UNION_BLOCKS,
    "auto": AUTO_BLOCKS
}


class StableDiffusionXLModularPipeline(
    ModularPipeline,
    StableDiffusionMixin,
    TextualInversionLoaderMixin,
    StableDiffusionXLLoraLoaderMixin,
    ModularIPAdapterMixin,
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
    def num_channels_unet(self):
        num_channels_unet = 4
        if hasattr(self, "unet") and self.unet is not None:
            num_channels_unet = self.unet.config.in_channels
        return num_channels_unet

    @property
    def num_channels_latents(self):
        num_channels_latents = 4
        if hasattr(self, "vae") and self.vae is not None:
            num_channels_latents = self.vae.config.latent_channels
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
    # 1. return image without apply any guidance
    # 2. add crops_coords and resize_mode to preprocess()
    def prepare_control_image(
        self,
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
            image = self.control_image_processor.preprocess(image, height=height, width=width, crops_coords=crops_coords, resize_mode="fill").to(dtype=torch.float32)
        else:
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
    # YiYi TODO: refactor using _encode_vae_image
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

    # Copied from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_inpaint.StableDiffusionXLInpaintPipeline.prepare_latents
    def prepare_latents_inpaint(
        self,
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
        return_noise=False,
        return_image_latents=False,
    ):
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

        if (image is None or timestep is None) and not is_strength_max:
            raise ValueError(
                "Since strength < 1. initial latents are to be initialised as a combination of Image + Noise."
                "However, either the image or the noise timestep has not been provided."
            )

        if image.shape[1] == 4:
            image_latents = image.to(device=device, dtype=dtype)
            image_latents = image_latents.repeat(batch_size // image_latents.shape[0], 1, 1, 1)
        elif return_image_latents or (latents is None and not is_strength_max):
            image = image.to(device=device, dtype=dtype)
            image_latents = self._encode_vae_image(image=image, generator=generator)
            image_latents = image_latents.repeat(batch_size // image_latents.shape[0], 1, 1, 1)

        if latents is None and add_noise:
            noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            # if strength is 1. then initialise the latents to noise, else initial to image + noise
            latents = noise if is_strength_max else self.scheduler.add_noise(image_latents, noise, timestep)
            # if pure noise then scale the initial latents by the  Scheduler's init sigma
            latents = latents * self.scheduler.init_noise_sigma if is_strength_max else latents
        elif add_noise:
            noise = latents.to(device)
            latents = noise * self.scheduler.init_noise_sigma
        else:
            noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            latents = image_latents.to(device)

        outputs = (latents,)

        if return_noise:
            outputs += (noise,)

        if return_image_latents:
            outputs += (image_latents,)

        return outputs

    
    # Modified from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_inpaint.StableDiffusionXLInpaintPipeline._encode_vae_image
    # YiYi TODO: update the _encode_vae_image so that we can use #Coped from
    def _encode_vae_image(self, image: torch.Tensor, generator: torch.Generator):
        
        latents_mean = latents_std = None
        if hasattr(self.vae.config, "latents_mean") and self.vae.config.latents_mean is not None:
            latents_mean = torch.tensor(self.vae.config.latents_mean).view(1, 4, 1, 1)
        if hasattr(self.vae.config, "latents_std") and self.vae.config.latents_std is not None:
            latents_std = torch.tensor(self.vae.config.latents_std).view(1, 4, 1, 1)
        
        dtype = image.dtype
        if self.vae.config.force_upcast:
            image = image.float()
            self.vae.to(dtype=torch.float32)

        if isinstance(generator, list):
            image_latents = [
                retrieve_latents(self.vae.encode(image[i : i + 1]), generator=generator[i])
                for i in range(image.shape[0])
            ]
            image_latents = torch.cat(image_latents, dim=0)
        else:
            image_latents = retrieve_latents(self.vae.encode(image), generator=generator)

        if self.vae.config.force_upcast:
            self.vae.to(dtype)

        image_latents = image_latents.to(dtype)
        if latents_mean is not None and latents_std is not None:
            latents_mean = latents_mean.to(device=image_latents.device, dtype=dtype)
            latents_std = latents_std.to(device=image_latents.device, dtype=dtype)
            image_latents = (image_latents - latents_mean) * self.vae.config.scaling_factor / latents_std
        else:
            image_latents = self.vae.config.scaling_factor * image_latents

        return image_latents 
    

    # modified from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_inpaint.StableDiffusionXLInpaintPipeline.prepare_mask_latents
    # do not accept do_classifier_free_guidance
    def prepare_mask_latents(
        self, mask, masked_image, batch_size, height, width, dtype, device, generator
    ):
        # resize the mask to latents shape as we concatenate the mask to the latents
        # we do that before converting to dtype to avoid breaking in case we're using cpu_offload
        # and half precision
        mask = torch.nn.functional.interpolate(
            mask, size=(height // self.vae_scale_factor, width // self.vae_scale_factor)
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
                masked_image_latents = self._encode_vae_image(masked_image, generator=generator)

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
