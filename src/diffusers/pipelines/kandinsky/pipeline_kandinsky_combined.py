# Copyright 2023 The HuggingFace Team. All rights reserved.
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
from typing import Callable, List, Optional, Union

import PIL.Image
import torch
from transformers import (
    CLIPImageProcessor,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
    XLMRobertaTokenizer,
)

from ...models import PriorTransformer, UNet2DConditionModel, VQModel
from ...schedulers import DDIMScheduler, DDPMScheduler, UnCLIPScheduler
from ...utils import (
    replace_example_docstring,
)
from ..pipeline_utils import DiffusionPipeline
from .pipeline_kandinsky import KandinskyPipeline
from .pipeline_kandinsky_img2img import KandinskyImg2ImgPipeline
from .pipeline_kandinsky_inpaint import KandinskyInpaintPipeline
from .pipeline_kandinsky_prior import KandinskyPriorPipeline
from .text_encoder import MultilingualCLIP


TEXT2IMAGE_EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        from diffusers import AutoPipelineForText2Image
        import torch

        pipe = AutoPipelineForText2Image.from_pretrained(
            "kandinsky-community/kandinsky-2-1", torch_dtype=torch.float16
        )
        pipe.enable_model_cpu_offload()

        prompt = "A lion in galaxies, spirals, nebulae, stars, smoke, iridescent, intricate detail, octane render, 8k"

        image = pipe(prompt=prompt, num_inference_steps=25).images[0]
        ```
"""

IMAGE2IMAGE_EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        from diffusers import AutoPipelineForImage2Image
        import torch
        import requests
        from io import BytesIO
        from PIL import Image
        import os

        pipe = AutoPipelineForImage2Image.from_pretrained(
            "kandinsky-community/kandinsky-2-1", torch_dtype=torch.float16
        )
        pipe.enable_model_cpu_offload()

        prompt = "A fantasy landscape, Cinematic lighting"
        negative_prompt = "low quality, bad quality"

        url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"

        response = requests.get(url)
        image = Image.open(BytesIO(response.content)).convert("RGB")
        image.thumbnail((768, 768))

        image = pipe(prompt=prompt, image=original_image, num_inference_steps=25).images[0]
        ```
"""

INPAINT_EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        from diffusers import AutoPipelineForInpainting
        from diffusers.utils import load_image
        import torch
        import numpy as np

        pipe = AutoPipelineForInpainting.from_pretrained(
            "kandinsky-community/kandinsky-2-1-inpaint", torch_dtype=torch.float16
        )
        pipe.enable_model_cpu_offload()

        prompt = "A fantasy landscape, Cinematic lighting"
        negative_prompt = "low quality, bad quality"

        original_image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main" "/kandinsky/cat.png"
        )

        mask = np.zeros((768, 768), dtype=np.float32)
        # Let's mask out an area above the cat's head
        mask[:250, 250:-250] = 1

        image = pipe(prompt=prompt, image=original_image, mask_image=mask, num_inference_steps=25).images[0]
        ```
"""


class KandinskyCombinedPipeline(DiffusionPipeline):
    """
    Combined Pipeline for text-to-image generation using Kandinsky

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        text_encoder ([`MultilingualCLIP`]):
            Frozen text-encoder.
        tokenizer ([`XLMRobertaTokenizer`]):
            Tokenizer of class
        scheduler (Union[`DDIMScheduler`,`DDPMScheduler`]):
            A scheduler to be used in combination with `unet` to generate image latents.
        unet ([`UNet2DConditionModel`]):
            Conditional U-Net architecture to denoise the image embedding.
        movq ([`VQModel`]):
            MoVQ Decoder to generate the image from the latents.
        prior_prior ([`PriorTransformer`]):
            The canonincal unCLIP prior to approximate the image embedding from the text embedding.
        prior_image_encoder ([`CLIPVisionModelWithProjection`]):
            Frozen image-encoder.
        prior_text_encoder ([`CLIPTextModelWithProjection`]):
            Frozen text-encoder.
        prior_tokenizer (`CLIPTokenizer`):
             Tokenizer of class
             [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        prior_scheduler ([`UnCLIPScheduler`]):
            A scheduler to be used in combination with `prior` to generate image embedding.
    """

    _load_connected_pipes = True
    model_cpu_offload_seq = "text_encoder->unet->movq->prior_prior->prior_image_encoder->prior_text_encoder"

    def __init__(
        self,
        text_encoder: MultilingualCLIP,
        tokenizer: XLMRobertaTokenizer,
        unet: UNet2DConditionModel,
        scheduler: Union[DDIMScheduler, DDPMScheduler],
        movq: VQModel,
        prior_prior: PriorTransformer,
        prior_image_encoder: CLIPVisionModelWithProjection,
        prior_text_encoder: CLIPTextModelWithProjection,
        prior_tokenizer: CLIPTokenizer,
        prior_scheduler: UnCLIPScheduler,
        prior_image_processor: CLIPImageProcessor,
    ):
        super().__init__()

        self.register_modules(
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            movq=movq,
            prior_prior=prior_prior,
            prior_image_encoder=prior_image_encoder,
            prior_text_encoder=prior_text_encoder,
            prior_tokenizer=prior_tokenizer,
            prior_scheduler=prior_scheduler,
            prior_image_processor=prior_image_processor,
        )
        self.prior_pipe = KandinskyPriorPipeline(
            prior=prior_prior,
            image_encoder=prior_image_encoder,
            text_encoder=prior_text_encoder,
            tokenizer=prior_tokenizer,
            scheduler=prior_scheduler,
            image_processor=prior_image_processor,
        )
        self.decoder_pipe = KandinskyPipeline(
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            movq=movq,
        )

    def enable_xformers_memory_efficient_attention(self, attention_op: Optional[Callable] = None):
        self.decoder_pipe.enable_xformers_memory_efficient_attention(attention_op)

    def enable_sequential_cpu_offload(self, gpu_id=0):
        r"""
        Offloads all models (`unet`, `text_encoder`, `vae`, and `safety checker` state dicts) to CPU using 🤗
        Accelerate, significantly reducing memory usage. Models are moved to a `torch.device('meta')` and loaded on a
        GPU only when their specific submodule's `forward` method is called. Offloading happens on a submodule basis.
        Memory savings are higher than using `enable_model_cpu_offload`, but performance is lower.
        """
        self.prior_pipe.enable_sequential_cpu_offload(gpu_id=gpu_id)
        self.decoder_pipe.enable_sequential_cpu_offload(gpu_id=gpu_id)

    def progress_bar(self, iterable=None, total=None):
        self.prior_pipe.progress_bar(iterable=iterable, total=total)
        self.decoder_pipe.progress_bar(iterable=iterable, total=total)
        self.decoder_pipe.enable_model_cpu_offload()

    def set_progress_bar_config(self, **kwargs):
        self.prior_pipe.set_progress_bar_config(**kwargs)
        self.decoder_pipe.set_progress_bar_config(**kwargs)

    @torch.no_grad()
    @replace_example_docstring(TEXT2IMAGE_EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_inference_steps: int = 100,
        guidance_scale: float = 4.0,
        num_images_per_prompt: int = 1,
        height: int = 512,
        width: int = 512,
        prior_guidance_scale: float = 4.0,
        prior_num_inference_steps: int = 25,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        return_dict: bool = True,
    ):
        """
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide the image generation.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            num_inference_steps (`int`, *optional*, defaults to 100):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            height (`int`, *optional*, defaults to 512):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to 512):
                The width in pixels of the generated image.
            prior_guidance_scale (`float`, *optional*, defaults to 4.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            prior_num_inference_steps (`int`, *optional*, defaults to 100):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 4.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between: `"pil"` (`PIL.Image.Image`), `"np"`
                (`np.array`) or `"pt"` (`torch.Tensor`).
            callback (`Callable`, *optional*):
                A function that calls every `callback_steps` steps during inference. The function is called with the
                following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function is called. If not specified, the callback is called at
                every step.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.

        Examples:

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`
        """
        prior_outputs = self.prior_pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt,
            num_inference_steps=prior_num_inference_steps,
            generator=generator,
            latents=latents,
            guidance_scale=prior_guidance_scale,
            output_type="pt",
            return_dict=False,
        )
        image_embeds = prior_outputs[0]
        negative_image_embeds = prior_outputs[1]

        prompt = [prompt] if not isinstance(prompt, (list, tuple)) else prompt

        if len(prompt) < image_embeds.shape[0] and image_embeds.shape[0] % len(prompt) == 0:
            prompt = (image_embeds.shape[0] // len(prompt)) * prompt

        outputs = self.decoder_pipe(
            prompt=prompt,
            image_embeds=image_embeds,
            negative_image_embeds=negative_image_embeds,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            generator=generator,
            guidance_scale=guidance_scale,
            output_type=output_type,
            callback=callback,
            callback_steps=callback_steps,
            return_dict=return_dict,
        )
        return outputs


class KandinskyImg2ImgCombinedPipeline(DiffusionPipeline):
    """
    Combined Pipeline for image-to-image generation using Kandinsky

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        text_encoder ([`MultilingualCLIP`]):
            Frozen text-encoder.
        tokenizer ([`XLMRobertaTokenizer`]):
            Tokenizer of class
        scheduler (Union[`DDIMScheduler`,`DDPMScheduler`]):
            A scheduler to be used in combination with `unet` to generate image latents.
        unet ([`UNet2DConditionModel`]):
            Conditional U-Net architecture to denoise the image embedding.
        movq ([`VQModel`]):
            MoVQ Decoder to generate the image from the latents.
        prior_prior ([`PriorTransformer`]):
            The canonincal unCLIP prior to approximate the image embedding from the text embedding.
        prior_image_encoder ([`CLIPVisionModelWithProjection`]):
            Frozen image-encoder.
        prior_text_encoder ([`CLIPTextModelWithProjection`]):
            Frozen text-encoder.
        prior_tokenizer (`CLIPTokenizer`):
             Tokenizer of class
             [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        prior_scheduler ([`UnCLIPScheduler`]):
            A scheduler to be used in combination with `prior` to generate image embedding.
    """

    _load_connected_pipes = True
    model_cpu_offload_seq = "prior_text_encoder->prior_image_encoder->prior_prior->" "text_encoder->unet->movq"

    def __init__(
        self,
        text_encoder: MultilingualCLIP,
        tokenizer: XLMRobertaTokenizer,
        unet: UNet2DConditionModel,
        scheduler: Union[DDIMScheduler, DDPMScheduler],
        movq: VQModel,
        prior_prior: PriorTransformer,
        prior_image_encoder: CLIPVisionModelWithProjection,
        prior_text_encoder: CLIPTextModelWithProjection,
        prior_tokenizer: CLIPTokenizer,
        prior_scheduler: UnCLIPScheduler,
        prior_image_processor: CLIPImageProcessor,
    ):
        super().__init__()

        self.register_modules(
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            movq=movq,
            prior_prior=prior_prior,
            prior_image_encoder=prior_image_encoder,
            prior_text_encoder=prior_text_encoder,
            prior_tokenizer=prior_tokenizer,
            prior_scheduler=prior_scheduler,
            prior_image_processor=prior_image_processor,
        )
        self.prior_pipe = KandinskyPriorPipeline(
            prior=prior_prior,
            image_encoder=prior_image_encoder,
            text_encoder=prior_text_encoder,
            tokenizer=prior_tokenizer,
            scheduler=prior_scheduler,
            image_processor=prior_image_processor,
        )
        self.decoder_pipe = KandinskyImg2ImgPipeline(
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            movq=movq,
        )

    def enable_xformers_memory_efficient_attention(self, attention_op: Optional[Callable] = None):
        self.decoder_pipe.enable_xformers_memory_efficient_attention(attention_op)

    def enable_sequential_cpu_offload(self, gpu_id=0):
        r"""
        Offloads all models to CPU using accelerate, significantly reducing memory usage. When called, unet,
        text_encoder, vae and safety checker have their state dicts saved to CPU and then are moved to a
        `torch.device('meta') and loaded to GPU only when their specific submodule has its `forward` method called.
        Note that offloading happens on a submodule basis. Memory savings are higher than with
        `enable_model_cpu_offload`, but performance is lower.
        """
        self.prior_pipe.enable_sequential_cpu_offload(gpu_id=gpu_id)
        self.decoder_pipe.enable_sequential_cpu_offload(gpu_id=gpu_id)

    def progress_bar(self, iterable=None, total=None):
        self.prior_pipe.progress_bar(iterable=iterable, total=total)
        self.decoder_pipe.progress_bar(iterable=iterable, total=total)
        self.decoder_pipe.enable_model_cpu_offload()

    def set_progress_bar_config(self, **kwargs):
        self.prior_pipe.set_progress_bar_config(**kwargs)
        self.decoder_pipe.set_progress_bar_config(**kwargs)

    @torch.no_grad()
    @replace_example_docstring(IMAGE2IMAGE_EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]],
        image: Union[torch.FloatTensor, PIL.Image.Image, List[torch.FloatTensor], List[PIL.Image.Image]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_inference_steps: int = 100,
        guidance_scale: float = 4.0,
        num_images_per_prompt: int = 1,
        strength: float = 0.3,
        height: int = 512,
        width: int = 512,
        prior_guidance_scale: float = 4.0,
        prior_num_inference_steps: int = 25,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        return_dict: bool = True,
    ):
        """
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide the image generation.
            image (`torch.FloatTensor`, `PIL.Image.Image`, `np.ndarray`, `List[torch.FloatTensor]`, `List[PIL.Image.Image]`, or `List[np.ndarray]`):
                `Image`, or tensor representing an image batch, that will be used as the starting point for the
                process. Can also accept image latents as `image`, if passing latents directly, it will not be encoded
                again.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            num_inference_steps (`int`, *optional*, defaults to 100):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            height (`int`, *optional*, defaults to 512):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to 512):
                The width in pixels of the generated image.
            strength (`float`, *optional*, defaults to 0.3):
                Conceptually, indicates how much to transform the reference `image`. Must be between 0 and 1. `image`
                will be used as a starting point, adding more noise to it the larger the `strength`. The number of
                denoising steps depends on the amount of noise initially added. When `strength` is 1, added noise will
                be maximum and the denoising process will run for the full number of iterations specified in
                `num_inference_steps`. A value of 1, therefore, essentially ignores `image`.
            prior_guidance_scale (`float`, *optional*, defaults to 4.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            prior_num_inference_steps (`int`, *optional*, defaults to 100):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 4.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between: `"pil"` (`PIL.Image.Image`), `"np"`
                (`np.array`) or `"pt"` (`torch.Tensor`).
            callback (`Callable`, *optional*):
                A function that calls every `callback_steps` steps during inference. The function is called with the
                following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function is called. If not specified, the callback is called at
                every step.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.

        Examples:

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`
        """
        prior_outputs = self.prior_pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt,
            num_inference_steps=prior_num_inference_steps,
            generator=generator,
            latents=latents,
            guidance_scale=prior_guidance_scale,
            output_type="pt",
            return_dict=False,
        )
        image_embeds = prior_outputs[0]
        negative_image_embeds = prior_outputs[1]

        prompt = [prompt] if not isinstance(prompt, (list, tuple)) else prompt
        image = [image] if isinstance(prompt, PIL.Image.Image) else image

        if len(prompt) < image_embeds.shape[0] and image_embeds.shape[0] % len(prompt) == 0:
            prompt = (image_embeds.shape[0] // len(prompt)) * prompt

        if (
            isinstance(image, (list, tuple))
            and len(image) < image_embeds.shape[0]
            and image_embeds.shape[0] % len(image) == 0
        ):
            image = (image_embeds.shape[0] // len(image)) * image

        outputs = self.decoder_pipe(
            prompt=prompt,
            image=image,
            image_embeds=image_embeds,
            negative_image_embeds=negative_image_embeds,
            strength=strength,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            generator=generator,
            guidance_scale=guidance_scale,
            output_type=output_type,
            callback=callback,
            callback_steps=callback_steps,
            return_dict=return_dict,
        )
        return outputs


class KandinskyInpaintCombinedPipeline(DiffusionPipeline):
    """
    Combined Pipeline for generation using Kandinsky

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        text_encoder ([`MultilingualCLIP`]):
            Frozen text-encoder.
        tokenizer ([`XLMRobertaTokenizer`]):
            Tokenizer of class
        scheduler (Union[`DDIMScheduler`,`DDPMScheduler`]):
            A scheduler to be used in combination with `unet` to generate image latents.
        unet ([`UNet2DConditionModel`]):
            Conditional U-Net architecture to denoise the image embedding.
        movq ([`VQModel`]):
            MoVQ Decoder to generate the image from the latents.
        prior_prior ([`PriorTransformer`]):
            The canonincal unCLIP prior to approximate the image embedding from the text embedding.
        prior_image_encoder ([`CLIPVisionModelWithProjection`]):
            Frozen image-encoder.
        prior_text_encoder ([`CLIPTextModelWithProjection`]):
            Frozen text-encoder.
        prior_tokenizer (`CLIPTokenizer`):
             Tokenizer of class
             [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        prior_scheduler ([`UnCLIPScheduler`]):
            A scheduler to be used in combination with `prior` to generate image embedding.
    """

    _load_connected_pipes = True
    model_cpu_offload_seq = "prior_text_encoder->prior_image_encoder->prior_prior->" "text_encoder->unet->movq"

    def __init__(
        self,
        text_encoder: MultilingualCLIP,
        tokenizer: XLMRobertaTokenizer,
        unet: UNet2DConditionModel,
        scheduler: Union[DDIMScheduler, DDPMScheduler],
        movq: VQModel,
        prior_prior: PriorTransformer,
        prior_image_encoder: CLIPVisionModelWithProjection,
        prior_text_encoder: CLIPTextModelWithProjection,
        prior_tokenizer: CLIPTokenizer,
        prior_scheduler: UnCLIPScheduler,
        prior_image_processor: CLIPImageProcessor,
    ):
        super().__init__()

        self.register_modules(
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            movq=movq,
            prior_prior=prior_prior,
            prior_image_encoder=prior_image_encoder,
            prior_text_encoder=prior_text_encoder,
            prior_tokenizer=prior_tokenizer,
            prior_scheduler=prior_scheduler,
            prior_image_processor=prior_image_processor,
        )
        self.prior_pipe = KandinskyPriorPipeline(
            prior=prior_prior,
            image_encoder=prior_image_encoder,
            text_encoder=prior_text_encoder,
            tokenizer=prior_tokenizer,
            scheduler=prior_scheduler,
            image_processor=prior_image_processor,
        )
        self.decoder_pipe = KandinskyInpaintPipeline(
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            movq=movq,
        )

    def enable_xformers_memory_efficient_attention(self, attention_op: Optional[Callable] = None):
        self.decoder_pipe.enable_xformers_memory_efficient_attention(attention_op)

    def enable_sequential_cpu_offload(self, gpu_id=0):
        r"""
        Offloads all models to CPU using accelerate, significantly reducing memory usage. When called, unet,
        text_encoder, vae and safety checker have their state dicts saved to CPU and then are moved to a
        `torch.device('meta') and loaded to GPU only when their specific submodule has its `forward` method called.
        Note that offloading happens on a submodule basis. Memory savings are higher than with
        `enable_model_cpu_offload`, but performance is lower.
        """
        self.prior_pipe.enable_sequential_cpu_offload(gpu_id=gpu_id)
        self.decoder_pipe.enable_sequential_cpu_offload(gpu_id=gpu_id)

    def progress_bar(self, iterable=None, total=None):
        self.prior_pipe.progress_bar(iterable=iterable, total=total)
        self.decoder_pipe.progress_bar(iterable=iterable, total=total)
        self.decoder_pipe.enable_model_cpu_offload()

    def set_progress_bar_config(self, **kwargs):
        self.prior_pipe.set_progress_bar_config(**kwargs)
        self.decoder_pipe.set_progress_bar_config(**kwargs)

    @torch.no_grad()
    @replace_example_docstring(INPAINT_EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]],
        image: Union[torch.FloatTensor, PIL.Image.Image, List[torch.FloatTensor], List[PIL.Image.Image]],
        mask_image: Union[torch.FloatTensor, PIL.Image.Image, List[torch.FloatTensor], List[PIL.Image.Image]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_inference_steps: int = 100,
        guidance_scale: float = 4.0,
        num_images_per_prompt: int = 1,
        height: int = 512,
        width: int = 512,
        prior_guidance_scale: float = 4.0,
        prior_num_inference_steps: int = 25,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        return_dict: bool = True,
    ):
        """
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide the image generation.
            image (`torch.FloatTensor`, `PIL.Image.Image`, `np.ndarray`, `List[torch.FloatTensor]`, `List[PIL.Image.Image]`, or `List[np.ndarray]`):
                `Image`, or tensor representing an image batch, that will be used as the starting point for the
                process. Can also accept image latents as `image`, if passing latents directly, it will not be encoded
                again.
            mask_image (`np.array`):
                Tensor representing an image batch, to mask `image`. White pixels in the mask will be repainted, while
                black pixels will be preserved. If `mask_image` is a PIL image, it will be converted to a single
                channel (luminance) before use. If it's a tensor, it should contain one color channel (L) instead of 3,
                so the expected shape would be `(B, H, W, 1)`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            num_inference_steps (`int`, *optional*, defaults to 100):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            height (`int`, *optional*, defaults to 512):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to 512):
                The width in pixels of the generated image.
            prior_guidance_scale (`float`, *optional*, defaults to 4.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            prior_num_inference_steps (`int`, *optional*, defaults to 100):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 4.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between: `"pil"` (`PIL.Image.Image`), `"np"`
                (`np.array`) or `"pt"` (`torch.Tensor`).
            callback (`Callable`, *optional*):
                A function that calls every `callback_steps` steps during inference. The function is called with the
                following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function is called. If not specified, the callback is called at
                every step.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.

        Examples:

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`
        """
        prior_outputs = self.prior_pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt,
            num_inference_steps=prior_num_inference_steps,
            generator=generator,
            latents=latents,
            guidance_scale=prior_guidance_scale,
            output_type="pt",
            return_dict=False,
        )
        image_embeds = prior_outputs[0]
        negative_image_embeds = prior_outputs[1]

        prompt = [prompt] if not isinstance(prompt, (list, tuple)) else prompt
        image = [image] if isinstance(prompt, PIL.Image.Image) else image
        mask_image = [mask_image] if isinstance(mask_image, PIL.Image.Image) else mask_image

        if len(prompt) < image_embeds.shape[0] and image_embeds.shape[0] % len(prompt) == 0:
            prompt = (image_embeds.shape[0] // len(prompt)) * prompt

        if (
            isinstance(image, (list, tuple))
            and len(image) < image_embeds.shape[0]
            and image_embeds.shape[0] % len(image) == 0
        ):
            image = (image_embeds.shape[0] // len(image)) * image

        if (
            isinstance(mask_image, (list, tuple))
            and len(mask_image) < image_embeds.shape[0]
            and image_embeds.shape[0] % len(mask_image) == 0
        ):
            mask_image = (image_embeds.shape[0] // len(mask_image)) * mask_image

        outputs = self.decoder_pipe(
            prompt=prompt,
            image=image,
            mask_image=mask_image,
            image_embeds=image_embeds,
            negative_image_embeds=negative_image_embeds,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            generator=generator,
            guidance_scale=guidance_scale,
            output_type=output_type,
            callback=callback,
            callback_steps=callback_steps,
            return_dict=return_dict,
        )
        return outputs
