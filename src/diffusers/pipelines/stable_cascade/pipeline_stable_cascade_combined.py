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
from typing import Callable, Dict, List, Optional, Union

import PIL
import torch
from transformers import CLIPImageProcessor, CLIPTextModelWithProjection, CLIPTokenizer, CLIPVisionModelWithProjection

from ...models import StableCascadeUNet
from ...schedulers import DDPMWuerstchenScheduler
from ...utils import is_torch_version, replace_example_docstring
from ..pipeline_utils import DeprecatedPipelineMixin, DiffusionPipeline
from ..wuerstchen.modeling_paella_vq_model import PaellaVQModel
from .pipeline_stable_cascade import StableCascadeDecoderPipeline
from .pipeline_stable_cascade_prior import StableCascadePriorPipeline


TEXT2IMAGE_EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import StableCascadeCombinedPipeline

        >>> pipe = StableCascadeCombinedPipeline.from_pretrained(
        ...     "stabilityai/stable-cascade", variant="bf16", torch_dtype=torch.bfloat16
        ... )
        >>> pipe.enable_model_cpu_offload()
        >>> prompt = "an image of a shiba inu, donning a spacesuit and helmet"
        >>> images = pipe(prompt=prompt)
        ```
"""


class StableCascadeCombinedPipeline(DeprecatedPipelineMixin, DiffusionPipeline):
    """
    Combined Pipeline for text-to-image generation using Stable Cascade.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        tokenizer (`CLIPTokenizer`):
            The decoder tokenizer to be used for text inputs.
        text_encoder (`CLIPTextModelWithProjection`):
            The decoder text encoder to be used for text inputs.
        decoder (`StableCascadeUNet`):
            The decoder model to be used for decoder image generation pipeline.
        scheduler (`DDPMWuerstchenScheduler`):
            The scheduler to be used for decoder image generation pipeline.
        vqgan (`PaellaVQModel`):
            The VQGAN model to be used for decoder image generation pipeline.
        prior_prior (`StableCascadeUNet`):
            The prior model to be used for prior pipeline.
        prior_text_encoder (`CLIPTextModelWithProjection`):
            The prior text encoder to be used for text inputs.
        prior_tokenizer (`CLIPTokenizer`):
            The prior tokenizer to be used for text inputs.
        prior_scheduler (`DDPMWuerstchenScheduler`):
            The scheduler to be used for prior pipeline.
        prior_feature_extractor ([`~transformers.CLIPImageProcessor`]):
            Model that extracts features from generated images to be used as inputs for the `image_encoder`.
        prior_image_encoder ([`CLIPVisionModelWithProjection`]):
            Frozen CLIP image-encoder ([clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)).
    """

    _last_supported_version = "0.35.2"

    _load_connected_pipes = True
    _optional_components = ["prior_feature_extractor", "prior_image_encoder"]

    def __init__(
        self,
        tokenizer: CLIPTokenizer,
        text_encoder: CLIPTextModelWithProjection,
        decoder: StableCascadeUNet,
        scheduler: DDPMWuerstchenScheduler,
        vqgan: PaellaVQModel,
        prior_prior: StableCascadeUNet,
        prior_text_encoder: CLIPTextModelWithProjection,
        prior_tokenizer: CLIPTokenizer,
        prior_scheduler: DDPMWuerstchenScheduler,
        prior_feature_extractor: Optional[CLIPImageProcessor] = None,
        prior_image_encoder: Optional[CLIPVisionModelWithProjection] = None,
    ):
        super().__init__()

        self.register_modules(
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            decoder=decoder,
            scheduler=scheduler,
            vqgan=vqgan,
            prior_text_encoder=prior_text_encoder,
            prior_tokenizer=prior_tokenizer,
            prior_prior=prior_prior,
            prior_scheduler=prior_scheduler,
            prior_feature_extractor=prior_feature_extractor,
            prior_image_encoder=prior_image_encoder,
        )
        self.prior_pipe = StableCascadePriorPipeline(
            prior=prior_prior,
            text_encoder=prior_text_encoder,
            tokenizer=prior_tokenizer,
            scheduler=prior_scheduler,
            image_encoder=prior_image_encoder,
            feature_extractor=prior_feature_extractor,
        )
        self.decoder_pipe = StableCascadeDecoderPipeline(
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            decoder=decoder,
            scheduler=scheduler,
            vqgan=vqgan,
        )

    def enable_xformers_memory_efficient_attention(self, attention_op: Optional[Callable] = None):
        self.decoder_pipe.enable_xformers_memory_efficient_attention(attention_op)

    def enable_model_cpu_offload(self, gpu_id: Optional[int] = None, device: Union[torch.device, str] = None):
        r"""
        Offloads all models to CPU using accelerate, reducing memory usage with a low impact on performance. Compared
        to `enable_sequential_cpu_offload`, this method moves one whole model at a time to the GPU when its `forward`
        method is called, and the model remains in GPU until the next model runs. Memory savings are lower than with
        `enable_sequential_cpu_offload`, but performance is much better due to the iterative execution of the `unet`.
        """
        self.prior_pipe.enable_model_cpu_offload(gpu_id=gpu_id, device=device)
        self.decoder_pipe.enable_model_cpu_offload(gpu_id=gpu_id, device=device)

    def enable_sequential_cpu_offload(self, gpu_id: Optional[int] = None, device: Union[torch.device, str] = None):
        r"""
        Offloads all models (`unet`, `text_encoder`, `vae`, and `safety checker` state dicts) to CPU using 🤗
        Accelerate, significantly reducing memory usage. Models are moved to a `torch.device('meta')` and loaded on a
        GPU only when their specific submodule's `forward` method is called. Offloading happens on a submodule basis.
        Memory savings are higher than using `enable_model_cpu_offload`, but performance is lower.
        """
        self.prior_pipe.enable_sequential_cpu_offload(gpu_id=gpu_id, device=device)
        self.decoder_pipe.enable_sequential_cpu_offload(gpu_id=gpu_id, device=device)

    def progress_bar(self, iterable=None, total=None):
        self.prior_pipe.progress_bar(iterable=iterable, total=total)
        self.decoder_pipe.progress_bar(iterable=iterable, total=total)

    def set_progress_bar_config(self, **kwargs):
        self.prior_pipe.set_progress_bar_config(**kwargs)
        self.decoder_pipe.set_progress_bar_config(**kwargs)

    @torch.no_grad()
    @replace_example_docstring(TEXT2IMAGE_EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        images: Union[torch.Tensor, PIL.Image.Image, List[torch.Tensor], List[PIL.Image.Image]] = None,
        height: int = 512,
        width: int = 512,
        prior_num_inference_steps: int = 60,
        prior_guidance_scale: float = 4.0,
        num_inference_steps: int = 12,
        decoder_guidance_scale: float = 0.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_embeds_pooled: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds_pooled: Optional[torch.Tensor] = None,
        num_images_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        prior_callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        prior_callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    ):
        """
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide the image generation for the prior and decoder.
            images (`torch.Tensor`, `PIL.Image.Image`, `List[torch.Tensor]`, `List[PIL.Image.Image]`, *optional*):
                The images to guide the image generation for the prior.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings for the prior. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, text embeddings will be generated from `prompt` input argument.
            prompt_embeds_pooled (`torch.Tensor`, *optional*):
                Pre-generated text embeddings for the prior. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings for the prior. Can be used to easily tweak text inputs, *e.g.*
                prompt weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt`
                input argument.
            negative_prompt_embeds_pooled (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings for the prior. Can be used to easily tweak text inputs, *e.g.*
                prompt weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt`
                input argument.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            height (`int`, *optional*, defaults to 512):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to 512):
                The width in pixels of the generated image.
            prior_guidance_scale (`float`, *optional*, defaults to 4.0):
                Guidance scale as defined in [Classifier-Free Diffusion
                Guidance](https://huggingface.co/papers/2207.12598). `prior_guidance_scale` is defined as `w` of
                equation 2. of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by
                setting `prior_guidance_scale > 1`. Higher guidance scale encourages to generate images that are
                closely linked to the text `prompt`, usually at the expense of lower image quality.
            prior_num_inference_steps (`Union[int, Dict[float, int]]`, *optional*, defaults to 60):
                The number of prior denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference. For more specific timestep spacing, you can pass customized
                `prior_timesteps`
            num_inference_steps (`int`, *optional*, defaults to 12):
                The number of decoder denoising steps. More denoising steps usually lead to a higher quality image at
                the expense of slower inference. For more specific timestep spacing, you can pass customized
                `timesteps`
            decoder_guidance_scale (`float`, *optional*, defaults to 0.0):
                Guidance scale as defined in [Classifier-Free Diffusion
                Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2.
                of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting
                `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to
                the text `prompt`, usually at the expense of lower image quality.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will be generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between: `"pil"` (`PIL.Image.Image`), `"np"`
                (`np.array`) or `"pt"` (`torch.Tensor`).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.
            prior_callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `prior_callback_on_step_end(self: DiffusionPipeline, step: int, timestep:
                int, callback_kwargs: Dict)`.
            prior_callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `prior_callback_on_step_end` function. The tensors specified in the
                list will be passed as `callback_kwargs` argument. You will only be able to include variables listed in
                the `._callback_tensor_inputs` attribute of your pipeline class.
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.

        Examples:

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple` [`~pipelines.ImagePipelineOutput`] if `return_dict` is True,
            otherwise a `tuple`. When returning a tuple, the first element is a list with the generated images.
        """
        dtype = self.decoder_pipe.decoder.dtype
        if is_torch_version("<", "2.2.0") and dtype == torch.bfloat16:
            raise ValueError(
                "`StableCascadeCombinedPipeline` requires torch>=2.2.0 when using `torch.bfloat16` dtype."
            )

        prior_outputs = self.prior_pipe(
            prompt=prompt if prompt_embeds is None else None,
            images=images,
            height=height,
            width=width,
            num_inference_steps=prior_num_inference_steps,
            guidance_scale=prior_guidance_scale,
            negative_prompt=negative_prompt if negative_prompt_embeds is None else None,
            prompt_embeds=prompt_embeds,
            prompt_embeds_pooled=prompt_embeds_pooled,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_prompt_embeds_pooled=negative_prompt_embeds_pooled,
            num_images_per_prompt=num_images_per_prompt,
            generator=generator,
            latents=latents,
            output_type="pt",
            return_dict=True,
            callback_on_step_end=prior_callback_on_step_end,
            callback_on_step_end_tensor_inputs=prior_callback_on_step_end_tensor_inputs,
        )
        image_embeddings = prior_outputs.image_embeddings
        prompt_embeds = prior_outputs.get("prompt_embeds", None)
        prompt_embeds_pooled = prior_outputs.get("prompt_embeds_pooled", None)
        negative_prompt_embeds = prior_outputs.get("negative_prompt_embeds", None)
        negative_prompt_embeds_pooled = prior_outputs.get("negative_prompt_embeds_pooled", None)

        outputs = self.decoder_pipe(
            image_embeddings=image_embeddings,
            prompt=prompt if prompt_embeds is None else None,
            num_inference_steps=num_inference_steps,
            guidance_scale=decoder_guidance_scale,
            negative_prompt=negative_prompt if negative_prompt_embeds is None else None,
            prompt_embeds=prompt_embeds,
            prompt_embeds_pooled=prompt_embeds_pooled,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_prompt_embeds_pooled=negative_prompt_embeds_pooled,
            generator=generator,
            output_type=output_type,
            return_dict=return_dict,
            callback_on_step_end=callback_on_step_end,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        )

        return outputs
