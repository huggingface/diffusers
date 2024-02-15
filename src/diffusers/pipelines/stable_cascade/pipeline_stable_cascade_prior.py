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

from math import ceil
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import PIL
import torch
from transformers import CLIPImageProcessor, CLIPTextModelWithProjection, CLIPTokenizer, CLIPVisionModelWithProjection

from ...loaders import LoraLoaderMixin
from ...schedulers import DDPMWuerstchenScheduler
from ...utils import deprecate, logging, replace_example_docstring
from ...utils.torch_utils import randn_tensor
from ..pipeline_utils import DiffusionPipeline
from ..wuerstchen.pipeline_wuerstchen_prior import WuerstchenPriorPipelineOutput
from .modeling_stable_cascade_common import StableCascadeUnet


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

DEFAULT_STAGE_C_TIMESTEPS = list(np.linspace(1.0, 2 / 3, 20)) + list(np.linspace(2 / 3, 0.0, 11))[1:]

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import StableCascadePriorPipeline

        >>> prior_pipe = StableCascadePriorPipeline.from_pretrained(
        ...     "warp-ai/wuerstchen-v3-prior", torch_dtype=torch.float16
        ... ).to("cuda")

        >>> prompt = "an image of a shiba inu, donning a spacesuit and helmet"
        >>> prior_output = pipe(prompt)
        ```
"""


class StableCascadePriorPipeline(DiffusionPipeline, LoraLoaderMixin):
    """
    Pipeline for generating image prior for Stable Cascade.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        prior ([`Prior`]):
            The Stable Cascade prior to approximate the image embedding from the text and/or image embedding.
        text_encoder ([`CLIPTextModelWithProjection`]):
            Frozen text-encoder ([laion/CLIP-ViT-bigG-14-laion2B-39B-b160k](https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k)).
        feature_extractor ([`~transformers.CLIPImageProcessor`]):
            Model that extracts features from generated images to be used as inputs for the `image_encoder`.
        image_encoder ([`CLIPVisionModelWithProjection`]):
            Frozen CLIP image-encoder ([clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)).
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        scheduler ([`DDPMWuerstchenScheduler`]):
            A scheduler to be used in combination with `prior` to generate image embedding.
        resolution_multiple ('float', *optional*, defaults to 42.67):
            Default resolution for multiple images generated.
    """

    unet_name = "prior"
    text_encoder_name = "text_encoder"
    model_cpu_offload_seq = "image_encoder->text_encoder->prior"
    _optional_components = ["image_encoder", "feature_extractor"]
    _callback_tensor_inputs = ["latents", "text_encoder_hidden_states", "negative_prompt_embeds"]

    def __init__(
        self,
        tokenizer: CLIPTokenizer,
        text_encoder: CLIPTextModelWithProjection,
        prior: StableCascadeUnet,
        scheduler: DDPMWuerstchenScheduler,
        resolution_multiple: float = 42.67,
        feature_extractor: Optional[CLIPImageProcessor] = None,
        image_encoder: Optional[CLIPVisionModelWithProjection] = None,
    ) -> None:
        super().__init__()
        self.register_modules(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            image_encoder=image_encoder,
            feature_extractor=feature_extractor,
            prior=prior,
            scheduler=scheduler,
        )
        self.register_to_config(resolution_multiple=resolution_multiple)

    # Copied from diffusers.pipelines.unclip.pipeline_unclip.UnCLIPPipeline.prepare_latents
    def prepare_latents(self, shape, dtype, device, generator, latents, scheduler):
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
            latents = latents.to(device)

        latents = latents * scheduler.init_noise_sigma
        return latents

    def encode_prompt(
        self,
        device,
        batch_size,
        num_images_per_prompt,
        do_classifier_free_guidance,
        prompt=None,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        prompt_embeds_pooled: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds_pooled: Optional[torch.FloatTensor] = None,
    ):
        if prompt_embeds is None:
            # get prompt text embeddings
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            attention_mask = text_inputs.attention_mask

            untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
                )
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                )
                text_input_ids = text_input_ids[:, : self.tokenizer.model_max_length]
                attention_mask = attention_mask[:, : self.tokenizer.model_max_length]

            text_encoder_output = self.text_encoder(
                text_input_ids.to(device), attention_mask=attention_mask.to(device), output_hidden_states=True
            )
            prompt_embeds = text_encoder_output.hidden_states[-1]
            if prompt_embeds_pooled is None:
                prompt_embeds_pooled = text_encoder_output.text_embeds.unsqueeze(1)

        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)
        prompt_embeds_pooled = prompt_embeds_pooled.to(dtype=self.text_encoder.dtype, device=device)
        prompt_embeds = prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)
        prompt_embeds_pooled = prompt_embeds_pooled.repeat_interleave(num_images_per_prompt, dim=0)

        if negative_prompt_embeds is None and do_classifier_free_guidance:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            negative_prompt_embeds_text_encoder_output = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=uncond_input.attention_mask.to(device),
                output_hidden_states=True,
            )

            negative_prompt_embeds = negative_prompt_embeds_text_encoder_output.hidden_states[-1]
            negative_prompt_embeds_pooled = negative_prompt_embeds_text_encoder_output.text_embeds.unsqueeze(1)

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]
            negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)
            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

            seq_len = negative_prompt_embeds_pooled.shape[1]
            negative_prompt_embeds_pooled = negative_prompt_embeds_pooled.to(
                dtype=self.text_encoder.dtype, device=device
            )
            negative_prompt_embeds_pooled = negative_prompt_embeds_pooled.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds_pooled = negative_prompt_embeds_pooled.view(
                batch_size * num_images_per_prompt, seq_len, -1
            )
            # done duplicates

        return prompt_embeds, prompt_embeds_pooled, negative_prompt_embeds, negative_prompt_embeds_pooled

    def encode_image(self, images, device, dtype, batch_size, num_images_per_prompt):
        image_embeds = []
        for image in images:
            if isinstance(image, torch.FloatTensor):
                image = (image * 255).type(torch.uint8)
            image = self.feature_extractor(image, return_tensors="pt").pixel_values
            image = image.to(device=device, dtype=dtype)
            image_embed = self.image_encoder(image).image_embeds.unsqueeze(1)
            image_embeds.append(image_embed)
        image_embeds = torch.cat(image_embeds, dim=1)

        image_embeds = image_embeds.repeat(batch_size * num_images_per_prompt)
        negative_image_embeds = torch.zeros_like(image_embeds)

        return image_embeds, negative_image_embeds

    def check_inputs(
        self,
        prompt,
        images,
        negative_prompt,
        num_inference_steps,
        prompt_embeds=None,
        prompt_embeds_pooled=None,
        negative_prompt_embeds=None,
        negative_prompt_embeds_pooled=None,
    ):
        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

        if prompt_embeds_pooled is not None and negative_prompt_embeds_pooled is not None:
            if prompt_embeds_pooled.shape != negative_prompt_embeds_pooled.shape:
                raise ValueError(
                    "`prompt_embeds_pooled` and `negative_prompt_embeds_pooled` must have the same shape when passed"
                    f"directly, but got: `prompt_embeds_pooled` {prompt_embeds_pooled.shape} !="
                    f"`negative_prompt_embeds_pooled` {negative_prompt_embeds_pooled.shape}."
                )

        if not isinstance(num_inference_steps, int):
            raise TypeError(
                f"'num_inference_steps' must be of type 'int', but got {type(num_inference_steps)}\
                           In Case you want to provide explicit timesteps, please use the 'timesteps' argument."
            )

        if images:
            for i, image in enumerate(images):
                if not isinstance(image, torch.Tensor) and not isinstance(image, PIL.Image.Image):
                    raise TypeError(
                        f"'images' must contain images of type 'torch.Tensor' or 'PIL.Image.Image, but got"
                        f"{type(image)} for image number {i}."
                    )

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1

    @property
    def num_timesteps(self):
        return self._num_timesteps

    def get_t_condioning(self, t, alphas_cumprod):
        s = torch.tensor([0.003])
        clamp_range = [0, 1]
        min_var = torch.cos(s / (1 + s) * torch.pi * 0.5) ** 2
        var = alphas_cumprod[t]
        var = var.clamp(*clamp_range)
        s, min_var = s.to(var.device), min_var.to(var.device)
        ratio = (((var * min_var) ** 0.5).acos() / (torch.pi * 0.5)) * (1 + s) - s
        return ratio

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        images: Union[torch.Tensor, PIL.Image.Image, List[torch.Tensor], List[PIL.Image.Image]] = None,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 20,
        timesteps: List[float] = None,
        guidance_scale: float = 4.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        prompt_embeds_pooled: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds_pooled: Optional[torch.FloatTensor] = None,
        image_embeds: Optional[torch.FloatTensor] = None,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pt",
        return_dict: bool = True,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        **kwargs,
    ):
        """
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide the image generation.
            height (`int`, *optional*, defaults to 1024):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to 1024):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 60):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process. If not defined, equal spaced `num_inference_steps`
                timesteps are used. Must be in descending order.
            guidance_scale (`float`, *optional*, defaults to 8.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `decoder_guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting
                `decoder_guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely
                linked to the text `prompt`, usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `decoder_guidance_scale` is less than `1`).
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            prompt_embeds_pooled (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            negative_prompt_embeds_pooled (`torch.FloatTensor`, *optional*):
                Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds_pooled will be generated from `negative_prompt` input
                argument.
            image_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated image embeddings. Can be used to easily tweak image inputs, *e.g.* prompt weighting.
                If not provided, image embeddings will be generated from `image` input argument if existing.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
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
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeine class.

        Examples:

        Returns:
            [`~pipelines.WuerstchenPriorPipelineOutput`] or `tuple` [`~pipelines.WuerstchenPriorPipelineOutput`] if
            `return_dict` is True, otherwise a `tuple`. When returning a tuple, the first element is a list with the
            generated image embeddings.
        """

        # 0. Define commonly used variables
        device = self._execution_device
        dtype = next(self.prior.parameters()).dtype
        self._guidance_scale = guidance_scale
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # 1. Check inputs. Raise error if not correct
        if prompt is not None and not isinstance(prompt, list):
            if isinstance(prompt, str):
                prompt = [prompt]
            else:
                raise TypeError(f"'prompt' must be of type 'list' or 'str', but got {type(prompt)}.")

        if images is not None and not isinstance(images, list):
            images = [images]

        if self.do_classifier_free_guidance:
            if negative_prompt is not None and not isinstance(negative_prompt, list):
                if isinstance(negative_prompt, str):
                    negative_prompt = [negative_prompt]
                else:
                    raise TypeError(
                        f"'negative_prompt' must be of type 'list' or 'str', but got {type(negative_prompt)}."
                    )

        self.check_inputs(
            prompt,
            images,
            negative_prompt,
            num_inference_steps,
            prompt_embeds=prompt_embeds,
            prompt_embeds_pooled=prompt_embeds_pooled,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_prompt_embeds_pooled=negative_prompt_embeds_pooled,
        )

        # 2. Encode caption + images
        (
            prompt_embeds,
            prompt_embeds_pooled,
            negative_prompt_embeds,
            negative_prompt_embeds_pooled,
        ) = self.encode_prompt(
            prompt=prompt,
            device=device,
            batch_size=batch_size,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            prompt_embeds_pooled=prompt_embeds_pooled,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_prompt_embeds_pooled=negative_prompt_embeds_pooled,
        )

        if images is not None:
            image_embeds_pooled, uncond_image_embeds_pooled = self.encode_image(
                images=images,
                device=device,
                dtype=dtype,
                batch_size=batch_size,
                num_images_per_prompt=num_images_per_prompt,
            )
        else:
            image_embeds_pooled = torch.zeros(
                batch_size * num_images_per_prompt, 1, self.prior.config.c_clip_img, device=device, dtype=dtype
            )
            uncond_image_embeds_pooled = torch.zeros(
                batch_size * num_images_per_prompt, 1, self.prior.config.c_clip_img, device=device, dtype=dtype
            )
        if self.do_classifier_free_guidance:
            image_embeds = torch.cat([image_embeds_pooled, uncond_image_embeds_pooled], dim=0)
        else:
            image_embeds = image_embeds_pooled

        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        text_encoder_hidden_states = (
            torch.cat([prompt_embeds, negative_prompt_embeds]) if negative_prompt_embeds is not None else prompt_embeds
        )
        text_encoder_pooled = (
            torch.cat([prompt_embeds_pooled, negative_prompt_embeds_pooled])
            if negative_prompt_embeds is not None
            else prompt_embeds_pooled
        )

        # 3. Determine latent shape of image embeddings
        latent_height = ceil(height / self.config.resolution_multiple)
        latent_width = ceil(width / self.config.resolution_multiple)
        effnet_features_shape = (
            num_images_per_prompt * batch_size,
            self.prior.config.in_channels,
            latent_height,
            latent_width,
        )

        # 4. Prepare and set timesteps
        if timesteps is not None:
            self.scheduler.set_timesteps(timesteps=timesteps, device=device)
            timesteps = self.scheduler.timesteps
            num_inference_steps = len(timesteps)
        else:
            self.scheduler.set_timesteps(num_inference_steps, device=device)
            timesteps = self.scheduler.timesteps

        # 5. Prepare latents
        latents = self.prepare_latents(effnet_features_shape, dtype, device, generator, latents, self.scheduler)

        if isinstance(self.scheduler, DDPMWuerstchenScheduler):
            timesteps = timesteps[:-1]
        else:
            self.scheduler.config.clip_sample = False  # disample sample clipping
        # 6. Run denoising loop
        if hasattr(self.scheduler, "betas"):
            alphas = 1.0 - self.scheduler.betas
            alphas_cumprod = torch.cumprod(alphas, dim=0)
        else:
            alphas_cumprod = []

        self._num_timesteps = len(timesteps)
        for i, t in enumerate(self.progress_bar(timesteps)):
            if not isinstance(self.scheduler, DDPMWuerstchenScheduler):
                if len(alphas_cumprod) > 0:
                    ratio = self.get_t_condioning(t.long().cpu(), alphas_cumprod)
                    ratio = ratio.expand(latents.size(0)).to(dtype).to(device)
                else:
                    ratio = t.float().div(self.scheduler.timesteps[-1]).expand(latents.size(0)).to(dtype)
            else:
                ratio = t.expand(latents.size(0)).to(dtype)
            # 7. Denoise image embeddings
            predicted_image_embedding = self.prior(
                x=torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents,
                r=torch.cat([ratio] * 2) if self.do_classifier_free_guidance else ratio,
                clip_text_pooled=text_encoder_pooled,
                clip_text=text_encoder_hidden_states,
                clip_img=image_embeds,
            )

            # 8. Check for classifier free guidance and apply it
            if self.do_classifier_free_guidance:
                predicted_image_embedding_text, predicted_image_embedding_uncond = predicted_image_embedding.chunk(2)
                predicted_image_embedding = torch.lerp(
                    predicted_image_embedding_uncond, predicted_image_embedding_text, self.guidance_scale
                )

            # 9. Renoise latents to next timestep
            if not isinstance(self.scheduler, DDPMWuerstchenScheduler):
                ratio = t
            latents = self.scheduler.step(
                model_output=predicted_image_embedding,
                timestep=ratio,
                sample=latents,
                generator=generator,
            ).prev_sample

            if callback_on_step_end is not None:
                callback_kwargs = {}
                for k in callback_on_step_end_tensor_inputs:
                    callback_kwargs[k] = locals()[k]
                callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                latents = callback_outputs.pop("latents", latents)
                text_encoder_hidden_states = callback_outputs.pop(
                    "text_encoder_hidden_states", text_encoder_hidden_states
                )
                negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

            # if callback is not None and i % callback_steps == 0:
            #     step_idx = i // getattr(self.scheduler, "order", 1)
            #     callback(step_idx, t, latents)

        # Offload all models
        self.maybe_free_model_hooks()

        if output_type == "np":
            latents = latents.cpu().float().numpy()

        if not return_dict:
            return (latents,)

        return WuerstchenPriorPipelineOutput(latents)
