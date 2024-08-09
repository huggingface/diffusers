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
from typing import Any, Callable, Dict, List, Optional, Union

import PIL.Image
import torch
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection

from ...image_processor import VaeImageProcessor
from ...loaders import StableDiffusionLoraLoaderMixin, TextualInversionLoaderMixin
from ...models import AutoencoderKL, UNet2DConditionModel
from ...models.embeddings import get_timestep_embedding
from ...models.lora import adjust_lora_scale_text_encoder
from ...schedulers import KarrasDiffusionSchedulers
from ...utils import (
    USE_PEFT_BACKEND,
    deprecate,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from ...utils.torch_utils import randn_tensor
from ..pipeline_utils import DiffusionPipeline, ImagePipelineOutput, StableDiffusionMixin
from .stable_unclip_image_normalizer import StableUnCLIPImageNormalizer


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import requests
        >>> import torch
        >>> from PIL import Image
        >>> from io import BytesIO

        >>> from diffusers import StableUnCLIPImg2ImgPipeline

        >>> pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(
        ...     "stabilityai/stable-diffusion-2-1-unclip-small", torch_dtype=torch.float16
        ... )
        >>> pipe = pipe.to("cuda")

        >>> url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"

        >>> response = requests.get(url)
        >>> init_image = Image.open(BytesIO(response.content)).convert("RGB")
        >>> init_image = init_image.resize((768, 512))

        >>> prompt = "A fantasy landscape, trending on artstation"

        >>> images = pipe(init_image, prompt).images
        >>> images[0].save("fantasy_landscape.png")
        ```
"""


class StableUnCLIPImg2ImgPipeline(
    DiffusionPipeline, StableDiffusionMixin, TextualInversionLoaderMixin, StableDiffusionLoraLoaderMixin
):
    """
    Pipeline for text-guided image-to-image generation using stable unCLIP.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    The pipeline also inherits the following loading methods:
        - [`~loaders.TextualInversionLoaderMixin.load_textual_inversion`] for loading textual inversion embeddings
        - [`~loaders.StableDiffusionLoraLoaderMixin.load_lora_weights`] for loading LoRA weights
        - [`~loaders.StableDiffusionLoraLoaderMixin.save_lora_weights`] for saving LoRA weights

    Args:
        feature_extractor ([`CLIPImageProcessor`]):
            Feature extractor for image pre-processing before being encoded.
        image_encoder ([`CLIPVisionModelWithProjection`]):
            CLIP vision model for encoding images.
        image_normalizer ([`StableUnCLIPImageNormalizer`]):
            Used to normalize the predicted image embeddings before the noise is applied and un-normalize the image
            embeddings after the noise has been applied.
        image_noising_scheduler ([`KarrasDiffusionSchedulers`]):
            Noise schedule for adding noise to the predicted image embeddings. The amount of noise to add is determined
            by the `noise_level`.
        tokenizer (`~transformers.CLIPTokenizer`):
            A [`~transformers.CLIPTokenizer`)].
        text_encoder ([`~transformers.CLIPTextModel`]):
            Frozen [`~transformers.CLIPTextModel`] text-encoder.
        unet ([`UNet2DConditionModel`]):
            A [`UNet2DConditionModel`] to denoise the encoded image latents.
        scheduler ([`KarrasDiffusionSchedulers`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents.
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
    """

    model_cpu_offload_seq = "text_encoder->image_encoder->unet->vae"
    _exclude_from_cpu_offload = ["image_normalizer"]

    # image encoding components
    feature_extractor: CLIPImageProcessor
    image_encoder: CLIPVisionModelWithProjection

    # image noising components
    image_normalizer: StableUnCLIPImageNormalizer
    image_noising_scheduler: KarrasDiffusionSchedulers

    # regular denoising components
    tokenizer: CLIPTokenizer
    text_encoder: CLIPTextModel
    unet: UNet2DConditionModel
    scheduler: KarrasDiffusionSchedulers

    vae: AutoencoderKL

    def __init__(
        self,
        # image encoding components
        feature_extractor: CLIPImageProcessor,
        image_encoder: CLIPVisionModelWithProjection,
        # image noising components
        image_normalizer: StableUnCLIPImageNormalizer,
        image_noising_scheduler: KarrasDiffusionSchedulers,
        # regular denoising components
        tokenizer: CLIPTokenizer,
        text_encoder: CLIPTextModel,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        # vae
        vae: AutoencoderKL,
    ):
        super().__init__()

        self.register_modules(
            feature_extractor=feature_extractor,
            image_encoder=image_encoder,
            image_normalizer=image_normalizer,
            image_noising_scheduler=image_noising_scheduler,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            unet=unet,
            scheduler=scheduler,
            vae=vae,
        )

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline._encode_prompt
    def _encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        lora_scale: Optional[float] = None,
        **kwargs,
    ):
        deprecation_message = "`_encode_prompt()` is deprecated and it will be removed in a future version. Use `encode_prompt()` instead. Also, be aware that the output format changed from a concatenated tensor to a tuple."
        deprecate("_encode_prompt()", "1.0.0", deprecation_message, standard_warn=False)

        prompt_embeds_tuple = self.encode_prompt(
            prompt=prompt,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=lora_scale,
            **kwargs,
        )

        # concatenate for backwards comp
        prompt_embeds = torch.cat([prompt_embeds_tuple[1], prompt_embeds_tuple[0]])

        return prompt_embeds

    def _encode_image(
        self,
        image,
        device,
        batch_size,
        num_images_per_prompt,
        do_classifier_free_guidance,
        noise_level,
        generator,
        image_embeds,
    ):
        dtype = next(self.image_encoder.parameters()).dtype

        if isinstance(image, PIL.Image.Image):
            # the image embedding should repeated so it matches the total batch size of the prompt
            repeat_by = batch_size
        else:
            # assume the image input is already properly batched and just needs to be repeated so
            # it matches the num_images_per_prompt.
            #
            # NOTE(will) this is probably missing a few number of side cases. I.e. batched/non-batched
            # `image_embeds`. If those happen to be common use cases, let's think harder about
            # what the expected dimensions of inputs should be and how we handle the encoding.
            repeat_by = num_images_per_prompt

        if image_embeds is None:
            if not isinstance(image, torch.Tensor):
                image = self.feature_extractor(images=image, return_tensors="pt").pixel_values

            image = image.to(device=device, dtype=dtype)
            image_embeds = self.image_encoder(image).image_embeds

        image_embeds = self.noise_image_embeddings(
            image_embeds=image_embeds,
            noise_level=noise_level,
            generator=generator,
        )

        # duplicate image embeddings for each generation per prompt, using mps friendly method
        image_embeds = image_embeds.unsqueeze(1)
        bs_embed, seq_len, _ = image_embeds.shape
        image_embeds = image_embeds.repeat(1, repeat_by, 1)
        image_embeds = image_embeds.view(bs_embed * repeat_by, seq_len, -1)
        image_embeds = image_embeds.squeeze(1)

        if do_classifier_free_guidance:
            negative_prompt_embeds = torch.zeros_like(image_embeds)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            image_embeds = torch.cat([negative_prompt_embeds, image_embeds])

        return image_embeds

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.encode_prompt
    def encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        lora_scale: Optional[float] = None,
        clip_skip: Optional[int] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
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
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            lora_scale (`float`, *optional*):
                A LoRA scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
        """
        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(self, StableDiffusionLoraLoaderMixin):
            self._lora_scale = lora_scale

            # dynamically adjust the LoRA scale
            if not USE_PEFT_BACKEND:
                adjust_lora_scale_text_encoder(self.text_encoder, lora_scale)
            else:
                scale_lora_layers(self.text_encoder, lora_scale)

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            # textual inversion: process multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                prompt = self.maybe_convert_prompt(prompt, self.tokenizer)

            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
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

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = text_inputs.attention_mask.to(device)
            else:
                attention_mask = None

            if clip_skip is None:
                prompt_embeds = self.text_encoder(text_input_ids.to(device), attention_mask=attention_mask)
                prompt_embeds = prompt_embeds[0]
            else:
                prompt_embeds = self.text_encoder(
                    text_input_ids.to(device), attention_mask=attention_mask, output_hidden_states=True
                )
                # Access the `hidden_states` first, that contains a tuple of
                # all the hidden states from the encoder layers. Then index into
                # the tuple to access the hidden states from the desired layer.
                prompt_embeds = prompt_embeds[-1][-(clip_skip + 1)]
                # We also need to apply the final LayerNorm here to not mess with the
                # representations. The `last_hidden_states` that we typically use for
                # obtaining the final prompt representations passes through the LayerNorm
                # layer.
                prompt_embeds = self.text_encoder.text_model.final_layer_norm(prompt_embeds)

        if self.text_encoder is not None:
            prompt_embeds_dtype = self.text_encoder.dtype
        elif self.unet is not None:
            prompt_embeds_dtype = self.unet.dtype
        else:
            prompt_embeds_dtype = prompt_embeds.dtype

        prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif prompt is not None and type(prompt) is not type(negative_prompt):
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

            # textual inversion: process multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                uncond_tokens = self.maybe_convert_prompt(uncond_tokens, self.tokenizer)

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        if self.text_encoder is not None:
            if isinstance(self, StableDiffusionLoraLoaderMixin) and USE_PEFT_BACKEND:
                # Retrieve the original scale by scaling back the LoRA layers
                unscale_lora_layers(self.text_encoder, lora_scale)

        return prompt_embeds, negative_prompt_embeds

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.decode_latents
    def decode_latents(self, latents):
        deprecation_message = "The decode_latents method is deprecated and will be removed in 1.0.0. Please use VaeImageProcessor.postprocess(...) instead"
        deprecate("decode_latents", "1.0.0", deprecation_message, standard_warn=False)

        latents = 1 / self.vae.config.scaling_factor * latents
        if torch.cuda.is_available() and torch.version.hip:
            latents = latents.to(dtype=self.vae.dtype)
        image = self.vae.decode(latents, return_dict=False)[0]
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image

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

    def check_inputs(
        self,
        prompt,
        image,
        height,
        width,
        callback_steps,
        noise_level,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        image_embeds=None,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Please make sure to define only one of the two."
            )

        if prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )

        if prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                "Provide either `negative_prompt` or `negative_prompt_embeds`. Cannot leave both `negative_prompt` and `negative_prompt_embeds` undefined."
            )

        if prompt is not None and negative_prompt is not None:
            if type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

        if noise_level < 0 or noise_level >= self.image_noising_scheduler.config.num_train_timesteps:
            raise ValueError(
                f"`noise_level` must be between 0 and {self.image_noising_scheduler.config.num_train_timesteps - 1}, inclusive."
            )

        if image is not None and image_embeds is not None:
            raise ValueError(
                "Provide either `image` or `image_embeds`. Please make sure to define only one of the two."
            )

        if image is None and image_embeds is None:
            raise ValueError(
                "Provide either `image` or `image_embeds`. Cannot leave both `image` and `image_embeds` undefined."
            )

        if image is not None:
            if (
                not isinstance(image, torch.Tensor)
                and not isinstance(image, PIL.Image.Image)
                and not isinstance(image, list)
            ):
                raise ValueError(
                    "`image` has to be of type `torch.Tensor` or `PIL.Image.Image` or `List[PIL.Image.Image]` but is"
                    f" {type(image)}"
                )

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

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_unclip.StableUnCLIPPipeline.noise_image_embeddings
    def noise_image_embeddings(
        self,
        image_embeds: torch.Tensor,
        noise_level: int,
        noise: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ):
        """
        Add noise to the image embeddings. The amount of noise is controlled by a `noise_level` input. A higher
        `noise_level` increases the variance in the final un-noised images.

        The noise is applied in two ways:
        1. A noise schedule is applied directly to the embeddings.
        2. A vector of sinusoidal time embeddings are appended to the output.

        In both cases, the amount of noise is controlled by the same `noise_level`.

        The embeddings are normalized before the noise is applied and un-normalized after the noise is applied.
        """
        if noise is None:
            noise = randn_tensor(
                image_embeds.shape, generator=generator, device=image_embeds.device, dtype=image_embeds.dtype
            )

        noise_level = torch.tensor([noise_level] * image_embeds.shape[0], device=image_embeds.device)

        self.image_normalizer.to(image_embeds.device)
        image_embeds = self.image_normalizer.scale(image_embeds)

        image_embeds = self.image_noising_scheduler.add_noise(image_embeds, timesteps=noise_level, noise=noise)

        image_embeds = self.image_normalizer.unscale(image_embeds)

        noise_level = get_timestep_embedding(
            timesteps=noise_level, embedding_dim=image_embeds.shape[-1], flip_sin_to_cos=True, downscale_freq_shift=0
        )

        # `get_timestep_embeddings` does not contain any weights and will always return f32 tensors,
        # but we might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        noise_level = noise_level.to(image_embeds.dtype)

        image_embeds = torch.cat((image_embeds, noise_level), 1)

        return image_embeds

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        image: Union[torch.Tensor, PIL.Image.Image] = None,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 20,
        guidance_scale: float = 10,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        noise_level: int = 0,
        image_embeds: Optional[torch.Tensor] = None,
        clip_skip: Optional[int] = None,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, either `prompt_embeds` will be
                used or prompt is initialized to `""`.
            image (`torch.Tensor` or `PIL.Image.Image`):
                `Image` or tensor representing an image batch. The image is encoded to its CLIP embedding which the
                `unet` is conditioned on. The image is _not_ encoded by the `vae` and then used as the latents in the
                denoising process like it is in the standard Stable Diffusion text-guided image variation process.
            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 20):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 10.0):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.
            callback (`Callable`, *optional*):
                A function that calls every `callback_steps` steps during inference. The function is called with the
                following arguments: `callback(step: int, timestep: int, latents: torch.Tensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function is called. If not specified, the callback is called at
                every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            noise_level (`int`, *optional*, defaults to `0`):
                The amount of noise to add to the image embeddings. A higher `noise_level` increases the variance in
                the final un-noised images. See [`StableUnCLIPPipeline.noise_image_embeddings`] for more details.
            image_embeds (`torch.Tensor`, *optional*):
                Pre-generated CLIP embeddings to condition the `unet` on. These latents are not used in the denoising
                process. If you want to provide pre-generated latents, pass them to `__call__` as `latents`.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.

        Examples:

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`:
                [`~ pipeline_utils.ImagePipelineOutput`] if `return_dict` is True, otherwise a `tuple`. When returning
                a tuple, the first element is a list with the generated images.
        """
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        if prompt is None and prompt_embeds is None:
            prompt = len(image) * [""] if isinstance(image, list) else ""

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt=prompt,
            image=image,
            height=height,
            width=width,
            callback_steps=callback_steps,
            noise_level=noise_level,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            image_embeds=image_embeds,
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        batch_size = batch_size * num_images_per_prompt

        device = self._execution_device

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
        )
        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        # 4. Encoder input image
        noise_level = torch.tensor([noise_level], device=device)
        image_embeds = self._encode_image(
            image=image,
            device=device,
            batch_size=batch_size,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            noise_level=noise_level,
            generator=generator,
            image_embeds=image_embeds,
        )

        # 5. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 6. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        if latents is None:
            latents = self.prepare_latents(
                batch_size=batch_size,
                num_channels_latents=num_channels_latents,
                height=height,
                width=width,
                dtype=prompt_embeds.dtype,
                device=device,
                generator=generator,
                latents=latents,
            )

        # 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 8. Denoising loop
        for i, t in enumerate(self.progress_bar(timesteps)):
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                class_labels=image_embeds,
                cross_attention_kwargs=cross_attention_kwargs,
                return_dict=False,
            )[0]

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

            if callback is not None and i % callback_steps == 0:
                step_idx = i // getattr(self.scheduler, "order", 1)
                callback(step_idx, t, latents)

        # 9. Post-processing
        if not output_type == "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
        else:
            image = latents

        image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)
