# Copyright 2025 The Photoroom and The HuggingFace Teams. All rights reserved.
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

import html
import inspect
from typing import Callable

import torch
from transformers import AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from ...image_processor import PixArtImageProcessor
from ...models.transformers.transformer_prx import PRXTransformer2DModel
from ...pipelines.pipeline_utils import DiffusionPipeline
from ...schedulers import FlowMatchEulerDiscreteScheduler
from ...utils import is_ftfy_available, logging, replace_example_docstring
from ...utils.torch_utils import randn_tensor
from .pipeline_output import PRXPipelineOutput
from .pipeline_prx import TextPreprocessor


if is_ftfy_available():
    import ftfy


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# PRXPixel is a 1024px model.
PRX_PIXEL_DEFAULT_RESOLUTION = 1024
# Number of text tokens used at training time (the Qwen tokenizer's own ``model_max_length`` is far larger).
PRX_PIXEL_DEFAULT_MAX_TOKENS = 256

# Predefined aspect-ratio bins for 1024px generation (mirrors ASPECT_RATIO_1024_BIN in pipeline_prx).
ASPECT_RATIO_1024_BIN = {
    "0.49": [704, 1440],
    "0.52": [736, 1408],
    "0.53": [736, 1376],
    "0.57": [768, 1344],
    "0.59": [768, 1312],
    "0.62": [800, 1280],
    "0.67": [832, 1248],
    "0.68": [832, 1216],
    "0.78": [896, 1152],
    "0.83": [928, 1120],
    "0.94": [992, 1056],
    "1.0": [1024, 1024],
    "1.06": [1056, 992],
    "1.13": [1088, 960],
    "1.21": [1120, 928],
    "1.29": [1152, 896],
    "1.37": [1184, 864],
    "1.46": [1216, 832],
    "1.5": [1248, 832],
    "1.71": [1312, 768],
    "1.75": [1344, 768],
    "1.87": [1376, 736],
    "1.91": [1408, 736],
    "2.05": [1440, 704],
}

ASPECT_RATIO_BINS = {
    1024: ASPECT_RATIO_1024_BIN,
}


def _basic_clean(text: str) -> str:
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import PRXPixelPipeline

        >>> pipe = PRXPixelPipeline.from_pretrained("Photoroom/prxpixel-t2i", torch_dtype=torch.bfloat16)
        >>> pipe.to("cuda")

        >>> prompt = "A front-facing portrait of a lion in the golden savanna at sunset."
        >>> image = pipe(prompt, num_inference_steps=28, guidance_scale=5.0).images[0]
        >>> image.save("prxpixel_output.png")
        ```
"""


class PRXPixelPipeline(DiffusionPipeline):
    r"""
    Pipeline for text-to-image generation with the PRXPixel model.

    PRXPixel is a standalone, pixel-space text-to-image pipeline. It denoises raw RGB directly with a ~7B-parameter
    [`PRXTransformer2DModel`] and has no VAE (generation happens entirely in pixel space, so the denoised output *is*
    the image). Prompts are encoded with a Qwen3-VL text encoder (the vision tower is discarded). Unlike
    [`PRXPipeline`] the transformer is trained with x-prediction: at every step it predicts the clean image `x0`, which
    is converted to a flow-matching velocity before the scheduler step. Sampling starts from `randn * noise_scale`
    (`noise_scale=2.0` by default) and the default resolution is 1024px.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Examples:
        ```py
        >>> import torch
        >>> from diffusers import PRXPixelPipeline

        >>> pipe = PRXPixelPipeline.from_pretrained("Photoroom/prxpixel-t2i", torch_dtype=torch.bfloat16)
        >>> pipe.to("cuda")

        >>> prompt = "A front-facing portrait of a lion in the golden savanna at sunset."
        >>> image = pipe(prompt, num_inference_steps=28, guidance_scale=5.0).images[0]
        >>> image.save("prxpixel_output.png")
        ```

    Args:
        transformer ([`PRXTransformer2DModel`]):
            The ~7B-parameter PRX denoiser. For PRXPixel this is built with `in_channels=3`, a bottleneck `img_in`, and
            `resolution_embeds=True`, and it is trained to predict the clean image `x0`.
        scheduler ([`FlowMatchEulerDiscreteScheduler`]):
            Flow-matching scheduler used to denoise the (pixel-space) latents.
        text_encoder ([`PreTrainedModel`]):
            The Qwen3-VL text backbone used to encode prompts (the vision tower is discarded). Must return a
            `last_hidden_state`.
        tokenizer ([`PreTrainedTokenizerBase`]):
            Tokenizer for `text_encoder` (typically loaded via `AutoTokenizer`).
        default_sample_size (`int`, *optional*, defaults to 1024):
            Default height/width used when none is provided to `__call__`.
        prompt_max_tokens (`int`, *optional*, defaults to 256):
            Number of text tokens the prompt is padded/truncated to before encoding.
        noise_scale (`float`, *optional*, defaults to 2.0):
            Scale applied to the initial Gaussian noise. PRXPixel trains with a non-unit initial-noise scale, so
            sampling must start from `randn * noise_scale`.
    """

    model_cpu_offload_seq = "text_encoder->transformer"
    _callback_tensor_inputs = ["latents", "prompt_embeds"]

    def __init__(
        self,
        transformer: PRXTransformer2DModel,
        scheduler: FlowMatchEulerDiscreteScheduler,
        text_encoder: PreTrainedModel,
        tokenizer: AutoTokenizer | PreTrainedTokenizerBase,
        default_sample_size: int | None = PRX_PIXEL_DEFAULT_RESOLUTION,
        prompt_max_tokens: int = PRX_PIXEL_DEFAULT_MAX_TOKENS,
        noise_scale: float = 2.0,
    ):
        super().__init__()

        self.text_preprocessor = TextPreprocessor()
        self._guidance_scale = 1.0

        self.register_modules(
            transformer=transformer,
            scheduler=scheduler,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
        )
        self.register_to_config(
            default_sample_size=default_sample_size,
            prompt_max_tokens=prompt_max_tokens,
            noise_scale=noise_scale,
        )

        # Pixel pipeline always has an image_processor (vae_scale_factor=1)
        # so that output_type="pil"/"np" work without a VAE.
        self.image_processor = PixArtImageProcessor(vae_scale_factor=self.vae_scale_factor)

    @property
    def vae_scale_factor(self):
        # PRXPixel operates directly in RGB pixel space: no VAE, no spatial compression.
        return 1

    @property
    # Copied from diffusers.pipelines.prx.pipeline_prx.PRXPipeline.do_classifier_free_guidance
    def do_classifier_free_guidance(self):
        """Check if classifier-free guidance is enabled based on guidance scale."""
        return self._guidance_scale > 1.0

    @property
    # Copied from diffusers.pipelines.prx.pipeline_prx.PRXPipeline.guidance_scale
    def guidance_scale(self):
        return self._guidance_scale

    def _tokenize_prompts(
        self,
        prompts: list[str],
        device: torch.device,
        tokenizer_max_length: int | None = None,
        skip_text_cleaning: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenize and (lightly) clean prompts.

        PRXPixel always uses light cleaning (`_basic_clean`) and the training-time token budget
        (`self.config.prompt_max_tokens`). The `tokenizer_max_length` and `skip_text_cleaning` arguments are accepted
        for API compatibility with the copied callers but are ignored.
        """
        cleaned = [_basic_clean(text) for text in prompts]
        tokens = self.tokenizer(
            cleaned,
            padding="max_length",
            max_length=self.config.prompt_max_tokens,
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return tokens["input_ids"].to(device), tokens["attention_mask"].bool().to(device)

    # Copied from diffusers.pipelines.prx.pipeline_prx.PRXPipeline._encode_prompt_standard
    def _encode_prompt_standard(
        self,
        prompt: list[str],
        device: torch.device,
        do_classifier_free_guidance: bool = True,
        negative_prompt: str = "",
        tokenizer_max_length: int | None = None,
        skip_text_cleaning: bool = False,
    ):
        """Encode prompt using standard text encoder and tokenizer with batch processing."""
        batch_size = len(prompt)

        if do_classifier_free_guidance:
            if isinstance(negative_prompt, str):
                negative_prompt = [negative_prompt] * batch_size

            prompts_to_encode = negative_prompt + prompt
        else:
            prompts_to_encode = prompt

        input_ids, attention_mask = self._tokenize_prompts(
            prompts_to_encode, device, tokenizer_max_length=tokenizer_max_length, skip_text_cleaning=skip_text_cleaning
        )

        with torch.no_grad():
            embeddings = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )["last_hidden_state"]

        if do_classifier_free_guidance:
            uncond_text_embeddings, text_embeddings = embeddings.split(batch_size, dim=0)
            uncond_cross_attn_mask, cross_attn_mask = attention_mask.split(batch_size, dim=0)
        else:
            text_embeddings = embeddings
            cross_attn_mask = attention_mask
            uncond_text_embeddings = None
            uncond_cross_attn_mask = None

        return text_embeddings, cross_attn_mask, uncond_text_embeddings, uncond_cross_attn_mask

    # Copied from diffusers.pipelines.prx.pipeline_prx.PRXPipeline.encode_prompt
    def encode_prompt(
        self,
        prompt: str | list[str],
        device: torch.device | None = None,
        do_classifier_free_guidance: bool = True,
        negative_prompt: str = "",
        num_images_per_prompt: int = 1,
        prompt_embeds: torch.FloatTensor | None = None,
        negative_prompt_embeds: torch.FloatTensor | None = None,
        prompt_attention_mask: torch.BoolTensor | None = None,
        negative_prompt_attention_mask: torch.BoolTensor | None = None,
        tokenizer_max_length: int | None = None,
        skip_text_cleaning: bool = False,
    ):
        """Encode text prompt using standard text encoder and tokenizer, or use precomputed embeddings."""
        if device is None:
            device = self._execution_device

        if prompt_embeds is None:
            if isinstance(prompt, str):
                prompt = [prompt]
            # Encode the prompts
            prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask = (
                self._encode_prompt_standard(
                    prompt,
                    device,
                    do_classifier_free_guidance,
                    negative_prompt,
                    tokenizer_max_length=tokenizer_max_length,
                    skip_text_cleaning=skip_text_cleaning,
                )
            )

        # Duplicate embeddings for each generation per prompt
        if num_images_per_prompt > 1:
            # Repeat prompt embeddings
            bs_embed, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
            prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

            if prompt_attention_mask is not None:
                prompt_attention_mask = prompt_attention_mask.view(bs_embed, -1)
                prompt_attention_mask = prompt_attention_mask.repeat(num_images_per_prompt, 1)

            # Repeat negative embeddings if using CFG
            if do_classifier_free_guidance and negative_prompt_embeds is not None:
                bs_embed, seq_len, _ = negative_prompt_embeds.shape
                negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
                negative_prompt_embeds = negative_prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

                if negative_prompt_attention_mask is not None:
                    negative_prompt_attention_mask = negative_prompt_attention_mask.view(bs_embed, -1)
                    negative_prompt_attention_mask = negative_prompt_attention_mask.repeat(num_images_per_prompt, 1)

        return (
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds if do_classifier_free_guidance else None,
            negative_prompt_attention_mask if do_classifier_free_guidance else None,
        )

    def check_inputs(
        self,
        prompt: str | list[str],
        height: int,
        width: int,
        guidance_scale: float,
        callback_on_step_end_tensor_inputs: list[str] | None = None,
        prompt_embeds: torch.FloatTensor | None = None,
        negative_prompt_embeds: torch.FloatTensor | None = None,
    ):
        """Check that all inputs are in correct format."""
        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )

        if prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )

        if prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if prompt_embeds is not None and guidance_scale > 1.0 and negative_prompt_embeds is None:
            raise ValueError(
                "When `prompt_embeds` is provided and `guidance_scale > 1.0`, "
                "`negative_prompt_embeds` must also be provided for classifier-free guidance."
            )

        # The latents must be divisible by the transformer's patch size after VAE compression.
        dimension_multiple = self.vae_scale_factor * self.transformer.config.patch_size
        if height % dimension_multiple != 0 or width % dimension_multiple != 0:
            raise ValueError(
                f"`height` and `width` have to be divisible by {dimension_multiple} (vae_scale_factor *"
                f" transformer patch_size) but are {height} and {width}."
            )

        if guidance_scale < 1.0:
            raise ValueError(f"guidance_scale has to be >= 1.0 but is {guidance_scale}")

        if callback_on_step_end_tensor_inputs is not None and not isinstance(callback_on_step_end_tensor_inputs, list):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be a list but is {callback_on_step_end_tensor_inputs}"
            )

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: torch.Generator | None = None,
        latents: torch.Tensor | None = None,
    ):
        """Prepare initial latents for the diffusion process.

        PRXPixel trains with a non-unit initial-noise scale, so the sampled noise is multiplied by
        `self.config.noise_scale`.
        """
        if latents is None:
            spatial_compression = self.vae_scale_factor
            latent_height, latent_width = (
                height // spatial_compression,
                width // spatial_compression,
            )
            shape = (batch_size, num_channels_latents, latent_height, latent_width)
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype) * self.config.noise_scale
        else:
            latents = latents.to(device)
        return latents

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: str | list[str] = None,
        negative_prompt: str = "",
        height: int | None = None,
        width: int | None = None,
        num_inference_steps: int = 28,
        timesteps: list[int] = None,
        guidance_scale: float = 4.0,
        num_images_per_prompt: int | None = 1,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        prompt_embeds: torch.FloatTensor | None = None,
        negative_prompt_embeds: torch.FloatTensor | None = None,
        prompt_attention_mask: torch.BoolTensor | None = None,
        negative_prompt_attention_mask: torch.BoolTensor | None = None,
        output_type: str | None = "pil",
        return_dict: bool = True,
        use_resolution_binning: bool = True,
        callback_on_step_end: Callable[[int, int], None] | None = None,
        callback_on_step_end_tensor_inputs: list[str] = ["latents"],
    ):
        """
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `list[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`
                instead.
            negative_prompt (`str`, *optional*, defaults to `""`):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
            height (`int`, *optional*, defaults to `default_sample_size`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `default_sample_size`):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 28):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`list[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            guidance_scale (`float`, *optional*, defaults to 4.0):
                Guidance scale as defined in [Classifier-Free Diffusion
                Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2.
                of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting
                `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to
                the text `prompt`, usually at the expense of lower image quality.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `list[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will be generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided and `guidance_scale > 1`, negative embeddings will be generated from an
                empty string.
            prompt_attention_mask (`torch.BoolTensor`, *optional*):
                Pre-generated attention mask for `prompt_embeds`. If not provided, attention mask will be generated
                from `prompt` input argument.
            negative_prompt_attention_mask (`torch.BoolTensor`, *optional*):
                Pre-generated attention mask for `negative_prompt_embeds`. If not provided and `guidance_scale > 1`,
                attention mask will be generated from an empty string.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.prx.PRXPipelineOutput`] instead of a plain tuple.
            use_resolution_binning (`bool`, *optional*, defaults to `True`):
                If set to `True`, the requested height and width are first mapped to the closest resolutions using
                predefined aspect ratio bins. After the produced latents are decoded into images, they are resized back
                to the requested resolution. Useful for generating non-square images at optimal resolutions.
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self, step, timestep, callback_kwargs)`.
                `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`list`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include tensors that are listed
                in the `._callback_tensor_inputs` attribute.

        Examples:

        Returns:
            [`~pipelines.prx.PRXPipelineOutput`] or `tuple`: [`~pipelines.prx.PRXPipelineOutput`] if `return_dict` is
            True, otherwise a `tuple. When returning a tuple, the first element is a list with the generated images.
        """

        # 0. Set height and width
        default_resolution = getattr(self.config, "default_sample_size", None) or PRX_PIXEL_DEFAULT_RESOLUTION
        height = height or default_resolution
        width = width or default_resolution

        if use_resolution_binning:
            if self.config.default_sample_size not in ASPECT_RATIO_BINS:
                raise ValueError(
                    f"Resolution binning is only supported for default_sample_size in {list(ASPECT_RATIO_BINS.keys())}, "
                    f"but got {self.config.default_sample_size}. Set use_resolution_binning=False to disable aspect ratio binning."
                )
            aspect_ratio_bin = ASPECT_RATIO_BINS[self.config.default_sample_size]

            # Store original dimensions
            orig_height, orig_width = height, width
            # Map to closest resolution in the bin
            height, width = self.image_processor.classify_height_width_bin(height, width, ratios=aspect_ratio_bin)

        # 1. Check inputs
        self.check_inputs(
            prompt,
            height,
            width,
            guidance_scale,
            callback_on_step_end_tensor_inputs,
            prompt_embeds,
            negative_prompt_embeds,
        )

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # Use execution device (handles offloading scenarios including group offloading)
        device = self._execution_device

        self._guidance_scale = guidance_scale

        # 2. Encode input prompt
        text_embeddings, cross_attn_mask, uncond_text_embeddings, uncond_cross_attn_mask = self.encode_prompt(
            prompt,
            device,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
        )
        # Expose standard names for callbacks parity
        prompt_embeds = text_embeddings
        negative_prompt_embeds = uncond_text_embeddings

        # 3. Prepare timesteps
        if timesteps is not None:
            self.scheduler.set_timesteps(timesteps=timesteps, device=device)
            timesteps = self.scheduler.timesteps
            num_inference_steps = len(timesteps)
        else:
            self.scheduler.set_timesteps(num_inference_steps, device=device)
            timesteps = self.scheduler.timesteps

        self.num_timesteps = len(timesteps)

        # 4. Prepare latent variables (pixel space: in_channels RGB tensors, no VAE)
        num_channels_latents = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            text_embeddings.dtype,
            device,
            generator,
            latents,
        )

        # 5. Prepare extra step kwargs
        extra_step_kwargs = {}
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_eta:
            extra_step_kwargs["eta"] = 0.0

        # 6. Prepare cross-attention embeddings and masks
        if self.do_classifier_free_guidance:
            ca_embed = torch.cat([uncond_text_embeddings, text_embeddings], dim=0)
            ca_mask = None
            if cross_attn_mask is not None and uncond_cross_attn_mask is not None:
                ca_mask = torch.cat([uncond_cross_attn_mask, cross_attn_mask], dim=0)
        else:
            ca_embed = text_embeddings
            ca_mask = cross_attn_mask

        # 7. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # Duplicate latents if using classifier-free guidance
                if self.do_classifier_free_guidance:
                    latents_in = torch.cat([latents, latents], dim=0)
                    # Normalize timestep for the transformer
                    t_cont = (t.float() / self.scheduler.config.num_train_timesteps).view(1).repeat(2).to(device)
                else:
                    latents_in = latents
                    # Normalize timestep for the transformer
                    t_cont = (t.float() / self.scheduler.config.num_train_timesteps).view(1).to(device)

                # Forward through transformer
                noise_pred = self.transformer(
                    hidden_states=latents_in,
                    timestep=t_cont,
                    encoder_hidden_states=ca_embed,
                    attention_mask=ca_mask,
                    return_dict=False,
                )[0]

                # Apply CFG
                if self.do_classifier_free_guidance:
                    noise_uncond, noise_text = noise_pred.chunk(2, dim=0)
                    noise_pred = noise_uncond + guidance_scale * (noise_text - noise_uncond)

                # PRXPixel predicts x0; convert to flow-matching velocity before the scheduler step.
                t_x = torch.clamp(t.float() / self.scheduler.config.num_train_timesteps, min=0.05)
                noise_pred = (latents - noise_pred) / t_x

                # Compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_on_step_end(self, i, t, callback_kwargs)

                # Call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        # 8. Post-processing (pixel space: the denoised output IS the image in [-1, 1]; no VAE decode).
        if output_type in ["latent", "pt"]:
            image = latents
        else:
            image = latents
            # Resize back to original resolution if using binning
            if use_resolution_binning:
                image = self.image_processor.resize_and_crop_tensor(image, orig_width, orig_height)

            # Use standard image processor for post-processing
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return PRXPipelineOutput(images=image)
