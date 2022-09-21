from typing import Dict, List, Optional, Union

import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict
from transformers import CLIPFeatureExtractor, CLIPTokenizer, FlaxCLIPTextModel

from ...models import FlaxAutoencoderKL, FlaxUNet2DConditionModel
from ...pipeline_flax_utils import FlaxDiffusionPipeline
from ...schedulers import FlaxDDIMScheduler, FlaxLMSDiscreteScheduler, FlaxPNDMScheduler
from . import FlaxStableDiffusionPipelineOutput
from .safety_checker_flax import FlaxStableDiffusionSafetyChecker


class FlaxStableDiffusionPipeline(FlaxDiffusionPipeline):
    r"""
    Pipeline for text-to-image generation using Stable Diffusion.

    This model inherits from [`FlaxDiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        vae ([`FlaxAutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`FlaxCLIPTextModel`]):
            Frozen text-encoder. Stable Diffusion uses the text portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.FlaxCLIPTextModel),
            specifically the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        unet ([`FlaxUNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latens. Can be one of
            [`FlaxDDIMScheduler`], [`FlaxLMSDiscreteScheduler`], or [`FlaxPNDMScheduler`].
        safety_checker ([`FlaxStableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offsensive or harmful.
            Please, refer to the [model card](https://huggingface.co/CompVis/stable-diffusion-v1-4) for details.
        feature_extractor ([`CLIPFeatureExtractor`]):
            Model that extracts features from generated images to be used as inputs for the `safety_checker`.
    """

    def __init__(
        self,
        vae: FlaxAutoencoderKL,
        text_encoder: FlaxCLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: FlaxUNet2DConditionModel,
        scheduler: Union[FlaxDDIMScheduler, FlaxPNDMScheduler, FlaxLMSDiscreteScheduler],
        safety_checker: FlaxStableDiffusionSafetyChecker,
        feature_extractor: CLIPFeatureExtractor,
        dtype: jnp.dtype = jnp.float32,
    ):
        super().__init__()
        scheduler = scheduler.set_format("np")
        self.dtype = dtype

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )

    def prepare_inputs(self, prompt: Union[str, List[str]]):
        if not isinstance(prompt, (str, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="np",
        )
        return text_input.input_ids

    def __call__(
        self,
        prompt_ids: jnp.array,
        params: Union[Dict, FrozenDict],
        prng_seed: jax.random.PRNGKey,
        num_inference_steps: Optional[int] = 50,
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        guidance_scale: Optional[float] = 7.5,
        latents: Optional[jnp.array] = None,
        return_dict: bool = True,
        debug: bool = False,
        **kwargs,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide the image generation.
            height (`int`, *optional*, defaults to 512):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to 512):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            generator (`torch.Generator`, *optional*):
                A [torch generator](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation
                deterministic.
            latents (`jnp.array`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.FlaxStableDiffusionPipelineOutput`] instead of
                a plain tuple.

        Returns:
            [`~pipelines.stable_diffusion.FlaxStableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.FlaxStableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a
            `tuple. When returning a tuple, the first element is a list with the generated images, and the second
            element is a list of `bool`s denoting whether the corresponding generated image likely represents
            "not-safe-for-work" (nsfw) content, according to the `safety_checker`.
        """
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        # get prompt text embeddings
        text_embeddings = self.text_encoder(prompt_ids, params=params["text_encoder"])[0]

        # TODO: currently it is assumed `do_classifier_free_guidance = guidance_scale > 1.0`
        # implement this conditional `do_classifier_free_guidance = guidance_scale > 1.0`
        batch_size = prompt_ids.shape[0]

        max_length = prompt_ids.shape[-1]
        uncond_input = self.tokenizer(
            [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="np"
        )
        uncond_embeddings = self.text_encoder(uncond_input.input_ids, params=params["text_encoder"])[0]
        context = jnp.concatenate([uncond_embeddings, text_embeddings])

        # TODO: check it because the shape is different from Pytorhc StableDiffusionPipeline
        latents_shape = (
            batch_size,
            self.unet.in_channels,
            self.unet.sample_size,
            self.unet.sample_size,
        )
        if latents is None:
            latents = jax.random.normal(prng_seed, shape=latents_shape, dtype=jnp.float32)
        else:
            if latents.shape != latents_shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {latents_shape}")

        def loop_body(step, args):
            latents, scheduler_state = args
            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            latents_input = jnp.concatenate([latents] * 2)

            t = jnp.array(scheduler_state.timesteps, dtype=jnp.int32)[step]
            timestep = jnp.broadcast_to(t, latents_input.shape[0])

            # predict the noise residual
            noise_pred = self.unet.apply(
                {"params": params["unet"]},
                jnp.array(latents_input),
                jnp.array(timestep, dtype=jnp.int32),
                encoder_hidden_states=context,
            ).sample
            # perform guidance
            noise_pred_uncond, noise_prediction_text = jnp.split(noise_pred, 2, axis=0)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents, scheduler_state = self.scheduler.step(scheduler_state, noise_pred, t, latents).to_tuple()
            return latents, scheduler_state

        scheduler_state = self.scheduler.set_timesteps(params["scheduler"], num_inference_steps=num_inference_steps)

        if debug:
            # run with python for loop
            for i in range(num_inference_steps):
                latents, scheduler_state = loop_body(i, (latents, scheduler_state))
        else:
            latents, _ = jax.lax.fori_loop(0, num_inference_steps, loop_body, (latents, scheduler_state))

        # scale and decode the image latents with vae
        latents = 1 / 0.18215 * latents
        # TODO: check when flax vae gets merged into main
        image = self.vae.apply({"params": params["vae"]}, latents, method=self.vae.decode).sample

        image = (image / 2 + 0.5).clip(0, 1).transpose(0, 2, 3, 1)

        #        image = jnp.asarray(image).transpose(0, 2, 3, 1)
        # run safety checker
        # TODO: check when flax safety checker gets merged into main
        #        safety_cheker_input = self.feature_extractor(self.numpy_to_pil(image), return_tensors="np")
        #        image, has_nsfw_concept = self.safety_checker(
        #            images=image, clip_input=safety_cheker_input.pixel_values, params=params["safety_params"]
        #        )
        has_nsfw_concept = False

        if not return_dict:
            return (image, has_nsfw_concept)

        return FlaxStableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
