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

import warnings
from functools import partial
from typing import Dict, List, Optional, Union

import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict
from flax.jax_utils import unreplicate
from flax.training.common_utils import shard
from PIL import Image
from transformers import CLIPFeatureExtractor, CLIPTokenizer, FlaxCLIPTextModel

from ...models import FlaxAutoencoderKL, FlaxControlNetModel, FlaxUNet2DConditionModel
from ...schedulers import (
    FlaxDDIMScheduler,
    FlaxDPMSolverMultistepScheduler,
    FlaxLMSDiscreteScheduler,
    FlaxPNDMScheduler,
)
from ...utils import PIL_INTERPOLATION, logging, replace_example_docstring
from ..pipeline_flax_utils import FlaxDiffusionPipeline
from ..stable_diffusion import FlaxStableDiffusionPipelineOutput
from ..stable_diffusion.safety_checker_flax import FlaxStableDiffusionSafetyChecker


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# Set to True to use python for loop instead of jax.fori_loop for easier debugging
DEBUG = False

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import jax
        >>> import numpy as np
        >>> import jax.numpy as jnp
        >>> from flax.jax_utils import replicate
        >>> from flax.training.common_utils import shard
        >>> from diffusers.utils import load_image, make_image_grid
        >>> from PIL import Image
        >>> from diffusers import FlaxStableDiffusionControlNetPipeline, FlaxControlNetModel


        >>> def create_key(seed=0):
        ...     return jax.random.PRNGKey(seed)


        >>> rng = create_key(0)

        >>> # get canny image
        >>> canny_image = load_image(
        ...     "https://huggingface.co/datasets/YiYiXu/test-doc-assets/resolve/main/blog_post_cell_10_output_0.jpeg"
        ... )

        >>> prompts = "best quality, extremely detailed"
        >>> negative_prompts = "monochrome, lowres, bad anatomy, worst quality, low quality"

        >>> # load control net and stable diffusion v1-5
        >>> controlnet, controlnet_params = FlaxControlNetModel.from_pretrained(
        ...     "lllyasviel/sd-controlnet-canny", from_pt=True, dtype=jnp.float32
        ... )
        >>> pipe, params = FlaxStableDiffusionControlNetPipeline.from_pretrained(
        ...     "runwayml/stable-diffusion-v1-5", controlnet=controlnet, revision="flax", dtype=jnp.float32
        ... )
        >>> params["controlnet"] = controlnet_params

        >>> num_samples = jax.device_count()
        >>> rng = jax.random.split(rng, jax.device_count())

        >>> prompt_ids = pipe.prepare_text_inputs([prompts] * num_samples)
        >>> negative_prompt_ids = pipe.prepare_text_inputs([negative_prompts] * num_samples)
        >>> processed_image = pipe.prepare_image_inputs([canny_image] * num_samples)

        >>> p_params = replicate(params)
        >>> prompt_ids = shard(prompt_ids)
        >>> negative_prompt_ids = shard(negative_prompt_ids)
        >>> processed_image = shard(processed_image)

        >>> output = pipe(
        ...     prompt_ids=prompt_ids,
        ...     image=processed_image,
        ...     params=p_params,
        ...     prng_seed=rng,
        ...     num_inference_steps=50,
        ...     neg_prompt_ids=negative_prompt_ids,
        ...     jit=True,
        ... ).images

        >>> output_images = pipe.numpy_to_pil(np.asarray(output.reshape((num_samples,) + output.shape[-3:])))
        >>> output_images = make_image_grid(output_images, num_samples // 4, 4)
        >>> output_images.save("generated_image.png")
        ```
"""


class FlaxStableDiffusionControlNetPipeline(FlaxDiffusionPipeline):
    r"""
    Flax-based pipeline for text-to-image generation using Stable Diffusion with ControlNet Guidance.

    This model inherits from [`FlaxDiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Args:
        vae ([`FlaxAutoencoderKL`]):
            Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations.
        text_encoder ([`~transformers.FlaxCLIPTextModel`]):
            Frozen text-encoder ([clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)).
        tokenizer ([`~transformers.CLIPTokenizer`]):
            A `CLIPTokenizer` to tokenize text.
        unet ([`FlaxUNet2DConditionModel`]):
            A `FlaxUNet2DConditionModel` to denoise the encoded image latents.
        controlnet ([`FlaxControlNetModel`]:
            Provides additional conditioning to the `unet` during the denoising process.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`FlaxDDIMScheduler`], [`FlaxLMSDiscreteScheduler`], [`FlaxPNDMScheduler`], or
            [`FlaxDPMSolverMultistepScheduler`].
        safety_checker ([`FlaxStableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please refer to the [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) for more details
            about a model's potential harms.
        feature_extractor ([`~transformers.CLIPImageProcessor`]):
            A `CLIPImageProcessor` to extract features from generated images; used as inputs to the `safety_checker`.
    """

    def __init__(
        self,
        vae: FlaxAutoencoderKL,
        text_encoder: FlaxCLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: FlaxUNet2DConditionModel,
        controlnet: FlaxControlNetModel,
        scheduler: Union[
            FlaxDDIMScheduler, FlaxPNDMScheduler, FlaxLMSDiscreteScheduler, FlaxDPMSolverMultistepScheduler
        ],
        safety_checker: FlaxStableDiffusionSafetyChecker,
        feature_extractor: CLIPFeatureExtractor,
        dtype: jnp.dtype = jnp.float32,
    ):
        super().__init__()
        self.dtype = dtype

        if safety_checker is None:
            logger.warning(
                f"You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`. Ensure"
                " that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered"
                " results in services or applications open to the public. Both the diffusers team and Hugging Face"
                " strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling"
                " it only for use-cases that involve analyzing network behavior or auditing its results. For more"
                " information, please have a look at https://github.com/huggingface/diffusers/pull/254 ."
            )

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            controlnet=controlnet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

    def prepare_text_inputs(self, prompt: Union[str, List[str]]):
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

    def prepare_image_inputs(self, image: Union[Image.Image, List[Image.Image]]):
        if not isinstance(image, (Image.Image, list)):
            raise ValueError(f"image has to be of type `PIL.Image.Image` or list but is {type(image)}")

        if isinstance(image, Image.Image):
            image = [image]

        processed_images = jnp.concatenate([preprocess(img, jnp.float32) for img in image])

        return processed_images

    def _get_has_nsfw_concepts(self, features, params):
        has_nsfw_concepts = self.safety_checker(features, params)
        return has_nsfw_concepts

    def _run_safety_checker(self, images, safety_model_params, jit=False):
        # safety_model_params should already be replicated when jit is True
        pil_images = [Image.fromarray(image) for image in images]
        features = self.feature_extractor(pil_images, return_tensors="np").pixel_values

        if jit:
            features = shard(features)
            has_nsfw_concepts = _p_get_has_nsfw_concepts(self, features, safety_model_params)
            has_nsfw_concepts = unshard(has_nsfw_concepts)
            safety_model_params = unreplicate(safety_model_params)
        else:
            has_nsfw_concepts = self._get_has_nsfw_concepts(features, safety_model_params)

        images_was_copied = False
        for idx, has_nsfw_concept in enumerate(has_nsfw_concepts):
            if has_nsfw_concept:
                if not images_was_copied:
                    images_was_copied = True
                    images = images.copy()

                images[idx] = np.zeros(images[idx].shape, dtype=np.uint8)  # black image

            if any(has_nsfw_concepts):
                warnings.warn(
                    "Potential NSFW content was detected in one or more images. A black image will be returned"
                    " instead. Try again with a different prompt and/or seed."
                )

        return images, has_nsfw_concepts

    def _generate(
        self,
        prompt_ids: jnp.ndarray,
        image: jnp.ndarray,
        params: Union[Dict, FrozenDict],
        prng_seed: jax.Array,
        num_inference_steps: int,
        guidance_scale: float,
        latents: Optional[jnp.ndarray] = None,
        neg_prompt_ids: Optional[jnp.ndarray] = None,
        controlnet_conditioning_scale: float = 1.0,
    ):
        height, width = image.shape[-2:]
        if height % 64 != 0 or width % 64 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 64 but are {height} and {width}.")

        # get prompt text embeddings
        prompt_embeds = self.text_encoder(prompt_ids, params=params["text_encoder"])[0]

        # TODO: currently it is assumed `do_classifier_free_guidance = guidance_scale > 1.0`
        # implement this conditional `do_classifier_free_guidance = guidance_scale > 1.0`
        batch_size = prompt_ids.shape[0]

        max_length = prompt_ids.shape[-1]

        if neg_prompt_ids is None:
            uncond_input = self.tokenizer(
                [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="np"
            ).input_ids
        else:
            uncond_input = neg_prompt_ids
        negative_prompt_embeds = self.text_encoder(uncond_input, params=params["text_encoder"])[0]
        context = jnp.concatenate([negative_prompt_embeds, prompt_embeds])

        image = jnp.concatenate([image] * 2)

        latents_shape = (
            batch_size,
            self.unet.config.in_channels,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
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

            latents_input = self.scheduler.scale_model_input(scheduler_state, latents_input, t)

            down_block_res_samples, mid_block_res_sample = self.controlnet.apply(
                {"params": params["controlnet"]},
                jnp.array(latents_input),
                jnp.array(timestep, dtype=jnp.int32),
                encoder_hidden_states=context,
                controlnet_cond=image,
                conditioning_scale=controlnet_conditioning_scale,
                return_dict=False,
            )

            # predict the noise residual
            noise_pred = self.unet.apply(
                {"params": params["unet"]},
                jnp.array(latents_input),
                jnp.array(timestep, dtype=jnp.int32),
                encoder_hidden_states=context,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
            ).sample

            # perform guidance
            noise_pred_uncond, noise_prediction_text = jnp.split(noise_pred, 2, axis=0)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents, scheduler_state = self.scheduler.step(scheduler_state, noise_pred, t, latents).to_tuple()
            return latents, scheduler_state

        scheduler_state = self.scheduler.set_timesteps(
            params["scheduler"], num_inference_steps=num_inference_steps, shape=latents_shape
        )

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * params["scheduler"].init_noise_sigma

        if DEBUG:
            # run with python for loop
            for i in range(num_inference_steps):
                latents, scheduler_state = loop_body(i, (latents, scheduler_state))
        else:
            latents, _ = jax.lax.fori_loop(0, num_inference_steps, loop_body, (latents, scheduler_state))

        # scale and decode the image latents with vae
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.apply({"params": params["vae"]}, latents, method=self.vae.decode).sample

        image = (image / 2 + 0.5).clip(0, 1).transpose(0, 2, 3, 1)
        return image

    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt_ids: jnp.ndarray,
        image: jnp.ndarray,
        params: Union[Dict, FrozenDict],
        prng_seed: jax.Array,
        num_inference_steps: int = 50,
        guidance_scale: Union[float, jnp.ndarray] = 7.5,
        latents: jnp.ndarray = None,
        neg_prompt_ids: jnp.ndarray = None,
        controlnet_conditioning_scale: Union[float, jnp.ndarray] = 1.0,
        return_dict: bool = True,
        jit: bool = False,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt_ids (`jnp.ndarray`):
                The prompt or prompts to guide the image generation.
            image (`jnp.ndarray`):
                Array representing the ControlNet input condition to provide guidance to the `unet` for generation.
            params (`Dict` or `FrozenDict`):
                Dictionary containing the model parameters/weights.
            prng_seed (`jax.Array`):
                Array containing random number generator key.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            latents (`jnp.ndarray`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                array is generated by sampling using the supplied random `generator`.
            controlnet_conditioning_scale (`float` or `jnp.ndarray`, *optional*, defaults to 1.0):
                The outputs of the ControlNet are multiplied by `controlnet_conditioning_scale` before they are added
                to the residual in the original `unet`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.FlaxStableDiffusionPipelineOutput`] instead of
                a plain tuple.
            jit (`bool`, defaults to `False`):
                Whether to run `pmap` versions of the generation and safety scoring functions.

                    <Tip warning={true}>

                    This argument exists because `__call__` is not yet end-to-end pmap-able. It will be removed in a
                    future release.

                    </Tip>

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.FlaxStableDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.FlaxStableDiffusionPipelineOutput`] is
                returned, otherwise a `tuple` is returned where the first element is a list with the generated images
                and the second element is a list of `bool`s indicating whether the corresponding generated image
                contains "not-safe-for-work" (nsfw) content.
        """

        height, width = image.shape[-2:]

        if isinstance(guidance_scale, float):
            # Convert to a tensor so each device gets a copy. Follow the prompt_ids for
            # shape information, as they may be sharded (when `jit` is `True`), or not.
            guidance_scale = jnp.array([guidance_scale] * prompt_ids.shape[0])
            if len(prompt_ids.shape) > 2:
                # Assume sharded
                guidance_scale = guidance_scale[:, None]

        if isinstance(controlnet_conditioning_scale, float):
            # Convert to a tensor so each device gets a copy. Follow the prompt_ids for
            # shape information, as they may be sharded (when `jit` is `True`), or not.
            controlnet_conditioning_scale = jnp.array([controlnet_conditioning_scale] * prompt_ids.shape[0])
            if len(prompt_ids.shape) > 2:
                # Assume sharded
                controlnet_conditioning_scale = controlnet_conditioning_scale[:, None]

        if jit:
            images = _p_generate(
                self,
                prompt_ids,
                image,
                params,
                prng_seed,
                num_inference_steps,
                guidance_scale,
                latents,
                neg_prompt_ids,
                controlnet_conditioning_scale,
            )
        else:
            images = self._generate(
                prompt_ids,
                image,
                params,
                prng_seed,
                num_inference_steps,
                guidance_scale,
                latents,
                neg_prompt_ids,
                controlnet_conditioning_scale,
            )

        if self.safety_checker is not None:
            safety_params = params["safety_checker"]
            images_uint8_casted = (images * 255).round().astype("uint8")
            num_devices, batch_size = images.shape[:2]

            images_uint8_casted = np.asarray(images_uint8_casted).reshape(num_devices * batch_size, height, width, 3)
            images_uint8_casted, has_nsfw_concept = self._run_safety_checker(images_uint8_casted, safety_params, jit)
            images = np.array(images)

            # block images
            if any(has_nsfw_concept):
                for i, is_nsfw in enumerate(has_nsfw_concept):
                    if is_nsfw:
                        images[i] = np.asarray(images_uint8_casted[i])

            images = images.reshape(num_devices, batch_size, height, width, 3)
        else:
            images = np.asarray(images)
            has_nsfw_concept = False

        if not return_dict:
            return (images, has_nsfw_concept)

        return FlaxStableDiffusionPipelineOutput(images=images, nsfw_content_detected=has_nsfw_concept)


# Static argnums are pipe, num_inference_steps. A change would trigger recompilation.
# Non-static args are (sharded) input tensors mapped over their first dimension (hence, `0`).
@partial(
    jax.pmap,
    in_axes=(None, 0, 0, 0, 0, None, 0, 0, 0, 0),
    static_broadcasted_argnums=(0, 5),
)
def _p_generate(
    pipe,
    prompt_ids,
    image,
    params,
    prng_seed,
    num_inference_steps,
    guidance_scale,
    latents,
    neg_prompt_ids,
    controlnet_conditioning_scale,
):
    return pipe._generate(
        prompt_ids,
        image,
        params,
        prng_seed,
        num_inference_steps,
        guidance_scale,
        latents,
        neg_prompt_ids,
        controlnet_conditioning_scale,
    )


@partial(jax.pmap, static_broadcasted_argnums=(0,))
def _p_get_has_nsfw_concepts(pipe, features, params):
    return pipe._get_has_nsfw_concepts(features, params)


def unshard(x: jnp.ndarray):
    # einops.rearrange(x, 'd b ... -> (d b) ...')
    num_devices, batch_size = x.shape[:2]
    rest = x.shape[2:]
    return x.reshape(num_devices * batch_size, *rest)


def preprocess(image, dtype):
    image = image.convert("RGB")
    w, h = image.size
    w, h = (x - x % 64 for x in (w, h))  # resize to integer multiple of 64
    image = image.resize((w, h), resample=PIL_INTERPOLATION["lanczos"])
    image = jnp.array(image).astype(dtype) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    return image
