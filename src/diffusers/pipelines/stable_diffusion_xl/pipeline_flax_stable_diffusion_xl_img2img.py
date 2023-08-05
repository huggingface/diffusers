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

from functools import partial
from typing import Dict, List, Optional, Union
from PIL import Image

import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict
from transformers import CLIPTokenizer, FlaxCLIPTextModel

from diffusers.utils import logging

from ...models import FlaxAutoencoderKL, FlaxUNet2DConditionModel
from ...schedulers import (
    FlaxDDIMScheduler,
    FlaxDPMSolverMultistepScheduler,
    FlaxLMSDiscreteScheduler,
    FlaxPNDMScheduler,
)
from ...utils import PIL_INTERPOLATION, logging, replace_example_docstring
from ..pipeline_flax_utils import FlaxDiffusionPipeline
from . import FlaxStableDiffusionXLPipelineOutput


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# Set to True to use python for loop instead of jax.fori_loop for easier debugging
DEBUG = False


class FlaxStableDiffusionXLImg2ImgPipeline(FlaxDiffusionPipeline):
    # ignore_for_config = ["dtype", "requires_aesthetics_score"]

    def __init__(
        self,
        text_encoder_2: FlaxCLIPTextModel,
        vae: FlaxAutoencoderKL,
        tokenizer_2: CLIPTokenizer,
        unet: FlaxUNet2DConditionModel,
        scheduler: Union[
            FlaxDDIMScheduler, FlaxPNDMScheduler, FlaxLMSDiscreteScheduler, FlaxDPMSolverMultistepScheduler
        ],
        requires_aesthetics_score: bool = False,
        # force_zeros_for_empty_prompt: bool = True,
        dtype: jnp.dtype = jnp.float32,
    ):
        super().__init__()
        self.dtype = dtype

        # tokenizer, text_encoder are null in the refiner
        self.register_modules(
            vae=vae,
            text_encoder_2=text_encoder_2,
            tokenizer_2=tokenizer_2,
            unet=unet,
            scheduler=scheduler,
        )
        self.register_to_config(requires_aesthetics_score=requires_aesthetics_score)
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

    def prepare_text_inputs(self, prompt: Union[str, List[str]]):
        if not isinstance(prompt, (str, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        input_ids = self.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=self.tokenizer_2.model_max_length,
            truncation=True,
            return_tensors="np",
        ).input_ids

        # Introduce an axis for consistency with FlaxStableDiffusionXLPipeline
        return input_ids[:, jnp.newaxis, :]

    def prepare_inputs(self, prompt: Union[str, List[str]], image: Union[Image.Image, List[Image.Image]]):
        if not isinstance(image, (Image.Image, list)):
            raise ValueError(f"image has to be of type `PIL.Image.Image` or list but is {type(image)}")

        if isinstance(image, Image.Image):
            image = [image]

        processed_images = jnp.concatenate([preprocess(img, jnp.float32) for img in image])
        text_input_ids = self.prepare_text_inputs(prompt)
        return text_input_ids, processed_images

    def __call__(
        self,
        prompt_ids: jax.Array,
        image: jnp.array,
        params: Union[Dict, FrozenDict],
        prng_seed: jax.random.KeyArray,
        strength: float = 0.3,
        num_inference_steps: int = 50,
        height: Optional[int] = None,
        width: Optional[int] = None,
        guidance_scale: Union[float, jax.Array] = 7.5,
        noise: jnp.array = None,
        neg_prompt_ids: jnp.array = None,
        aesthetic_score: float = 6.0,
        negative_aesthetic_score: float = 2.5,
        return_dict: bool = True,
        jit: bool = False,
    ):
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        if isinstance(guidance_scale, float) and jit:
            # Convert to a tensor so each device gets a copy.
            guidance_scale = jnp.array([guidance_scale] * prompt_ids.shape[0])
            guidance_scale = guidance_scale[:, None]

        start_timestep = self.get_timestep_start(num_inference_steps, strength)

        if jit:
            images = _p_generate(
                self,
                prompt_ids,
                image,
                params,
                prng_seed,
                start_timestep,
                num_inference_steps,
                height,
                width,
                guidance_scale,
                noise,
                neg_prompt_ids,
                aesthetic_score,
                negative_aesthetic_score,
            )
        else:
            images = self._generate(
                prompt_ids,
                image,
                params,
                prng_seed,
                start_timestep,
                num_inference_steps,
                height,
                width,
                guidance_scale,
                noise,
                neg_prompt_ids,
                aesthetic_score,
                negative_aesthetic_score,
            )

        if not return_dict:
            return (images,)

        return FlaxStableDiffusionXLPipelineOutput(images=images)

    def get_embeddings(self, prompt_ids: jnp.array, params):
        # bs, encoder_input, seq_length
        te_inputs = prompt_ids[:, 0, :]

        prompt_embeds_2_out = self.text_encoder_2(
            te_inputs, params=params["text_encoder_2"], output_hidden_states=True
        )
        text_embeds = prompt_embeds_2_out["text_embeds"]
        prompt_embeds = prompt_embeds_2_out["hidden_states"][-2]
        return prompt_embeds, text_embeds
    
    def _get_add_time_ids(
        self, original_size, crops_coords_top_left, target_size, aesthetic_score, negative_aesthetic_score, bs, dtype
    ):
        if self.config.requires_aesthetics_score:
            add_time_ids = list(original_size + crops_coords_top_left + (aesthetic_score,))
            add_neg_time_ids = list(original_size + crops_coords_top_left + (negative_aesthetic_score,))
        else:
            add_time_ids = list(original_size + crops_coords_top_left + target_size)
            add_neg_time_ids = list(original_size + crops_coords_top_left + target_size)

        passed_add_embed_dim = (
            self.unet.config.addition_time_embed_dim * len(add_time_ids) + self.text_encoder_2.config.projection_dim
        )
        # TODO: verify (params["unet"]["add_embedding"]["linear_1"]["kernel"].shape[0] ?)
        # expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features

        # if (
        #     expected_add_embed_dim > passed_add_embed_dim
        #     and (expected_add_embed_dim - passed_add_embed_dim) == self.unet.config.addition_time_embed_dim
        # ):
        #     raise ValueError(
        #         f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. Please make sure to enable `requires_aesthetics_score` with `pipe.register_to_config(requires_aesthetics_score=True)` to make sure `aesthetic_score` {aesthetic_score} and `negative_aesthetic_score` {negative_aesthetic_score} is correctly used by the model."
        #     )
        # elif (
        #     expected_add_embed_dim < passed_add_embed_dim
        #     and (passed_add_embed_dim - expected_add_embed_dim) == self.unet.config.addition_time_embed_dim
        # ):
        #     raise ValueError(
        #         f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. Please make sure to disable `requires_aesthetics_score` with `pipe.register_to_config(requires_aesthetics_score=False)` to make sure `target_size` {target_size} is correctly used by the model."
        #     )
        # elif expected_add_embed_dim != passed_add_embed_dim:
        #     raise ValueError(
        #         f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
        #     )

        add_time_ids = jnp.array([add_time_ids] * bs, dtype=dtype)
        add_neg_time_ids = jnp.array([add_neg_time_ids] * bs, dtype=dtype)

        return add_time_ids, add_neg_time_ids
    
    def get_timestep_start(self, num_inference_steps, strength):
        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)

        return t_start


    def _generate(
        self,
        prompt_ids: jnp.array,
        image: jnp.array,
        params: Union[Dict, FrozenDict],
        prng_seed: jax.random.KeyArray,
        start_timestep: int,
        num_inference_steps: int,
        height: int,
        width: int,
        guidance_scale: float,
        noise: Optional[jnp.array] = None,
        neg_prompt_ids: Optional[jnp.array] = None,
        aesthetic_score: float = 6.0,
        negative_aesthetic_score: float = 2.5,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        # 1. Encode input prompt
        prompt_embeds, pooled_embeds = self.get_embeddings(prompt_ids, params)

        # 2. Get unconditional embeddings
        batch_size = prompt_embeds.shape[0]
        if neg_prompt_ids is None:
            neg_prompt_ids = self.prepare_text_inputs([""] * batch_size)

        neg_prompt_embeds, negative_pooled_embeds = self.get_embeddings(neg_prompt_ids, params)

        add_time_ids, add_neg_time_ids = self._get_add_time_ids(
            (height, width),
            (0, 0),
            (height, width),
            aesthetic_score,
            negative_aesthetic_score,
            prompt_embeds.shape[0],
            dtype=prompt_embeds.dtype,
        )

        prompt_embeds = jnp.concatenate([neg_prompt_embeds, prompt_embeds], axis=0)
        add_text_embeds = jnp.concatenate([negative_pooled_embeds, pooled_embeds], axis=0)
        add_time_ids = jnp.concatenate([add_neg_time_ids, add_time_ids], axis=0)

        latents_shape = (
            batch_size,
            self.unet.config.in_channels,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        if noise is None:
            noise = jax.random.normal(prng_seed, shape=latents_shape, dtype=jnp.float32)
        else:
            if noise.shape != latents_shape:
                raise ValueError(f"Unexpected latents shape, got {noise.shape}, expected {latents_shape}")

        if image.shape[1] == 4:
            # Skip encoding if using latents as input
            init_latents = image
        else:
            # Create init_latents
            init_latent_dist = self.vae.apply({"params": params["vae"]}, image, method=self.vae.encode).latent_dist
            init_latents = init_latent_dist.sample(key=prng_seed).transpose((0, 3, 1, 2))
            init_latents = self.vae.config.scaling_factor * init_latents

        scheduler_state = self.scheduler.set_timesteps(
            params["scheduler"], num_inference_steps=num_inference_steps, shape=latents_shape
        )

        latent_timestep = scheduler_state.timesteps[start_timestep : start_timestep + 1].repeat(batch_size)
        latents = self.scheduler.add_noise(params["scheduler"], init_latents, noise, latent_timestep)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * params["scheduler"].init_noise_sigma

        # Ensure model output will be `float32` before going into the scheduler
        guidance_scale = jnp.array([guidance_scale], dtype=jnp.float32)

        added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}

        # 6. Denoising loop
        def loop_body(step, args):
            latents, scheduler_state = args
            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            latents_input = jnp.concatenate([latents] * 2)

            t = jnp.array(scheduler_state.timesteps, dtype=jnp.int32)[step]
            timestep = jnp.broadcast_to(t, latents_input.shape[0])

            latents_input = self.scheduler.scale_model_input(scheduler_state, latents_input, t)

            # predict the noise residual
            noise_pred = self.unet.apply(
                {"params": params["unet"]},
                jnp.array(latents_input),
                jnp.array(timestep, dtype=jnp.int32),
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs=added_cond_kwargs,
            ).sample
            # perform guidance
            noise_pred_uncond, noise_prediction_text = jnp.split(noise_pred, 2, axis=0)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents, scheduler_state = self.scheduler.step(scheduler_state, noise_pred, t, latents).to_tuple()
            return latents, scheduler_state

        if DEBUG:
            # run with python for loop
            for i in range(start_timestep, num_inference_steps):
                latents, scheduler_state = loop_body(i, (latents, scheduler_state))
        else:
            latents, _ = jax.lax.fori_loop(start_timestep, num_inference_steps, loop_body, (latents, scheduler_state))

        # 7. Decode latents
        # scale and decode the image latents with vae
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.apply({"params": params["vae"]}, latents, method=self.vae.decode).sample

        image = (image / 2 + 0.5).clip(0, 1).transpose(0, 2, 3, 1)
        return image


# Static argnums are pipe, start_timestep, num_inference_steps, height, width, aesthetic_score, negative_aesthetic_score.
# A change would trigger recompilation.
# Non-static args are (sharded) input tensors mapped over their first dimension (hence, `0`).
@partial(
    jax.pmap,
    in_axes=(None, 0, 0, 0, 0, None, None, None, None, 0, 0, 0, None, None),
    static_broadcasted_argnums=(0, 5, 6, 7, 8, 12, 13),
)
def _p_generate(
    pipe,
    prompt_ids,
    image,
    params,
    prng_seed,
    start_timestep,
    num_inference_steps,
    height,
    width,
    guidance_scale,
    noise,
    neg_prompt_ids,
    aesthetic_score,
    negative_aesthetic_score,
):
    return pipe._generate(
        prompt_ids,
        image,
        params,
        prng_seed,
        start_timestep,
        num_inference_steps,
        height,
        width,
        guidance_scale,
        noise,
        neg_prompt_ids,
        aesthetic_score,
        negative_aesthetic_score,
    )

def preprocess(image, dtype):
    w, h = image.size
    w, h = (x - x % 32 for x in (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=PIL_INTERPOLATION["lanczos"])
    image = jnp.array(image).astype(dtype) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    return 2.0 * image - 1.0
