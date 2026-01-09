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

from functools import partial
from typing import Dict, List, Optional, Union

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
from ..pipeline_flax_utils import FlaxDiffusionPipeline
from .pipeline_output import FlaxStableDiffusionXLPipelineOutput


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# Set to True to use python for loop instead of jax.fori_loop for easier debugging
DEBUG = False


class FlaxStableDiffusionXLPipeline(FlaxDiffusionPipeline):
    def __init__(
        self,
        text_encoder: FlaxCLIPTextModel,
        text_encoder_2: FlaxCLIPTextModel,
        vae: FlaxAutoencoderKL,
        tokenizer: CLIPTokenizer,
        tokenizer_2: CLIPTokenizer,
        unet: FlaxUNet2DConditionModel,
        scheduler: Union[
            FlaxDDIMScheduler, FlaxPNDMScheduler, FlaxLMSDiscreteScheduler, FlaxDPMSolverMultistepScheduler
        ],
        dtype: jnp.dtype = jnp.float32,
    ):
        super().__init__()
        self.dtype = dtype

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            unet=unet,
            scheduler=scheduler,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1) if getattr(self, "vae", None) else 8

    def prepare_inputs(self, prompt: Union[str, List[str]]):
        if not isinstance(prompt, (str, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        # Assume we have the two encoders
        inputs = []
        for tokenizer in [self.tokenizer, self.tokenizer_2]:
            text_inputs = tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="np",
            )
            inputs.append(text_inputs.input_ids)
        inputs = jnp.stack(inputs, axis=1)
        return inputs

    def __call__(
        self,
        prompt_ids: jax.Array,
        params: Union[Dict, FrozenDict],
        prng_seed: jax.Array,
        num_inference_steps: int = 50,
        guidance_scale: Union[float, jax.Array] = 7.5,
        height: Optional[int] = None,
        width: Optional[int] = None,
        latents: jnp.array = None,
        neg_prompt_ids: jnp.array = None,
        return_dict: bool = True,
        output_type: str = None,
        jit: bool = False,
    ):
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        if isinstance(guidance_scale, float) and jit:
            # Convert to a tensor so each device gets a copy.
            guidance_scale = jnp.array([guidance_scale] * prompt_ids.shape[0])
            guidance_scale = guidance_scale[:, None]

        return_latents = output_type == "latent"

        if jit:
            images = _p_generate(
                self,
                prompt_ids,
                params,
                prng_seed,
                num_inference_steps,
                height,
                width,
                guidance_scale,
                latents,
                neg_prompt_ids,
                return_latents,
            )
        else:
            images = self._generate(
                prompt_ids,
                params,
                prng_seed,
                num_inference_steps,
                height,
                width,
                guidance_scale,
                latents,
                neg_prompt_ids,
                return_latents,
            )

        if not return_dict:
            return (images,)

        return FlaxStableDiffusionXLPipelineOutput(images=images)

    def get_embeddings(self, prompt_ids: jnp.array, params):
        # We assume we have the two encoders

        # bs, encoder_input, seq_length
        te_1_inputs = prompt_ids[:, 0, :]
        te_2_inputs = prompt_ids[:, 1, :]

        prompt_embeds = self.text_encoder(te_1_inputs, params=params["text_encoder"], output_hidden_states=True)
        prompt_embeds = prompt_embeds["hidden_states"][-2]
        prompt_embeds_2_out = self.text_encoder_2(
            te_2_inputs, params=params["text_encoder_2"], output_hidden_states=True
        )
        prompt_embeds_2 = prompt_embeds_2_out["hidden_states"][-2]
        text_embeds = prompt_embeds_2_out["text_embeds"]
        prompt_embeds = jnp.concatenate([prompt_embeds, prompt_embeds_2], axis=-1)
        return prompt_embeds, text_embeds

    def _get_add_time_ids(self, original_size, crops_coords_top_left, target_size, bs, dtype):
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = jnp.array([add_time_ids] * bs, dtype=dtype)
        return add_time_ids

    def _generate(
        self,
        prompt_ids: jnp.array,
        params: Union[Dict, FrozenDict],
        prng_seed: jax.Array,
        num_inference_steps: int,
        height: int,
        width: int,
        guidance_scale: float,
        latents: Optional[jnp.array] = None,
        neg_prompt_ids: Optional[jnp.array] = None,
        return_latents=False,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        # Encode input prompt
        prompt_embeds, pooled_embeds = self.get_embeddings(prompt_ids, params)

        # Get unconditional embeddings
        batch_size = prompt_embeds.shape[0]
        if neg_prompt_ids is None:
            neg_prompt_embeds = jnp.zeros_like(prompt_embeds)
            negative_pooled_embeds = jnp.zeros_like(pooled_embeds)
        else:
            neg_prompt_embeds, negative_pooled_embeds = self.get_embeddings(neg_prompt_ids, params)

        add_time_ids = self._get_add_time_ids(
            (height, width), (0, 0), (height, width), prompt_embeds.shape[0], dtype=prompt_embeds.dtype
        )

        prompt_embeds = jnp.concatenate([neg_prompt_embeds, prompt_embeds], axis=0)  # (2, 77, 2048)
        add_text_embeds = jnp.concatenate([negative_pooled_embeds, pooled_embeds], axis=0)
        add_time_ids = jnp.concatenate([add_time_ids, add_time_ids], axis=0)

        # Ensure model output will be `float32` before going into the scheduler
        guidance_scale = jnp.array([guidance_scale], dtype=jnp.float32)

        # Create random latents
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

        # Prepare scheduler state
        scheduler_state = self.scheduler.set_timesteps(
            params["scheduler"], num_inference_steps=num_inference_steps, shape=latents.shape
        )

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * scheduler_state.init_noise_sigma

        added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}

        # Denoising loop
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
            for i in range(num_inference_steps):
                latents, scheduler_state = loop_body(i, (latents, scheduler_state))
        else:
            latents, _ = jax.lax.fori_loop(0, num_inference_steps, loop_body, (latents, scheduler_state))

        if return_latents:
            return latents

        # Decode latents
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.apply({"params": params["vae"]}, latents, method=self.vae.decode).sample

        image = (image / 2 + 0.5).clip(0, 1).transpose(0, 2, 3, 1)
        return image


# Static argnums are pipe, num_inference_steps, height, width, return_latents. A change would trigger recompilation.
# Non-static args are (sharded) input tensors mapped over their first dimension (hence, `0`).
@partial(
    jax.pmap,
    in_axes=(None, 0, 0, 0, None, None, None, 0, 0, 0, None),
    static_broadcasted_argnums=(0, 4, 5, 6, 10),
)
def _p_generate(
    pipe,
    prompt_ids,
    params,
    prng_seed,
    num_inference_steps,
    height,
    width,
    guidance_scale,
    latents,
    neg_prompt_ids,
    return_latents,
):
    return pipe._generate(
        prompt_ids,
        params,
        prng_seed,
        num_inference_steps,
        height,
        width,
        guidance_scale,
        latents,
        neg_prompt_ids,
        return_latents,
    )
