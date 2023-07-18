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

from typing import List, Union, Dict, Optional
from flax.core.frozen_dict import FrozenDict
from transformers import CLIPTokenizer, FlaxCLIPTextModel
import jax.numpy as jnp
import jax

from ...models import FlaxAutoencoderKL, FlaxUNet2DConditionModel
from ...schedulers import (
    FlaxDDIMScheduler,
    FlaxDPMSolverMultistepScheduler,
    FlaxLMSDiscreteScheduler,
    FlaxPNDMScheduler,
)
from ..pipeline_flax_utils import FlaxDiffusionPipeline
from diffusers.utils import logging

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
            scheduler=scheduler
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

    def prepare_inputs(self, prompt: Union[str, List[str]]):
        if not isinstance(prompt, (str, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if self.tokenizer is not None:
            assert self.tokenizer_2 is not None
            tokenizers = [self.tokenizer, self.tokenizer_2]
        else:
            tokenizers = [self.tokenizer_2]
        inputs = []
        for tokenizer in enumerate(tokenizers):
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="np"
            )
            inputs.append(text_inputs.input_ids)
        inputs = jnp.stack(inputs)
        return inputs

    def __call__(
        self,
        prompt_ids: jax.Array,
        params: Union[Dict, FrozenDict],
        prng_seed: jax.random.KeyArray,
        num_inference_steps: int = 50,
        guidance_scale: Union[float, jax.Array] = 7.5,
        height: Optional[int] = None,
        width: Optional[int] = None,
    ):

        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        do_classifier_free_guidance = guidance_scale > 1.0

        if isinstance(guidance_scale, float):
            # Convert to a tensor so each device gets a copy. Follow the prompt_ids for
            # shape information, as they may be sharded (when `jit` is `True`), or not.
            guidance_scale = jnp.array([guidance_scale] * prompt_ids.shape[0])
            if len(prompt_ids.shape) > 2:
                # Assume sharded
                guidance_scale = guidance_scale[:, None]

        # TODO: support jit
        images = self._generate(
            prompt_ids,
            params,
            prng_seed,
            num_inference_steps,
            height,
            width,
            guidance_scale,
            do_classifier_free_guidance
        )

    def get_embeddings(self, prompt_ids: jax.Array, params: Union[Dict, FrozenDict]):
        if prompt_ids.shape[0] == 2:
            # using both CLIP models
            prompt_embeds = self.text_encoder(prompt_ids[0], params=params['text_encoder'], output_hidden_states=True)
            prompt_embeds = prompt_embeds['hidden_states'][-2]
            prompt_embeds_2_out = self.text_encoder_2(prompt_ids[1], params=params['text_encoder_2'], output_hidden_states=True)
            prompt_embeds_2 = prompt_embeds_2_out['hidden_states'][-2]
        else:
            prompt_embeds = jnp.array([])  # dummy embedding for first CLIP model
            prompt_embeds_2_out = self.text_encoder_2(prompt_ids[1], params=params['text_encoder_2'], output_hidden_states=True)
            prompt_embeds_2 = prompt_embeds_2_out['hidden_states'][-2]
        pooled_embeds = prompt_embeds_2_out['pooler_output']  # use second text encoder's pooled output
        prompt_embeds = jnp.concatenate([prompt_embeds, prompt_embeds_2], axis=-1)
        return prompt_embeds, pooled_embeds

    def _get_add_time_ids(self, original_size, crops_coords_top_left, target_size, dtype):
        add_time_ids = list(original_size + crops_coords_top_left + target_size)  # TODO: This is currently not jit'able - probably need pass add_time_ids as input to __call__
        add_time_ids = jnp.array([add_time_ids], dtype=dtype)
        return add_time_ids

    def _generate(
        self,
        prompt_ids: jax.Array,
        params: Union[Dict, FrozenDict],
        prng_seed: jax.random.KeyArray,
        num_inference_steps: int,
        height: int,
        width: int,
        guidance_scale: float = 7.5,
        do_classifier_free_guidance: bool = True,
        latents: Optional[jax.Array] = None,
        neg_prompt_ids: Optional[jax.Array] = None,
        original_size: tuple = (1024, 1024),
        crops_coords_top_left: tuple = (0, 0),
        target_size: tuple = (1024, 1024),
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        # 1. Encode input prompt
        prompt_embeds, pooled_embeds = self.get_embeddings(prompt_ids, params)

        # 2. Get unconditional embeddings
        batch_size = prompt_embeds.shape[0]
        if neg_prompt_ids is None:
            neg_prompt_ids = self.prepare_inputs([""] * batch_size)
        # TODO: properly support without classifier guidance here (or drop support entirely)
        neg_prompt_embeds, pooled_neg_embeds = self.get_embeddings(neg_prompt_ids, params)
        context = jnp.concatenate([neg_prompt_embeds, prompt_embeds], axis=0)  # (2, 77, 2048)

        # 3. Create random latents
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
        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * params["scheduler"].init_noise_sigma

        # 4. Prepare scheduler state
        scheduler_state = self.scheduler.set_timesteps(
            params["scheduler"], num_inference_steps=num_inference_steps, shape=latents.shape
        )

        # 5. Prepare added embeddings
        add_text_embeds = prompt_embeds
        add_time_ids = self._get_add_time_ids(
            original_size, crops_coords_top_left, target_size, dtype=prompt_embeds.dtype
        )

        # 6. Denoising loop
        def loop_body(step, args):
            # TODO
            pass

        if DEBUG:
            # run with python for loop
            for i in range(num_inference_steps):
                latents, scheduler_state = loop_body(i, (latents, scheduler_state))
        else:
            latents, _ = jax.lax.fori_loop(0, num_inference_steps, loop_body, (latents, scheduler_state))

        # 7. Deocde latents
        # TODO
