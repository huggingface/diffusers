# coding=utf-8
# Copyright 2024 HuggingFace Inc.
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

import gc
import unittest

from diffusers import FlaxDPMSolverMultistepScheduler, FlaxStableDiffusionPipeline
from diffusers.utils import is_flax_available
from diffusers.utils.testing_utils import nightly, require_flax


if is_flax_available():
    import jax
    import jax.numpy as jnp
    from flax.jax_utils import replicate
    from flax.training.common_utils import shard


@nightly
@require_flax
class FlaxStableDiffusion2PipelineIntegrationTests(unittest.TestCase):
    def tearDown(self):
        # clean up the VRAM after each test
        super().tearDown()
        gc.collect()

    def test_stable_diffusion_flax(self):
        sd_pipe, params = FlaxStableDiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2",
            variant="bf16",
            dtype=jnp.bfloat16,
        )

        prompt = "A painting of a squirrel eating a burger"
        num_samples = jax.device_count()
        prompt = num_samples * [prompt]
        prompt_ids = sd_pipe.prepare_inputs(prompt)

        params = replicate(params)
        prompt_ids = shard(prompt_ids)

        prng_seed = jax.random.PRNGKey(0)
        prng_seed = jax.random.split(prng_seed, jax.device_count())

        images = sd_pipe(prompt_ids, params, prng_seed, num_inference_steps=25, jit=True)[0]
        assert images.shape == (jax.device_count(), 1, 768, 768, 3)

        images = images.reshape((images.shape[0] * images.shape[1],) + images.shape[-3:])
        image_slice = images[0, 253:256, 253:256, -1]

        output_slice = jnp.asarray(jax.device_get(image_slice.flatten()))
        expected_slice = jnp.array([0.4238, 0.4414, 0.4395, 0.4453, 0.4629, 0.4590, 0.4531, 0.45508, 0.4512])
        print(f"output_slice: {output_slice}")
        assert jnp.abs(output_slice - expected_slice).max() < 1e-2


@nightly
@require_flax
class FlaxStableDiffusion2PipelineNightlyTests(unittest.TestCase):
    def tearDown(self):
        # clean up the VRAM after each test
        super().tearDown()
        gc.collect()

    def test_stable_diffusion_dpm_flax(self):
        model_id = "stabilityai/stable-diffusion-2"
        scheduler, scheduler_params = FlaxDPMSolverMultistepScheduler.from_pretrained(model_id, subfolder="scheduler")
        sd_pipe, params = FlaxStableDiffusionPipeline.from_pretrained(
            model_id,
            scheduler=scheduler,
            variant="bf16",
            dtype=jnp.bfloat16,
        )
        params["scheduler"] = scheduler_params

        prompt = "A painting of a squirrel eating a burger"
        num_samples = jax.device_count()
        prompt = num_samples * [prompt]
        prompt_ids = sd_pipe.prepare_inputs(prompt)

        params = replicate(params)
        prompt_ids = shard(prompt_ids)

        prng_seed = jax.random.PRNGKey(0)
        prng_seed = jax.random.split(prng_seed, jax.device_count())

        images = sd_pipe(prompt_ids, params, prng_seed, num_inference_steps=25, jit=True)[0]
        assert images.shape == (jax.device_count(), 1, 768, 768, 3)

        images = images.reshape((images.shape[0] * images.shape[1],) + images.shape[-3:])
        image_slice = images[0, 253:256, 253:256, -1]

        output_slice = jnp.asarray(jax.device_get(image_slice.flatten()))
        expected_slice = jnp.array([0.4336, 0.42969, 0.4453, 0.4199, 0.4297, 0.4531, 0.4434, 0.4434, 0.4297])
        print(f"output_slice: {output_slice}")
        assert jnp.abs(output_slice - expected_slice).max() < 1e-2
