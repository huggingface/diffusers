# coding=utf-8
# Copyright 2022 HuggingFace Inc.
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

import unittest

import numpy as np

from diffusers.utils import is_flax_available
from diffusers.utils.testing_utils import require_flax, slow


if is_flax_available():
    import jax
    import jax.numpy as jnp
    from diffusers import FlaxStableDiffusionPipeline
    from flax.jax_utils import replicate
    from flax.training.common_utils import shard
    from jax import pmap


@require_flax
@slow
class FlaxPipelineTests(unittest.TestCase):
    def test_dummy_all_tpus(self):
        pipeline, params = FlaxStableDiffusionPipeline.from_pretrained(
            "hf-internal-testing/tiny-stable-diffusion-pipe", safety_checker=None
        )

        prompt = (
            "A cinematic film still of Morgan Freeman starring as Jimi Hendrix, portrait, 40mm lens, shallow depth of"
            " field, close up, split lighting, cinematic"
        )

        prng_seed = jax.random.PRNGKey(0)
        num_inference_steps = 4

        num_samples = jax.device_count()
        prompt = num_samples * [prompt]
        prompt_ids = pipeline.prepare_inputs(prompt)

        p_sample = pmap(pipeline.__call__, static_broadcasted_argnums=(3,))

        # shard inputs and rng
        params = replicate(params)
        prng_seed = jax.random.split(prng_seed, 8)
        prompt_ids = shard(prompt_ids)

        images = p_sample(prompt_ids, params, prng_seed, num_inference_steps).images

        assert images.shape == (8, 1, 64, 64, 3)
        assert np.abs((np.abs(images[0, 0, :2, :2, -2:], dtype=np.float32).sum() - 4.151474)) < 1e-3
        assert np.abs((np.abs(images, dtype=np.float32).sum() - 49947.875)) < 1e-2

        images_pil = pipeline.numpy_to_pil(np.asarray(images.reshape((num_samples,) + images.shape[-3:])))

        assert len(images_pil) == 8

    def test_stable_diffusion_v1_4(self):
        pipeline, params = FlaxStableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4", revision="flax", safety_checker=None
        )

        prompt = (
            "A cinematic film still of Morgan Freeman starring as Jimi Hendrix, portrait, 40mm lens, shallow depth of"
            " field, close up, split lighting, cinematic"
        )

        prng_seed = jax.random.PRNGKey(0)
        num_inference_steps = 50

        num_samples = jax.device_count()
        prompt = num_samples * [prompt]
        prompt_ids = pipeline.prepare_inputs(prompt)

        p_sample = pmap(pipeline.__call__, static_broadcasted_argnums=(3,))

        # shard inputs and rng
        params = replicate(params)
        prng_seed = jax.random.split(prng_seed, 8)
        prompt_ids = shard(prompt_ids)

        images = p_sample(prompt_ids, params, prng_seed, num_inference_steps).images

        images_pil = pipeline.numpy_to_pil(np.asarray(images.reshape((num_samples,) + images.shape[-3:])))
        for i, image in enumerate(images_pil):
            image.save(f"/home/patrick/images/flax-test-{i}_fp32.png")

        assert images.shape == (8, 1, 512, 512, 3)
        assert np.abs((np.abs(images[0, 0, :2, :2, -2:], dtype=np.float32).sum() - 0.05652401)) < 1e-3
        assert np.abs((np.abs(images, dtype=np.float32).sum() - 2383808.2)) < 1e-2

    def test_stable_diffusion_v1_4_bfloat_16(self):
        pipeline, params = FlaxStableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4", revision="bf16", dtype=jnp.bfloat16, safety_checker=None
        )

        prompt = (
            "A cinematic film still of Morgan Freeman starring as Jimi Hendrix, portrait, 40mm lens, shallow depth of"
            " field, close up, split lighting, cinematic"
        )

        prng_seed = jax.random.PRNGKey(0)
        num_inference_steps = 50

        num_samples = jax.device_count()
        prompt = num_samples * [prompt]
        prompt_ids = pipeline.prepare_inputs(prompt)

        p_sample = pmap(pipeline.__call__, static_broadcasted_argnums=(3,))

        # shard inputs and rng
        params = replicate(params)
        prng_seed = jax.random.split(prng_seed, 8)
        prompt_ids = shard(prompt_ids)

        images = p_sample(prompt_ids, params, prng_seed, num_inference_steps).images

        assert images.shape == (8, 1, 512, 512, 3)
        assert np.abs((np.abs(images[0, 0, :2, :2, -2:], dtype=np.float32).sum() - 0.06652832)) < 1e-3
        assert np.abs((np.abs(images, dtype=np.float32).sum() - 2384849.8)) < 1e-2

    def test_stable_diffusion_v1_4_bfloat_16_with_safety(self):
        pipeline, params = FlaxStableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4", revision="bf16", dtype=jnp.bfloat16
        )

        prompt = (
            "A cinematic film still of Morgan Freeman starring as Jimi Hendrix, portrait, 40mm lens, shallow depth of"
            " field, close up, split lighting, cinematic"
        )

        prng_seed = jax.random.PRNGKey(0)
        num_inference_steps = 50

        num_samples = jax.device_count()
        prompt = num_samples * [prompt]
        prompt_ids = pipeline.prepare_inputs(prompt)

        # shard inputs and rng
        params = replicate(params)
        prng_seed = jax.random.split(prng_seed, 8)
        prompt_ids = shard(prompt_ids)

        images = pipeline(prompt_ids, params, prng_seed, num_inference_steps, jit=True).images

        assert images.shape == (8, 1, 512, 512, 3)
        assert np.abs((np.abs(images[0, 0, :2, :2, -2:], dtype=np.float32).sum() - 0.06652832)) < 1e-3
        assert np.abs((np.abs(images, dtype=np.float32).sum() - 2384849.8)) < 1e-2
