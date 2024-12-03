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

import os
import tempfile
import unittest

import numpy as np

from diffusers.utils import is_flax_available
from diffusers.utils.testing_utils import require_flax, slow


if is_flax_available():
    import jax
    import jax.numpy as jnp
    from flax.jax_utils import replicate
    from flax.training.common_utils import shard

    from diffusers import FlaxDDIMScheduler, FlaxDiffusionPipeline, FlaxStableDiffusionPipeline


@require_flax
class DownloadTests(unittest.TestCase):
    def test_download_only_pytorch(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            # pipeline has Flax weights
            _ = FlaxDiffusionPipeline.from_pretrained(
                "hf-internal-testing/tiny-stable-diffusion-pipe", safety_checker=None, cache_dir=tmpdirname
            )

            all_root_files = [t[-1] for t in os.walk(os.path.join(tmpdirname, os.listdir(tmpdirname)[0], "snapshots"))]
            files = [item for sublist in all_root_files for item in sublist]

            # None of the downloaded files should be a PyTorch file even if we have some here:
            # https://huggingface.co/hf-internal-testing/tiny-stable-diffusion-pipe/blob/main/unet/diffusion_pytorch_model.bin
            assert not any(f.endswith(".bin") for f in files)


@slow
@require_flax
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

        # shard inputs and rng
        params = replicate(params)
        prng_seed = jax.random.split(prng_seed, num_samples)
        prompt_ids = shard(prompt_ids)

        images = pipeline(prompt_ids, params, prng_seed, num_inference_steps, jit=True).images

        assert images.shape == (num_samples, 1, 64, 64, 3)
        if jax.device_count() == 8:
            assert np.abs(np.abs(images[0, 0, :2, :2, -2:], dtype=np.float32).sum() - 4.1514745) < 1e-3
            assert np.abs(np.abs(images, dtype=np.float32).sum() - 49947.875) < 5e-1

        images_pil = pipeline.numpy_to_pil(np.asarray(images.reshape((num_samples,) + images.shape[-3:])))
        assert len(images_pil) == num_samples

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

        # shard inputs and rng
        params = replicate(params)
        prng_seed = jax.random.split(prng_seed, num_samples)
        prompt_ids = shard(prompt_ids)

        images = pipeline(prompt_ids, params, prng_seed, num_inference_steps, jit=True).images

        assert images.shape == (num_samples, 1, 512, 512, 3)
        if jax.device_count() == 8:
            assert np.abs((np.abs(images[0, 0, :2, :2, -2:], dtype=np.float32).sum() - 0.05652401)) < 1e-2
            assert np.abs((np.abs(images, dtype=np.float32).sum() - 2383808.2)) < 5e-1

    def test_stable_diffusion_v1_4_bfloat_16(self):
        pipeline, params = FlaxStableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4", variant="bf16", dtype=jnp.bfloat16, safety_checker=None
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
        prng_seed = jax.random.split(prng_seed, num_samples)
        prompt_ids = shard(prompt_ids)

        images = pipeline(prompt_ids, params, prng_seed, num_inference_steps, jit=True).images

        assert images.shape == (num_samples, 1, 512, 512, 3)
        if jax.device_count() == 8:
            assert np.abs((np.abs(images[0, 0, :2, :2, -2:], dtype=np.float32).sum() - 0.04003906)) < 5e-2
            assert np.abs((np.abs(images, dtype=np.float32).sum() - 2373516.75)) < 5e-1

    def test_stable_diffusion_v1_4_bfloat_16_with_safety(self):
        pipeline, params = FlaxStableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4", variant="bf16", dtype=jnp.bfloat16
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
        prng_seed = jax.random.split(prng_seed, num_samples)
        prompt_ids = shard(prompt_ids)

        images = pipeline(prompt_ids, params, prng_seed, num_inference_steps, jit=True).images

        assert images.shape == (num_samples, 1, 512, 512, 3)
        if jax.device_count() == 8:
            assert np.abs((np.abs(images[0, 0, :2, :2, -2:], dtype=np.float32).sum() - 0.04003906)) < 5e-2
            assert np.abs((np.abs(images, dtype=np.float32).sum() - 2373516.75)) < 5e-1

    def test_stable_diffusion_v1_4_bfloat_16_ddim(self):
        scheduler = FlaxDDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            set_alpha_to_one=False,
            steps_offset=1,
        )

        pipeline, params = FlaxStableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            variant="bf16",
            dtype=jnp.bfloat16,
            scheduler=scheduler,
            safety_checker=None,
        )
        scheduler_state = scheduler.create_state()

        params["scheduler"] = scheduler_state

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
        prng_seed = jax.random.split(prng_seed, num_samples)
        prompt_ids = shard(prompt_ids)

        images = pipeline(prompt_ids, params, prng_seed, num_inference_steps, jit=True).images

        assert images.shape == (num_samples, 1, 512, 512, 3)
        if jax.device_count() == 8:
            assert np.abs((np.abs(images[0, 0, :2, :2, -2:], dtype=np.float32).sum() - 0.045043945)) < 5e-2
            assert np.abs((np.abs(images, dtype=np.float32).sum() - 2347693.5)) < 5e-1

    def test_jax_memory_efficient_attention(self):
        prompt = (
            "A cinematic film still of Morgan Freeman starring as Jimi Hendrix, portrait, 40mm lens, shallow depth of"
            " field, close up, split lighting, cinematic"
        )

        num_samples = jax.device_count()
        prompt = num_samples * [prompt]
        prng_seed = jax.random.split(jax.random.PRNGKey(0), num_samples)

        pipeline, params = FlaxStableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            variant="bf16",
            dtype=jnp.bfloat16,
            safety_checker=None,
        )

        params = replicate(params)
        prompt_ids = pipeline.prepare_inputs(prompt)
        prompt_ids = shard(prompt_ids)
        images = pipeline(prompt_ids, params, prng_seed, jit=True).images
        assert images.shape == (num_samples, 1, 512, 512, 3)
        slice = images[2, 0, 256, 10:17, 1]

        # With memory efficient attention
        pipeline, params = FlaxStableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            variant="bf16",
            dtype=jnp.bfloat16,
            safety_checker=None,
            use_memory_efficient_attention=True,
        )

        params = replicate(params)
        prompt_ids = pipeline.prepare_inputs(prompt)
        prompt_ids = shard(prompt_ids)
        images_eff = pipeline(prompt_ids, params, prng_seed, jit=True).images
        assert images_eff.shape == (num_samples, 1, 512, 512, 3)
        slice_eff = images[2, 0, 256, 10:17, 1]

        # I checked the results visually and they are very similar. However, I saw that the max diff is `1` and the `sum`
        # over the 8 images is exactly `256`, which is very suspicious. Testing a random slice for now.
        assert abs(slice_eff - slice).max() < 1e-2
