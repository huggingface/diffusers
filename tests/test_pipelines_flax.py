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
    from diffusers import FlaxStableDiffusionPipeline
    from flax.jax_utils import replicate
    from flax.training.common_utils import shard
    from jax import pmap


@require_flax
@slow
class FlaxPipelineTests(unittest.TestCase):
    def test_dummy_all_tpus(self):
        pipeline, params = FlaxStableDiffusionPipeline.from_pretrained(
            "hf-internal-testing/tiny-stable-diffusion-pipe"
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
        images_pil = pipeline.numpy_to_pil(np.asarray(images.reshape((num_samples,) + images.shape[-3:])))

        assert len(images_pil) == 8
