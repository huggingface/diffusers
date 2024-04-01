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

from diffusers import FlaxStableDiffusionInpaintPipeline
from diffusers.utils import is_flax_available, load_image
from diffusers.utils.testing_utils import require_flax, slow


if is_flax_available():
    import jax
    import jax.numpy as jnp
    from flax.jax_utils import replicate
    from flax.training.common_utils import shard


@slow
@require_flax
class FlaxStableDiffusionInpaintPipelineIntegrationTests(unittest.TestCase):
    def tearDown(self):
        # clean up the VRAM after each test
        super().tearDown()
        gc.collect()

    def test_stable_diffusion_inpaint_pipeline(self):
        init_image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
            "/sd2-inpaint/init_image.png"
        )
        mask_image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd2-inpaint/mask.png"
        )

        model_id = "xvjiarui/stable-diffusion-2-inpainting"
        pipeline, params = FlaxStableDiffusionInpaintPipeline.from_pretrained(model_id, safety_checker=None)

        prompt = "Face of a yellow cat, high resolution, sitting on a park bench"

        prng_seed = jax.random.PRNGKey(0)
        num_inference_steps = 50

        num_samples = jax.device_count()
        prompt = num_samples * [prompt]
        init_image = num_samples * [init_image]
        mask_image = num_samples * [mask_image]
        prompt_ids, processed_masked_images, processed_masks = pipeline.prepare_inputs(prompt, init_image, mask_image)

        # shard inputs and rng
        params = replicate(params)
        prng_seed = jax.random.split(prng_seed, jax.device_count())
        prompt_ids = shard(prompt_ids)
        processed_masked_images = shard(processed_masked_images)
        processed_masks = shard(processed_masks)

        output = pipeline(
            prompt_ids, processed_masks, processed_masked_images, params, prng_seed, num_inference_steps, jit=True
        )

        images = output.images.reshape(num_samples, 512, 512, 3)

        image_slice = images[0, 253:256, 253:256, -1]

        output_slice = jnp.asarray(jax.device_get(image_slice.flatten()))
        expected_slice = jnp.array(
            [0.3611307, 0.37649736, 0.3757408, 0.38213953, 0.39295167, 0.3841631, 0.41554978, 0.4137475, 0.4217084]
        )
        print(f"output_slice: {output_slice}")
        assert jnp.abs(output_slice - expected_slice).max() < 1e-2
