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

import numpy as np
import torch

from diffusers import AutoencoderKL, DDIMScheduler, DiTPipeline, DiTTransformer2DModel, DPMSolverMultistepScheduler
from diffusers.utils import is_xformers_available
from diffusers.utils.testing_utils import enable_full_determinism, load_numpy, nightly, require_torch_gpu, torch_device

from ..pipeline_params import (
    CLASS_CONDITIONED_IMAGE_GENERATION_BATCH_PARAMS,
    CLASS_CONDITIONED_IMAGE_GENERATION_PARAMS,
)
from ..test_pipelines_common import PipelineTesterMixin


enable_full_determinism()


class DiTPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = DiTPipeline
    params = CLASS_CONDITIONED_IMAGE_GENERATION_PARAMS
    required_optional_params = PipelineTesterMixin.required_optional_params - {
        "latents",
        "num_images_per_prompt",
        "callback",
        "callback_steps",
    }
    batch_params = CLASS_CONDITIONED_IMAGE_GENERATION_BATCH_PARAMS

    def get_dummy_components(self):
        torch.manual_seed(0)
        transformer = DiTTransformer2DModel(
            sample_size=16,
            num_layers=2,
            patch_size=4,
            attention_head_dim=8,
            num_attention_heads=2,
            in_channels=4,
            out_channels=8,
            attention_bias=True,
            activation_fn="gelu-approximate",
            num_embeds_ada_norm=1000,
            norm_type="ada_norm_zero",
            norm_elementwise_affine=False,
        )
        vae = AutoencoderKL()
        scheduler = DDIMScheduler()
        components = {"transformer": transformer.eval(), "vae": vae.eval(), "scheduler": scheduler}
        return components

    def get_dummy_inputs(self, device, seed=0):
        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device=device).manual_seed(seed)
        inputs = {
            "class_labels": [1],
            "generator": generator,
            "num_inference_steps": 2,
            "output_type": "np",
        }
        return inputs

    def test_inference(self):
        device = "cpu"

        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        image = pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1]

        self.assertEqual(image.shape, (1, 16, 16, 3))
        expected_slice = np.array([0.2946, 0.6601, 0.4329, 0.3296, 0.4144, 0.5319, 0.7273, 0.5013, 0.4457])
        max_diff = np.abs(image_slice.flatten() - expected_slice).max()
        self.assertLessEqual(max_diff, 1e-3)

    def test_inference_batch_single_identical(self):
        self._test_inference_batch_single_identical(expected_max_diff=1e-3)

    @unittest.skipIf(
        torch_device != "cuda" or not is_xformers_available(),
        reason="XFormers attention is only available with CUDA and `xformers` installed",
    )
    def test_xformers_attention_forwardGenerator_pass(self):
        self._test_xformers_attention_forwardGenerator_pass(expected_max_diff=1e-3)


@nightly
@require_torch_gpu
class DiTPipelineIntegrationTests(unittest.TestCase):
    def setUp(self):
        super().setUp()
        gc.collect()
        torch.cuda.empty_cache()

    def tearDown(self):
        super().tearDown()
        gc.collect()
        torch.cuda.empty_cache()

    def test_dit_256(self):
        generator = torch.manual_seed(0)

        pipe = DiTPipeline.from_pretrained("facebook/DiT-XL-2-256")
        pipe.to("cuda")

        words = ["vase", "umbrella", "white shark", "white wolf"]
        ids = pipe.get_label_ids(words)

        images = pipe(ids, generator=generator, num_inference_steps=40, output_type="np").images

        for word, image in zip(words, images):
            expected_image = load_numpy(
                f"https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/dit/{word}.npy"
            )
            assert np.abs((expected_image - image).max()) < 1e-2

    def test_dit_512(self):
        pipe = DiTPipeline.from_pretrained("facebook/DiT-XL-2-512")
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.to("cuda")

        words = ["vase", "umbrella"]
        ids = pipe.get_label_ids(words)

        generator = torch.manual_seed(0)
        images = pipe(ids, generator=generator, num_inference_steps=25, output_type="np").images

        for word, image in zip(words, images):
            expected_image = load_numpy(
                "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
                f"/dit/{word}_512.npy"
            )

            assert np.abs((expected_image - image).max()) < 1e-1
