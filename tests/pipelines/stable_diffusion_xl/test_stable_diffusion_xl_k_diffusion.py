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

from diffusers import StableDiffusionXLKDiffusionPipeline
from diffusers.utils.testing_utils import enable_full_determinism, require_torch_gpu, slow, torch_device


enable_full_determinism()


@slow
@require_torch_gpu
class StableDiffusionXLKPipelineIntegrationTests(unittest.TestCase):
    dtype = torch.float16

    def setUp(self):
        # clean up the VRAM before each test
        super().setUp()
        gc.collect()
        torch.cuda.empty_cache()

    def tearDown(self):
        # clean up the VRAM after each test
        super().tearDown()
        gc.collect()
        torch.cuda.empty_cache()

    def test_stable_diffusion_xl(self):
        sd_pipe = StableDiffusionXLKDiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=self.dtype
        )
        sd_pipe = sd_pipe.to(torch_device)
        sd_pipe.set_progress_bar_config(disable=None)

        sd_pipe.set_scheduler("sample_euler")

        prompt = "A painting of a squirrel eating a burger"
        generator = torch.manual_seed(0)
        output = sd_pipe(
            [prompt],
            generator=generator,
            guidance_scale=9.0,
            num_inference_steps=2,
            height=512,
            width=512,
            output_type="np",
        )

        image = output.images

        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array([0.5420, 0.5038, 0.2439, 0.5371, 0.4660, 0.1906, 0.5221, 0.4290, 0.2566])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

    def test_stable_diffusion_karras_sigmas(self):
        sd_pipe = StableDiffusionXLKDiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=self.dtype
        )
        sd_pipe = sd_pipe.to(torch_device)
        sd_pipe.set_progress_bar_config(disable=None)

        sd_pipe.set_scheduler("sample_dpmpp_2m")

        prompt = "A painting of a squirrel eating a burger"
        generator = torch.manual_seed(0)
        output = sd_pipe(
            [prompt],
            generator=generator,
            guidance_scale=7.5,
            num_inference_steps=2,
            output_type="np",
            use_karras_sigmas=True,
            height=512,
            width=512,
        )

        image = output.images

        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array([0.6418, 0.6424, 0.6462, 0.6271, 0.6314, 0.6295, 0.6249, 0.6339, 0.6335])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

    def test_stable_diffusion_noise_sampler_seed(self):
        sd_pipe = StableDiffusionXLKDiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=self.dtype
        )
        sd_pipe = sd_pipe.to(torch_device)
        sd_pipe.set_progress_bar_config(disable=None)

        sd_pipe.set_scheduler("sample_dpmpp_sde")

        prompt = "A painting of a squirrel eating a burger"
        seed = 0
        images1 = sd_pipe(
            [prompt],
            generator=torch.manual_seed(seed),
            noise_sampler_seed=seed,
            guidance_scale=9.0,
            num_inference_steps=2,
            output_type="np",
            height=512,
            width=512,
        ).images
        images2 = sd_pipe(
            [prompt],
            generator=torch.manual_seed(seed),
            noise_sampler_seed=seed,
            guidance_scale=9.0,
            num_inference_steps=2,
            output_type="np",
            height=512,
            width=512,
        ).images
        assert images1.shape == (1, 512, 512, 3)
        assert images2.shape == (1, 512, 512, 3)
        assert np.abs(images1.flatten() - images2.flatten()).max() < 1e-2
