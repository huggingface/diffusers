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

from diffusers import StableDiffusionKDiffusionPipeline
from diffusers.utils.testing_utils import enable_full_determinism, nightly, require_torch_gpu, torch_device


enable_full_determinism()


@nightly
@require_torch_gpu
class StableDiffusionPipelineIntegrationTests(unittest.TestCase):
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

    def test_stable_diffusion_1(self):
        sd_pipe = StableDiffusionKDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
        sd_pipe = sd_pipe.to(torch_device)
        sd_pipe.set_progress_bar_config(disable=None)

        sd_pipe.set_scheduler("sample_euler")

        prompt = "A painting of a squirrel eating a burger"
        generator = torch.manual_seed(0)
        output = sd_pipe([prompt], generator=generator, guidance_scale=9.0, num_inference_steps=20, output_type="np")

        image = output.images

        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array([0.0447, 0.0492, 0.0468, 0.0408, 0.0383, 0.0408, 0.0354, 0.0380, 0.0339])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

    def test_stable_diffusion_2(self):
        sd_pipe = StableDiffusionKDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-base")
        sd_pipe = sd_pipe.to(torch_device)
        sd_pipe.set_progress_bar_config(disable=None)

        sd_pipe.set_scheduler("sample_euler")

        prompt = "A painting of a squirrel eating a burger"
        generator = torch.manual_seed(0)
        output = sd_pipe([prompt], generator=generator, guidance_scale=9.0, num_inference_steps=20, output_type="np")

        image = output.images

        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array([0.1237, 0.1320, 0.1438, 0.1359, 0.1390, 0.1132, 0.1277, 0.1175, 0.1112])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 5e-1

    def test_stable_diffusion_karras_sigmas(self):
        sd_pipe = StableDiffusionKDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-base")
        sd_pipe = sd_pipe.to(torch_device)
        sd_pipe.set_progress_bar_config(disable=None)

        sd_pipe.set_scheduler("sample_dpmpp_2m")

        prompt = "A painting of a squirrel eating a burger"
        generator = torch.manual_seed(0)
        output = sd_pipe(
            [prompt],
            generator=generator,
            guidance_scale=7.5,
            num_inference_steps=15,
            output_type="np",
            use_karras_sigmas=True,
        )

        image = output.images

        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array(
            [0.11381689, 0.12112921, 0.1389457, 0.12549606, 0.1244964, 0.10831517, 0.11562866, 0.10867816, 0.10499048]
        )

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

    def test_stable_diffusion_noise_sampler_seed(self):
        sd_pipe = StableDiffusionKDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
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
            num_inference_steps=20,
            output_type="np",
        ).images
        images2 = sd_pipe(
            [prompt],
            generator=torch.manual_seed(seed),
            noise_sampler_seed=seed,
            guidance_scale=9.0,
            num_inference_steps=20,
            output_type="np",
        ).images

        assert images1.shape == (1, 512, 512, 3)
        assert images2.shape == (1, 512, 512, 3)
        assert np.abs(images1.flatten() - images2.flatten()).max() < 1e-2
