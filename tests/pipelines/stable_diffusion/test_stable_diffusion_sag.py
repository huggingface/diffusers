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

import gc
import unittest

import numpy as np
import torch

from diffusers import StableDiffusionSAGPipeline
from diffusers.utils import slow, torch_device
from diffusers.utils.testing_utils import require_torch_gpu


torch.backends.cuda.matmul.allow_tf32 = False


@slow
@require_torch_gpu
class StableDiffusionPipelineIntegrationTests(unittest.TestCase):
    def tearDown(self):
        # clean up the VRAM after each test
        super().tearDown()
        gc.collect()
        torch.cuda.empty_cache()

    def test_stable_diffusion_1(self):
        sag_pipe = StableDiffusionSAGPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
        sag_pipe = sag_pipe.to(torch_device)
        sag_pipe.set_progress_bar_config(disable=None)

        prompt = "."
        generator = torch.manual_seed(0)
        output = sag_pipe([prompt], generator=generator, guidance_scale=7.5, sag_scale=1.0, num_inference_steps=20, output_type="np")

        image = output.images

        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array([0.1645, 0.1758, 0.1716, 0.1746, 0.1459, 0.1763, 0.1558, 0.1696, 0.2003])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 5e-5

    def test_stable_diffusion_2(self):
        sag_pipe = StableDiffusionSAGPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-base")
        sag_pipe = sag_pipe.to(torch_device)
        sag_pipe.set_progress_bar_config(disable=None)

        prompt = "."
        generator = torch.manual_seed(0)
        output = sag_pipe([prompt], generator=generator, guidance_scale=7.5, sag_scale=1.0, num_inference_steps=20, output_type="np")

        image = output.images

        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array([0.3485, 0.2882, 0.2547, 0.2982, 0.2684, 0.2154, 0.3005, 0.2252, 0.2368])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 5e-5
