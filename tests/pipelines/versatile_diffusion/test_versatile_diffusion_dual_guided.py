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
import tempfile
import unittest

import numpy as np
import torch

from diffusers import VersatileDiffusionDualGuidedPipeline
from diffusers.utils.testing_utils import load_image, require_torch_gpu, slow, torch_device


torch.backends.cuda.matmul.allow_tf32 = False


class VersatileDiffusionDualGuidedPipelineFastTests(unittest.TestCase):
    pass


@slow
@require_torch_gpu
class VersatileDiffusionDualGuidedPipelineIntegrationTests(unittest.TestCase):
    def tearDown(self):
        # clean up the VRAM after each test
        super().tearDown()
        gc.collect()
        torch.cuda.empty_cache()

    def test_remove_unused_weights_save_load(self):
        pipe = VersatileDiffusionDualGuidedPipeline.from_pretrained("shi-labs/versatile-diffusion")
        # remove text_unet
        pipe.remove_unused_weights()
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        second_prompt = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/versatile_diffusion/benz.jpg"
        )

        generator = torch.manual_seed(0)
        image = pipe(
            prompt="first prompt",
            image=second_prompt,
            text_to_image_strength=0.75,
            generator=generator,
            guidance_scale=7.5,
            num_inference_steps=2,
            output_type="numpy",
        ).images

        with tempfile.TemporaryDirectory() as tmpdirname:
            pipe.save_pretrained(tmpdirname)
            pipe = VersatileDiffusionDualGuidedPipeline.from_pretrained(tmpdirname)

        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        generator = generator.manual_seed(0)
        new_image = pipe(
            prompt="first prompt",
            image=second_prompt,
            text_to_image_strength=0.75,
            generator=generator,
            guidance_scale=7.5,
            num_inference_steps=2,
            output_type="numpy",
        ).images

        assert np.abs(image - new_image).sum() < 1e-5, "Models don't have the same forward pass"

    def test_inference_dual_guided(self):
        pipe = VersatileDiffusionDualGuidedPipeline.from_pretrained("shi-labs/versatile-diffusion")
        pipe.remove_unused_weights()
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        first_prompt = "cyberpunk 2077"
        second_prompt = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/versatile_diffusion/benz.jpg"
        )
        generator = torch.manual_seed(0)
        image = pipe(
            prompt=first_prompt,
            image=second_prompt,
            text_to_image_strength=0.75,
            generator=generator,
            guidance_scale=7.5,
            num_inference_steps=50,
            output_type="numpy",
        ).images

        image_slice = image[0, 253:256, 253:256, -1]

        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array([0.0787, 0.0849, 0.0826, 0.0812, 0.0807, 0.0795, 0.0818, 0.0798, 0.0779])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2
