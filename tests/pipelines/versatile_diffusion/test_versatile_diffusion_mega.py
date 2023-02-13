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

from diffusers import VersatileDiffusionPipeline
from diffusers.utils.testing_utils import load_image, require_torch_gpu, slow, torch_device


torch.backends.cuda.matmul.allow_tf32 = False


class VersatileDiffusionMegaPipelineFastTests(unittest.TestCase):
    pass


@slow
@require_torch_gpu
class VersatileDiffusionMegaPipelineIntegrationTests(unittest.TestCase):
    def tearDown(self):
        # clean up the VRAM after each test
        super().tearDown()
        gc.collect()
        torch.cuda.empty_cache()

    def test_from_save_pretrained(self):
        pipe = VersatileDiffusionPipeline.from_pretrained("shi-labs/versatile-diffusion", torch_dtype=torch.float16)
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        prompt_image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/blob/main/versatile_diffusion/benz.jpg"
        )

        generator = torch.manual_seed(0)
        image = pipe.dual_guided(
            prompt="first prompt",
            image=prompt_image,
            text_to_image_strength=0.75,
            generator=generator,
            guidance_scale=7.5,
            num_inference_steps=2,
            output_type="numpy",
        ).images

        with tempfile.TemporaryDirectory() as tmpdirname:
            pipe.save_pretrained(tmpdirname)
            pipe = VersatileDiffusionPipeline.from_pretrained(tmpdirname, torch_dtype=torch.float16)
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        generator = generator.manual_seed(0)
        new_image = pipe.dual_guided(
            prompt="first prompt",
            image=prompt_image,
            text_to_image_strength=0.75,
            generator=generator,
            guidance_scale=7.5,
            num_inference_steps=2,
            output_type="numpy",
        ).images

        assert np.abs(image - new_image).sum() < 1e-5, "Models don't have the same forward pass"

    def test_inference_dual_guided_then_text_to_image(self):
        pipe = VersatileDiffusionPipeline.from_pretrained("shi-labs/versatile-diffusion", torch_dtype=torch.float16)
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        prompt = "cyberpunk 2077"
        init_image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/blob/main/versatile_diffusion/benz.jpg"
        )
        generator = torch.manual_seed(0)
        image = pipe.dual_guided(
            prompt=prompt,
            image=init_image,
            text_to_image_strength=0.75,
            generator=generator,
            guidance_scale=7.5,
            num_inference_steps=50,
            output_type="numpy",
        ).images

        image_slice = image[0, 253:256, 253:256, -1]

        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array([0.1448, 0.1619, 0.1741, 0.1086, 0.1147, 0.1128, 0.1199, 0.1165, 0.1001])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-1

        prompt = "A painting of a squirrel eating a burger "
        generator = torch.manual_seed(0)
        image = pipe.text_to_image(
            prompt=prompt, generator=generator, guidance_scale=7.5, num_inference_steps=50, output_type="numpy"
        ).images

        image_slice = image[0, 253:256, 253:256, -1]

        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array([0.3367, 0.3169, 0.2656, 0.3870, 0.4790, 0.3796, 0.4009, 0.4878, 0.4778])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-1

        image = pipe.image_variation(init_image, generator=generator, output_type="numpy").images

        image_slice = image[0, 253:256, 253:256, -1]

        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array([0.3076, 0.3123, 0.3284, 0.3782, 0.3770, 0.3894, 0.4297, 0.4331, 0.4456])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-1
