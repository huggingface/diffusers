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

import torch

from diffusers import DDIMScheduler, TextToVideoZeroPipeline
from diffusers.utils.testing_utils import (
    backend_empty_cache,
    load_pt,
    nightly,
    require_torch_accelerator,
    torch_device,
)

from ..test_pipelines_common import assert_mean_pixel_difference


@nightly
@require_torch_accelerator
class TextToVideoZeroPipelineSlowTests(unittest.TestCase):
    def setUp(self):
        # clean up the VRAM before each test
        super().setUp()
        gc.collect()
        backend_empty_cache(torch_device)

    def tearDown(self):
        # clean up the VRAM after each test
        super().tearDown()
        gc.collect()
        backend_empty_cache(torch_device)

    def test_full_model(self):
        model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
        pipe = TextToVideoZeroPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(torch_device)
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        generator = torch.Generator(device="cpu").manual_seed(0)

        prompt = "A bear is playing a guitar on Times Square"
        result = pipe(prompt=prompt, generator=generator).images

        expected_result = load_pt(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/text-to-video/A bear is playing a guitar on Times Square.pt",
            weights_only=False,
        )

        assert_mean_pixel_difference(result, expected_result)
