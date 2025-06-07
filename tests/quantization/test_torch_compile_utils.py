# coding=utf-8
# Copyright 2024 The HuggingFace Team Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a clone of the License at
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

from diffusers import DiffusionPipeline
from diffusers.utils.testing_utils import backend_empty_cache, require_torch_gpu, slow, torch_device


@require_torch_gpu
@slow
class QuantCompileMiscTests(unittest.TestCase):
    def setUp(self):
        super().setUp()
        gc.collect()
        backend_empty_cache(torch_device)
        torch.compiler.reset()

    def tearDown(self):
        super().tearDown()
        gc.collect()
        backend_empty_cache(torch_device)
        torch.compiler.reset()

    def test_torch_compile(self, quantization_config, torch_dtype=torch.bfloat16):
        pipe = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-3-medium-diffusers",
            quantization_config=quantization_config,
            torch_dtype=torch_dtype,
        ).to("cuda")
        pipe.transformer.compile(fullgraph=True)

        for _ in range(2):
            # small resolutions to ensure speedy execution.
            pipe("a dog", num_inference_steps=4, max_sequence_length=16, height=256, width=256)
