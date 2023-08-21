# coding=utf-8
# Copyright 2023 HuggingFace Inc.
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
import torch

from diffusers import DDIMScheduler, TextToVideoZeroSDXLPipeline
from diffusers.utils import require_torch_gpu, slow


@slow
@require_torch_gpu
class TextToVideoZeroSDXLPipelineSlowTests(unittest.TestCase):
    def test_full_model(self):
        model_id = "stabilityai/stable-diffusion-xl-base-1.0"
        pipe = TextToVideoZeroSDXLPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
        ).to("cuda")
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        generator = torch.Generator(device="cuda").manual_seed(0)

        prompt = "A panda dancing in Antarctica"
        result = pipe(prompt=prompt, generator=generator).images

        first_frame_slice = result[0, -3:, -3:, -1]
        last_frame_slice = result[-1, -3:, -3:, 0]

        expected_slice1 = np.array([0.12, 0.11, 0.11, 0.11, 0.11, 0.11, 0.10, 0.12, 0.12])
        expected_slice2 = np.array([0.53, 0.53, 0.53, 0.53, 0.54, 0.54, 0.53, 0.55, 0.55])

        assert np.abs(first_frame_slice.flatten() - expected_slice1).max() < 1e-2
        assert np.abs(last_frame_slice.flatten() - expected_slice2).max() < 1e-2
