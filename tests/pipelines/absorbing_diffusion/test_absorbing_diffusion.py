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

import torch

from diffusers import AbsorbingDiffusionPipeline
from diffusers.utils import slow, torch_device
from diffusers.utils.testing_utils import require_torch_gpu


torch.backends.cuda.matmul.allow_tf32 = False


@slow
@require_torch_gpu
class PipelineIntegrationTests(unittest.TestCase):
    def tearDown(self):
        # clean up the VRAM after each test
        super().tearDown()
        gc.collect()
        torch.cuda.empty_cache()

    def test_absorbing_diffusion(self):
        device = torch_device

        pipe = AbsorbingDiffusionPipeline.from_pretrained("nielsr/absorbing-diffusion-churches")
        pipe = pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        generator = torch.Generator(device=device).manual_seed(0)

        output = pipe(generator=generator, batch_size=10, num_inference_steps=256)
        images = output.images
        print(len(images))

        raise NotImplementedError("To do")
