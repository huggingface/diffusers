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

import unittest

import numpy as np
import torch

from diffusers import VersatileDiffusionDualGuidedPipeline
from diffusers.utils.testing_utils import load_image, require_torch_gpu, slow, torch_device

from ...test_pipelines_common import PipelineTesterMixin


torch.backends.cuda.matmul.allow_tf32 = False


class VersatileDiffusionDualGuidedPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pass


@slow
@require_torch_gpu
class VersatileDiffusionDualGuidedPipelineIntegrationTests(unittest.TestCase):
    def test_inference_image_variations(self):
        pipe = VersatileDiffusionDualGuidedPipeline.from_pretrained("diffusers/vd-official-test")
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        first_prompt = "cyberpunk 2077"
        second_prompt = load_image(
            "https://raw.githubusercontent.com/SHI-Labs/Versatile-Diffusion/master/assets/benz.jpg"
        )
        generator = torch.Generator(device=torch_device).manual_seed(22)
        image = pipe(
            first_prompt=first_prompt,
            second_prompt=second_prompt,
            prompt_mix_ratio=0.75,
            generator=generator,
            guidance_scale=7.5,
            num_inference_steps=50,
            output_type="numpy",
        ).images

        image_slice = image[0, 253:256, 253:256, -1]

        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array([0.1811, 0.0430, 0.0433, 0.1082, 0.0144, 0.0306, 0.0683, 0.0248, 0.0876])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2
