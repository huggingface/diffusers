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

from diffusers import VersatileDiffusionImageToTextPipeline
from diffusers.utils.testing_utils import load_image, require_torch_gpu, slow, torch_device

from ...test_pipelines_common import PipelineTesterMixin


torch.backends.cuda.matmul.allow_tf32 = False


class VersatileDiffusionImageToTextPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pass


@slow
@require_torch_gpu
class VersatileDiffusionImageToTextPipelineIntegrationTests(unittest.TestCase):
    def test_inference_image_to_text(self):
        pipe = VersatileDiffusionImageToTextPipeline.from_pretrained("diffusers/vd-official-test")
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        image_prompt = load_image(
            "https://raw.githubusercontent.com/SHI-Labs/Versatile-Diffusion/master/assets/benz.jpg"
        )
        generator = torch.Generator(device=torch_device).manual_seed(0)
        tokens = pipe(
            image=image_prompt,
            generator=generator,
            guidance_scale=7.5,
            num_inference_steps=50,
            output_type="numpy",
        ).text

        assert tokens.shape == (1, 30)
        expected_tokens = np.array([0, 1, 2, 3, 4, 5, 6, 7])
        assert self.assertItemsEqual(tokens[0], expected_tokens)
