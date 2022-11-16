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

from diffusers import VersatileDiffusionPipeline
from diffusers.utils.testing_utils import require_torch, slow, torch_device

from ...test_pipelines_common import PipelineTesterMixin


torch.backends.cuda.matmul.allow_tf32 = False


class VersatileDiffusionPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pass


@slow
@require_torch
class VersatileDiffusionPipelineIntegrationTests(unittest.TestCase):
    def test_inference_text2img(self):
        pipe = VersatileDiffusionPipeline.from_pretrained("scripts/vd-diffusers")
        pipe.to(torch_device)
        #pipe.set_progress_bar_config(disable=None)

        prompt = "a dream of a village in china, by Caspar David Friedrich, matte painting trending on artstation HQ"
        generator = torch.Generator(device=torch_device).manual_seed(0)
        image = pipe(
            [prompt], generator=generator, guidance_scale=7.5, num_inference_steps=50, output_type="numpy"
        ).images

        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 256, 256, 3)
        expected_slice = np.array([0.9256, 0.9340, 0.8933, 0.9361, 0.9113, 0.8727, 0.9122, 0.8745, 0.8099])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2
