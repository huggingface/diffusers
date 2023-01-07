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

from diffusers import KarrasVePipeline, KarrasVeScheduler, UNet2DModel
from diffusers.utils.testing_utils import require_torch, slow, torch_device


torch.backends.cuda.matmul.allow_tf32 = False


@slow
@require_torch
class KarrasVePipelineIntegrationTests(unittest.TestCase):
    def test_inference(self):
        model_id = "google/ncsnpp-celebahq-256"
        model = UNet2DModel.from_pretrained(model_id)
        scheduler = KarrasVeScheduler()

        pipe = KarrasVePipeline(unet=model, scheduler=scheduler)
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        generator = torch.manual_seed(0)
        image = pipe(num_inference_steps=20, generator=generator, output_type="numpy").images

        image_slice = image[0, -3:, -3:, -1]
        assert image.shape == (1, 256, 256, 3)
        expected_slice = np.array([0.578, 0.5811, 0.5924, 0.5809, 0.587, 0.5886, 0.5861, 0.5802, 0.586])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2
