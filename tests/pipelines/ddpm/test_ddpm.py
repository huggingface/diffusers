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

from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel
from diffusers.utils.testing_utils import require_torch, slow, torch_device

from ...test_pipelines_common import PipelineTesterMixin


torch.backends.cuda.matmul.allow_tf32 = False


class DDPMPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    # FIXME: add fast tests
    pass


@slow
@require_torch
class DDPMPipelineIntegrationTests(unittest.TestCase):
    def test_inference_cifar10(self):
        model_id = "google/ddpm-cifar10-32"

        unet = UNet2DModel.from_pretrained(model_id, device_map="auto")
        scheduler = DDPMScheduler.from_config(model_id)

        ddpm = DDPMPipeline(unet=unet, scheduler=scheduler)
        ddpm.to(torch_device)
        ddpm.set_progress_bar_config(disable=None)

        generator = torch.manual_seed(0)
        image = ddpm(generator=generator, output_type="numpy").images

        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 32, 32, 3)
        expected_slice = np.array([0.41995, 0.35885, 0.19385, 0.38475, 0.3382, 0.2647, 0.41545, 0.3582, 0.33845])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2
