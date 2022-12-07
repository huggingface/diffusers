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

from diffusers import DDIMPipeline, DDIMScheduler, UNet2DModel
from diffusers.utils.testing_utils import require_torch_gpu, slow, torch_device

from ...test_pipelines_common import PipelineTesterMixin


torch.backends.cuda.matmul.allow_tf32 = False


class DDIMPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = DDIMPipeline

    def get_dummy_components(self):
        torch.manual_seed(0)
        unet = UNet2DModel(
            block_out_channels=(32, 64),
            layers_per_block=2,
            sample_size=32,
            in_channels=3,
            out_channels=3,
            down_block_types=("DownBlock2D", "AttnDownBlock2D"),
            up_block_types=("AttnUpBlock2D", "UpBlock2D"),
        )
        scheduler = DDIMScheduler()
        components = {"unet": unet, "scheduler": scheduler}
        return components

    def get_dummy_inputs(self, device, seed=0):
        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device=device).manual_seed(seed)
        inputs = {
            "generator": generator,
            "num_inference_steps": 2,
            "output_type": "numpy",
        }
        return inputs

    def test_inference(self):
        device = "cpu"

        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        image = pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1]

        self.assertEqual(image.shape, (1, 32, 32, 3))
        expected_slice = np.array(
            [1.000e00, 5.717e-01, 4.717e-01, 1.000e00, 0.000e00, 1.000e00, 3.000e-04, 0.000e00, 9.000e-04]
        )
        max_diff = np.abs(image_slice.flatten() - expected_slice).max()
        self.assertLessEqual(max_diff, 1e-3)


@slow
@require_torch_gpu
class DDIMPipelineIntegrationTests(unittest.TestCase):
    def test_inference_ema_bedroom(self):
        model_id = "google/ddpm-ema-bedroom-256"

        unet = UNet2DModel.from_pretrained(model_id)
        scheduler = DDIMScheduler.from_pretrained(model_id)

        ddpm = DDIMPipeline(unet=unet, scheduler=scheduler)
        ddpm.to(torch_device)
        ddpm.set_progress_bar_config(disable=None)

        generator = torch.Generator(device=torch_device).manual_seed(0)
        image = ddpm(generator=generator, output_type="numpy").images

        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 256, 256, 3)
        expected_slice = np.array([0.1546, 0.1561, 0.1595, 0.1564, 0.1569, 0.1585, 0.1554, 0.1550, 0.1575])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

    def test_inference_cifar10(self):
        model_id = "google/ddpm-cifar10-32"

        unet = UNet2DModel.from_pretrained(model_id)
        scheduler = DDIMScheduler()

        ddim = DDIMPipeline(unet=unet, scheduler=scheduler)
        ddim.to(torch_device)
        ddim.set_progress_bar_config(disable=None)

        generator = torch.Generator(device=torch_device).manual_seed(0)
        image = ddim(generator=generator, eta=0.0, output_type="numpy").images

        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 32, 32, 3)
        expected_slice = np.array([0.2060, 0.2042, 0.2022, 0.2193, 0.2146, 0.2110, 0.2471, 0.2446, 0.2388])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2
