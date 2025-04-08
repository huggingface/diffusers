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

import unittest

import numpy as np
import torch

from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel
from diffusers.utils.testing_utils import enable_full_determinism, require_torch_accelerator, slow, torch_device


enable_full_determinism()


class DDPMPipelineFastTests(unittest.TestCase):
    @property
    def dummy_uncond_unet(self):
        torch.manual_seed(0)
        model = UNet2DModel(
            block_out_channels=(4, 8),
            layers_per_block=1,
            norm_num_groups=4,
            sample_size=8,
            in_channels=3,
            out_channels=3,
            down_block_types=("DownBlock2D", "AttnDownBlock2D"),
            up_block_types=("AttnUpBlock2D", "UpBlock2D"),
        )
        return model

    def test_fast_inference(self):
        device = "cpu"
        unet = self.dummy_uncond_unet
        scheduler = DDPMScheduler()

        ddpm = DDPMPipeline(unet=unet, scheduler=scheduler)
        ddpm.to(device)
        ddpm.set_progress_bar_config(disable=None)

        generator = torch.Generator(device=device).manual_seed(0)
        image = ddpm(generator=generator, num_inference_steps=2, output_type="np").images

        generator = torch.Generator(device=device).manual_seed(0)
        image_from_tuple = ddpm(generator=generator, num_inference_steps=2, output_type="np", return_dict=False)[0]

        image_slice = image[0, -3:, -3:, -1]
        image_from_tuple_slice = image_from_tuple[0, -3:, -3:, -1]

        assert image.shape == (1, 8, 8, 3)
        expected_slice = np.array([0.0, 0.9996672, 0.00329116, 1.0, 0.9995991, 1.0, 0.0060907, 0.00115037, 0.0])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2
        assert np.abs(image_from_tuple_slice.flatten() - expected_slice).max() < 1e-2

    def test_inference_predict_sample(self):
        unet = self.dummy_uncond_unet
        scheduler = DDPMScheduler(prediction_type="sample")

        ddpm = DDPMPipeline(unet=unet, scheduler=scheduler)
        ddpm.to(torch_device)
        ddpm.set_progress_bar_config(disable=None)

        generator = torch.manual_seed(0)
        image = ddpm(generator=generator, num_inference_steps=2, output_type="np").images

        generator = torch.manual_seed(0)
        image_eps = ddpm(generator=generator, num_inference_steps=2, output_type="np")[0]

        image_slice = image[0, -3:, -3:, -1]
        image_eps_slice = image_eps[0, -3:, -3:, -1]

        assert image.shape == (1, 8, 8, 3)
        tolerance = 1e-2 if torch_device != "mps" else 3e-2
        assert np.abs(image_slice.flatten() - image_eps_slice.flatten()).max() < tolerance


@slow
@require_torch_accelerator
class DDPMPipelineIntegrationTests(unittest.TestCase):
    def test_inference_cifar10(self):
        model_id = "google/ddpm-cifar10-32"

        unet = UNet2DModel.from_pretrained(model_id)
        scheduler = DDPMScheduler.from_pretrained(model_id)

        ddpm = DDPMPipeline(unet=unet, scheduler=scheduler)
        ddpm.to(torch_device)
        ddpm.set_progress_bar_config(disable=None)

        generator = torch.manual_seed(0)
        image = ddpm(generator=generator, output_type="np").images

        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 32, 32, 3)
        expected_slice = np.array([0.4200, 0.3588, 0.1939, 0.3847, 0.3382, 0.2647, 0.4155, 0.3582, 0.3385])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2
