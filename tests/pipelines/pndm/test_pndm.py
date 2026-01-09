# coding=utf-8
# Copyright 2025 HuggingFace Inc.
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

from diffusers import PNDMPipeline, PNDMScheduler, UNet2DModel

from ...testing_utils import enable_full_determinism, nightly, require_torch, torch_device


enable_full_determinism()


class PNDMPipelineFastTests(unittest.TestCase):
    @property
    def dummy_uncond_unet(self):
        torch.manual_seed(0)
        model = UNet2DModel(
            block_out_channels=(32, 64),
            layers_per_block=2,
            sample_size=32,
            in_channels=3,
            out_channels=3,
            down_block_types=("DownBlock2D", "AttnDownBlock2D"),
            up_block_types=("AttnUpBlock2D", "UpBlock2D"),
        )
        return model

    def test_inference(self):
        unet = self.dummy_uncond_unet
        scheduler = PNDMScheduler()

        pndm = PNDMPipeline(unet=unet, scheduler=scheduler)
        pndm.to(torch_device)
        pndm.set_progress_bar_config(disable=None)

        generator = torch.manual_seed(0)
        image = pndm(generator=generator, num_inference_steps=20, output_type="np").images

        generator = torch.manual_seed(0)
        image_from_tuple = pndm(generator=generator, num_inference_steps=20, output_type="np", return_dict=False)[0]

        image_slice = image[0, -3:, -3:, -1]
        image_from_tuple_slice = image_from_tuple[0, -3:, -3:, -1]

        assert image.shape == (1, 32, 32, 3)
        expected_slice = np.array([1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2
        assert np.abs(image_from_tuple_slice.flatten() - expected_slice).max() < 1e-2


@nightly
@require_torch
class PNDMPipelineIntegrationTests(unittest.TestCase):
    def test_inference_cifar10(self):
        model_id = "google/ddpm-cifar10-32"

        unet = UNet2DModel.from_pretrained(model_id)
        scheduler = PNDMScheduler()

        pndm = PNDMPipeline(unet=unet, scheduler=scheduler)
        pndm.to(torch_device)
        pndm.set_progress_bar_config(disable=None)
        generator = torch.manual_seed(0)
        image = pndm(generator=generator, output_type="np").images

        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 32, 32, 3)
        expected_slice = np.array([0.1564, 0.14645, 0.1406, 0.14715, 0.12425, 0.14045, 0.13115, 0.12175, 0.125])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2
