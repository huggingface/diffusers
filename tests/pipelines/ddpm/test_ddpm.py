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
from diffusers.utils import deprecate
from diffusers.utils.testing_utils import require_torch_gpu, slow, torch_device


torch.backends.cuda.matmul.allow_tf32 = False


class DDPMPipelineFastTests(unittest.TestCase):
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
        device = "cpu"
        unet = self.dummy_uncond_unet
        scheduler = DDPMScheduler()

        ddpm = DDPMPipeline(unet=unet, scheduler=scheduler)
        ddpm.to(device)
        ddpm.set_progress_bar_config(disable=None)

        generator = torch.Generator(device=device).manual_seed(0)
        image = ddpm(generator=generator, num_inference_steps=2, output_type="numpy").images

        generator = torch.Generator(device=device).manual_seed(0)
        image_from_tuple = ddpm(generator=generator, num_inference_steps=2, output_type="numpy", return_dict=False)[0]

        image_slice = image[0, -3:, -3:, -1]
        image_from_tuple_slice = image_from_tuple[0, -3:, -3:, -1]

        assert image.shape == (1, 32, 32, 3)
        expected_slice = np.array(
            [5.589e-01, 7.089e-01, 2.632e-01, 6.841e-01, 1.000e-04, 9.999e-01, 1.973e-01, 1.000e-04, 8.010e-02]
        )
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2
        assert np.abs(image_from_tuple_slice.flatten() - expected_slice).max() < 1e-2

    def test_inference_deprecated_predict_epsilon(self):
        deprecate("remove this test", "0.12.0", "remove")
        unet = self.dummy_uncond_unet
        scheduler = DDPMScheduler(predict_epsilon=False)

        ddpm = DDPMPipeline(unet=unet, scheduler=scheduler)
        ddpm.to(torch_device)
        ddpm.set_progress_bar_config(disable=None)

        # Warmup pass when using mps (see #372)
        if torch_device == "mps":
            _ = ddpm(num_inference_steps=1)

        if torch_device == "mps":
            # device type MPS is not supported for torch.Generator() api.
            generator = torch.manual_seed(0)
        else:
            generator = torch.Generator(device=torch_device).manual_seed(0)
        image = ddpm(generator=generator, num_inference_steps=2, output_type="numpy").images

        generator = generator.manual_seed(0)
        image_eps = ddpm(generator=generator, num_inference_steps=2, output_type="numpy", predict_epsilon=False)[0]

        image_slice = image[0, -3:, -3:, -1]
        image_eps_slice = image_eps[0, -3:, -3:, -1]

        assert image.shape == (1, 32, 32, 3)
        tolerance = 1e-2 if torch_device != "mps" else 3e-2
        assert np.abs(image_slice.flatten() - image_eps_slice.flatten()).max() < tolerance

    def test_inference_predict_sample(self):
        unet = self.dummy_uncond_unet
        scheduler = DDPMScheduler(prediction_type="sample")

        ddpm = DDPMPipeline(unet=unet, scheduler=scheduler)
        ddpm.to(torch_device)
        ddpm.set_progress_bar_config(disable=None)

        # Warmup pass when using mps (see #372)
        if torch_device == "mps":
            _ = ddpm(num_inference_steps=1)

        if torch_device == "mps":
            # device type MPS is not supported for torch.Generator() api.
            generator = torch.manual_seed(0)
        else:
            generator = torch.Generator(device=torch_device).manual_seed(0)
        image = ddpm(generator=generator, num_inference_steps=2, output_type="numpy").images

        generator = generator.manual_seed(0)
        image_eps = ddpm(generator=generator, num_inference_steps=2, output_type="numpy")[0]

        image_slice = image[0, -3:, -3:, -1]
        image_eps_slice = image_eps[0, -3:, -3:, -1]

        assert image.shape == (1, 32, 32, 3)
        tolerance = 1e-2 if torch_device != "mps" else 3e-2
        assert np.abs(image_slice.flatten() - image_eps_slice.flatten()).max() < tolerance


@slow
@require_torch_gpu
class DDPMPipelineIntegrationTests(unittest.TestCase):
    def test_inference_cifar10(self):
        model_id = "google/ddpm-cifar10-32"

        unet = UNet2DModel.from_pretrained(model_id)
        scheduler = DDPMScheduler.from_pretrained(model_id)

        ddpm = DDPMPipeline(unet=unet, scheduler=scheduler)
        ddpm.to(torch_device)
        ddpm.set_progress_bar_config(disable=None)

        generator = torch.Generator(device=torch_device).manual_seed(0)
        image = ddpm(generator=generator, output_type="numpy").images

        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 32, 32, 3)
        expected_slice = np.array([0.4454, 0.2025, 0.0315, 0.3023, 0.2575, 0.1031, 0.0953, 0.1604, 0.2020])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2
