# coding=utf-8
# Copyright 2023 HuggingFace Inc.
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
from diffusers.utils.testing_utils import enable_full_determinism, require_torch_gpu, slow, torch_device

from ..pipeline_params import UNCONDITIONAL_IMAGE_GENERATION_BATCH_PARAMS, UNCONDITIONAL_IMAGE_GENERATION_PARAMS
from ..test_pipelines_common import PipelineTesterMixin


enable_full_determinism()


class DDIMPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = DDIMPipeline
    params = UNCONDITIONAL_IMAGE_GENERATION_PARAMS
    required_optional_params = PipelineTesterMixin.required_optional_params - {
        "num_images_per_prompt",
        "latents",
        "callback",
        "callback_steps",
    }
    batch_params = UNCONDITIONAL_IMAGE_GENERATION_BATCH_PARAMS

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
            "batch_size": 1,
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

    def test_dict_tuple_outputs_equivalent(self):
        super().test_dict_tuple_outputs_equivalent(expected_max_difference=3e-3)

    def test_save_load_local(self):
        super().test_save_load_local(expected_max_difference=3e-3)

    def test_save_load_optional_components(self):
        super().test_save_load_optional_components(expected_max_difference=3e-3)

    def test_inference_batch_single_identical(self):
        super().test_inference_batch_single_identical(expected_max_diff=3e-3)


@slow
@require_torch_gpu
class DDIMPipelineIntegrationTests(unittest.TestCase):
    def test_inference_cifar10(self):
        model_id = "google/ddpm-cifar10-32"

        unet = UNet2DModel.from_pretrained(model_id)
        scheduler = DDIMScheduler()

        ddim = DDIMPipeline(unet=unet, scheduler=scheduler)
        ddim.to(torch_device)
        ddim.set_progress_bar_config(disable=None)

        generator = torch.manual_seed(0)
        image = ddim(generator=generator, eta=0.0, output_type="numpy").images

        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 32, 32, 3)
        expected_slice = np.array([0.1723, 0.1617, 0.1600, 0.1626, 0.1497, 0.1513, 0.1505, 0.1442, 0.1453])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

    def test_inference_ema_bedroom(self):
        model_id = "google/ddpm-ema-bedroom-256"

        unet = UNet2DModel.from_pretrained(model_id)
        scheduler = DDIMScheduler.from_pretrained(model_id)

        ddpm = DDIMPipeline(unet=unet, scheduler=scheduler)
        ddpm.to(torch_device)
        ddpm.set_progress_bar_config(disable=None)

        generator = torch.manual_seed(0)
        image = ddpm(generator=generator, output_type="numpy").images

        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 256, 256, 3)
        expected_slice = np.array([0.0060, 0.0201, 0.0344, 0.0024, 0.0018, 0.0002, 0.0022, 0.0000, 0.0069])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2
