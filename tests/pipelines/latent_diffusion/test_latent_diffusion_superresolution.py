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

import random
import unittest

import numpy as np
import torch

import PIL
from diffusers import DDIMScheduler, LDMSuperResolutionPipeline, UNet2DModel, VQModel
from diffusers.utils import floats_tensor, load_image, slow, torch_device
from diffusers.utils.testing_utils import require_torch

from ...test_pipelines_common import PipelineTesterMixin


torch.backends.cuda.matmul.allow_tf32 = False


class LDMSuperResolutionPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    @property
    def dummy_image(self):
        batch_size = 1
        num_channels = 3
        sizes = (32, 32)

        image = floats_tensor((batch_size, num_channels) + sizes, rng=random.Random(0)).to(torch_device)
        return image

    @property
    def dummy_uncond_unet(self):
        torch.manual_seed(0)
        model = UNet2DModel(
            block_out_channels=(32, 64),
            layers_per_block=2,
            sample_size=32,
            in_channels=6,
            out_channels=3,
            down_block_types=("DownBlock2D", "AttnDownBlock2D"),
            up_block_types=("AttnUpBlock2D", "UpBlock2D"),
        )
        return model

    @property
    def dummy_vq_model(self):
        torch.manual_seed(0)
        model = VQModel(
            block_out_channels=[32, 64],
            in_channels=3,
            out_channels=3,
            down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
            up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"],
            latent_channels=3,
        )
        return model

    def test_inference_superresolution(self):
        unet = self.dummy_uncond_unet
        scheduler = DDIMScheduler()
        vqvae = self.dummy_vq_model

        ldm = LDMSuperResolutionPipeline(unet=unet, vqvae=vqvae, scheduler=scheduler)
        ldm.to(torch_device)
        ldm.set_progress_bar_config(disable=None)

        init_image = self.dummy_image.to(torch_device)

        # Warmup pass when using mps (see #372)
        if torch_device == "mps":
            generator = torch.manual_seed(0)
            _ = ldm(init_image, generator=generator, num_inference_steps=1, output_type="numpy").images

        generator = torch.manual_seed(0)
        image = ldm(init_image, generator=generator, num_inference_steps=2, output_type="numpy").images

        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 64, 64, 3)
        expected_slice = np.array([0.8634, 0.8186, 0.6416, 0.6846, 0.4427, 0.5676, 0.4679, 0.6247, 0.5176])
        tolerance = 1e-2 if torch_device != "mps" else 3e-2
        assert np.abs(image_slice.flatten() - expected_slice).max() < tolerance


@slow
@require_torch
class LDMSuperResolutionPipelineIntegrationTests(unittest.TestCase):
    def test_inference_superresolution(self):
        init_image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
            "/vq_diffusion/teddy_bear_pool.png"
        )
        init_image = init_image.resize((64, 64), resample=PIL.Image.LANCZOS)

        ldm = LDMSuperResolutionPipeline.from_pretrained("duongna/ldm-super-resolution", device_map="auto")
        ldm.to(torch_device)
        ldm.set_progress_bar_config(disable=None)

        generator = torch.Generator(device=torch_device).manual_seed(0)
        image = ldm(init_image, generator=generator, num_inference_steps=20, output_type="numpy").images

        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 256, 256, 3)
        expected_slice = np.array([0.7418, 0.7472, 0.7424, 0.7422, 0.7463, 0.726, 0.7382, 0.7248, 0.6828])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2
