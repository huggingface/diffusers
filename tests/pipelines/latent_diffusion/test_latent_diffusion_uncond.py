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
from transformers import CLIPTextConfig, CLIPTextModel

from diffusers import DDIMScheduler, LDMPipeline, UNet2DModel, VQModel
from diffusers.utils.testing_utils import require_torch, slow, torch_device


torch.backends.cuda.matmul.allow_tf32 = False


class LDMPipelineFastTests(unittest.TestCase):
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

    @property
    def dummy_text_encoder(self):
        torch.manual_seed(0)
        config = CLIPTextConfig(
            bos_token_id=0,
            eos_token_id=2,
            hidden_size=32,
            intermediate_size=37,
            layer_norm_eps=1e-05,
            num_attention_heads=4,
            num_hidden_layers=5,
            pad_token_id=1,
            vocab_size=1000,
        )
        return CLIPTextModel(config)

    def test_inference_uncond(self):
        unet = self.dummy_uncond_unet
        scheduler = DDIMScheduler()
        vae = self.dummy_vq_model

        ldm = LDMPipeline(unet=unet, vqvae=vae, scheduler=scheduler)
        ldm.to(torch_device)
        ldm.set_progress_bar_config(disable=None)

        # Warmup pass when using mps (see #372)
        if torch_device == "mps":
            generator = torch.manual_seed(0)
            _ = ldm(generator=generator, num_inference_steps=1, output_type="numpy").images

        generator = torch.manual_seed(0)
        image = ldm(generator=generator, num_inference_steps=2, output_type="numpy").images

        generator = torch.manual_seed(0)
        image_from_tuple = ldm(generator=generator, num_inference_steps=2, output_type="numpy", return_dict=False)[0]

        image_slice = image[0, -3:, -3:, -1]
        image_from_tuple_slice = image_from_tuple[0, -3:, -3:, -1]

        assert image.shape == (1, 64, 64, 3)
        expected_slice = np.array([0.8512, 0.818, 0.6411, 0.6808, 0.4465, 0.5618, 0.46, 0.6231, 0.5172])
        tolerance = 1e-2 if torch_device != "mps" else 3e-2

        assert np.abs(image_slice.flatten() - expected_slice).max() < tolerance
        assert np.abs(image_from_tuple_slice.flatten() - expected_slice).max() < tolerance


@slow
@require_torch
class LDMPipelineIntegrationTests(unittest.TestCase):
    def test_inference_uncond(self):
        ldm = LDMPipeline.from_pretrained("CompVis/ldm-celebahq-256")
        ldm.to(torch_device)
        ldm.set_progress_bar_config(disable=None)

        generator = torch.manual_seed(0)
        image = ldm(generator=generator, num_inference_steps=5, output_type="numpy").images

        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 256, 256, 3)
        expected_slice = np.array([0.4399, 0.44975, 0.46825, 0.474, 0.4359, 0.4581, 0.45095, 0.4341, 0.4447])
        tolerance = 1e-2 if torch_device != "mps" else 3e-2

        assert np.abs(image_slice.flatten() - expected_slice).max() < tolerance
