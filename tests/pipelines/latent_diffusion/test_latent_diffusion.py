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

from diffusers import AutoencoderKL, DDIMScheduler, LDMTextToImagePipeline, UNet2DConditionModel
from diffusers.utils.testing_utils import require_torch, slow, torch_device
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer


torch.backends.cuda.matmul.allow_tf32 = False


class LDMTextToImagePipelineFastTests(unittest.TestCase):
    @property
    def dummy_cond_unet(self):
        torch.manual_seed(0)
        model = UNet2DConditionModel(
            block_out_channels=(32, 64),
            layers_per_block=2,
            sample_size=32,
            in_channels=4,
            out_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
            cross_attention_dim=32,
        )
        return model

    @property
    def dummy_vae(self):
        torch.manual_seed(0)
        model = AutoencoderKL(
            block_out_channels=[32, 64],
            in_channels=3,
            out_channels=3,
            down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
            up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"],
            latent_channels=4,
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

    def test_inference_text2img(self):
        if torch_device != "cpu":
            return

        unet = self.dummy_cond_unet
        scheduler = DDIMScheduler()
        vae = self.dummy_vae
        bert = self.dummy_text_encoder
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        ldm = LDMTextToImagePipeline(vqvae=vae, bert=bert, tokenizer=tokenizer, unet=unet, scheduler=scheduler)
        ldm.to(torch_device)
        ldm.set_progress_bar_config(disable=None)

        prompt = "A painting of a squirrel eating a burger"

        # Warmup pass when using mps (see #372)
        if torch_device == "mps":
            generator = torch.manual_seed(0)
            _ = ldm(
                [prompt], generator=generator, guidance_scale=6.0, num_inference_steps=1, output_type="numpy"
            ).images

        device = torch_device if torch_device != "mps" else "cpu"
        generator = torch.Generator(device=device).manual_seed(0)

        image = ldm(
            [prompt], generator=generator, guidance_scale=6.0, num_inference_steps=2, output_type="numpy"
        ).images

        device = torch_device if torch_device != "mps" else "cpu"
        generator = torch.Generator(device=device).manual_seed(0)

        image_from_tuple = ldm(
            [prompt],
            generator=generator,
            guidance_scale=6.0,
            num_inference_steps=2,
            output_type="numpy",
            return_dict=False,
        )[0]

        image_slice = image[0, -3:, -3:, -1]
        image_from_tuple_slice = image_from_tuple[0, -3:, -3:, -1]

        assert image.shape == (1, 16, 16, 3)
        expected_slice = np.array([0.6806, 0.5454, 0.5638, 0.4893, 0.4656, 0.4257, 0.6248, 0.5217, 0.5498])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2
        assert np.abs(image_from_tuple_slice.flatten() - expected_slice).max() < 1e-2


@slow
@require_torch
class LDMTextToImagePipelineIntegrationTests(unittest.TestCase):
    def test_inference_text2img(self):
        ldm = LDMTextToImagePipeline.from_pretrained("CompVis/ldm-text2im-large-256")
        ldm.to(torch_device)
        ldm.set_progress_bar_config(disable=None)

        prompt = "A painting of a squirrel eating a burger"

        device = torch_device if torch_device != "mps" else "cpu"
        generator = torch.Generator(device=device).manual_seed(0)

        image = ldm(
            [prompt], generator=generator, guidance_scale=6.0, num_inference_steps=20, output_type="numpy"
        ).images

        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 256, 256, 3)
        expected_slice = np.array([0.9256, 0.9340, 0.8933, 0.9361, 0.9113, 0.8727, 0.9122, 0.8745, 0.8099])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

    def test_inference_text2img_fast(self):
        ldm = LDMTextToImagePipeline.from_pretrained("CompVis/ldm-text2im-large-256")
        ldm.to(torch_device)
        ldm.set_progress_bar_config(disable=None)

        prompt = "A painting of a squirrel eating a burger"

        device = torch_device if torch_device != "mps" else "cpu"
        generator = torch.Generator(device=device).manual_seed(0)

        image = ldm(prompt, generator=generator, num_inference_steps=1, output_type="numpy").images

        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 256, 256, 3)
        expected_slice = np.array([0.3163, 0.8670, 0.6465, 0.1865, 0.6291, 0.5139, 0.2824, 0.3723, 0.4344])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2
