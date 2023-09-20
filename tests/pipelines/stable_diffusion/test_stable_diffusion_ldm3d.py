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


import gc
import unittest

import numpy as np
import torch
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer

from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    PNDMScheduler,
    StableDiffusionLDM3DPipeline,
    UNet2DConditionModel,
)
from diffusers.utils.testing_utils import enable_full_determinism, nightly, require_torch_gpu, torch_device

from ..pipeline_params import TEXT_TO_IMAGE_BATCH_PARAMS, TEXT_TO_IMAGE_IMAGE_PARAMS, TEXT_TO_IMAGE_PARAMS


enable_full_determinism()


class StableDiffusionLDM3DPipelineFastTests(unittest.TestCase):
    pipeline_class = StableDiffusionLDM3DPipeline
    params = TEXT_TO_IMAGE_PARAMS
    batch_params = TEXT_TO_IMAGE_BATCH_PARAMS
    image_params = TEXT_TO_IMAGE_IMAGE_PARAMS

    def get_dummy_components(self):
        torch.manual_seed(0)
        unet = UNet2DConditionModel(
            block_out_channels=(32, 64),
            layers_per_block=2,
            sample_size=32,
            in_channels=4,
            out_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
            cross_attention_dim=32,
        )
        scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
        )
        torch.manual_seed(0)
        vae = AutoencoderKL(
            block_out_channels=[32, 64],
            in_channels=6,
            out_channels=6,
            down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
            up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"],
            latent_channels=4,
        )
        torch.manual_seed(0)
        text_encoder_config = CLIPTextConfig(
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
        text_encoder = CLIPTextModel(text_encoder_config)
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        components = {
            "unet": unet,
            "scheduler": scheduler,
            "vae": vae,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "safety_checker": None,
            "feature_extractor": None,
        }
        return components

    def get_dummy_inputs(self, device, seed=0):
        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device=device).manual_seed(seed)
        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            "output_type": "numpy",
        }
        return inputs

    def test_stable_diffusion_ddim(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator

        components = self.get_dummy_components()
        ldm3d_pipe = StableDiffusionLDM3DPipeline(**components)
        ldm3d_pipe = ldm3d_pipe.to(torch_device)
        ldm3d_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        output = ldm3d_pipe(**inputs)
        rgb, depth = output.rgb, output.depth

        image_slice_rgb = rgb[0, -3:, -3:, -1]
        image_slice_depth = depth[0, -3:, -1]

        assert rgb.shape == (1, 64, 64, 3)
        assert depth.shape == (1, 64, 64)

        expected_slice_rgb = np.array(
            [0.37338176, 0.70247, 0.74203193, 0.51643604, 0.58256793, 0.60932136, 0.4181095, 0.48355877, 0.46535262]
        )
        expected_slice_depth = np.array([103.46727, 85.812004, 87.849236])

        assert np.abs(image_slice_rgb.flatten() - expected_slice_rgb).max() < 1e-2
        assert np.abs(image_slice_depth.flatten() - expected_slice_depth).max() < 1e-2

    def test_stable_diffusion_prompt_embeds(self):
        components = self.get_dummy_components()
        ldm3d_pipe = StableDiffusionLDM3DPipeline(**components)
        ldm3d_pipe = ldm3d_pipe.to(torch_device)
        ldm3d_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(torch_device)
        inputs["prompt"] = 3 * [inputs["prompt"]]

        # forward
        output = ldm3d_pipe(**inputs)
        rgb_slice_1, depth_slice_1 = output.rgb, output.depth
        rgb_slice_1 = rgb_slice_1[0, -3:, -3:, -1]
        depth_slice_1 = depth_slice_1[0, -3:, -1]

        inputs = self.get_dummy_inputs(torch_device)
        prompt = 3 * [inputs.pop("prompt")]

        text_inputs = ldm3d_pipe.tokenizer(
            prompt,
            padding="max_length",
            max_length=ldm3d_pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_inputs = text_inputs["input_ids"].to(torch_device)

        prompt_embeds = ldm3d_pipe.text_encoder(text_inputs)[0]

        inputs["prompt_embeds"] = prompt_embeds

        # forward
        output = ldm3d_pipe(**inputs)
        rgb_slice_2, depth_slice_2 = output.rgb, output.depth
        rgb_slice_2 = rgb_slice_2[0, -3:, -3:, -1]
        depth_slice_2 = depth_slice_2[0, -3:, -1]

        assert np.abs(rgb_slice_1.flatten() - rgb_slice_2.flatten()).max() < 1e-4
        assert np.abs(depth_slice_1.flatten() - depth_slice_2.flatten()).max() < 1e-4

    def test_stable_diffusion_negative_prompt(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()
        components["scheduler"] = PNDMScheduler(skip_prk_steps=True)
        ldm3d_pipe = StableDiffusionLDM3DPipeline(**components)
        ldm3d_pipe = ldm3d_pipe.to(device)
        ldm3d_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        negative_prompt = "french fries"
        output = ldm3d_pipe(**inputs, negative_prompt=negative_prompt)

        rgb, depth = output.rgb, output.depth
        rgb_slice = rgb[0, -3:, -3:, -1]
        depth_slice = depth[0, -3:, -1]

        assert rgb.shape == (1, 64, 64, 3)
        assert depth.shape == (1, 64, 64)

        expected_slice_rgb = np.array(
            [0.37044, 0.71811503, 0.7223251, 0.48603675, 0.5638391, 0.6364948, 0.42833704, 0.4901315, 0.47926217]
        )
        expected_slice_depth = np.array([107.84738, 84.62802, 89.962135])
        assert np.abs(rgb_slice.flatten() - expected_slice_rgb).max() < 1e-2
        assert np.abs(depth_slice.flatten() - expected_slice_depth).max() < 1e-2


@nightly
@require_torch_gpu
class StableDiffusionLDM3DPipelineSlowTests(unittest.TestCase):
    def tearDown(self):
        super().tearDown()
        gc.collect()
        torch.cuda.empty_cache()

    def get_inputs(self, device, generator_device="cpu", dtype=torch.float32, seed=0):
        generator = torch.Generator(device=generator_device).manual_seed(seed)
        latents = np.random.RandomState(seed).standard_normal((1, 4, 64, 64))
        latents = torch.from_numpy(latents).to(device=device, dtype=dtype)
        inputs = {
            "prompt": "a photograph of an astronaut riding a horse",
            "latents": latents,
            "generator": generator,
            "num_inference_steps": 3,
            "guidance_scale": 7.5,
            "output_type": "numpy",
        }
        return inputs

    def test_ldm3d_stable_diffusion(self):
        ldm3d_pipe = StableDiffusionLDM3DPipeline.from_pretrained("Intel/ldm3d")
        ldm3d_pipe = ldm3d_pipe.to(torch_device)
        ldm3d_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs(torch_device)
        output = ldm3d_pipe(**inputs)
        rgb, depth = output.rgb, output.depth
        rgb_slice = rgb[0, -3:, -3:, -1].flatten()
        depth_slice = rgb[0, -3:, -1].flatten()

        assert rgb.shape == (1, 512, 512, 3)
        assert depth.shape == (1, 512, 512)

        expected_slice_rgb = np.array(
            [0.53805465, 0.56707305, 0.5486515, 0.57012236, 0.5814511, 0.56253487, 0.54843014, 0.55092263, 0.6459706]
        )
        expected_slice_depth = np.array(
            [0.9263781, 0.6678672, 0.5486515, 0.92202145, 0.67831135, 0.56253487, 0.9241694, 0.7551478, 0.6459706]
        )
        assert np.abs(rgb_slice - expected_slice_rgb).max() < 3e-3
        assert np.abs(depth_slice - expected_slice_depth).max() < 3e-3


@nightly
@require_torch_gpu
class StableDiffusionPipelineNightlyTests(unittest.TestCase):
    def tearDown(self):
        super().tearDown()
        gc.collect()
        torch.cuda.empty_cache()

    def get_inputs(self, device, generator_device="cpu", dtype=torch.float32, seed=0):
        generator = torch.Generator(device=generator_device).manual_seed(seed)
        latents = np.random.RandomState(seed).standard_normal((1, 4, 64, 64))
        latents = torch.from_numpy(latents).to(device=device, dtype=dtype)
        inputs = {
            "prompt": "a photograph of an astronaut riding a horse",
            "latents": latents,
            "generator": generator,
            "num_inference_steps": 50,
            "guidance_scale": 7.5,
            "output_type": "numpy",
        }
        return inputs

    def test_ldm3d(self):
        ldm3d_pipe = StableDiffusionLDM3DPipeline.from_pretrained("Intel/ldm3d").to(torch_device)
        ldm3d_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs(torch_device)
        output = ldm3d_pipe(**inputs)
        rgb, depth = output.rgb, output.depth

        expected_rgb_mean = 0.495586
        expected_rgb_std = 0.33795515
        expected_depth_mean = 112.48518
        expected_depth_std = 98.489746
        assert np.abs(expected_rgb_mean - rgb.mean()) < 1e-3
        assert np.abs(expected_rgb_std - rgb.std()) < 1e-3
        assert np.abs(expected_depth_mean - depth.mean()) < 1e-3
        assert np.abs(expected_depth_std - depth.std()) < 1e-3

    def test_ldm3d_v2(self):
        ldm3d_pipe = StableDiffusionLDM3DPipeline.from_pretrained("Intel/ldm3d-4c").to(torch_device)
        ldm3d_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs(torch_device)
        output = ldm3d_pipe(**inputs)
        rgb, depth = output.rgb, output.depth

        expected_rgb_mean = 0.4194127
        expected_rgb_std = 0.35375586
        expected_depth_mean = 0.5638502
        expected_depth_std = 0.34686103

        assert rgb.shape == (1, 512, 512, 3)
        assert depth.shape == (1, 512, 512, 1)
        assert np.abs(expected_rgb_mean - rgb.mean()) < 1e-3
        assert np.abs(expected_rgb_std - rgb.std()) < 1e-3
        assert np.abs(expected_depth_mean - depth.mean()) < 1e-3
        assert np.abs(expected_depth_std - depth.std()) < 1e-3
