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
import requests
import numpy as np
import torch
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer
from PIL import Image
from io import BytesIO
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DDPMScheduler,
    StableDiffusionUpscaleLDM3DPipeline,
    UNet2DConditionModel,
)
from diffusers.utils.testing_utils import enable_full_determinism, nightly, require_torch_gpu, torch_device

# from ..pipeline_params import TEXT_TO_IMAGE_BATCH_PARAMS, TEXT_TO_IMAGE_IMAGE_PARAMS, TEXT_TO_IMAGE_PARAMS
from pipeline_params import TEXT_TO_IMAGE_BATCH_PARAMS, TEXT_TO_IMAGE_IMAGE_PARAMS, TEXT_TO_IMAGE_PARAMS

enable_full_determinism()

def load_image(res=64):
    rgb_path = "https://huggingface.co/Intel/ldm3d-sr/resolve/main/lemons_ldm3d_rgb.jpg"
    depth_path = "https://huggingface.co/Intel/ldm3d-sr/resolve/main/lemons_ldm3d_depth.png"
    low_res_rgb = Image.open(BytesIO(requests.get(rgb_path).content)).convert("RGB").resize((res, res))
    low_res_depth = Image.open(BytesIO(requests.get(depth_path).content)).convert("L").resize((res, res))
    return low_res_rgb, low_res_depth

class StableDiffusionUpscaleLDM3DPipelineFastTests(unittest.TestCase):
    pipeline_class = StableDiffusionUpscaleLDM3DPipeline
    params = TEXT_TO_IMAGE_PARAMS
    batch_params = TEXT_TO_IMAGE_BATCH_PARAMS
    image_params = TEXT_TO_IMAGE_IMAGE_PARAMS

    def get_dummy_components(self):
        torch.manual_seed(0)
        unet = UNet2DConditionModel(
            block_out_channels=(32, 64),
            layers_per_block=2,
            sample_size=32,
            in_channels=8,
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
            in_channels=4,
            out_channels=4,
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
            "low_res_scheduler": DDPMScheduler,
        }
        return components
    
    def get_dummy_inputs(self, device, seed=0):
        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device=device).manual_seed(seed)
        target_res = 128
        rgb, depth = load_image(res=64)
        inputs = {
            "rgb": rgb,
            "depth": depth,
            "prompt": "high quality high resolution uhd 4k image",
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            "output_type": "numpy",
            "target_res": [target_res, target_res]
            
        }
        return inputs

    def test_stable_diffusion_ddim(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator

        components = self.get_dummy_components()
        ldm3d_pipe = StableDiffusionUpscaleLDM3DPipeline(**components)
        ldm3d_pipe = ldm3d_pipe.to(torch_device)
        ldm3d_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        target_res, target_res = inputs['target_res']
        output = ldm3d_pipe(**inputs)
        rgb, depth = output.rgb, output.depth

        image_slice_rgb = rgb[0, -3:, -3:, -1]
        image_slice_depth = depth[0, -3:, -1]

        assert rgb.shape == (1, target_res, target_res, 3)
        assert depth.shape == (1, target_res, target_res, 1)

        expected_slice_rgb = np.array(
            [0.66053814, 0.74662584, 0.6431943,  0.74867487, 0.71609235, 0.5595806, 0.5559061,  0.54281414, 0.58524626]
        )
        expected_slice_depth = np.array([0.30766925, 0.4785453,  0.5751028 ])

        assert np.abs(image_slice_rgb.flatten() - expected_slice_rgb).max() < 1e-2
        assert np.abs(image_slice_depth.flatten() - expected_slice_depth).max() < 1e-2


@nightly
@require_torch_gpu
class StableDiffusionUpscaleLDM3DPipelineSlowTests(unittest.TestCase):
    def tearDown(self):
        super().tearDown()
        gc.collect()
        torch.cuda.empty_cache()

    def get_inputs(self, device, generator_device="cpu", dtype=torch.float32, seed=0):
        generator = torch.Generator(device=generator_device).manual_seed(seed)
        res = 128
        latents = np.random.RandomState(seed).standard_normal((1, 4, int(res/8), int(res/8)))
        latents = torch.from_numpy(latents).to(device=device, dtype=dtype)
        rgb, depth = load_image(res=64)
        inputs = {
            "prompt": "high quality high resolution uhd 4k image",
            "rgb": rgb,
            "depth": depth,
            "latents": latents,
            "generator": generator,
            "num_inference_steps": 3,
            "guidance_scale": 7.5,
            "output_type": "numpy",
            "target_res": [res, res]
            }
        return inputs

    def test_ldm3d_stable_diffusion(self):
        ldm3d_pipe = StableDiffusionUpscaleLDM3DPipeline.from_pretrained("Intel/ldm3d-sr")
        ldm3d_pipe = ldm3d_pipe.to(torch_device)
        ldm3d_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs(torch_device)
        output = ldm3d_pipe(**inputs)
        rgb, depth = output.rgb, output.depth
        rgb_slice = rgb[0, -3:, -3:, -1].flatten()
        depth_slice = rgb[0, -3:, -1].flatten()
        res, res = inputs["target_res"]
        assert rgb.shape == (1, res, res, 3)
        assert depth.shape == (1, res, res, 1)

        expected_slice_rgb = np.array(
            [0.70745385, 0.7183809,  0.6794662,  0.6393902 , 0.64071167, 0.6380512,0.5901599,  0.5722568,  0.56403804]
        )
        expected_slice_depth = np.array(
            [0.68841934, 0.6916517,  0.6794662,  0.64727414, 0.65219307, 0.6380512, 0.58779067, 0.5852662,  0.56403804]
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
        target_res = 128
        latents = np.random.RandomState(seed).standard_normal((1, 4, int(target_res/8), int(target_res/8)))
        latents = torch.from_numpy(latents).to(device=device, dtype=dtype)
        rgb, depth = load_image(res=64)
        inputs = {
            "prompt": "high quality high resolution uhd 4k image",
            "rgb": rgb,
            "depth": depth,
            "latents": latents,
            "generator": generator,
            "num_inference_steps": 50,
            "guidance_scale": 7.5,
            "output_type": "numpy",
            "target_res": [target_res, target_res]
        }
        return inputs

    def test_ldm3d(self):
        ldm3d_pipe = StableDiffusionUpscaleLDM3DPipeline.from_pretrained("Intel/ldm3d-sr").to(torch_device)
        ldm3d_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs(torch_device)
        output = ldm3d_pipe(**inputs)
        rgb, depth = output.rgb, output.depth

        expected_rgb_mean = 0.53158706
        expected_rgb_std = 0.38129386
        expected_depth_mean = 0.19618212
        expected_depth_std = 0.07465989

        assert np.abs(expected_rgb_mean - rgb.mean()) < 1e-3
        assert np.abs(expected_rgb_std - rgb.std()) < 1e-3
        assert np.abs(expected_depth_mean - depth.mean()) < 1e-3
        assert np.abs(expected_depth_std - depth.std()) < 1e-3

    def test_ldm3d_v2(self):
        ldm3d_pipe = StableDiffusionUpscaleLDM3DPipeline.from_pretrained("Intel/ldm3d-sr").to(torch_device)
        ldm3d_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs(torch_device)
        output = ldm3d_pipe(**inputs)
        res, res = inputs['target_res']
        rgb, depth = output.rgb, output.depth

        expected_rgb_mean = 0.53158706
        expected_rgb_std = 0.38129386
        expected_depth_mean = 0.19618212
        expected_depth_std = 0.07465989

        assert rgb.shape == (1, res, res, 3)
        assert depth.shape == (1, res, res, 1)
        assert np.abs(expected_rgb_mean - rgb.mean()) < 1e-3
        assert np.abs(expected_rgb_std - rgb.std()) < 1e-3
        assert np.abs(expected_depth_mean - depth.mean()) < 1e-3
        assert np.abs(expected_depth_std - depth.std()) < 1e-3
