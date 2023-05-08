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

from diffusers import AutoencoderKL, DDIMScheduler, LDMTextToImagePipeline, UNet2DConditionModel
from diffusers.utils.testing_utils import load_numpy, nightly, require_torch_gpu, slow, torch_device

from ..pipeline_params import TEXT_TO_IMAGE_BATCH_PARAMS, TEXT_TO_IMAGE_PARAMS
from ..test_pipelines_common import PipelineTesterMixin


torch.backends.cuda.matmul.allow_tf32 = False


class LDMTextToImagePipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = LDMTextToImagePipeline
    params = TEXT_TO_IMAGE_PARAMS - {
        "negative_prompt",
        "negative_prompt_embeds",
        "cross_attention_kwargs",
        "prompt_embeds",
    }
    required_optional_params = PipelineTesterMixin.required_optional_params - {
        "num_images_per_prompt",
        "callback",
        "callback_steps",
    }
    batch_params = TEXT_TO_IMAGE_BATCH_PARAMS
    test_cpu_offload = False

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
            block_out_channels=(32, 64),
            in_channels=3,
            out_channels=3,
            down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D"),
            up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D"),
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
            "vqvae": vae,
            "bert": text_encoder,
            "tokenizer": tokenizer,
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

    def test_inference_text2img(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator

        components = self.get_dummy_components()
        pipe = LDMTextToImagePipeline(**components)
        pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        image = pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 16, 16, 3)
        expected_slice = np.array([0.6101, 0.6156, 0.5622, 0.4895, 0.6661, 0.3804, 0.5748, 0.6136, 0.5014])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-3


@slow
@require_torch_gpu
class LDMTextToImagePipelineSlowTests(unittest.TestCase):
    def tearDown(self):
        super().tearDown()
        gc.collect()
        torch.cuda.empty_cache()

    def get_inputs(self, device, dtype=torch.float32, seed=0):
        generator = torch.manual_seed(seed)
        latents = np.random.RandomState(seed).standard_normal((1, 4, 32, 32))
        latents = torch.from_numpy(latents).to(device=device, dtype=dtype)
        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "latents": latents,
            "generator": generator,
            "num_inference_steps": 3,
            "guidance_scale": 6.0,
            "output_type": "numpy",
        }
        return inputs

    def test_ldm_default_ddim(self):
        pipe = LDMTextToImagePipeline.from_pretrained("CompVis/ldm-text2im-large-256").to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs(torch_device)
        image = pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1].flatten()

        assert image.shape == (1, 256, 256, 3)
        expected_slice = np.array([0.51825, 0.52850, 0.52543, 0.54258, 0.52304, 0.52569, 0.54363, 0.55276, 0.56878])
        max_diff = np.abs(expected_slice - image_slice).max()
        assert max_diff < 1e-3


@nightly
@require_torch_gpu
class LDMTextToImagePipelineNightlyTests(unittest.TestCase):
    def tearDown(self):
        super().tearDown()
        gc.collect()
        torch.cuda.empty_cache()

    def get_inputs(self, device, dtype=torch.float32, seed=0):
        generator = torch.manual_seed(seed)
        latents = np.random.RandomState(seed).standard_normal((1, 4, 32, 32))
        latents = torch.from_numpy(latents).to(device=device, dtype=dtype)
        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "latents": latents,
            "generator": generator,
            "num_inference_steps": 50,
            "guidance_scale": 6.0,
            "output_type": "numpy",
        }
        return inputs

    def test_ldm_default_ddim(self):
        pipe = LDMTextToImagePipeline.from_pretrained("CompVis/ldm-text2im-large-256").to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs(torch_device)
        image = pipe(**inputs).images[0]

        expected_image = load_numpy(
            "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main/ldm_text2img/ldm_large_256_ddim.npy"
        )
        max_diff = np.abs(expected_image - image).max()
        assert max_diff < 1e-3
