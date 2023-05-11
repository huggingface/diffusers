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
    StableDiffusionSAGPipeline,
    UNet2DConditionModel,
)
from diffusers.utils import slow, torch_device
from diffusers.utils.testing_utils import require_torch_gpu

from ..pipeline_params import TEXT_TO_IMAGE_BATCH_PARAMS, TEXT_TO_IMAGE_IMAGE_PARAMS, TEXT_TO_IMAGE_PARAMS
from ..test_pipelines_common import PipelineLatentTesterMixin, PipelineTesterMixin


torch.backends.cuda.matmul.allow_tf32 = False
torch.use_deterministic_algorithms(True)


class StableDiffusionSAGPipelineFastTests(PipelineLatentTesterMixin, PipelineTesterMixin, unittest.TestCase):
    pipeline_class = StableDiffusionSAGPipeline
    params = TEXT_TO_IMAGE_PARAMS
    batch_params = TEXT_TO_IMAGE_BATCH_PARAMS
    image_params = TEXT_TO_IMAGE_IMAGE_PARAMS
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
            block_out_channels=[32, 64],
            in_channels=3,
            out_channels=3,
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
            "prompt": ".",
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 1.0,
            "sag_scale": 1.0,
            "output_type": "numpy",
        }
        return inputs

    def test_inference_batch_single_identical(self):
        super().test_inference_batch_single_identical(expected_max_diff=3e-3)


@slow
@require_torch_gpu
class StableDiffusionPipelineIntegrationTests(unittest.TestCase):
    def tearDown(self):
        # clean up the VRAM after each test
        super().tearDown()
        gc.collect()
        torch.cuda.empty_cache()

    def test_stable_diffusion_1(self):
        sag_pipe = StableDiffusionSAGPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
        sag_pipe = sag_pipe.to(torch_device)
        sag_pipe.set_progress_bar_config(disable=None)

        prompt = "."
        generator = torch.manual_seed(0)
        output = sag_pipe(
            [prompt], generator=generator, guidance_scale=7.5, sag_scale=1.0, num_inference_steps=20, output_type="np"
        )

        image = output.images

        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array([0.1568, 0.1738, 0.1695, 0.1693, 0.1507, 0.1705, 0.1547, 0.1751, 0.1949])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 5e-2

    def test_stable_diffusion_2(self):
        sag_pipe = StableDiffusionSAGPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-base")
        sag_pipe = sag_pipe.to(torch_device)
        sag_pipe.set_progress_bar_config(disable=None)

        prompt = "."
        generator = torch.manual_seed(0)
        output = sag_pipe(
            [prompt], generator=generator, guidance_scale=7.5, sag_scale=1.0, num_inference_steps=20, output_type="np"
        )

        image = output.images

        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array([0.3459, 0.2876, 0.2537, 0.3002, 0.2671, 0.2160, 0.3026, 0.2262, 0.2371])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 5e-2

    def test_stable_diffusion_2_non_square(self):
        sag_pipe = StableDiffusionSAGPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-base")
        sag_pipe = sag_pipe.to(torch_device)
        sag_pipe.set_progress_bar_config(disable=None)

        prompt = "."
        generator = torch.manual_seed(0)
        output = sag_pipe(
            [prompt],
            width=768,
            height=512,
            generator=generator,
            guidance_scale=7.5,
            sag_scale=1.0,
            num_inference_steps=20,
            output_type="np",
        )

        image = output.images

        assert image.shape == (1, 512, 768, 3)
