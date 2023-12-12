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
import random
import unittest

import numpy as np
import torch
from PIL import Image
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer

from diffusers import (
    AutoencoderKL,
    DPMSolverMultistepScheduler,
    LEditsPPPipelineStableDiffusion,
    UNet2DConditionModel,
)

# from diffusers.image_processor import VaeImageProcessor
from diffusers.utils.testing_utils import (
    enable_full_determinism,
    floats_tensor,
    load_image,
    require_torch_gpu,
    skip_mps,
    slow,
    torch_device,
)


# from ..test_pipelines_common import PipelineKarrasSchedulerTesterMixin, PipelineLatentTesterMixin, PipelineTesterMixin
# from ..test_pipelines_common import PipelineLatentTesterMixin, PipelineTesterMixin


enable_full_determinism()


@skip_mps
class LEditsPPPipelineStableDiffusionFastTests(
    #    PipelineLatentTesterMixin, PipelineTesterMixin,
    unittest.TestCase
):
    pipeline_class = LEditsPPPipelineStableDiffusion
    # params = TEXT_GUIDED_IMAGE_VARIATION_PARAMS - {"height", "width", "cross_attention_kwargs"}
    # batch_params = TEXT_GUIDED_IMAGE_INPAINTING_BATCH_PARAMS
    # image_params = IMAGE_TO_IMAGE_IMAGE_PARAMS
    # image_latents_params = IMAGE_TO_IMAGE_IMAGE_PARAMS
    # callback_cfg_params = TEXT_TO_IMAGE_CALLBACK_CFG_PARAMS.union({"image_latents"}) - {"negative_prompt_embeds"}

    def get_dummy_components(self):
        torch.manual_seed(0)
        unet = UNet2DConditionModel(
            block_out_channels=(32, 64, 64),
            layers_per_block=2,
            sample_size=32,
            in_channels=4,
            out_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D"),
            cross_attention_dim=32,
        )
        scheduler = DPMSolverMultistepScheduler(algorithm_type="sde-dpmsolver++", solver_order=2)
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
            "generator": generator,
            "editing_prompt": ["wearing glasses", "sunshine"],
            "reverse_editing_direction": [False, True],
            "edit_guidance_scale": [10.0, 5.0],
        }
        return inputs

    def get_dummy_inversion_inputs(self, device, seed=0):
        images = floats_tensor((2, 3, 32, 32), rng=random.Random(0)).cpu().permute(0, 2, 3, 1)
        images = 255 * images
        image_1 = Image.fromarray(np.uint8(images[0])).convert("RGB")
        image_2 = Image.fromarray(np.uint8(images[1])).convert("RGB")

        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device=device).manual_seed(seed)

        inputs = {
            "image": [image_1, image_2],
            "source_prompt": "",
            "source_guidance_scale": 3.5,
            "num_inversion_steps": 20,
            "skip": 0.15,
            "generator": generator,
        }
        return inputs

    def test_ledits_pp_inversion(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()
        sd_pipe = LEditsPPPipelineStableDiffusion(**components)
        sd_pipe = sd_pipe.to(device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inversion_inputs(device)
        inputs["image"] = inputs["image"][0]
        sd_pipe.invert(**inputs)
        assert sd_pipe.init_latents.shape == (1, 4, 32, 32)

        latent_slice = sd_pipe.init_latents[0, -1, -3:, -3:]
        print(latent_slice.flatten())
        expected_slice = np.array([0.2591, -0.8992, 0.0135, 0.8945, 0.6812, -0.3078, -0.1397, 0.9919, 0.7559])
        assert np.abs(latent_slice.flatten() - expected_slice).max() < 1e-3

    def test_ledits_pp_inversion_batch(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()
        sd_pipe = LEditsPPPipelineStableDiffusion(**components)
        sd_pipe = sd_pipe.to(device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inversion_inputs(device)
        sd_pipe.invert(**inputs)
        assert sd_pipe.init_latents.shape == (2, 4, 32, 32)

        latent_slice = sd_pipe.init_latents[0, -1, -3:, -3:]
        print(latent_slice.flatten())
        expected_slice = np.array([0.9312, 0.9823, 0.8902, 0.2546, -0.4952, -0.3061, -0.1626, -1.5873, -1.0343])
        assert np.abs(latent_slice.flatten() - expected_slice).max() < 1e-3

        latent_slice = sd_pipe.init_latents[1, -1, -3:, -3:]
        print(latent_slice.flatten())
        expected_slice = np.array([-1.4121, -1.8028, 1.3368, -0.4232, 0.1431, 0.5343, -0.2898, -0.0392, 0.5292])
        assert np.abs(latent_slice.flatten() - expected_slice).max() < 1e-3

    def test_ledits_pp_warmup_steps(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()
        pipe = LEditsPPPipelineStableDiffusion(**components)
        pipe = pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        inversion_inputs = self.get_dummy_inversion_inputs(device)
        pipe.invert(**inversion_inputs)

        inputs = self.get_dummy_inputs(torch_device)

        inputs["edit_warmup_steps"] = [0, 5]
        pipe(**inputs).images

        inputs["edit_warmup_steps"] = [5, 0]
        pipe(**inputs).images

        inputs["edit_warmup_steps"] = [5, 10]
        pipe(**inputs).images

        inputs["edit_warmup_steps"] = [10, 5]
        pipe(**inputs).images


@slow
@require_torch_gpu
class LEditsPPPipelineStableDiffusionSlowTests(unittest.TestCase):
    def tearDown(self):
        super().tearDown()
        gc.collect()
        torch.cuda.empty_cache()

    @classmethod
    def setUpClass(cls):
        raw_image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/pix2pix/cat_6.png"
        )
        raw_image = raw_image.convert("RGB").resize((512, 512))
        cls.raw_image = raw_image

    def test_ledits_pp_perfect_reconstruction(self):
        pipe = LEditsPPPipelineStableDiffusion.from_pretrained(
            "runwayml/stable-diffusion-v1-5", safety_checker=None, torch_dtype=torch.float16
        )
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        generator = torch.manual_seed(0)
        inversion_output = pipe.invert(image=self.raw_image, generator=generator)

        reconstruction = pipe(output_type="np", generator=generator).images[0]
        image = np.array(inversion_output.vae_reconstruction_images[0]).astype(np.float32) / 255.0

        print(np.abs(image - reconstruction).max())
        assert np.abs(image - reconstruction).max() < 1e-2
