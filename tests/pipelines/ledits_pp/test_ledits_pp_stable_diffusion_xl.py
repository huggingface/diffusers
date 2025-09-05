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

import random
import unittest

import numpy as np
import torch
from PIL import Image
from transformers import (
    CLIPImageProcessor,
    CLIPTextConfig,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    CLIPVisionConfig,
    CLIPVisionModelWithProjection,
)

from diffusers import (
    AutoencoderKL,
    DPMSolverMultistepScheduler,
    LEditsPPPipelineStableDiffusionXL,
    UNet2DConditionModel,
)

# from diffusers.image_processor import VaeImageProcessor
from ...testing_utils import (
    enable_full_determinism,
    floats_tensor,
    load_image,
    require_torch_accelerator,
    skip_mps,
    slow,
    torch_device,
)


enable_full_determinism()


@skip_mps
class LEditsPPPipelineStableDiffusionXLFastTests(unittest.TestCase):
    pipeline_class = LEditsPPPipelineStableDiffusionXL

    def get_dummy_components(self, skip_first_text_encoder=False, time_cond_proj_dim=None):
        torch.manual_seed(0)
        unet = UNet2DConditionModel(
            block_out_channels=(32, 64),
            layers_per_block=2,
            sample_size=32,
            in_channels=4,
            out_channels=4,
            time_cond_proj_dim=time_cond_proj_dim,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
            # SD2-specific config below
            attention_head_dim=(2, 4),
            use_linear_projection=True,
            addition_embed_type="text_time",
            addition_time_embed_dim=8,
            transformer_layers_per_block=(1, 2),
            projection_class_embeddings_input_dim=80,  # 6 * 8 + 32
            cross_attention_dim=64 if not skip_first_text_encoder else 32,
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
            sample_size=128,
        )
        torch.manual_seed(0)
        image_encoder_config = CLIPVisionConfig(
            hidden_size=32,
            image_size=224,
            projection_dim=32,
            intermediate_size=37,
            num_attention_heads=4,
            num_channels=3,
            num_hidden_layers=5,
            patch_size=14,
        )
        image_encoder = CLIPVisionModelWithProjection(image_encoder_config)

        feature_extractor = CLIPImageProcessor(
            crop_size=224,
            do_center_crop=True,
            do_normalize=True,
            do_resize=True,
            image_mean=[0.48145466, 0.4578275, 0.40821073],
            image_std=[0.26862954, 0.26130258, 0.27577711],
            resample=3,
            size=224,
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
            # SD2-specific config below
            hidden_act="gelu",
            projection_dim=32,
        )
        text_encoder = CLIPTextModel(text_encoder_config)
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        text_encoder_2 = CLIPTextModelWithProjection(text_encoder_config)
        tokenizer_2 = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        components = {
            "unet": unet,
            "scheduler": scheduler,
            "vae": vae,
            "text_encoder": text_encoder if not skip_first_text_encoder else None,
            "tokenizer": tokenizer if not skip_first_text_encoder else None,
            "text_encoder_2": text_encoder_2,
            "tokenizer_2": tokenizer_2,
            "image_encoder": image_encoder,
            "feature_extractor": feature_extractor,
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
        sd_pipe = LEditsPPPipelineStableDiffusionXL(**components)
        sd_pipe = sd_pipe.to(torch_device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inversion_inputs(device)
        inputs["image"] = inputs["image"][0]
        sd_pipe.invert(**inputs)
        assert sd_pipe.init_latents.shape == (
            1,
            4,
            int(32 / sd_pipe.vae_scale_factor),
            int(32 / sd_pipe.vae_scale_factor),
        )

        latent_slice = sd_pipe.init_latents[0, -1, -3:, -3:].to(device)
        expected_slice = np.array([-0.9084, -0.0367, 0.2940, 0.0839, 0.6890, 0.2651, -0.7103, 2.1090, -0.7821])
        assert np.abs(latent_slice.flatten() - expected_slice).max() < 1e-3

    def test_ledits_pp_inversion_batch(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()
        sd_pipe = LEditsPPPipelineStableDiffusionXL(**components)
        sd_pipe = sd_pipe.to(torch_device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inversion_inputs(device)
        sd_pipe.invert(**inputs)
        assert sd_pipe.init_latents.shape == (
            2,
            4,
            int(32 / sd_pipe.vae_scale_factor),
            int(32 / sd_pipe.vae_scale_factor),
        )

        latent_slice = sd_pipe.init_latents[0, -1, -3:, -3:].to(device)

        expected_slice = np.array([0.2528, 0.1458, -0.2166, 0.4565, -0.5656, -1.0286, -0.9961, 0.5933, 1.1172])
        assert np.abs(latent_slice.flatten() - expected_slice).max() < 1e-3

        latent_slice = sd_pipe.init_latents[1, -1, -3:, -3:].to(device)

        expected_slice = np.array([-0.0796, 2.0583, 0.5500, 0.5358, 0.0282, -0.2803, -1.0470, 0.7024, -0.0072])

        assert np.abs(latent_slice.flatten() - expected_slice).max() < 1e-3

    def test_ledits_pp_warmup_steps(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()
        pipe = LEditsPPPipelineStableDiffusionXL(**components)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        inversion_inputs = self.get_dummy_inversion_inputs(device)
        inversion_inputs["image"] = inversion_inputs["image"][0]
        pipe.invert(**inversion_inputs)

        inputs = self.get_dummy_inputs(device)

        inputs["edit_warmup_steps"] = [0, 5]
        pipe(**inputs).images

        inputs["edit_warmup_steps"] = [5, 0]
        pipe(**inputs).images

        inputs["edit_warmup_steps"] = [5, 10]
        pipe(**inputs).images

        inputs["edit_warmup_steps"] = [10, 5]
        pipe(**inputs).images


@slow
@require_torch_accelerator
class LEditsPPPipelineStableDiffusionXLSlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        raw_image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/pix2pix/cat_6.png"
        )
        raw_image = raw_image.convert("RGB").resize((512, 512))
        cls.raw_image = raw_image

    def test_ledits_pp_edit(self):
        pipe = LEditsPPPipelineStableDiffusionXL.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", safety_checker=None, add_watermarker=None
        )
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        generator = torch.manual_seed(0)
        _ = pipe.invert(image=self.raw_image, generator=generator, num_zero_noise_steps=0)
        inputs = {
            "generator": generator,
            "editing_prompt": ["cat", "dog"],
            "reverse_editing_direction": [True, False],
            "edit_guidance_scale": [2.0, 4.0],
            "edit_threshold": [0.8, 0.8],
        }
        reconstruction = pipe(**inputs, output_type="np").images[0]

        output_slice = reconstruction[150:153, 140:143, -1]
        output_slice = output_slice.flatten()
        expected_slice = np.array(
            [0.56419, 0.44121838, 0.2765603, 0.5708484, 0.42763475, 0.30945742, 0.5387106, 0.4735807, 0.3547244]
        )
        assert np.abs(output_slice - expected_slice).max() < 1e-3
