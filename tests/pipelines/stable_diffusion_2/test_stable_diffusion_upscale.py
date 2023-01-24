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

import gc
import random
import unittest

import numpy as np
import torch

from diffusers import AutoencoderKL, DDIMScheduler, DDPMScheduler, StableDiffusionUpscalePipeline, UNet2DConditionModel
from diffusers.utils import floats_tensor, load_image, load_numpy, slow, torch_device
from diffusers.utils.testing_utils import require_torch_gpu
from PIL import Image
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer


torch.backends.cuda.matmul.allow_tf32 = False


class StableDiffusionUpscalePipelineFastTests(unittest.TestCase):
    def tearDown(self):
        # clean up the VRAM after each test
        super().tearDown()
        gc.collect()
        torch.cuda.empty_cache()

    @property
    def dummy_image(self):
        batch_size = 1
        num_channels = 3
        sizes = (32, 32)

        image = floats_tensor((batch_size, num_channels) + sizes, rng=random.Random(0)).to(torch_device)
        return image

    @property
    def dummy_cond_unet_upscale(self):
        torch.manual_seed(0)
        model = UNet2DConditionModel(
            block_out_channels=(32, 32, 64),
            layers_per_block=2,
            sample_size=32,
            in_channels=7,
            out_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D"),
            cross_attention_dim=32,
            # SD2-specific config below
            attention_head_dim=8,
            use_linear_projection=True,
            only_cross_attention=(True, True, False),
            num_class_embeds=100,
        )
        return model

    @property
    def dummy_vae(self):
        torch.manual_seed(0)
        model = AutoencoderKL(
            block_out_channels=[32, 32, 64],
            in_channels=3,
            out_channels=3,
            down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"],
            up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"],
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
            # SD2-specific config below
            hidden_act="gelu",
            projection_dim=512,
        )
        return CLIPTextModel(config)

    def test_stable_diffusion_upscale(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        unet = self.dummy_cond_unet_upscale
        low_res_scheduler = DDPMScheduler()
        scheduler = DDIMScheduler(prediction_type="v_prediction")
        vae = self.dummy_vae
        text_encoder = self.dummy_text_encoder
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        image = self.dummy_image.cpu().permute(0, 2, 3, 1)[0]
        low_res_image = Image.fromarray(np.uint8(image)).convert("RGB").resize((64, 64))

        # make sure here that pndm scheduler skips prk
        sd_pipe = StableDiffusionUpscalePipeline(
            unet=unet,
            low_res_scheduler=low_res_scheduler,
            scheduler=scheduler,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            max_noise_level=350,
        )
        sd_pipe = sd_pipe.to(device)
        sd_pipe.set_progress_bar_config(disable=None)

        prompt = "A painting of a squirrel eating a burger"
        generator = torch.Generator(device=device).manual_seed(0)
        output = sd_pipe(
            [prompt],
            image=low_res_image,
            generator=generator,
            guidance_scale=6.0,
            noise_level=20,
            num_inference_steps=2,
            output_type="np",
        )

        image = output.images

        generator = torch.Generator(device=device).manual_seed(0)
        image_from_tuple = sd_pipe(
            [prompt],
            image=low_res_image,
            generator=generator,
            guidance_scale=6.0,
            noise_level=20,
            num_inference_steps=2,
            output_type="np",
            return_dict=False,
        )[0]

        image_slice = image[0, -3:, -3:, -1]
        image_from_tuple_slice = image_from_tuple[0, -3:, -3:, -1]

        expected_height_width = low_res_image.size[0] * 4
        assert image.shape == (1, expected_height_width, expected_height_width, 3)
        expected_slice = np.array([0.2562, 0.3606, 0.4204, 0.4469, 0.4822, 0.4647, 0.5315, 0.5748, 0.5606])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2
        assert np.abs(image_from_tuple_slice.flatten() - expected_slice).max() < 1e-2

    def test_stable_diffusion_upscale_batch(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        unet = self.dummy_cond_unet_upscale
        low_res_scheduler = DDPMScheduler()
        scheduler = DDIMScheduler(prediction_type="v_prediction")
        vae = self.dummy_vae
        text_encoder = self.dummy_text_encoder
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        image = self.dummy_image.cpu().permute(0, 2, 3, 1)[0]
        low_res_image = Image.fromarray(np.uint8(image)).convert("RGB").resize((64, 64))

        # make sure here that pndm scheduler skips prk
        sd_pipe = StableDiffusionUpscalePipeline(
            unet=unet,
            low_res_scheduler=low_res_scheduler,
            scheduler=scheduler,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            max_noise_level=350,
        )
        sd_pipe = sd_pipe.to(device)
        sd_pipe.set_progress_bar_config(disable=None)

        prompt = "A painting of a squirrel eating a burger"
        output = sd_pipe(
            2 * [prompt],
            image=2 * [low_res_image],
            guidance_scale=6.0,
            noise_level=20,
            num_inference_steps=2,
            output_type="np",
        )
        image = output.images
        assert image.shape[0] == 2

        generator = torch.Generator(device=device).manual_seed(0)
        output = sd_pipe(
            [prompt],
            image=low_res_image,
            generator=generator,
            num_images_per_prompt=2,
            guidance_scale=6.0,
            noise_level=20,
            num_inference_steps=2,
            output_type="np",
        )
        image = output.images
        assert image.shape[0] == 2

    @unittest.skipIf(torch_device != "cuda", "This test requires a GPU")
    def test_stable_diffusion_upscale_fp16(self):
        """Test that stable diffusion upscale works with fp16"""
        unet = self.dummy_cond_unet_upscale
        low_res_scheduler = DDPMScheduler()
        scheduler = DDIMScheduler(prediction_type="v_prediction")
        vae = self.dummy_vae
        text_encoder = self.dummy_text_encoder
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        image = self.dummy_image.cpu().permute(0, 2, 3, 1)[0]
        low_res_image = Image.fromarray(np.uint8(image)).convert("RGB").resize((64, 64))

        # put models in fp16, except vae as it overflows in fp16
        unet = unet.half()
        text_encoder = text_encoder.half()

        # make sure here that pndm scheduler skips prk
        sd_pipe = StableDiffusionUpscalePipeline(
            unet=unet,
            low_res_scheduler=low_res_scheduler,
            scheduler=scheduler,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            max_noise_level=350,
        )
        sd_pipe = sd_pipe.to(torch_device)
        sd_pipe.set_progress_bar_config(disable=None)

        prompt = "A painting of a squirrel eating a burger"
        generator = torch.manual_seed(0)
        image = sd_pipe(
            [prompt],
            image=low_res_image,
            generator=generator,
            num_inference_steps=2,
            output_type="np",
        ).images

        expected_height_width = low_res_image.size[0] * 4
        assert image.shape == (1, expected_height_width, expected_height_width, 3)


@slow
@require_torch_gpu
class StableDiffusionUpscalePipelineIntegrationTests(unittest.TestCase):
    def tearDown(self):
        # clean up the VRAM after each test
        super().tearDown()
        gc.collect()
        torch.cuda.empty_cache()

    def test_stable_diffusion_upscale_pipeline(self):
        image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
            "/sd2-upscale/low_res_cat.png"
        )
        expected_image = load_numpy(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd2-upscale"
            "/upsampled_cat.npy"
        )

        model_id = "stabilityai/stable-diffusion-x4-upscaler"
        pipe = StableDiffusionUpscalePipeline.from_pretrained(model_id)
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()

        prompt = "a cat sitting on a park bench"

        generator = torch.manual_seed(0)
        output = pipe(
            prompt=prompt,
            image=image,
            generator=generator,
            output_type="np",
        )
        image = output.images[0]

        assert image.shape == (512, 512, 3)
        assert np.abs(expected_image - image).max() < 1e-3

    def test_stable_diffusion_upscale_pipeline_fp16(self):
        image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
            "/sd2-upscale/low_res_cat.png"
        )
        expected_image = load_numpy(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd2-upscale"
            "/upsampled_cat_fp16.npy"
        )

        model_id = "stabilityai/stable-diffusion-x4-upscaler"
        pipe = StableDiffusionUpscalePipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
        )
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()

        prompt = "a cat sitting on a park bench"

        generator = torch.manual_seed(0)
        output = pipe(
            prompt=prompt,
            image=image,
            generator=generator,
            output_type="np",
        )
        image = output.images[0]

        assert image.shape == (512, 512, 3)
        assert np.abs(expected_image - image).max() < 5e-1

    def test_stable_diffusion_pipeline_with_sequential_cpu_offloading(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()

        image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
            "/sd2-upscale/low_res_cat.png"
        )

        model_id = "stabilityai/stable-diffusion-x4-upscaler"
        pipe = StableDiffusionUpscalePipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
        )
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing(1)
        pipe.enable_sequential_cpu_offload()

        prompt = "a cat sitting on a park bench"

        generator = torch.manual_seed(0)
        _ = pipe(
            prompt=prompt,
            image=image,
            generator=generator,
            num_inference_steps=5,
            output_type="np",
        )

        mem_bytes = torch.cuda.max_memory_allocated()
        # make sure that less than 2.65 GB is allocated
        assert mem_bytes < 2.65 * 10**9
