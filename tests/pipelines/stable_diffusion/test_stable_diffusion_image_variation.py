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

from diffusers import (
    AutoencoderKL,
    LMSDiscreteScheduler,
    PNDMScheduler,
    StableDiffusionImageVariationPipeline,
    UNet2DConditionModel,
)
from diffusers.utils import floats_tensor, load_image, load_numpy, slow, torch_device
from diffusers.utils.testing_utils import require_torch_gpu
from transformers import CLIPVisionConfig, CLIPVisionModelWithProjection

from ...test_pipelines_common import PipelineTesterMixin


torch.backends.cuda.matmul.allow_tf32 = False


class StableDiffusionImageVariationPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = StableDiffusionImageVariationPipeline

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
        scheduler = PNDMScheduler(skip_prk_steps=True)
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
        image_encoder_config = CLIPVisionConfig(
            hidden_size=32,
            projection_dim=32,
            intermediate_size=37,
            layer_norm_eps=1e-05,
            num_attention_heads=4,
            num_hidden_layers=5,
            image_size=32,
            patch_size=4,
        )
        image_encoder = CLIPVisionModelWithProjection(image_encoder_config)

        components = {
            "unet": unet,
            "scheduler": scheduler,
            "vae": vae,
            "image_encoder": image_encoder,
            "safety_checker": None,
            "feature_extractor": None,
        }
        return components

    def get_dummy_inputs(self, device, seed=0):
        image = floats_tensor((1, 3, 32, 32), rng=random.Random(seed)).to(device)
        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device=device).manual_seed(seed)
        inputs = {
            "image": image,
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            "output_type": "numpy",
        }
        return inputs

    def test_stable_diffusion_img_variation_default_case(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()
        sd_pipe = StableDiffusionImageVariationPipeline(**components)
        sd_pipe = sd_pipe.to(device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        image = sd_pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 64, 64, 3)
        expected_slice = np.array([0.5093, 0.5717, 0.4806, 0.4891, 0.5552, 0.4594, 0.5177, 0.4894, 0.4904])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-3

    def test_stable_diffusion_img_variation_multiple_images(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()
        sd_pipe = StableDiffusionImageVariationPipeline(**components)
        sd_pipe = sd_pipe.to(device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        inputs["image"] = inputs["image"].repeat(2, 1, 1, 1)
        output = sd_pipe(**inputs)

        image = output.images

        image_slice = image[-1, -3:, -3:, -1]

        assert image.shape == (2, 64, 64, 3)
        expected_slice = np.array([0.6427, 0.5452, 0.5602, 0.5478, 0.5968, 0.6211, 0.5538, 0.5514, 0.5281])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-3

    def test_stable_diffusion_img_variation_num_images_per_prompt(self):
        device = "cpu"
        components = self.get_dummy_components()
        sd_pipe = StableDiffusionImageVariationPipeline(**components)
        sd_pipe = sd_pipe.to(device)
        sd_pipe.set_progress_bar_config(disable=None)

        # test num_images_per_prompt=1 (default)
        inputs = self.get_dummy_inputs(device)
        images = sd_pipe(**inputs).images

        assert images.shape == (1, 64, 64, 3)

        # test num_images_per_prompt=1 (default) for batch of images
        batch_size = 2
        inputs = self.get_dummy_inputs(device)
        inputs["image"] = inputs["image"].repeat(batch_size, 1, 1, 1)
        images = sd_pipe(**inputs).images

        assert images.shape == (batch_size, 64, 64, 3)

        # test num_images_per_prompt for single prompt
        num_images_per_prompt = 2
        inputs = self.get_dummy_inputs(device)
        images = sd_pipe(**inputs, num_images_per_prompt=num_images_per_prompt).images

        assert images.shape == (num_images_per_prompt, 64, 64, 3)

        # test num_images_per_prompt for batch of prompts
        batch_size = 2
        inputs = self.get_dummy_inputs(device)
        inputs["image"] = inputs["image"].repeat(batch_size, 1, 1, 1)
        images = sd_pipe(**inputs, num_images_per_prompt=num_images_per_prompt).images

        assert images.shape == (batch_size * num_images_per_prompt, 64, 64, 3)


@slow
@require_torch_gpu
class StableDiffusionImageVariationPipelineIntegrationTests(unittest.TestCase):
    def tearDown(self):
        # clean up the VRAM after each test
        super().tearDown()
        gc.collect()
        torch.cuda.empty_cache()

    def test_stable_diffusion_img_variation_pipeline_default(self):
        init_image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/img2img/vermeer.jpg"
        )
        init_image = init_image.resize((512, 512))
        expected_image = load_numpy(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/img2img/vermeer.npy"
        )

        model_id = "fusing/sd-image-variations-diffusers"
        pipe = StableDiffusionImageVariationPipeline.from_pretrained(
            model_id,
            safety_checker=None,
        )
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()

        generator = torch.Generator(device=torch_device).manual_seed(0)
        output = pipe(
            init_image,
            guidance_scale=7.5,
            generator=generator,
            output_type="np",
        )
        image = output.images[0]

        assert image.shape == (512, 512, 3)
        # img2img is flaky across GPUs even in fp32, so using MAE here
        assert np.abs(expected_image - image).max() < 1e-3

    def test_stable_diffusion_img_variation_intermediate_state(self):
        number_of_steps = 0

        def test_callback_fn(step: int, timestep: int, latents: torch.FloatTensor) -> None:
            test_callback_fn.has_been_called = True
            nonlocal number_of_steps
            number_of_steps += 1
            if step == 0:
                latents = latents.detach().cpu().numpy()
                assert latents.shape == (1, 4, 64, 64)
                latents_slice = latents[0, -3:, -3:, -1]
                expected_slice = np.array([1.83, 1.293, -0.09705, 1.256, -2.293, 1.091, -0.0809, -0.65, -2.953])
                assert np.abs(latents_slice.flatten() - expected_slice).max() < 5e-3
            elif step == 37:
                latents = latents.detach().cpu().numpy()
                assert latents.shape == (1, 4, 64, 64)
                latents_slice = latents[0, -3:, -3:, -1]
                expected_slice = np.array([2.285, 2.703, 1.969, 0.696, -1.323, 0.9253, -0.5464, -1.521, -2.537])
                assert np.abs(latents_slice.flatten() - expected_slice).max() < 5e-2

        test_callback_fn.has_been_called = False

        init_image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
            "/img2img/sketch-mountains-input.jpg"
        )
        init_image = init_image.resize((512, 512))

        pipe = StableDiffusionImageVariationPipeline.from_pretrained(
            "fusing/sd-image-variations-diffusers",
            torch_dtype=torch.float16,
        )
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()

        generator = torch.Generator(device=torch_device).manual_seed(0)
        with torch.autocast(torch_device):
            pipe(
                init_image,
                num_inference_steps=50,
                guidance_scale=7.5,
                generator=generator,
                callback=test_callback_fn,
                callback_steps=1,
            )
        assert test_callback_fn.has_been_called
        assert number_of_steps == 50

    def test_stable_diffusion_pipeline_with_sequential_cpu_offloading(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()

        init_image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
            "/img2img/sketch-mountains-input.jpg"
        )
        init_image = init_image.resize((512, 512))

        model_id = "fusing/sd-image-variations-diffusers"
        lms = LMSDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
        pipe = StableDiffusionImageVariationPipeline.from_pretrained(
            model_id, scheduler=lms, safety_checker=None, torch_dtype=torch.float16
        )
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing(1)
        pipe.enable_sequential_cpu_offload()

        generator = torch.Generator(device=torch_device).manual_seed(0)
        _ = pipe(
            init_image,
            guidance_scale=7.5,
            generator=generator,
            output_type="np",
            num_inference_steps=5,
        )

        mem_bytes = torch.cuda.max_memory_allocated()
        # make sure that less than 2.6 GB is allocated
        assert mem_bytes < 2.6 * 10**9
