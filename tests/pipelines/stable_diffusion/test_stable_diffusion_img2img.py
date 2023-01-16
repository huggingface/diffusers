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
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    StableDiffusionImg2ImgPipeline,
    UNet2DConditionModel,
)
from diffusers.utils import floats_tensor, load_image, load_numpy, nightly, slow, torch_device
from diffusers.utils.testing_utils import require_torch_gpu
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer

from ...test_pipelines_common import PipelineTesterMixin


torch.backends.cuda.matmul.allow_tf32 = False


class StableDiffusionImg2ImgPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = StableDiffusionImg2ImgPipeline

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
        image = floats_tensor((1, 3, 32, 32), rng=random.Random(seed)).to(device)
        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device=device).manual_seed(seed)
        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "image": image,
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            "output_type": "numpy",
        }
        return inputs

    def test_stable_diffusion_img2img_default_case(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()
        sd_pipe = StableDiffusionImg2ImgPipeline(**components)
        sd_pipe = sd_pipe.to(device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        image = sd_pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 32, 32, 3)
        expected_slice = np.array([0.4492, 0.3865, 0.4222, 0.5854, 0.5139, 0.4379, 0.4193, 0.48, 0.4218])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-3

    def test_stable_diffusion_img2img_negative_prompt(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()
        sd_pipe = StableDiffusionImg2ImgPipeline(**components)
        sd_pipe = sd_pipe.to(device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        negative_prompt = "french fries"
        output = sd_pipe(**inputs, negative_prompt=negative_prompt)
        image = output.images
        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 32, 32, 3)
        expected_slice = np.array([0.4065, 0.3783, 0.4050, 0.5266, 0.4781, 0.4252, 0.4203, 0.4692, 0.4365])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-3

    def test_stable_diffusion_img2img_multiple_init_images(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()
        sd_pipe = StableDiffusionImg2ImgPipeline(**components)
        sd_pipe = sd_pipe.to(device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        inputs["prompt"] = [inputs["prompt"]] * 2
        inputs["image"] = inputs["image"].repeat(2, 1, 1, 1)
        image = sd_pipe(**inputs).images
        image_slice = image[-1, -3:, -3:, -1]

        assert image.shape == (2, 32, 32, 3)
        expected_slice = np.array([0.5144, 0.4447, 0.4735, 0.6676, 0.5526, 0.5454, 0.645, 0.5149, 0.4689])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-3

    def test_stable_diffusion_img2img_k_lms(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()
        components["scheduler"] = LMSDiscreteScheduler(
            beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear"
        )
        sd_pipe = StableDiffusionImg2ImgPipeline(**components)
        sd_pipe = sd_pipe.to(device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        image = sd_pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 32, 32, 3)
        expected_slice = np.array([0.4367, 0.4986, 0.4372, 0.6706, 0.5665, 0.444, 0.5864, 0.6019, 0.5203])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-3

    def test_stable_diffusion_img2img_num_images_per_prompt(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()
        sd_pipe = StableDiffusionImg2ImgPipeline(**components)
        sd_pipe = sd_pipe.to(device)
        sd_pipe.set_progress_bar_config(disable=None)

        # test num_images_per_prompt=1 (default)
        inputs = self.get_dummy_inputs(device)
        images = sd_pipe(**inputs).images

        assert images.shape == (1, 32, 32, 3)

        # test num_images_per_prompt=1 (default) for batch of prompts
        batch_size = 2
        inputs = self.get_dummy_inputs(device)
        inputs["prompt"] = [inputs["prompt"]] * batch_size
        images = sd_pipe(**inputs).images

        assert images.shape == (batch_size, 32, 32, 3)

        # test num_images_per_prompt for single prompt
        num_images_per_prompt = 2
        inputs = self.get_dummy_inputs(device)
        images = sd_pipe(**inputs, num_images_per_prompt=num_images_per_prompt).images

        assert images.shape == (num_images_per_prompt, 32, 32, 3)

        # test num_images_per_prompt for batch of prompts
        batch_size = 2
        inputs = self.get_dummy_inputs(device)
        inputs["prompt"] = [inputs["prompt"]] * batch_size
        images = sd_pipe(**inputs, num_images_per_prompt=num_images_per_prompt).images

        assert images.shape == (batch_size * num_images_per_prompt, 32, 32, 3)


@slow
@require_torch_gpu
class StableDiffusionImg2ImgPipelineSlowTests(unittest.TestCase):
    def tearDown(self):
        super().tearDown()
        gc.collect()
        torch.cuda.empty_cache()

    def get_inputs(self, device, dtype=torch.float32, seed=0):
        generator = torch.Generator(device=device).manual_seed(seed)
        init_image = load_image(
            "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main"
            "/stable_diffusion_img2img/sketch-mountains-input.png"
        )
        inputs = {
            "prompt": "a fantasy landscape, concept art, high resolution",
            "image": init_image,
            "generator": generator,
            "num_inference_steps": 3,
            "strength": 0.75,
            "guidance_scale": 7.5,
            "output_type": "numpy",
        }
        return inputs

    def test_stable_diffusion_img2img_default(self):
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", safety_checker=None)
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()

        inputs = self.get_inputs(torch_device)
        image = pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1].flatten()

        assert image.shape == (1, 512, 768, 3)
        expected_slice = np.array([0.27150, 0.14849, 0.15605, 0.26740, 0.16954, 0.18204, 0.31470, 0.26311, 0.24525])
        assert np.abs(expected_slice - image_slice).max() < 1e-3

    def test_stable_diffusion_img2img_k_lms(self):
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", safety_checker=None)
        pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config)
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()

        inputs = self.get_inputs(torch_device)
        image = pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1].flatten()

        assert image.shape == (1, 512, 768, 3)
        expected_slice = np.array([0.04890, 0.04862, 0.06422, 0.04655, 0.05108, 0.05307, 0.05926, 0.08759, 0.06852])
        assert np.abs(expected_slice - image_slice).max() < 1e-3

    def test_stable_diffusion_img2img_ddim(self):
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", safety_checker=None)
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()

        inputs = self.get_inputs(torch_device)
        image = pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1].flatten()

        assert image.shape == (1, 512, 768, 3)
        expected_slice = np.array([0.06069, 0.05703, 0.08054, 0.05797, 0.06286, 0.06234, 0.08438, 0.11151, 0.08068])
        assert np.abs(expected_slice - image_slice).max() < 1e-3

    def test_stable_diffusion_img2img_intermediate_state(self):
        number_of_steps = 0

        def callback_fn(step: int, timestep: int, latents: torch.FloatTensor) -> None:
            callback_fn.has_been_called = True
            nonlocal number_of_steps
            number_of_steps += 1
            if step == 1:
                latents = latents.detach().cpu().numpy()
                assert latents.shape == (1, 4, 64, 96)
                latents_slice = latents[0, -3:, -3:, -1]
                expected_slice = np.array([0.7705, 0.1045, 0.5, 3.393, 3.723, 4.273, 2.467, 3.486, 1.758])
                assert np.abs(latents_slice.flatten() - expected_slice).max() < 1e-3
            elif step == 2:
                latents = latents.detach().cpu().numpy()
                assert latents.shape == (1, 4, 64, 96)
                latents_slice = latents[0, -3:, -3:, -1]
                expected_slice = np.array([0.765, 0.1047, 0.4973, 3.375, 3.709, 4.258, 2.451, 3.46, 1.755])
                assert np.abs(latents_slice.flatten() - expected_slice).max() < 1e-3

        callback_fn.has_been_called = False

        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4", safety_checker=None, torch_dtype=torch.float16
        )
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()

        inputs = self.get_inputs(torch_device, dtype=torch.float16)
        pipe(**inputs, callback=callback_fn, callback_steps=1)
        assert callback_fn.has_been_called
        assert number_of_steps == 2

    def test_stable_diffusion_pipeline_with_sequential_cpu_offloading(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()

        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4", safety_checker=None, torch_dtype=torch.float16
        )
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing(1)
        pipe.enable_sequential_cpu_offload()

        inputs = self.get_inputs(torch_device, dtype=torch.float16)
        _ = pipe(**inputs)

        mem_bytes = torch.cuda.max_memory_allocated()
        # make sure that less than 2.2 GB is allocated
        assert mem_bytes < 2.2 * 10**9

    def test_stable_diffusion_img2img_pipeline_multiple_of_8(self):
        init_image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
            "/img2img/sketch-mountains-input.jpg"
        )
        # resize to resolution that is divisible by 8 but not 16 or 32
        init_image = init_image.resize((760, 504))

        model_id = "CompVis/stable-diffusion-v1-4"
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_id,
            safety_checker=None,
        )
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()

        prompt = "A fantasy landscape, trending on artstation"

        generator = torch.Generator(device=torch_device).manual_seed(0)
        output = pipe(
            prompt=prompt,
            image=init_image,
            strength=0.75,
            guidance_scale=7.5,
            generator=generator,
            output_type="np",
        )
        image = output.images[0]

        image_slice = image[255:258, 383:386, -1]

        assert image.shape == (504, 760, 3)
        expected_slice = np.array([0.7124, 0.7105, 0.6993, 0.7140, 0.7106, 0.6945, 0.7198, 0.7172, 0.7031])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-3


@nightly
@require_torch_gpu
class StableDiffusionImg2ImgPipelineNightlyTests(unittest.TestCase):
    def tearDown(self):
        super().tearDown()
        gc.collect()
        torch.cuda.empty_cache()

    def get_inputs(self, device, dtype=torch.float32, seed=0):
        generator = torch.Generator(device=device).manual_seed(seed)
        init_image = load_image(
            "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main"
            "/stable_diffusion_img2img/sketch-mountains-input.png"
        )
        inputs = {
            "prompt": "a fantasy landscape, concept art, high resolution",
            "image": init_image,
            "generator": generator,
            "num_inference_steps": 50,
            "strength": 0.75,
            "guidance_scale": 7.5,
            "output_type": "numpy",
        }
        return inputs

    def test_img2img_pndm(self):
        sd_pipe = StableDiffusionImg2ImgPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
        sd_pipe.to(torch_device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs(torch_device)
        image = sd_pipe(**inputs).images[0]

        expected_image = load_numpy(
            "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main"
            "/stable_diffusion_img2img/stable_diffusion_1_5_pndm.npy"
        )
        max_diff = np.abs(expected_image - image).max()
        assert max_diff < 1e-3

    def test_img2img_ddim(self):
        sd_pipe = StableDiffusionImg2ImgPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
        sd_pipe.scheduler = DDIMScheduler.from_config(sd_pipe.scheduler.config)
        sd_pipe.to(torch_device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs(torch_device)
        image = sd_pipe(**inputs).images[0]

        expected_image = load_numpy(
            "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main"
            "/stable_diffusion_img2img/stable_diffusion_1_5_ddim.npy"
        )
        max_diff = np.abs(expected_image - image).max()
        assert max_diff < 1e-3

    def test_img2img_lms(self):
        sd_pipe = StableDiffusionImg2ImgPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
        sd_pipe.scheduler = LMSDiscreteScheduler.from_config(sd_pipe.scheduler.config)
        sd_pipe.to(torch_device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs(torch_device)
        image = sd_pipe(**inputs).images[0]

        expected_image = load_numpy(
            "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main"
            "/stable_diffusion_img2img/stable_diffusion_1_5_lms.npy"
        )
        max_diff = np.abs(expected_image - image).max()
        assert max_diff < 1e-3

    def test_img2img_dpm(self):
        sd_pipe = StableDiffusionImg2ImgPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
        sd_pipe.scheduler = DPMSolverMultistepScheduler.from_config(sd_pipe.scheduler.config)
        sd_pipe.to(torch_device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs(torch_device)
        inputs["num_inference_steps"] = 30
        image = sd_pipe(**inputs).images[0]

        expected_image = load_numpy(
            "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main"
            "/stable_diffusion_img2img/stable_diffusion_1_5_dpm.npy"
        )
        max_diff = np.abs(expected_image - image).max()
        assert max_diff < 1e-3
