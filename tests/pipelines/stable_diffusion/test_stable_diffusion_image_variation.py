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
from transformers import CLIPImageProcessor, CLIPVisionConfig, CLIPVisionModelWithProjection

from diffusers import (
    AutoencoderKL,
    DPMSolverMultistepScheduler,
    PNDMScheduler,
    StableDiffusionImageVariationPipeline,
    UNet2DConditionModel,
)
from diffusers.utils import floats_tensor, load_image, load_numpy, nightly, slow, torch_device
from diffusers.utils.testing_utils import enable_full_determinism, require_torch_gpu

from ..pipeline_params import IMAGE_VARIATION_BATCH_PARAMS, IMAGE_VARIATION_PARAMS
from ..test_pipelines_common import PipelineLatentTesterMixin, PipelineTesterMixin


enable_full_determinism()


class StableDiffusionImageVariationPipelineFastTests(
    PipelineLatentTesterMixin, PipelineTesterMixin, unittest.TestCase
):
    pipeline_class = StableDiffusionImageVariationPipeline
    params = IMAGE_VARIATION_PARAMS
    batch_params = IMAGE_VARIATION_BATCH_PARAMS
    image_params = frozenset([])
    # TO-DO: update image_params once pipeline is refactored with VaeImageProcessor.preprocess
    image_latents_params = frozenset([])

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
        feature_extractor = CLIPImageProcessor(crop_size=32, size=32)

        components = {
            "unet": unet,
            "scheduler": scheduler,
            "vae": vae,
            "image_encoder": image_encoder,
            "feature_extractor": feature_extractor,
            "safety_checker": None,
        }
        return components

    def get_dummy_inputs(self, device, seed=0):
        image = floats_tensor((1, 3, 32, 32), rng=random.Random(seed))
        image = image.cpu().permute(0, 2, 3, 1)[0]
        image = Image.fromarray(np.uint8(image)).convert("RGB").resize((32, 32))
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
        expected_slice = np.array([0.5239, 0.5723, 0.4796, 0.5049, 0.5550, 0.4685, 0.5329, 0.4891, 0.4921])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-3

    def test_stable_diffusion_img_variation_multiple_images(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()
        sd_pipe = StableDiffusionImageVariationPipeline(**components)
        sd_pipe = sd_pipe.to(device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        inputs["image"] = 2 * [inputs["image"]]
        output = sd_pipe(**inputs)

        image = output.images

        image_slice = image[-1, -3:, -3:, -1]

        assert image.shape == (2, 64, 64, 3)
        expected_slice = np.array([0.6892, 0.5637, 0.5836, 0.5771, 0.6254, 0.6409, 0.5580, 0.5569, 0.5289])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-3

    def test_inference_batch_single_identical(self):
        super().test_inference_batch_single_identical(expected_max_diff=3e-3)


@slow
@require_torch_gpu
class StableDiffusionImageVariationPipelineSlowTests(unittest.TestCase):
    def tearDown(self):
        super().tearDown()
        gc.collect()
        torch.cuda.empty_cache()

    def get_inputs(self, device, generator_device="cpu", dtype=torch.float32, seed=0):
        generator = torch.Generator(device=generator_device).manual_seed(seed)
        init_image = load_image(
            "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main"
            "/stable_diffusion_imgvar/input_image_vermeer.png"
        )
        latents = np.random.RandomState(seed).standard_normal((1, 4, 64, 64))
        latents = torch.from_numpy(latents).to(device=device, dtype=dtype)
        inputs = {
            "image": init_image,
            "latents": latents,
            "generator": generator,
            "num_inference_steps": 3,
            "guidance_scale": 7.5,
            "output_type": "numpy",
        }
        return inputs

    def test_stable_diffusion_img_variation_pipeline_default(self):
        sd_pipe = StableDiffusionImageVariationPipeline.from_pretrained(
            "lambdalabs/sd-image-variations-diffusers", safety_checker=None
        )
        sd_pipe = sd_pipe.to(torch_device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs(torch_device)
        image = sd_pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1].flatten()

        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array([0.84491, 0.90789, 0.75708, 0.78734, 0.83485, 0.70099, 0.66938, 0.68727, 0.61379])
        assert np.abs(image_slice - expected_slice).max() < 6e-3

    def test_stable_diffusion_img_variation_intermediate_state(self):
        number_of_steps = 0

        def callback_fn(step: int, timestep: int, latents: torch.FloatTensor) -> None:
            callback_fn.has_been_called = True
            nonlocal number_of_steps
            number_of_steps += 1
            if step == 1:
                latents = latents.detach().cpu().numpy()
                assert latents.shape == (1, 4, 64, 64)
                latents_slice = latents[0, -3:, -3:, -1]
                expected_slice = np.array(
                    [-0.1621, 0.2837, -0.7979, -0.1221, -1.3057, 0.7681, -2.1191, 0.0464, 1.6309]
                )

                assert np.abs(latents_slice.flatten() - expected_slice).max() < 5e-2
            elif step == 2:
                latents = latents.detach().cpu().numpy()
                assert latents.shape == (1, 4, 64, 64)
                latents_slice = latents[0, -3:, -3:, -1]
                expected_slice = np.array([0.6299, 1.7500, 1.1992, -2.1582, -1.8994, 0.7334, -0.7090, 1.0137, 1.5273])

                assert np.abs(latents_slice.flatten() - expected_slice).max() < 5e-2

        callback_fn.has_been_called = False

        pipe = StableDiffusionImageVariationPipeline.from_pretrained(
            "fusing/sd-image-variations-diffusers",
            safety_checker=None,
            torch_dtype=torch.float16,
        )
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()

        inputs = self.get_inputs(torch_device, dtype=torch.float16)
        pipe(**inputs, callback=callback_fn, callback_steps=1)
        assert callback_fn.has_been_called
        assert number_of_steps == inputs["num_inference_steps"]

    def test_stable_diffusion_pipeline_with_sequential_cpu_offloading(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()

        model_id = "fusing/sd-image-variations-diffusers"
        pipe = StableDiffusionImageVariationPipeline.from_pretrained(
            model_id, safety_checker=None, torch_dtype=torch.float16
        )
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing(1)
        pipe.enable_sequential_cpu_offload()

        inputs = self.get_inputs(torch_device, dtype=torch.float16)
        _ = pipe(**inputs)

        mem_bytes = torch.cuda.max_memory_allocated()
        # make sure that less than 2.6 GB is allocated
        assert mem_bytes < 2.6 * 10**9


@nightly
@require_torch_gpu
class StableDiffusionImageVariationPipelineNightlyTests(unittest.TestCase):
    def tearDown(self):
        super().tearDown()
        gc.collect()
        torch.cuda.empty_cache()

    def get_inputs(self, device, generator_device="cpu", dtype=torch.float32, seed=0):
        generator = torch.Generator(device=generator_device).manual_seed(seed)
        init_image = load_image(
            "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main"
            "/stable_diffusion_imgvar/input_image_vermeer.png"
        )
        latents = np.random.RandomState(seed).standard_normal((1, 4, 64, 64))
        latents = torch.from_numpy(latents).to(device=device, dtype=dtype)
        inputs = {
            "image": init_image,
            "latents": latents,
            "generator": generator,
            "num_inference_steps": 50,
            "guidance_scale": 7.5,
            "output_type": "numpy",
        }
        return inputs

    def test_img_variation_pndm(self):
        sd_pipe = StableDiffusionImageVariationPipeline.from_pretrained("fusing/sd-image-variations-diffusers")
        sd_pipe.to(torch_device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs(torch_device)
        image = sd_pipe(**inputs).images[0]

        expected_image = load_numpy(
            "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main"
            "/stable_diffusion_imgvar/lambdalabs_variations_pndm.npy"
        )
        max_diff = np.abs(expected_image - image).max()
        assert max_diff < 1e-3

    def test_img_variation_dpm(self):
        sd_pipe = StableDiffusionImageVariationPipeline.from_pretrained("fusing/sd-image-variations-diffusers")
        sd_pipe.scheduler = DPMSolverMultistepScheduler.from_config(sd_pipe.scheduler.config)
        sd_pipe.to(torch_device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs(torch_device)
        inputs["num_inference_steps"] = 25
        image = sd_pipe(**inputs).images[0]

        expected_image = load_numpy(
            "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main"
            "/stable_diffusion_imgvar/lambdalabs_variations_dpm_multi.npy"
        )
        max_diff = np.abs(expected_image - image).max()
        assert max_diff < 1e-3
