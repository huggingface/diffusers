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
    EulerAncestralDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    StableDiffusionPanoramaPipeline,
    UNet2DConditionModel,
)
from diffusers.utils import slow, torch_device
from diffusers.utils.testing_utils import require_torch_gpu, skip_mps

from ..pipeline_params import TEXT_TO_IMAGE_BATCH_PARAMS, TEXT_TO_IMAGE_IMAGE_PARAMS, TEXT_TO_IMAGE_PARAMS
from ..test_pipelines_common import PipelineLatentTesterMixin, PipelineTesterMixin


torch.backends.cuda.matmul.allow_tf32 = False
torch.use_deterministic_algorithms(True)


@skip_mps
class StableDiffusionPanoramaPipelineFastTests(PipelineLatentTesterMixin, PipelineTesterMixin, unittest.TestCase):
    pipeline_class = StableDiffusionPanoramaPipeline
    params = TEXT_TO_IMAGE_PARAMS
    batch_params = TEXT_TO_IMAGE_BATCH_PARAMS
    image_params = TEXT_TO_IMAGE_IMAGE_PARAMS

    def get_dummy_components(self):
        torch.manual_seed(0)
        unet = UNet2DConditionModel(
            block_out_channels=(32, 64),
            layers_per_block=1,
            sample_size=32,
            in_channels=4,
            out_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
            cross_attention_dim=32,
        )
        scheduler = DDIMScheduler()
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
        generator = torch.manual_seed(seed)
        inputs = {
            "prompt": "a photo of the dolomites",
            "generator": generator,
            # Setting height and width to None to prevent OOMs on CPU.
            "height": None,
            "width": None,
            "num_inference_steps": 1,
            "guidance_scale": 6.0,
            "output_type": "numpy",
        }
        return inputs

    def test_stable_diffusion_panorama_default_case(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()
        sd_pipe = StableDiffusionPanoramaPipeline(**components)
        sd_pipe = sd_pipe.to(device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        image = sd_pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1]
        assert image.shape == (1, 64, 64, 3)

        expected_slice = np.array([0.6186, 0.5374, 0.4915, 0.4135, 0.4114, 0.4563, 0.5128, 0.4977, 0.4757])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

    # override to speed the overall test timing up.
    def test_inference_batch_consistent(self):
        super().test_inference_batch_consistent(batch_sizes=[1, 2])

    # override to speed the overall test timing up.
    def test_inference_batch_single_identical(self):
        super().test_inference_batch_single_identical(batch_size=2, expected_max_diff=3e-3)

    def test_stable_diffusion_panorama_negative_prompt(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()
        sd_pipe = StableDiffusionPanoramaPipeline(**components)
        sd_pipe = sd_pipe.to(device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        negative_prompt = "french fries"
        output = sd_pipe(**inputs, negative_prompt=negative_prompt)
        image = output.images
        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 64, 64, 3)

        expected_slice = np.array([0.6187, 0.5375, 0.4915, 0.4136, 0.4114, 0.4563, 0.5128, 0.4976, 0.4757])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

    def test_stable_diffusion_panorama_euler(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()
        components["scheduler"] = EulerAncestralDiscreteScheduler(
            beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear"
        )
        sd_pipe = StableDiffusionPanoramaPipeline(**components)
        sd_pipe = sd_pipe.to(device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        image = sd_pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 64, 64, 3)

        expected_slice = np.array([0.4886, 0.5586, 0.4476, 0.5053, 0.6013, 0.4737, 0.5538, 0.5100, 0.4927])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

    def test_stable_diffusion_panorama_pndm(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()
        components["scheduler"] = PNDMScheduler()
        sd_pipe = StableDiffusionPanoramaPipeline(**components)
        sd_pipe = sd_pipe.to(device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        # the pipeline does not expect pndm so test if it raises error.
        with self.assertRaises(ValueError):
            _ = sd_pipe(**inputs).images


@slow
@require_torch_gpu
class StableDiffusionPanoramaSlowTests(unittest.TestCase):
    def tearDown(self):
        super().tearDown()
        gc.collect()
        torch.cuda.empty_cache()

    def get_inputs(self, seed=0):
        generator = torch.manual_seed(seed)
        inputs = {
            "prompt": "a photo of the dolomites",
            "generator": generator,
            "num_inference_steps": 3,
            "guidance_scale": 7.5,
            "output_type": "numpy",
        }
        return inputs

    def test_stable_diffusion_panorama_default(self):
        model_ckpt = "stabilityai/stable-diffusion-2-base"
        scheduler = DDIMScheduler.from_pretrained(model_ckpt, subfolder="scheduler")
        pipe = StableDiffusionPanoramaPipeline.from_pretrained(model_ckpt, scheduler=scheduler, safety_checker=None)
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()

        inputs = self.get_inputs()
        image = pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1].flatten()

        assert image.shape == (1, 512, 2048, 3)

        expected_slice = np.array(
            [
                0.36968392,
                0.27025372,
                0.32446766,
                0.28379387,
                0.36363274,
                0.30733347,
                0.27100027,
                0.27054125,
                0.25536096,
            ]
        )

        assert np.abs(expected_slice - image_slice).max() < 1e-2

    def test_stable_diffusion_panorama_k_lms(self):
        pipe = StableDiffusionPanoramaPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-base", safety_checker=None
        )
        pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config)
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()

        inputs = self.get_inputs()
        image = pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1].flatten()

        assert image.shape == (1, 512, 2048, 3)

        expected_slice = np.array(
            [
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
            ]
        )

        assert np.abs(expected_slice - image_slice).max() < 1e-3

    def test_stable_diffusion_panorama_intermediate_state(self):
        number_of_steps = 0

        def callback_fn(step: int, timestep: int, latents: torch.FloatTensor) -> None:
            callback_fn.has_been_called = True
            nonlocal number_of_steps
            number_of_steps += 1
            if step == 1:
                latents = latents.detach().cpu().numpy()
                assert latents.shape == (1, 4, 64, 256)
                latents_slice = latents[0, -3:, -3:, -1]

                expected_slice = np.array(
                    [
                        0.18681869,
                        0.33907816,
                        0.5361276,
                        0.14432865,
                        -0.02856611,
                        -0.73941123,
                        0.23397987,
                        0.47322682,
                        -0.37823164,
                    ]
                )
                assert np.abs(latents_slice.flatten() - expected_slice).max() < 5e-2
            elif step == 2:
                latents = latents.detach().cpu().numpy()
                assert latents.shape == (1, 4, 64, 256)
                latents_slice = latents[0, -3:, -3:, -1]

                expected_slice = np.array(
                    [
                        0.18539645,
                        0.33987248,
                        0.5378559,
                        0.14437142,
                        -0.02455261,
                        -0.7338317,
                        0.23990755,
                        0.47356272,
                        -0.3786505,
                    ]
                )

                assert np.abs(latents_slice.flatten() - expected_slice).max() < 5e-2

        callback_fn.has_been_called = False

        model_ckpt = "stabilityai/stable-diffusion-2-base"
        scheduler = DDIMScheduler.from_pretrained(model_ckpt, subfolder="scheduler")
        pipe = StableDiffusionPanoramaPipeline.from_pretrained(model_ckpt, scheduler=scheduler, safety_checker=None)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()

        inputs = self.get_inputs()
        pipe(**inputs, callback=callback_fn, callback_steps=1)
        assert callback_fn.has_been_called
        assert number_of_steps == 3

    def test_stable_diffusion_panorama_pipeline_with_sequential_cpu_offloading(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()

        model_ckpt = "stabilityai/stable-diffusion-2-base"
        scheduler = DDIMScheduler.from_pretrained(model_ckpt, subfolder="scheduler")
        pipe = StableDiffusionPanoramaPipeline.from_pretrained(model_ckpt, scheduler=scheduler, safety_checker=None)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing(1)
        pipe.enable_sequential_cpu_offload()

        inputs = self.get_inputs()
        _ = pipe(**inputs)

        mem_bytes = torch.cuda.max_memory_allocated()
        # make sure that less than 5.2 GB is allocated
        assert mem_bytes < 5.5 * 10**9
