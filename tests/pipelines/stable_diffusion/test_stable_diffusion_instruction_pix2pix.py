# coding=utf-8
# Copyright 2024 HuggingFace Inc.
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
    DDIMScheduler,
    EulerAncestralDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    StableDiffusionInstructPix2PixPipeline,
    UNet2DConditionModel,
)
from diffusers.image_processor import VaeImageProcessor
from diffusers.utils.testing_utils import (
    enable_full_determinism,
    floats_tensor,
    load_image,
    require_torch_gpu,
    slow,
    torch_device,
)

from ..pipeline_params import (
    IMAGE_TO_IMAGE_IMAGE_PARAMS,
    TEXT_GUIDED_IMAGE_INPAINTING_BATCH_PARAMS,
    TEXT_GUIDED_IMAGE_VARIATION_PARAMS,
    TEXT_TO_IMAGE_CALLBACK_CFG_PARAMS,
)
from ..test_pipelines_common import (
    PipelineKarrasSchedulerTesterMixin,
    PipelineLatentTesterMixin,
    PipelineTesterMixin,
)


enable_full_determinism()


class StableDiffusionInstructPix2PixPipelineFastTests(
    PipelineLatentTesterMixin, PipelineKarrasSchedulerTesterMixin, PipelineTesterMixin, unittest.TestCase
):
    pipeline_class = StableDiffusionInstructPix2PixPipeline
    params = TEXT_GUIDED_IMAGE_VARIATION_PARAMS - {"height", "width", "cross_attention_kwargs"}
    batch_params = TEXT_GUIDED_IMAGE_INPAINTING_BATCH_PARAMS
    image_params = IMAGE_TO_IMAGE_IMAGE_PARAMS
    image_latents_params = IMAGE_TO_IMAGE_IMAGE_PARAMS
    callback_cfg_params = TEXT_TO_IMAGE_CALLBACK_CFG_PARAMS.union({"image_latents"}) - {"negative_prompt_embeds"}

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
            "image_encoder": None,
        }
        return components

    def get_dummy_inputs(self, device, seed=0):
        image = floats_tensor((1, 3, 32, 32), rng=random.Random(seed)).to(device)
        image = image.cpu().permute(0, 2, 3, 1)[0]
        image = Image.fromarray(np.uint8(image)).convert("RGB")
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
            "image_guidance_scale": 1,
            "output_type": "np",
        }
        return inputs

    def test_stable_diffusion_pix2pix_default_case(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()
        sd_pipe = StableDiffusionInstructPix2PixPipeline(**components)
        sd_pipe = sd_pipe.to(device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        image = sd_pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1]
        assert image.shape == (1, 32, 32, 3)
        expected_slice = np.array([0.7526, 0.3750, 0.4547, 0.6117, 0.5866, 0.5016, 0.4327, 0.5642, 0.4815])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-3

    def test_stable_diffusion_pix2pix_negative_prompt(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()
        sd_pipe = StableDiffusionInstructPix2PixPipeline(**components)
        sd_pipe = sd_pipe.to(device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        negative_prompt = "french fries"
        output = sd_pipe(**inputs, negative_prompt=negative_prompt)
        image = output.images
        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 32, 32, 3)
        expected_slice = np.array([0.7511, 0.3642, 0.4553, 0.6236, 0.5797, 0.5013, 0.4343, 0.5611, 0.4831])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-3

    def test_stable_diffusion_pix2pix_multiple_init_images(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()
        sd_pipe = StableDiffusionInstructPix2PixPipeline(**components)
        sd_pipe = sd_pipe.to(device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        inputs["prompt"] = [inputs["prompt"]] * 2

        image = np.array(inputs["image"]).astype(np.float32) / 255.0
        image = torch.from_numpy(image).unsqueeze(0).to(device)
        image = image / 2 + 0.5
        image = image.permute(0, 3, 1, 2)
        inputs["image"] = image.repeat(2, 1, 1, 1)

        image = sd_pipe(**inputs).images
        image_slice = image[-1, -3:, -3:, -1]

        assert image.shape == (2, 32, 32, 3)
        expected_slice = np.array([0.5812, 0.5748, 0.5222, 0.5908, 0.5695, 0.7174, 0.6804, 0.5523, 0.5579])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-3

    def test_stable_diffusion_pix2pix_euler(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()
        components["scheduler"] = EulerAncestralDiscreteScheduler(
            beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear"
        )
        sd_pipe = StableDiffusionInstructPix2PixPipeline(**components)
        sd_pipe = sd_pipe.to(device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        image = sd_pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 32, 32, 3)
        expected_slice = np.array([0.7417, 0.3842, 0.4732, 0.5776, 0.5891, 0.5139, 0.4052, 0.5673, 0.4986])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-3

    def test_inference_batch_single_identical(self):
        super().test_inference_batch_single_identical(expected_max_diff=3e-3)

    # Overwrite the default test_latents_inputs because pix2pix encode the image differently
    def test_latents_input(self):
        components = self.get_dummy_components()
        pipe = StableDiffusionInstructPix2PixPipeline(**components)
        pipe.image_processor = VaeImageProcessor(do_resize=False, do_normalize=False)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        out = pipe(**self.get_dummy_inputs_by_type(torch_device, input_image_type="pt"))[0]

        vae = components["vae"]
        inputs = self.get_dummy_inputs_by_type(torch_device, input_image_type="pt")

        for image_param in self.image_latents_params:
            if image_param in inputs.keys():
                inputs[image_param] = vae.encode(inputs[image_param]).latent_dist.mode()

        out_latents_inputs = pipe(**inputs)[0]

        max_diff = np.abs(out - out_latents_inputs).max()
        self.assertLess(max_diff, 1e-4, "passing latents as image input generate different result from passing image")

    # Override the default test_callback_cfg because pix2pix create inputs for cfg differently
    def test_callback_cfg(self):
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        def callback_no_cfg(pipe, i, t, callback_kwargs):
            if i == 1:
                for k, w in callback_kwargs.items():
                    if k in self.callback_cfg_params:
                        callback_kwargs[k] = callback_kwargs[k].chunk(3)[0]
                pipe._guidance_scale = 1.0

            return callback_kwargs

        inputs = self.get_dummy_inputs(torch_device)
        inputs["guidance_scale"] = 1.0
        inputs["num_inference_steps"] = 2
        out_no_cfg = pipe(**inputs)[0]

        inputs["guidance_scale"] = 7.5
        inputs["callback_on_step_end"] = callback_no_cfg
        inputs["callback_on_step_end_tensor_inputs"] = pipe._callback_tensor_inputs
        out_callback_no_cfg = pipe(**inputs)[0]

        assert out_no_cfg.shape == out_callback_no_cfg.shape


@slow
@require_torch_gpu
class StableDiffusionInstructPix2PixPipelineSlowTests(unittest.TestCase):
    def setUp(self):
        super().setUp()
        gc.collect()
        torch.cuda.empty_cache()

    def tearDown(self):
        super().tearDown()
        gc.collect()
        torch.cuda.empty_cache()

    def get_inputs(self, seed=0):
        generator = torch.manual_seed(seed)
        image = load_image(
            "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main/stable_diffusion_pix2pix/example.jpg"
        )
        inputs = {
            "prompt": "turn him into a cyborg",
            "image": image,
            "generator": generator,
            "num_inference_steps": 3,
            "guidance_scale": 7.5,
            "image_guidance_scale": 1.0,
            "output_type": "np",
        }
        return inputs

    def test_stable_diffusion_pix2pix_default(self):
        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            "timbrooks/instruct-pix2pix", safety_checker=None
        )
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()

        inputs = self.get_inputs()
        image = pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1].flatten()

        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array([0.5902, 0.6015, 0.6027, 0.5983, 0.6092, 0.6061, 0.5765, 0.5785, 0.5555])

        assert np.abs(expected_slice - image_slice).max() < 1e-3

    def test_stable_diffusion_pix2pix_k_lms(self):
        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            "timbrooks/instruct-pix2pix", safety_checker=None
        )
        pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config)
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()

        inputs = self.get_inputs()
        image = pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1].flatten()

        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array([0.6578, 0.6817, 0.6972, 0.6761, 0.6856, 0.6916, 0.6428, 0.6516, 0.6301])

        assert np.abs(expected_slice - image_slice).max() < 1e-3

    def test_stable_diffusion_pix2pix_ddim(self):
        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            "timbrooks/instruct-pix2pix", safety_checker=None
        )
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()

        inputs = self.get_inputs()
        image = pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1].flatten()

        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array([0.3828, 0.3834, 0.3818, 0.3792, 0.3865, 0.3752, 0.3792, 0.3847, 0.3753])

        assert np.abs(expected_slice - image_slice).max() < 1e-3

    def test_stable_diffusion_pix2pix_intermediate_state(self):
        number_of_steps = 0

        def callback_fn(step: int, timestep: int, latents: torch.Tensor) -> None:
            callback_fn.has_been_called = True
            nonlocal number_of_steps
            number_of_steps += 1
            if step == 1:
                latents = latents.detach().cpu().numpy()
                assert latents.shape == (1, 4, 64, 64)
                latents_slice = latents[0, -3:, -3:, -1]
                expected_slice = np.array([-0.2463, -0.4644, -0.9756, 1.5176, 1.4414, 0.7866, 0.9897, 0.8521, 0.7983])

                assert np.abs(latents_slice.flatten() - expected_slice).max() < 5e-2
            elif step == 2:
                latents = latents.detach().cpu().numpy()
                assert latents.shape == (1, 4, 64, 64)
                latents_slice = latents[0, -3:, -3:, -1]
                expected_slice = np.array([-0.2644, -0.4626, -0.9653, 1.5176, 1.4551, 0.7686, 0.9805, 0.8452, 0.8115])

                assert np.abs(latents_slice.flatten() - expected_slice).max() < 5e-2

        callback_fn.has_been_called = False

        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            "timbrooks/instruct-pix2pix", safety_checker=None, torch_dtype=torch.float16
        )
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()

        inputs = self.get_inputs()
        pipe(**inputs, callback=callback_fn, callback_steps=1)
        assert callback_fn.has_been_called
        assert number_of_steps == 3

    def test_stable_diffusion_pipeline_with_sequential_cpu_offloading(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()

        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            "timbrooks/instruct-pix2pix", safety_checker=None, torch_dtype=torch.float16
        )
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing(1)
        pipe.enable_sequential_cpu_offload()

        inputs = self.get_inputs()
        _ = pipe(**inputs)

        mem_bytes = torch.cuda.max_memory_allocated()
        # make sure that less than 2.2 GB is allocated
        assert mem_bytes < 2.2 * 10**9

    def test_stable_diffusion_pix2pix_pipeline_multiple_of_8(self):
        inputs = self.get_inputs()
        # resize to resolution that is divisible by 8 but not 16 or 32
        inputs["image"] = inputs["image"].resize((504, 504))

        model_id = "timbrooks/instruct-pix2pix"
        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            model_id,
            safety_checker=None,
        )
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()

        output = pipe(**inputs)
        image = output.images[0]

        image_slice = image[255:258, 383:386, -1]

        assert image.shape == (504, 504, 3)
        expected_slice = np.array([0.2726, 0.2529, 0.2664, 0.2655, 0.2641, 0.2642, 0.2591, 0.2649, 0.2590])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 5e-3
