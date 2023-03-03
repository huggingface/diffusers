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
    DDIMInverseScheduler,
    DDIMScheduler,
    DDPMScheduler,
    EulerAncestralDiscreteScheduler,
    LMSDiscreteScheduler,
    StableDiffusionPix2PixZeroPipeline,
    UNet2DConditionModel,
)
from diffusers.utils import load_numpy, slow, torch_device
from diffusers.utils.testing_utils import load_image, load_pt, require_torch_gpu, skip_mps

from ...pipeline_params import TEXT_GUIDED_IMAGE_VARIATION_BATCH_PARAMS, TEXT_GUIDED_IMAGE_VARIATION_PARAMS
from ...test_pipelines_common import PipelineTesterMixin


torch.backends.cuda.matmul.allow_tf32 = False


@skip_mps
class StableDiffusionPix2PixZeroPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = StableDiffusionPix2PixZeroPipeline
    params = TEXT_GUIDED_IMAGE_VARIATION_PARAMS
    batch_params = TEXT_GUIDED_IMAGE_VARIATION_BATCH_PARAMS

    @classmethod
    def setUpClass(cls):
        cls.source_embeds = load_pt(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/pix2pix/src_emb_0.pt"
        )

        cls.target_embeds = load_pt(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/pix2pix/tgt_emb_0.pt"
        )

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
            "inverse_scheduler": None,
            "caption_generator": None,
            "caption_processor": None,
        }
        return components

    def get_dummy_inputs(self, device, seed=0):
        generator = torch.manual_seed(seed)

        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            "cross_attention_guidance_amount": 0.15,
            "source_embeds": self.source_embeds,
            "target_embeds": self.target_embeds,
            "output_type": "numpy",
        }
        return inputs

    def test_stable_diffusion_pix2pix_zero_default_case(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()
        sd_pipe = StableDiffusionPix2PixZeroPipeline(**components)
        sd_pipe = sd_pipe.to(device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        image = sd_pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1]
        assert image.shape == (1, 64, 64, 3)
        expected_slice = np.array([0.5184, 0.503, 0.4917, 0.4022, 0.3455, 0.464, 0.5324, 0.5323, 0.4894])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-3

    def test_stable_diffusion_pix2pix_zero_negative_prompt(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()
        sd_pipe = StableDiffusionPix2PixZeroPipeline(**components)
        sd_pipe = sd_pipe.to(device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        negative_prompt = "french fries"
        output = sd_pipe(**inputs, negative_prompt=negative_prompt)
        image = output.images
        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 64, 64, 3)
        expected_slice = np.array([0.5464, 0.5072, 0.5012, 0.4124, 0.3624, 0.466, 0.5413, 0.5468, 0.4927])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-3

    def test_stable_diffusion_pix2pix_zero_euler(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()
        components["scheduler"] = EulerAncestralDiscreteScheduler(
            beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear"
        )
        sd_pipe = StableDiffusionPix2PixZeroPipeline(**components)
        sd_pipe = sd_pipe.to(device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        image = sd_pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 64, 64, 3)
        expected_slice = np.array([0.5114, 0.5051, 0.5222, 0.5279, 0.5037, 0.5156, 0.4604, 0.4966, 0.504])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-3

    def test_stable_diffusion_pix2pix_zero_ddpm(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()
        components["scheduler"] = DDPMScheduler()
        sd_pipe = StableDiffusionPix2PixZeroPipeline(**components)
        sd_pipe = sd_pipe.to(device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        image = sd_pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 64, 64, 3)
        expected_slice = np.array([0.5185, 0.5027, 0.492, 0.401, 0.3445, 0.464, 0.5321, 0.5327, 0.4892])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-3

    def test_stable_diffusion_pix2pix_zero_num_images_per_prompt(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()
        sd_pipe = StableDiffusionPix2PixZeroPipeline(**components)
        sd_pipe = sd_pipe.to(device)
        sd_pipe.set_progress_bar_config(disable=None)

        # test num_images_per_prompt=1 (default)
        inputs = self.get_dummy_inputs(device)
        images = sd_pipe(**inputs).images

        assert images.shape == (1, 64, 64, 3)

        # test num_images_per_prompt=2 for a single prompt
        num_images_per_prompt = 2
        inputs = self.get_dummy_inputs(device)
        images = sd_pipe(**inputs, num_images_per_prompt=num_images_per_prompt).images

        assert images.shape == (num_images_per_prompt, 64, 64, 3)

        # test num_images_per_prompt for batch of prompts
        batch_size = 2
        inputs = self.get_dummy_inputs(device)
        inputs["prompt"] = [inputs["prompt"]] * batch_size
        images = sd_pipe(**inputs, num_images_per_prompt=num_images_per_prompt).images

        assert images.shape == (batch_size * num_images_per_prompt, 64, 64, 3)

    # Non-determinism caused by the scheduler optimizing the latent inputs during inference
    @unittest.skip("non-deterministic pipeline")
    def test_inference_batch_single_identical(self):
        return super().test_inference_batch_single_identical()


@slow
@require_torch_gpu
class StableDiffusionPix2PixZeroPipelineSlowTests(unittest.TestCase):
    def tearDown(self):
        super().tearDown()
        gc.collect()
        torch.cuda.empty_cache()

    @classmethod
    def setUpClass(cls):
        cls.source_embeds = load_pt(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/pix2pix/cat.pt"
        )

        cls.target_embeds = load_pt(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/pix2pix/dog.pt"
        )

    def get_inputs(self, seed=0):
        generator = torch.manual_seed(seed)

        inputs = {
            "prompt": "turn him into a cyborg",
            "generator": generator,
            "num_inference_steps": 3,
            "guidance_scale": 7.5,
            "cross_attention_guidance_amount": 0.15,
            "source_embeds": self.source_embeds,
            "target_embeds": self.target_embeds,
            "output_type": "numpy",
        }
        return inputs

    def test_stable_diffusion_pix2pix_zero_default(self):
        pipe = StableDiffusionPix2PixZeroPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4", safety_checker=None, torch_dtype=torch.float16
        )
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()

        inputs = self.get_inputs()
        image = pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1].flatten()

        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array([0.5742, 0.5757, 0.5747, 0.5781, 0.5688, 0.5713, 0.5742, 0.5664, 0.5747])

        assert np.abs(expected_slice - image_slice).max() < 5e-2

    def test_stable_diffusion_pix2pix_zero_k_lms(self):
        pipe = StableDiffusionPix2PixZeroPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4", safety_checker=None, torch_dtype=torch.float16
        )
        pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config)
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()

        inputs = self.get_inputs()
        image = pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1].flatten()

        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array([0.6367, 0.5459, 0.5146, 0.5479, 0.4905, 0.4753, 0.4961, 0.4629, 0.4624])

        assert np.abs(expected_slice - image_slice).max() < 5e-2

    def test_stable_diffusion_pix2pix_zero_intermediate_state(self):
        number_of_steps = 0

        def callback_fn(step: int, timestep: int, latents: torch.FloatTensor) -> None:
            callback_fn.has_been_called = True
            nonlocal number_of_steps
            number_of_steps += 1
            if step == 1:
                latents = latents.detach().cpu().numpy()
                assert latents.shape == (1, 4, 64, 64)
                latents_slice = latents[0, -3:, -3:, -1]
                expected_slice = np.array([0.1345, 0.268, 0.1539, 0.0726, 0.0959, 0.2261, -0.2673, 0.0277, -0.2062])

                assert np.abs(latents_slice.flatten() - expected_slice).max() < 5e-2
            elif step == 2:
                latents = latents.detach().cpu().numpy()
                assert latents.shape == (1, 4, 64, 64)
                latents_slice = latents[0, -3:, -3:, -1]
                expected_slice = np.array([0.1393, 0.2637, 0.1617, 0.0724, 0.0987, 0.2271, -0.2666, 0.0299, -0.2104])

                assert np.abs(latents_slice.flatten() - expected_slice).max() < 5e-2

        callback_fn.has_been_called = False

        pipe = StableDiffusionPix2PixZeroPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4", safety_checker=None, torch_dtype=torch.float16
        )
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
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

        pipe = StableDiffusionPix2PixZeroPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4", safety_checker=None, torch_dtype=torch.float16
        )
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing(1)
        pipe.enable_sequential_cpu_offload()

        inputs = self.get_inputs()
        _ = pipe(**inputs)

        mem_bytes = torch.cuda.max_memory_allocated()
        # make sure that less than 8.2 GB is allocated
        assert mem_bytes < 8.2 * 10**9


@slow
@require_torch_gpu
class InversionPipelineSlowTests(unittest.TestCase):
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

    def test_stable_diffusion_pix2pix_inversion(self):
        pipe = StableDiffusionPix2PixZeroPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4", safety_checker=None, torch_dtype=torch.float16
        )
        pipe.inverse_scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.inverse_scheduler = DDIMInverseScheduler.from_config(pipe.scheduler.config)

        caption = "a photography of a cat with flowers"
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.enable_model_cpu_offload()
        pipe.set_progress_bar_config(disable=None)

        generator = torch.manual_seed(0)
        output = pipe.invert(caption, image=self.raw_image, generator=generator, num_inference_steps=10)
        inv_latents = output[0]

        image_slice = inv_latents[0, -3:, -3:, -1].flatten()

        assert inv_latents.shape == (1, 4, 64, 64)
        expected_slice = np.array([0.8877, 0.0587, 0.7700, -1.6035, -0.5962, 0.4827, -0.6265, 1.0498, -0.8599])

        assert np.abs(expected_slice - image_slice.cpu().numpy()).max() < 5e-2

    def test_stable_diffusion_pix2pix_full(self):
        # numpy array of https://huggingface.co/datasets/hf-internal-testing/diffusers-images/blob/main/pix2pix/dog.png
        expected_image = load_numpy(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/pix2pix/dog.npy"
        )

        pipe = StableDiffusionPix2PixZeroPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4", safety_checker=None, torch_dtype=torch.float16
        )
        pipe.inverse_scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.inverse_scheduler = DDIMInverseScheduler.from_config(pipe.scheduler.config)

        caption = "a photography of a cat with flowers"
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.enable_model_cpu_offload()
        pipe.set_progress_bar_config(disable=None)

        generator = torch.manual_seed(0)
        output = pipe.invert(caption, image=self.raw_image, generator=generator)
        inv_latents = output[0]

        source_prompts = 4 * ["a cat sitting on the street", "a cat playing in the field", "a face of a cat"]
        target_prompts = 4 * ["a dog sitting on the street", "a dog playing in the field", "a face of a dog"]

        source_embeds = pipe.get_embeds(source_prompts)
        target_embeds = pipe.get_embeds(target_prompts)

        image = pipe(
            caption,
            source_embeds=source_embeds,
            target_embeds=target_embeds,
            num_inference_steps=50,
            cross_attention_guidance_amount=0.15,
            generator=generator,
            latents=inv_latents,
            negative_prompt=caption,
            output_type="np",
        ).images

        max_diff = np.abs(expected_image - image).mean()
        assert max_diff < 0.05
