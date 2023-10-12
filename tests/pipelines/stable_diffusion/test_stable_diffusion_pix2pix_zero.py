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
import tempfile
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
from diffusers.image_processor import VaeImageProcessor
from diffusers.utils.testing_utils import (
    enable_full_determinism,
    floats_tensor,
    load_image,
    load_numpy,
    load_pt,
    nightly,
    require_torch_gpu,
    skip_mps,
    torch_device,
)

from ..pipeline_params import (
    TEXT_GUIDED_IMAGE_VARIATION_BATCH_PARAMS,
    TEXT_GUIDED_IMAGE_VARIATION_PARAMS,
    TEXT_TO_IMAGE_IMAGE_PARAMS,
)
from ..test_pipelines_common import (
    PipelineLatentTesterMixin,
    PipelineTesterMixin,
    assert_mean_pixel_difference,
)


enable_full_determinism()


@skip_mps
class StableDiffusionPix2PixZeroPipelineFastTests(PipelineLatentTesterMixin, PipelineTesterMixin, unittest.TestCase):
    pipeline_class = StableDiffusionPix2PixZeroPipeline
    params = TEXT_GUIDED_IMAGE_VARIATION_PARAMS - {"image"}
    batch_params = TEXT_GUIDED_IMAGE_VARIATION_BATCH_PARAMS
    image_params = TEXT_TO_IMAGE_IMAGE_PARAMS
    image_latents_params = TEXT_TO_IMAGE_IMAGE_PARAMS

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
        inverse_scheduler = DDIMInverseScheduler()
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
            "inverse_scheduler": inverse_scheduler,
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

    def get_dummy_inversion_inputs(self, device, seed=0):
        dummy_image = floats_tensor((2, 3, 32, 32), rng=random.Random(seed)).to(torch_device)
        dummy_image = dummy_image / 2 + 0.5
        generator = torch.manual_seed(seed)

        inputs = {
            "prompt": [
                "A painting of a squirrel eating a burger",
                "A painting of a burger eating a squirrel",
            ],
            "image": dummy_image.cpu(),
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            "generator": generator,
            "output_type": "numpy",
        }
        return inputs

    def get_dummy_inversion_inputs_by_type(self, device, seed=0, input_image_type="pt", output_type="np"):
        inputs = self.get_dummy_inversion_inputs(device, seed)

        if input_image_type == "pt":
            image = inputs["image"]
        elif input_image_type == "np":
            image = VaeImageProcessor.pt_to_numpy(inputs["image"])
        elif input_image_type == "pil":
            image = VaeImageProcessor.pt_to_numpy(inputs["image"])
            image = VaeImageProcessor.numpy_to_pil(image)
        else:
            raise ValueError(f"unsupported input_image_type {input_image_type}")

        inputs["image"] = image
        inputs["output_type"] = output_type

        return inputs

    def test_save_load_optional_components(self):
        if not hasattr(self.pipeline_class, "_optional_components"):
            return

        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        # set all optional components to None and update pipeline config accordingly
        for optional_component in pipe._optional_components:
            setattr(pipe, optional_component, None)
        pipe.register_modules(**{optional_component: None for optional_component in pipe._optional_components})

        inputs = self.get_dummy_inputs(torch_device)
        output = pipe(**inputs)[0]

        with tempfile.TemporaryDirectory() as tmpdir:
            pipe.save_pretrained(tmpdir)
            pipe_loaded = self.pipeline_class.from_pretrained(tmpdir)
            pipe_loaded.to(torch_device)
            pipe_loaded.set_progress_bar_config(disable=None)

        for optional_component in pipe._optional_components:
            self.assertTrue(
                getattr(pipe_loaded, optional_component) is None,
                f"`{optional_component}` did not stay set to None after loading.",
            )

        inputs = self.get_dummy_inputs(torch_device)
        output_loaded = pipe_loaded(**inputs)[0]

        max_diff = np.abs(output - output_loaded).max()
        self.assertLess(max_diff, 1e-4)

    def test_stable_diffusion_pix2pix_zero_inversion(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()
        sd_pipe = StableDiffusionPix2PixZeroPipeline(**components)
        sd_pipe = sd_pipe.to(device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inversion_inputs(device)
        inputs["image"] = inputs["image"][:1]
        inputs["prompt"] = inputs["prompt"][:1]
        image = sd_pipe.invert(**inputs).images
        image_slice = image[0, -3:, -3:, -1]
        assert image.shape == (1, 32, 32, 3)
        expected_slice = np.array([0.4732, 0.4630, 0.5722, 0.5103, 0.5140, 0.5622, 0.5104, 0.5390, 0.5020])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-3

    def test_stable_diffusion_pix2pix_zero_inversion_batch(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()
        sd_pipe = StableDiffusionPix2PixZeroPipeline(**components)
        sd_pipe = sd_pipe.to(device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inversion_inputs(device)
        image = sd_pipe.invert(**inputs).images
        image_slice = image[1, -3:, -3:, -1]
        assert image.shape == (2, 32, 32, 3)
        expected_slice = np.array([0.6046, 0.5400, 0.4902, 0.4448, 0.4694, 0.5498, 0.4857, 0.5073, 0.5089])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-3

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
        expected_slice = np.array([0.4863, 0.5053, 0.5033, 0.4007, 0.3571, 0.4768, 0.5176, 0.5277, 0.4940])

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
        expected_slice = np.array([0.5177, 0.5097, 0.5047, 0.4076, 0.3667, 0.4767, 0.5238, 0.5307, 0.4958])

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
        expected_slice = np.array([0.5421, 0.5525, 0.6085, 0.5279, 0.4658, 0.5317, 0.4418, 0.4815, 0.5132])

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
        expected_slice = np.array([0.4861, 0.5053, 0.5038, 0.3994, 0.3562, 0.4768, 0.5172, 0.5280, 0.4938])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-3

    def test_stable_diffusion_pix2pix_zero_inversion_pt_np_pil_outputs_equivalent(self):
        device = torch_device
        components = self.get_dummy_components()
        sd_pipe = StableDiffusionPix2PixZeroPipeline(**components)
        sd_pipe = sd_pipe.to(device)
        sd_pipe.set_progress_bar_config(disable=None)

        output_pt = sd_pipe.invert(**self.get_dummy_inversion_inputs_by_type(device, output_type="pt")).images
        output_np = sd_pipe.invert(**self.get_dummy_inversion_inputs_by_type(device, output_type="np")).images
        output_pil = sd_pipe.invert(**self.get_dummy_inversion_inputs_by_type(device, output_type="pil")).images

        max_diff = np.abs(output_pt.cpu().numpy().transpose(0, 2, 3, 1) - output_np).max()
        self.assertLess(max_diff, 1e-4, "`output_type=='pt'` generate different results from `output_type=='np'`")

        max_diff = np.abs(np.array(output_pil[0]) - (output_np[0] * 255).round()).max()
        self.assertLess(max_diff, 2.0, "`output_type=='pil'` generate different results from `output_type=='np'`")

    def test_stable_diffusion_pix2pix_zero_inversion_pt_np_pil_inputs_equivalent(self):
        device = torch_device
        components = self.get_dummy_components()
        sd_pipe = StableDiffusionPix2PixZeroPipeline(**components)
        sd_pipe = sd_pipe.to(device)
        sd_pipe.set_progress_bar_config(disable=None)

        out_input_pt = sd_pipe.invert(**self.get_dummy_inversion_inputs_by_type(device, input_image_type="pt")).images
        out_input_np = sd_pipe.invert(**self.get_dummy_inversion_inputs_by_type(device, input_image_type="np")).images
        out_input_pil = sd_pipe.invert(
            **self.get_dummy_inversion_inputs_by_type(device, input_image_type="pil")
        ).images

        max_diff = np.abs(out_input_pt - out_input_np).max()
        self.assertLess(max_diff, 1e-4, "`input_type=='pt'` generate different result from `input_type=='np'`")

        assert_mean_pixel_difference(out_input_pil, out_input_np, expected_max_diff=1)

    # Non-determinism caused by the scheduler optimizing the latent inputs during inference
    @unittest.skip("non-deterministic pipeline")
    def test_inference_batch_single_identical(self):
        return super().test_inference_batch_single_identical()


@nightly
@require_torch_gpu
class StableDiffusionPix2PixZeroPipelineNightlyTests(unittest.TestCase):
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


@nightly
@require_torch_gpu
class InversionPipelineNightlyTests(unittest.TestCase):
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
        expected_slice = np.array([0.8447, -0.0730, 0.7588, -1.2070, -0.4678, 0.1511, -0.8555, 1.1816, -0.7666])

        assert np.abs(expected_slice - image_slice.cpu().numpy()).max() < 5e-2

    def test_stable_diffusion_2_pix2pix_inversion(self):
        pipe = StableDiffusionPix2PixZeroPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-1", safety_checker=None, torch_dtype=torch.float16
        )
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
        expected_slice = np.array([0.8970, -0.1611, 0.4766, -1.1162, -0.5923, 0.1050, -0.9678, 1.0537, -0.6050])

        assert np.abs(expected_slice - image_slice.cpu().numpy()).max() < 5e-2

    def test_stable_diffusion_2_pix2pix_full(self):
        # numpy array of https://huggingface.co/datasets/hf-internal-testing/diffusers-images/blob/main/pix2pix/dog_2.png
        expected_image = load_numpy(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/pix2pix/dog_2.npy"
        )

        pipe = StableDiffusionPix2PixZeroPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-1", safety_checker=None, torch_dtype=torch.float16
        )
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
            num_inference_steps=125,
            cross_attention_guidance_amount=0.015,
            generator=generator,
            latents=inv_latents,
            negative_prompt=caption,
            output_type="np",
        ).images

        mean_diff = np.abs(expected_image - image).mean()
        assert mean_diff < 0.25
