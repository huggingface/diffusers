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
import tempfile
import time
import unittest

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer

from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
    logging,
)
from diffusers.utils import load_numpy, nightly, slow, torch_device
from diffusers.utils.testing_utils import CaptureLogger, require_torch_gpu

from ...models.test_models_unet_2d_condition import create_lora_layers
from ...pipeline_params import TEXT_TO_IMAGE_BATCH_PARAMS, TEXT_TO_IMAGE_PARAMS
from ...test_pipelines_common import PipelineTesterMixin


torch.backends.cuda.matmul.allow_tf32 = False


class StableDiffusionPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = StableDiffusionPipeline
    params = TEXT_TO_IMAGE_PARAMS
    batch_params = TEXT_TO_IMAGE_BATCH_PARAMS

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
        scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
        )
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
            "prompt": "A painting of a squirrel eating a burger",
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            "output_type": "numpy",
        }
        return inputs

    def test_stable_diffusion_ddim(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator

        components = self.get_dummy_components()
        sd_pipe = StableDiffusionPipeline(**components)
        sd_pipe = sd_pipe.to(torch_device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        output = sd_pipe(**inputs)
        image = output.images

        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 64, 64, 3)
        expected_slice = np.array([0.5643, 0.6017, 0.4799, 0.5267, 0.5584, 0.4641, 0.5159, 0.4963, 0.4791])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

    def test_stable_diffusion_lora(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator

        components = self.get_dummy_components()
        sd_pipe = StableDiffusionPipeline(**components)
        sd_pipe = sd_pipe.to(torch_device)
        sd_pipe.set_progress_bar_config(disable=None)

        # forward 1
        inputs = self.get_dummy_inputs(device)
        output = sd_pipe(**inputs)
        image = output.images
        image_slice = image[0, -3:, -3:, -1]

        # set lora layers
        lora_attn_procs = create_lora_layers(sd_pipe.unet)
        sd_pipe.unet.set_attn_processor(lora_attn_procs)
        sd_pipe = sd_pipe.to(torch_device)

        # forward 2
        inputs = self.get_dummy_inputs(device)
        output = sd_pipe(**inputs, cross_attention_kwargs={"scale": 0.0})
        image = output.images
        image_slice_1 = image[0, -3:, -3:, -1]

        # forward 3
        inputs = self.get_dummy_inputs(device)
        output = sd_pipe(**inputs, cross_attention_kwargs={"scale": 0.5})
        image = output.images
        image_slice_2 = image[0, -3:, -3:, -1]

        assert np.abs(image_slice - image_slice_1).max() < 1e-2
        assert np.abs(image_slice - image_slice_2).max() > 1e-2

    def test_stable_diffusion_prompt_embeds(self):
        components = self.get_dummy_components()
        sd_pipe = StableDiffusionPipeline(**components)
        sd_pipe = sd_pipe.to(torch_device)
        sd_pipe = sd_pipe.to(torch_device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(torch_device)
        inputs["prompt"] = 3 * [inputs["prompt"]]

        # forward
        output = sd_pipe(**inputs)
        image_slice_1 = output.images[0, -3:, -3:, -1]

        inputs = self.get_dummy_inputs(torch_device)
        prompt = 3 * [inputs.pop("prompt")]

        text_inputs = sd_pipe.tokenizer(
            prompt,
            padding="max_length",
            max_length=sd_pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_inputs = text_inputs["input_ids"].to(torch_device)

        prompt_embeds = sd_pipe.text_encoder(text_inputs)[0]

        inputs["prompt_embeds"] = prompt_embeds

        # forward
        output = sd_pipe(**inputs)
        image_slice_2 = output.images[0, -3:, -3:, -1]

        assert np.abs(image_slice_1.flatten() - image_slice_2.flatten()).max() < 1e-4

    def test_stable_diffusion_negative_prompt_embeds(self):
        components = self.get_dummy_components()
        sd_pipe = StableDiffusionPipeline(**components)
        sd_pipe = sd_pipe.to(torch_device)
        sd_pipe = sd_pipe.to(torch_device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(torch_device)
        negative_prompt = 3 * ["this is a negative prompt"]
        inputs["negative_prompt"] = negative_prompt
        inputs["prompt"] = 3 * [inputs["prompt"]]

        # forward
        output = sd_pipe(**inputs)
        image_slice_1 = output.images[0, -3:, -3:, -1]

        inputs = self.get_dummy_inputs(torch_device)
        prompt = 3 * [inputs.pop("prompt")]

        embeds = []
        for p in [prompt, negative_prompt]:
            text_inputs = sd_pipe.tokenizer(
                p,
                padding="max_length",
                max_length=sd_pipe.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_inputs = text_inputs["input_ids"].to(torch_device)

            embeds.append(sd_pipe.text_encoder(text_inputs)[0])

        inputs["prompt_embeds"], inputs["negative_prompt_embeds"] = embeds

        # forward
        output = sd_pipe(**inputs)
        image_slice_2 = output.images[0, -3:, -3:, -1]

        assert np.abs(image_slice_1.flatten() - image_slice_2.flatten()).max() < 1e-4

    def test_stable_diffusion_ddim_factor_8(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator

        components = self.get_dummy_components()
        sd_pipe = StableDiffusionPipeline(**components)
        sd_pipe = sd_pipe.to(device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        output = sd_pipe(**inputs, height=136, width=136)
        image = output.images

        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 136, 136, 3)
        expected_slice = np.array([0.5524, 0.5626, 0.6069, 0.4727, 0.386, 0.3995, 0.4613, 0.4328, 0.4269])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

    def test_stable_diffusion_pndm(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()
        sd_pipe = StableDiffusionPipeline(**components)
        sd_pipe.scheduler = PNDMScheduler(skip_prk_steps=True)
        sd_pipe = sd_pipe.to(device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        output = sd_pipe(**inputs)
        image = output.images
        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 64, 64, 3)
        expected_slice = np.array([0.5094, 0.5674, 0.4667, 0.5125, 0.5696, 0.4674, 0.5277, 0.4964, 0.4945])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

    def test_stable_diffusion_no_safety_checker(self):
        pipe = StableDiffusionPipeline.from_pretrained(
            "hf-internal-testing/tiny-stable-diffusion-lms-pipe", safety_checker=None
        )
        assert isinstance(pipe, StableDiffusionPipeline)
        assert isinstance(pipe.scheduler, LMSDiscreteScheduler)
        assert pipe.safety_checker is None

        image = pipe("example prompt", num_inference_steps=2).images[0]
        assert image is not None

        # check that there's no error when saving a pipeline with one of the models being None
        with tempfile.TemporaryDirectory() as tmpdirname:
            pipe.save_pretrained(tmpdirname)
            pipe = StableDiffusionPipeline.from_pretrained(tmpdirname)

        # sanity check that the pipeline still works
        assert pipe.safety_checker is None
        image = pipe("example prompt", num_inference_steps=2).images[0]
        assert image is not None

    def test_stable_diffusion_k_lms(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator

        components = self.get_dummy_components()
        sd_pipe = StableDiffusionPipeline(**components)
        sd_pipe.scheduler = LMSDiscreteScheduler.from_config(sd_pipe.scheduler.config)
        sd_pipe = sd_pipe.to(device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        output = sd_pipe(**inputs)
        image = output.images
        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 64, 64, 3)
        expected_slice = np.array(
            [
                0.47082293033599854,
                0.5371589064598083,
                0.4562119245529175,
                0.5220914483070374,
                0.5733777284622192,
                0.4795039892196655,
                0.5465868711471558,
                0.5074326395988464,
                0.5042197108268738,
            ]
        )

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

    def test_stable_diffusion_k_euler_ancestral(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator

        components = self.get_dummy_components()
        sd_pipe = StableDiffusionPipeline(**components)
        sd_pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(sd_pipe.scheduler.config)
        sd_pipe = sd_pipe.to(device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        output = sd_pipe(**inputs)
        image = output.images
        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 64, 64, 3)
        expected_slice = np.array(
            [
                0.4707113206386566,
                0.5372191071510315,
                0.4563021957874298,
                0.5220003724098206,
                0.5734264850616455,
                0.4794946610927582,
                0.5463782548904419,
                0.5074145197868347,
                0.504422664642334,
            ]
        )

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

    def test_stable_diffusion_k_euler(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator

        components = self.get_dummy_components()
        sd_pipe = StableDiffusionPipeline(**components)
        sd_pipe.scheduler = EulerDiscreteScheduler.from_config(sd_pipe.scheduler.config)
        sd_pipe = sd_pipe.to(device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        output = sd_pipe(**inputs)
        image = output.images
        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 64, 64, 3)
        expected_slice = np.array(
            [
                0.47082313895225525,
                0.5371587872505188,
                0.4562119245529175,
                0.5220913887023926,
                0.5733776688575745,
                0.47950395941734314,
                0.546586811542511,
                0.5074326992034912,
                0.5042197108268738,
            ]
        )

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

    def test_stable_diffusion_vae_slicing(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()
        components["scheduler"] = LMSDiscreteScheduler.from_config(components["scheduler"].config)
        sd_pipe = StableDiffusionPipeline(**components)
        sd_pipe = sd_pipe.to(device)
        sd_pipe.set_progress_bar_config(disable=None)

        image_count = 4

        inputs = self.get_dummy_inputs(device)
        inputs["prompt"] = [inputs["prompt"]] * image_count
        output_1 = sd_pipe(**inputs)

        # make sure sliced vae decode yields the same result
        sd_pipe.enable_vae_slicing()
        inputs = self.get_dummy_inputs(device)
        inputs["prompt"] = [inputs["prompt"]] * image_count
        output_2 = sd_pipe(**inputs)

        # there is a small discrepancy at image borders vs. full batch decode
        assert np.abs(output_2.images.flatten() - output_1.images.flatten()).max() < 3e-3

    def test_stable_diffusion_vae_tiling(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()

        # make sure here that pndm scheduler skips prk
        components["safety_checker"] = None
        sd_pipe = StableDiffusionPipeline(**components)
        sd_pipe = sd_pipe.to(device)
        sd_pipe.set_progress_bar_config(disable=None)

        prompt = "A painting of a squirrel eating a burger"

        # Test that tiled decode at 512x512 yields the same result as the non-tiled decode
        generator = torch.Generator(device=device).manual_seed(0)
        output_1 = sd_pipe([prompt], generator=generator, guidance_scale=6.0, num_inference_steps=2, output_type="np")

        # make sure tiled vae decode yields the same result
        sd_pipe.enable_vae_tiling()
        generator = torch.Generator(device=device).manual_seed(0)
        output_2 = sd_pipe([prompt], generator=generator, guidance_scale=6.0, num_inference_steps=2, output_type="np")

        assert np.abs(output_2.images.flatten() - output_1.images.flatten()).max() < 5e-1

        # test that tiled decode works with various shapes
        shapes = [(1, 4, 73, 97), (1, 4, 97, 73), (1, 4, 49, 65), (1, 4, 65, 49)]
        for shape in shapes:
            zeros = torch.zeros(shape).to(device)
            sd_pipe.vae.decode(zeros)

    def test_stable_diffusion_negative_prompt(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()
        components["scheduler"] = PNDMScheduler(skip_prk_steps=True)
        sd_pipe = StableDiffusionPipeline(**components)
        sd_pipe = sd_pipe.to(device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        negative_prompt = "french fries"
        output = sd_pipe(**inputs, negative_prompt=negative_prompt)

        image = output.images
        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 64, 64, 3)
        expected_slice = np.array(
            [
                0.5108221173286438,
                0.5688379406929016,
                0.4685141146183014,
                0.5098261833190918,
                0.5657756328582764,
                0.4631010890007019,
                0.5226285457611084,
                0.49129390716552734,
                0.4899061322212219,
            ]
        )

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

    def test_stable_diffusion_long_prompt(self):
        components = self.get_dummy_components()
        components["scheduler"] = LMSDiscreteScheduler.from_config(components["scheduler"].config)
        sd_pipe = StableDiffusionPipeline(**components)
        sd_pipe = sd_pipe.to(torch_device)
        sd_pipe.set_progress_bar_config(disable=None)

        do_classifier_free_guidance = True
        negative_prompt = None
        num_images_per_prompt = 1
        logger = logging.get_logger("diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion")

        prompt = 25 * "@"
        with CaptureLogger(logger) as cap_logger_3:
            text_embeddings_3 = sd_pipe._encode_prompt(
                prompt, torch_device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
            )

        prompt = 100 * "@"
        with CaptureLogger(logger) as cap_logger:
            text_embeddings = sd_pipe._encode_prompt(
                prompt, torch_device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
            )

        negative_prompt = "Hello"
        with CaptureLogger(logger) as cap_logger_2:
            text_embeddings_2 = sd_pipe._encode_prompt(
                prompt, torch_device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
            )

        assert text_embeddings_3.shape == text_embeddings_2.shape == text_embeddings.shape
        assert text_embeddings.shape[1] == 77

        assert cap_logger.out == cap_logger_2.out
        # 100 - 77 + 1 (BOS token) + 1 (EOS token) = 25
        assert cap_logger.out.count("@") == 25
        assert cap_logger_3.out == ""

    def test_stable_diffusion_height_width_opt(self):
        components = self.get_dummy_components()
        components["scheduler"] = LMSDiscreteScheduler.from_config(components["scheduler"].config)
        sd_pipe = StableDiffusionPipeline(**components)
        sd_pipe = sd_pipe.to(torch_device)
        sd_pipe.set_progress_bar_config(disable=None)

        prompt = "hey"

        output = sd_pipe(prompt, num_inference_steps=1, output_type="np")
        image_shape = output.images[0].shape[:2]
        assert image_shape == (64, 64)

        output = sd_pipe(prompt, num_inference_steps=1, height=96, width=96, output_type="np")
        image_shape = output.images[0].shape[:2]
        assert image_shape == (96, 96)

        config = dict(sd_pipe.unet.config)
        config["sample_size"] = 96
        sd_pipe.unet = UNet2DConditionModel.from_config(config).to(torch_device)
        output = sd_pipe(prompt, num_inference_steps=1, output_type="np")
        image_shape = output.images[0].shape[:2]
        assert image_shape == (192, 192)


@slow
@require_torch_gpu
class StableDiffusionPipelineSlowTests(unittest.TestCase):
    def tearDown(self):
        super().tearDown()
        gc.collect()
        torch.cuda.empty_cache()

    def get_inputs(self, device, generator_device="cpu", dtype=torch.float32, seed=0):
        generator = torch.Generator(device=generator_device).manual_seed(seed)
        latents = np.random.RandomState(seed).standard_normal((1, 4, 64, 64))
        latents = torch.from_numpy(latents).to(device=device, dtype=dtype)
        inputs = {
            "prompt": "a photograph of an astronaut riding a horse",
            "latents": latents,
            "generator": generator,
            "num_inference_steps": 3,
            "guidance_scale": 7.5,
            "output_type": "numpy",
        }
        return inputs

    def test_stable_diffusion_1_1_pndm(self):
        sd_pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-1")
        sd_pipe = sd_pipe.to(torch_device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs(torch_device)
        image = sd_pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1].flatten()

        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array([0.43625, 0.43554, 0.36670, 0.40660, 0.39703, 0.38658, 0.43936, 0.43557, 0.40592])
        assert np.abs(image_slice - expected_slice).max() < 1e-4

    def test_stable_diffusion_1_4_pndm(self):
        sd_pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
        sd_pipe = sd_pipe.to(torch_device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs(torch_device)
        image = sd_pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1].flatten()

        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array([0.57400, 0.47841, 0.31625, 0.63583, 0.58306, 0.55056, 0.50825, 0.56306, 0.55748])
        assert np.abs(image_slice - expected_slice).max() < 1e-4

    def test_stable_diffusion_ddim(self):
        sd_pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", safety_checker=None)
        sd_pipe.scheduler = DDIMScheduler.from_config(sd_pipe.scheduler.config)
        sd_pipe = sd_pipe.to(torch_device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs(torch_device)
        image = sd_pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1].flatten()

        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array([0.38019, 0.28647, 0.27321, 0.40377, 0.38290, 0.35446, 0.39218, 0.38165, 0.42239])
        assert np.abs(image_slice - expected_slice).max() < 1e-4

    def test_stable_diffusion_lms(self):
        sd_pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", safety_checker=None)
        sd_pipe.scheduler = LMSDiscreteScheduler.from_config(sd_pipe.scheduler.config)
        sd_pipe = sd_pipe.to(torch_device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs(torch_device)
        image = sd_pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1].flatten()

        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array([0.10542, 0.09620, 0.07332, 0.09015, 0.09382, 0.07597, 0.08496, 0.07806, 0.06455])
        assert np.abs(image_slice - expected_slice).max() < 1e-4

    def test_stable_diffusion_dpm(self):
        sd_pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", safety_checker=None)
        sd_pipe.scheduler = DPMSolverMultistepScheduler.from_config(sd_pipe.scheduler.config)
        sd_pipe = sd_pipe.to(torch_device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs(torch_device)
        image = sd_pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1].flatten()

        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array([0.03503, 0.03494, 0.01087, 0.03128, 0.02552, 0.00803, 0.00742, 0.00372, 0.00000])
        assert np.abs(image_slice - expected_slice).max() < 1e-4

    def test_stable_diffusion_attention_slicing(self):
        torch.cuda.reset_peak_memory_stats()
        pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        # enable attention slicing
        pipe.enable_attention_slicing()
        inputs = self.get_inputs(torch_device, dtype=torch.float16)
        image_sliced = pipe(**inputs).images

        mem_bytes = torch.cuda.max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        # make sure that less than 3.75 GB is allocated
        assert mem_bytes < 3.75 * 10**9

        # disable slicing
        pipe.disable_attention_slicing()
        inputs = self.get_inputs(torch_device, dtype=torch.float16)
        image = pipe(**inputs).images

        # make sure that more than 3.75 GB is allocated
        mem_bytes = torch.cuda.max_memory_allocated()
        assert mem_bytes > 3.75 * 10**9
        assert np.abs(image_sliced - image).max() < 1e-3

    def test_stable_diffusion_vae_slicing(self):
        torch.cuda.reset_peak_memory_stats()
        pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()

        # enable vae slicing
        pipe.enable_vae_slicing()
        inputs = self.get_inputs(torch_device, dtype=torch.float16)
        inputs["prompt"] = [inputs["prompt"]] * 4
        inputs["latents"] = torch.cat([inputs["latents"]] * 4)
        image_sliced = pipe(**inputs).images

        mem_bytes = torch.cuda.max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        # make sure that less than 4 GB is allocated
        assert mem_bytes < 4e9

        # disable vae slicing
        pipe.disable_vae_slicing()
        inputs = self.get_inputs(torch_device, dtype=torch.float16)
        inputs["prompt"] = [inputs["prompt"]] * 4
        inputs["latents"] = torch.cat([inputs["latents"]] * 4)
        image = pipe(**inputs).images

        # make sure that more than 4 GB is allocated
        mem_bytes = torch.cuda.max_memory_allocated()
        assert mem_bytes > 4e9
        # There is a small discrepancy at the image borders vs. a fully batched version.
        assert np.abs(image_sliced - image).max() < 1e-2

    def test_stable_diffusion_vae_tiling(self):
        torch.cuda.reset_peak_memory_stats()
        model_id = "CompVis/stable-diffusion-v1-4"
        pipe = StableDiffusionPipeline.from_pretrained(model_id, revision="fp16", torch_dtype=torch.float16)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()
        pipe.unet = pipe.unet.to(memory_format=torch.channels_last)
        pipe.vae = pipe.vae.to(memory_format=torch.channels_last)

        prompt = "a photograph of an astronaut riding a horse"

        # enable vae tiling
        pipe.enable_vae_tiling()
        pipe.enable_model_cpu_offload()
        generator = torch.Generator(device="cpu").manual_seed(0)
        output_chunked = pipe(
            [prompt],
            width=1024,
            height=1024,
            generator=generator,
            guidance_scale=7.5,
            num_inference_steps=2,
            output_type="numpy",
        )
        image_chunked = output_chunked.images

        mem_bytes = torch.cuda.max_memory_allocated()

        # disable vae tiling
        pipe.disable_vae_tiling()
        generator = torch.Generator(device="cpu").manual_seed(0)
        output = pipe(
            [prompt],
            width=1024,
            height=1024,
            generator=generator,
            guidance_scale=7.5,
            num_inference_steps=2,
            output_type="numpy",
        )
        image = output.images

        assert mem_bytes < 1e10
        assert np.abs(image_chunked.flatten() - image.flatten()).max() < 1e-2

    def test_stable_diffusion_fp16_vs_autocast(self):
        # this test makes sure that the original model with autocast
        # and the new model with fp16 yield the same result
        pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs(torch_device, dtype=torch.float16)
        image_fp16 = pipe(**inputs).images

        with torch.autocast(torch_device):
            inputs = self.get_inputs(torch_device)
            image_autocast = pipe(**inputs).images

        # Make sure results are close enough
        diff = np.abs(image_fp16.flatten() - image_autocast.flatten())
        # They ARE different since ops are not run always at the same precision
        # however, they should be extremely close.
        assert diff.mean() < 2e-2

    def test_stable_diffusion_intermediate_state(self):
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
                    [-0.5693, -0.3018, -0.9746, 0.0518, -0.8770, 0.7559, -1.7402, 0.1022, 1.1582]
                )

                assert np.abs(latents_slice.flatten() - expected_slice).max() < 5e-2
            elif step == 2:
                latents = latents.detach().cpu().numpy()
                assert latents.shape == (1, 4, 64, 64)
                latents_slice = latents[0, -3:, -3:, -1]
                expected_slice = np.array(
                    [-0.1958, -0.2993, -1.0166, -0.5005, -0.4810, 0.6162, -0.9492, 0.6621, 1.4492]
                )

                assert np.abs(latents_slice.flatten() - expected_slice).max() < 5e-2

        callback_fn.has_been_called = False

        pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()

        inputs = self.get_inputs(torch_device, dtype=torch.float16)
        pipe(**inputs, callback=callback_fn, callback_steps=1)
        assert callback_fn.has_been_called
        assert number_of_steps == inputs["num_inference_steps"]

    def test_stable_diffusion_low_cpu_mem_usage(self):
        pipeline_id = "CompVis/stable-diffusion-v1-4"

        start_time = time.time()
        pipeline_low_cpu_mem_usage = StableDiffusionPipeline.from_pretrained(pipeline_id, torch_dtype=torch.float16)
        pipeline_low_cpu_mem_usage.to(torch_device)
        low_cpu_mem_usage_time = time.time() - start_time

        start_time = time.time()
        _ = StableDiffusionPipeline.from_pretrained(pipeline_id, torch_dtype=torch.float16, low_cpu_mem_usage=False)
        normal_load_time = time.time() - start_time

        assert 2 * low_cpu_mem_usage_time < normal_load_time

    def test_stable_diffusion_pipeline_with_sequential_cpu_offloading(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()

        pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing(1)
        pipe.enable_sequential_cpu_offload()

        inputs = self.get_inputs(torch_device, dtype=torch.float16)
        _ = pipe(**inputs)

        mem_bytes = torch.cuda.max_memory_allocated()
        # make sure that less than 2.8 GB is allocated
        assert mem_bytes < 2.8 * 10**9

    def test_stable_diffusion_pipeline_with_model_offloading(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()

        inputs = self.get_inputs(torch_device, dtype=torch.float16)

        # Normal inference

        pipe = StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            torch_dtype=torch.float16,
        )
        pipe.unet.set_default_attn_processor()
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        outputs = pipe(**inputs)
        mem_bytes = torch.cuda.max_memory_allocated()

        # With model offloading

        # Reload but don't move to cuda
        pipe = StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            torch_dtype=torch.float16,
        )
        pipe.unet.set_default_attn_processor()

        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()

        pipe.enable_model_cpu_offload()
        pipe.set_progress_bar_config(disable=None)
        inputs = self.get_inputs(torch_device, dtype=torch.float16)

        outputs_offloaded = pipe(**inputs)
        mem_bytes_offloaded = torch.cuda.max_memory_allocated()

        assert np.abs(outputs.images - outputs_offloaded.images).max() < 1e-3
        assert mem_bytes_offloaded < mem_bytes
        assert mem_bytes_offloaded < 3.5 * 10**9
        for module in pipe.text_encoder, pipe.unet, pipe.vae, pipe.safety_checker:
            assert module.device == torch.device("cpu")

        # With attention slicing
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()

        pipe.enable_attention_slicing()
        _ = pipe(**inputs)
        mem_bytes_slicing = torch.cuda.max_memory_allocated()

        assert mem_bytes_slicing < mem_bytes_offloaded
        assert mem_bytes_slicing < 3 * 10**9

    def test_stable_diffusion_textual_inversion(self):
        pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
        pipe.load_textual_inversion("sd-concepts-library/low-poly-hd-logos-icons")

        a111_file = hf_hub_download("hf-internal-testing/text_inv_embedding_a1111_format", "winter_style.pt")
        a111_file_neg = hf_hub_download(
            "hf-internal-testing/text_inv_embedding_a1111_format", "winter_style_negative.pt"
        )
        pipe.load_textual_inversion(a111_file)
        pipe.load_textual_inversion(a111_file_neg)
        pipe.to("cuda")

        generator = torch.Generator(device="cpu").manual_seed(1)

        prompt = "An logo of a turtle in strong Style-Winter with <low-poly-hd-logos-icons>"
        neg_prompt = "Style-Winter-neg"

        image = pipe(prompt=prompt, negative_prompt=neg_prompt, generator=generator, output_type="np").images[0]
        expected_image = load_numpy(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/text_inv/winter_logo_style.npy"
        )

        max_diff = np.abs(expected_image - image).max()
        assert max_diff < 5e-2


@nightly
@require_torch_gpu
class StableDiffusionPipelineNightlyTests(unittest.TestCase):
    def tearDown(self):
        super().tearDown()
        gc.collect()
        torch.cuda.empty_cache()

    def get_inputs(self, device, generator_device="cpu", dtype=torch.float32, seed=0):
        generator = torch.Generator(device=generator_device).manual_seed(seed)
        latents = np.random.RandomState(seed).standard_normal((1, 4, 64, 64))
        latents = torch.from_numpy(latents).to(device=device, dtype=dtype)
        inputs = {
            "prompt": "a photograph of an astronaut riding a horse",
            "latents": latents,
            "generator": generator,
            "num_inference_steps": 50,
            "guidance_scale": 7.5,
            "output_type": "numpy",
        }
        return inputs

    def test_stable_diffusion_1_4_pndm(self):
        sd_pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4").to(torch_device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs(torch_device)
        image = sd_pipe(**inputs).images[0]

        expected_image = load_numpy(
            "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main"
            "/stable_diffusion_text2img/stable_diffusion_1_4_pndm.npy"
        )
        max_diff = np.abs(expected_image - image).max()
        assert max_diff < 1e-3

    def test_stable_diffusion_1_5_pndm(self):
        sd_pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to(torch_device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs(torch_device)
        image = sd_pipe(**inputs).images[0]

        expected_image = load_numpy(
            "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main"
            "/stable_diffusion_text2img/stable_diffusion_1_5_pndm.npy"
        )
        max_diff = np.abs(expected_image - image).max()
        assert max_diff < 1e-3

    def test_stable_diffusion_ddim(self):
        sd_pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4").to(torch_device)
        sd_pipe.scheduler = DDIMScheduler.from_config(sd_pipe.scheduler.config)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs(torch_device)
        image = sd_pipe(**inputs).images[0]

        expected_image = load_numpy(
            "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main"
            "/stable_diffusion_text2img/stable_diffusion_1_4_ddim.npy"
        )
        max_diff = np.abs(expected_image - image).max()
        assert max_diff < 1e-3

    def test_stable_diffusion_lms(self):
        sd_pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4").to(torch_device)
        sd_pipe.scheduler = LMSDiscreteScheduler.from_config(sd_pipe.scheduler.config)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs(torch_device)
        image = sd_pipe(**inputs).images[0]

        expected_image = load_numpy(
            "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main"
            "/stable_diffusion_text2img/stable_diffusion_1_4_lms.npy"
        )
        max_diff = np.abs(expected_image - image).max()
        assert max_diff < 1e-3

    def test_stable_diffusion_euler(self):
        sd_pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4").to(torch_device)
        sd_pipe.scheduler = EulerDiscreteScheduler.from_config(sd_pipe.scheduler.config)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs(torch_device)
        image = sd_pipe(**inputs).images[0]

        expected_image = load_numpy(
            "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main"
            "/stable_diffusion_text2img/stable_diffusion_1_4_euler.npy"
        )
        max_diff = np.abs(expected_image - image).max()
        assert max_diff < 1e-3

    def test_stable_diffusion_dpm(self):
        sd_pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4").to(torch_device)
        sd_pipe.scheduler = DPMSolverMultistepScheduler.from_config(sd_pipe.scheduler.config)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs(torch_device)
        inputs["num_inference_steps"] = 25
        image = sd_pipe(**inputs).images[0]

        expected_image = load_numpy(
            "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main"
            "/stable_diffusion_text2img/stable_diffusion_1_4_dpm_multi.npy"
        )
        max_diff = np.abs(expected_image - image).max()
        assert max_diff < 1e-3
