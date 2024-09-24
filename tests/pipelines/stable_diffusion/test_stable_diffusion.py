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
import tempfile
import time
import traceback
import unittest

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import (
    CLIPTextConfig,
    CLIPTextModel,
    CLIPTokenizer,
)

from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LCMScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
    logging,
)
from diffusers.utils.testing_utils import (
    CaptureLogger,
    enable_full_determinism,
    is_torch_compile,
    load_image,
    load_numpy,
    nightly,
    numpy_cosine_similarity_distance,
    require_accelerate_version_greater,
    require_torch_2,
    require_torch_gpu,
    require_torch_multi_gpu,
    run_test_in_subprocess,
    skip_mps,
    slow,
    torch_device,
)

from ..pipeline_params import (
    TEXT_TO_IMAGE_BATCH_PARAMS,
    TEXT_TO_IMAGE_CALLBACK_CFG_PARAMS,
    TEXT_TO_IMAGE_IMAGE_PARAMS,
    TEXT_TO_IMAGE_PARAMS,
)
from ..test_pipelines_common import (
    IPAdapterTesterMixin,
    PipelineKarrasSchedulerTesterMixin,
    PipelineLatentTesterMixin,
    PipelineTesterMixin,
)


enable_full_determinism()


# Will be run via run_test_in_subprocess
def _test_stable_diffusion_compile(in_queue, out_queue, timeout):
    error = None
    try:
        inputs = in_queue.get(timeout=timeout)
        torch_device = inputs.pop("torch_device")
        seed = inputs.pop("seed")
        inputs["generator"] = torch.Generator(device=torch_device).manual_seed(seed)

        sd_pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", safety_checker=None)
        sd_pipe.scheduler = DDIMScheduler.from_config(sd_pipe.scheduler.config)
        sd_pipe = sd_pipe.to(torch_device)

        sd_pipe.unet.to(memory_format=torch.channels_last)
        sd_pipe.unet = torch.compile(sd_pipe.unet, mode="reduce-overhead", fullgraph=True)

        sd_pipe.set_progress_bar_config(disable=None)

        image = sd_pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1].flatten()

        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array([0.38019, 0.28647, 0.27321, 0.40377, 0.38290, 0.35446, 0.39218, 0.38165, 0.42239])

        assert np.abs(image_slice - expected_slice).max() < 5e-3
    except Exception:
        error = f"{traceback.format_exc()}"

    results = {"error": error}
    out_queue.put(results, timeout=timeout)
    out_queue.join()


class StableDiffusionPipelineFastTests(
    IPAdapterTesterMixin,
    PipelineLatentTesterMixin,
    PipelineKarrasSchedulerTesterMixin,
    PipelineTesterMixin,
    unittest.TestCase,
):
    pipeline_class = StableDiffusionPipeline
    params = TEXT_TO_IMAGE_PARAMS
    batch_params = TEXT_TO_IMAGE_BATCH_PARAMS
    image_params = TEXT_TO_IMAGE_IMAGE_PARAMS
    image_latents_params = TEXT_TO_IMAGE_IMAGE_PARAMS
    callback_cfg_params = TEXT_TO_IMAGE_CALLBACK_CFG_PARAMS

    def get_dummy_components(self, time_cond_proj_dim=None):
        cross_attention_dim = 8

        torch.manual_seed(0)
        unet = UNet2DConditionModel(
            block_out_channels=(4, 8),
            layers_per_block=1,
            sample_size=32,
            time_cond_proj_dim=time_cond_proj_dim,
            in_channels=4,
            out_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
            cross_attention_dim=cross_attention_dim,
            norm_num_groups=2,
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
            block_out_channels=[4, 8],
            in_channels=3,
            out_channels=3,
            down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
            up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"],
            latent_channels=4,
            norm_num_groups=2,
        )
        torch.manual_seed(0)
        text_encoder_config = CLIPTextConfig(
            bos_token_id=0,
            eos_token_id=2,
            hidden_size=cross_attention_dim,
            intermediate_size=16,
            layer_norm_eps=1e-05,
            num_attention_heads=2,
            num_hidden_layers=2,
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
        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device=device).manual_seed(seed)
        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            "output_type": "np",
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
        expected_slice = np.array([0.1763, 0.4776, 0.4986, 0.2566, 0.3802, 0.4596, 0.5363, 0.3277, 0.3949])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

    def test_stable_diffusion_lcm(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator

        components = self.get_dummy_components(time_cond_proj_dim=256)
        sd_pipe = StableDiffusionPipeline(**components)
        sd_pipe.scheduler = LCMScheduler.from_config(sd_pipe.scheduler.config)
        sd_pipe = sd_pipe.to(torch_device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        output = sd_pipe(**inputs)
        image = output.images

        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 64, 64, 3)
        expected_slice = np.array([0.2368, 0.4900, 0.5019, 0.2723, 0.4473, 0.4578, 0.4551, 0.3532, 0.4133])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

    def test_stable_diffusion_lcm_custom_timesteps(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator

        components = self.get_dummy_components(time_cond_proj_dim=256)
        sd_pipe = StableDiffusionPipeline(**components)
        sd_pipe.scheduler = LCMScheduler.from_config(sd_pipe.scheduler.config)
        sd_pipe = sd_pipe.to(torch_device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        del inputs["num_inference_steps"]
        inputs["timesteps"] = [999, 499]
        output = sd_pipe(**inputs)
        image = output.images

        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 64, 64, 3)
        expected_slice = np.array([0.2368, 0.4900, 0.5019, 0.2723, 0.4473, 0.4578, 0.4551, 0.3532, 0.4133])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

    def test_stable_diffusion_ays(self):
        from diffusers.schedulers import AysSchedules

        timestep_schedule = AysSchedules["StableDiffusionTimesteps"]
        sigma_schedule = AysSchedules["StableDiffusionSigmas"]

        device = "cpu"  # ensure determinism for the device-dependent torch.Generator

        components = self.get_dummy_components(time_cond_proj_dim=256)
        sd_pipe = StableDiffusionPipeline(**components)
        sd_pipe.scheduler = EulerDiscreteScheduler.from_config(sd_pipe.scheduler.config)
        sd_pipe = sd_pipe.to(torch_device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        inputs["num_inference_steps"] = 10
        output = sd_pipe(**inputs).images

        inputs = self.get_dummy_inputs(device)
        inputs["num_inference_steps"] = None
        inputs["timesteps"] = timestep_schedule
        output_ts = sd_pipe(**inputs).images

        inputs = self.get_dummy_inputs(device)
        inputs["num_inference_steps"] = None
        inputs["sigmas"] = sigma_schedule
        output_sigmas = sd_pipe(**inputs).images

        assert (
            np.abs(output_sigmas.flatten() - output_ts.flatten()).max() < 1e-3
        ), "ays timesteps and ays sigmas should have the same outputs"
        assert (
            np.abs(output.flatten() - output_ts.flatten()).max() > 1e-3
        ), "use ays timesteps should have different outputs"
        assert (
            np.abs(output.flatten() - output_sigmas.flatten()).max() > 1e-3
        ), "use ays sigmas should have different outputs"

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

    def test_stable_diffusion_prompt_embeds_no_text_encoder_or_tokenizer(self):
        components = self.get_dummy_components()
        sd_pipe = StableDiffusionPipeline(**components)
        sd_pipe = sd_pipe.to(torch_device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(torch_device)
        inputs["negative_prompt"] = "this is a negative prompt"

        # forward
        output = sd_pipe(**inputs)
        image_slice_1 = output.images[0, -3:, -3:, -1]

        inputs = self.get_dummy_inputs(torch_device)
        prompt = inputs.pop("prompt")
        negative_prompt = "this is a negative prompt"

        prompt_embeds, negative_prompt_embeds = sd_pipe.encode_prompt(
            prompt,
            torch_device,
            1,
            True,
            negative_prompt=negative_prompt,
            prompt_embeds=None,
            negative_prompt_embeds=None,
        )

        inputs["prompt_embeds"] = prompt_embeds
        inputs["negative_prompt_embeds"] = negative_prompt_embeds

        sd_pipe.text_encoder = None
        sd_pipe.tokenizer = None

        # forward
        output = sd_pipe(**inputs)
        image_slice_2 = output.images[0, -3:, -3:, -1]

        assert np.abs(image_slice_1.flatten() - image_slice_2.flatten()).max() < 1e-4

    def test_stable_diffusion_prompt_embeds_with_plain_negative_prompt_list(self):
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
        inputs["negative_prompt"] = negative_prompt
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
        expected_slice = np.array([0.4720, 0.5426, 0.5160, 0.3961, 0.4696, 0.4296, 0.5738, 0.5888, 0.5481])

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
        expected_slice = np.array([0.1941, 0.4748, 0.4880, 0.2222, 0.4221, 0.4545, 0.5604, 0.3488, 0.3902])

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
        expected_slice = np.array([0.2681, 0.4785, 0.4857, 0.2426, 0.4473, 0.4481, 0.5610, 0.3676, 0.3855])

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
        expected_slice = np.array([0.2682, 0.4782, 0.4855, 0.2424, 0.4472, 0.4479, 0.5612, 0.3676, 0.3854])

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
        expected_slice = np.array([0.2681, 0.4785, 0.4857, 0.2426, 0.4473, 0.4481, 0.5610, 0.3676, 0.3855])

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
        expected_slice = np.array([0.1907, 0.4709, 0.4858, 0.2224, 0.4223, 0.4539, 0.5606, 0.3489, 0.3900])

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
        logger.setLevel(logging.WARNING)

        prompt = 100 * "@"
        with CaptureLogger(logger) as cap_logger:
            negative_text_embeddings, text_embeddings = sd_pipe.encode_prompt(
                prompt, torch_device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
            )
            if negative_text_embeddings is not None:
                text_embeddings = torch.cat([negative_text_embeddings, text_embeddings])

        # 100 - 77 + 1 (BOS token) + 1 (EOS token) = 25
        assert cap_logger.out.count("@") == 25

        negative_prompt = "Hello"
        with CaptureLogger(logger) as cap_logger_2:
            negative_text_embeddings_2, text_embeddings_2 = sd_pipe.encode_prompt(
                prompt, torch_device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
            )
            if negative_text_embeddings_2 is not None:
                text_embeddings_2 = torch.cat([negative_text_embeddings_2, text_embeddings_2])

        assert cap_logger.out == cap_logger_2.out

        prompt = 25 * "@"
        with CaptureLogger(logger) as cap_logger_3:
            negative_text_embeddings_3, text_embeddings_3 = sd_pipe.encode_prompt(
                prompt, torch_device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
            )
            if negative_text_embeddings_3 is not None:
                text_embeddings_3 = torch.cat([negative_text_embeddings_3, text_embeddings_3])

        assert text_embeddings_3.shape == text_embeddings_2.shape == text_embeddings.shape
        assert text_embeddings.shape[1] == 77
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

    def test_attention_slicing_forward_pass(self):
        super().test_attention_slicing_forward_pass(expected_max_diff=3e-3)

    def test_inference_batch_single_identical(self):
        super().test_inference_batch_single_identical(expected_max_diff=3e-3)

    # MPS currently doesn't support ComplexFloats, which are required for freeU - see https://github.com/huggingface/diffusers/issues/7569.
    @skip_mps
    def test_freeu_enabled(self):
        components = self.get_dummy_components()
        sd_pipe = StableDiffusionPipeline(**components)
        sd_pipe = sd_pipe.to(torch_device)
        sd_pipe.set_progress_bar_config(disable=None)

        prompt = "hey"
        output = sd_pipe(prompt, num_inference_steps=1, output_type="np", generator=torch.manual_seed(0)).images

        sd_pipe.enable_freeu(s1=0.9, s2=0.2, b1=1.2, b2=1.4)
        output_freeu = sd_pipe(prompt, num_inference_steps=1, output_type="np", generator=torch.manual_seed(0)).images

        assert not np.allclose(
            output[0, -3:, -3:, -1], output_freeu[0, -3:, -3:, -1]
        ), "Enabling of FreeU should lead to different results."

    def test_freeu_disabled(self):
        components = self.get_dummy_components()
        sd_pipe = StableDiffusionPipeline(**components)
        sd_pipe = sd_pipe.to(torch_device)
        sd_pipe.set_progress_bar_config(disable=None)

        prompt = "hey"
        output = sd_pipe(prompt, num_inference_steps=1, output_type="np", generator=torch.manual_seed(0)).images

        sd_pipe.enable_freeu(s1=0.9, s2=0.2, b1=1.2, b2=1.4)
        sd_pipe.disable_freeu()

        freeu_keys = {"s1", "s2", "b1", "b2"}
        for upsample_block in sd_pipe.unet.up_blocks:
            for key in freeu_keys:
                assert getattr(upsample_block, key) is None, f"Disabling of FreeU should have set {key} to None."

        output_no_freeu = sd_pipe(
            prompt, num_inference_steps=1, output_type="np", generator=torch.manual_seed(0)
        ).images

        assert np.allclose(
            output[0, -3:, -3:, -1], output_no_freeu[0, -3:, -3:, -1]
        ), "Disabling of FreeU should lead to results similar to the default pipeline results."

    def test_fused_qkv_projections(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()
        sd_pipe = StableDiffusionPipeline(**components)
        sd_pipe = sd_pipe.to(device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        image = sd_pipe(**inputs).images
        original_image_slice = image[0, -3:, -3:, -1]

        sd_pipe.fuse_qkv_projections()
        inputs = self.get_dummy_inputs(device)
        image = sd_pipe(**inputs).images
        image_slice_fused = image[0, -3:, -3:, -1]

        sd_pipe.unfuse_qkv_projections()
        inputs = self.get_dummy_inputs(device)
        image = sd_pipe(**inputs).images
        image_slice_disabled = image[0, -3:, -3:, -1]

        assert np.allclose(
            original_image_slice, image_slice_fused, atol=1e-2, rtol=1e-2
        ), "Fusion of QKV projections shouldn't affect the outputs."
        assert np.allclose(
            image_slice_fused, image_slice_disabled, atol=1e-2, rtol=1e-2
        ), "Outputs, with QKV projection fusion enabled, shouldn't change when fused QKV projections are disabled."
        assert np.allclose(
            original_image_slice, image_slice_disabled, atol=1e-2, rtol=1e-2
        ), "Original outputs should match when fused QKV projections are disabled."

    def test_pipeline_interrupt(self):
        components = self.get_dummy_components()
        sd_pipe = StableDiffusionPipeline(**components)
        sd_pipe = sd_pipe.to(torch_device)
        sd_pipe.set_progress_bar_config(disable=None)

        prompt = "hey"
        num_inference_steps = 3

        # store intermediate latents from the generation process
        class PipelineState:
            def __init__(self):
                self.state = []

            def apply(self, pipe, i, t, callback_kwargs):
                self.state.append(callback_kwargs["latents"])
                return callback_kwargs

        pipe_state = PipelineState()
        sd_pipe(
            prompt,
            num_inference_steps=num_inference_steps,
            output_type="np",
            generator=torch.Generator("cpu").manual_seed(0),
            callback_on_step_end=pipe_state.apply,
        ).images

        # interrupt generation at step index
        interrupt_step_idx = 1

        def callback_on_step_end(pipe, i, t, callback_kwargs):
            if i == interrupt_step_idx:
                pipe._interrupt = True

            return callback_kwargs

        output_interrupted = sd_pipe(
            prompt,
            num_inference_steps=num_inference_steps,
            output_type="latent",
            generator=torch.Generator("cpu").manual_seed(0),
            callback_on_step_end=callback_on_step_end,
        ).images

        # fetch intermediate latents at the interrupted step
        # from the completed generation process
        intermediate_latent = pipe_state.state[interrupt_step_idx]

        # compare the intermediate latent to the output of the interrupted process
        # they should be the same
        assert torch.allclose(intermediate_latent, output_interrupted, atol=1e-4)


@slow
@require_torch_gpu
class StableDiffusionPipelineSlowTests(unittest.TestCase):
    def setUp(self):
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
            "output_type": "np",
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
        expected_slice = np.array([0.4363, 0.4355, 0.3667, 0.4066, 0.3970, 0.3866, 0.4394, 0.4356, 0.4059])
        assert np.abs(image_slice - expected_slice).max() < 3e-3

    def test_stable_diffusion_v1_4_with_freeu(self):
        sd_pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4").to(torch_device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs(torch_device)
        inputs["num_inference_steps"] = 25

        sd_pipe.enable_freeu(s1=0.9, s2=0.2, b1=1.2, b2=1.4)
        image = sd_pipe(**inputs).images
        image = image[0, -3:, -3:, -1].flatten()
        expected_image = [0.0721, 0.0588, 0.0268, 0.0384, 0.0636, 0.0, 0.0429, 0.0344, 0.0309]
        max_diff = np.abs(expected_image - image).max()
        assert max_diff < 1e-3

    def test_stable_diffusion_1_4_pndm(self):
        sd_pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
        sd_pipe = sd_pipe.to(torch_device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs(torch_device)
        image = sd_pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1].flatten()

        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array([0.5740, 0.4784, 0.3162, 0.6358, 0.5831, 0.5505, 0.5082, 0.5631, 0.5575])
        assert np.abs(image_slice - expected_slice).max() < 3e-3

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
        assert np.abs(image_slice - expected_slice).max() < 3e-3

    def test_stable_diffusion_dpm(self):
        sd_pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", safety_checker=None)
        sd_pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            sd_pipe.scheduler.config,
            final_sigmas_type="sigma_min",
        )
        sd_pipe = sd_pipe.to(torch_device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs(torch_device)
        image = sd_pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1].flatten()

        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array([0.03503, 0.03494, 0.01087, 0.03128, 0.02552, 0.00803, 0.00742, 0.00372, 0.00000])
        assert np.abs(image_slice - expected_slice).max() < 3e-3

    def test_stable_diffusion_attention_slicing(self):
        torch.cuda.reset_peak_memory_stats()
        pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
        pipe.unet.set_default_attn_processor()
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
        pipe.unet.set_default_attn_processor()
        inputs = self.get_inputs(torch_device, dtype=torch.float16)
        image = pipe(**inputs).images

        # make sure that more than 3.75 GB is allocated
        mem_bytes = torch.cuda.max_memory_allocated()
        assert mem_bytes > 3.75 * 10**9
        max_diff = numpy_cosine_similarity_distance(image_sliced.flatten(), image.flatten())
        assert max_diff < 1e-3

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
        max_diff = numpy_cosine_similarity_distance(image_sliced.flatten(), image.flatten())
        assert max_diff < 1e-2

    def test_stable_diffusion_vae_tiling(self):
        torch.cuda.reset_peak_memory_stats()
        model_id = "CompVis/stable-diffusion-v1-4"
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id, variant="fp16", torch_dtype=torch.float16, safety_checker=None
        )
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
            output_type="np",
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
            output_type="np",
        )
        image = output.images

        assert mem_bytes < 1e10
        max_diff = numpy_cosine_similarity_distance(image_chunked.flatten(), image.flatten())
        assert max_diff < 1e-2

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

        def callback_fn(step: int, timestep: int, latents: torch.Tensor) -> None:
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

        images = outputs.images
        offloaded_images = outputs_offloaded.images

        max_diff = numpy_cosine_similarity_distance(images.flatten(), offloaded_images.flatten())
        assert max_diff < 1e-3
        assert mem_bytes_offloaded < mem_bytes
        assert mem_bytes_offloaded < 3.5 * 10**9
        for module in pipe.text_encoder, pipe.unet, pipe.vae:
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
        assert max_diff < 8e-1

    def test_stable_diffusion_textual_inversion_with_model_cpu_offload(self):
        pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
        pipe.enable_model_cpu_offload()
        pipe.load_textual_inversion("sd-concepts-library/low-poly-hd-logos-icons")

        a111_file = hf_hub_download("hf-internal-testing/text_inv_embedding_a1111_format", "winter_style.pt")
        a111_file_neg = hf_hub_download(
            "hf-internal-testing/text_inv_embedding_a1111_format", "winter_style_negative.pt"
        )
        pipe.load_textual_inversion(a111_file)
        pipe.load_textual_inversion(a111_file_neg)

        generator = torch.Generator(device="cpu").manual_seed(1)

        prompt = "An logo of a turtle in strong Style-Winter with <low-poly-hd-logos-icons>"
        neg_prompt = "Style-Winter-neg"

        image = pipe(prompt=prompt, negative_prompt=neg_prompt, generator=generator, output_type="np").images[0]
        expected_image = load_numpy(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/text_inv/winter_logo_style.npy"
        )

        max_diff = np.abs(expected_image - image).max()
        assert max_diff < 8e-1

    def test_stable_diffusion_textual_inversion_with_sequential_cpu_offload(self):
        pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
        pipe.enable_sequential_cpu_offload()
        pipe.load_textual_inversion("sd-concepts-library/low-poly-hd-logos-icons")

        a111_file = hf_hub_download("hf-internal-testing/text_inv_embedding_a1111_format", "winter_style.pt")
        a111_file_neg = hf_hub_download(
            "hf-internal-testing/text_inv_embedding_a1111_format", "winter_style_negative.pt"
        )
        pipe.load_textual_inversion(a111_file)
        pipe.load_textual_inversion(a111_file_neg)

        generator = torch.Generator(device="cpu").manual_seed(1)

        prompt = "An logo of a turtle in strong Style-Winter with <low-poly-hd-logos-icons>"
        neg_prompt = "Style-Winter-neg"

        image = pipe(prompt=prompt, negative_prompt=neg_prompt, generator=generator, output_type="np").images[0]
        expected_image = load_numpy(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/text_inv/winter_logo_style.npy"
        )

        max_diff = np.abs(expected_image - image).max()
        assert max_diff < 8e-1

    @is_torch_compile
    @require_torch_2
    def test_stable_diffusion_compile(self):
        seed = 0
        inputs = self.get_inputs(torch_device, seed=seed)
        # Can't pickle a Generator object
        del inputs["generator"]
        inputs["torch_device"] = torch_device
        inputs["seed"] = seed
        run_test_in_subprocess(test_case=self, target_func=_test_stable_diffusion_compile, inputs=inputs)

    def test_stable_diffusion_lcm(self):
        unet = UNet2DConditionModel.from_pretrained("SimianLuo/LCM_Dreamshaper_v7", subfolder="unet")
        sd_pipe = StableDiffusionPipeline.from_pretrained("Lykon/dreamshaper-7", unet=unet).to(torch_device)
        sd_pipe.scheduler = LCMScheduler.from_config(sd_pipe.scheduler.config)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs(torch_device)
        inputs["num_inference_steps"] = 6
        inputs["output_type"] = "pil"

        image = sd_pipe(**inputs).images[0]

        expected_image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/lcm_full/stable_diffusion_lcm.png"
        )

        image = sd_pipe.image_processor.pil_to_numpy(image)
        expected_image = sd_pipe.image_processor.pil_to_numpy(expected_image)

        max_diff = numpy_cosine_similarity_distance(image.flatten(), expected_image.flatten())

        assert max_diff < 1e-2


@slow
@require_torch_gpu
class StableDiffusionPipelineCkptTests(unittest.TestCase):
    def setUp(self):
        super().setUp()
        gc.collect()
        torch.cuda.empty_cache()

    def tearDown(self):
        super().tearDown()
        gc.collect()
        torch.cuda.empty_cache()

    def test_download_from_hub(self):
        ckpt_paths = [
            "https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/blob/main/v1-5-pruned-emaonly.safetensors",
            "https://huggingface.co/WarriorMama777/OrangeMixs/blob/main/Models/AbyssOrangeMix/AbyssOrangeMix.safetensors",
        ]

        for ckpt_path in ckpt_paths:
            pipe = StableDiffusionPipeline.from_single_file(ckpt_path, torch_dtype=torch.float16)
            pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
            pipe.to("cuda")

        image_out = pipe("test", num_inference_steps=1, output_type="np").images[0]

        assert image_out.shape == (512, 512, 3)

    def test_download_local(self):
        ckpt_filename = hf_hub_download(
            "stable-diffusion-v1-5/stable-diffusion-v1-5", filename="v1-5-pruned-emaonly.safetensors"
        )
        config_filename = hf_hub_download("stable-diffusion-v1-5/stable-diffusion-v1-5", filename="v1-inference.yaml")

        pipe = StableDiffusionPipeline.from_single_file(
            ckpt_filename, config_files={"v1": config_filename}, torch_dtype=torch.float16
        )
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.to("cuda")

        image_out = pipe("test", num_inference_steps=1, output_type="np").images[0]

        assert image_out.shape == (512, 512, 3)


@nightly
@require_torch_gpu
class StableDiffusionPipelineNightlyTests(unittest.TestCase):
    def setUp(self):
        super().setUp()
        gc.collect()
        torch.cuda.empty_cache()

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
            "output_type": "np",
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
        sd_pipe = StableDiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5").to(
            torch_device
        )
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
        assert max_diff < 3e-3

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


# (sayakpaul): This test suite was run in the DGX with two GPUs (1, 2).
@slow
@require_torch_multi_gpu
@require_accelerate_version_greater("0.27.0")
class StableDiffusionPipelineDeviceMapTests(unittest.TestCase):
    def tearDown(self):
        super().tearDown()
        gc.collect()
        torch.cuda.empty_cache()

    def get_inputs(self, generator_device="cpu", seed=0):
        generator = torch.Generator(device=generator_device).manual_seed(seed)
        inputs = {
            "prompt": "a photograph of an astronaut riding a horse",
            "generator": generator,
            "num_inference_steps": 50,
            "guidance_scale": 7.5,
            "output_type": "np",
        }
        return inputs

    def get_pipeline_output_without_device_map(self):
        sd_pipe = StableDiffusionPipeline.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16
        ).to(torch_device)
        sd_pipe.set_progress_bar_config(disable=True)
        inputs = self.get_inputs()
        no_device_map_image = sd_pipe(**inputs).images

        del sd_pipe

        return no_device_map_image

    def test_forward_pass_balanced_device_map(self):
        no_device_map_image = self.get_pipeline_output_without_device_map()

        sd_pipe_with_device_map = StableDiffusionPipeline.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5", device_map="balanced", torch_dtype=torch.float16
        )
        sd_pipe_with_device_map.set_progress_bar_config(disable=True)
        inputs = self.get_inputs()
        device_map_image = sd_pipe_with_device_map(**inputs).images

        max_diff = np.abs(device_map_image - no_device_map_image).max()
        assert max_diff < 1e-3

    def test_components_put_in_right_devices(self):
        sd_pipe_with_device_map = StableDiffusionPipeline.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5", device_map="balanced", torch_dtype=torch.float16
        )

        assert len(set(sd_pipe_with_device_map.hf_device_map.values())) >= 2

    def test_max_memory(self):
        no_device_map_image = self.get_pipeline_output_without_device_map()

        sd_pipe_with_device_map = StableDiffusionPipeline.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5",
            device_map="balanced",
            max_memory={0: "1GB", 1: "1GB"},
            torch_dtype=torch.float16,
        )
        sd_pipe_with_device_map.set_progress_bar_config(disable=True)
        inputs = self.get_inputs()
        device_map_image = sd_pipe_with_device_map(**inputs).images

        max_diff = np.abs(device_map_image - no_device_map_image).max()
        assert max_diff < 1e-3

    def test_reset_device_map(self):
        sd_pipe_with_device_map = StableDiffusionPipeline.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5", device_map="balanced", torch_dtype=torch.float16
        )
        sd_pipe_with_device_map.reset_device_map()

        assert sd_pipe_with_device_map.hf_device_map is None

        for name, component in sd_pipe_with_device_map.components.items():
            if isinstance(component, torch.nn.Module):
                assert component.device.type == "cpu"

    def test_reset_device_map_to(self):
        sd_pipe_with_device_map = StableDiffusionPipeline.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5", device_map="balanced", torch_dtype=torch.float16
        )
        sd_pipe_with_device_map.reset_device_map()

        assert sd_pipe_with_device_map.hf_device_map is None

        # Make sure `to()` can be used and the pipeline can be called.
        pipe = sd_pipe_with_device_map.to("cuda")
        _ = pipe("hello", num_inference_steps=2)

    def test_reset_device_map_enable_model_cpu_offload(self):
        sd_pipe_with_device_map = StableDiffusionPipeline.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5", device_map="balanced", torch_dtype=torch.float16
        )
        sd_pipe_with_device_map.reset_device_map()

        assert sd_pipe_with_device_map.hf_device_map is None

        # Make sure `enable_model_cpu_offload()` can be used and the pipeline can be called.
        sd_pipe_with_device_map.enable_model_cpu_offload()
        _ = sd_pipe_with_device_map("hello", num_inference_steps=2)

    def test_reset_device_map_enable_sequential_cpu_offload(self):
        sd_pipe_with_device_map = StableDiffusionPipeline.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5", device_map="balanced", torch_dtype=torch.float16
        )
        sd_pipe_with_device_map.reset_device_map()

        assert sd_pipe_with_device_map.hf_device_map is None

        # Make sure `enable_sequential_cpu_offload()` can be used and the pipeline can be called.
        sd_pipe_with_device_map.enable_sequential_cpu_offload()
        _ = sd_pipe_with_device_map("hello", num_inference_steps=2)
