# coding=utf-8
# Copyright 2026 HuggingFace Inc.
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

import numpy as np
import pytest
import torch
from PIL import Image
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer

from diffusers import AutoencoderKL, DDIMScheduler, DDPMScheduler, StableDiffusionUpscalePipeline, UNet2DConditionModel

from ...testing_utils import (
    assert_tensors_close,
    backend_empty_cache,
    backend_max_memory_allocated,
    backend_reset_max_memory_allocated,
    backend_reset_peak_memory_stats,
    enable_full_determinism,
    floats_tensor,
    load_image,
    load_numpy,
    require_accelerator,
    require_torch_accelerator,
    slow,
    torch_device,
)
from ..pipeline_params import TEXT_GUIDED_IMAGE_VARIATION_BATCH_PARAMS
from ..testing_utils import (
    BasePipelineTesterConfig,
    MemoryTesterMixin,
    PipelineTesterMixin,
)


enable_full_determinism()


class StableDiffusionUpscalePipelineTesterConfig(BasePipelineTesterConfig):
    pipeline_class = StableDiffusionUpscalePipeline
    # `TEXT_GUIDED_IMAGE_VARIATION_PARAMS` without `height` / `width`: the output resolution follows the image.
    required_input_params_in_call_signature = frozenset(
        ["prompt", "image", "guidance_scale", "negative_prompt", "prompt_embeds", "negative_prompt_embeds"]
    )
    batch_input_params = TEXT_GUIDED_IMAGE_VARIATION_BATCH_PARAMS
    # The 64x64 low-resolution input is upscaled 4x.
    output_shape = (3, 256, 256)

    @property
    def dummy_image(self):
        batch_size = 1
        num_channels = 3
        sizes = (32, 32)

        image = floats_tensor((batch_size, num_channels) + sizes, rng=random.Random(0))
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

    @property
    def dummy_low_res_image(self):
        image = self.dummy_image.permute(0, 2, 3, 1)[0]
        return Image.fromarray(np.uint8(image)).convert("RGB").resize((64, 64))

    def get_dummy_components(self):
        return {
            "unet": self.dummy_cond_unet_upscale,
            "low_res_scheduler": DDPMScheduler(),
            "scheduler": DDIMScheduler(prediction_type="v_prediction"),
            "vae": self.dummy_vae,
            "text_encoder": self.dummy_text_encoder,
            "tokenizer": CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip"),
            "safety_checker": None,
            "feature_extractor": None,
            "watermarker": None,
            "max_noise_level": 350,
        }

    def get_dummy_inputs(self):
        return {
            "prompt": "A painting of a squirrel eating a burger",
            "image": self.dummy_low_res_image,
            "generator": self.get_generator(0),
            "guidance_scale": 6.0,
            "noise_level": 20,
            "num_inference_steps": 2,
            # Request torch outputs so tests compare torch tensors directly (see `BasePipelineTesterConfig`).
            # Note `"pt"` images are `(batch, channels, height, width)`, unlike `"np"` (`(batch, h, w, c)`).
            "output_type": "pt",
        }


class TestStableDiffusionUpscalePipeline(StableDiffusionUpscalePipelineTesterConfig, PipelineTesterMixin):
    def test_stable_diffusion_upscale(self):
        # Run on CPU: the expected slice below is CPU-specific.
        sd_pipe = self.get_pipeline()

        image = sd_pipe(**self.get_dummy_inputs()).images
        image_slice = image[0, -1, -3:, -3:]

        assert image.shape == (1, *self.output_shape)
        # fmt: off
        expected_slice = torch.tensor([0.2631, 0.4038, 0.4338, 0.4254, 0.5002, 0.4831, 0.5073, 0.5619, 0.5597])
        # fmt: on
        assert_tensors_close(image_slice.flatten(), expected_slice, atol=1e-2)

    def test_stable_diffusion_upscale_batch(self):
        sd_pipe = self.get_pipeline()

        inputs = self.get_dummy_inputs()
        inputs["prompt"] = 2 * [inputs["prompt"]]
        inputs["image"] = 2 * [inputs["image"]]
        del inputs["generator"]
        image = sd_pipe(**inputs).images
        assert image.shape[0] == 2

        inputs = self.get_dummy_inputs()
        inputs["prompt"] = [inputs["prompt"]]
        image = sd_pipe(**inputs, num_images_per_prompt=2).images
        assert image.shape[0] == 2

    def test_stable_diffusion_upscale_prompt_embeds(self):
        # `encode_prompt` is called with `torch_device` below, so the pipeline has to live there too.
        sd_pipe = self.get_pipeline().to(torch_device)

        image = sd_pipe(**self.get_dummy_inputs()).images

        inputs = self.get_dummy_inputs()
        prompt = inputs.pop("prompt")
        prompt_embeds, negative_prompt_embeds = sd_pipe.encode_prompt(prompt, torch_device, 1, False)
        if negative_prompt_embeds is not None:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        inputs["image"] = [inputs["image"]]
        image_from_prompt_embeds = sd_pipe(prompt_embeds=prompt_embeds, **inputs, return_dict=False)[0]

        assert_tensors_close(
            image_from_prompt_embeds[0, -1, -3:, -3:],
            image[0, -1, -3:, -3:],
            atol=1e-2,
            msg="Passing `prompt_embeds` changed the output.",
        )

    def test_encode_prompt_works_in_isolation(self):
        extra_required_param_value_dict = {
            "device": torch.device(torch_device).type,
            "do_classifier_free_guidance": self.get_dummy_inputs().get("guidance_scale", 1.0) > 1.0,
        }
        return super().test_encode_prompt_works_in_isolation(extra_required_param_value_dict)

    @require_accelerator
    def test_stable_diffusion_upscale_fp16(self):
        """Test that stable diffusion upscale works with fp16"""
        components = self.get_dummy_components()
        # put models in fp16, except vae as it overflows in fp16
        components["unet"] = components["unet"].half()
        components["text_encoder"] = components["text_encoder"].half()
        sd_pipe = self.get_pipeline(**components).to(torch_device)

        inputs = self.get_dummy_inputs()
        del inputs["guidance_scale"]
        del inputs["noise_level"]
        image = sd_pipe(**inputs).images

        assert image.shape == (1, *self.output_shape)

    def test_stable_diffusion_upscale_from_save_pretrained(self, tmp_path):
        sd_pipe = self.get_pipeline()
        sd_pipe.save_pretrained(tmp_path)
        sd_pipe_loaded = StableDiffusionUpscalePipeline.from_pretrained(tmp_path)
        sd_pipe_loaded.set_progress_bar_config(disable=None)

        image_slices = []
        for pipe in [sd_pipe, sd_pipe_loaded]:
            image = pipe(**self.get_dummy_inputs()).images
            image_slices.append(image[0, -1, -3:, -3:].flatten())

        assert_tensors_close(image_slices[1], image_slices[0], atol=1e-3)


class TestStableDiffusionUpscalePipelineMemory(StableDiffusionUpscalePipelineTesterConfig, MemoryTesterMixin):
    """Memory optimization tests (CPU offload, group offload, layerwise casting) for the SD upscale pipeline."""


@slow
@require_torch_accelerator
class TestStableDiffusionUpscalePipelineIntegration:
    @pytest.fixture(autouse=True)
    def cleanup(self):
        # clean up the VRAM before and after each test
        gc.collect()
        backend_empty_cache(torch_device)
        yield
        gc.collect()
        backend_empty_cache(torch_device)

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
        backend_empty_cache(torch_device)
        backend_reset_max_memory_allocated(torch_device)
        backend_reset_peak_memory_stats(torch_device)

        image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
            "/sd2-upscale/low_res_cat.png"
        )

        model_id = "stabilityai/stable-diffusion-x4-upscaler"
        pipe = StableDiffusionUpscalePipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
        )
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing(1)
        pipe.enable_sequential_cpu_offload(device=torch_device)

        prompt = "a cat sitting on a park bench"

        generator = torch.manual_seed(0)
        _ = pipe(
            prompt=prompt,
            image=image,
            generator=generator,
            num_inference_steps=5,
            output_type="np",
        )

        mem_bytes = backend_max_memory_allocated(torch_device)
        # make sure that less than 2.9 GB is allocated
        assert mem_bytes < 2.9 * 10**9
