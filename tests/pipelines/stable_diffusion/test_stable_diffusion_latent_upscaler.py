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

import random
import unittest

import numpy as np
import torch
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer

from diffusers import (
    AutoencoderKL,
    EulerDiscreteScheduler,
    StableDiffusionLatentUpscalePipeline,
    UNet2DConditionModel,
)
from diffusers.utils.testing_utils import (
    enable_full_determinism,
    floats_tensor,
    skip_mps,
)

from ..pipeline_params import (
    IMAGE_TO_IMAGE_IMAGE_PARAMS,
    TEXT_GUIDED_IMAGE_VARIATION_BATCH_PARAMS,
    TEXT_GUIDED_IMAGE_VARIATION_PARAMS,
    TEXT_TO_IMAGE_CALLBACK_CFG_PARAMS,
)
from ..test_pipelines_common import (
    PipelineLatentTesterMixin,
    PipelineTesterMixin,
)


enable_full_determinism()


class StableDiffusionLatentUpscalerPipelineFastTests(
    PipelineLatentTesterMixin,
    PipelineTesterMixin,
    unittest.TestCase,
):
    pipeline_class = StableDiffusionLatentUpscalePipeline
    params = TEXT_GUIDED_IMAGE_VARIATION_PARAMS - {"height", "width"}
    required_optional_params = PipelineTesterMixin.required_optional_params - {"num_images_per_prompt"}
    batch_params = TEXT_GUIDED_IMAGE_VARIATION_BATCH_PARAMS
    image_params = IMAGE_TO_IMAGE_IMAGE_PARAMS
    image_latents_params = IMAGE_TO_IMAGE_IMAGE_PARAMS
    callback_cfg_params = TEXT_TO_IMAGE_CALLBACK_CFG_PARAMS

    def get_dummy_components(self):
        torch.manual_seed(0)
        unet = UNet2DConditionModel(
            act_fn="gelu",
            block_out_channels=(384, 768, 768),
            layers_per_block=1,
            time_cond_proj_dim=896,
            sample_size=None,
            in_channels=8,
            out_channels=5,
            attention_head_dim=64,
            down_block_types=("KDownBlock2D", "KCrossAttnDownBlock2D", "KCrossAttnDownBlock2D"),
            up_block_types=("KCrossAttnUpBlock2D", "KCrossAttnUpBlock2D", "KUpBlock2D"),
            conv_in_kernel=1,
            conv_out_kernel=1,
            mid_block_type=None,
            cross_attention_dim=768,
            norm_num_groups=None,
            resnet_time_scale_shift="scale_shift",
            time_embedding_type="fourier",
            timestep_post_act="gelu",
        )
        scheduler = EulerDiscreteScheduler()
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
            hidden_size=768,
            intermediate_size=3072,
            layer_norm_eps=1e-05,
            num_attention_heads=12,
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
        }
        return components

    def get_dummy_inputs(self, device, seed=0):
        image = floats_tensor((1, 3, 16, 16), rng=random.Random(seed)).to(device)
        image = image / 2 + 0.5
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
            "output_type": "pt",
        }
        return inputs

    def test_stable_diffusion_latent_upscaler_default_case(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()
        sd_pipe = StableDiffusionLatentUpscalePipeline(**components)
        sd_pipe = sd_pipe.to(device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        image = sd_pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 3, 32, 32)
        expected_slice = np.array([0.5124, 0.4909, 0.4620, 0.5250, 0.5045, 0.4327, 0.4345, 0.3971, 0.3820])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-3

    def test_stable_diffusion_latent_upscaler_negative_prompt(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()
        sd_pipe = StableDiffusionLatentUpscalePipeline(**components)
        sd_pipe = sd_pipe.to(device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        negative_prompt = "french fries"
        output = sd_pipe(**inputs, negative_prompt=negative_prompt)
        image = output.images
        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 3, 32, 32)
        expected_slice = np.array([0.5876, 0.6150, 0.5728, 0.6009, 0.5441, 0.4278, 0.5181, 0.5461, 0.4822])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-3

    def test_stable_diffusion_latent_upscaler_multiple_init_images(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()
        sd_pipe = StableDiffusionLatentUpscalePipeline(**components)
        sd_pipe = sd_pipe.to(device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        inputs["prompt"] = [inputs["prompt"]] * 2
        inputs["image"] = inputs["image"].repeat(2, 1, 1, 1)
        image = sd_pipe(**inputs).images
        image_slice = image[-1, -3:, -3:, -1]

        assert image.shape == (2, 3, 32, 32)
        expected_slice = np.array([0.3902, 0.4474, 0.4671, 0.5388, 0.5603, 0.6732, 0.3680, 0.4603, 0.4492])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-3

    @skip_mps
    def test_save_load_local(self):
        return super().test_save_load_local()

    @skip_mps
    def test_dict_tuple_outputs_equivalent(self):
        return super().test_dict_tuple_outputs_equivalent()

    @skip_mps
    def test_save_load_optional_components(self):
        return super().test_save_load_optional_components()

    @skip_mps
    def test_attention_slicing_forward_pass(self):
        return super().test_attention_slicing_forward_pass(expected_max_diff=5e-3)

    def test_inference_batch_single_identical(self):
        super().test_inference_batch_single_identical(expected_max_diff=3e-3)

    def test_float16_inference(self):
        super().test_float16_inference(expected_max_diff=5e-1)
