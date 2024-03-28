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
import unittest

import torch

from diffusers import StableCascadeUNet
from diffusers.utils import logging
from diffusers.utils.testing_utils import (
    enable_full_determinism,
    numpy_cosine_similarity_distance,
    require_torch_gpu,
    slow,
)
from diffusers.utils.torch_utils import randn_tensor


logger = logging.get_logger(__name__)

enable_full_determinism()


@slow
class StableCascadeUNetModelSlowTests(unittest.TestCase):
    def tearDown(self) -> None:
        super().tearDown()
        gc.collect()
        torch.cuda.empty_cache()

    def test_stable_cascade_unet_prior_single_file_components(self):
        single_file_url = "https://huggingface.co/stabilityai/stable-cascade/blob/main/stage_c_bf16.safetensors"
        single_file_unet = StableCascadeUNet.from_single_file(single_file_url)

        single_file_unet_config = single_file_unet.config
        del single_file_unet
        gc.collect()
        torch.cuda.empty_cache()

        unet = StableCascadeUNet.from_pretrained("stabilityai/stable-cascade-prior", subfolder="prior", variant="bf16")
        unet_config = unet.config
        del unet
        gc.collect()
        torch.cuda.empty_cache()

        PARAMS_TO_IGNORE = ["torch_dtype", "_name_or_path", "_use_default_values"]
        for param_name, param_value in single_file_unet_config.items():
            if param_name in PARAMS_TO_IGNORE:
                continue

            assert unet_config[param_name] == param_value

    def test_stable_cascade_unet_decoder_single_file_components(self):
        single_file_url = "https://huggingface.co/stabilityai/stable-cascade/blob/main/stage_b_bf16.safetensors"
        single_file_unet = StableCascadeUNet.from_single_file(single_file_url)

        single_file_unet_config = single_file_unet.config
        del single_file_unet
        gc.collect()
        torch.cuda.empty_cache()

        unet = StableCascadeUNet.from_pretrained("stabilityai/stable-cascade", subfolder="decoder", variant="bf16")
        unet_config = unet.config
        del unet
        gc.collect()
        torch.cuda.empty_cache()

        PARAMS_TO_IGNORE = ["torch_dtype", "_name_or_path", "_use_default_values"]
        for param_name, param_value in single_file_unet_config.items():
            if param_name in PARAMS_TO_IGNORE:
                continue

            assert unet_config[param_name] == param_value

    def test_stable_cascade_unet_config_loading(self):
        config = StableCascadeUNet.load_config(
            pretrained_model_name_or_path="diffusers/stable-cascade-configs", subfolder="prior"
        )
        single_file_url = "https://huggingface.co/stabilityai/stable-cascade/blob/main/stage_c_bf16.safetensors"

        single_file_unet = StableCascadeUNet.from_single_file(single_file_url, config=config)
        single_file_unet_config = single_file_unet.config
        del single_file_unet
        gc.collect()
        torch.cuda.empty_cache()

        PARAMS_TO_IGNORE = ["torch_dtype", "_name_or_path", "_use_default_values"]
        for param_name, param_value in config.items():
            if param_name in PARAMS_TO_IGNORE:
                continue

            assert single_file_unet_config[param_name] == param_value

    @require_torch_gpu
    def test_stable_cascade_unet_single_file_prior_forward_pass(self):
        dtype = torch.bfloat16
        generator = torch.Generator("cpu")

        model_inputs = {
            "sample": randn_tensor((1, 16, 24, 24), generator=generator.manual_seed(0)).to("cuda", dtype),
            "timestep_ratio": torch.tensor([1]).to("cuda", dtype),
            "clip_text_pooled": randn_tensor((1, 1, 1280), generator=generator.manual_seed(0)).to("cuda", dtype),
            "clip_text": randn_tensor((1, 77, 1280), generator=generator.manual_seed(0)).to("cuda", dtype),
            "clip_img": randn_tensor((1, 1, 768), generator=generator.manual_seed(0)).to("cuda", dtype),
            "pixels": randn_tensor((1, 3, 8, 8), generator=generator.manual_seed(0)).to("cuda", dtype),
        }

        unet = StableCascadeUNet.from_pretrained(
            "stabilityai/stable-cascade-prior",
            subfolder="prior",
            revision="refs/pr/2",
            variant="bf16",
            torch_dtype=dtype,
        )
        unet.to("cuda")
        with torch.no_grad():
            prior_output = unet(**model_inputs).sample.float().cpu().numpy()

        # Remove UNet from GPU memory before loading the single file UNet model
        del unet
        gc.collect()
        torch.cuda.empty_cache()

        single_file_url = "https://huggingface.co/stabilityai/stable-cascade/blob/main/stage_c_bf16.safetensors"
        single_file_unet = StableCascadeUNet.from_single_file(single_file_url, torch_dtype=dtype)
        single_file_unet.to("cuda")
        with torch.no_grad():
            prior_single_file_output = single_file_unet(**model_inputs).sample.float().cpu().numpy()

        # Remove UNet from GPU memory before loading the single file UNet model
        del single_file_unet
        gc.collect()
        torch.cuda.empty_cache()

        max_diff = numpy_cosine_similarity_distance(prior_output.flatten(), prior_single_file_output.flatten())
        assert max_diff < 8e-3

    @require_torch_gpu
    def test_stable_cascade_unet_single_file_decoder_forward_pass(self):
        dtype = torch.float32
        generator = torch.Generator("cpu")

        model_inputs = {
            "sample": randn_tensor((1, 4, 256, 256), generator=generator.manual_seed(0)).to("cuda", dtype),
            "timestep_ratio": torch.tensor([1]).to("cuda", dtype),
            "clip_text": randn_tensor((1, 77, 1280), generator=generator.manual_seed(0)).to("cuda", dtype),
            "clip_text_pooled": randn_tensor((1, 1, 1280), generator=generator.manual_seed(0)).to("cuda", dtype),
            "pixels": randn_tensor((1, 3, 8, 8), generator=generator.manual_seed(0)).to("cuda", dtype),
        }

        unet = StableCascadeUNet.from_pretrained(
            "stabilityai/stable-cascade",
            subfolder="decoder",
            revision="refs/pr/44",
            torch_dtype=dtype,
        )
        unet.to("cuda")
        with torch.no_grad():
            prior_output = unet(**model_inputs).sample.float().cpu().numpy()

        # Remove UNet from GPU memory before loading the single file UNet model
        del unet
        gc.collect()
        torch.cuda.empty_cache()

        single_file_url = "https://huggingface.co/stabilityai/stable-cascade/blob/main/stage_b.safetensors"
        single_file_unet = StableCascadeUNet.from_single_file(single_file_url, torch_dtype=dtype)
        single_file_unet.to("cuda")
        with torch.no_grad():
            prior_single_file_output = single_file_unet(**model_inputs).sample.float().cpu().numpy()

        # Remove UNet from GPU memory before loading the single file UNet model
        del single_file_unet
        gc.collect()
        torch.cuda.empty_cache()

        max_diff = numpy_cosine_similarity_distance(prior_output.flatten(), prior_single_file_output.flatten())
        assert max_diff < 1e-4
