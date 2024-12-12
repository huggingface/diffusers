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

from diffusers import (
    AutoencoderDC,
)
from diffusers.utils.testing_utils import (
    backend_empty_cache,
    enable_full_determinism,
    load_hf_numpy,
    numpy_cosine_similarity_distance,
    require_torch_accelerator,
    slow,
    torch_device,
)


enable_full_determinism()


@slow
@require_torch_accelerator
class AutoencoderDCSingleFileTests(unittest.TestCase):
    model_class = AutoencoderDC
    ckpt_path = "https://huggingface.co/mit-han-lab/dc-ae-f32c32-sana-1.0/blob/main/model.safetensors"
    repo_id = "mit-han-lab/dc-ae-f32c32-sana-1.0-diffusers"
    main_input_name = "sample"
    base_precision = 1e-2

    def setUp(self):
        super().setUp()
        gc.collect()
        backend_empty_cache(torch_device)

    def tearDown(self):
        super().tearDown()
        gc.collect()
        backend_empty_cache(torch_device)

    def get_file_format(self, seed, shape):
        return f"gaussian_noise_s={seed}_shape={'_'.join([str(s) for s in shape])}.npy"

    def get_sd_image(self, seed=0, shape=(4, 3, 512, 512), fp16=False):
        dtype = torch.float16 if fp16 else torch.float32
        image = torch.from_numpy(load_hf_numpy(self.get_file_format(seed, shape))).to(torch_device).to(dtype)
        return image

    def test_single_file_inference_same_as_pretrained(self):
        model_1 = self.model_class.from_pretrained(self.repo_id).to(torch_device)
        model_2 = self.model_class.from_single_file(self.ckpt_path, config=self.repo_id).to(torch_device)

        image = self.get_sd_image(33)

        with torch.no_grad():
            sample_1 = model_1(image).sample
            sample_2 = model_2(image).sample

        assert sample_1.shape == sample_2.shape

        output_slice_1 = sample_1.flatten().float().cpu()
        output_slice_2 = sample_2.flatten().float().cpu()

        assert numpy_cosine_similarity_distance(output_slice_1, output_slice_2) < 1e-4

    def test_single_file_components(self):
        model = self.model_class.from_pretrained(self.repo_id)
        model_single_file = self.model_class.from_single_file(self.ckpt_path)

        PARAMS_TO_IGNORE = ["torch_dtype", "_name_or_path", "_use_default_values", "_diffusers_version"]
        for param_name, param_value in model_single_file.config.items():
            if param_name in PARAMS_TO_IGNORE:
                continue
            assert (
                model.config[param_name] == param_value
            ), f"{param_name} differs between pretrained loading and single file loading"

    def test_single_file_in_type_variant_components(self):
        # `in` variant checkpoints require passing in a `config` parameter
        # in order to set the scaling factor correctly.
        # `in` and `mix` variants have the same keys and we cannot automatically infer a scaling factor.
        # We default to using teh `mix` config
        repo_id = "mit-han-lab/dc-ae-f128c512-in-1.0-diffusers"
        ckpt_path = "https://huggingface.co/mit-han-lab/dc-ae-f128c512-in-1.0/blob/main/model.safetensors"

        model = self.model_class.from_pretrained(repo_id)
        model_single_file = self.model_class.from_single_file(ckpt_path, config=repo_id)

        PARAMS_TO_IGNORE = ["torch_dtype", "_name_or_path", "_use_default_values", "_diffusers_version"]
        for param_name, param_value in model_single_file.config.items():
            if param_name in PARAMS_TO_IGNORE:
                continue
            assert (
                model.config[param_name] == param_value
            ), f"{param_name} differs between pretrained loading and single file loading"

    def test_single_file_mix_type_variant_components(self):
        repo_id = "mit-han-lab/dc-ae-f128c512-mix-1.0-diffusers"
        ckpt_path = "https://huggingface.co/mit-han-lab/dc-ae-f128c512-mix-1.0/blob/main/model.safetensors"

        model = self.model_class.from_pretrained(repo_id)
        model_single_file = self.model_class.from_single_file(ckpt_path, config=repo_id)

        PARAMS_TO_IGNORE = ["torch_dtype", "_name_or_path", "_use_default_values", "_diffusers_version"]
        for param_name, param_value in model_single_file.config.items():
            if param_name in PARAMS_TO_IGNORE:
                continue
            assert (
                model.config[param_name] == param_value
            ), f"{param_name} differs between pretrained loading and single file loading"
