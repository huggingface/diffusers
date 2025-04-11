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

import unittest

from diffusers import (
    MotionAdapter,
)
from diffusers.utils.testing_utils import (
    enable_full_determinism,
)


enable_full_determinism()


class MotionAdapterSingleFileTests(unittest.TestCase):
    model_class = MotionAdapter

    def test_single_file_components_version_v1_5(self):
        ckpt_path = "https://huggingface.co/guoyww/animatediff/blob/main/mm_sd_v15.ckpt"
        repo_id = "guoyww/animatediff-motion-adapter-v1-5"

        model = self.model_class.from_pretrained(repo_id)
        model_single_file = self.model_class.from_single_file(ckpt_path)

        PARAMS_TO_IGNORE = ["torch_dtype", "_name_or_path", "_use_default_values", "_diffusers_version"]
        for param_name, param_value in model_single_file.config.items():
            if param_name in PARAMS_TO_IGNORE:
                continue
            assert (
                model.config[param_name] == param_value
            ), f"{param_name} differs between pretrained loading and single file loading"

    def test_single_file_components_version_v1_5_2(self):
        ckpt_path = "https://huggingface.co/guoyww/animatediff/blob/main/mm_sd_v15_v2.ckpt"
        repo_id = "guoyww/animatediff-motion-adapter-v1-5-2"

        model = self.model_class.from_pretrained(repo_id)
        model_single_file = self.model_class.from_single_file(ckpt_path)

        PARAMS_TO_IGNORE = ["torch_dtype", "_name_or_path", "_use_default_values", "_diffusers_version"]
        for param_name, param_value in model_single_file.config.items():
            if param_name in PARAMS_TO_IGNORE:
                continue
            assert (
                model.config[param_name] == param_value
            ), f"{param_name} differs between pretrained loading and single file loading"

    def test_single_file_components_version_v1_5_3(self):
        ckpt_path = "https://huggingface.co/guoyww/animatediff/blob/main/v3_sd15_mm.ckpt"
        repo_id = "guoyww/animatediff-motion-adapter-v1-5-3"

        model = self.model_class.from_pretrained(repo_id)
        model_single_file = self.model_class.from_single_file(ckpt_path)

        PARAMS_TO_IGNORE = ["torch_dtype", "_name_or_path", "_use_default_values", "_diffusers_version"]
        for param_name, param_value in model_single_file.config.items():
            if param_name in PARAMS_TO_IGNORE:
                continue
            assert (
                model.config[param_name] == param_value
            ), f"{param_name} differs between pretrained loading and single file loading"

    def test_single_file_components_version_sdxl_beta(self):
        ckpt_path = "https://huggingface.co/guoyww/animatediff/blob/main/mm_sdxl_v10_beta.ckpt"
        repo_id = "guoyww/animatediff-motion-adapter-sdxl-beta"

        model = self.model_class.from_pretrained(repo_id)
        model_single_file = self.model_class.from_single_file(ckpt_path)

        PARAMS_TO_IGNORE = ["torch_dtype", "_name_or_path", "_use_default_values", "_diffusers_version"]
        for param_name, param_value in model_single_file.config.items():
            if param_name in PARAMS_TO_IGNORE:
                continue
            assert (
                model.config[param_name] == param_value
            ), f"{param_name} differs between pretrained loading and single file loading"
