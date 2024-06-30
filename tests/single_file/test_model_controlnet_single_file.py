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
    ControlNetModel,
)
from diffusers.utils.testing_utils import (
    enable_full_determinism,
    require_torch_gpu,
    slow,
)


enable_full_determinism()


@slow
@require_torch_gpu
class ControlNetModelSingleFileTests(unittest.TestCase):
    model_class = ControlNetModel
    ckpt_path = "https://huggingface.co/lllyasviel/ControlNet-v1-1/blob/main/control_v11p_sd15_canny.pth"
    repo_id = "lllyasviel/control_v11p_sd15_canny"

    def setUp(self):
        super().setUp()
        gc.collect()
        torch.cuda.empty_cache()

    def tearDown(self):
        super().tearDown()
        gc.collect()
        torch.cuda.empty_cache()

    def test_single_file_components(self):
        model = self.model_class.from_pretrained(self.repo_id)
        model_single_file = self.model_class.from_single_file(self.ckpt_path)

        PARAMS_TO_IGNORE = ["torch_dtype", "_name_or_path", "_use_default_values", "_diffusers_version"]
        for param_name, param_value in model_single_file.config.items():
            if param_name in PARAMS_TO_IGNORE:
                continue
            assert (
                model.config[param_name] == param_value
            ), f"{param_name} differs between single file loading and pretrained loading"

    def test_single_file_arguments(self):
        model_default = self.model_class.from_single_file(self.ckpt_path)

        assert model_default.config.upcast_attention is False
        assert model_default.dtype == torch.float32

        torch_dtype = torch.float16
        upcast_attention = True

        model = self.model_class.from_single_file(
            self.ckpt_path,
            upcast_attention=upcast_attention,
            torch_dtype=torch_dtype,
        )
        assert model.config.upcast_attention == upcast_attention
        assert model.dtype == torch_dtype
