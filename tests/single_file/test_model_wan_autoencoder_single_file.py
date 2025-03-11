# coding=utf-8
# Copyright 2025 HuggingFace Inc.
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

from diffusers import (
    AutoencoderKLWan,
)
from diffusers.utils.testing_utils import (
    backend_empty_cache,
    enable_full_determinism,
    require_torch_accelerator,
    torch_device,
)


enable_full_determinism()


@require_torch_accelerator
class AutoencoderKLWanSingleFileTests(unittest.TestCase):
    model_class = AutoencoderKLWan
    ckpt_path = (
        "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/blob/main/split_files/vae/wan_2.1_vae.safetensors"
    )
    repo_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"

    def setUp(self):
        super().setUp()
        gc.collect()
        backend_empty_cache(torch_device)

    def tearDown(self):
        super().tearDown()
        gc.collect()
        backend_empty_cache(torch_device)

    def test_single_file_components(self):
        model = self.model_class.from_pretrained(self.repo_id, subfolder="vae")
        model_single_file = self.model_class.from_single_file(self.ckpt_path)

        PARAMS_TO_IGNORE = ["torch_dtype", "_name_or_path", "_use_default_values", "_diffusers_version"]
        for param_name, param_value in model_single_file.config.items():
            if param_name in PARAMS_TO_IGNORE:
                continue
            assert model.config[param_name] == param_value, (
                f"{param_name} differs between single file loading and pretrained loading"
            )
