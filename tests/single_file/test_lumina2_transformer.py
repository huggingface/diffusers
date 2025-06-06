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
    Lumina2Transformer2DModel,
)
from diffusers.utils.testing_utils import (
    backend_empty_cache,
    enable_full_determinism,
    require_torch_accelerator,
    torch_device,
)


enable_full_determinism()


@require_torch_accelerator
class Lumina2Transformer2DModelSingleFileTests(unittest.TestCase):
    model_class = Lumina2Transformer2DModel
    ckpt_path = "https://huggingface.co/Comfy-Org/Lumina_Image_2.0_Repackaged/blob/main/split_files/diffusion_models/lumina_2_model_bf16.safetensors"
    alternate_keys_ckpt_paths = [
        "https://huggingface.co/Comfy-Org/Lumina_Image_2.0_Repackaged/blob/main/split_files/diffusion_models/lumina_2_model_bf16.safetensors"
    ]

    repo_id = "Alpha-VLLM/Lumina-Image-2.0"

    def setUp(self):
        super().setUp()
        gc.collect()
        backend_empty_cache(torch_device)

    def tearDown(self):
        super().tearDown()
        gc.collect()
        backend_empty_cache(torch_device)

    def test_single_file_components(self):
        model = self.model_class.from_pretrained(self.repo_id, subfolder="transformer")
        model_single_file = self.model_class.from_single_file(self.ckpt_path)

        PARAMS_TO_IGNORE = ["torch_dtype", "_name_or_path", "_use_default_values", "_diffusers_version"]
        for param_name, param_value in model_single_file.config.items():
            if param_name in PARAMS_TO_IGNORE:
                continue
            assert model.config[param_name] == param_value, (
                f"{param_name} differs between single file loading and pretrained loading"
            )

    def test_checkpoint_loading(self):
        for ckpt_path in self.alternate_keys_ckpt_paths:
            backend_empty_cache(torch_device)
            model = self.model_class.from_single_file(ckpt_path)

            del model
            gc.collect()
            backend_empty_cache(torch_device)
