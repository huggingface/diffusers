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
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from diffusers.loaders import StableDiffusionLoraLoaderMixin
from diffusers.loaders.lora_base import _best_guess_weight_name


class LoraBaseBestGuessWeightNameTests(unittest.TestCase):
    def test_local_file_with_local_files_only_returns_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            weight_path = Path(tmpdir) / "adapter.safetensors"
            save_file({"unet.test.lora_A.weight": torch.ones(1)}, str(weight_path))

            weight_name = _best_guess_weight_name(str(weight_path), local_files_only=True)
            self.assertIsNone(weight_name)

    def test_local_directory_with_local_files_only_discovers_weight(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            weight_path = Path(tmpdir) / "adapter.safetensors"
            save_file({"unet.test.lora_A.weight": torch.ones(1)}, str(weight_path))

            weight_name = _best_guess_weight_name(tmpdir, local_files_only=True)
            self.assertEqual(weight_name, "adapter.safetensors")

    def test_local_directory_with_hf_hub_offline_discovers_weight(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            weight_path = Path(tmpdir) / "adapter.safetensors"
            save_file({"unet.test.lora_A.weight": torch.ones(1)}, str(weight_path))

            with patch("diffusers.loaders.lora_base.HF_HUB_OFFLINE", True):
                weight_name = _best_guess_weight_name(tmpdir)
            self.assertEqual(weight_name, "adapter.safetensors")

    def test_remote_repo_with_local_files_only_raises(self):
        with self.assertRaises(ValueError) as cm:
            _best_guess_weight_name("some/repo", local_files_only=True)
        self.assertIn("offline mode", str(cm.exception).lower())


class LoraBaseOfflineLoadingTests(unittest.TestCase):
    def test_lora_state_dict_local_directory_with_local_files_only(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            weight_path = Path(tmpdir) / "adapter.safetensors"
            save_file({"unet.test.lora_A.weight": torch.ones(1)}, str(weight_path))

            state_dict, _ = StableDiffusionLoraLoaderMixin.lora_state_dict(tmpdir, local_files_only=True)
            self.assertIn("unet.test.lora_A.weight", state_dict)

    def test_lora_state_dict_local_file_with_local_files_only(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            weight_path = Path(tmpdir) / "adapter.safetensors"
            save_file({"unet.test.lora_A.weight": torch.ones(1)}, str(weight_path))

            state_dict, _ = StableDiffusionLoraLoaderMixin.lora_state_dict(str(weight_path), local_files_only=True)
            self.assertIn("unet.test.lora_A.weight", state_dict)

    @patch.dict(os.environ, {"HF_HUB_OFFLINE": "1"})
    def test_lora_state_dict_local_directory_with_hf_hub_offline(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            weight_path = Path(tmpdir) / "adapter.safetensors"
            save_file({"unet.test.lora_A.weight": torch.ones(1)}, str(weight_path))

            state_dict, _ = StableDiffusionLoraLoaderMixin.lora_state_dict(tmpdir)
            self.assertIn("unet.test.lora_A.weight", state_dict)


if __name__ == "__main__":
    unittest.main()
