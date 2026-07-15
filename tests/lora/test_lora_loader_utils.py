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
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from diffusers.loaders import StableDiffusionLoraLoaderMixin
from diffusers.loaders.lora_base import _best_guess_weight_name


LORA_KEY = "unet.test.lora_A.weight"


def _write_lora_weights(path):
    save_file({LORA_KEY: torch.ones(1)}, path)


class LoraWeightNameDiscoveryTests(unittest.TestCase):
    def test_local_directory_in_offline_mode(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            _write_lora_weights(Path(tmpdirname) / "adapter.safetensors")

            with (
                patch("diffusers.loaders.lora_base.HF_HUB_OFFLINE", True),
                patch("diffusers.loaders.lora_base.model_info") as model_info_mock,
            ):
                state_dict, _ = StableDiffusionLoraLoaderMixin.lora_state_dict(tmpdirname)

            self.assertTrue(torch.equal(state_dict[LORA_KEY], torch.ones(1)))
            model_info_mock.assert_not_called()

    def test_local_directory_with_local_files_only(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            _write_lora_weights(Path(tmpdirname) / "adapter.safetensors")

            with patch("diffusers.loaders.lora_base.model_info") as model_info_mock:
                state_dict, _ = StableDiffusionLoraLoaderMixin.lora_state_dict(tmpdirname, local_files_only=True)

            self.assertTrue(torch.equal(state_dict[LORA_KEY], torch.ones(1)))
            model_info_mock.assert_not_called()

    def test_local_file_in_offline_mode(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            weight_path = Path(tmpdirname) / "adapter.safetensors"
            _write_lora_weights(weight_path)

            with (
                patch("diffusers.loaders.lora_base.HF_HUB_OFFLINE", True),
                patch("diffusers.loaders.lora_base.model_info") as model_info_mock,
            ):
                state_dict, _ = StableDiffusionLoraLoaderMixin.lora_state_dict(weight_path)

            self.assertTrue(torch.equal(state_dict[LORA_KEY], torch.ones(1)))
            model_info_mock.assert_not_called()

    def test_remote_repository_in_offline_mode_requires_weight_name(self):
        with (
            patch("diffusers.loaders.lora_base.HF_HUB_OFFLINE", True),
            patch("diffusers.loaders.lora_base.model_info") as model_info_mock,
        ):
            with self.assertRaisesRegex(ValueError, "offline mode.*weight_name"):
                StableDiffusionLoraLoaderMixin.lora_state_dict("organization/repository")

        model_info_mock.assert_not_called()

    def test_local_directory_without_matching_files_returns_none(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            (Path(tmpdirname) / "notes.txt").touch()

            with patch("diffusers.loaders.lora_base.HF_HUB_OFFLINE", True):
                weight_name = _best_guess_weight_name(tmpdirname)

        self.assertIsNone(weight_name)

    def test_local_directory_with_multiple_files_warns_and_uses_first(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            first_path = Path(tmpdirname) / "first.safetensors"
            second_path = Path(tmpdirname) / "second.safetensors"
            first_path.touch()
            second_path.touch()

            with (
                patch("diffusers.loaders.lora_base.HF_HUB_OFFLINE", True),
                patch(
                    "diffusers.loaders.lora_base.os.listdir",
                    return_value=[first_path.name, second_path.name],
                ),
                self.assertLogs("diffusers.loaders.lora_base", level="WARNING") as logs,
            ):
                weight_name = _best_guess_weight_name(tmpdirname)

        self.assertEqual(weight_name, first_path.name)
        self.assertIn("contains more than one weights file", logs.output[0])
