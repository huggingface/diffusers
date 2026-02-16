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

import pathlib
import tempfile
import unittest

from diffusers.models.auto_model import AutoModel
from diffusers import AutoPipelineForText2Image

from ..testing_utils import require_flashpack, require_torch_gpu, is_torch_available

if is_torch_available():
    import torch


class FlashPackTests(unittest.TestCase):
    model_id: str = "hf-internal-testing/tiny-flux-pipe"

    @require_flashpack
    def test_save_load_model(self):
        model = AutoModel.from_pretrained(self.model_id, subfolder="transformer")
        with tempfile.TemporaryDirectory() as temp_dir:
            model.save_pretrained(temp_dir, use_flashpack=True)
            self.assertTrue((pathlib.Path(temp_dir) / "model.flashpack").exists())
            model = AutoModel.from_pretrained(temp_dir, use_flashpack=True)

    @require_flashpack
    def test_save_load_pipeline(self):
        pipeline = AutoPipelineForText2Image.from_pretrained(self.model_id)
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline.save_pretrained(temp_dir, use_flashpack=True)
            self.assertTrue((pathlib.Path(temp_dir) / "transformer" / "model.flashpack").exists())
            self.assertTrue((pathlib.Path(temp_dir) / "vae" / "model.flashpack").exists())
            pipeline = AutoPipelineForText2Image.from_pretrained(temp_dir, use_flashpack=True)

    @require_torch_gpu
    @require_flashpack
    def test_load_model_device_str(self):
        model = AutoModel.from_pretrained(self.model_id, subfolder="transformer")
        with tempfile.TemporaryDirectory() as temp_dir:
            model.save_pretrained(temp_dir, use_flashpack=True)
            model = AutoModel.from_pretrained(temp_dir, use_flashpack=True, device_map={"": "cuda"})
            self.assertTrue(model.device.type == "cuda")

    @require_torch_gpu
    @require_flashpack
    def test_load_model_device(self):
        model = AutoModel.from_pretrained(self.model_id, subfolder="transformer")
        with tempfile.TemporaryDirectory() as temp_dir:
            model.save_pretrained(temp_dir, use_flashpack=True)
            model = AutoModel.from_pretrained(temp_dir, use_flashpack=True, device_map={"": torch.device("cuda")})
            self.assertTrue(model.device.type == "cuda")

    @require_flashpack
    def test_load_model_device_auto(self):
        model = AutoModel.from_pretrained(self.model_id, subfolder="transformer")
        with tempfile.TemporaryDirectory() as temp_dir:
            model.save_pretrained(temp_dir, use_flashpack=True)
            with self.assertRaises(ValueError):
                model = AutoModel.from_pretrained(temp_dir, use_flashpack=True, device_map={"": "auto"})
