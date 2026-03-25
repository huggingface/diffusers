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

import torch

from diffusers.models.autoencoders import AutoencoderULEncoder


class AutoencoderULEncoderTests(unittest.TestCase):
    def test_ul_encoder_api(self):
        model = AutoencoderULEncoder()
        self.assertEqual(tuple(model.config.layers_per_block), (2, 2, 2, 3))

        x = torch.randn(2, 3, 32, 32)
        encoded = model.encode(x)
        self.assertEqual(encoded.latent.shape, (2, 4, 4, 4))

    def test_ul_encoder_save_load(self):
        model = AutoencoderULEncoder()
        x = torch.randn(1, 3, 32, 32)
        y_ref = model.encode(x).latent

        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir)
            loaded = AutoencoderULEncoder.from_pretrained(tmp_dir)
            y_loaded = loaded.encode(x).latent

        self.assertTrue(torch.allclose(y_ref, y_loaded, atol=1e-6))
