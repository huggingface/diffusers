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

import torch
from huggingface_hub import hf_hub_download

from diffusers import AutoencoderKLMagi
from diffusers.utils.testing_utils import (
    require_torch_gpu,
    slow,
    torch_device,
)


class AutoencoderKLMagiSingleFileTests(unittest.TestCase):
    model_class = AutoencoderKLMagi
    ckpt_path = (
        "https://huggingface.co/sand-ai/MAGI-1/blob/main/vae/diffusion_pytorch_model.safetensors"
    )
    repo_id = "sand-ai/MAGI-1"

    @slow
    @require_torch_gpu
    def test_single_file_components(self):
        model = self.model_class.from_single_file(self.ckpt_path)
        model.to(torch_device)

        batch_size = 1
        num_frames = 2
        num_channels = 3
        sizes = (16, 16)
        image = torch.randn((batch_size, num_channels, num_frames) + sizes).to(torch_device)

        with torch.no_grad():
            model(image, return_dict=False)

    @slow
    @require_torch_gpu
    def test_single_file_components_from_hub(self):
        model = self.model_class.from_pretrained(self.repo_id, subfolder="vae")
        model.to(torch_device)

        batch_size = 1
        num_frames = 2
        num_channels = 3
        sizes = (16, 16)
        image = torch.randn((batch_size, num_channels, num_frames) + sizes).to(torch_device)

        with torch.no_grad():
            model(image, return_dict=False)