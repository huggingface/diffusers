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

from diffusers import MagiTransformer3DModel
from diffusers.utils.testing_utils import (
    require_torch_gpu,
    slow,
    torch_device,
)


class MagiTransformer3DModelText2VideoSingleFileTest(unittest.TestCase):
    model_class = MagiTransformer3DModel
    ckpt_path = "https://huggingface.co/sand-ai/MAGI-1/blob/main/transformer/diffusion_pytorch_model.safetensors"
    repo_id = "sand-ai/MAGI-1"

    @slow
    @require_torch_gpu
    def test_single_file_components(self):
        model = self.model_class.from_single_file(self.ckpt_path)
        model.to(torch_device)

        batch_size = 1
        num_channels = 4
        num_frames = 2
        height = 16
        width = 16
        text_encoder_embedding_dim = 16
        sequence_length = 12

        hidden_states = torch.randn((batch_size, num_channels, num_frames, height, width)).to(torch_device)
        timestep = torch.randint(0, 1000, size=(batch_size,)).to(torch_device)
        encoder_hidden_states = torch.randn((batch_size, sequence_length, text_encoder_embedding_dim)).to(torch_device)

        with torch.no_grad():
            model(
                hidden_states=hidden_states,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                return_dict=False,
            )

    @slow
    @require_torch_gpu
    def test_single_file_components_from_hub(self):
        model = self.model_class.from_pretrained(self.repo_id, subfolder="transformer")
        model.to(torch_device)

        batch_size = 1
        num_channels = 4
        num_frames = 2
        height = 16
        width = 16
        text_encoder_embedding_dim = 16
        sequence_length = 12

        hidden_states = torch.randn((batch_size, num_channels, num_frames, height, width)).to(torch_device)
        timestep = torch.randint(0, 1000, size=(batch_size,)).to(torch_device)
        encoder_hidden_states = torch.randn((batch_size, sequence_length, text_encoder_embedding_dim)).to(torch_device)

        with torch.no_grad():
            model(
                hidden_states=hidden_states,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                return_dict=False,
            )
