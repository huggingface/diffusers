# coding=utf-8
# Copyright 2023 HuggingFace Inc.
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
import copy
import os
import tempfile
import unittest

import torch
from transformers import CLIPTextConfig, CLIPTextModel

from diffusers.loaders import TextEncoderLoRAMixin
from diffusers.utils import torch_device


class TextEncoderLoRATests(unittest.TestCase):
    def get_dummy_components(self):
        torch.manual_seed(0)
        text_encoder_config = CLIPTextConfig(
            bos_token_id=0,
            eos_token_id=2,
            hidden_size=32,
            intermediate_size=37,
            layer_norm_eps=1e-05,
            num_attention_heads=4,
            num_hidden_layers=5,
            pad_token_id=1,
            vocab_size=1000,
        )
        text_encoder = CLIPTextModel(text_encoder_config).to(torch_device)
        text_encoder_lora_wrapper = TextEncoderLoRAMixin(copy.deepcopy(text_encoder))
        lora_attn_procs = text_encoder_lora_wrapper.lora_attn_procs
        text_encoder_lora = text_encoder_lora_wrapper._modify_text_encoder(lora_attn_procs)
        return text_encoder, text_encoder_lora, text_encoder_lora_wrapper

    def get_dummy_inputs(self):
        batch_size = 1
        sequence_length = 10
        generator = torch.manual_seed(0)
        input_ids = torch.randint(1, sequence_length, size=(batch_size, sequence_length), generator=generator)
        return input_ids

    def test_lora_default_case(self):
        text_encoder, text_encoder_lora, _ = self.get_dummy_components()
        inputs = self.get_dummy_inputs()

        with torch.no_grad():
            original_outputs = text_encoder(inputs)[0]
            lora_outputs = text_encoder_lora(inputs)[0]

        # Outputs shouldn't match.
        self.assertFalse(torch.allclose(original_outputs, lora_outputs))

    def test_lora_save_load(self):
        text_encoder, _, text_encoder_lora_wrapper = self.get_dummy_components()
        inputs = self.get_dummy_inputs()

        with torch.no_grad():
            original_outputs = text_encoder(inputs)[0]

        with tempfile.TemporaryDirectory() as tmpdirname:
            text_encoder_lora_wrapper.save_attn_procs(tmpdirname)
            self.assertTrue(os.path.isfile(os.path.join(tmpdirname, "pytorch_text_encoder_lora_weights.bin")))
            text_encoder_lora = text_encoder_lora_wrapper.load_attn_procs(tmpdirname)

        with torch.no_grad():
            lora_outputs = text_encoder_lora(inputs)[0]

        # Outputs shouldn't match.
        self.assertFalse(torch.allclose(original_outputs, lora_outputs))

    def test_lora_save_load_safetensors(self):
        text_encoder, _, text_encoder_lora_wrapper = self.get_dummy_components()
        inputs = self.get_dummy_inputs()

        with torch.no_grad():
            original_outputs = text_encoder(inputs)[0]

        with tempfile.TemporaryDirectory() as tmpdirname:
            text_encoder_lora_wrapper.save_attn_procs(tmpdirname, safe_serialization=True)
            self.assertTrue(os.path.isfile(os.path.join(tmpdirname, "pytorch_text_encoder_lora_weights.safetensors")))
            text_encoder_lora = text_encoder_lora_wrapper.load_attn_procs(tmpdirname)

        with torch.no_grad():
            lora_outputs = text_encoder_lora(inputs)[0]

        # Outputs shouldn't match.
        self.assertFalse(torch.allclose(original_outputs, lora_outputs))
