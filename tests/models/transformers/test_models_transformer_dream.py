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

import unittest

import torch

from diffusers import DreamTransformer1DModel
from diffusers.utils.testing_utils import (
    enable_full_determinism,
    require_torch_accelerator_with_training,
    torch_device,
)

from ..test_modeling_common import ModelTesterMixin, TorchCompileTesterMixin


enable_full_determinism()


class DreamTransformerTests(ModelTesterMixin, unittest.TestCase):
    model_class = DreamTransformer1DModel
    main_input_name = "text_ids"

    # Skip setting testing with default: AttnProcessor
    uses_custom_attn_processor = True

    @property
    def dummy_input(self):
        return self.prepare_dummy_input()

    @property
    def input_shape(self):
        return (48,)  # (sequence_length,)

    @property
    def output_shape(self):
        return (48, 100)  # (sequence_length, vocab_size)

    def prepare_dummy_input(self, batch_size: int = 1, sequence_length: int = 48):
        vocab_size = 100

        text_ids = torch.randint(vocab_size, size=(batch_size, sequence_length), device=torch_device)
        # NOTE: dummy timestep input for now (not used)
        # timestep = torch.tensor([1.0]).to(torch_device).expand(batch_size)

        inputs_dict = {"text_ids": text_ids}
        return inputs_dict

    def prepare_init_args_and_inputs_for_common(self):
        init_dict = {
            "num_layers": 1,
            "attention_head_dim": 16,
            "num_attention_heads": 4,
            "num_attention_kv_heads": 2,
            "ff_intermediate_dim": 256,  # 4 * (attention_head_dim * num_attention_heads)
            "vocab_size": 100,
            "pad_token_id": 90,
        }

        inputs_dict = self.dummy_input
        return init_dict, inputs_dict

    # NOTE: override ModelTesterMixin.test_output to supply a custom expected_output_shape as the expected output
    # shape of the Dream transformer is not the same as the input shape
    def test_output(self, expected_output_shape=None):
        if expected_output_shape is None:
            init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
            vocab_size = init_dict["vocab_size"]
            batch_size, seq_len = inputs_dict["text_ids"].shape
            expected_output_shape = (batch_size, seq_len, vocab_size)
        super().test_output(expected_output_shape=expected_output_shape)

    def test_output_hidden_states_supplied(self):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()\

        # Prepare hidden_states argument manually, remove text_ids arg.
        hidden_dim = init_dict["attention_head_dim"] * init_dict["num_attention_heads"]
        vocab_size = init_dict["vocab_size"]
        batch_size, seq_len = inputs_dict["text_ids"].shape
        hidden_states = torch.randn((batch_size, seq_len, hidden_dim), device=torch_device)
        inputs_dict["hidden_states"] = hidden_states
        del inputs_dict["text_ids"]

        expected_output_shape = (batch_size, seq_len, vocab_size)
        super().test_output(expected_output_shape=expected_output_shape)

    def test_output_positions_ids_supplied(self):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()

        # Prepare position_ids argument manually.
        vocab_size = init_dict["vocab_size"]
        position_ids = torch.arange(inputs_dict["text_ids"].shape[1], device=torch_device)
        position_ids = position_ids.unsqueeze(0).expand(inputs_dict["text_ids"].shape[0], -1)
        inputs_dict["position_ids"] = position_ids

        expected_output_shape = (inputs_dict["text_ids"].shape[0], inputs_dict["text_ids"].shape[1], vocab_size)
        super().test_output(expected_output_shape=expected_output_shape)

    @require_torch_accelerator_with_training
    def test_training_attention_mask_supplied(self):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()

        vocab_size = init_dict["vocab_size"]
        batch_size, seq_len = inputs_dict["text_ids"].shape

        model = self.model_class(**init_dict)
        model.to(torch_device)
        model.train()
        dtype = model.dtype

        # Prepare causal attention mask for training, specifically a transformers-style 4D additive causal mask with
        # the upper triangular entries filled with -inf
        attention_mask = None
        causal_mask = torch.full((seq_len, seq_len), torch.finfo(dtype).min, device=torch_device)
        positions = torch.arange(causal_mask.size(-1), device=torch_device)
        causal_mask.masked_fill_(positions < (positions + 1).view(causal_mask.size(-1), 1), 0)
        attention_mask = causal_mask[None, None, :, :].expand(batch_size, 1, seq_len, seq_len)
        inputs_dict["attention_mask"] = attention_mask

        logits = model(**inputs_dict)

        if isinstance(logits, dict):
            logits = logits.to_tuple()[0]

        input_tensor = inputs_dict[self.main_input_name]
        target = torch.randint(vocab_size, (input_tensor.shape[0], input_tensor.shape[1]), device=torch_device)
        loss = torch.nn.functional.cross_entropy(logits.view(-1, vocab_size), target.view(-1))
        loss.backward()


class DreamTransformerCompileTests(TorchCompileTesterMixin, unittest.TestCase):
    model_class = DreamTransformer1DModel
    # NOTE: set to None to skip TorchCompileTesterMixin.test_compile_on_different_shapes because this test
    # currently assumes the input is image-like (specifically, that prepare_dummy_inputs accepts `height` and `width`
    # argunments). We could consider overriding this test to make it specific to the Dream transformer.
    different_shapes_for_compilation = None

    def prepare_dummy_input(self, batch_size: int = 1, sequence_length: int = 48):
        return DreamTransformerTests().prepare_dummy_input(batch_size=batch_size, sequence_length=sequence_length)

    def prepare_init_args_and_inputs_for_common(self):
        return DreamTransformerTests().prepare_init_args_and_inputs_for_common()
