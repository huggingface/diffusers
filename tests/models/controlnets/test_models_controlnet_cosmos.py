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

from diffusers import CosmosControlNetModel

from ...testing_utils import enable_full_determinism, torch_device
from ..test_modeling_common import ModelTesterMixin


enable_full_determinism()


class CosmosControlNetModelTests(ModelTesterMixin, unittest.TestCase):
    model_class = CosmosControlNetModel
    main_input_name = "controls_latents"
    uses_custom_attn_processor = True

    @property
    def dummy_input(self):
        batch_size = 1
        num_channels = 16
        num_frames = 1
        height = 16
        width = 16
        text_embed_dim = 32
        sequence_length = 12
        img_context_dim = 32
        img_context_num_tokens = 4

        controls_latents = torch.randn((batch_size, num_channels, num_frames, height, width)).to(torch_device)
        latents = torch.randn((batch_size, num_frames * (height // 2) * (width // 2), 32)).to(torch_device)  # patchified
        condition_mask = torch.ones(batch_size, 1, num_frames, height, width).to(torch_device)
        padding_mask = torch.zeros(batch_size, 1, height, width).to(torch_device)

        # Text embeddings
        text_context = torch.randn((batch_size, sequence_length, text_embed_dim)).to(torch_device)
        # Image context for Cosmos 2.5
        img_context = torch.randn((batch_size, img_context_num_tokens, img_context_dim)).to(torch_device)
        encoder_hidden_states = (text_context, img_context)

        temb = torch.randn((batch_size, num_frames * (height // 2) * (width // 2), 32 * 3)).to(torch_device)
        embedded_timestep = torch.randn((batch_size, num_frames * (height // 2) * (width // 2), 32)).to(torch_device)

        return {
            "controls_latents": controls_latents,
            "latents": latents,
            "condition_mask": condition_mask,
            "conditioning_scale": 1.0,
            "padding_mask": padding_mask,
            "encoder_hidden_states": encoder_hidden_states,
            "temb": temb,
            "embedded_timestep": embedded_timestep,
        }

    @property
    def input_shape(self):
        return (16, 1, 16, 16)

    @property
    def output_shape(self):
        # Output is a list of control blocks - this property not directly applicable
        return None

    def prepare_init_args_and_inputs_for_common(self):
        init_dict = {
            "n_controlnet_blocks": 2,
            "in_channels": 16 + 1 + 1,  # latent_channels + condition_mask + padding_mask
            "model_channels": 32,
            "num_attention_heads": 2,
            "attention_head_dim": 16,
            "mlp_ratio": 2,
            "text_embed_dim": 32,
            "adaln_lora_dim": 4,
            "patch_size": (1, 2, 2),
            "max_size": (4, 32, 32),
            "rope_scale": (2.0, 1.0, 1.0),
        }
        inputs_dict = self.dummy_input
        return init_dict, inputs_dict

    def test_output_is_list_of_tensors(self):
        """Test that the model outputs a list of control tensors."""
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
        model = self.model_class(**init_dict)
        model.to(torch_device)
        model.eval()

        with torch.no_grad():
            output = model(**inputs_dict)

        self.assertIsInstance(output, list)
        self.assertEqual(len(output), init_dict["n_controlnet_blocks"])
        for tensor in output:
            self.assertIsInstance(tensor, torch.Tensor)

    def test_conditioning_scale_single(self):
        """Test that a single conditioning scale is broadcast to all blocks."""
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
        model = self.model_class(**init_dict)
        model.to(torch_device)
        model.eval()

        inputs_dict["conditioning_scale"] = 0.5

        with torch.no_grad():
            output = model(**inputs_dict)

        self.assertEqual(len(output), init_dict["n_controlnet_blocks"])

    def test_conditioning_scale_list(self):
        """Test that a list of conditioning scales is applied per block."""
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
        model = self.model_class(**init_dict)
        model.to(torch_device)
        model.eval()

        # Provide a scale for each block
        inputs_dict["conditioning_scale"] = [0.5, 1.0]

        with torch.no_grad():
            output = model(**inputs_dict)

        self.assertEqual(len(output), init_dict["n_controlnet_blocks"])

    def test_forward_with_none_img_context(self):
        """Test forward pass when img_context is None."""
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
        model = self.model_class(**init_dict)
        model.to(torch_device)
        model.eval()

        # Set encoder_hidden_states to (text_context, None)
        text_context = inputs_dict["encoder_hidden_states"][0]
        inputs_dict["encoder_hidden_states"] = (text_context, None)

        with torch.no_grad():
            output = model(**inputs_dict)

        self.assertIsInstance(output, list)
        self.assertEqual(len(output), init_dict["n_controlnet_blocks"])

    def test_gradient_checkpointing_is_applied(self):
        expected_set = {"CosmosControlNetModel"}
        super().test_gradient_checkpointing_is_applied(expected_set=expected_set)

    @unittest.skip("CosmosControlNetModel outputs a list, not compatible with standard output test.")
    def test_determinism(self):
        pass

    @unittest.skip("CosmosControlNetModel outputs a list, not compatible with standard output test.")
    def test_outputs_equivalence(self):
        pass

    @unittest.skip("CosmosControlNetModel uses custom attention processor.")
    def test_forward_signature(self):
        pass

    @unittest.skip("CosmosControlNetModel doesn't use standard forward output shape.")
    def test_forward_with_norm_groups(self):
        pass

    @unittest.skip("CosmosControlNetModel outputs a list, not compatible with EMA training test.")
    def test_ema_training(self):
        pass

    @unittest.skip("CosmosControlNetModel outputs a list, not compatible with standard variant test.")
    def test_model_from_pretrained_hub_subfolder(self):
        pass

    @unittest.skip("CosmosControlNetModel outputs a list, not compatible with standard variant test.")
    def test_model_from_pretrained_subfolder(self):
        pass

    @unittest.skip("CosmosControlNetModel outputs a list, not compatible with standard output test.")
    def test_from_save_pretrained(self):
        pass

    @unittest.skip("CosmosControlNetModel outputs a list, not compatible with standard output test.")
    def test_from_save_pretrained_variant(self):
        pass

    @unittest.skip("CosmosControlNetModel outputs a list, not compatible with standard output test.")
    def test_set_xformers_attn_processor_for_determinism(self):
        pass

    @unittest.skip("CosmosControlNetModel outputs a list, not compatible with standard output test.")
    def test_set_default_attn_processor(self):
        pass

    @unittest.skip("CosmosControlNetModel outputs a list, not compatible with standard output test.")
    def test_set_attn_processor_for_determinism(self):
        pass

    @unittest.skip("Layerwise casting test has compatibility issues with this model's output format.")
    def test_layerwise_casting_inference(self):
        pass

    @unittest.skip("CosmosControlNetModel outputs a list, output_shape is None.")
    def test_layerwise_casting_memory(self):
        pass

    @unittest.skip("CosmosControlNetModel outputs a list, output_shape is None.")
    def test_layerwise_casting_training(self):
        pass

    @unittest.skip("CosmosControlNetModel outputs a list, not compatible with output shape comparison.")
    def test_output(self):
        pass

    @unittest.skip("CosmosControlNetModel outputs a list, output_shape is None.")
    def test_training(self):
        pass
