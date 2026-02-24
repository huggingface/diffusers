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
from diffusers.models.controlnets.controlnet_cosmos import CosmosControlNetOutput

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
        img_context_dim_in = 32
        img_context_num_tokens = 4

        # Raw latents (not patchified) - the controlnet computes embeddings internally
        controls_latents = torch.randn((batch_size, num_channels, num_frames, height, width)).to(torch_device)
        latents = torch.randn((batch_size, num_channels, num_frames, height, width)).to(torch_device)
        timestep = torch.tensor([0.5]).to(torch_device)  # Diffusion timestep
        condition_mask = torch.ones(batch_size, 1, num_frames, height, width).to(torch_device)
        padding_mask = torch.zeros(batch_size, 1, height, width).to(torch_device)

        # Text embeddings
        text_context = torch.randn((batch_size, sequence_length, text_embed_dim)).to(torch_device)
        # Image context for Cosmos 2.5
        img_context = torch.randn((batch_size, img_context_num_tokens, img_context_dim_in)).to(torch_device)
        encoder_hidden_states = (text_context, img_context)

        return {
            "controls_latents": controls_latents,
            "latents": latents,
            "timestep": timestep,
            "encoder_hidden_states": encoder_hidden_states,
            "condition_mask": condition_mask,
            "conditioning_scale": 1.0,
            "padding_mask": padding_mask,
        }

    @property
    def input_shape(self):
        return (16, 1, 16, 16)

    @property
    def output_shape(self):
        # Output is tuple of n_controlnet_blocks tensors, each with shape (batch, num_patches, model_channels)
        # After stacking by normalize_output: (n_blocks, batch, num_patches, model_channels)
        # For test config: n_blocks=2, num_patches=64 (1*8*8), model_channels=32
        # output_shape is used as (batch_size,) + output_shape, so: (2, 64, 32)
        return (2, 64, 32)

    def prepare_init_args_and_inputs_for_common(self):
        init_dict = {
            "n_controlnet_blocks": 2,
            "in_channels": 16 + 1 + 1,  # control_latent_channels + condition_mask + padding_mask
            "latent_channels": 16 + 1 + 1,  # base_latent_channels (16) + condition_mask (1) + padding_mask (1) = 18
            "model_channels": 32,
            "num_attention_heads": 2,
            "attention_head_dim": 16,
            "mlp_ratio": 2,
            "text_embed_dim": 32,
            "adaln_lora_dim": 4,
            "patch_size": (1, 2, 2),
            "max_size": (4, 32, 32),
            "rope_scale": (2.0, 1.0, 1.0),
            "extra_pos_embed_type": None,
            "img_context_dim_in": 32,
            "img_context_dim_out": 32,
            "use_crossattn_projection": False,  # Test doesn't need this projection
        }
        inputs_dict = self.dummy_input
        return init_dict, inputs_dict

    def test_output_format(self):
        """Test that the model outputs CosmosControlNetOutput with correct structure."""
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
        model = self.model_class(**init_dict)
        model.to(torch_device)
        model.eval()

        with torch.no_grad():
            output = model(**inputs_dict)

        self.assertIsInstance(output, CosmosControlNetOutput)
        self.assertIsInstance(output.control_block_samples, list)
        self.assertEqual(len(output.control_block_samples), init_dict["n_controlnet_blocks"])
        for tensor in output.control_block_samples:
            self.assertIsInstance(tensor, torch.Tensor)

    def test_output_list_format(self):
        """Test that return_dict=False returns a tuple containing a list."""
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
        model = self.model_class(**init_dict)
        model.to(torch_device)
        model.eval()

        with torch.no_grad():
            output = model(**inputs_dict, return_dict=False)

        self.assertIsInstance(output, tuple)
        self.assertEqual(len(output), 1)
        self.assertIsInstance(output[0], list)
        self.assertEqual(len(output[0]), init_dict["n_controlnet_blocks"])

    def test_conditioning_scale_single(self):
        """Test that a single conditioning scale is broadcast to all blocks."""
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
        model = self.model_class(**init_dict)
        model.to(torch_device)
        model.eval()

        inputs_dict["conditioning_scale"] = 0.5

        with torch.no_grad():
            output = model(**inputs_dict)

        self.assertEqual(len(output.control_block_samples), init_dict["n_controlnet_blocks"])

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

        self.assertEqual(len(output.control_block_samples), init_dict["n_controlnet_blocks"])

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

        self.assertIsInstance(output, CosmosControlNetOutput)
        self.assertEqual(len(output.control_block_samples), init_dict["n_controlnet_blocks"])

    def test_forward_without_img_context_proj(self):
        """Test forward pass when img_context_proj is not configured."""
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
        # Disable img_context_proj
        init_dict["img_context_dim_in"] = None
        model = self.model_class(**init_dict)
        model.to(torch_device)
        model.eval()

        # When img_context is disabled, pass only text context (not a tuple)
        text_context = inputs_dict["encoder_hidden_states"][0]
        inputs_dict["encoder_hidden_states"] = text_context

        with torch.no_grad():
            output = model(**inputs_dict)

        self.assertIsInstance(output, CosmosControlNetOutput)
        self.assertEqual(len(output.control_block_samples), init_dict["n_controlnet_blocks"])

    def test_gradient_checkpointing_is_applied(self):
        expected_set = {"CosmosControlNetModel"}
        super().test_gradient_checkpointing_is_applied(expected_set=expected_set)

    # Note: test_set_attn_processor_for_determinism already handles uses_custom_attn_processor=True
    # so no explicit skip needed for it
    # Note: test_forward_signature and test_set_default_attn_processor don't exist in base class

    # Skip tests that don't apply to this architecture
    @unittest.skip("CosmosControlNetModel doesn't use norm groups.")
    def test_forward_with_norm_groups(self):
        pass

    # Skip tests that expect .sample attribute - ControlNets don't have this
    @unittest.skip("ControlNet output doesn't have .sample attribute")
    def test_effective_gradient_checkpointing(self):
        pass

    # Skip tests that compute MSE loss against single tensor output
    @unittest.skip("ControlNet outputs list of control blocks, not single tensor for MSE loss")
    def test_ema_training(self):
        pass

    @unittest.skip("ControlNet outputs list of control blocks, not single tensor for MSE loss")
    def test_training(self):
        pass

    # Skip tests where output shape comparison doesn't apply to ControlNets
    @unittest.skip("ControlNet output shape doesn't match input shape by design")
    def test_output(self):
        pass

    # Skip outputs_equivalence - dict/list comparison logic not compatible (recursive_check expects dict.values())
    @unittest.skip("ControlNet output structure not compatible with recursive dict check")
    def test_outputs_equivalence(self):
        pass

    # Skip model parallelism - base test uses torch.allclose(base_output[0], new_output[0]) which fails
    # because output[0] is the list of control_block_samples, not a tensor
    @unittest.skip("test_model_parallelism uses torch.allclose on output[0] which is a list, not a tensor")
    def test_model_parallelism(self):
        pass

    # Skip layerwise casting tests - these have two issues:
    # 1. _inference and _memory: dtype compatibility issues with learnable_pos_embed and float8/bfloat16
    # 2. _training: same as test_training - mse_loss expects tensor, not list
    @unittest.skip("Layerwise casting has dtype issues with learnable_pos_embed")
    def test_layerwise_casting_inference(self):
        pass

    @unittest.skip("Layerwise casting has dtype issues with learnable_pos_embed")
    def test_layerwise_casting_memory(self):
        pass

    @unittest.skip("test_layerwise_casting_training computes mse_loss on list output")
    def test_layerwise_casting_training(self):
        pass
