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

import unittest

import torch

from diffusers.models.transformers.transformer_mirage import MirageTransformer2DModel, MirageParams

from ...testing_utils import enable_full_determinism, torch_device
from ..test_modeling_common import ModelTesterMixin


enable_full_determinism()


class MirageTransformerTests(ModelTesterMixin, unittest.TestCase):
    model_class = MirageTransformer2DModel
    main_input_name = "image_latent"

    @property
    def dummy_input(self):
        return self.prepare_dummy_input()

    @property
    def input_shape(self):
        return (16, 4, 4)

    @property
    def output_shape(self):
        return (16, 4, 4)

    def prepare_dummy_input(self, height=32, width=32):
        batch_size = 1
        num_latent_channels = 16
        sequence_length = 16
        embedding_dim = 1792

        image_latent = torch.randn((batch_size, num_latent_channels, height, width)).to(torch_device)
        cross_attn_conditioning = torch.randn((batch_size, sequence_length, embedding_dim)).to(torch_device)
        micro_conditioning = torch.randn((batch_size, embedding_dim)).to(torch_device)
        timestep = torch.tensor([1.0]).to(torch_device).expand(batch_size)

        return {
            "image_latent": image_latent,
            "timestep": timestep,
            "cross_attn_conditioning": cross_attn_conditioning,
            "micro_conditioning": micro_conditioning,
        }

    def prepare_init_args_and_inputs_for_common(self):
        init_dict = {
            "in_channels": 16,
            "patch_size": 2,
            "context_in_dim": 1792,
            "hidden_size": 1792,
            "mlp_ratio": 3.5,
            "num_heads": 28,
            "depth": 4,  # Smaller depth for testing
            "axes_dim": [32, 32],
            "theta": 10_000,
        }
        inputs_dict = self.prepare_dummy_input()
        return init_dict, inputs_dict

    def test_forward_signature(self):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()

        model = self.model_class(**init_dict)
        model.to(torch_device)
        model.eval()

        with torch.no_grad():
            # Test forward
            outputs = model(**inputs_dict)

        self.assertIsNotNone(outputs)
        expected_shape = inputs_dict["image_latent"].shape
        self.assertEqual(outputs.shape, expected_shape)

    def test_mirage_params_initialization(self):
        # Test model initialization
        model = MirageTransformer2DModel(
            in_channels=16,
            patch_size=2,
            context_in_dim=1792,
            hidden_size=1792,
            mlp_ratio=3.5,
            num_heads=28,
            depth=4,
            axes_dim=[32, 32],
            theta=10_000,
        )
        self.assertEqual(model.config.in_channels, 16)
        self.assertEqual(model.config.hidden_size, 1792)
        self.assertEqual(model.config.num_heads, 28)

    def test_model_with_dict_config(self):
        # Test model initialization with from_config
        config_dict = {
            "in_channels": 16,
            "patch_size": 2,
            "context_in_dim": 1792,
            "hidden_size": 1792,
            "mlp_ratio": 3.5,
            "num_heads": 28,
            "depth": 4,
            "axes_dim": [32, 32],
            "theta": 10_000,
        }

        model = MirageTransformer2DModel.from_config(config_dict)
        self.assertEqual(model.config.in_channels, 16)
        self.assertEqual(model.config.hidden_size, 1792)

    def test_process_inputs(self):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
        model = self.model_class(**init_dict)
        model.to(torch_device)
        model.eval()

        with torch.no_grad():
            img_seq, txt, pe = model.process_inputs(
                inputs_dict["image_latent"],
                inputs_dict["cross_attn_conditioning"]
            )

        # Check shapes
        batch_size = inputs_dict["image_latent"].shape[0]
        height, width = inputs_dict["image_latent"].shape[2:]
        patch_size = init_dict["patch_size"]
        expected_seq_len = (height // patch_size) * (width // patch_size)

        self.assertEqual(img_seq.shape, (batch_size, expected_seq_len, init_dict["in_channels"] * patch_size**2))
        self.assertEqual(txt.shape, (batch_size, inputs_dict["cross_attn_conditioning"].shape[1], init_dict["hidden_size"]))
        # Check that pe has the correct batch size, sequence length and some embedding dimension
        self.assertEqual(pe.shape[0], batch_size)  # batch size
        self.assertEqual(pe.shape[1], 1)  # unsqueeze(1) in EmbedND
        self.assertEqual(pe.shape[2], expected_seq_len)  # sequence length
        self.assertEqual(pe.shape[-2:], (2, 2))  # rope rearrange output

    def test_forward_transformers(self):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
        model = self.model_class(**init_dict)
        model.to(torch_device)
        model.eval()

        with torch.no_grad():
            # Process inputs first
            img_seq, txt, pe = model.process_inputs(
                inputs_dict["image_latent"],
                inputs_dict["cross_attn_conditioning"]
            )

            # Test forward_transformers
            output_seq = model.forward_transformers(
                img_seq,
                txt,
                timestep=inputs_dict["timestep"],
                pe=pe
            )

        # Check output shape
        expected_out_channels = init_dict["in_channels"] * init_dict["patch_size"]**2
        self.assertEqual(output_seq.shape, (img_seq.shape[0], img_seq.shape[1], expected_out_channels))

    def test_attention_mask(self):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
        model = self.model_class(**init_dict)
        model.to(torch_device)
        model.eval()

        # Create attention mask
        batch_size = inputs_dict["cross_attn_conditioning"].shape[0]
        seq_len = inputs_dict["cross_attn_conditioning"].shape[1]
        attention_mask = torch.ones((batch_size, seq_len), dtype=torch.bool).to(torch_device)
        attention_mask[:, seq_len//2:] = False  # Mask second half

        with torch.no_grad():
            outputs = model(
                **inputs_dict,
                cross_attn_mask=attention_mask
            )

        self.assertIsNotNone(outputs)
        expected_shape = inputs_dict["image_latent"].shape
        self.assertEqual(outputs.shape, expected_shape)

    def test_invalid_config(self):
        # Test invalid configuration - hidden_size not divisible by num_heads
        with self.assertRaises(ValueError):
            MirageTransformer2DModel(
                in_channels=16,
                patch_size=2,
                context_in_dim=1792,
                hidden_size=1793,  # Not divisible by 28
                mlp_ratio=3.5,
                num_heads=28,
                depth=4,
                axes_dim=[32, 32],
                theta=10_000,
            )

        # Test invalid axes_dim that doesn't sum to pe_dim
        with self.assertRaises(ValueError):
            MirageTransformer2DModel(
                in_channels=16,
                patch_size=2,
                context_in_dim=1792,
                hidden_size=1792,
                mlp_ratio=3.5,
                num_heads=28,
                depth=4,
                axes_dim=[30, 30],  # Sum = 60, but pe_dim = 1792/28 = 64
                theta=10_000,
            )

    def test_gradient_checkpointing_enable(self):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
        model = self.model_class(**init_dict)
        model.to(torch_device)

        # Enable gradient checkpointing
        model.enable_gradient_checkpointing()

        # Check that _activation_checkpointing is set
        for block in model.blocks:
            self.assertTrue(hasattr(block, '_activation_checkpointing'))

    def test_from_config(self):
        init_dict, _ = self.prepare_init_args_and_inputs_for_common()

        # Create model from config
        model = self.model_class.from_config(init_dict)
        self.assertIsInstance(model, self.model_class)
        self.assertEqual(model.config.in_channels, init_dict["in_channels"])


if __name__ == "__main__":
    unittest.main()