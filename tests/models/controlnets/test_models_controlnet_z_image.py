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

from diffusers.models.controlnets.controlnet_z_image import ZImageControlNetModel
from diffusers.models.transformers.transformer_z_image import ZImageTransformer2DModel


class ZImageControlNetFromTransformerTests(unittest.TestCase):
    """Tests for ZImageControlNetModel.from_transformer weight independence.

    Verifies that from_transformer creates independent copies of weights,
    so modifying the controlnet does not affect the original transformer.
    Regression test for https://github.com/huggingface/diffusers/issues/13077
    """

    def get_transformer_config(self):
        return {
            "all_patch_size": (2,),
            "all_f_patch_size": (1,),
            "in_channels": 16,
            "dim": 256,
            "n_layers": 2,
            "n_refiner_layers": 2,
            "n_heads": 8,
            "n_kv_heads": 8,
            "cap_feat_dim": 256,
            "axes_dims": [8, 12, 12],
            "axes_lens": [64, 64, 64],
        }

    def get_controlnet_config(self):
        return {
            "control_layers_places": [0, 1],
            "control_refiner_layers_places": [0, 1],
            "add_control_noise_refiner": "control_noise_refiner",
            "control_in_dim": 16,
            "dim": 256,
            "n_refiner_layers": 2,
            "n_heads": 8,
            "n_kv_heads": 8,
        }

    def test_t_embedder_independence(self):
        """Modifying controlnet.t_embedder should not affect transformer.t_embedder."""
        transformer = ZImageTransformer2DModel(**self.get_transformer_config())
        controlnet = ZImageControlNetModel(**self.get_controlnet_config())
        controlnet = ZImageControlNetModel.from_transformer(controlnet=controlnet, transformer=transformer)

        original = transformer.t_embedder.mlp[0].weight.clone()
        torch.nn.init.constant_(controlnet.t_embedder.mlp[0].weight, 42.0)

        self.assertTrue(
            torch.equal(transformer.t_embedder.mlp[0].weight, original),
            "Transformer t_embedder weights were corrupted by controlnet modification",
        )

    def test_cap_embedder_independence(self):
        """Modifying controlnet.cap_embedder should not affect transformer.cap_embedder."""
        transformer = ZImageTransformer2DModel(**self.get_transformer_config())
        controlnet = ZImageControlNetModel(**self.get_controlnet_config())
        controlnet = ZImageControlNetModel.from_transformer(controlnet=controlnet, transformer=transformer)

        original = transformer.cap_embedder[1].weight.clone()
        torch.nn.init.constant_(controlnet.cap_embedder[1].weight, 42.0)

        self.assertTrue(
            torch.equal(transformer.cap_embedder[1].weight, original),
            "Transformer cap_embedder weights were corrupted by controlnet modification",
        )

    def test_all_x_embedder_independence(self):
        """Modifying controlnet.all_x_embedder should not affect transformer.all_x_embedder."""
        transformer = ZImageTransformer2DModel(**self.get_transformer_config())
        controlnet = ZImageControlNetModel(**self.get_controlnet_config())
        controlnet = ZImageControlNetModel.from_transformer(controlnet=controlnet, transformer=transformer)

        first_key = list(transformer.all_x_embedder.keys())[0]
        original = transformer.all_x_embedder[first_key].weight.clone()
        torch.nn.init.constant_(controlnet.all_x_embedder[first_key].weight, 42.0)

        self.assertTrue(
            torch.equal(transformer.all_x_embedder[first_key].weight, original),
            "Transformer all_x_embedder weights were corrupted by controlnet modification",
        )

    def test_noise_refiner_independence(self):
        """Modifying controlnet.noise_refiner should not affect transformer.noise_refiner."""
        transformer = ZImageTransformer2DModel(**self.get_transformer_config())
        controlnet = ZImageControlNetModel(**self.get_controlnet_config())
        controlnet = ZImageControlNetModel.from_transformer(controlnet=controlnet, transformer=transformer)

        original = list(transformer.noise_refiner.parameters())[0].clone()
        torch.nn.init.constant_(list(controlnet.noise_refiner.parameters())[0], 42.0)

        self.assertTrue(
            torch.equal(list(transformer.noise_refiner.parameters())[0], original),
            "Transformer noise_refiner weights were corrupted by controlnet modification",
        )

    def test_context_refiner_independence(self):
        """Modifying controlnet.context_refiner should not affect transformer.context_refiner."""
        transformer = ZImageTransformer2DModel(**self.get_transformer_config())
        controlnet = ZImageControlNetModel(**self.get_controlnet_config())
        controlnet = ZImageControlNetModel.from_transformer(controlnet=controlnet, transformer=transformer)

        original = list(transformer.context_refiner.parameters())[0].clone()
        torch.nn.init.constant_(list(controlnet.context_refiner.parameters())[0], 42.0)

        self.assertTrue(
            torch.equal(list(transformer.context_refiner.parameters())[0], original),
            "Transformer context_refiner weights were corrupted by controlnet modification",
        )

    def test_x_pad_token_independence(self):
        """Modifying controlnet.x_pad_token should not affect transformer.x_pad_token."""
        transformer = ZImageTransformer2DModel(**self.get_transformer_config())
        controlnet = ZImageControlNetModel(**self.get_controlnet_config())
        controlnet = ZImageControlNetModel.from_transformer(controlnet=controlnet, transformer=transformer)

        original = transformer.x_pad_token.data.clone()
        controlnet.x_pad_token.data.fill_(99.0)

        self.assertTrue(
            torch.equal(transformer.x_pad_token.data, original),
            "Transformer x_pad_token was corrupted by controlnet modification",
        )

    def test_cap_pad_token_independence(self):
        """Modifying controlnet.cap_pad_token should not affect transformer.cap_pad_token."""
        transformer = ZImageTransformer2DModel(**self.get_transformer_config())
        controlnet = ZImageControlNetModel(**self.get_controlnet_config())
        controlnet = ZImageControlNetModel.from_transformer(controlnet=controlnet, transformer=transformer)

        original = transformer.cap_pad_token.data.clone()
        controlnet.cap_pad_token.data.fill_(99.0)

        self.assertTrue(
            torch.equal(transformer.cap_pad_token.data, original),
            "Transformer cap_pad_token was corrupted by controlnet modification",
        )

    def test_rope_embedder_independence(self):
        """Controlnet.rope_embedder should be a different instance from transformer.rope_embedder."""
        transformer = ZImageTransformer2DModel(**self.get_transformer_config())
        controlnet = ZImageControlNetModel(**self.get_controlnet_config())
        controlnet = ZImageControlNetModel.from_transformer(controlnet=controlnet, transformer=transformer)

        self.assertIsNot(
            controlnet.rope_embedder,
            transformer.rope_embedder,
            "Controlnet and transformer share the same rope_embedder instance",
        )

    def test_weights_correctly_copied(self):
        """Verify that weights are correctly copied from transformer to controlnet."""
        transformer = ZImageTransformer2DModel(**self.get_transformer_config())
        controlnet = ZImageControlNetModel(**self.get_controlnet_config())
        controlnet = ZImageControlNetModel.from_transformer(controlnet=controlnet, transformer=transformer)

        modules_to_check = ["t_embedder", "all_x_embedder", "cap_embedder", "noise_refiner", "context_refiner"]

        for name in modules_to_check:
            t_sd = getattr(transformer, name).state_dict()
            c_sd = getattr(controlnet, name).state_dict()
            for key in t_sd:
                self.assertTrue(
                    torch.equal(t_sd[key], c_sd[key]),
                    f"Weights not correctly copied for {name}.{key}",
                )

    def test_t_scale_correctly_copied(self):
        """Verify that t_scale is correctly copied from transformer config."""
        transformer = ZImageTransformer2DModel(**self.get_transformer_config())
        controlnet = ZImageControlNetModel(**self.get_controlnet_config())
        controlnet = ZImageControlNetModel.from_transformer(controlnet=controlnet, transformer=transformer)

        self.assertEqual(
            controlnet.t_scale,
            transformer.config.t_scale,
            "t_scale not correctly copied from transformer config",
        )
