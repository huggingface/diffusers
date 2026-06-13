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

from diffusers import ZImageControlNetModel, ZImageTransformer2DModel


def _get_tiny_transformer():
    return ZImageTransformer2DModel(
        all_patch_size=(2,),
        all_f_patch_size=(1,),
        in_channels=16,
        dim=16,
        n_layers=1,
        n_refiner_layers=1,
        n_heads=1,
        n_kv_heads=2,
        qk_norm=True,
        cap_feat_dim=16,
        rope_theta=256.0,
        t_scale=1000.0,
        axes_dims=[8, 4, 4],
        axes_lens=[256, 32, 32],
    )


def _get_tiny_controlnet():
    return ZImageControlNetModel(
        control_layers_places=[0],
        control_refiner_layers_places=[0],
        control_in_dim=16,
        all_patch_size=(2,),
        all_f_patch_size=(1,),
        dim=16,
        n_refiner_layers=1,
        n_heads=1,
        n_kv_heads=2,
        qk_norm=True,
    )


class ZImageControlNetModelTests(unittest.TestCase):
    def test_from_transformer_copies_weights_without_sharing(self):
        torch.manual_seed(0)
        transformer = _get_tiny_transformer()
        # Z-Image initializes a few tensors with ``torch.empty``; fill them with finite values
        # so the mutation checks below are not thrown off by uninitialized memory.
        with torch.no_grad():
            for p in transformer.parameters():
                p.normal_()

        controlnet = ZImageControlNetModel.from_transformer(_get_tiny_controlnet(), transformer)

        # The components carried over from the transformer must be independent objects, not shared
        # references -- otherwise casting or optimizing the control net silently mutates the
        # transformer too.
        self.assertIsNot(controlnet.t_embedder, transformer.t_embedder)
        self.assertIsNot(controlnet.all_x_embedder, transformer.all_x_embedder)
        self.assertIsNot(controlnet.cap_embedder, transformer.cap_embedder)
        self.assertIsNot(controlnet.noise_refiner, transformer.noise_refiner)
        self.assertIsNot(controlnet.context_refiner, transformer.context_refiner)
        self.assertIsNot(controlnet.x_pad_token, transformer.x_pad_token)
        self.assertIsNot(controlnet.cap_pad_token, transformer.cap_pad_token)

        # The carried-over weights must start out identical to the transformer's (a copy, not a re-init).
        for (_, copied), (_, original) in zip(
            controlnet.noise_refiner.named_parameters(), transformer.noise_refiner.named_parameters()
        ):
            self.assertTrue(torch.equal(copied, original))

        # Concretely: mutating every control-net parameter must leave the transformer untouched.
        reference = {name: param.detach().clone() for name, param in transformer.named_parameters()}
        with torch.no_grad():
            for param in controlnet.parameters():
                param.add_(1.0)
        for name, param in transformer.named_parameters():
            self.assertTrue(torch.equal(reference[name], param), f"transformer parameter '{name}' was mutated")


if __name__ == "__main__":
    unittest.main()
