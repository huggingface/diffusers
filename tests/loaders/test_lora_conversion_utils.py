# Copyright 2026 The HuggingFace Team. All rights reserved.
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

from diffusers.loaders.lora_conversion_utils import _convert_non_diffusers_anima_lora_to_diffusers


class AnimaLoraConversionUtilsTests(unittest.TestCase):
    def test_comfy_lora_alpha_is_folded_into_converted_weights(self):
        state_dict = {
            "diffusion_model.blocks.0.cross_attn.k_proj.lora_down.weight": torch.ones(4, 8),
            "diffusion_model.blocks.0.cross_attn.k_proj.lora_up.weight": torch.ones(8, 4),
            "diffusion_model.blocks.0.cross_attn.k_proj.alpha": torch.tensor(2.0),
        }

        converted_state_dict = _convert_non_diffusers_anima_lora_to_diffusers(state_dict)

        down_key = "transformer.transformer_blocks.0.attn2.to_k.lora_A.weight"
        up_key = "transformer.transformer_blocks.0.attn2.to_k.lora_B.weight"
        self.assertEqual(set(converted_state_dict), {down_key, up_key})
        self.assertTrue(torch.allclose(converted_state_dict[down_key], torch.full((4, 8), 0.5)))
        self.assertTrue(torch.allclose(converted_state_dict[up_key], torch.ones(8, 4)))

    def test_comfy_diffusers_style_lora_alpha_is_removed(self):
        state_dict = {
            "diffusion_model.blocks.0.cross_attn.k_proj.lora_A.weight": torch.ones(4, 8),
            "diffusion_model.blocks.0.cross_attn.k_proj.lora_B.weight": torch.ones(8, 4),
            "diffusion_model.blocks.0.cross_attn.k_proj.alpha": torch.tensor(2.0),
        }

        converted_state_dict = _convert_non_diffusers_anima_lora_to_diffusers(state_dict)

        down_key = "transformer.transformer_blocks.0.attn2.to_k.lora_A.weight"
        up_key = "transformer.transformer_blocks.0.attn2.to_k.lora_B.weight"
        self.assertEqual(set(converted_state_dict), {down_key, up_key})
        self.assertTrue(torch.allclose(converted_state_dict[down_key], torch.full((4, 8), 0.5)))
        self.assertTrue(torch.allclose(converted_state_dict[up_key], torch.ones(8, 4)))


if __name__ == "__main__":
    unittest.main()
