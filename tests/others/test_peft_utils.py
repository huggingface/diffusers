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

from diffusers.utils.peft_utils import get_peft_kwargs


class GetPeftKwargsTest(unittest.TestCase):
    """Tests for diffusers.utils.peft_utils.get_peft_kwargs."""

    def test_empty_rank_dict_raises_actionable_value_error(self):
        """Regression for huggingface/peft#3238 (failure path is in diffusers,
        not peft). When the caller's rank-discovery loop produces an empty
        `rank_dict` (typical when state_dict keys carry an extra/missing
        prefix or an adapter-name infix that the loop did not match), we
        used to crash with a cryptic IndexError on
        `list(rank_dict.values())[0]`. Now we raise a `ValueError` whose
        message names the underlying mismatch and shows a few state_dict
        keys so the user can diagnose.
        """
        peft_state_dict = {
            "encoder.layers.0.self_attn.k_proj.lora_A.default_0.weight": object(),
            "encoder.layers.0.self_attn.k_proj.lora_B.default_0.weight": object(),
            "encoder.layers.0.self_attn.q_proj.lora_A.default_0.weight": object(),
        }
        with self.assertRaises(ValueError) as cm:
            get_peft_kwargs(
                rank_dict={},
                network_alpha_dict=None,
                peft_state_dict=peft_state_dict,
                is_unet=False,
            )
        message = str(cm.exception)
        self.assertIn("`rank_dict` is empty", message)
        self.assertIn("lora_B.weight", message)
        # The message includes a sample of the state_dict so the user can spot
        # the prefix/infix mismatch from the error alone.
        self.assertIn("State dict has 3 keys", message)

    def test_empty_rank_dict_with_none_state_dict_is_safe(self):
        """The diagnostic message should not crash on a None peft_state_dict."""
        with self.assertRaises(ValueError) as cm:
            get_peft_kwargs(
                rank_dict={},
                network_alpha_dict=None,
                peft_state_dict=None,
                is_unet=True,
            )
        self.assertIn("State dict has 0 keys", str(cm.exception))

    def test_non_empty_rank_dict_unchanged(self):
        """The fast-path (rank_dict populated as before) must remain
        functionally identical. Smoke-test that get_peft_kwargs returns the
        expected keys for a minimal one-module rank_dict.
        """
        rank_dict = {"q_proj.lora_B.weight": 4}
        peft_state_dict = {
            "q_proj.lora_A.weight": object(),
            "q_proj.lora_B.weight": object(),
        }
        kwargs = get_peft_kwargs(
            rank_dict=rank_dict,
            network_alpha_dict=None,
            peft_state_dict=peft_state_dict,
            is_unet=True,
        )
        self.assertEqual(kwargs["r"], 4)
        self.assertEqual(kwargs["lora_alpha"], 4)
        self.assertEqual(kwargs["rank_pattern"], {})
        self.assertIn("q_proj", kwargs["target_modules"])
        self.assertFalse(kwargs["use_dora"])
        self.assertFalse(kwargs["lora_bias"])
