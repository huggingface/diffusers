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

import numpy as np
import torch

from diffusers import MagCacheConfig, apply_mag_cache
from diffusers.hooks._helpers import TransformerBlockMetadata, TransformerBlockRegistry
from diffusers.models import ModelMixin
from diffusers.utils import logging


logger = logging.get_logger(__name__)


class DummyBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states, encoder_hidden_states=None, **kwargs):
        # Output is double input
        # This ensures Residual = 2*Input - Input = Input
        return hidden_states * 2.0


class DummyTransformer(ModelMixin):
    def __init__(self):
        super().__init__()
        self.transformer_blocks = torch.nn.ModuleList([DummyBlock(), DummyBlock()])

    def forward(self, hidden_states, encoder_hidden_states=None):
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, encoder_hidden_states=encoder_hidden_states)
        return hidden_states


class TupleOutputBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states, encoder_hidden_states=None, **kwargs):
        # Returns a tuple
        return hidden_states * 2.0, encoder_hidden_states


class TupleTransformer(ModelMixin):
    def __init__(self):
        super().__init__()
        self.transformer_blocks = torch.nn.ModuleList([TupleOutputBlock()])

    def forward(self, hidden_states, encoder_hidden_states=None):
        for block in self.transformer_blocks:
            # Emulate Flux-like behavior
            output = block(hidden_states, encoder_hidden_states=encoder_hidden_states)
            hidden_states = output[0]
            encoder_hidden_states = output[1]
        return hidden_states, encoder_hidden_states


class MagCacheTests(unittest.TestCase):
    def setUp(self):
        # Register standard dummy block
        TransformerBlockRegistry.register(
            DummyBlock,
            TransformerBlockMetadata(return_hidden_states_index=None, return_encoder_hidden_states_index=None),
        )
        # Register tuple block (Flux style)
        TransformerBlockRegistry.register(
            TupleOutputBlock,
            TransformerBlockMetadata(return_hidden_states_index=0, return_encoder_hidden_states_index=1),
        )

    def _set_context(self, model, context_name):
        """Helper to set context on all hooks in the model."""
        for module in model.modules():
            if hasattr(module, "_diffusers_hook"):
                module._diffusers_hook._set_context(context_name)

    def _get_calibration_data(self, model):
        for module in model.modules():
            if hasattr(module, "_diffusers_hook"):
                hook = module._diffusers_hook.get_hook("mag_cache_block_hook")
                if hook:
                    return hook.state_manager.get_state().calibration_ratios
        return []

    def test_mag_cache_validation(self):
        """Test that missing mag_ratios raises ValueError."""
        with self.assertRaises(ValueError):
            MagCacheConfig(num_inference_steps=10, calibrate=False)

    def test_mag_cache_skipping_logic(self):
        """
        Tests that MagCache correctly calculates residuals and skips blocks when conditions are met.
        """
        model = DummyTransformer()

        # Dummy ratios: [1.0, 1.0] implies 0 accumulated error if we skip
        ratios = np.array([1.0, 1.0])

        config = MagCacheConfig(
            threshold=100.0,
            num_inference_steps=2,
            retention_ratio=0.0,  # Enable immediate skipping
            max_skip_steps=5,
            mag_ratios=ratios,
        )

        apply_mag_cache(model, config)
        self._set_context(model, "test_context")

        # Step 0: Input 10.0 -> Output 40.0 (2 blocks * 2x each)
        # HeadInput=10. Output=40. Residual=30.
        input_t0 = torch.tensor([[[10.0]]])
        output_t0 = model(input_t0)
        self.assertTrue(torch.allclose(output_t0, torch.tensor([[[40.0]]])), "Step 0 failed")

        # Step 1: Input 11.0.
        # If Skipped: Output = Input(11) + Residual(30) = 41.0
        # If Computed: Output = 11 * 4 = 44.0
        input_t1 = torch.tensor([[[11.0]]])
        output_t1 = model(input_t1)

        self.assertTrue(
            torch.allclose(output_t1, torch.tensor([[[41.0]]])), f"Expected Skip (41.0), got {output_t1.item()}"
        )

    def test_mag_cache_retention(self):
        """Test that retention_ratio prevents skipping even if error is low."""
        model = DummyTransformer()
        # Ratios that imply 0 error, so it *would* skip if retention allowed it
        ratios = np.array([1.0, 1.0])

        config = MagCacheConfig(
            threshold=100.0,
            num_inference_steps=2,
            retention_ratio=1.0,  # Force retention for ALL steps
            mag_ratios=ratios,
        )

        apply_mag_cache(model, config)
        self._set_context(model, "test_context")

        # Step 0
        model(torch.tensor([[[10.0]]]))

        # Step 1: Should COMPUTE (44.0) not SKIP (41.0) because of retention
        input_t1 = torch.tensor([[[11.0]]])
        output_t1 = model(input_t1)

        self.assertTrue(
            torch.allclose(output_t1, torch.tensor([[[44.0]]])),
            f"Expected Compute (44.0) due to retention, got {output_t1.item()}",
        )

    def test_mag_cache_tuple_outputs(self):
        """Test compatibility with models returning (hidden, encoder_hidden) like Flux."""
        model = TupleTransformer()
        ratios = np.array([1.0, 1.0])

        config = MagCacheConfig(threshold=100.0, num_inference_steps=2, retention_ratio=0.0, mag_ratios=ratios)

        apply_mag_cache(model, config)
        self._set_context(model, "test_context")

        # Step 0: Compute. Input 10.0 -> Output 20.0 (1 block * 2x)
        # Residual = 10.0
        input_t0 = torch.tensor([[[10.0]]])
        enc_t0 = torch.tensor([[[1.0]]])
        out_0, _ = model(input_t0, encoder_hidden_states=enc_t0)
        self.assertTrue(torch.allclose(out_0, torch.tensor([[[20.0]]])))

        # Step 1: Skip. Input 11.0.
        # Skipped Output = 11 + 10 = 21.0
        input_t1 = torch.tensor([[[11.0]]])
        out_1, _ = model(input_t1, encoder_hidden_states=enc_t0)

        self.assertTrue(
            torch.allclose(out_1, torch.tensor([[[21.0]]])), f"Tuple skip failed. Expected 21.0, got {out_1.item()}"
        )

    def test_mag_cache_reset(self):
        """Test that state resets correctly after num_inference_steps."""
        model = DummyTransformer()
        config = MagCacheConfig(
            threshold=100.0, num_inference_steps=2, retention_ratio=0.0, mag_ratios=np.array([1.0, 1.0])
        )
        apply_mag_cache(model, config)
        self._set_context(model, "test_context")

        input_t = torch.ones(1, 1, 1)

        model(input_t)  # Step 0
        model(input_t)  # Step 1 (Skipped)

        # Step 2 (Reset -> Step 0) -> Should Compute
        # Input 2.0 -> Output 8.0
        input_t2 = torch.tensor([[[2.0]]])
        output_t2 = model(input_t2)

        self.assertTrue(torch.allclose(output_t2, torch.tensor([[[8.0]]])), "State did not reset correctly")

    def test_mag_cache_calibration(self):
        """Test that calibration mode records ratios."""
        model = DummyTransformer()
        config = MagCacheConfig(num_inference_steps=2, calibrate=True)
        apply_mag_cache(model, config)
        self._set_context(model, "test_context")

        # Step 0
        # HeadInput = 10. Output = 40. Residual = 30.
        # Ratio 0 is placeholder 1.0
        model(torch.tensor([[[10.0]]]))

        # Check intermediate state
        ratios = self._get_calibration_data(model)
        self.assertEqual(len(ratios), 1)
        self.assertEqual(ratios[0], 1.0)

        # Step 1
        # HeadInput = 10. Output = 40. Residual = 30.
        # PrevResidual = 30. CurrResidual = 30.
        # Ratio = 30/30 = 1.0
        model(torch.tensor([[[10.0]]]))

        # Verify it computes fully (no skip)
        # If it skipped, output would be 41.0. It should be 40.0
        # Actually in test setup, input is same (10.0) so output 40.0.
        # Let's ensure list is empty after reset (end of step 1)
        ratios_after = self._get_calibration_data(model)
        self.assertEqual(ratios_after, [])
