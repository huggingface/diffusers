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

from diffusers import MagCacheConfig, apply_mag_cache
from diffusers.hooks._helpers import TransformerBlockMetadata, TransformerBlockRegistry
from diffusers.models import ModelMixin
from diffusers.utils import logging


logger = logging.get_logger(__name__)

class DummyBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states, encoder_hidden_states=None, **kwargs):
        return hidden_states * 2.0

class DummyTransformer(ModelMixin):
    def __init__(self):
        super().__init__()
        self.transformer_blocks = torch.nn.ModuleList([DummyBlock(), DummyBlock()])

    def forward(self, hidden_states, encoder_hidden_states=None):
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, encoder_hidden_states=encoder_hidden_states)
        return hidden_states

class MagCacheTests(unittest.TestCase):
    def setUp(self):
        TransformerBlockRegistry.register(
            DummyBlock,
            TransformerBlockMetadata(
                return_hidden_states_index=None,
                return_encoder_hidden_states_index=None
            )
        )

    def _set_context(self, model, context_name):
        """Helper to set context on all hooks in the model."""
        for module in model.modules():
            if hasattr(module, "_diffusers_hook"):
                module._diffusers_hook._set_context(context_name)

    def test_mag_cache_skipping_logic(self):
        """
        Tests that MagCache correctly calculates residuals and skips blocks when conditions are met.
        """
        model = DummyTransformer()

        # Config:
        # num_inference_steps=2
        # retention_ratio=0.0 (Allow skipping immediately)
        # threshold=100.0 (Always skip if residual exists)
        config = MagCacheConfig(
            threshold=100.0,
            num_inference_steps=2,
            retention_ratio=0.0,
            max_skip_steps=5
        )

        # Apply Hook
        apply_mag_cache(model, config)

        # Set Context
        self._set_context(model, "test_context")

        # First run (Cannot skip, calculates residual)
        # Input: 10.0
        # Expected Output: 10 * 2 (Block 0) * 2 (Block 1) = 40.0
        input_t0 = torch.tensor([[[10.0]]])
        output_t0 = model(input_t0)

        self.assertTrue(torch.allclose(output_t0, torch.tensor([[[40.0]]])), "Step 0 computation failed")

        # Second run (Should SKIP based on config)
        # Input: 11.0
        # If Computed: 11 * 2 * 2 = 44.0
        # If Skipped: Input + Previous_Residual (30.0) = 11.0 + 30.0 = 41.0

        input_t1 = torch.tensor([[[11.0]]])
        output_t1 = model(input_t1)

        # Assert we got the SKIPPED result (41.0)
        self.assertTrue(
            torch.allclose(output_t1, torch.tensor([[[41.0]]])),
            f"MagCache failed to skip. Expected 41.0 (Cached), got {output_t1.item()} (Computed?)"
        )

    def test_mag_cache_reset(self):
        """Test that state resets correctly."""
        model = DummyTransformer()
        config = MagCacheConfig(threshold=100.0, num_inference_steps=2, retention_ratio=0.0)
        apply_mag_cache(model, config)
        self._set_context(model, "test_context")

        input_t = torch.ones(1, 1, 1)

        model(input_t)

        model(input_t)

        input_t2 = torch.tensor([[[2.0]]])
        output_t2 = model(input_t2)

        # Expected Compute: 2 * 2 * 2 = 8.0
        self.assertTrue(
            torch.allclose(output_t2, torch.tensor([[[8.0]]])),
            "MagCache did not reset loop correctly; might have applied stale residual."
        )

    def test_mag_cache_structure_validation(self):
        """Test that apply_mag_cache handles models without appropriate blocks gracefully."""
        class EmptyModel(torch.nn.Module):
            def forward(self, x): return x

        model = EmptyModel()
        apply_mag_cache(model, MagCacheConfig()) # Should not raise error
