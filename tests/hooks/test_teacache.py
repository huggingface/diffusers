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
import warnings

import torch

from diffusers.hooks import HookRegistry, TeaCacheConfig


class TeaCacheConfigTests(unittest.TestCase):
    """Tests for TeaCacheConfig parameter validation."""

    def test_valid_config(self):
        """Test valid configuration is accepted."""
        config = TeaCacheConfig(rel_l1_thresh=0.2)
        self.assertEqual(config.rel_l1_thresh, 0.2)
        self.assertIsNotNone(config.coefficients)
        self.assertEqual(len(config.coefficients), 5)

    def test_invalid_type(self):
        """Test invalid type for rel_l1_thresh raises TypeError."""
        with self.assertRaises(TypeError) as context:
            TeaCacheConfig(rel_l1_thresh="invalid")
        self.assertIn("must be a number", str(context.exception))

    def test_negative_value(self):
        """Test negative threshold raises ValueError."""
        with self.assertRaises(ValueError) as context:
            TeaCacheConfig(rel_l1_thresh=-0.5)
        self.assertIn("must be positive", str(context.exception))

    def test_invalid_coefficients_length(self):
        """Test wrong coefficient count raises ValueError."""
        with self.assertRaises(ValueError) as context:
            TeaCacheConfig(rel_l1_thresh=0.2, coefficients=[1.0, 2.0, 3.0])
        self.assertIn("exactly 5 elements", str(context.exception))

    def test_invalid_coefficients_type(self):
        """Test invalid coefficient types raise TypeError."""
        with self.assertRaises(TypeError) as context:
            TeaCacheConfig(rel_l1_thresh=0.2, coefficients=[1.0, 2.0, "invalid", 4.0, 5.0])
        self.assertIn("must be numbers", str(context.exception))

    def test_warning_very_low_threshold(self):
        """Test warning is issued for very low threshold."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            TeaCacheConfig(rel_l1_thresh=0.01)
            self.assertEqual(len(w), 1)
            self.assertIn("very low", str(w[0].message))

    def test_warning_very_high_threshold(self):
        """Test warning is issued for very high threshold."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            TeaCacheConfig(rel_l1_thresh=1.5)
            self.assertEqual(len(w), 1)
            self.assertIn("very high", str(w[0].message))

    def test_config_repr(self):
        """Test __repr__ method works correctly."""
        config = TeaCacheConfig(rel_l1_thresh=0.25)
        repr_str = repr(config)
        self.assertIn("TeaCacheConfig", repr_str)
        self.assertIn("0.25", repr_str)

    def test_custom_coefficients(self):
        """Test custom coefficients are accepted."""
        custom_coeffs = [1.0, 2.0, 3.0, 4.0, 5.0]
        config = TeaCacheConfig(rel_l1_thresh=0.2, coefficients=custom_coeffs)
        self.assertEqual(config.coefficients, custom_coeffs)


class TeaCacheStateTests(unittest.TestCase):
    """Tests for TeaCacheState."""

    def test_state_initialization(self):
        """Test state initializes with correct default values."""
        from diffusers.hooks.teacache import TeaCacheState

        state = TeaCacheState()
        self.assertEqual(state.cnt, 0)
        self.assertEqual(state.num_steps, 0)
        self.assertEqual(state.accumulated_rel_l1_distance, 0.0)
        self.assertIsNone(state.previous_modulated_input)
        self.assertIsNone(state.previous_residual)

    def test_state_reset(self):
        """Test state reset clears all values."""
        from diffusers.hooks.teacache import TeaCacheState

        state = TeaCacheState()
        # Modify state
        state.cnt = 5
        state.num_steps = 10
        state.accumulated_rel_l1_distance = 0.5
        state.previous_modulated_input = torch.randn(1, 10)
        state.previous_residual = torch.randn(1, 10)

        # Reset
        state.reset()

        # Verify reset
        self.assertEqual(state.cnt, 0)
        self.assertEqual(state.num_steps, 0)
        self.assertEqual(state.accumulated_rel_l1_distance, 0.0)
        self.assertIsNone(state.previous_modulated_input)
        self.assertIsNone(state.previous_residual)

    def test_state_repr(self):
        """Test __repr__ method works correctly."""
        from diffusers.hooks.teacache import TeaCacheState

        state = TeaCacheState()
        state.cnt = 3
        state.num_steps = 10
        repr_str = repr(state)
        self.assertIn("TeaCacheState", repr_str)
        self.assertIn("cnt=3", repr_str)
        self.assertIn("num_steps=10", repr_str)


class TeaCacheHookTests(unittest.TestCase):
    """Tests for TeaCacheHook functionality."""

    def test_hook_initialization(self):
        """Test hook initializes correctly with config."""
        from diffusers.hooks.teacache import TeaCacheHook

        config = TeaCacheConfig(rel_l1_thresh=0.2)
        hook = TeaCacheHook(config)

        self.assertEqual(hook.config.rel_l1_thresh, 0.2)
        self.assertIsNotNone(hook.rescale_func)
        self.assertIsNotNone(hook.state_manager)

    def test_should_compute_full_transformer_logic(self):
        """Test _should_compute_full_transformer decision logic."""
        from diffusers.hooks.teacache import TeaCacheHook, TeaCacheState

        config = TeaCacheConfig(rel_l1_thresh=1.0, coefficients=[1, 0, 0, 0, 0])
        hook = TeaCacheHook(config)
        state = TeaCacheState()

        x0 = torch.ones(1, 4)
        x1 = torch.ones(1, 4) * 1.1

        # First step should always compute
        self.assertTrue(hook._should_compute_full_transformer(state, x0))

        state.previous_modulated_input = x0
        state.cnt = 1
        state.num_steps = 4

        # Middle step: accumulate distance and stay below threshold => reuse cache
        self.assertFalse(hook._should_compute_full_transformer(state, x1))

        # Last step: must compute regardless of distance
        state.cnt = state.num_steps - 1
        self.assertTrue(hook._should_compute_full_transformer(state, x1))

    def test_apply_teacache_with_custom_extractor(self):
        """Test apply_teacache works with custom extractor function."""
        from diffusers.hooks import apply_teacache
        from diffusers.models import CacheMixin

        class DummyModule(torch.nn.Module, CacheMixin):
            def __init__(self):
                super().__init__()
                self.dummy = torch.nn.Linear(4, 4)

        module = DummyModule()

        # Custom extractor function
        def custom_extractor(mod, hidden_states, temb):
            return hidden_states

        config = TeaCacheConfig(rel_l1_thresh=0.2, extract_modulated_input_fn=custom_extractor)

        # Should not raise - TeaCache is now model-agnostic
        apply_teacache(module, config)

        # Verify registry and disable path work
        registry = HookRegistry.check_if_exists_or_initialize(module)
        self.assertIn("teacache", registry.hooks)

        module.disable_cache()


if __name__ == "__main__":
    unittest.main()
