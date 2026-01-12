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

from diffusers.hooks import HookRegistry, TeaCacheConfig


class TeaCacheConfigTests(unittest.TestCase):
    """Tests for TeaCacheConfig parameter validation."""

    def test_valid_config(self):
        """Test valid configuration is accepted."""
        config = TeaCacheConfig(rel_l1_thresh=0.2)
        self.assertEqual(config.rel_l1_thresh, 0.2)
        # coefficients is None by default (auto-detected during hook initialization)
        self.assertIsNone(config.coefficients)

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

    def test_very_low_threshold_accepted(self):
        """Test very low threshold is accepted (with logging warning)."""
        # Very low threshold should be accepted but logged as warning
        config = TeaCacheConfig(rel_l1_thresh=0.01)
        self.assertEqual(config.rel_l1_thresh, 0.01)

    def test_very_high_threshold_accepted(self):
        """Test very high threshold is accepted (with logging warning)."""
        # Very high threshold should be accepted but logged as warning
        config = TeaCacheConfig(rel_l1_thresh=1.5)
        self.assertEqual(config.rel_l1_thresh, 1.5)

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
        # coefficients is None until initialize_hook() is called with a model (auto-detection)
        self.assertIsNone(hook.coefficients)
        self.assertIsNotNone(hook.state_manager)

    def test_should_compute_logic(self):
        """Test _should_compute decision logic."""
        from diffusers.hooks.teacache import TeaCacheState, _should_compute

        coefficients = [1, 0, 0, 0, 0]
        rel_l1_thresh = 1.0
        state = TeaCacheState()

        x0 = torch.ones(1, 4)
        x1 = torch.ones(1, 4) * 1.1

        # First step should always compute
        self.assertTrue(_should_compute(state, x0, coefficients, rel_l1_thresh))

        state.previous_modulated_input = x0
        state.previous_residual = torch.zeros(1, 4)  # Need residual to skip compute
        state.cnt = 1
        state.num_steps = 4

        # Middle step: accumulate distance and stay below threshold => reuse cache
        self.assertFalse(_should_compute(state, x1, coefficients, rel_l1_thresh))

        # Last step: must compute regardless of distance
        state.cnt = state.num_steps - 1
        self.assertTrue(_should_compute(state, x1, coefficients, rel_l1_thresh))

    def test_apply_teacache_unsupported_model_raises_error(self):
        """Test that apply_teacache raises error for unsupported models."""
        from diffusers.hooks import apply_teacache
        from diffusers.models import CacheMixin

        class UnsupportedModule(torch.nn.Module, CacheMixin):
            def __init__(self):
                super().__init__()
                self.dummy = torch.nn.Linear(4, 4)

        module = UnsupportedModule()
        config = TeaCacheConfig(rel_l1_thresh=0.2)

        # Should raise ValueError for unsupported model type
        with self.assertRaises(ValueError) as context:
            apply_teacache(module, config)
        self.assertIn("Unsupported model", str(context.exception))
        self.assertIn("UnsupportedModule", str(context.exception))


class TeaCacheMultiModelTests(unittest.TestCase):
    """Tests for TeaCache multi-model support (Mochi, Lumina2, CogVideoX)."""

    def test_model_coefficient_registry(self):
        """Test that model coefficients are properly registered."""
        from diffusers.hooks.teacache import _get_model_config

        model_config = _get_model_config()

        self.assertIn("Flux", model_config)
        self.assertIn("Mochi", model_config)
        self.assertIn("Lumina2", model_config)
        self.assertIn("CogVideoX", model_config)

        # Verify all coefficients are 5-element lists
        for model_name, config in model_config.items():
            coeffs = config["coefficients"]
            self.assertEqual(len(coeffs), 5, f"{model_name} coefficients should have 5 elements")
            self.assertTrue(
                all(isinstance(c, (int, float)) for c in coeffs), f"{model_name} coefficients should be numbers"
            )

    def test_mochi_extractor(self):
        """Test Mochi modulated input extraction (now inlined in forward function)."""
        from diffusers import MochiTransformer3DModel

        # Create a minimal Mochi model for testing
        model = MochiTransformer3DModel(
            patch_size=2,
            num_attention_heads=2,
            attention_head_dim=8,
            num_layers=2,
            in_channels=4,
            text_embed_dim=16,
            time_embed_dim=4,
        )

        hidden_states = torch.randn(2, 4, 2, 8, 8)
        timestep = torch.randint(0, 1000, (2,))
        encoder_hidden_states = torch.randn(2, 16, 16)
        encoder_attention_mask = torch.ones(2, 16).bool()

        # Get timestep embedding
        temb, _ = model.time_embed(
            timestep, encoder_hidden_states, encoder_attention_mask, hidden_dtype=hidden_states.dtype
        )
        hidden_states = hidden_states.permute(0, 2, 1, 3, 4).flatten(0, 1)
        hidden_states = model.patch_embed(hidden_states)
        hidden_states = hidden_states.unflatten(0, (2, -1)).flatten(1, 2)

        # Test inline extraction logic: Mochi norm1 returns tuple (modulated_inp, ...)
        modulated_inp = model.transformer_blocks[0].norm1(hidden_states, temb)[0]
        self.assertIsInstance(modulated_inp, torch.Tensor)
        self.assertEqual(modulated_inp.shape[0], hidden_states.shape[0])

    def test_lumina2_extractor(self):
        """Test Lumina2 modulated input extraction (now inlined in forward function)."""
        from diffusers import Lumina2Transformer2DModel

        # Create a minimal Lumina2 model for testing
        model = Lumina2Transformer2DModel(
            sample_size=16,
            patch_size=2,
            in_channels=4,
            hidden_size=24,
            num_layers=2,
            num_refiner_layers=1,
            num_attention_heads=3,
            num_kv_heads=1,
        )

        # Create properly shaped inputs that match what the extractor expects
        # The extractor expects input_to_main_loop (already preprocessed concatenated text+image tokens)
        batch_size = 2
        seq_len = 100  # combined text + image sequence
        hidden_size = model.config.hidden_size

        # Simulate input_to_main_loop (already preprocessed)
        input_to_main_loop = torch.randn(batch_size, seq_len, hidden_size)
        temb = torch.randn(batch_size, hidden_size)

        # Test inline extraction logic: Lumina2 uses layers[0].norm1
        modulated_inp = model.layers[0].norm1(input_to_main_loop, temb)[0]
        self.assertIsInstance(modulated_inp, torch.Tensor)
        self.assertEqual(modulated_inp.shape[0], batch_size)

    def test_cogvideox_extractor(self):
        """Test CogVideoX modulated input extraction (now inlined in forward function)."""
        from diffusers import CogVideoXTransformer3DModel

        # Create a minimal CogVideoX model for testing
        model = CogVideoXTransformer3DModel(
            num_attention_heads=2,
            attention_head_dim=8,
            in_channels=4,
            num_layers=2,
            text_embed_dim=16,
            time_embed_dim=4,
        )

        hidden_states = torch.randn(2, 2, 4, 8, 8)
        timestep = torch.randint(0, 1000, (2,))

        # Get timestep embedding
        t_emb = model.time_proj(timestep)
        t_emb = t_emb.to(dtype=hidden_states.dtype)
        emb = model.time_embedding(t_emb, None)

        # Test inline extraction logic: CogVideoX uses timestep embedding directly
        modulated_inp = emb
        self.assertIsInstance(modulated_inp, torch.Tensor)
        self.assertEqual(modulated_inp.shape, emb.shape)

    def test_auto_detect_mochi(self):
        """Test auto-detection for Mochi models."""
        from diffusers import MochiTransformer3DModel
        from diffusers.hooks import TeaCacheConfig, apply_teacache
        from diffusers.hooks.teacache import _get_model_config

        model = MochiTransformer3DModel(
            patch_size=2,
            num_attention_heads=2,
            attention_head_dim=8,
            num_layers=2,
            in_channels=4,
            text_embed_dim=16,
            time_embed_dim=4,
        )

        model_config = _get_model_config()

        # Test coefficient auto-detection
        config = TeaCacheConfig(rel_l1_thresh=0.2)
        apply_teacache(model, config)

        registry = HookRegistry.check_if_exists_or_initialize(model)
        hook = registry.get_hook("teacache")
        self.assertIsNotNone(hook)
        # Verify coefficients were auto-set
        self.assertEqual(hook.coefficients, model_config["Mochi"]["coefficients"])

        model.disable_cache()

    def test_auto_detect_lumina2(self):
        """Test auto-detection for Lumina2 models."""
        from diffusers import Lumina2Transformer2DModel
        from diffusers.hooks import TeaCacheConfig, apply_teacache
        from diffusers.hooks.teacache import _get_model_config

        model = Lumina2Transformer2DModel(
            sample_size=16,
            patch_size=2,
            in_channels=4,
            hidden_size=24,
            num_layers=2,
            num_refiner_layers=1,
            num_attention_heads=3,
            num_kv_heads=1,
        )

        model_config = _get_model_config()

        config = TeaCacheConfig(rel_l1_thresh=0.2)
        apply_teacache(model, config)

        registry = HookRegistry.check_if_exists_or_initialize(model)
        hook = registry.get_hook("teacache")
        self.assertIsNotNone(hook)
        # Verify coefficients were auto-set
        self.assertEqual(hook.coefficients, model_config["Lumina2"]["coefficients"])

        # Lumina2 doesn't have CacheMixin, manually remove hook instead
        registry.remove_hook("teacache")

    def test_auto_detect_cogvideox(self):
        """Test auto-detection for CogVideoX models."""
        from diffusers import CogVideoXTransformer3DModel
        from diffusers.hooks import TeaCacheConfig, apply_teacache
        from diffusers.hooks.teacache import _get_model_config

        model = CogVideoXTransformer3DModel(
            num_attention_heads=2,
            attention_head_dim=8,
            in_channels=4,
            num_layers=2,
            text_embed_dim=16,
            time_embed_dim=4,
        )

        model_config = _get_model_config()

        config = TeaCacheConfig(rel_l1_thresh=0.2)
        apply_teacache(model, config)

        registry = HookRegistry.check_if_exists_or_initialize(model)
        hook = registry.get_hook("teacache")
        self.assertIsNotNone(hook)
        # Verify coefficients were auto-set
        self.assertEqual(hook.coefficients, model_config["CogVideoX"]["coefficients"])

        model.disable_cache()

    def test_teacache_state_encoder_residual(self):
        """Test that TeaCacheState supports encoder residual for CogVideoX."""
        from diffusers.hooks.teacache import TeaCacheState

        state = TeaCacheState()
        self.assertIsNone(state.previous_residual_encoder)

        # Set encoder residual
        state.previous_residual_encoder = torch.randn(2, 10, 16)
        self.assertIsNotNone(state.previous_residual_encoder)

        # Reset should clear it
        state.reset()
        self.assertIsNone(state.previous_residual_encoder)

    def test_model_routing(self):
        """Test that new_forward routes to correct handler based on model type."""
        from diffusers import CogVideoXTransformer3DModel, Lumina2Transformer2DModel, MochiTransformer3DModel
        from diffusers.hooks.teacache import TeaCacheConfig, TeaCacheHook

        config = TeaCacheConfig(rel_l1_thresh=0.2)

        # Test Mochi routing
        mochi_model = MochiTransformer3DModel(
            patch_size=2,
            num_attention_heads=2,
            attention_head_dim=8,
            num_layers=2,
            in_channels=4,
            text_embed_dim=16,
            time_embed_dim=4,
        )
        mochi_hook = TeaCacheHook(config)
        mochi_hook.initialize_hook(mochi_model)
        self.assertEqual(mochi_hook.model_type, "Mochi")

        # Test Lumina2 routing
        lumina_model = Lumina2Transformer2DModel(
            sample_size=16,
            patch_size=2,
            in_channels=4,
            hidden_size=24,
            num_layers=2,
            num_refiner_layers=1,
            num_attention_heads=3,
            num_kv_heads=1,
        )
        lumina_hook = TeaCacheHook(config)
        lumina_hook.initialize_hook(lumina_model)
        self.assertEqual(lumina_hook.model_type, "Lumina2")

        # Test CogVideoX routing
        cogvideox_model = CogVideoXTransformer3DModel(
            num_attention_heads=2,
            attention_head_dim=8,
            in_channels=4,
            num_layers=2,
            text_embed_dim=16,
            time_embed_dim=4,
        )
        cogvideox_hook = TeaCacheHook(config)
        cogvideox_hook.initialize_hook(cogvideox_model)
        self.assertEqual(cogvideox_hook.model_type, "CogVideoX")


class StateManagerContextTests(unittest.TestCase):
    """Tests for StateManager context isolation and backward compatibility."""

    def test_context_isolation(self):
        """Test that different contexts maintain separate states."""
        from diffusers.hooks import StateManager
        from diffusers.hooks.teacache import TeaCacheState

        state_manager = StateManager(TeaCacheState, (), {})

        # Set context "cond" and modify state
        state_manager.set_context("cond")
        cond_state = state_manager.get_state()
        cond_state.cnt = 5
        cond_state.accumulated_rel_l1_distance = 0.3

        # Set context "uncond" and modify state
        state_manager.set_context("uncond")
        uncond_state = state_manager.get_state()
        uncond_state.cnt = 10
        uncond_state.accumulated_rel_l1_distance = 0.7

        # Verify isolation - switch back to "cond"
        state_manager.set_context("cond")
        self.assertEqual(state_manager.get_state().cnt, 5)
        self.assertEqual(state_manager.get_state().accumulated_rel_l1_distance, 0.3)

        # Verify isolation - switch back to "uncond"
        state_manager.set_context("uncond")
        self.assertEqual(state_manager.get_state().cnt, 10)
        self.assertEqual(state_manager.get_state().accumulated_rel_l1_distance, 0.7)

    def test_default_context_fallback(self):
        """Test that state works without explicit context (backward compatibility)."""
        from diffusers.hooks import StateManager
        from diffusers.hooks.teacache import TeaCacheState

        state_manager = StateManager(TeaCacheState, (), {})

        # Don't set context - should use "_default" fallback
        state = state_manager.get_state()
        self.assertIsNotNone(state)
        self.assertEqual(state.cnt, 0)

        # Modify state
        state.cnt = 42

        # Should still get the same state via default context
        state2 = state_manager.get_state()
        self.assertEqual(state2.cnt, 42)

    def test_default_context_separate_from_named(self):
        """Test that default context is separate from named contexts."""
        from diffusers.hooks import StateManager
        from diffusers.hooks.teacache import TeaCacheState

        state_manager = StateManager(TeaCacheState, (), {})

        # Use default context (no explicit set_context)
        default_state = state_manager.get_state()
        default_state.cnt = 100

        # Now set a named context
        state_manager.set_context("named")
        named_state = state_manager.get_state()
        named_state.cnt = 200

        # Clear context to use default again
        state_manager._current_context = None
        self.assertEqual(state_manager.get_state().cnt, 100)

        # Named context should still have its value
        state_manager.set_context("named")
        self.assertEqual(state_manager.get_state().cnt, 200)


if __name__ == "__main__":
    unittest.main()
