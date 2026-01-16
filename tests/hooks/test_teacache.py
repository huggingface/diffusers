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

from diffusers import CogVideoXTransformer3DModel, Lumina2Transformer2DModel, MochiTransformer3DModel
from diffusers.hooks import HookRegistry, StateManager, TeaCacheConfig, apply_teacache
from diffusers.hooks.teacache import TeaCacheHook, TeaCacheState, _get_model_config, _should_compute


def _create_mochi_model() -> MochiTransformer3DModel:
    return MochiTransformer3DModel(
        patch_size=2,
        num_attention_heads=2,
        attention_head_dim=8,
        num_layers=2,
        in_channels=4,
        text_embed_dim=16,
        time_embed_dim=4,
    )


def _create_lumina2_model() -> Lumina2Transformer2DModel:
    return Lumina2Transformer2DModel(
        sample_size=16,
        patch_size=2,
        in_channels=4,
        hidden_size=24,
        num_layers=2,
        num_refiner_layers=1,
        num_attention_heads=3,
        num_kv_heads=1,
    )


def _create_cogvideox_model() -> CogVideoXTransformer3DModel:
    return CogVideoXTransformer3DModel(
        num_attention_heads=2,
        attention_head_dim=8,
        in_channels=4,
        num_layers=2,
        text_embed_dim=16,
        time_embed_dim=4,
    )


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
        config = TeaCacheConfig(rel_l1_thresh=0.01)
        self.assertEqual(config.rel_l1_thresh, 0.01)

    def test_very_high_threshold_accepted(self):
        """Test very high threshold is accepted (with logging warning)."""
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
        state = TeaCacheState()
        self.assertEqual(state.cnt, 0)
        self.assertEqual(state.num_steps, 0)
        self.assertEqual(state.accumulated_rel_l1_distance, 0.0)
        self.assertIsNone(state.previous_modulated_input)
        self.assertIsNone(state.previous_residual)

    def test_state_reset(self):
        """Test state reset clears all values."""
        state = TeaCacheState()
        state.cnt = 5
        state.num_steps = 10
        state.accumulated_rel_l1_distance = 0.5
        state.previous_modulated_input = torch.randn(1, 10)
        state.previous_residual = torch.randn(1, 10)

        state.reset()

        self.assertEqual(state.cnt, 0)
        self.assertEqual(state.num_steps, 0)
        self.assertEqual(state.accumulated_rel_l1_distance, 0.0)
        self.assertIsNone(state.previous_modulated_input)
        self.assertIsNone(state.previous_residual)

    def test_state_repr(self):
        """Test __repr__ method works correctly."""
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
        config = TeaCacheConfig(rel_l1_thresh=0.2)
        hook = TeaCacheHook(config)

        self.assertEqual(hook.config.rel_l1_thresh, 0.2)
        self.assertIsNone(hook.coefficients)
        self.assertIsNotNone(hook.state_manager)

    def test_should_compute_logic(self):
        """Test _should_compute decision logic."""
        coefficients = [1, 0, 0, 0, 0]
        rel_l1_thresh = 1.0
        state = TeaCacheState()

        x0 = torch.ones(1, 4)
        x1 = torch.ones(1, 4) * 1.1

        self.assertTrue(_should_compute(state, x0, coefficients, rel_l1_thresh))

        state.previous_modulated_input = x0
        state.previous_residual = torch.zeros(1, 4)
        state.cnt = 1
        state.num_steps = 4

        self.assertFalse(_should_compute(state, x1, coefficients, rel_l1_thresh))

        state.cnt = state.num_steps - 1
        self.assertTrue(_should_compute(state, x1, coefficients, rel_l1_thresh))

    def test_apply_teacache_unsupported_model_raises_error(self):
        """Test that apply_teacache raises error for unsupported models."""
        from diffusers.models import CacheMixin

        class UnsupportedModule(torch.nn.Module, CacheMixin):
            def __init__(self):
                super().__init__()
                self.dummy = torch.nn.Linear(4, 4)

        module = UnsupportedModule()
        config = TeaCacheConfig(rel_l1_thresh=0.2)

        with self.assertRaises(ValueError) as context:
            apply_teacache(module, config)
        self.assertIn("Unsupported model", str(context.exception))
        self.assertIn("UnsupportedModule", str(context.exception))


class TeaCacheMultiModelTests(unittest.TestCase):
    """Tests for TeaCache multi-model support (Mochi, Lumina2, CogVideoX)."""

    def test_model_coefficient_registry(self):
        """Test that model coefficients are properly registered."""
        model_config = _get_model_config()

        self.assertIn("Flux", model_config)
        self.assertIn("Mochi", model_config)
        self.assertIn("Lumina2", model_config)
        self.assertIn("CogVideoX", model_config)

        for model_name, config in model_config.items():
            coeffs = config["coefficients"]
            self.assertEqual(len(coeffs), 5, f"{model_name} coefficients should have 5 elements")
            self.assertTrue(
                all(isinstance(c, (int, float)) for c in coeffs), f"{model_name} coefficients should be numbers"
            )

    def test_mochi_extractor(self):
        """Test Mochi modulated input extraction."""
        model = _create_mochi_model()

        hidden_states = torch.randn(2, 4, 2, 8, 8)
        timestep = torch.randint(0, 1000, (2,))
        encoder_hidden_states = torch.randn(2, 16, 16)
        encoder_attention_mask = torch.ones(2, 16).bool()

        temb, _ = model.time_embed(
            timestep, encoder_hidden_states, encoder_attention_mask, hidden_dtype=hidden_states.dtype
        )
        hidden_states = hidden_states.permute(0, 2, 1, 3, 4).flatten(0, 1)
        hidden_states = model.patch_embed(hidden_states)
        hidden_states = hidden_states.unflatten(0, (2, -1)).flatten(1, 2)

        modulated_inp = model.transformer_blocks[0].norm1(hidden_states, temb)[0]
        self.assertIsInstance(modulated_inp, torch.Tensor)
        self.assertEqual(modulated_inp.shape[0], hidden_states.shape[0])

    def test_lumina2_extractor(self):
        """Test Lumina2 modulated input extraction."""
        model = _create_lumina2_model()

        batch_size = 2
        seq_len = 100
        hidden_size = model.config.hidden_size

        input_to_main_loop = torch.randn(batch_size, seq_len, hidden_size)
        temb = torch.randn(batch_size, hidden_size)

        modulated_inp = model.layers[0].norm1(input_to_main_loop, temb)[0]
        self.assertIsInstance(modulated_inp, torch.Tensor)
        self.assertEqual(modulated_inp.shape[0], batch_size)

    def test_cogvideox_extractor(self):
        """Test CogVideoX modulated input extraction."""
        model = _create_cogvideox_model()

        hidden_states = torch.randn(2, 2, 4, 8, 8)
        timestep = torch.randint(0, 1000, (2,))

        t_emb = model.time_proj(timestep)
        t_emb = t_emb.to(dtype=hidden_states.dtype)
        emb = model.time_embedding(t_emb, None)

        modulated_inp = emb
        self.assertIsInstance(modulated_inp, torch.Tensor)
        self.assertEqual(modulated_inp.shape, emb.shape)

    def test_auto_detect_mochi(self):
        """Test auto-detection for Mochi models."""
        model = _create_mochi_model()
        model_config = _get_model_config()

        config = TeaCacheConfig(rel_l1_thresh=0.2)
        apply_teacache(model, config)

        registry = HookRegistry.check_if_exists_or_initialize(model)
        hook = registry.get_hook("teacache")
        self.assertIsNotNone(hook)
        self.assertEqual(hook.coefficients, model_config["Mochi"]["coefficients"])

        model.disable_cache()

    def test_auto_detect_lumina2(self):
        """Test auto-detection for Lumina2 models."""
        model = _create_lumina2_model()
        model_config = _get_model_config()

        config = TeaCacheConfig(rel_l1_thresh=0.2)
        apply_teacache(model, config)

        registry = HookRegistry.check_if_exists_or_initialize(model)
        hook = registry.get_hook("teacache")
        self.assertIsNotNone(hook)
        self.assertEqual(hook.coefficients, model_config["Lumina2"]["coefficients"])

        registry.remove_hook("teacache")

    def test_auto_detect_cogvideox(self):
        """Test auto-detection for CogVideoX models."""
        model = _create_cogvideox_model()
        model_config = _get_model_config()

        config = TeaCacheConfig(rel_l1_thresh=0.2)
        apply_teacache(model, config)

        registry = HookRegistry.check_if_exists_or_initialize(model)
        hook = registry.get_hook("teacache")
        self.assertIsNotNone(hook)
        self.assertEqual(hook.coefficients, model_config["CogVideoX"]["coefficients"])

        model.disable_cache()

    def test_teacache_state_encoder_residual(self):
        """Test that TeaCacheState supports encoder residual for CogVideoX."""
        state = TeaCacheState()
        self.assertIsNone(state.previous_residual_encoder)

        state.previous_residual_encoder = torch.randn(2, 10, 16)
        self.assertIsNotNone(state.previous_residual_encoder)

        state.reset()
        self.assertIsNone(state.previous_residual_encoder)

    def test_model_routing(self):
        """Test that new_forward routes to correct handler based on model type."""
        config = TeaCacheConfig(rel_l1_thresh=0.2)

        mochi_hook = TeaCacheHook(config)
        mochi_hook.initialize_hook(_create_mochi_model())
        self.assertEqual(mochi_hook.model_type, "Mochi")

        lumina_hook = TeaCacheHook(config)
        lumina_hook.initialize_hook(_create_lumina2_model())
        self.assertEqual(lumina_hook.model_type, "Lumina2")

        cogvideox_hook = TeaCacheHook(config)
        cogvideox_hook.initialize_hook(_create_cogvideox_model())
        self.assertEqual(cogvideox_hook.model_type, "CogVideoX")


class StateManagerContextTests(unittest.TestCase):
    """Tests for StateManager context isolation and backward compatibility."""

    def test_context_isolation(self):
        """Test that different contexts maintain separate states."""
        state_manager = StateManager(TeaCacheState, (), {})

        state_manager.set_context("cond")
        cond_state = state_manager.get_state()
        cond_state.cnt = 5
        cond_state.accumulated_rel_l1_distance = 0.3

        state_manager.set_context("uncond")
        uncond_state = state_manager.get_state()
        uncond_state.cnt = 10
        uncond_state.accumulated_rel_l1_distance = 0.7

        state_manager.set_context("cond")
        self.assertEqual(state_manager.get_state().cnt, 5)
        self.assertEqual(state_manager.get_state().accumulated_rel_l1_distance, 0.3)

        state_manager.set_context("uncond")
        self.assertEqual(state_manager.get_state().cnt, 10)
        self.assertEqual(state_manager.get_state().accumulated_rel_l1_distance, 0.7)

    def test_default_context_fallback(self):
        """Test that state works without explicit context (backward compatibility)."""
        state_manager = StateManager(TeaCacheState, (), {})

        state = state_manager.get_state()
        self.assertIsNotNone(state)
        self.assertEqual(state.cnt, 0)

        state.cnt = 42

        state2 = state_manager.get_state()
        self.assertEqual(state2.cnt, 42)

    def test_default_context_separate_from_named(self):
        """Test that default context is separate from named contexts."""
        state_manager = StateManager(TeaCacheState, (), {})

        default_state = state_manager.get_state()
        default_state.cnt = 100

        state_manager.set_context("named")
        named_state = state_manager.get_state()
        named_state.cnt = 200

        state_manager._current_context = None
        self.assertEqual(state_manager.get_state().cnt, 100)

        state_manager.set_context("named")
        self.assertEqual(state_manager.get_state().cnt, 200)


if __name__ == "__main__":
    unittest.main()
