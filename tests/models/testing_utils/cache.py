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

import gc

import pytest
import torch

from diffusers.hooks import FasterCacheConfig, FirstBlockCacheConfig, PyramidAttentionBroadcastConfig
from diffusers.hooks.faster_cache import _FASTER_CACHE_BLOCK_HOOK, _FASTER_CACHE_DENOISER_HOOK
from diffusers.hooks.first_block_cache import _FBC_BLOCK_HOOK, _FBC_LEADER_BLOCK_HOOK
from diffusers.hooks.pyramid_attention_broadcast import _PYRAMID_ATTENTION_BROADCAST_HOOK
from diffusers.models.cache_utils import CacheMixin

from ...testing_utils import assert_tensors_close, backend_empty_cache, is_cache, torch_device


def require_cache_mixin(func):
    """Decorator to skip tests if model doesn't use CacheMixin."""

    def wrapper(self, *args, **kwargs):
        if not issubclass(self.model_class, CacheMixin):
            pytest.skip(f"{self.model_class.__name__} does not use CacheMixin.")
        return func(self, *args, **kwargs)

    return wrapper


class CacheTesterMixin:
    """
    Base mixin class providing common test implementations for cache testing.

    Cache-specific mixins should:
    1. Inherit from their respective config mixin (e.g., PyramidAttentionBroadcastConfigMixin)
    2. Inherit from this mixin
    3. Define the cache config to use for tests

    Expected class attributes:
        - model_class: The model class to test (must use CacheMixin)

    Expected methods in test classes:
        - get_init_dict(): Returns dict of arguments to initialize the model
        - get_dummy_inputs(): Returns dict of inputs to pass to the model forward pass

    Optional overrides:
        - cache_input_key: Property returning the input tensor key to vary between passes (default: "hidden_states")
    """

    @property
    def cache_input_key(self):
        return "hidden_states"

    def setup_method(self):
        gc.collect()
        backend_empty_cache(torch_device)

    def teardown_method(self):
        gc.collect()
        backend_empty_cache(torch_device)

    def _get_cache_config(self):
        """
        Get the cache config for testing.
        Should be implemented by subclasses.
        """
        raise NotImplementedError("Subclass must implement _get_cache_config")

    def _get_hook_names(self):
        """
        Get the hook names to check for this cache type.
        Should be implemented by subclasses.
        Returns a list of hook name strings.
        """
        raise NotImplementedError("Subclass must implement _get_hook_names")

    def _test_cache_enable_disable_state(self):
        """Test that cache enable/disable updates the is_cache_enabled state correctly."""
        init_dict = self.get_init_dict()
        model = self.model_class(**init_dict).to(torch_device)

        # Initially cache should not be enabled
        assert not model.is_cache_enabled, "Cache should not be enabled initially."

        config = self._get_cache_config()

        # Enable cache
        model.enable_cache(config)
        assert model.is_cache_enabled, "Cache should be enabled after enable_cache()."

        # Disable cache
        model.disable_cache()
        assert not model.is_cache_enabled, "Cache should not be enabled after disable_cache()."

    def _test_cache_double_enable_raises_error(self):
        """Test that enabling cache twice raises an error."""
        init_dict = self.get_init_dict()
        model = self.model_class(**init_dict).to(torch_device)

        config = self._get_cache_config()

        model.enable_cache(config)

        # Trying to enable again should raise ValueError
        with pytest.raises(ValueError, match="Caching has already been enabled"):
            model.enable_cache(config)

        # Cleanup
        model.disable_cache()

    def _test_cache_hooks_registered(self):
        """Test that cache hooks are properly registered and removed."""
        init_dict = self.get_init_dict()
        model = self.model_class(**init_dict).to(torch_device)

        config = self._get_cache_config()
        hook_names = self._get_hook_names()

        model.enable_cache(config)

        # Check that at least one hook was registered
        hook_count = 0
        for module in model.modules():
            if hasattr(module, "_diffusers_hook"):
                for hook_name in hook_names:
                    hook = module._diffusers_hook.get_hook(hook_name)
                    if hook is not None:
                        hook_count += 1

        assert hook_count > 0, f"At least one cache hook should be registered. Hook names: {hook_names}"

        # Disable and verify hooks are removed
        model.disable_cache()

        hook_count_after = 0
        for module in model.modules():
            if hasattr(module, "_diffusers_hook"):
                for hook_name in hook_names:
                    hook = module._diffusers_hook.get_hook(hook_name)
                    if hook is not None:
                        hook_count_after += 1

        assert hook_count_after == 0, "Cache hooks should be removed after disable_cache()."

    @torch.no_grad()
    def _test_cache_inference(self):
        """Test that model can run inference with cache enabled."""
        init_dict = self.get_init_dict()
        inputs_dict = self.get_dummy_inputs()
        model = self.model_class(**init_dict).to(torch_device)
        model.eval()

        config = self._get_cache_config()

        model.enable_cache(config)

        # First pass populates the cache
        _ = model(**inputs_dict, return_dict=False)[0]

        # Create modified inputs for second pass (vary input tensor to simulate denoising)
        inputs_dict_step2 = inputs_dict.copy()
        if self.cache_input_key in inputs_dict_step2:
            inputs_dict_step2[self.cache_input_key] = inputs_dict_step2[self.cache_input_key] + torch.randn_like(
                inputs_dict_step2[self.cache_input_key]
            )

        # Second pass uses cached attention with different inputs (produces approximated output)
        output_with_cache = model(**inputs_dict_step2, return_dict=False)[0]

        assert output_with_cache is not None, "Model output should not be None with cache enabled."
        assert not torch.isnan(output_with_cache).any(), "Model output contains NaN with cache enabled."

        # Run same inputs without cache to compare
        model.disable_cache()
        output_without_cache = model(**inputs_dict_step2, return_dict=False)[0]

        # Cached output should be different from non-cached output (due to approximation)
        assert not torch.allclose(output_without_cache, output_with_cache, atol=1e-5), (
            "Cached output should be different from non-cached output due to cache approximation."
        )

    @torch.no_grad()
    def _test_cache_context_manager(self, atol=1e-5, rtol=0):
        """Test the cache_context context manager properly isolates cache state."""
        init_dict = self.get_init_dict()
        inputs_dict = self.get_dummy_inputs()
        model = self.model_class(**init_dict).to(torch_device)
        model.eval()

        config = self._get_cache_config()
        model.enable_cache(config)

        # Run inference in first context
        with model.cache_context("context_1"):
            output_ctx1 = model(**inputs_dict, return_dict=False)[0]

        # Run same inference in second context (cache should be reset)
        with model.cache_context("context_2"):
            output_ctx2 = model(**inputs_dict, return_dict=False)[0]

        # Both contexts should produce the same output (first pass in each)
        assert_tensors_close(
            output_ctx1,
            output_ctx2,
            atol=atol,
            rtol=rtol,
            msg="First pass in different cache contexts should produce the same output.",
        )

        model.disable_cache()

    @torch.no_grad()
    def _test_reset_stateful_cache(self):
        """Test that _reset_stateful_cache resets the cache state."""
        init_dict = self.get_init_dict()
        inputs_dict = self.get_dummy_inputs()
        model = self.model_class(**init_dict).to(torch_device)
        model.eval()

        config = self._get_cache_config()

        model.enable_cache(config)

        _ = model(**inputs_dict, return_dict=False)[0]

        model._reset_stateful_cache()

        model.disable_cache()


@is_cache
class PyramidAttentionBroadcastConfigMixin:
    """
    Base mixin providing PyramidAttentionBroadcast cache config.

    Expected class attributes:
        - model_class: The model class to test (must use CacheMixin)
    """

    # Default PAB config - can be overridden by subclasses
    PAB_CONFIG = {
        "spatial_attention_block_skip_range": 2,
    }

    # Store timestep for callback (must be within default range (100, 800) for skipping to trigger)
    _current_timestep = 500

    def _get_cache_config(self):
        config_kwargs = self.PAB_CONFIG.copy()
        config_kwargs["current_timestep_callback"] = lambda: self._current_timestep
        return PyramidAttentionBroadcastConfig(**config_kwargs)

    def _get_hook_names(self):
        return [_PYRAMID_ATTENTION_BROADCAST_HOOK]


@is_cache
class PyramidAttentionBroadcastTesterMixin(PyramidAttentionBroadcastConfigMixin, CacheTesterMixin):
    """
    Mixin class for testing PyramidAttentionBroadcast caching on models.

    Expected class attributes:
        - model_class: The model class to test (must use CacheMixin)

    Expected methods to be implemented by subclasses:
        - get_init_dict(): Returns dict of arguments to initialize the model
        - get_dummy_inputs(): Returns dict of inputs to pass to the model forward pass

    Pytest mark: cache
        Use `pytest -m "not cache"` to skip these tests
    """

    @require_cache_mixin
    def test_pab_cache_enable_disable_state(self):
        self._test_cache_enable_disable_state()

    @require_cache_mixin
    def test_pab_cache_double_enable_raises_error(self):
        self._test_cache_double_enable_raises_error()

    @require_cache_mixin
    def test_pab_cache_hooks_registered(self):
        self._test_cache_hooks_registered()

    @require_cache_mixin
    def test_pab_cache_inference(self):
        self._test_cache_inference()

    @require_cache_mixin
    def test_pab_cache_context_manager(self):
        self._test_cache_context_manager()

    @require_cache_mixin
    def test_pab_reset_stateful_cache(self):
        self._test_reset_stateful_cache()


@is_cache
class FirstBlockCacheConfigMixin:
    """
    Base mixin providing FirstBlockCache config.

    Expected class attributes:
        - model_class: The model class to test (must use CacheMixin)
    """

    # Default FBC config - can be overridden by subclasses
    # Higher threshold makes FBC more aggressive about caching (skips more often)
    FBC_CONFIG = {
        "threshold": 1.0,
    }

    def _get_cache_config(self):
        return FirstBlockCacheConfig(**self.FBC_CONFIG)

    def _get_hook_names(self):
        return [_FBC_LEADER_BLOCK_HOOK, _FBC_BLOCK_HOOK]


@is_cache
class FirstBlockCacheTesterMixin(FirstBlockCacheConfigMixin, CacheTesterMixin):
    """
    Mixin class for testing FirstBlockCache on models.

    Expected class attributes:
        - model_class: The model class to test (must use CacheMixin)

    Expected methods to be implemented by subclasses:
        - get_init_dict(): Returns dict of arguments to initialize the model
        - get_dummy_inputs(): Returns dict of inputs to pass to the model forward pass

    Pytest mark: cache
        Use `pytest -m "not cache"` to skip these tests
    """

    @torch.no_grad()
    def _test_cache_inference(self):
        """Test that model can run inference with FBC cache enabled (requires cache_context)."""
        init_dict = self.get_init_dict()
        inputs_dict = self.get_dummy_inputs()
        model = self.model_class(**init_dict).to(torch_device)
        model.eval()

        config = self._get_cache_config()
        model.enable_cache(config)

        # FBC requires cache_context to be set for inference
        with model.cache_context("fbc_test"):
            # First pass populates the cache
            _ = model(**inputs_dict, return_dict=False)[0]

            # Create modified inputs for second pass
            inputs_dict_step2 = inputs_dict.copy()
            if self.cache_input_key in inputs_dict_step2:
                inputs_dict_step2[self.cache_input_key] = inputs_dict_step2[self.cache_input_key] + torch.randn_like(
                    inputs_dict_step2[self.cache_input_key]
                )

            # Second pass - FBC should skip remaining blocks and use cached residuals
            output_with_cache = model(**inputs_dict_step2, return_dict=False)[0]

        assert output_with_cache is not None, "Model output should not be None with cache enabled."
        assert not torch.isnan(output_with_cache).any(), "Model output contains NaN with cache enabled."

        # Run same inputs without cache to compare
        model.disable_cache()
        output_without_cache = model(**inputs_dict_step2, return_dict=False)[0]

        # Cached output should be different from non-cached output (due to approximation)
        assert not torch.allclose(output_without_cache, output_with_cache, atol=1e-5), (
            "Cached output should be different from non-cached output due to cache approximation."
        )

    @torch.no_grad()
    def _test_reset_stateful_cache(self):
        """Test that _reset_stateful_cache resets the FBC cache state (requires cache_context)."""
        init_dict = self.get_init_dict()
        inputs_dict = self.get_dummy_inputs()
        model = self.model_class(**init_dict).to(torch_device)
        model.eval()

        config = self._get_cache_config()
        model.enable_cache(config)

        with model.cache_context("fbc_test"):
            _ = model(**inputs_dict, return_dict=False)[0]

        model._reset_stateful_cache()

        model.disable_cache()

    @require_cache_mixin
    def test_fbc_cache_enable_disable_state(self):
        self._test_cache_enable_disable_state()

    @require_cache_mixin
    def test_fbc_cache_double_enable_raises_error(self):
        self._test_cache_double_enable_raises_error()

    @require_cache_mixin
    def test_fbc_cache_hooks_registered(self):
        self._test_cache_hooks_registered()

    @require_cache_mixin
    def test_fbc_cache_inference(self):
        self._test_cache_inference()

    @require_cache_mixin
    def test_fbc_cache_context_manager(self):
        self._test_cache_context_manager()

    @require_cache_mixin
    def test_fbc_reset_stateful_cache(self):
        self._test_reset_stateful_cache()


@is_cache
class FasterCacheConfigMixin:
    """
    Base mixin providing FasterCache config.

    Expected class attributes:
        - model_class: The model class to test (must use CacheMixin)
    """

    # Default FasterCache config - can be overridden by subclasses
    FASTER_CACHE_CONFIG = {
        "spatial_attention_block_skip_range": 2,
        "spatial_attention_timestep_skip_range": (-1, 901),
        "tensor_format": "BCHW",
    }

    def _get_cache_config(self, current_timestep_callback=None):
        config_kwargs = self.FASTER_CACHE_CONFIG.copy()
        if current_timestep_callback is None:
            current_timestep_callback = lambda: 1000  # noqa: E731
        config_kwargs["current_timestep_callback"] = current_timestep_callback
        return FasterCacheConfig(**config_kwargs)

    def _get_hook_names(self):
        return [_FASTER_CACHE_DENOISER_HOOK, _FASTER_CACHE_BLOCK_HOOK]


@is_cache
class FasterCacheTesterMixin(FasterCacheConfigMixin, CacheTesterMixin):
    """
    Mixin class for testing FasterCache on models.

    Note: FasterCache is designed for pipeline-level inference with proper CFG batch handling
    and timestep management. Inference tests are skipped at model level - FasterCache should
    be tested via pipeline tests (e.g., FluxPipeline, HunyuanVideoPipeline).

    Expected class attributes:
        - model_class: The model class to test (must use CacheMixin)

    Expected methods to be implemented by subclasses:
        - get_init_dict(): Returns dict of arguments to initialize the model
        - get_dummy_inputs(): Returns dict of inputs to pass to the model forward pass

    Pytest mark: cache
        Use `pytest -m "not cache"` to skip these tests
    """

    @torch.no_grad()
    def _test_cache_inference(self):
        """Test that model can run inference with FasterCache enabled."""
        init_dict = self.get_init_dict()
        inputs_dict = self.get_dummy_inputs()
        model = self.model_class(**init_dict).to(torch_device)
        model.eval()

        current_timestep = [1000]
        config = self._get_cache_config(current_timestep_callback=lambda: current_timestep[0])

        model.enable_cache(config)

        # First pass with timestep outside skip range - computes and populates cache
        current_timestep[0] = 1000
        _ = model(**inputs_dict, return_dict=False)[0]

        # Move timestep inside skip range so subsequent passes use cache
        current_timestep[0] = 500

        # Create modified inputs for second pass
        inputs_dict_step2 = inputs_dict.copy()
        if self.cache_input_key in inputs_dict_step2:
            inputs_dict_step2[self.cache_input_key] = inputs_dict_step2[self.cache_input_key] + torch.randn_like(
                inputs_dict_step2[self.cache_input_key]
            )

        # Second pass uses cached attention with different inputs
        output_with_cache = model(**inputs_dict_step2, return_dict=False)[0]

        assert output_with_cache is not None, "Model output should not be None with cache enabled."
        assert not torch.isnan(output_with_cache).any(), "Model output contains NaN with cache enabled."

        # Run same inputs without cache to compare
        model.disable_cache()
        output_without_cache = model(**inputs_dict_step2, return_dict=False)[0]

        # Cached output should be different from non-cached output (due to approximation)
        assert not torch.allclose(output_without_cache, output_with_cache, atol=1e-5), (
            "Cached output should be different from non-cached output due to cache approximation."
        )

    @torch.no_grad()
    def _test_reset_stateful_cache(self):
        """Test that _reset_stateful_cache resets the FasterCache state."""
        init_dict = self.get_init_dict()
        inputs_dict = self.get_dummy_inputs()
        model = self.model_class(**init_dict).to(torch_device)
        model.eval()

        config = self._get_cache_config()
        model.enable_cache(config)

        _ = model(**inputs_dict, return_dict=False)[0]

        model._reset_stateful_cache()

        model.disable_cache()

    @require_cache_mixin
    def test_faster_cache_enable_disable_state(self):
        self._test_cache_enable_disable_state()

    @require_cache_mixin
    def test_faster_cache_double_enable_raises_error(self):
        self._test_cache_double_enable_raises_error()

    @require_cache_mixin
    def test_faster_cache_hooks_registered(self):
        self._test_cache_hooks_registered()

    @require_cache_mixin
    def test_faster_cache_inference(self):
        self._test_cache_inference()

    @require_cache_mixin
    def test_faster_cache_context_manager(self):
        self._test_cache_context_manager()

    @require_cache_mixin
    def test_faster_cache_reset_stateful_cache(self):
        self._test_reset_stateful_cache()
