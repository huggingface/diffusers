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

import inspect

import numpy as np
import pytest
import torch

from diffusers import FasterCacheConfig, PyramidAttentionBroadcastConfig, apply_faster_cache
from diffusers.hooks.faster_cache import FasterCacheBlockHook, FasterCacheDenoiserHook
from diffusers.hooks.first_block_cache import FirstBlockCacheConfig
from diffusers.hooks.mag_cache import MagCacheConfig
from diffusers.hooks.pyramid_attention_broadcast import PyramidAttentionBroadcastHook
from diffusers.hooks.taylorseer_cache import TaylorSeerCacheConfig
from diffusers.utils import logging

from ...testing_utils import CaptureLogger, is_cache
from .utils import assert_outputs_close


class CacheTesterMixin:
    """
    Shared machinery for cache-hook tester mixins. Each cache backend subclasses this and supplies its own config,
    mirroring the model-level `cache.py` layout. The denoiser-level enable/disable inference comparison is shared
    via `_test_cache_inference`; backend-specific state/layer checks live on the subclasses.
    """

    def _test_cache_inference(self, cache_config, num_inference_steps, expected_atol=0.1, set_timestep_callback=False):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator

        def create_pipe():
            torch.manual_seed(0)
            components = self.get_dummy_components(num_layers=2)
            pipe = self.pipeline_class(**components)
            pipe = pipe.to(device)
            pipe.set_progress_bar_config(disable=None)
            return pipe

        def run_forward(pipe):
            torch.manual_seed(0)
            inputs = self.get_dummy_inputs(device)
            inputs["num_inference_steps"] = num_inference_steps
            return pipe(**inputs)[0]

        # Run inference without cache
        pipe = create_pipe()
        output = run_forward(pipe).flatten()
        original_image_slice = np.concatenate((output[:8], output[-8:]))

        # Run inference with cache enabled
        pipe = create_pipe()
        if set_timestep_callback:
            cache_config.current_timestep_callback = lambda: pipe.current_timestep
        pipe.transformer.enable_cache(cache_config)
        output = run_forward(pipe).flatten()
        image_slice_enabled = np.concatenate((output[:8], output[-8:]))

        # Run inference with cache disabled
        pipe.transformer.disable_cache()
        output = run_forward(pipe).flatten()
        image_slice_disabled = np.concatenate((output[:8], output[-8:]))

        assert_outputs_close(
            image_slice_enabled,
            original_image_slice,
            atol=expected_atol,
            rtol=1e-5,
            msg="Cached outputs should not differ much in the specified timestep range.",
        )
        assert_outputs_close(
            image_slice_disabled,
            original_image_slice,
            atol=1e-4,
            rtol=1e-5,
            msg="Outputs from normal inference and after disabling cache should not differ.",
        )


@is_cache
class PyramidAttentionBroadcastTesterMixin(CacheTesterMixin):
    pab_config = PyramidAttentionBroadcastConfig(
        spatial_attention_block_skip_range=2,
        spatial_attention_timestep_skip_range=(100, 800),
        spatial_attention_block_identifiers=["transformer_blocks"],
    )

    def test_pyramid_attention_broadcast_layers(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator

        num_layers = 0
        num_single_layers = 0
        dummy_component_kwargs = {}
        dummy_component_parameters = inspect.signature(self.get_dummy_components).parameters
        if "num_layers" in dummy_component_parameters:
            num_layers = 2
            dummy_component_kwargs["num_layers"] = num_layers
        if "num_single_layers" in dummy_component_parameters:
            num_single_layers = 2
            dummy_component_kwargs["num_single_layers"] = num_single_layers

        components = self.get_dummy_components(**dummy_component_kwargs)
        pipe = self.pipeline_class(**components)
        pipe.set_progress_bar_config(disable=None)

        self.pab_config.current_timestep_callback = lambda: pipe.current_timestep
        denoiser = pipe.transformer if hasattr(pipe, "transformer") else pipe.unet
        denoiser.enable_cache(self.pab_config)

        expected_hooks = 0
        if self.pab_config.spatial_attention_block_skip_range is not None:
            expected_hooks += num_layers + num_single_layers
        if self.pab_config.temporal_attention_block_skip_range is not None:
            expected_hooks += num_layers + num_single_layers
        if self.pab_config.cross_attention_block_skip_range is not None:
            expected_hooks += num_layers + num_single_layers

        denoiser = pipe.transformer if hasattr(pipe, "transformer") else pipe.unet
        count = 0
        for module in denoiser.modules():
            if hasattr(module, "_diffusers_hook"):
                hook = module._diffusers_hook.get_hook("pyramid_attention_broadcast")
                if hook is None:
                    continue
                count += 1
                assert isinstance(hook, PyramidAttentionBroadcastHook), (
                    "Hook should be of type PyramidAttentionBroadcastHook."
                )
                assert hook.state.cache is None, "Cache should be None at initialization."
        assert count == expected_hooks, "Number of hooks should match the expected number."

        # Perform dummy inference step to ensure state is updated
        def pab_state_check_callback(pipe, i, t, kwargs):
            for module in denoiser.modules():
                if hasattr(module, "_diffusers_hook"):
                    hook = module._diffusers_hook.get_hook("pyramid_attention_broadcast")
                    if hook is None:
                        continue
                    assert hook.state.cache is not None, "Cache should have updated during inference."
                    assert hook.state.iteration == i + 1, "Hook iteration state should have updated during inference."
            return {}

        inputs = self.get_dummy_inputs(device)
        inputs["num_inference_steps"] = 2
        inputs["callback_on_step_end"] = pab_state_check_callback
        pipe(**inputs)[0]

        # After inference, reset_stateful_hooks is called within the pipeline, which should have reset the states
        for module in denoiser.modules():
            if hasattr(module, "_diffusers_hook"):
                hook = module._diffusers_hook.get_hook("pyramid_attention_broadcast")
                if hook is None:
                    continue
                assert hook.state.cache is None, "Cache should be reset to None after inference."
                assert hook.state.iteration == 0, "Iteration should be reset to 0 after inference."

    def test_pyramid_attention_broadcast_inference(self, expected_atol: float = 0.2):
        # We need to use higher tolerance because we are using a random model. With a converged/trained model, the
        # tolerance can be lower.
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        num_layers = 2
        components = self.get_dummy_components(num_layers=num_layers)
        for key in components:
            if "text_encoder" in key and hasattr(components[key], "eval"):
                components[key].eval()
        pipe = self.pipeline_class(**components)
        pipe = pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        # Run inference without PAB
        inputs = self.get_dummy_inputs(device)
        inputs["num_inference_steps"] = 4
        output = pipe(**inputs)[0]
        original_image_slice = output.flatten()
        original_image_slice = np.concatenate((original_image_slice[:8], original_image_slice[-8:]))

        # Run inference with PAB enabled
        self.pab_config.current_timestep_callback = lambda: pipe.current_timestep
        denoiser = pipe.transformer if hasattr(pipe, "transformer") else pipe.unet
        denoiser.enable_cache(self.pab_config)

        inputs = self.get_dummy_inputs(device)
        inputs["num_inference_steps"] = 4
        output = pipe(**inputs)[0]
        image_slice_pab_enabled = output.flatten()
        image_slice_pab_enabled = np.concatenate((image_slice_pab_enabled[:8], image_slice_pab_enabled[-8:]))

        # Run inference with PAB disabled
        denoiser.disable_cache()

        inputs = self.get_dummy_inputs(device)
        inputs["num_inference_steps"] = 4
        output = pipe(**inputs)[0]
        image_slice_pab_disabled = output.flatten()
        image_slice_pab_disabled = np.concatenate((image_slice_pab_disabled[:8], image_slice_pab_disabled[-8:]))

        assert_outputs_close(
            image_slice_pab_enabled,
            original_image_slice,
            atol=expected_atol,
            rtol=1e-5,
            msg="PAB outputs should not differ much in specified timestep range.",
        )
        assert_outputs_close(
            image_slice_pab_disabled,
            original_image_slice,
            atol=1e-4,
            rtol=1e-5,
            msg="Outputs from normal inference and after disabling cache should not differ.",
        )


@is_cache
class FasterCacheTesterMixin(CacheTesterMixin):
    faster_cache_config = FasterCacheConfig(
        spatial_attention_block_skip_range=2,
        spatial_attention_timestep_skip_range=(-1, 901),
        unconditional_batch_skip_range=2,
        attention_weight_callback=lambda _: 0.5,
    )

    def test_faster_cache_basic_warning_or_errors_raised(self):
        components = self.get_dummy_components()

        logger = logging.get_logger("diffusers.hooks.faster_cache")
        logger.setLevel(logging.INFO)

        # Check if warning is raised when no attention_weight_callback is provided
        pipe = self.pipeline_class(**components)
        with CaptureLogger(logger) as cap_logger:
            config = FasterCacheConfig(spatial_attention_block_skip_range=2, attention_weight_callback=None)
            apply_faster_cache(pipe.transformer, config)
        assert "No `attention_weight_callback` provided when enabling FasterCache" in cap_logger.out

        # Check if error raised when unsupported tensor format used
        pipe = self.pipeline_class(**components)
        with pytest.raises(ValueError):
            config = FasterCacheConfig(spatial_attention_block_skip_range=2, tensor_format="BFHWC")
            apply_faster_cache(pipe.transformer, config)

    def test_faster_cache_inference(self, expected_atol: float = 0.1):
        self._test_cache_inference(
            self.faster_cache_config, num_inference_steps=4, expected_atol=expected_atol, set_timestep_callback=True
        )

    def test_faster_cache_state(self):
        from diffusers.hooks.faster_cache import _FASTER_CACHE_BLOCK_HOOK, _FASTER_CACHE_DENOISER_HOOK

        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        num_layers = 0
        num_single_layers = 0
        dummy_component_kwargs = {}
        dummy_component_parameters = inspect.signature(self.get_dummy_components).parameters
        if "num_layers" in dummy_component_parameters:
            num_layers = 2
            dummy_component_kwargs["num_layers"] = num_layers
        if "num_single_layers" in dummy_component_parameters:
            num_single_layers = 2
            dummy_component_kwargs["num_single_layers"] = num_single_layers

        components = self.get_dummy_components(**dummy_component_kwargs)
        pipe = self.pipeline_class(**components)
        pipe.set_progress_bar_config(disable=None)

        self.faster_cache_config.current_timestep_callback = lambda: pipe.current_timestep
        pipe.transformer.enable_cache(self.faster_cache_config)

        expected_hooks = 0
        if self.faster_cache_config.spatial_attention_block_skip_range is not None:
            expected_hooks += num_layers + num_single_layers
        if self.faster_cache_config.temporal_attention_block_skip_range is not None:
            expected_hooks += num_layers + num_single_layers

        # Check if faster_cache denoiser hook is attached
        denoiser = pipe.transformer if hasattr(pipe, "transformer") else pipe.unet
        assert hasattr(denoiser, "_diffusers_hook") and isinstance(
            denoiser._diffusers_hook.get_hook(_FASTER_CACHE_DENOISER_HOOK), FasterCacheDenoiserHook
        ), "Hook should be of type FasterCacheDenoiserHook."

        # Check if all blocks have faster_cache block hook attached
        count = 0
        for name, module in denoiser.named_modules():
            if hasattr(module, "_diffusers_hook"):
                if name == "":
                    # Skip the root denoiser module
                    continue
                count += 1
                assert isinstance(module._diffusers_hook.get_hook(_FASTER_CACHE_BLOCK_HOOK), FasterCacheBlockHook), (
                    "Hook should be of type FasterCacheBlockHook."
                )
        assert count == expected_hooks, "Number of hooks should match expected number."

        # Perform inference to ensure that states are updated correctly
        def faster_cache_state_check_callback(pipe, i, t, kwargs):
            for name, module in denoiser.named_modules():
                if not hasattr(module, "_diffusers_hook"):
                    continue
                if name == "":
                    # Root denoiser module
                    state = module._diffusers_hook.get_hook(_FASTER_CACHE_DENOISER_HOOK).state
                    if not self.faster_cache_config.is_guidance_distilled:
                        assert state.low_frequency_delta is not None, "Low frequency delta should be set."
                        assert state.high_frequency_delta is not None, "High frequency delta should be set."
                else:
                    # Internal blocks
                    state = module._diffusers_hook.get_hook(_FASTER_CACHE_BLOCK_HOOK).state
                    assert state.cache is not None and len(state.cache) == 2, "Cache should be set."
                assert state.iteration == i + 1, "Hook iteration state should have updated during inference."
            return {}

        inputs = self.get_dummy_inputs(device)
        inputs["num_inference_steps"] = 4
        inputs["callback_on_step_end"] = faster_cache_state_check_callback
        _ = pipe(**inputs)[0]

        # After inference, reset_stateful_hooks is called within the pipeline, which should have reset the states
        for name, module in denoiser.named_modules():
            if not hasattr(module, "_diffusers_hook"):
                continue

            if name == "":
                # Root denoiser module
                state = module._diffusers_hook.get_hook(_FASTER_CACHE_DENOISER_HOOK).state
                assert state.iteration == 0, "Iteration should be reset to 0."
                assert state.low_frequency_delta is None, "Low frequency delta should be reset to None."
                assert state.high_frequency_delta is None, "High frequency delta should be reset to None."
            else:
                # Internal blocks
                state = module._diffusers_hook.get_hook(_FASTER_CACHE_BLOCK_HOOK).state
                assert state.iteration == 0, "Iteration should be reset to 0."
                assert state.batch_size is None, "Batch size should be reset to None."
                assert state.cache is None, "Cache should be reset to None."


# TODO(aryan, dhruv): the cache tester mixins should probably be rewritten so that more models can be tested out
# of the box once there is better cache support/implementation
@is_cache
class FirstBlockCacheTesterMixin(CacheTesterMixin):
    # threshold is intentionally set higher than usual values since we're testing with random unconverged models
    # that will not satisfy the expected properties of the denoiser for caching to be effective
    first_block_cache_config = FirstBlockCacheConfig(threshold=0.8)

    def test_first_block_cache_inference(self, expected_atol: float = 0.1):
        self._test_cache_inference(self.first_block_cache_config, num_inference_steps=4, expected_atol=expected_atol)


@is_cache
class TaylorSeerCacheTesterMixin(CacheTesterMixin):
    taylorseer_cache_config = TaylorSeerCacheConfig(
        cache_interval=5,
        disable_cache_before_step=10,
        max_order=1,
        taylor_factors_dtype=torch.bfloat16,
        use_lite_mode=True,
    )

    def test_taylorseer_cache_inference(self, expected_atol: float = 0.1):
        self._test_cache_inference(self.taylorseer_cache_config, num_inference_steps=50, expected_atol=expected_atol)


@is_cache
class MagCacheTesterMixin(CacheTesterMixin):
    mag_cache_config = MagCacheConfig(
        threshold=0.06,
        max_skip_steps=3,
        retention_ratio=0.2,
        num_inference_steps=50,
        mag_ratios=torch.ones(50),
    )

    def test_mag_cache_inference(self, expected_atol: float = 0.1):
        self._test_cache_inference(self.mag_cache_config, num_inference_steps=50, expected_atol=expected_atol)
