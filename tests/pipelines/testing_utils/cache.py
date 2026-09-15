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

import torch

from diffusers import FasterCacheConfig, PyramidAttentionBroadcastConfig
from diffusers.hooks.first_block_cache import FirstBlockCacheConfig
from diffusers.hooks.mag_cache import MagCacheConfig
from diffusers.hooks.pyramid_attention_broadcast import PyramidAttentionBroadcastHook
from diffusers.hooks.taylorseer_cache import TaylorSeerCacheConfig

from ...testing_utils import assert_tensors_close, is_cache, torch_device
from .common import BasePipelineOutputMixin


class CacheTesterMixin(BasePipelineOutputMixin):
    """
    Shared machinery for cache-hook tester mixins. Each cache backend subclasses this and supplies its own config,
    mirroring the model-level `cache.py` layout. Backends store their config *kwargs* as a dict class attribute and
    build a fresh config instance per test via `_get_cache_config()`; a shared config instance would leak per-test
    mutations (e.g. `current_timestep_callback`) across tests. The denoiser-level enable/disable inference comparison
    is shared via `_test_cache_inference`; backend-specific state/layer checks live on the subclasses.
    """

    def _get_cache_config(self):
        raise NotImplementedError("Subclass must implement `_get_cache_config`.")

    def _test_cache_inference(self, cache_config, num_inference_steps, expected_atol=0.1, set_timestep_callback=False):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator

        def create_pipe():
            torch.manual_seed(0)
            return self.get_pipeline(**self.get_dummy_components(num_layers=2)).to(device)

        def run_forward(pipe):
            torch.manual_seed(0)
            inputs = self.get_dummy_inputs()
            inputs["num_inference_steps"] = num_inference_steps
            return pipe(**inputs)[0]

        # Run inference without cache
        pipe = create_pipe()
        output = run_forward(pipe).flatten()
        original_image_slice = torch.cat((output[:8], output[-8:]))

        # Run inference with cache enabled
        pipe = create_pipe()
        if set_timestep_callback:
            cache_config.current_timestep_callback = lambda: pipe.current_timestep
        pipe.transformer.enable_cache(cache_config)
        output = run_forward(pipe).flatten()
        image_slice_enabled = torch.cat((output[:8], output[-8:]))

        # Run inference with cache disabled
        pipe.transformer.disable_cache()
        output = run_forward(pipe).flatten()
        image_slice_disabled = torch.cat((output[:8], output[-8:]))

        assert_tensors_close(
            image_slice_enabled,
            original_image_slice,
            atol=expected_atol,
            rtol=1e-5,
            msg="Cached outputs should not differ much in the specified timestep range.",
        )
        assert_tensors_close(
            image_slice_disabled,
            original_image_slice,
            atol=1e-4,
            rtol=1e-5,
            msg="Outputs from normal inference and after disabling cache should not differ.",
        )


@is_cache
class PyramidAttentionBroadcastTesterMixin(CacheTesterMixin):
    PAB_CONFIG = {
        "spatial_attention_block_skip_range": 2,
        "spatial_attention_timestep_skip_range": (100, 800),
        "spatial_attention_block_identifiers": ["transformer_blocks"],
    }

    def _get_cache_config(self):
        return PyramidAttentionBroadcastConfig(**self.PAB_CONFIG)

    def test_pyramid_attention_broadcast_layers(self):
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

        pipe = self.get_pipeline(**self.get_dummy_components(**dummy_component_kwargs))

        pab_config = self._get_cache_config()
        pab_config.current_timestep_callback = lambda: pipe.current_timestep
        denoiser = pipe.transformer if hasattr(pipe, "transformer") else pipe.unet
        denoiser.enable_cache(pab_config)

        expected_hooks = 0
        if pab_config.spatial_attention_block_skip_range is not None:
            expected_hooks += num_layers + num_single_layers
        if pab_config.temporal_attention_block_skip_range is not None:
            expected_hooks += num_layers + num_single_layers
        if pab_config.cross_attention_block_skip_range is not None:
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

        inputs = self.get_dummy_inputs()
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

    def test_pyramid_attention_broadcast_inference(self, base_pipe_output, expected_atol: float = 0.2):
        # We need to use higher tolerance because we are using a random model. With a converged/trained model, the
        # tolerance can be lower. The no-PAB reference is the shared, class-cached `base_pipe_output`, so the PAB
        # runs below reuse the same components/inputs the fixture does (default `get_dummy_components()` and
        # `get_dummy_inputs()`) and reseed the global RNG before each forward, so the disabled run reproduces the
        # reference and the only remaining differences come from PAB itself.
        pipe = self.get_pipeline().to(torch_device)

        # Reference (no PAB) output, computed once per test class by the `base_pipe_output` fixture.
        original_image_slice = base_pipe_output.flatten()
        original_image_slice = torch.cat((original_image_slice[:8], original_image_slice[-8:]))

        # Run inference with PAB enabled
        pab_config = self._get_cache_config()
        pab_config.current_timestep_callback = lambda: pipe.current_timestep
        denoiser = pipe.transformer if hasattr(pipe, "transformer") else pipe.unet
        denoiser.enable_cache(pab_config)

        torch.manual_seed(0)
        output = pipe(**self.get_dummy_inputs())[0]
        image_slice_pab_enabled = output.flatten()
        image_slice_pab_enabled = torch.cat((image_slice_pab_enabled[:8], image_slice_pab_enabled[-8:]))

        # Run inference with PAB disabled — should reproduce the reference output
        denoiser.disable_cache()

        torch.manual_seed(0)
        output = pipe(**self.get_dummy_inputs())[0]
        image_slice_pab_disabled = output.flatten()
        image_slice_pab_disabled = torch.cat((image_slice_pab_disabled[:8], image_slice_pab_disabled[-8:]))

        assert_tensors_close(
            image_slice_pab_enabled,
            original_image_slice,
            atol=expected_atol,
            rtol=1e-5,
            msg="PAB outputs should not differ much in specified timestep range.",
        )
        assert_tensors_close(
            image_slice_pab_disabled,
            original_image_slice,
            atol=1e-4,
            rtol=1e-5,
            msg="Outputs from normal inference and after disabling cache should not differ.",
        )


@is_cache
class FasterCacheTesterMixin(CacheTesterMixin):
    FASTER_CACHE_CONFIG = {
        "spatial_attention_block_skip_range": 2,
        "spatial_attention_timestep_skip_range": (-1, 901),
        "unconditional_batch_skip_range": 2,
        "attention_weight_callback": lambda _: 0.5,
    }

    def _get_cache_config(self):
        return FasterCacheConfig(**self.FASTER_CACHE_CONFIG)

    def test_faster_cache_inference(self, expected_atol: float = 0.1):
        self._test_cache_inference(
            self._get_cache_config(), num_inference_steps=4, expected_atol=expected_atol, set_timestep_callback=True
        )

    def test_faster_cache_state(self):
        from diffusers.hooks.faster_cache import _FASTER_CACHE_BLOCK_HOOK, _FASTER_CACHE_DENOISER_HOOK

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

        pipe = self.get_pipeline(**self.get_dummy_components(**dummy_component_kwargs))

        faster_cache_config = self._get_cache_config()
        faster_cache_config.current_timestep_callback = lambda: pipe.current_timestep
        pipe.transformer.enable_cache(faster_cache_config)

        # Hook registration/removal is covered at the model level (`_test_cache_hooks_registered`). Here we only
        # verify the pipeline-specific behaviour: that hook state evolves during the denoising loop and is reset
        # afterwards.
        denoiser = pipe.transformer if hasattr(pipe, "transformer") else pipe.unet

        # Perform inference to ensure that states are updated correctly
        def faster_cache_state_check_callback(pipe, i, t, kwargs):
            for name, module in denoiser.named_modules():
                if not hasattr(module, "_diffusers_hook"):
                    continue
                if name == "":
                    # Root denoiser module
                    state = module._diffusers_hook.get_hook(_FASTER_CACHE_DENOISER_HOOK).state
                    if not faster_cache_config.is_guidance_distilled:
                        assert state.low_frequency_delta is not None, "Low frequency delta should be set."
                        assert state.high_frequency_delta is not None, "High frequency delta should be set."
                else:
                    # Internal blocks
                    state = module._diffusers_hook.get_hook(_FASTER_CACHE_BLOCK_HOOK).state
                    assert state.cache is not None and len(state.cache) == 2, "Cache should be set."
                assert state.iteration == i + 1, "Hook iteration state should have updated during inference."
            return {}

        inputs = self.get_dummy_inputs()
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
    FIRST_BLOCK_CACHE_CONFIG = {"threshold": 0.8}

    def _get_cache_config(self):
        return FirstBlockCacheConfig(**self.FIRST_BLOCK_CACHE_CONFIG)

    def test_first_block_cache_inference(self, expected_atol: float = 0.1):
        self._test_cache_inference(self._get_cache_config(), num_inference_steps=4, expected_atol=expected_atol)


@is_cache
class TaylorSeerCacheTesterMixin(CacheTesterMixin):
    TAYLORSEER_CACHE_CONFIG = {
        "cache_interval": 5,
        "disable_cache_before_step": 10,
        "max_order": 1,
        "taylor_factors_dtype": torch.bfloat16,
        "use_lite_mode": True,
    }

    def _get_cache_config(self):
        return TaylorSeerCacheConfig(**self.TAYLORSEER_CACHE_CONFIG)

    def test_taylorseer_cache_inference(self, expected_atol: float = 0.1):
        self._test_cache_inference(self._get_cache_config(), num_inference_steps=50, expected_atol=expected_atol)


@is_cache
class MagCacheTesterMixin(CacheTesterMixin):
    MAG_CACHE_CONFIG = {
        "threshold": 0.06,
        "max_skip_steps": 3,
        "retention_ratio": 0.2,
        "num_inference_steps": 50,
        "mag_ratios": torch.ones(50),
    }

    def _get_cache_config(self):
        return MagCacheConfig(**self.MAG_CACHE_CONFIG)

    def test_mag_cache_inference(self, expected_atol: float = 0.1):
        self._test_cache_inference(self._get_cache_config(), num_inference_steps=50, expected_atol=expected_atol)
