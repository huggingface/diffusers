# coding=utf-8
# Copyright 2026 HuggingFace Inc.
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
import logging
from typing import NamedTuple

import pytest
import torch

from diffusers.models.attention import AttentionMixin, AttentionModuleMixin
from diffusers.models.attention_dispatch import AttentionBackendName, _AttentionBackendRegistry, attention_backend
from diffusers.models.attention_processor import Attention, AttnProcessor
from diffusers.utils import is_kernels_available, is_torch_version

from ...testing_utils import assert_tensors_close, backend_empty_cache, is_attention, is_torch_compile, torch_device
from .utils import _maybe_cast_to_bf16


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level backend parameter sets for AttentionBackendTesterMixin
# ---------------------------------------------------------------------------

_CUDA_AVAILABLE = torch.cuda.is_available()

_PARAM_NATIVE_CUDNN = pytest.param(
    AttentionBackendName._NATIVE_CUDNN,
    id="native_cudnn",
    marks=pytest.mark.skipif(
        not _CUDA_AVAILABLE,
        reason="CUDA is required for _native_cudnn backend.",
    ),
)

_PARAM_FLASH_HUB = pytest.param(
    AttentionBackendName.FLASH_HUB,
    id="flash_hub",
    marks=[
        pytest.mark.skipif(not _CUDA_AVAILABLE, reason="CUDA is required for flash_hub backend."),
        pytest.mark.skipif(
            not is_kernels_available(),
            reason="`kernels` package is required for flash_hub backend. Install with `pip install kernels`.",
        ),
    ],
)

_PARAM_FLASH_3_HUB = pytest.param(
    AttentionBackendName._FLASH_3_HUB,
    id="flash_3_hub",
    marks=[
        pytest.mark.skipif(not _CUDA_AVAILABLE, reason="CUDA is required for _flash_3_hub backend."),
        pytest.mark.skipif(
            not is_kernels_available(),
            reason="`kernels` package is required for _flash_3_hub backend. Install with `pip install kernels`.",
        ),
    ],
)

_PARAM_FLASH_VARLEN_HUB = pytest.param(
    AttentionBackendName.FLASH_VARLEN_HUB,
    id="flash_varlen_hub",
    marks=[
        pytest.mark.skipif(not _CUDA_AVAILABLE, reason="CUDA is required for flash_varlen_hub backend."),
        pytest.mark.skipif(
            not is_kernels_available(),
            reason="`kernels` package is required for flash_varlen_hub backend. Install with `pip install kernels`.",
        ),
    ],
)

_PARAM_FLASH_3_VARLEN_HUB = pytest.param(
    AttentionBackendName._FLASH_3_VARLEN_HUB,
    id="flash_3_varlen_hub",
    marks=[
        pytest.mark.skipif(not _CUDA_AVAILABLE, reason="CUDA is required for _flash_3_varlen_hub backend."),
        pytest.mark.skipif(
            not is_kernels_available(),
            reason="`kernels` package is required for _flash_3_varlen_hub backend. Install with `pip install kernels`.",
        ),
    ],
)

# All backends under test.
_ALL_BACKEND_PARAMS = [
    _PARAM_NATIVE_CUDNN,
    _PARAM_FLASH_HUB,
    _PARAM_FLASH_3_HUB,
    _PARAM_FLASH_VARLEN_HUB,
    _PARAM_FLASH_3_VARLEN_HUB,
]

# Backends that perform non-deterministic operations and therefore cannot run when
# torch.use_deterministic_algorithms(True) is active (e.g. after enable_full_determinism()).
_NON_DETERMINISTIC_BACKENDS = {AttentionBackendName._NATIVE_CUDNN}


def _skip_if_backend_requires_nondeterminism(backend):
    """Skip at runtime when torch.use_deterministic_algorithms(True) blocks the backend.

    This check is intentionally deferred to test execution time because
    enable_full_determinism() is typically called at module level in test files *after*
    the module-level pytest.param() objects in this file have already been evaluated,
    making it impossible to catch via a collection-time skipif condition.
    """
    if backend in _NON_DETERMINISTIC_BACKENDS and torch.are_deterministic_algorithms_enabled():
        pytest.skip(
            f"Backend '{backend.value}' performs non-deterministic operations and cannot run "
            f"while `torch.use_deterministic_algorithms(True)` is active."
        )


class _FusionPath(NamedTuple):
    """How one of the two `fuse_qkv_projections()` implementations behaves.

    `attention_cls` is the attention base class that implementation walks: a model built on the *other* base class
    gets fused right past, which is why `test_fuse_unfuse_qkv_projections` checks it before anything else.
    `swaps_in_fused_processors` says what unfusing is expected to undo.
    """

    attention_cls: type
    swaps_in_fused_processors: bool


# `AttentionMixin.fuse_qkv_projections()`, inherited by Flux, Flux2, Chroma, Wan, ... It walks `AttentionModuleMixin`
# modules and leaves the processors alone, so `unfuse_qkv_projections()` deletes the fused layers again.
_SHARED_FUSION = _FusionPath(attention_cls=AttentionModuleMixin, swaps_in_fused_processors=False)

# The per-model implementations on UNets, SD3, PixArt, AuraFlow, CogVideoX, AutoencoderKL, ... They walk `Attention`
# modules, swap in a dedicated `Fused*` processor and stash the originals in `original_attn_processors`, so
# `unfuse_qkv_projections()` only puts those processors back - the fused layers stay on the modules.
_LEGACY_FUSION = _FusionPath(attention_cls=Attention, swaps_in_fused_processors=True)


def _fusion_path(model):
    """Which of the two `fuse_qkv_projections()` implementations this model inherits."""
    if type(model).fuse_qkv_projections is AttentionMixin.fuse_qkv_projections:
        return _SHARED_FUSION
    return _LEGACY_FUSION


def _fused_layer_name(module):
    """Name of the layer `fuse_projections()` creates: cross-attention fuses K/V only, self-attention fuses Q/K/V."""
    return "to_kv" if getattr(module, "is_cross_attention", False) else "to_qkv"


@is_attention
class AttentionTesterMixin:
    """
    Mixin class for testing attention processor and module functionality on models.

    Tests functionality from AttentionModuleMixin including:
        - Attention processor management (set/get)
        - QKV projection fusion/unfusion

    Expected from config mixin:
        - model_class: The model class to test

    Expected methods from config mixin:
        - get_init_dict(): Returns dict of arguments to initialize the model
        - get_dummy_inputs(): Returns dict of inputs to pass to the model forward pass

    Pytest mark: attention
        Use `pytest -m "not attention"` to skip these tests
    """

    def setup_method(self):
        gc.collect()
        backend_empty_cache(torch_device)

    def teardown_method(self):
        gc.collect()
        backend_empty_cache(torch_device)

    @torch.no_grad()
    def test_fuse_unfuse_qkv_projections(self, request, atol=1e-3, rtol=0):
        init_dict = self.get_init_dict()
        inputs_dict = self.get_dummy_inputs()
        model = self.model_class(**init_dict)
        model.to(torch_device)
        model.eval()

        if not hasattr(model, "fuse_qkv_projections"):
            pytest.skip("Model does not support QKV projection fusion.")

        fusion = _fusion_path(model)

        attention_modules = [
            module for module in model.modules() if isinstance(module, (AttentionModuleMixin, Attention))
        ]
        if not attention_modules:
            pytest.skip("Model has no attention modules to fuse.")

        walked_modules = [module for module in attention_modules if isinstance(module, fusion.attention_cls)]
        stranded_modules = [module for module in attention_modules if not isinstance(module, fusion.attention_cls)]

        if not walked_modules:
            # Every attention module in this model is of the base class the model's `fuse_qkv_projections()` does not
            # walk, so fusing silently does nothing. Marked xfail rather than skipped so that fixing the model shows
            # up as an XPASS instead of staying quietly green.
            request.node.add_marker(
                pytest.mark.xfail(
                    reason=(
                        f"{type(model).__name__}.fuse_qkv_projections() only walks "
                        f"`{fusion.attention_cls.__name__}` modules, but every attention module in this model is "
                        f"a `{type(stranded_modules[0]).__name__}` instance, so nothing is ever fused."
                    ),
                    strict=True,
                )
            )

        assert walked_modules, (
            f"{type(model).__name__}.fuse_qkv_projections() does not reach any of this model's attention modules."
        )

        fusable_modules = [module for module in walked_modules if getattr(module, "_supports_qkv_fusion", True)]
        if not fusable_modules:
            pytest.skip("Model's attention modules do not support QKV projection fusion.")

        output_before_fusion = model(**inputs_dict, return_dict=False)[0]
        processors_before_fusion = model.attn_processors

        model.fuse_qkv_projections()

        for module in fusable_modules:
            assert module.fused_projections, "fused_projections flag should be True"
            layer_name = _fused_layer_name(module)
            assert getattr(module, layer_name, None) is not None, (
                f"{type(module).__name__} should expose a fused `{layer_name}` layer after fusing."
            )

        processors_after_fusion = model.attn_processors
        assert len(processors_after_fusion) == len(processors_before_fusion), (
            "Fusing projections should not change the number of attention processors."
        )
        if fusion.swaps_in_fused_processors:
            for name, processor in processors_after_fusion.items():
                assert type(processor).__name__.startswith("Fused"), (
                    f"Processor {name} should be a fused processor after fusing, got {type(processor).__name__}."
                )

        output_after_fusion = model(**inputs_dict, return_dict=False)[0]

        assert_tensors_close(
            output_before_fusion,
            output_after_fusion,
            atol=atol,
            rtol=rtol,
            msg="Output should not change after fusing projections",
        )

        model.unfuse_qkv_projections()

        if fusion.swaps_in_fused_processors:
            # This path only restores the original processors; the fused layers themselves are left in place.
            processors_after_unfusion = model.attn_processors
            assert len(processors_after_unfusion) == len(processors_before_fusion), (
                "Unfusing projections should not change the number of attention processors."
            )
            for name, processor in processors_after_unfusion.items():
                assert type(processor) is type(processors_before_fusion[name]), (
                    f"Processor {name} should be restored to {type(processors_before_fusion[name]).__name__} "
                    f"after unfusing, got {type(processor).__name__}."
                )
        else:
            for module in fusable_modules:
                assert not hasattr(module, "to_qkv"), "to_qkv should be removed after unfusing"
                assert not hasattr(module, "to_kv"), "to_kv should be removed after unfusing"
                assert not module.fused_projections, "fused_projections flag should be False"

        output_after_unfusion = model(**inputs_dict, return_dict=False)[0]

        assert_tensors_close(
            output_before_fusion,
            output_after_unfusion,
            atol=atol,
            rtol=rtol,
            msg="Output should match original after unfusing projections",
        )

    def test_get_set_processor(self):
        init_dict = self.get_init_dict()
        model = self.model_class(**init_dict)
        model.to(torch_device)

        # Check if model has attention processors
        if not hasattr(model, "attn_processors"):
            pytest.skip("Model does not have attention processors.")

        # Test getting processors
        processors = model.attn_processors
        assert isinstance(processors, dict), "attn_processors should return a dict"
        assert len(processors) > 0, "Model should have at least one attention processor"

        # Test that all processors can be retrieved via get_processor
        for module in model.modules():
            if isinstance(module, AttentionModuleMixin):
                processor = module.get_processor()
                assert processor is not None, "get_processor should return a processor"

                # Test setting a new processor
                new_processor = AttnProcessor()
                module.set_processor(new_processor)
                retrieved_processor = module.get_processor()
                assert retrieved_processor is new_processor, "Retrieved processor should be the same as the one set"

    def test_attention_processor_dict(self):
        init_dict = self.get_init_dict()
        model = self.model_class(**init_dict)
        model.to(torch_device)

        if not hasattr(model, "set_attn_processor"):
            pytest.skip("Model does not support setting attention processors.")

        # Get current processors
        current_processors = model.attn_processors

        # Create a dict of new processors
        new_processors = {key: AttnProcessor() for key in current_processors.keys()}

        # Set processors using dict
        model.set_attn_processor(new_processors)

        # Verify all processors were set
        updated_processors = model.attn_processors
        for key in current_processors.keys():
            assert type(updated_processors[key]) == AttnProcessor, f"Processor {key} should be AttnProcessor"

    def test_attention_processor_count_mismatch_raises_error(self):
        init_dict = self.get_init_dict()
        model = self.model_class(**init_dict)
        model.to(torch_device)

        if not hasattr(model, "set_attn_processor"):
            pytest.skip("Model does not support setting attention processors.")

        # Get current processors
        current_processors = model.attn_processors

        # Create a dict with wrong number of processors
        wrong_processors = {list(current_processors.keys())[0]: AttnProcessor()}

        # Verify error is raised
        with pytest.raises(ValueError) as exc_info:
            model.set_attn_processor(wrong_processors)

        assert "number of processors" in str(exc_info.value).lower(), "Error should mention processor count mismatch"


@is_attention
class AttentionBackendTesterMixin:
    """
    Mixin class for testing attention backends on models. Following things are tested:

    1. Backends can be set with the `attention_backend` context manager and with
    `set_attention_backend()` method.
    2. SDPA outputs don't deviate too much from backend outputs.
    3. Backend works with (regional) compilation.
    4. Backends can be restored.

    Tests the backends using the model provided by the host test class. The backends to test
    are defined in `_ALL_BACKEND_PARAMS`.

    Expected from the host test class:
        - model_class: The model class to instantiate.

    Expected methods from the host test class:
        - get_init_dict(): Returns dict of kwargs to construct the model.
        - get_dummy_inputs(): Returns dict of inputs for the model's forward pass.

    Pytest mark: attention
        Use `pytest -m "not attention"` to skip these tests.
    """

    unsupported_attn_backends: list[str] = []

    def setup_method(self):
        gc.collect()
        backend_empty_cache(torch_device)

    def teardown_method(self):
        gc.collect()
        backend_empty_cache(torch_device)

    @torch.no_grad()
    @pytest.mark.parametrize("backend", _ALL_BACKEND_PARAMS)
    def test_set_attention_backend_matches_context_manager(self, backend):
        """set_attention_backend() and the attention_backend() context manager must yield identical outputs."""
        _skip_if_backend_requires_nondeterminism(backend)
        if backend.value in self.unsupported_attn_backends:
            pytest.skip(f"{backend.value} is not supported for this model.")

        init_dict = self.get_init_dict()
        inputs_dict = self.get_dummy_inputs()
        model = self.model_class(**init_dict)
        model.to(torch_device)
        model.eval()

        model, inputs_dict = _maybe_cast_to_bf16(backend, model, inputs_dict)

        with attention_backend(backend):
            ctx_output = model(**inputs_dict, return_dict=False)[0]

        initial_registry_backend, _ = _AttentionBackendRegistry.get_active_backend()

        model.set_attention_backend(backend.value)

        try:
            set_output = model(**inputs_dict, return_dict=False)[0]
        finally:
            model.reset_attention_backend()
            _AttentionBackendRegistry.set_active_backend(initial_registry_backend)

        assert_tensors_close(
            set_output,
            ctx_output,
            atol=0,
            rtol=0,
            msg=(
                f"Output from model.set_attention_backend('{backend.value}') should be identical "
                f"to the output from `with attention_backend('{backend.value}'):`."
            ),
        )

    @torch.no_grad()
    @pytest.mark.parametrize("backend", _ALL_BACKEND_PARAMS)
    def test_output_close_to_native(self, backend, atol=1e-2, rtol=1e-2):
        """All backends should produce model output numerically close to the native SDPA reference."""
        _skip_if_backend_requires_nondeterminism(backend)
        if backend.value in self.unsupported_attn_backends:
            pytest.skip(f"{backend.value} is not supported for this model.")

        init_dict = self.get_init_dict()
        inputs_dict = self.get_dummy_inputs()
        model = self.model_class(**init_dict)
        model.to(torch_device)
        model.eval()

        model, inputs_dict = _maybe_cast_to_bf16(backend, model, inputs_dict)

        with attention_backend(AttentionBackendName.NATIVE):
            native_output = model(**inputs_dict, return_dict=False)[0]

        initial_registry_backend, _ = _AttentionBackendRegistry.get_active_backend()

        try:
            model.set_attention_backend(backend.value)
        except Exception as e:
            logger.warning("Skipping test for backend '%s': %s", backend.value, e)
            pytest.skip(str(e))

        try:
            backend_output = model(**inputs_dict, return_dict=False)[0]
        finally:
            model.reset_attention_backend()
            _AttentionBackendRegistry.set_active_backend(initial_registry_backend)

        assert_tensors_close(
            backend_output,
            native_output,
            atol=atol,
            rtol=rtol,
            msg=f"Output from {backend} should be numerically close to native SDPA.",
        )

    @pytest.mark.parametrize("backend", _ALL_BACKEND_PARAMS)
    def test_context_manager_switches_and_restores_backend(self, backend):
        """attention_backend() should activate the requested backend and restore the previous one on exit."""
        initial_backend, _ = _AttentionBackendRegistry.get_active_backend()

        with attention_backend(backend):
            active_backend, _ = _AttentionBackendRegistry.get_active_backend()
            assert active_backend == backend, (
                f"Backend should be {backend} inside the context manager, got {active_backend}."
            )

        restored_backend, _ = _AttentionBackendRegistry.get_active_backend()
        assert restored_backend == initial_backend, (
            f"Backend should be restored to {initial_backend} after exiting the context manager, "
            f"got {restored_backend}."
        )

    @pytest.mark.parametrize("backend", _ALL_BACKEND_PARAMS)
    @is_torch_compile
    def test_compile(self, backend, atol=1e-2, rtol=1e-2):
        """
        `torch.compile` tests checking for recompilation, graph breaks, forward can run, etc.
        For speed, we use regional compilation here (`model.compile_repeated_blocks()`
        as opposed to `model.compile`).
        """
        _skip_if_backend_requires_nondeterminism(backend)
        if backend.value in self.unsupported_attn_backends:
            pytest.skip(f"{backend.value} is not supported for this model.")
        if getattr(self.model_class, "_repeated_blocks", None) is None:
            pytest.skip("Skipping tests as regional compilation is not supported.")

        if backend == AttentionBackendName.NATIVE and not is_torch_version(">=", "2.9.0"):
            pytest.xfail(
                "test_compile with the native backend requires torch >= 2.9.0 for stable "
                "fullgraph compilation with error_on_recompile=True."
            )

        init_dict = self.get_init_dict()
        inputs_dict = self.get_dummy_inputs()
        model = self.model_class(**init_dict)
        model.to(torch_device)
        model.eval()

        model, inputs_dict = _maybe_cast_to_bf16(backend, model, inputs_dict)

        with torch.no_grad(), attention_backend(AttentionBackendName.NATIVE):
            native_output = model(**inputs_dict, return_dict=False)[0]

        initial_registry_backend, _ = _AttentionBackendRegistry.get_active_backend()

        try:
            model.set_attention_backend(backend.value)
        except Exception as e:
            logger.warning("Skipping test for backend '%s': %s", backend.value, e)
            pytest.skip(str(e))

        try:
            model.compile_repeated_blocks(fullgraph=True)
            torch.compiler.reset()

            with (
                torch._inductor.utils.fresh_inductor_cache(),
                torch._dynamo.config.patch(error_on_recompile=True),
            ):
                with torch.no_grad():
                    compile_output = model(**inputs_dict, return_dict=False)[0]
                    model(**inputs_dict, return_dict=False)
        finally:
            model.reset_attention_backend()
            _AttentionBackendRegistry.set_active_backend(initial_registry_backend)

        assert_tensors_close(
            compile_output,
            native_output,
            atol=atol,
            rtol=rtol,
            msg=f"Compiled output with backend '{backend.value}' should be numerically close to eager native SDPA.",
        )
