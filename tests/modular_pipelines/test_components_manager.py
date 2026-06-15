# Copyright 2025 The HuggingFace Team. All rights reserved.
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
from unittest import mock

import pytest
import torch

from diffusers import ComponentsManager
from diffusers.models import ModelMixin
from diffusers.utils import is_accelerate_available

from ..testing_utils import backend_empty_cache, require_accelerate, require_accelerator, torch_device


if is_accelerate_available():
    from diffusers.modular_pipelines.components_manager import AutoOffloadStrategy


# The offload logic deals in bytes. We keep the test models tiny (a few KB of real
# parameters) and express every size as a multiple of this unit, then *simulate* the
# device's free memory at the same scale. This is what lets us exercise the offloading
# decisions deterministically instead of relying on the real free memory of the test
# hardware (an 80GB GPU never runs low on a handful of KB-sized models).
UNIT = 1024


class DummyModel(ModelMixin):
    def __init__(self, footprint_bytes: int = UNIT):
        super().__init__()
        # A float32 parameter of `footprint_bytes // 4` elements weighs exactly
        # `footprint_bytes`, so callers control the reported size directly.
        self.weight = torch.nn.Parameter(torch.zeros(footprint_bytes // 4))

    def forward(self, x):
        return x + self.weight.sum()


class _FakeHook:
    """Minimal stand-in for `UserCustomOffloadHook` in strategy-level unit tests.

    `AutoOffloadStrategy` only reads `hook.model_id` and
    `hook.model.get_memory_footprint()`, so we avoid attaching real accelerate hooks
    (which would move modules around) and keep the logic test pure.
    """

    def __init__(self, model_id: str, model: torch.nn.Module):
        self.model_id = model_id
        self.model = model


def _patch_cuda_mem_get_info(free_bytes: int, total_bytes: int = 80 * UNIT):
    # Strategy unit tests use a `cuda:0` execution-device *descriptor* (which needs no
    # real GPU), so they patch `torch.cuda.mem_get_info` directly.
    return mock.patch.object(torch.cuda, "mem_get_info", return_value=(free_bytes, total_bytes))


def _patch_free_memory(free_bytes: int, total_bytes: int = 80 * UNIT):
    # Integration tests run on the real `torch_device`; patch `mem_get_info` on
    # whichever backend module (cuda/xpu/...) actually backs it. `mem_get_info` returns
    # `(free, total)` and is the single point where the strategy learns how much memory
    # is available, so patching it simulates arbitrary memory pressure.
    device_type = torch.device(torch_device).type
    device_module = getattr(torch, device_type, torch.cuda)
    return mock.patch.object(device_module, "mem_get_info", return_value=(free_bytes, total_bytes))


@require_accelerate
class ComponentsManagerTesterMixin:
    """Common tests for `ComponentsManager` and its auto-offload strategy.

    The whole suite requires accelerate (the offload machinery is built on it), hence
    the class-level `require_accelerate`. Tests are ordered so the hardware-independent
    ones (strategy unit tests, which mock `mem_get_info`) come first, and the few that
    need a real accelerator are grouped together at the end behind `require_accelerator`.

    Subclasses may override `get_dummy_model` to drive the same offload logic with a
    different `ModelMixin` type.
    """

    # A `cuda:0` device descriptor is enough to drive the strategy's device-type and
    # index logic; no real GPU is required because `mem_get_info` is mocked.
    strategy_execution_device = torch.device("cuda:0")

    def setup_method(self):
        # Mirror `ModularPipelineTesterMixin` cleanup so this mixin stays interchangeable
        # in the MRO when stacked into a pipeline test class.
        torch.compiler.reset()
        gc.collect()
        backend_empty_cache(torch_device)

    def teardown_method(self):
        torch.compiler.reset()
        gc.collect()
        backend_empty_cache(torch_device)

    def get_dummy_model(self, footprint_bytes: int = UNIT) -> ModelMixin:
        return DummyModel(footprint_bytes=footprint_bytes)

    # ------------------------------------------------------------------
    # AutoOffloadStrategy unit tests (hardware-independent)
    # ------------------------------------------------------------------
    def _select_offload(self, *, incoming_footprint, free_bytes, hook_sizes, memory_reserve_margin=UNIT):
        strategy = AutoOffloadStrategy(memory_reserve_margin=memory_reserve_margin)
        hooks = [_FakeHook(model_id, self.get_dummy_model(fp)) for model_id, fp in hook_sizes.items()]
        incoming = self.get_dummy_model(incoming_footprint)
        with _patch_cuda_mem_get_info(free_bytes):
            selected = strategy(
                hooks=hooks,
                model_id="incoming",
                model=incoming,
                execution_device=self.strategy_execution_device,
            )
        return sorted(hook.model_id for hook in selected)

    def test_strategy_no_offload_when_memory_is_sufficient(self):
        # 70 units free, 1 reserved -> 69 usable, incoming needs 4: nothing to offload.
        selected = self._select_offload(
            incoming_footprint=4 * UNIT,
            free_bytes=70 * UNIT,
            hook_sizes={"a": 5 * UNIT, "b": 3 * UNIT},
        )
        assert selected == []

    def test_strategy_offloads_minimal_single_model(self):
        # usable = 4 - 1 = 3, incoming needs 6 -> must free 3.
        # Smallest combination that frees >= 3 is "b" (exactly 3) on its own.
        selected = self._select_offload(
            incoming_footprint=6 * UNIT,
            free_bytes=4 * UNIT,
            hook_sizes={"a": 5 * UNIT, "b": 3 * UNIT, "c": 2 * UNIT},
        )
        assert selected == ["b"]

    def test_strategy_offloads_smallest_sufficient_combination(self):
        # usable = 4 - 1 = 3, incoming needs 8 -> must free 5.
        # No single model frees 5 (max is 4), so the smallest sufficient combination of
        # models is chosen: a (4) + c (1) = 5.
        selected = self._select_offload(
            incoming_footprint=8 * UNIT,
            free_bytes=4 * UNIT,
            hook_sizes={"a": 4 * UNIT, "b": 4 * UNIT, "c": 1 * UNIT},
        )
        assert selected == ["a", "c"]

    def test_strategy_offloads_all_when_freeing_enough_is_impossible(self):
        # incoming needs more than the sum of everything on device -> offload all.
        selected = self._select_offload(
            incoming_footprint=11 * UNIT,
            free_bytes=1 * UNIT,
            hook_sizes={"a": 5 * UNIT, "b": 3 * UNIT, "c": 2 * UNIT},
        )
        assert selected == ["a", "b", "c"]

    def test_strategy_no_hooks_returns_empty(self):
        selected = self._select_offload(
            incoming_footprint=11 * UNIT,
            free_bytes=0,
            hook_sizes={},
        )
        assert selected == []

    def test_strategy_memory_reserve_margin_changes_decision(self):
        # Same device free memory and incoming model; only the reserve margin differs.
        # A small margin leaves enough room; a large margin forces an offload. We check
        # this both with a single resident model and with several, to confirm the margin
        # participates in the selection regardless of how many candidates exist.

        # Single candidate: free=5, incoming=3. margin 1 -> usable 4 (fits); margin 3 ->
        # usable 2, must free 1 -> offload "a".
        assert (
            self._select_offload(
                incoming_footprint=3 * UNIT,
                free_bytes=5 * UNIT,
                hook_sizes={"a": 2 * UNIT},
                memory_reserve_margin=1 * UNIT,
            )
            == []
        )
        assert self._select_offload(
            incoming_footprint=3 * UNIT,
            free_bytes=5 * UNIT,
            hook_sizes={"a": 2 * UNIT},
            memory_reserve_margin=3 * UNIT,
        ) == ["a"]

        # Multiple candidates: free=6, incoming=4. margin 1 -> usable 5 (fits); margin 3
        # -> usable 3, must free 1 -> smallest sufficient model "c" (1) is offloaded.
        multi_hooks = {"a": 3 * UNIT, "b": 2 * UNIT, "c": 1 * UNIT}
        assert (
            self._select_offload(
                incoming_footprint=4 * UNIT,
                free_bytes=6 * UNIT,
                hook_sizes=multi_hooks,
                memory_reserve_margin=1 * UNIT,
            )
            == []
        )
        assert self._select_offload(
            incoming_footprint=4 * UNIT,
            free_bytes=6 * UNIT,
            hook_sizes=multi_hooks,
            memory_reserve_margin=3 * UNIT,
        ) == ["c"]

    def test_strategy_raises_for_model_without_memory_footprint(self):
        strategy = AutoOffloadStrategy(memory_reserve_margin=UNIT)
        hooks = [_FakeHook("a", self.get_dummy_model(2 * UNIT))]
        # A bare nn.Module does not implement get_memory_footprint().
        with _patch_cuda_mem_get_info(1 * UNIT):
            with pytest.raises(AttributeError):
                strategy(
                    hooks=hooks,
                    model_id="incoming",
                    model=torch.nn.Linear(4, 4),
                    execution_device=self.strategy_execution_device,
                )

    # ------------------------------------------------------------------
    # Registry tests (hardware-independent)
    # ------------------------------------------------------------------
    def test_add_and_get_one(self):
        cm = ComponentsManager()
        model = self.get_dummy_model()
        component_id = cm.add("unet", model)
        assert component_id in cm.components
        assert cm.get_one(name="unet") is model
        assert cm.get_one(component_id=component_id) is model

    def test_add_same_component_twice_reuses_id(self):
        cm = ComponentsManager()
        model = self.get_dummy_model()
        first_id = cm.add("unet", model)
        second_id = cm.add("unet", model)
        assert first_id == second_id
        assert len(cm.components) == 1

    def test_remove(self):
        cm = ComponentsManager()
        component_id = cm.add("unet", self.get_dummy_model())
        cm.remove(component_id)
        assert component_id not in cm.components

    def test_get_model_info_reports_size(self):
        cm = ComponentsManager()
        model = self.get_dummy_model(footprint_bytes=2 * UNIT)
        component_id = cm.add("unet", model)
        info = cm.get_model_info(component_id, fields="size_gb")
        assert info["size_gb"] == model.get_memory_footprint() / (1024**3)

    # ------------------------------------------------------------------
    # Auto-offload integration tests (require an accelerator)
    # ------------------------------------------------------------------
    @require_accelerator
    def test_auto_offload_starts_with_all_components_on_cpu(self):
        cm = ComponentsManager()
        model = self.get_dummy_model(4 * UNIT)
        cm.add("m1", model)
        cm.enable_auto_cpu_offload(device=torch_device, memory_reserve_margin=UNIT)
        try:
            assert next(model.parameters()).device.type == "cpu"
        finally:
            cm.disable_auto_cpu_offload()

    @require_accelerator
    def test_auto_offload_evicts_resident_model_under_memory_pressure(self):
        device_type = torch.device(torch_device).type
        cm = ComponentsManager()
        m1 = self.get_dummy_model(4 * UNIT)
        m2 = self.get_dummy_model(4 * UNIT)
        cm.add("m1", m1)
        cm.add("m2", m2)
        cm.enable_auto_cpu_offload(device=torch_device, memory_reserve_margin=UNIT)
        try:
            # Both components start offloaded on the CPU.
            assert next(m1.parameters()).device.type == "cpu"
            assert next(m2.parameters()).device.type == "cpu"

            x = torch.randn(2, 4, device=torch_device)

            # Ample free memory: running m1 just moves it onto the device, evicting
            # nothing (m2 is not resident, so it is not even a candidate).
            with _patch_free_memory(70 * UNIT):
                m1(x)
            assert next(m1.parameters()).device.type == device_type

            # Memory pressure: usable = 4 - 1 = 3 but m2 needs 4, so the only resident
            # model (m1) must be evicted back to the CPU to make room for m2.
            with _patch_free_memory(4 * UNIT):
                m2(x)
            assert next(m2.parameters()).device.type == device_type
            assert next(m1.parameters()).device.type == "cpu"
        finally:
            cm.disable_auto_cpu_offload()

    @require_accelerator
    def test_auto_offload_keeps_models_resident_when_memory_is_ample(self):
        device_type = torch.device(torch_device).type
        cm = ComponentsManager()
        m1 = self.get_dummy_model(4 * UNIT)
        m2 = self.get_dummy_model(4 * UNIT)
        cm.add("m1", m1)
        cm.add("m2", m2)
        cm.enable_auto_cpu_offload(device=torch_device, memory_reserve_margin=UNIT)
        try:
            x = torch.randn(2, 4, device=torch_device)
            with _patch_free_memory(70 * UNIT):
                m1(x)
                m2(x)
            # Both fit comfortably, so neither gets evicted.
            assert next(m1.parameters()).device.type == device_type
            assert next(m2.parameters()).device.type == device_type
        finally:
            cm.disable_auto_cpu_offload()


class TestComponentsManager(ComponentsManagerTesterMixin):
    pass


# More free memory than any tiny test checkpoint could ever need, so the strategy never
# decides to offload. Used to assert the *negative*: no eviction without memory pressure.
_AMPLE_FREE_BYTES = 1024**4


class ModularPipelineOffloadTesterMixin:
    """Auto-CPU-offload tests for a *real* modular pipeline's components.

    Designed to be mixed into a pipeline test class alongside
    `ModularPipelineTesterMixin`, whose `get_pipeline`, `get_dummy_inputs` and
    `output_name` it relies on. It registers the pipeline's real components in a
    `ComponentsManager` and mocks `mem_get_info` to control the *simulated* free memory,
    so the offloading path can be exercised on any hardware (on an 80GB GPU with tiny
    test checkpoints nothing would otherwise ever get offloaded).
    """

    @staticmethod
    def _managed_models(cm):
        """The registered components that the offloader actually manages (parameterized
        `nn.Module`s)."""
        models = []
        for component in cm.components.values():
            if isinstance(component, torch.nn.Module) and next(component.parameters(), None) is not None:
                models.append(component)
        return models

    @staticmethod
    def _is_resident(model):
        return next(model.parameters()).device.type == torch.device(torch_device).type

    def _run_offloaded(self, free_bytes):
        """
        Run the pipeline with auto offload on and `free_bytes` of *simulated* device
        memory, recording every offload decision the strategy makes.

        Each record is `{"incoming", "resident_before", "offloaded"}` (lists of model
        ids), captured by spying on `AutoOffloadStrategy.__call__`, which the hooks call
        each time a model is about to be moved onto the device.
        """
        cm = ComponentsManager()
        pipe = self.get_pipeline(components_manager=cm)
        cm.enable_auto_cpu_offload(device=torch_device, memory_reserve_margin=0)

        records = []
        original_call = AutoOffloadStrategy.__call__

        def spy_call(strategy, hooks, model_id, model, execution_device):
            selected = original_call(
                strategy, hooks=hooks, model_id=model_id, model=model, execution_device=execution_device
            )
            records.append(
                {
                    "incoming": model_id,
                    "resident_before": [hook.model_id for hook in hooks],
                    "offloaded": [hook.model_id for hook in selected],
                }
            )
            return selected

        with _patch_free_memory(free_bytes), mock.patch.object(AutoOffloadStrategy, "__call__", spy_call):
            output = pipe(**self.get_dummy_inputs(), output=self.output_name)
        return cm, records, output

    @staticmethod
    def _peak_co_residency(records):
        """
        Largest number of models simultaneously on the device, reconstructed from the
        strategy's view of residency just before each load.
        """
        peak = 0
        for record in records:
            resident = (set(record["resident_before"]) - set(record["offloaded"])) | {record["incoming"]}
            peak = max(peak, len(resident))
        return peak

    @require_accelerate
    @require_accelerator
    def test_auto_cpu_offload_serializes_models_under_memory_pressure(self):
        # Zero simulated free memory: every model that runs must first evict whatever is
        # currently resident (comfy-style serialized execution).
        cm, records, _ = self._run_offloaded(free_bytes=0)
        try:
            distinct_models = {record["incoming"] for record in records}
            if len(distinct_models) < 2:
                pytest.skip("pipeline has fewer than two offloadable model components")

            # Offloading actually fired (at least one eviction happened).
            assert any(record["offloaded"] for record in records), "expected at least one eviction"

            # Sequencing: models run one at a time, never two co-resident on the device.
            peak = self._peak_co_residency(records)
            assert peak == 1, f"expected serialized execution under pressure, saw {peak} models co-resident"

            # Device placement after the run: at most the last-run model stays on the
            # accelerator, and at least one managed model was pushed back to the CPU.
            models = self._managed_models(cm)
            resident = [m for m in models if self._is_resident(m)]
            assert len(resident) <= 1
            assert any(not self._is_resident(m) for m in models), "expected some model offloaded to CPU"
        finally:
            cm.disable_auto_cpu_offload()

    @require_accelerate
    @require_accelerator
    def test_auto_cpu_offload_keeps_models_resident_without_memory_pressure(self):
        # Negative case: with ample simulated memory the strategy is still consulted on
        # every load, but it must never decide to evict anything.
        cm, records, _ = self._run_offloaded(free_bytes=_AMPLE_FREE_BYTES)
        try:
            distinct_models = {record["incoming"] for record in records}
            if len(distinct_models) < 2:
                pytest.skip("pipeline has fewer than two offloadable model components")

            # Nothing was ever offloaded...
            assert all(record["offloaded"] == [] for record in records), "no model should be evicted"

            # ...and models accumulate on the device instead of being serialized.
            peak = self._peak_co_residency(records)
            assert peak >= 2, f"expected models to co-reside without pressure, saw peak {peak}"

            models = self._managed_models(cm)
            assert sum(self._is_resident(m) for m in models) >= 2, "expected multiple models resident on device"
        finally:
            cm.disable_auto_cpu_offload()

    @require_accelerate
    @require_accelerator
    def test_auto_cpu_offload_inference_consistent_under_memory_pressure(self, expected_max_diff=1e-3):
        # Sensible results: forcing offload (zero simulated free memory) must not change
        # the output relative to an ordinary, non-offloaded run.
        base_pipe = self.get_pipeline().to(torch_device)
        baseline = base_pipe(**self.get_dummy_inputs(), output=self.output_name)

        cm, _, offloaded = self._run_offloaded(free_bytes=0)
        try:
            max_diff = torch.abs(baseline - offloaded).max()
            assert max_diff < expected_max_diff, f"offloaded output diverged from baseline (max diff {max_diff})"
        finally:
            cm.disable_auto_cpu_offload()
