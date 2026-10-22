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

import torch

from diffusers import ComponentsManager
from diffusers.models import ModelMixin
from diffusers.utils import is_accelerate_available

from ..testing_utils import backend_empty_cache, require_accelerate, require_accelerator, torch_device
from .testing_utils import patch_free_memory


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
    """
    Minimal stand-in for `UserCustomOffloadHook` in strategy-level unit tests.

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


@require_accelerate
class TestComponentsManager:
    """
    Tests for `ComponentsManager` and its auto-offload strategy.
    """

    # A `cuda:0` device descriptor is enough to drive the strategy's device-type and
    # index logic; no real GPU is required because `mem_get_info` is mocked.
    strategy_execution_device = torch.device("cuda:0")

    def setup_method(self):
        # Free VRAM before/after each test; the auto-offload integration tests below reason
        # about device residency, so they must not inherit another test's allocations.
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
            with patch_free_memory(70 * UNIT):
                m1(x)
            assert next(m1.parameters()).device.type == device_type

            # Memory pressure: usable = 4 - 1 = 3 but m2 needs 4, so the only resident
            # model (m1) must be evicted back to the CPU to make room for m2.
            with patch_free_memory(4 * UNIT):
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
            with patch_free_memory(70 * UNIT):
                m1(x)
                m2(x)
            # Both fit comfortably, so neither gets evicted.
            assert next(m1.parameters()).device.type == device_type
            assert next(m2.parameters()).device.type == device_type
        finally:
            cm.disable_auto_cpu_offload()
