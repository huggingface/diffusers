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

import collections
import contextlib
import gc
from unittest import mock

import pytest
import torch

from diffusers import ComponentsManager
from diffusers.models import ModelMixin
from diffusers.utils import is_accelerate_available
from diffusers.utils.accelerate_utils import apply_forward_hook

from ..testing_utils import (
    backend_empty_cache,
    require_accelerate,
    require_accelerator,
    simulate_accelerator_memory,
    torch_device,
)


if is_accelerate_available():
    from diffusers.modular_pipelines.components_manager import AutoOffloadStrategy


# The offload logic deals in bytes. We keep the test models tiny (a few KB of real
# parameters) and express every size as a multiple of this unit, then *simulate* the
# device's free memory at the same scale. This is what lets us exercise the offloading
# decisions deterministically instead of relying on the real free memory of the test
# hardware (an 80GB GPU never runs low on a handful of KB-sized models).
UNIT = 1024

# More free memory than any tiny test checkpoint could ever need, so the strategy never
# decides to offload. Used to assert the *negative*: no offloading without memory pressure.
_AMPLE_FREE_BYTES = 1024**4


def _peak_co_residency(events):
    """Largest number of models simultaneously on the device, replayed from the recorded moves."""
    resident, peak = set(), 0
    for event in events:
        if event.action == "onload":
            resident.add(event.model_id)
            peak = max(peak, len(resident))
        else:
            resident.discard(event.model_id)
    return peak


class DummyModel(ModelMixin):
    def __init__(self, footprint_bytes: int = UNIT):
        super().__init__()
        # A float32 parameter of `footprint_bytes // 4` elements weighs exactly
        # `footprint_bytes`, so callers control the reported size directly.
        self.weight = torch.nn.Parameter(torch.zeros(footprint_bytes // 4))

    def forward(self, x):
        return x + self.weight.sum()


class OOMOnceModel(DummyModel):
    """Raises a fake device OOM on its first `_oom_calls` forwards (or on every forward with `_repeated_oom`)."""

    _repeated_oom = False
    _oom_calls = 1

    def forward(self, x):
        self.forward_calls = getattr(self, "forward_calls", 0) + 1
        if self.forward_calls <= self._oom_calls or self._repeated_oom:
            raise torch.OutOfMemoryError("fake OOM")
        return super().forward(x)


class DecodeModel(DummyModel):
    """Autoencoder-style model: runs through a `decode` entry point (via `apply_forward_hook`, like the VAEs)
    instead of `forward`."""

    @apply_forward_hook
    def decode(self, x):
        return x + self.weight.sum()


class OOMOnceDecodeModel(DecodeModel):
    """`DecodeModel` that raises a fake device OOM on its first decode."""

    @apply_forward_hook
    def decode(self, x):
        self.decode_calls = getattr(self, "decode_calls", 0) + 1
        if self.decode_calls == 1:
            raise torch.OutOfMemoryError("fake OOM")
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


@contextlib.contextmanager
def _patch_memory_stats(device_module, free_bytes, total_bytes, cached_bytes=0):
    # `mem_get_info` returns `(free, total)` and is where the strategy learns how much
    # memory is free; the strategy additionally counts the allocator's reusable cache
    # (`memory_reserved - memory_allocated`) as available, so those are pinned too —
    # `cached_bytes` simulates reusable cache on top of `free_bytes`.
    with contextlib.ExitStack() as stack:
        stack.enter_context(mock.patch.object(device_module, "mem_get_info", return_value=(free_bytes, total_bytes)))
        if hasattr(device_module, "memory_reserved") and hasattr(device_module, "memory_allocated"):
            stack.enter_context(mock.patch.object(device_module, "memory_reserved", return_value=cached_bytes))
            stack.enter_context(mock.patch.object(device_module, "memory_allocated", return_value=0))
        yield


def _patch_cuda_mem_get_info(free_bytes: int, total_bytes: int = 80 * UNIT, cached_bytes: int = 0):
    # Strategy unit tests use a `cuda:0` execution-device *descriptor* (which needs no
    # real GPU), so they patch the `torch.cuda` memory introspection directly.
    return _patch_memory_stats(torch.cuda, free_bytes, total_bytes, cached_bytes=cached_bytes)


def _patch_free_memory(free_bytes: int, total_bytes: int = 80 * UNIT):
    # Integration tests run on the real `torch_device`; patch the memory introspection
    # on whichever backend module (cuda/xpu/...) actually backs it to simulate
    # arbitrary memory pressure.
    device_type = torch.device(torch_device).type
    device_module = getattr(torch, device_type, torch.cuda)
    return _patch_memory_stats(device_module, free_bytes, total_bytes)


@require_accelerate
class ComponentsManagerTesterMixin:
    """
    Common tests for `ComponentsManager` and its auto-offload strategy.
    """

    # A `cuda:0` device descriptor is enough to drive the strategy's device-type and
    # index logic; no real GPU is required because the memory readings are mocked.
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
    def _select_offload(
        self,
        *,
        incoming_footprint,
        free_bytes,
        hook_sizes,
        memory_reserve=UNIT,
        cached_bytes=0,
    ):
        strategy = AutoOffloadStrategy(memory_reserve=memory_reserve)
        hooks = [_FakeHook(model_id, self.get_dummy_model(fp)) for model_id, fp in hook_sizes.items()]
        incoming = self.get_dummy_model(incoming_footprint)
        with _patch_cuda_mem_get_info(free_bytes, cached_bytes=cached_bytes):
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

    def test_strategy_memory_reserve_changes_decision(self):
        # Same device free memory and incoming model; only the reserve differs.
        # A small reserve leaves enough room; a large reserve forces an offload. We check
        # this both with a single resident model and with several, to confirm the reserve
        # participates in the selection regardless of how many candidates exist.

        # Single candidate: free=5, incoming=3. reserve 1 -> usable 4 (fits); reserve 3 ->
        # usable 2, must free 1 -> offload "a".
        assert (
            self._select_offload(
                incoming_footprint=3 * UNIT,
                free_bytes=5 * UNIT,
                hook_sizes={"a": 2 * UNIT},
                memory_reserve=1 * UNIT,
            )
            == []
        )
        assert self._select_offload(
            incoming_footprint=3 * UNIT,
            free_bytes=5 * UNIT,
            hook_sizes={"a": 2 * UNIT},
            memory_reserve=3 * UNIT,
        ) == ["a"]

        # Multiple candidates: free=6, incoming=4. reserve 1 -> usable 5 (fits); reserve 3
        # -> usable 3, must free 1 -> smallest sufficient model "c" (1) is offloaded.
        multi_hooks = {"a": 3 * UNIT, "b": 2 * UNIT, "c": 1 * UNIT}
        assert (
            self._select_offload(
                incoming_footprint=4 * UNIT,
                free_bytes=6 * UNIT,
                hook_sizes=multi_hooks,
                memory_reserve=1 * UNIT,
            )
            == []
        )
        assert self._select_offload(
            incoming_footprint=4 * UNIT,
            free_bytes=6 * UNIT,
            hook_sizes=multi_hooks,
            memory_reserve=3 * UNIT,
        ) == ["c"]

    def test_strategy_reserve_zero_packs_tight(self):
        # `memory_reserve=0` uses every last byte for weights: incoming exactly equals
        # the free memory and still fits without offloading anything.
        selected = self._select_offload(
            incoming_footprint=4 * UNIT,
            free_bytes=4 * UNIT,
            hook_sizes={"a": 4 * UNIT},
            memory_reserve=0,
        )
        assert selected == []

    def test_strategy_allocator_cache_counts_as_available(self):
        # `mem_get_info` reports only 2 units free, but 6 units of the allocator's cache
        # are reusable: available = 2 + 6 = 8, minus 1 reserve -> the incoming 4 fits
        # with nothing offloaded. Without the cache add-back this would offload needlessly.
        selected = self._select_offload(
            incoming_footprint=4 * UNIT,
            free_bytes=2 * UNIT,
            hook_sizes={"a": 4 * UNIT},
            cached_bytes=6 * UNIT,
        )
        assert selected == []

    def test_strategy_memory_reserve_accepts_file_size_strings(self):
        # "3KiB" = 3 * 1024 bytes = a 3-unit reserve; free 4 - 3 = 1 usable but the
        # incoming needs 4, so the resident model must go.
        selected = self._select_offload(
            incoming_footprint=4 * UNIT,
            free_bytes=4 * UNIT,
            hook_sizes={"a": 4 * UNIT},
            memory_reserve="3KiB",
        )
        assert selected == ["a"]

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
        cm.enable_auto_cpu_offload(device=torch_device, memory_reserve=UNIT)
        try:
            assert next(model.parameters()).device.type == "cpu"
        finally:
            cm.disable_auto_cpu_offload()

    @require_accelerator
    def test_auto_offload_offloads_resident_model_under_memory_pressure(self):
        device_type = torch.device(torch_device).type
        cm = ComponentsManager()
        m1 = self.get_dummy_model(4 * UNIT)
        m2 = self.get_dummy_model(4 * UNIT)
        cm.add("m1", m1)
        cm.add("m2", m2)
        cm.enable_auto_cpu_offload(device=torch_device, memory_reserve=UNIT)
        try:
            # Both components start offloaded on the CPU.
            assert next(m1.parameters()).device.type == "cpu"
            assert next(m2.parameters()).device.type == "cpu"

            x = torch.randn(2, 4, device=torch_device)

            # Ample free memory: running m1 just moves it onto the device, offloading
            # nothing (m2 is not resident, so it is not even a candidate).
            with _patch_free_memory(70 * UNIT):
                m1(x)
            assert next(m1.parameters()).device.type == device_type

            # Memory pressure: usable = 4 - 1 = 3 but m2 needs 4, so the only resident
            # model (m1) must be offloaded back to the CPU to make room for m2.
            with _patch_free_memory(4 * UNIT):
                m2(x)
            assert next(m2.parameters()).device.type == device_type
            assert next(m1.parameters()).device.type == "cpu"
        finally:
            cm.disable_auto_cpu_offload()

    @require_accelerator
    def test_record_captures_what_the_strategy_saw(self):
        cm = ComponentsManager()
        m1 = self.get_dummy_model(4 * UNIT)
        m2 = self.get_dummy_model(4 * UNIT)
        cm.add("m1", m1)
        cm.add("m2", m2)
        cm.enable_auto_cpu_offload(device=torch_device, memory_reserve=UNIT)
        try:
            x = torch.randn(2, 4, device=torch_device)
            with _patch_free_memory(70 * UNIT):
                m1(x)
            # m2 does not fit alongside m1 (usable 4 - 1 = 3 < 4), so m1 is offloaded for it
            with _patch_free_memory(4 * UNIT):
                m2(x)

            onloads = [event for event in cm.offload_record.events if event.action == "onload"]
            names = [event.model_id.rsplit("_", 1)[0] for event in onloads]
            assert names == ["m1", "m2"]
            # The reading behind each decision is recorded, so a run can be explained after the fact.
            assert onloads[0].available_memory == 70 * UNIT
            assert onloads[1].available_memory == 4 * UNIT
            # the offload is its own event, attributed to the onload that caused it, with its own reading
            offload_event = next(event for event in cm.offload_record.events if event.action == "offload")
            assert offload_event.model_id == cm.model_hooks[0].model_id
            assert offload_event.reason == f"release_memory_for:{cm.model_hooks[1].model_id}"
            assert offload_event.available_memory == 4 * UNIT
            assert _peak_co_residency(cm.offload_record.events) == 1
            # the printed table shows one row per decision: m2's row carries the m1 offload it caused
            m2_row = next(line for line in repr(cm.offload_record).splitlines() if line.startswith("2 "))
            assert cm.model_hooks[1].model_id in m2_row and cm.model_hooks[0].model_id in m2_row
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
        cm.enable_auto_cpu_offload(device=torch_device, memory_reserve=UNIT)
        try:
            x = torch.randn(2, 4, device=torch_device)
            with _patch_free_memory(70 * UNIT):
                m1(x)
                m2(x)
            # Both fit comfortably, so neither gets offloaded.
            assert next(m1.parameters()).device.type == device_type
            assert next(m2.parameters()).device.type == device_type
        finally:
            cm.disable_auto_cpu_offload()

    @require_accelerator
    def test_record_captures_the_onload_sequence(self):
        cm = ComponentsManager()
        model_a = self.get_dummy_model(UNIT)
        model_b = self.get_dummy_model(4 * UNIT)
        cm.add("a", model_a)
        cm.add("b", model_b)
        cm.enable_auto_cpu_offload(device=torch_device, memory_reserve=UNIT)
        try:
            x = torch.randn(2, 4, device=torch_device)
            with _patch_free_memory(70 * UNIT):
                model_a(x)
                model_b(x)

            onloads = [event for event in cm.offload_record.events if event.action == "onload"]
            assert [event.model_id.rsplit("_", 1)[0] for event in onloads] == ["a", "b"]
            assert [event.model_size for event in onloads] == [UNIT, 4 * UNIT]
            assert [event.action for event in cm.offload_record.events] == ["onload", "onload"]
        finally:
            cm.disable_auto_cpu_offload()

    @require_accelerator
    def test_record_tracks_peak_co_residency(self):
        cm = ComponentsManager()
        m1 = self.get_dummy_model(4 * UNIT)
        m2 = self.get_dummy_model(4 * UNIT)
        cm.add("m1", m1)
        cm.add("m2", m2)
        cm.enable_auto_cpu_offload(device=torch_device, memory_reserve=UNIT)
        try:
            x = torch.randn(2, 4, device=torch_device)
            with _patch_free_memory(70 * UNIT):
                m1(x)
                m2(x)
            assert _peak_co_residency(cm.offload_record.events) == 2

            # an offload brings residency back down, so a later onload does not raise the peak:
            # m3 needs 4 but only 4 - 1 = 3 is usable, costing one of the resident models its spot
            m3 = self.get_dummy_model(4 * UNIT)
            cm.add("m3", m3)
            with _patch_free_memory(4 * UNIT):
                m3(x)
            assert _peak_co_residency(cm.offload_record.events) == 2
        finally:
            cm.disable_auto_cpu_offload()

    @require_accelerator
    def test_record_is_bounded_and_clearable(self):
        cm = ComponentsManager()
        m1 = self.get_dummy_model(4 * UNIT)
        m2 = self.get_dummy_model(4 * UNIT)
        cm.add("m1", m1)
        cm.add("m2", m2)
        cm.enable_auto_cpu_offload(device=torch_device, memory_reserve=UNIT)
        cm.offload_record.events = collections.deque(maxlen=2)
        try:
            x = torch.randn(2, 4, device=torch_device)
            # under pressure the two models trade places on every call, producing a steady
            # stream of events; the record keeps only the last `maxlen`
            with _patch_free_memory(4 * UNIT):
                m1(x)
                m2(x)
                m1(x)

            assert len(cm.offload_record.events) == 2

            cm.offload_record.clear()
            assert len(cm.offload_record.events) == 0
            assert "nothing recorded yet" in repr(cm.offload_record)
        finally:
            cm.disable_auto_cpu_offload()

    @require_accelerator
    def test_record_survives_disable(self):
        cm = ComponentsManager()
        m1 = self.get_dummy_model(4 * UNIT)
        cm.add("m1", m1)
        cm.enable_auto_cpu_offload(device=torch_device, memory_reserve=UNIT)
        x = torch.randn(2, 4, device=torch_device)
        with _patch_free_memory(70 * UNIT):
            m1(x)

        cm.disable_auto_cpu_offload()

        # the record is the post-mortem of the run that just ended
        events = list(cm.offload_record.events)
        assert sum(event.action == "onload" for event in events) == 1
        # disabling is recorded too: the resident model's final move back to the CPU
        assert events[-1].action == "offload"
        assert events[-1].reason == "offloading_disabled"
        assert events[-1].available_memory is not None


class TestComponentsManager(ComponentsManagerTesterMixin):
    pass


@require_accelerate
class TestOffloadHookBehavior:
    """CPU-level tests of the hook machinery: OOM retry and non-disruptive add/remove."""

    @pytest.fixture(autouse=True)
    def _fake_cpu_mem_get_info(self):
        # These tests exercise the hook machinery only, on a cpu execution device — but enabling
        # offloading requires `mem_get_info`, and the OOM-retry records read it, neither of which cpu
        # implements, so provide an ample fake one for the whole test.
        with mock.patch.object(
            torch.cpu, "mem_get_info", create=True, return_value=(_AMPLE_FREE_BYTES, _AMPLE_FREE_BYTES)
        ):
            yield

    def _manager(self, components, **offload_kwargs):
        manager = ComponentsManager()
        for name, component in components.items():
            manager.add(name, component)
        manager.enable_auto_cpu_offload(device="cpu", **offload_kwargs)
        return manager

    @staticmethod
    def _offloaded_names(manager):
        """The models the record shows going back to the CPU, in order, by the name they were added under."""
        # model ids are `<name>_<id(component)>`
        return [
            event.model_id.rsplit("_", 1)[0] for event in manager.offload_record.events if event.action == "offload"
        ]

    def test_oom_retry_offloads_smallest_and_reruns(self):
        model = OOMOnceModel()
        smallest = DummyModel(UNIT)
        larger = DummyModel(4 * UNIT)
        manager = self._manager({"model": model, "larger": larger, "smallest": smallest})

        out = model(torch.zeros(2, 4))
        assert out.shape == (2, 4)
        assert model.forward_calls == 2
        # one OOM costs the smallest resident model, not everything on the device
        assert self._offloaded_names(manager) == ["smallest"]

    def test_oom_retry_escalates_one_model_at_a_time(self):
        model = OOMOnceModel()
        model._oom_calls = 2
        smallest = DummyModel(UNIT)
        larger = DummyModel(4 * UNIT)
        manager = self._manager({"model": model, "larger": larger, "smallest": smallest})

        out = model(torch.zeros(2, 4))
        assert out.shape == (2, 4)
        assert model.forward_calls == 3
        assert self._offloaded_names(manager) == ["smallest", "larger"]

    def test_oom_reraises_when_nothing_to_offload(self):
        model = OOMOnceModel()
        self._manager({"model": model})

        with pytest.raises(torch.OutOfMemoryError, match="group offloading"):
            model(torch.zeros(2, 4))

    def test_oom_reraises_once_everything_is_offloaded(self):
        model = OOMOnceModel()
        model._repeated_oom = True
        smallest = DummyModel(UNIT)
        larger = DummyModel(4 * UNIT)
        manager = self._manager({"model": model, "larger": larger, "smallest": smallest})

        with pytest.raises(torch.OutOfMemoryError, match="group offloading"):
            model(torch.zeros(2, 4))
        # one attempt per model offloaded, plus the initial one
        assert model.forward_calls == 3
        assert self._offloaded_names(manager) == ["smallest", "larger"]

    def test_retry_on_oom_false_leaves_the_forward_alone(self):
        model = OOMOnceModel()
        other = DummyModel()
        manager = self._manager({"model": model, "other": other}, retry_on_oom=False)

        with pytest.raises(torch.OutOfMemoryError, match="fake OOM"):
            model(torch.zeros(2, 4))
        assert model.forward_calls == 1
        assert all(not user_hook.hook._wrapped_methods for user_hook in manager.model_hooks)

    def test_oom_retry_covers_apply_forward_hook_entry_points(self):
        # Autoencoders run through `encode`/`decode` (via `apply_forward_hook`), not `forward` - an OOM
        # there must be retried just like one raised inside `forward`, first call included.
        model = OOMOnceDecodeModel()
        smallest = DummyModel(UNIT)
        manager = self._manager({"model": model, "smallest": smallest})

        out = model.decode(torch.zeros(2, 4))
        assert out.shape == (2, 4)
        assert model.decode_calls == 2
        assert self._offloaded_names(manager) == ["smallest"]

    def test_record_captures_oom_and_the_escalation(self):
        model = OOMOnceModel()
        smallest = DummyModel(UNIT)
        manager = self._manager({"model": model, "smallest": smallest})

        model(torch.zeros(2, 4))

        # the OOM shows up as the offload it caused, attributed to the model that ran out of memory
        offloads = [event for event in manager.offload_record.events if event.action == "offload"]
        assert len(offloads) == 1
        offload_event = offloads[0]
        assert offload_event.model_size == UNIT
        assert offload_event.model_id.rsplit("_", 1)[0] == "smallest"
        assert offload_event.reason == f"oom_retry:{manager.model_hooks[0].model_id}"

    def test_add_does_not_rebuild_existing_hooks(self):
        model_a = DummyModel()
        manager = self._manager({"a": model_a})
        hooks_before = list(manager.model_hooks)

        model_b = DummyModel()
        manager.add("b", model_b)

        assert len(manager.model_hooks) == 2
        assert manager.model_hooks[0] is hooks_before[0], "existing hook was rebuilt"
        hook_a, hook_b = manager.model_hooks
        assert hook_b.hook.other_hooks == [hook_a]
        assert hook_a.hook.other_hooks == [hook_b]

    def test_remove_detaches_only_that_hook(self):
        model_a = DummyModel()
        model_b = DummyModel()
        manager = self._manager({"a": model_a, "b": model_b})
        hook_a = manager.model_hooks[0]
        b_id = [cid for cid in manager.components if cid.startswith("b_")][0]

        manager.remove(b_id)

        assert manager.model_hooks == [hook_a]
        assert hook_a.hook.other_hooks == []
        assert not hasattr(model_b, "_hf_hook")
        # model a still hooked and functional
        assert model_a(torch.zeros(2, 4)).shape == (2, 4)


class ModularPipelineOffloadTesterMixin:
    """
    Auto-CPU-offload tests for a modular pipeline's components.
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
        memory, recording every move in `cm.offload_record`: an `"onload"` event per model
        that ran, and an `"offload"` event (`reason="release_memory_for:<onloader>"`) per model
        offloaded to make room.
        """
        cm = ComponentsManager()
        pipe = self.get_pipeline(components_manager=cm)
        cm.enable_auto_cpu_offload(device=torch_device, memory_reserve=0)

        with _patch_free_memory(free_bytes):
            output = pipe(**self.get_dummy_inputs(), output=self.output_name)
        return cm, list(cm.offload_record.events), output

    @require_accelerate
    @require_accelerator
    def test_auto_cpu_offload_serializes_models_under_memory_pressure(self):
        # Zero simulated free memory: every model that runs must first offload whatever is
        # currently resident (comfy-style serialized execution).
        cm, events, _ = self._run_offloaded(free_bytes=0)
        try:
            distinct_models = {event.model_id for event in events if event.action == "onload"}
            if len(distinct_models) < 2:
                pytest.skip("pipeline has fewer than two offloadable model components")

            # Offloading actually fired (at least one model was pushed off the device).
            assert any(event.action == "offload" for event in events), "expected at least one offload"

            # Sequencing: models run one at a time, never two co-resident on the device.
            peak = _peak_co_residency(events)
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
        # every load, but it must never decide to offload anything.
        cm, events, _ = self._run_offloaded(free_bytes=_AMPLE_FREE_BYTES)
        try:
            distinct_models = {event.model_id for event in events if event.action == "onload"}
            if len(distinct_models) < 2:
                pytest.skip("pipeline has fewer than two offloadable model components")

            # Nothing was ever offloaded...
            assert all(event.action == "onload" for event in events), "no model should be offloaded"

            # ...and models accumulate on the device instead of being serialized.
            peak = _peak_co_residency(events)
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


@require_accelerator
class TestSimulateAcceleratorMemory:
    """
    Validates the `simulate_accelerator_memory` testing util itself against the real device: unlike
    `_patch_memory_stats` (which pins scripted readings for hardware-independent unit tests), the
    util wraps `mem_get_info` so real allocations show up live on a simulated smaller card, and
    hard-caps the allocator so overshooting the simulated capacity raises a real OOM.
    """

    MB = 2**20

    def setup_method(self):
        gc.collect()
        backend_empty_cache(torch_device)

    def teardown_method(self):
        gc.collect()
        backend_empty_cache(torch_device)

    @staticmethod
    def _device_module():
        return getattr(torch, torch.device(torch_device).type)

    def test_readings_translate_to_simulated_card(self):
        device_module = self._device_module()
        free, real_total = device_module.mem_get_info()
        # Simulate a card with 64MB of headroom on top of whatever the device currently holds.
        sim_total = (real_total - free) + 64 * self.MB

        with simulate_accelerator_memory(sim_total, device=torch_device, hard=False):
            free0, total0 = device_module.mem_get_info()
            assert total0 == sim_total
            assert free0 <= 64 * self.MB

            # A real allocation is visible on the simulated card: free drops by at least its size.
            x = torch.zeros(8 * self.MB, dtype=torch.float32, device=torch_device)  # 32MB
            free1, total1 = device_module.mem_get_info()
            assert total1 == sim_total
            assert free1 <= free0 - 32 * self.MB
            del x

        # Exiting the context restores the real readings.
        assert device_module.mem_get_info()[1] == real_total

    def test_run_ooms_when_simulated_card_is_too_small(self):
        device_module = self._device_module()
        model = DummyModel(footprint_bytes=64 * self.MB)
        x = torch.randn(4, device=torch_device)

        # Measure what the run actually needs on the device (weights + forward).
        device_module.reset_peak_memory_stats()
        model.to(torch_device)
        model(x)
        requirement = device_module.max_memory_allocated()
        model.to("cpu")
        backend_empty_cache(torch_device)

        try:
            # Control: a simulated card with comfortable headroom runs the model fine.
            free, real_total = device_module.mem_get_info()
            with simulate_accelerator_memory((real_total - free) + 2 * requirement, device=torch_device):
                model.to(torch_device)
                model(x)
            model.to("cpu")
            backend_empty_cache(torch_device)

            # A card with only half the requirement left cannot: the hard cap turns the
            # overshoot into a real device OOM instead of using the extra physical memory.
            free, real_total = device_module.mem_get_info()
            with simulate_accelerator_memory((real_total - free) + requirement // 2, device=torch_device):
                with pytest.raises(torch.OutOfMemoryError):
                    model.to(torch_device)
                    model(x)
        finally:
            model.to("cpu")
            backend_empty_cache(torch_device)
