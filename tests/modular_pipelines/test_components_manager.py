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

import contextlib
import gc
from collections import Counter
from unittest import mock

import pytest
import torch

from diffusers import ComponentsManager
from diffusers.models import ModelMixin
from diffusers.utils import is_accelerate_available
from diffusers.utils.accelerate_utils import apply_forward_hook

from ..testing_utils import (
    backend_empty_cache,
    backend_max_memory_allocated,
    backend_reset_peak_memory_stats,
    require_accelerate,
    require_accelerator,
    simulate_accelerator_memory,
    slow,
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


class DummyModel(ModelMixin):
    def __init__(self, footprint_bytes: int = UNIT):
        super().__init__()
        # A float32 parameter of `footprint_bytes // 4` elements weighs exactly
        # `footprint_bytes`, so callers control the reported size directly.
        self.weight = torch.nn.Parameter(torch.zeros(footprint_bytes // 4))

    def forward(self, x):
        return x + self.weight.sum()


class DummyDecodeModel(DummyModel):
    """Autoencoder-style model: runs through a `decode` entry point (via `apply_forward_hook`, like the VAEs)
    instead of `forward`."""

    @apply_forward_hook
    def decode(self, x):
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
def _patch_memory_stats(free_bytes, total_bytes=80 * UNIT, device_module=None):
    # `mem_get_info` returns `(free, total)` and is where the offloader learns how much memory is
    # free; it additionally counts the allocator's reusable cache (`memory_reserved -
    # memory_allocated`) as available, so those are pinned to zero — `free_bytes` is the whole
    # story. `device_module=None` targets the backend of the real `torch_device` (integration
    # tests); the strategy unit tests pass `torch.cuda` to match their `cuda:0` device descriptor.
    if device_module is None:
        device_module = getattr(torch, torch.device(torch_device).type, torch.cuda)
    with contextlib.ExitStack() as stack:
        # `create=True` lets this fake `mem_get_info` onto backends that don't implement it (cpu)
        stack.enter_context(
            mock.patch.object(device_module, "mem_get_info", create=True, return_value=(free_bytes, total_bytes))
        )
        if hasattr(device_module, "memory_reserved") and hasattr(device_module, "memory_allocated"):
            stack.enter_context(mock.patch.object(device_module, "memory_reserved", return_value=0))
            stack.enter_context(mock.patch.object(device_module, "memory_allocated", return_value=0))
        yield


def _simulate_card_with_headroom(headroom_bytes):
    """
    A `simulate_accelerator_memory` card sized to whatever the device currently holds plus
    `headroom_bytes` — so tests can reason in absolute headroom, independent of the node's state.
    """
    device_module = getattr(torch, torch.device(torch_device).type)
    free, real_total = device_module.mem_get_info()
    return simulate_accelerator_memory((real_total - free) + headroom_bytes, device=torch_device)


class TestComponentsManagerRegistry:
    """Registry behavior only: add / look up / remove components. No offloading, no hardware."""

    def test_add_and_get_one(self):
        cm = ComponentsManager()
        model = DummyModel(UNIT)
        component_id = cm.add("unet", model)
        assert component_id in cm.components
        assert cm.get_one(name="unet") is model
        assert cm.get_one(component_id=component_id) is model

    def test_add_same_component_twice_reuses_id(self):
        cm = ComponentsManager()
        model = DummyModel(UNIT)
        first_id = cm.add("unet", model)
        second_id = cm.add("unet", model)
        assert first_id == second_id
        assert len(cm.components) == 1

    def test_remove(self):
        cm = ComponentsManager()
        component_id = cm.add("unet", DummyModel(UNIT))
        cm.remove(component_id)
        assert component_id not in cm.components

    def test_get_model_info_reports_size(self):
        cm = ComponentsManager()
        model = DummyModel(footprint_bytes=2 * UNIT)
        component_id = cm.add("unet", model)
        info = cm.get_model_info(component_id, fields="size_gb")
        assert info["size_gb"] == model.get_memory_footprint() / (1024**3)


@require_accelerate
class TestAutoOffloadStrategy:
    """AutoOffloadStrategy unit tests: no real GPU — a `cuda:0` device *descriptor* plus scripted
    memory readings are enough to drive the selection logic."""

    strategy_execution_device = torch.device("cuda:0")

    def _select_offload(self, *, incoming_footprint, free_bytes, hook_sizes, memory_reserve=UNIT):
        strategy = AutoOffloadStrategy(memory_reserve=memory_reserve)
        hooks = [_FakeHook(model_id, DummyModel(fp)) for model_id, fp in hook_sizes.items()]
        incoming = DummyModel(incoming_footprint)
        with _patch_memory_stats(free_bytes, device_module=torch.cuda):
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

    @require_accelerator
    def test_readings_translate_to_simulated_card(self):
        # this test validates the simulated readings themselves, so it reads `mem_get_info` raw
        device_module = getattr(torch, torch.device(torch_device).type)
        free, real_total = device_module.mem_get_info()
        # Simulate a card with 64MB of headroom on top of whatever the device currently holds.
        sim_total = (real_total - free) + 64 * self.MB

        with simulate_accelerator_memory(sim_total, device=torch_device, hard=False):
            free0, total0 = device_module.mem_get_info()
            assert total0 == sim_total
            assert free0 <= 64 * self.MB

            # A real allocation is visible on the simulated card: free drops by at least its size.
            # (8M float32 elements x 4 bytes each = 32MB; "at least" because the allocator grabs
            # memory from the driver in larger blocks)
            x = torch.zeros(8 * self.MB, dtype=torch.float32, device=torch_device)
            free1, total1 = device_module.mem_get_info()
            assert total1 == sim_total
            assert free1 <= free0 - 32 * self.MB
            del x

        # Exiting the context restores the real readings.
        assert device_module.mem_get_info()[1] == real_total

    @require_accelerator
    def test_run_ooms_when_simulated_card_is_too_small(self):
        model = DummyModel(footprint_bytes=64 * self.MB)
        x = torch.randn(4, device=torch_device)

        # Measure what the run actually needs on the device (weights + forward).
        backend_reset_peak_memory_stats(torch_device)
        model.to(torch_device)
        model(x)
        requirement = backend_max_memory_allocated(torch_device)
        model.to("cpu")
        backend_empty_cache(torch_device)

        try:
            # Control: a simulated card with comfortable headroom runs the model fine.
            with _simulate_card_with_headroom(2 * requirement):
                model.to(torch_device)
                model(x)
            model.to("cpu")
            backend_empty_cache(torch_device)

            # A card with only half the requirement left cannot: the hard cap turns the
            # overshoot into a real device OOM instead of using the extra physical memory.
            with _simulate_card_with_headroom(requirement // 2):
                with pytest.raises(torch.OutOfMemoryError):
                    model.to(torch_device)
                    model(x)
        finally:
            model.to("cpu")
            backend_empty_cache(torch_device)


class TestAutoOffload:
    """Auto-offload behavior on a real accelerator: genuine device moves. Memory pressure is
    simulated two ways — `_patch_memory_stats` scripts the *readings* the strategy decides on, and
    `simulate_accelerator_memory` (hard cap) makes the device behave like a smaller card, so the
    OOM-retry tests recover from real `torch.OutOfMemoryError`s."""

    MB = 2**20

    def setup_method(self):
        torch.compiler.reset()
        gc.collect()
        backend_empty_cache(torch_device)

    def teardown_method(self):
        torch.compiler.reset()
        gc.collect()
        backend_empty_cache(torch_device)

    # ------------------------------------------------------------------
    # OOM retry (real OOMs on a simulated small card)
    # ------------------------------------------------------------------

    @require_accelerate
    @require_accelerator
    def test_oom_retry_offloads_a_resident_model_and_reruns(self):
        device_type = torch.device(torch_device).type

        # the model that OOMs mid-forward: its 32MB of weights fit the card easily
        model = DummyModel(32 * self.MB)
        torch.nn.init.normal_(model.weight)  # non-zero weights, so the equivalence check below has teeth
        # the bystander whose eviction is what saves the retry: offloading it frees 64MB
        resident = DummyModel(64 * self.MB)

        x = torch.zeros(10 * self.MB, device=torch_device)  # the forward's output weighs 40MB
        x_tiny = torch.zeros(4, device=torch_device)

        # baseline output from a plain, unhooked run - the retried run must reproduce it exactly.
        # (empty the cache afterwards: blocks this run leaves behind would otherwise be reusable
        # free headroom the simulated card below doesn't know about)
        with torch.no_grad():
            model.to(torch_device)
            baseline = model(x)
            model.to("cpu")
        backend_empty_cache(torch_device)

        manager = ComponentsManager()
        manager.add("model", model)
        manager.add("resident", resident)
        # reserve 0 lets the weights pack tight, so it is the forward's own output that overshoots
        manager.enable_auto_cpu_offload(device=torch_device, memory_reserve=0)
        try:
            # both models are actually under the offloader before anything is measured
            assert all(hasattr(m, "_hf_hook") for m in (model, resident))

            # A card where both models' weights fit (64 + 32 = 96 < 120) but the forward's output
            # does not fit on top (96 + 40 = 136 > 120): the run really OOMs, and offloading the
            # resident model (frees 64) is really what lets the retry succeed (32 + 40 = 72 < 120).
            with _simulate_card_with_headroom(120 * self.MB), torch.no_grad():
                # resident onloads (64MB, plenty of room) and runs; it stays on the device
                resident(x_tiny)
                # model onloads (96MB resident now, still fits) -> its forward OOMs allocating the
                # 40MB output -> the retry offloads `resident`, freeing 64MB -> the forward re-runs
                # from the original input and fits
                out = model(x)

            # surviving an OOM must not change the result
            assert torch.allclose(out, baseline, atol=1e-5)
            # the OOM cost the resident model its spot; the running model stayed
            assert next(resident.parameters()).device.type == "cpu"
            assert next(model.parameters()).device.type == device_type
            # The retry's offload is the last event: the OOM struck the forward's output allocation,
            # after the model's weights had already onloaded - so the re-run needs no new onload,
            # only the resident model's eviction.
            last_record = list(manager.offload_record.events)[-1]
            assert last_record.action == "offload"
            assert last_record.model_id == manager.model_hooks[1].model_id
            assert last_record.reason == f"oom_retry:{manager.model_hooks[0].model_id}"
        finally:
            manager.disable_auto_cpu_offload()

    @require_accelerate
    @require_accelerator
    def test_oom_reraises_when_nothing_to_offload(self):
        # 32MB of weights fit the 64MB card on their own; only the forward's output overshoots
        model = DummyModel(32 * self.MB)
        manager = ComponentsManager()
        manager.add("model", model)
        manager.enable_auto_cpu_offload(device=torch_device, memory_reserve=0)
        try:
            x = torch.zeros(12 * self.MB, device=torch_device)  # the forward's output weighs 48MB
            # The weights (32) fit the 64MB card, weights + output (32 + 48 = 80) never can - and
            # with no other managed model to offload, the retry gives up with advice.
            with _simulate_card_with_headroom(64 * self.MB), torch.no_grad():
                with pytest.raises(torch.OutOfMemoryError, match="group offloading"):
                    model(x)
        finally:
            manager.disable_auto_cpu_offload()

    @require_accelerate
    @require_accelerator
    def test_oom_retry_covers_apply_forward_hook_entry_points(self):
        # Autoencoders run through `encode`/`decode` (via `apply_forward_hook`), which fire the
        # offload hook but route around `forward` - an OOM there must be retried just like one
        # raised inside `forward`, first call included.
        device_type = torch.device(torch_device).type
        # the decoder that OOMs: 32MB of weights, called through `decode` instead of `forward`
        model = DummyDecodeModel(32 * self.MB)
        # the bystander whose 64MB eviction is what lets the retried decode fit
        resident = DummyModel(64 * self.MB)
        manager = ComponentsManager()
        manager.add("model", model)
        manager.add("resident", resident)
        manager.enable_auto_cpu_offload(device=torch_device, memory_reserve=0)
        try:
            x = torch.zeros(10 * self.MB, device=torch_device)  # the decode's output weighs 40MB
            x_tiny = torch.zeros(4, device=torch_device)
            # Same card arithmetic as the forward-based retry test above: the weights fit
            # (64 + 32 = 96 < 120), the decode's 40MB output on top does not (136 > 120).
            with _simulate_card_with_headroom(120 * self.MB), torch.no_grad():
                # resident onloads and runs (recorded: onload); it stays on the device
                resident(x_tiny)
                # `decode` fires `pre_forward` via `apply_forward_hook`: model onloads (recorded:
                # onload) -> decode OOMs allocating its output -> the retry offloads `resident`
                # (recorded: offload, reason "oom_retry:<model>") -> decode re-runs and fits
                out = model.decode(x)

            assert out.shape == x.shape
            assert next(resident.parameters()).device.type == "cpu"
            assert next(model.parameters()).device.type == device_type
        finally:
            manager.disable_auto_cpu_offload()

    @require_accelerate
    def test_enable_rejects_devices_without_memory_introspection(self):
        # every offloading decision starts from `mem_get_info`, which cpu doesn't implement -
        # enabling on such a backend must fail loudly, not silently misbehave
        cm = ComponentsManager()
        cm.add("m1", DummyModel(UNIT))
        with pytest.raises(NotImplementedError, match="mem_get_info"):
            cm.enable_auto_cpu_offload(device="cpu")

    @require_accelerate
    @require_accelerator
    def test_offloading_reduces_peak_memory(self):
        # The feature's headline claim, measured: the tighter the card, the more the offloader
        # serializes the models, and the lower the real peak device memory.
        m1 = DummyModel(64 * self.MB)
        m2 = DummyModel(64 * self.MB)
        m3 = DummyModel(64 * self.MB)
        models = (m1, m2, m3)
        x = torch.randn(4, device=torch_device)

        # baseline: no offloading, all three resident together -> the peak covers 192MB of weights
        backend_reset_peak_memory_stats(torch_device)
        with torch.no_grad():
            for model in models:
                model.to(torch_device)
                model(x)
        baseline_peak = backend_max_memory_allocated(torch_device)
        for model in models:
            model.to("cpu")

        manager = ComponentsManager()
        manager.add("m1", m1)
        manager.add("m2", m2)
        manager.add("m3", m3)

        def peak_on_card(headroom_bytes):
            # a fresh enable puts every model back on the CPU, so phases don't leak into each other
            manager.enable_auto_cpu_offload(device=torch_device, memory_reserve=0)
            backend_empty_cache(torch_device)
            backend_reset_peak_memory_stats(torch_device)
            with _simulate_card_with_headroom(headroom_bytes), torch.no_grad():
                for model in models:
                    model(x)
            peak = backend_max_memory_allocated(torch_device)
            manager.disable_auto_cpu_offload()
            return peak

        try:
            # a 160MB card fits two models (128) but not three (192): the third load offloads one
            # resident, capping the peak at two models
            partial_peak = peak_on_card(160 * self.MB)
            # an 80MB card fits only one model at a time: fully serialized execution
            serialized_peak = peak_on_card(80 * self.MB)

            # the hard cap doubles as a silent assert: neither run OOMed, so the offloader really
            # kept each within its card
            assert serialized_peak < partial_peak < baseline_peak
        finally:
            manager.disable_auto_cpu_offload()

    @require_accelerate
    @require_accelerator
    def test_add_does_not_rebuild_existing_hooks(self):
        model_a = DummyModel(UNIT)
        manager = ComponentsManager()
        manager.add("a", model_a)
        manager.enable_auto_cpu_offload(device=torch_device)
        try:
            hooks_before = list(manager.model_hooks)

            model_b = DummyModel(UNIT)
            manager.add("b", model_b)

            assert len(manager.model_hooks) == 2
            assert manager.model_hooks[0] is hooks_before[0], "existing hook was rebuilt"
            hook_a, hook_b = manager.model_hooks
            assert hook_b.hook.other_hooks == [hook_a]
            assert hook_a.hook.other_hooks == [hook_b]
        finally:
            manager.disable_auto_cpu_offload()

    @require_accelerate
    @require_accelerator
    def test_remove_detaches_only_that_hook(self):
        model_a = DummyModel(UNIT)
        model_b = DummyModel(UNIT)
        manager = ComponentsManager()
        manager.add("a", model_a)
        b_id = manager.add("b", model_b)
        manager.enable_auto_cpu_offload(device=torch_device)
        try:
            hook_a = manager.model_hooks[0]

            manager.remove(b_id)

            assert manager.model_hooks == [hook_a]
            assert hook_a.hook.other_hooks == []
            assert not hasattr(model_b, "_hf_hook")
            # model a keeps its hook: running it still moves it to the device
            assert hasattr(model_a, "_hf_hook")
            out = model_a(torch.zeros(2, 4, device=torch_device))
            assert out.shape == (2, 4)
            assert next(model_a.parameters()).device.type == torch.device(torch_device).type
        finally:
            manager.disable_auto_cpu_offload()

    # ------------------------------------------------------------------
    # Integration (real accelerator, simulated memory pressure)
    # ------------------------------------------------------------------
    @require_accelerate
    @require_accelerator
    def test_auto_offload_starts_with_all_components_on_cpu(self):
        cm = ComponentsManager()
        model = DummyModel(4 * UNIT)
        cm.add("m1", model)
        cm.enable_auto_cpu_offload(device=torch_device, memory_reserve=UNIT)
        try:
            assert next(model.parameters()).device.type == "cpu"
        finally:
            cm.disable_auto_cpu_offload()

    @require_accelerate
    @require_accelerator
    def test_auto_offload_offloads_resident_model_under_memory_pressure(self):
        device_type = torch.device(torch_device).type
        cm = ComponentsManager()
        m1 = DummyModel(4 * UNIT)
        m2 = DummyModel(4 * UNIT)
        cm.add("m1", m1)
        cm.add("m2", m2)
        cm.enable_auto_cpu_offload(device=torch_device, memory_reserve=UNIT)
        try:
            # Both components start offloaded on the CPU.
            assert next(m1.parameters()).device.type == "cpu"
            assert next(m2.parameters()).device.type == "cpu"

            x = torch.randn(2, 4, device=torch_device)

            # Ample free memory: running m1 just moves it onto the device.
            with _patch_memory_stats(70 * UNIT):
                m1(x)
            assert next(m1.parameters()).device.type == device_type

            # Memory pressure: usable = 4 - 1 = 3 but m2 needs 4, so the only resident model (m1)
            # must be offloaded back to the CPU to make room.
            with _patch_memory_stats(4 * UNIT):
                m2(x)
            assert next(m2.parameters()).device.type == device_type
            assert next(m1.parameters()).device.type == "cpu"

            # Ample again: m1 re-loads next to m2 without pushing it off - models co-reside
            # whenever memory allows.
            with _patch_memory_stats(70 * UNIT):
                m1(x)
            assert next(m1.parameters()).device.type == device_type
            assert next(m2.parameters()).device.type == device_type
        finally:
            cm.disable_auto_cpu_offload()

        # Disabling moves the residents back to the CPU.
        assert next(m1.parameters()).device.type == "cpu"
        assert next(m2.parameters()).device.type == "cpu"

    @require_accelerate
    @require_accelerator
    def test_record_captures_what_the_strategy_saw(self):
        cm = ComponentsManager()
        m1 = DummyModel(4 * UNIT)
        m2 = DummyModel(4 * UNIT)
        cm.add("m1", m1)
        cm.add("m2", m2)
        cm.enable_auto_cpu_offload(device=torch_device, memory_reserve=UNIT)
        try:
            m1_id = cm.model_hooks[0].model_id
            m2_id = cm.model_hooks[1].model_id

            x = torch.randn(2, 4, device=torch_device)
            with _patch_memory_stats(70 * UNIT):
                m1(x)  # -> records[0]: m1's onload
            # m2 does not fit alongside m1 (usable 4 - 1 = 3 < 4), so m1 is offloaded for it
            with _patch_memory_stats(4 * UNIT):
                m2(x)  # -> records[1] + records[2]: m1's offload, then m2's onload

            records = list(cm.offload_record.events)
            assert len(records) == 3

            # records[0]: m1 onloads into ample memory - the 70-unit reading it decided on is kept
            assert records[0].action == "onload"
            assert records[0].model_id == m1_id
            assert records[0].model_size == 4 * UNIT
            assert records[0].available_memory == 70 * UNIT
            assert records[0].reason is None

            # records[1]: m1 offloads to make room, attributed to m2 with the 4-unit reading that
            # forced the decision (usable 4 - 1 = 3 < m2's 4)
            assert records[1].action == "offload"
            assert records[1].model_id == m1_id
            assert records[1].reason == f"release_memory_for:{m2_id}"
            assert records[1].available_memory == 4 * UNIT

            # records[2]: m2 onloads into the freed memory
            assert records[2].action == "onload"
            assert records[2].model_id == m2_id
            assert records[2].model_size == 4 * UNIT
            assert records[2].available_memory == 4 * UNIT

            # records[3]: disabling offloading is recorded too - m2, the one model still resident,
            # moves back to the CPU
            cm.disable_auto_cpu_offload()
            records = list(cm.offload_record.events)
            assert len(records) == 4
            assert records[3].action == "offload"
            assert records[3].model_id == m2_id
            assert records[3].reason == "offloading_disabled"
            assert records[3].available_memory is not None

            # every event carries the device's cumulative peak-memory reading, which never decreases
            peaks = [record.peak_memory for record in records]
            assert all(peak is not None for peak in peaks)
            assert peaks == sorted(peaks)

            # the printed table shows one row per decision: m2's row carries the m1 offload it caused
            m2_row = next(line for line in repr(cm.offload_record).splitlines() if line.startswith("2 "))
            assert m2_id in m2_row and m1_id in m2_row
        finally:
            cm.disable_auto_cpu_offload()

    @require_accelerate
    @require_accelerator
    def test_add_model_after_a_run_keeps_residents_in_place(self):
        device_type = torch.device(torch_device).type
        cm = ComponentsManager()
        m1 = DummyModel(4 * UNIT)
        m2 = DummyModel(4 * UNIT)
        cm.add("m1", m1)
        cm.add("m2", m2)
        cm.enable_auto_cpu_offload(device=torch_device, memory_reserve=UNIT)
        try:
            x = torch.randn(2, 4, device=torch_device)
            with _patch_memory_stats(70 * UNIT):
                m1(x)
                m2(x)
            assert next(m1.parameters()).device.type == device_type
            assert next(m2.parameters()).device.type == device_type

            # Adding a model mid-run hooks it into the managed set without disturbing it: the
            # residents stay where they are and the newcomer starts on the CPU.
            m3 = DummyModel(4 * UNIT)
            cm.add("m3", m3)
            hook1, hook2, hook3 = cm.model_hooks
            assert hook3.hook.other_hooks == [hook1, hook2]
            assert hook1.hook.other_hooks == [hook2, hook3]
            assert hook2.hook.other_hooks == [hook1, hook3]
            assert next(m1.parameters()).device.type == device_type
            assert next(m2.parameters()).device.type == device_type
            assert next(m3.parameters()).device.type == "cpu"

            # With enough memory the newcomer loads next to the residents.
            with _patch_memory_stats(70 * UNIT):
                m3(x)
            assert next(m1.parameters()).device.type == device_type
            assert next(m2.parameters()).device.type == device_type
            assert next(m3.parameters()).device.type == device_type
        finally:
            cm.disable_auto_cpu_offload()

    @require_accelerate
    @require_accelerator
    def test_added_model_can_displace_residents_under_pressure(self):
        device_type = torch.device(torch_device).type
        cm = ComponentsManager()
        m1 = DummyModel(4 * UNIT)
        m2 = DummyModel(4 * UNIT)
        cm.add("m1", m1)
        cm.add("m2", m2)
        cm.enable_auto_cpu_offload(device=torch_device, memory_reserve=UNIT)
        try:
            x = torch.randn(2, 4, device=torch_device)
            with _patch_memory_stats(70 * UNIT):
                m1(x)
                m2(x)

            m3 = DummyModel(4 * UNIT)
            cm.add("m3", m3)
            # Insufficient memory: m3 needs 4 but usable is 4 - 1 = 3, so the strategy frees one
            # resident (the first-added, with all sizes equal) to make room.
            with _patch_memory_stats(4 * UNIT):
                m3(x)
            assert next(m3.parameters()).device.type == device_type
            assert next(m1.parameters()).device.type == "cpu"
            assert next(m2.parameters()).device.type == device_type
        finally:
            cm.disable_auto_cpu_offload()

    @require_accelerate
    @require_accelerator
    def test_remove_model_after_a_run_leaves_others_resident(self):
        device_type = torch.device(torch_device).type
        cm = ComponentsManager()
        m1 = DummyModel(4 * UNIT)
        m2 = DummyModel(4 * UNIT)
        m3 = DummyModel(4 * UNIT)
        cm.add("m1", m1)
        m2_id = cm.add("m2", m2)
        cm.add("m3", m3)
        cm.enable_auto_cpu_offload(device=torch_device, memory_reserve=UNIT)
        try:
            x = torch.randn(2, 4, device=torch_device)
            with _patch_memory_stats(70 * UNIT):
                m1(x)
                m2(x)
                m3(x)

            # Removing a component detaches only that component: it lands on the CPU unhooked,
            # the others keep their hooks (now linked to each other only) and stay resident.
            cm.remove(m2_id)
            assert not hasattr(m2, "_hf_hook")
            assert next(m2.parameters()).device.type == "cpu"
            hook1, hook3 = cm.model_hooks
            assert hook1.hook.other_hooks == [hook3]
            assert hook3.hook.other_hooks == [hook1]
            assert next(m1.parameters()).device.type == device_type
            assert next(m3.parameters()).device.type == device_type

            # ...and the remaining models still run.
            with _patch_memory_stats(70 * UNIT):
                m1(x)
            assert next(m1.parameters()).device.type == device_type
        finally:
            cm.disable_auto_cpu_offload()


class ModularPipelineOffloadTesterMixin:
    """
    Auto-CPU-offload tests for a modular pipeline's components.
    """

    @staticmethod
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

        with _patch_memory_stats(free_bytes):
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
            peak = self._peak_co_residency(events)
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
            peak = self._peak_co_residency(events)
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


class ModularPipelineIntegrationTesterMixin:
    """
    Slow integration tests that run a modular pipeline against its real checkpoint. Auto
    offloading is one aspect covered here: `test_auto_cpu_offload_on_cards` verifies that it
    behaves exactly as declared on a few simulated card sizes, and that it leaves the output
    untouched. The tests need a device large enough to run the pipeline normally - the simulated
    cards only cap the memory readings.

    Subclasses set `repo_id`, `get_inputs()`, and `offload_cards` - the expected offloading
    behavior on each simulated card:

        offload_cards = {
            "16GB": {
                "offload": {"text_encoder": 1, "transformer": 1},  # model -> times offloaded
                "oom": {},  # model whose forward ran out of memory -> times
                "final_device": {"text_encoder": "cpu", "transformer": "cpu", "vae": "cuda"},
            },
        }

    A model missing from "offload" must never have been offloaded, `"oom": {}` means no forward
    pass needed OOM recovery, and "final_device" is where every model sits right after the run
    (compared by device type). To discover the expected behavior for a new pipeline or card, run
    it once and read the record:

        cm.enable_auto_cpu_offload(device="cuda")
        with simulate_accelerator_memory("16GB", hard=False):
            pipe(**inputs, output="images")
        print(cm.offload_record)  # the decisions, one row each -> "offload" and "oom" counts
        for component_id, component in cm.components.items():
            print(component_id, component.device)  # -> "final_device"

    check the decisions make sense for the component sizes, then transcribe them into the spec.
    Pick card sizes with some margin from the fits/doesn't-fit boundary: the offloader reads live
    free memory, which activations and allocator cache have already reduced, so a card within
    ~2GB of the weights + reserve line can behave differently across environments.
    """

    repo_id = None
    torch_dtype = torch.bfloat16
    output_name = "images"

    @property
    def offload_cards(self):
        # {card label: expected behavior} - see the class docstring
        raise NotImplementedError

    def get_inputs(self):
        raise NotImplementedError

    def get_pipeline(self, components_manager=None):
        from diffusers import ModularPipeline

        pipe = ModularPipeline.from_pretrained(self.repo_id, components_manager=components_manager)
        pipe.load_components(torch_dtype=self.torch_dtype)
        return pipe

    def setup_method(self):
        gc.collect()
        backend_empty_cache(torch_device)

    def teardown_method(self):
        gc.collect()
        backend_empty_cache(torch_device)

    @slow
    @require_accelerate
    @require_accelerator
    def test_auto_cpu_offload_on_cards(self, memory_reserve="3GB", expected_max_diff=5e-2):
        from accelerate.utils.modeling import convert_file_size_to_int

        # baseline: an ordinary, fully-resident run on the real card
        baseline_pipe = self.get_pipeline()
        baseline_pipe.to(torch_device)
        baseline = baseline_pipe(**self.get_inputs(), output=self.output_name)
        # move off the device explicitly - `del` alone frees nothing while any reference to the
        # components survives, and stale weights would occupy every simulated card below
        baseline_pipe.to("cpu")
        del baseline_pipe
        gc.collect()
        backend_empty_cache(torch_device)

        cm = ComponentsManager()
        pipe = self.get_pipeline(components_manager=cm)

        for label, expected in self.offload_cards.items():
            card = convert_file_size_to_int(label)
            cm.enable_auto_cpu_offload(device=torch_device, memory_reserve=memory_reserve)
            # Soft simulation: only the memory *readings* see the card, so heavy activations can
            # never OOM the test - the decisions are verified against the declared spec instead.
            with simulate_accelerator_memory(card, device=torch_device, hard=False):
                output = pipe(**self.get_inputs(), output=self.output_name)
            events = list(cm.offload_record.events)
            # manager ids are "{name}_{id(model)}" - strip the suffix to compare against the
            # spec's plain component names
            final_devices = {
                component_id.rsplit("_", 1)[0]: component.device.type
                for component_id, component in cm.components.items()
                if isinstance(component, torch.nn.Module) and next(component.parameters(), None) is not None
            }
            cm.disable_auto_cpu_offload()
            backend_empty_cache(torch_device)

            offloads = Counter(event.model_id.rsplit("_", 1)[0] for event in events if event.action == "offload")
            assert offloads == Counter(expected["offload"]), (
                f"[{label}] offloads {dict(offloads)} != expected {expected['offload']}"
            )

            ooms = Counter(
                event.reason.removeprefix("oom_retry:").rsplit("_", 1)[0]
                for event in events
                if event.reason is not None and event.reason.startswith("oom_retry:")
            )
            assert ooms == Counter(expected["oom"]), f"[{label}] OOMs {dict(ooms)} != expected {expected['oom']}"

            assert final_devices == expected["final_device"], (
                f"[{label}] final devices {final_devices} != expected {expected['final_device']}"
            )

            # and none of it changed the result
            max_diff = torch.abs(baseline - output).max()
            assert max_diff < expected_max_diff, (
                f"[{label}] offloaded output diverged from baseline (max diff {max_diff})"
            )
