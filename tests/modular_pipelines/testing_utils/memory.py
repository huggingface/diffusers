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
from unittest import mock

import pytest
import torch

from diffusers import ComponentsManager, ModularPipeline
from diffusers.hooks import apply_group_offloading
from diffusers.utils import is_accelerate_available

from ...testing_utils import (
    backend_empty_cache,
    is_cpu_offload,
    is_group_offload,
    is_memory,
    require_accelerate,
    require_accelerator,
    torch_device,
)
from .common import BaseModularPipelineOutputMixin
from .utils import backend_memory_allocated, patch_free_memory


if is_accelerate_available():
    from diffusers.modular_pipelines.components_manager import AutoOffloadStrategy


# More free memory than any tiny test checkpoint could ever need, so the strategy never
# decides to offload. Used to assert the *negative*: no eviction without memory pressure.
_AMPLE_FREE_BYTES = 1024**4


@is_cpu_offload
class ModularOffloadTesterMixin(BaseModularPipelineOutputMixin):
    """Auto CPU offload driven by a `ComponentsManager`: inference stays correct, and unloading a component
    re-applies the offload hooks to the ones that are left."""

    @require_accelerator
    def test_components_auto_cpu_offload_inference_consistent(self, base_pipe_output):
        cm = ComponentsManager()
        cm.enable_auto_cpu_offload(device=torch_device)
        offload_pipe = self.get_pipeline(components_manager=cm)

        image = offload_pipe(**self.get_dummy_inputs(), output=self.output_name)

        expected_slice = base_pipe_output[0, -3:, -3:, -1].flatten()
        image_slice = image[0, -3:, -3:, -1].flatten()
        assert torch.abs(expected_slice - image_slice).max() < 1e-3

    @require_accelerator
    def test_unload_components_auto_cpu_offload(self, base_pipe_output):
        cm = ComponentsManager()
        cm.enable_auto_cpu_offload(device=torch_device)
        pipe = self.get_pipeline(components_manager=cm)
        name = next(
            name for name in pipe.pretrained_component_names if isinstance(pipe.components.get(name), torch.nn.Module)
        )
        component_id = f"{name}_{id(pipe.components[name])}"

        pipe.unload_components(name)

        # removing a component re-applies auto offload to the ones that are left
        assert component_id not in cm.components
        assert component_id not in {hook.model_id for hook in cm.model_hooks}
        remaining = [component for component in cm.components.values() if isinstance(component, torch.nn.Module)]
        assert all(hasattr(component, "_hf_hook") for component in remaining)

        # the reloaded component is hooked up again, so the pipeline still runs
        pipe.load_components(names=name, dtype=torch.float32)
        image = pipe(**self.get_dummy_inputs(), output=self.output_name)
        assert torch.abs(base_pipe_output - image).max() < 1e-3


@is_group_offload
class ModularGroupOffloadTesterMixin(BaseModularPipelineOutputMixin):
    """Group offloading applied to the pipeline's components."""

    @require_accelerator
    def test_group_offloading_execution_device(self):
        pipe = self.get_pipeline().to("cpu")
        assert pipe._execution_device.type == "cpu"

        offloaded = None
        for name, component in pipe.components.items():
            if not isinstance(component, torch.nn.Module):
                continue
            if not getattr(component, "_supports_group_offloading", True):
                continue
            apply_group_offloading(
                component,
                onload_device=torch.device(torch_device),
                offload_device=torch.device("cpu"),
                offload_type="leaf_level",
            )
            offloaded = name
            break

        if offloaded is None:
            pytest.skip("No component supports group offloading.")

        assert pipe._execution_device.type == torch.device(torch_device).type


@is_memory
class ModularMemoryTesterMixin(ModularOffloadTesterMixin, ModularGroupOffloadTesterMixin):
    """Combined mixin for the memory optimizations every modular pipeline is expected to support: auto CPU offload,
    group offload, and reclaiming device memory on `unload_components`."""

    @require_accelerator
    def test_unload_components_frees_device_memory(self):
        pipe = ModularPipeline.from_pretrained(self.pretrained_model_name_or_path)
        pipe.load_components(dtype=torch.float32)
        name = next(
            name for name in pipe.pretrained_component_names if isinstance(pipe.components.get(name), torch.nn.Module)
        )
        pipe.to(torch_device)

        component = pipe.components[name]
        footprint = sum(t.numel() * t.element_size() for t in [*component.parameters(), *component.buffers()])
        del component

        gc.collect()
        backend_empty_cache(torch_device)
        allocated_before = backend_memory_allocated(torch_device)

        pipe.unload_components(name)
        freed = allocated_before - backend_memory_allocated(torch_device)

        assert freed >= 0.9 * footprint, (
            f"Unloading '{name}' freed {freed} bytes on {torch_device}, expected around {footprint}"
        )


@is_cpu_offload
class ModularAutoOffloadTesterMixin(BaseModularPipelineOutputMixin):
    """
    Auto-CPU-offload *decisions* for a modular pipeline's components, driven by simulated device memory.

    Opt-in on top of `ModularMemoryTesterMixin`: these tests spy on `AutoOffloadStrategy` to assert how models are
    sequenced onto the device, which is only meaningful for pipelines with several offloadable model components.
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

        with patch_free_memory(free_bytes), mock.patch.object(AutoOffloadStrategy, "__call__", spy_call):
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
    def test_auto_cpu_offload_inference_consistent_under_memory_pressure(
        self, base_pipe_output, expected_max_diff=1e-3
    ):
        # Sensible results: forcing offload (zero simulated free memory) must not change
        # the output relative to an ordinary, non-offloaded run.
        cm, _, offloaded = self._run_offloaded(free_bytes=0)
        try:
            max_diff = torch.abs(base_pipe_output - offloaded).max()
            assert max_diff < expected_max_diff, f"offloaded output diverged from baseline (max diff {max_diff})"
        finally:
            cm.disable_auto_cpu_offload()
