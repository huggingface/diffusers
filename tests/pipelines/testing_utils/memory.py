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

import pytest
import torch

from diffusers import DiffusionPipeline
from diffusers.hooks import apply_group_offloading

from ...testing_utils import (
    assert_tensors_close,
    is_accelerate_available,
    is_cpu_offload,
    is_group_offload,
    is_memory,
    require_accelerate_version_greater,
    require_accelerator,
    require_torch_accelerator,
    torch_device,
)
from .common import BasePipelineOutputMixin


if is_accelerate_available():
    import accelerate


@is_cpu_offload
class PipelineOffloadTesterMixin(BasePipelineOutputMixin):
    """CPU/sequential offload and accelerate `device_map` loading for pipelines."""

    @require_accelerator
    @require_accelerate_version_greater("0.14.0")
    def test_sequential_cpu_offload_forward_pass(self, base_pipe_output, expected_max_diff=1e-4):
        pipe = self.get_pipeline().to(torch_device)

        pipe.enable_sequential_cpu_offload(device=torch_device)
        assert pipe._execution_device.type == torch_device

        inputs = self.get_dummy_inputs()
        torch.manual_seed(0)
        output_with_offload = pipe(**inputs)[0]

        assert_tensors_close(
            output_with_offload,
            base_pipe_output,
            atol=expected_max_diff,
            msg="CPU offloading should not affect the inference results",
        )

        # make sure all `torch.nn.Module` components (except those in `self._exclude_from_cpu_offload`) are offloaded
        offloaded_modules = {
            k: v
            for k, v in pipe.components.items()
            if isinstance(v, torch.nn.Module) and k not in pipe._exclude_from_cpu_offload
        }
        # 1. all offloaded modules should be saved to cpu and moved to meta device
        assert all(v.device.type == "meta" for v in offloaded_modules.values()), (
            f"Not offloaded: {[k for k, v in offloaded_modules.items() if v.device.type != 'meta']}"
        )
        # 2. all offloaded modules should have hook installed
        assert all(hasattr(v, "_hf_hook") for k, v in offloaded_modules.items()), (
            f"No hook attached: {[k for k, v in offloaded_modules.items() if not hasattr(v, '_hf_hook')]}"
        )
        # 3. all offloaded modules should have correct hooks installed, should be either one of these two
        #    - `AlignDevicesHook`
        #    - a `SequentialHook` that contains `AlignDevicesHook`
        offloaded_modules_with_incorrect_hooks = {}
        for k, v in offloaded_modules.items():
            if hasattr(v, "_hf_hook"):
                if isinstance(v._hf_hook, accelerate.hooks.SequentialHook):
                    for hook in v._hf_hook.hooks:
                        if not isinstance(hook, accelerate.hooks.AlignDevicesHook):
                            offloaded_modules_with_incorrect_hooks[k] = type(v._hf_hook.hooks[0])
                elif not isinstance(v._hf_hook, accelerate.hooks.AlignDevicesHook):
                    offloaded_modules_with_incorrect_hooks[k] = type(v._hf_hook)

        assert len(offloaded_modules_with_incorrect_hooks) == 0, (
            f"Not installed correct hook: {offloaded_modules_with_incorrect_hooks}"
        )

    @require_accelerator
    @require_accelerate_version_greater("0.17.0")
    def test_model_cpu_offload_forward_pass(self, base_pipe_output, expected_max_diff=2e-4):
        pipe = self.get_pipeline().to(torch_device)

        pipe.enable_model_cpu_offload(device=torch_device)
        assert pipe._execution_device.type == torch_device

        inputs = self.get_dummy_inputs()
        torch.manual_seed(0)
        output_with_offload = pipe(**inputs)[0]

        assert_tensors_close(
            output_with_offload,
            base_pipe_output,
            atol=expected_max_diff,
            msg="CPU offloading should not affect the inference results",
        )

        offloaded_modules = {
            k: v
            for k, v in pipe.components.items()
            if isinstance(v, torch.nn.Module) and k not in pipe._exclude_from_cpu_offload
        }
        # 1. check if all offloaded modules are saved to cpu
        assert all(v.device.type == "cpu" for v in offloaded_modules.values()), (
            f"Not offloaded: {[k for k, v in offloaded_modules.items() if v.device.type != 'cpu']}"
        )
        # 2. check if all offloaded modules have hooks installed
        assert all(hasattr(v, "_hf_hook") for k, v in offloaded_modules.items()), (
            f"No hook attached: {[k for k, v in offloaded_modules.items() if not hasattr(v, '_hf_hook')]}"
        )
        # 3. check if all offloaded modules have correct type of hooks installed, should be `CpuOffload`
        offloaded_modules_with_incorrect_hooks = {}
        for k, v in offloaded_modules.items():
            if hasattr(v, "_hf_hook") and not isinstance(v._hf_hook, accelerate.hooks.CpuOffload):
                offloaded_modules_with_incorrect_hooks[k] = type(v._hf_hook)

        assert len(offloaded_modules_with_incorrect_hooks) == 0, (
            f"Not installed correct hook: {offloaded_modules_with_incorrect_hooks}"
        )

    @require_accelerator
    @require_accelerate_version_greater("0.17.0")
    def test_cpu_offload_forward_pass_twice(self, expected_max_diff=2e-4):
        pipe = self.get_pipeline()

        pipe.enable_model_cpu_offload()
        inputs = self.get_dummy_inputs()
        output_with_offload = pipe(**inputs)[0]

        pipe.enable_model_cpu_offload()
        inputs = self.get_dummy_inputs()
        output_with_offload_twice = pipe(**inputs)[0]

        assert_tensors_close(
            output_with_offload,
            output_with_offload_twice,
            atol=expected_max_diff,
            msg="running CPU offloading 2nd time should not affect the inference results",
        )

        offloaded_modules = {
            k: v
            for k, v in pipe.components.items()
            if isinstance(v, torch.nn.Module) and k not in pipe._exclude_from_cpu_offload
        }
        # 1. check if all offloaded modules are saved to cpu
        assert all(v.device.type == "cpu" for v in offloaded_modules.values()), (
            f"Not offloaded: {[k for k, v in offloaded_modules.items() if v.device.type != 'cpu']}"
        )
        # 2. check if all offloaded modules have hooks installed
        assert all(hasattr(v, "_hf_hook") for k, v in offloaded_modules.items()), (
            f"No hook attached: {[k for k, v in offloaded_modules.items() if not hasattr(v, '_hf_hook')]}"
        )
        # 3. check if all offloaded modules have correct type of hooks installed, should be `CpuOffload`
        offloaded_modules_with_incorrect_hooks = {}
        for k, v in offloaded_modules.items():
            if hasattr(v, "_hf_hook") and not isinstance(v._hf_hook, accelerate.hooks.CpuOffload):
                offloaded_modules_with_incorrect_hooks[k] = type(v._hf_hook)

        assert len(offloaded_modules_with_incorrect_hooks) == 0, (
            f"Not installed correct hook: {offloaded_modules_with_incorrect_hooks}"
        )

    @require_accelerator
    @require_accelerate_version_greater("0.14.0")
    def test_sequential_offload_forward_pass_twice(self, expected_max_diff=2e-4):
        pipe = self.get_pipeline()

        pipe.enable_sequential_cpu_offload(device=torch_device)
        inputs = self.get_dummy_inputs()
        output_with_offload = pipe(**inputs)[0]

        pipe.enable_sequential_cpu_offload(device=torch_device)
        inputs = self.get_dummy_inputs()
        output_with_offload_twice = pipe(**inputs)[0]

        assert_tensors_close(
            output_with_offload,
            output_with_offload_twice,
            atol=expected_max_diff,
            msg="running sequential offloading second time should have the inference results",
        )

        offloaded_modules = {
            k: v
            for k, v in pipe.components.items()
            if isinstance(v, torch.nn.Module) and k not in pipe._exclude_from_cpu_offload
        }
        # 1. check if all offloaded modules are moved to meta device
        assert all(v.device.type == "meta" for v in offloaded_modules.values()), (
            f"Not offloaded: {[k for k, v in offloaded_modules.items() if v.device.type != 'meta']}"
        )
        # 2. check if all offloaded modules have hook installed
        assert all(hasattr(v, "_hf_hook") for k, v in offloaded_modules.items()), (
            f"No hook attached: {[k for k, v in offloaded_modules.items() if not hasattr(v, '_hf_hook')]}"
        )
        # 3. check if all offloaded modules have correct hooks installed, should be either one of these two
        #    - `AlignDevicesHook`
        #    - a `SequentialHook` that contains `AlignDevicesHook`
        offloaded_modules_with_incorrect_hooks = {}
        for k, v in offloaded_modules.items():
            if hasattr(v, "_hf_hook"):
                if isinstance(v._hf_hook, accelerate.hooks.SequentialHook):
                    for hook in v._hf_hook.hooks:
                        if not isinstance(hook, accelerate.hooks.AlignDevicesHook):
                            offloaded_modules_with_incorrect_hooks[k] = type(v._hf_hook.hooks[0])
                elif not isinstance(v._hf_hook, accelerate.hooks.AlignDevicesHook):
                    offloaded_modules_with_incorrect_hooks[k] = type(v._hf_hook)

        assert len(offloaded_modules_with_incorrect_hooks) == 0, (
            f"Not installed correct hook: {offloaded_modules_with_incorrect_hooks}"
        )

    def test_pipeline_with_accelerator_device_map(self, tmp_path, base_pipe_output, expected_max_difference=1e-4):
        pipe = self.get_pipeline().to(torch_device)

        pipe.save_pretrained(tmp_path)

        loaded_pipe = self.pipeline_class.from_pretrained(tmp_path, device_map=torch_device)

        inputs = self.get_dummy_inputs()
        torch.manual_seed(0)
        loaded_out = loaded_pipe(**inputs)[0]
        assert_tensors_close(
            loaded_out, base_pipe_output, atol=expected_max_difference, msg="device_map loaded output changed."
        )


class LayerwiseCastingTesterMixin(BasePipelineOutputMixin):
    """Layerwise FP8 casting during pipeline inference."""

    def test_layerwise_casting_inference(self):
        pipe = self.get_pipeline()
        denoiser = getattr(pipe, "transformer", None) or getattr(pipe, "unet", None)
        if denoiser is None or not hasattr(denoiser, "enable_layerwise_casting"):
            pytest.skip(f"{self.pipeline_class.__name__} has no denoiser that supports layerwise casting.")

        pipe.to(torch_device, dtype=torch.bfloat16)
        pipe.set_progress_bar_config(disable=None)

        denoiser.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16)

        inputs = self.get_dummy_inputs()
        output = pipe(**inputs)[0]

        assert not torch.isnan(output).any(), (
            f"`{self.pipeline_class.__name__}` produced NaNs during layerwise casting inference."
        )


@is_group_offload
class GroupOffloadTesterMixin(BasePipelineOutputMixin):
    """Block/leaf-level group offload, both component-scoped and pipeline-level orchestration."""

    def create_pipe(self):
            torch.manual_seed(0)
            return self.get_pipeline()

    def _skip_if_group_offloading_unsupported(self, pipe):
        for component in pipe.components.values():
            if hasattr(component, "_supports_group_offloading") and not component._supports_group_offloading:
                pytest.skip(f"{self.pipeline_class.__name__} has a component that does not support group offloading.")

    def _group_offload_exclude_modules(self, pipe, offload_type):
        """Config-declared components to keep out of group offloading at `offload_type`.

        Every group offload test routes its exclusions through here, so a name that matches no component on the
        pipeline is reported as the typo it is rather than silently costing coverage and surfacing later as a
        device mismatch. The onload names are not checked: they are a shared default covering several pipelines,
        most of which have only some of them.
        """
        exclude = set(self.group_offloading_exclude_modules)
        if offload_type == "leaf_level":
            exclude |= set(self.group_offloading_leaf_level_exclude_modules)

        # Checked against every registered component rather than the module-valued ones, so that excluding an
        # optional component a config leaves unset reads as the no-op it is instead of a typo.
        unknown = sorted(exclude - set(pipe.components))
        assert not unknown, (
            f"{type(self).__name__} excludes {unknown} from group offloading, but "
            f"{self.pipeline_class.__name__} has no such component. Its components are "
            f"{sorted(pipe.components)}."
        )
        return exclude

    def _split_group_offload_components(self, pipe, offload_type):
        """Split the pipeline's module components into the ones to offload and the ones to keep on the accelerator.

        Everything is offloaded unless the config lists it, so a component a pipeline adds under a name this file
        has never heard of is covered by default rather than silently left on CPU. See the three list attributes on
        `BasePipelineTesterConfig`.
        """
        module_names = [name for name, component in pipe.components.items() if isinstance(component, torch.nn.Module)]
        onload_names = self._group_offload_exclude_modules(pipe, offload_type) | set(
            self.group_offloading_onload_component_names
        )
        offload = [name for name in module_names if name not in onload_names]
        onload = [name for name in module_names if name in onload_names]
        return offload, onload

    def _enable_group_offload_on_components(self, pipe, **group_offloading_kwargs):
        offload_names, onload_names = self._split_group_offload_components(
            pipe, group_offloading_kwargs["offload_type"]
        )
        for component_name in offload_names:
            component = getattr(pipe, component_name)
            if hasattr(component, "enable_group_offload"):
                # For diffusers ModelMixin implementations
                component.enable_group_offload(torch.device(torch_device), **group_offloading_kwargs)
            else:
                # For other models not part of diffusers
                apply_group_offloading(component, onload_device=torch.device(torch_device), **group_offloading_kwargs)
            assert all(
                module._diffusers_hook.get_hook("group_offloading") is not None
                for module in component.modules()
                if hasattr(module, "_diffusers_hook")
            )
        for component_name in onload_names:
            getattr(pipe, component_name).to(torch_device)

    def _run_group_offload_inference(self, base_pipe_output, expected_max_difference, msg, **group_offloading_kwargs):
        # Build the offload pipeline the same way as `base_pipe_output` so that group offloading is the only
        # difference under test. It stays on CPU here — the components are placed as they are hooked.
        pipe = self.create_pipe()
        self._skip_if_group_offloading_unsupported(pipe)
        self._enable_group_offload_on_components(pipe, **group_offloading_kwargs)

        assert_tensors_close(self.run_pipe(pipe), base_pipe_output, atol=expected_max_difference, rtol=1e-5, msg=msg)

    @require_torch_accelerator
    def test_group_offloading_inference_block_level(self, base_pipe_output, expected_max_difference=1e-4):
        self._run_group_offload_inference(
            base_pipe_output,
            expected_max_difference,
            msg="block-level group offloading should not affect the inference results",
            offload_type="block_level",
            num_blocks_per_group=1,
        )

    @require_torch_accelerator
    def test_group_offloading_inference_leaf_level(self, base_pipe_output, expected_max_difference=1e-4):
        self._run_group_offload_inference(
            base_pipe_output,
            expected_max_difference,
            msg="leaf-level group offloading should not affect the inference results",
            offload_type="leaf_level",
        )

    @require_torch_accelerator
    def test_pipeline_level_group_offloading_sanity_checks(self):
        pipe: DiffusionPipeline = self.get_pipeline()
        self._skip_if_group_offloading_unsupported(pipe)

        module_names = sorted(
            [name for name, component in pipe.components.items() if isinstance(component, torch.nn.Module)]
        )
        exclude_module_name = module_names[0]
        offload_device = "cpu"
        pipe.enable_group_offload(
            onload_device=torch_device,
            offload_device=offload_device,
            offload_type="leaf_level",
            exclude_modules=exclude_module_name,
        )
        excluded_module = getattr(pipe, exclude_module_name)
        assert torch.device(excluded_module.device).type == torch.device(torch_device).type

        for name, component in pipe.components.items():
            if name not in [exclude_module_name] and isinstance(component, torch.nn.Module):
                # `component.device` prints the `onload_device` type. We should probably override the `device`
                # property in `ModelMixin`. Skip modules with no parameters (e.g., dummy safety checkers).
                params = list(component.parameters())
                if not params:
                    continue
                component_device = params[0].device
                assert torch.device(component_device).type == torch.device(offload_device).type

    @require_torch_accelerator
    def test_pipeline_level_group_offloading_inference(self, base_pipe_output, expected_max_difference=1e-4):
        # Build the offload pipeline the same way as `base_pipe_output` so that group offloading is the only
        # difference under test. It stays on CPU here — `enable_group_offload` places the components.
        pipe: DiffusionPipeline = self.get_pipeline()
        self._skip_if_group_offloading_unsupported(pipe)

        offload_device = "cpu"
        offload_type = "leaf_level"
        pipe.enable_group_offload(
            onload_device=torch_device,
            offload_device=offload_device,
            offload_type=offload_type,
            exclude_modules=sorted(self._group_offload_exclude_modules(pipe, offload_type)),
        )
        pipe.set_progress_bar_config(disable=None)
        inputs = self.get_dummy_inputs()
        torch.manual_seed(0)
        out_offload = pipe(**inputs)[0]

        assert_tensors_close(
            out_offload,
            base_pipe_output,
            atol=expected_max_difference,
            msg="pipeline-level group offloading should not affect the inference results",
        )


@is_memory
@require_accelerator
class MemoryTesterMixin(PipelineOffloadTesterMixin, GroupOffloadTesterMixin, LayerwiseCastingTesterMixin):
    """Umbrella mixin bundling all memory-placement tests (cf. model-level `MemoryTesterMixin`)."""
