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
    is_cpu_offload,
    is_group_offload,
    is_memory,
    require_accelerate_version_greater,
    require_accelerator,
    require_torch_accelerator,
    torch_device,
)
from .utils import assert_outputs_close


@is_cpu_offload
class PipelineOffloadTesterMixin:
    """CPU/sequential offload and accelerate `device_map` loading for pipelines."""

    @require_accelerator
    @require_accelerate_version_greater("0.14.0")
    def test_sequential_cpu_offload_forward_pass(self, base_pipe_output, expected_max_diff=1e-4):
        import accelerate

        pipe = self.build_pipe()

        pipe.enable_sequential_cpu_offload(device=torch_device)
        assert pipe._execution_device.type == torch_device

        inputs = self.get_dummy_inputs(torch_device)
        torch.manual_seed(0)
        output_with_offload = pipe(**inputs)[0]

        assert_outputs_close(
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
        import accelerate

        pipe = self.build_pipe()

        pipe.enable_model_cpu_offload(device=torch_device)
        assert pipe._execution_device.type == torch_device

        inputs = self.get_dummy_inputs(torch_device)
        torch.manual_seed(0)
        output_with_offload = pipe(**inputs)[0]

        assert_outputs_close(
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
        import accelerate

        generator_device = "cpu"
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)

        for component in pipe.components.values():
            if hasattr(component, "set_default_attn_processor"):
                component.set_default_attn_processor()

        pipe.set_progress_bar_config(disable=None)

        pipe.enable_model_cpu_offload()
        inputs = self.get_dummy_inputs(generator_device)
        output_with_offload = pipe(**inputs)[0]

        pipe.enable_model_cpu_offload()
        inputs = self.get_dummy_inputs(generator_device)
        output_with_offload_twice = pipe(**inputs)[0]

        assert_outputs_close(
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
        import accelerate

        generator_device = "cpu"
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)

        for component in pipe.components.values():
            if hasattr(component, "set_default_attn_processor"):
                component.set_default_attn_processor()

        pipe.set_progress_bar_config(disable=None)

        pipe.enable_sequential_cpu_offload(device=torch_device)
        inputs = self.get_dummy_inputs(generator_device)
        output_with_offload = pipe(**inputs)[0]

        pipe.enable_sequential_cpu_offload(device=torch_device)
        inputs = self.get_dummy_inputs(generator_device)
        output_with_offload_twice = pipe(**inputs)[0]

        assert_outputs_close(
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
        pipe = self.build_pipe()
        pipe.save_pretrained(tmp_path)

        loaded_pipe = self.pipeline_class.from_pretrained(tmp_path, device_map=torch_device)
        for component in loaded_pipe.components.values():
            if hasattr(component, "set_default_attn_processor"):
                component.set_default_attn_processor()

        inputs = self.get_dummy_inputs(torch_device)
        torch.manual_seed(0)
        loaded_out = loaded_pipe(**inputs)[0]
        assert_outputs_close(
            loaded_out, base_pipe_output, atol=expected_max_difference, msg="device_map loaded output changed."
        )


class LayerwiseCastingTesterMixin:
    """Layerwise FP8 casting during pipeline inference (gated by `test_layerwise_casting`)."""

    def test_layerwise_casting_inference(self):
        if not self.test_layerwise_casting:
            return

        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.to(torch_device, dtype=torch.bfloat16)
        pipe.set_progress_bar_config(disable=None)

        denoiser = pipe.transformer if hasattr(pipe, "transformer") else pipe.unet
        denoiser.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16)

        inputs = self.get_dummy_inputs(torch_device)
        _ = pipe(**inputs)[0]


@is_group_offload
class GroupOffloadTesterMixin:
    """Block/leaf-level group offload, both component-scoped and pipeline-level orchestration."""

    @require_torch_accelerator
    def test_group_offloading_inference(self):
        if not self.test_group_offloading:
            return

        def create_pipe():
            torch.manual_seed(0)
            components = self.get_dummy_components()
            pipe = self.pipeline_class(**components)
            pipe.set_progress_bar_config(disable=None)
            return pipe

        def enable_group_offload_on_component(pipe, group_offloading_kwargs):
            # We intentionally don't test VAE's here. This is because some tests enable tiling on the VAE. If
            # tiling is enabled and a forward pass is run, when accelerator streams are used, the execution order
            # of the layers is not traced correctly. This causes errors. For apply group offloading to VAE, a
            # warmup forward pass (even with dummy small inputs) is recommended.
            for component_name in [
                "text_encoder",
                "text_encoder_2",
                "text_encoder_3",
                "transformer",
                "unet",
                "controlnet",
            ]:
                if not hasattr(pipe, component_name):
                    continue
                component = getattr(pipe, component_name)
                if not getattr(component, "_supports_group_offloading", True):
                    continue
                if hasattr(component, "enable_group_offload"):
                    # For diffusers ModelMixin implementations
                    component.enable_group_offload(torch.device(torch_device), **group_offloading_kwargs)
                else:
                    # For other models not part of diffusers
                    apply_group_offloading(
                        component, onload_device=torch.device(torch_device), **group_offloading_kwargs
                    )
                assert all(
                    module._diffusers_hook.get_hook("group_offloading") is not None
                    for module in component.modules()
                    if hasattr(module, "_diffusers_hook")
                )
            for component_name in ["vae", "vqvae", "image_encoder"]:
                component = getattr(pipe, component_name, None)
                if isinstance(component, torch.nn.Module):
                    component.to(torch_device)

        def run_forward(pipe):
            torch.manual_seed(0)
            inputs = self.get_dummy_inputs(torch_device)
            return pipe(**inputs)[0]

        pipe = create_pipe().to(torch_device)
        output_without_group_offloading = run_forward(pipe)

        pipe = create_pipe()
        enable_group_offload_on_component(pipe, {"offload_type": "block_level", "num_blocks_per_group": 1})
        output_with_group_offloading1 = run_forward(pipe)

        pipe = create_pipe()
        enable_group_offload_on_component(pipe, {"offload_type": "leaf_level"})
        output_with_group_offloading2 = run_forward(pipe)

        if torch.is_tensor(output_without_group_offloading):
            output_without_group_offloading = output_without_group_offloading.detach().cpu().numpy()
            output_with_group_offloading1 = output_with_group_offloading1.detach().cpu().numpy()
            output_with_group_offloading2 = output_with_group_offloading2.detach().cpu().numpy()

        assert_outputs_close(
            output_with_group_offloading1,
            output_without_group_offloading,
            atol=1e-4,
            rtol=1e-5,
            msg="block-level group offloading should not affect the inference results",
        )
        assert_outputs_close(
            output_with_group_offloading2,
            output_without_group_offloading,
            atol=1e-4,
            rtol=1e-5,
            msg="leaf-level group offloading should not affect the inference results",
        )

    @require_torch_accelerator
    def test_pipeline_level_group_offloading_sanity_checks(self):
        components = self.get_dummy_components()
        pipe: DiffusionPipeline = self.pipeline_class(**components)

        for name, component in pipe.components.items():
            if hasattr(component, "_supports_group_offloading"):
                if not component._supports_group_offloading:
                    pytest.skip(f"{self.pipeline_class.__name__} is not suitable for this test.")

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
        # Build the offload pipeline with the same canonical preamble as `base_pipe_output` (eval text encoders +
        # default attn processors) so that group offloading is the only difference under test.
        components = self.get_dummy_components()
        for key in components:
            if "text_encoder" in key and hasattr(components[key], "eval"):
                components[key].eval()
        pipe: DiffusionPipeline = self.pipeline_class(**components)
        for component in pipe.components.values():
            if hasattr(component, "set_default_attn_processor"):
                component.set_default_attn_processor()

        for name, component in pipe.components.items():
            if hasattr(component, "_supports_group_offloading"):
                if not component._supports_group_offloading:
                    pytest.skip(f"{self.pipeline_class.__name__} is not suitable for this test.")

        offload_device = "cpu"
        pipe.enable_group_offload(
            onload_device=torch_device,
            offload_device=offload_device,
            offload_type="leaf_level",
        )
        pipe.set_progress_bar_config(disable=None)
        inputs = self.get_dummy_inputs(torch_device)
        torch.manual_seed(0)
        out_offload = pipe(**inputs)[0]

        assert_outputs_close(
            out_offload,
            base_pipe_output,
            atol=expected_max_difference,
            msg="pipeline-level group offloading should not affect the inference results",
        )


@is_memory
@require_accelerator
class MemoryTesterMixin(PipelineOffloadTesterMixin, GroupOffloadTesterMixin, LayerwiseCastingTesterMixin):
    """Umbrella mixin bundling all memory-placement tests (cf. model-level `MemoryTesterMixin`)."""
