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

import gc
import glob
import inspect
from functools import wraps

import pytest
import torch
from accelerate.utils.modeling import compute_module_sizes

from diffusers.utils.testing_utils import _check_safetensors_serialization
from diffusers.utils.torch_utils import get_torch_cuda_device_capability

from ...testing_utils import (
    assert_tensors_close,
    backend_empty_cache,
    backend_max_memory_allocated,
    backend_reset_peak_memory_stats,
    backend_synchronize,
    is_cpu_offload,
    is_group_offload,
    is_memory,
    require_accelerator,
    torch_device,
)
from .common import cast_inputs_to_dtype, check_device_map_is_respected


def require_offload_support(func):
    """
    Decorator to skip tests if model doesn't support offloading (requires _no_split_modules).
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if self.model_class._no_split_modules is None:
            pytest.skip("Test not supported for this model as `_no_split_modules` is not set.")
        return func(self, *args, **kwargs)

    return wrapper


def require_group_offload_support(func):
    """
    Decorator to skip tests if model doesn't support group offloading.
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not self.model_class._supports_group_offloading:
            pytest.skip("Model does not support group offloading.")
        return func(self, *args, **kwargs)

    return wrapper


@is_cpu_offload
class CPUOffloadTesterMixin:
    """
    Mixin class for testing CPU offloading functionality.

    Expected from config mixin:
        - model_class: The model class to test

    Optional properties:
        - model_split_percents: List of percentages for splitting model across devices (default: [0.5, 0.7])

    Expected methods from config mixin:
        - get_init_dict(): Returns dict of arguments to initialize the model
        - get_dummy_inputs(): Returns dict of inputs to pass to the model forward pass

    Pytest mark: cpu_offload
        Use `pytest -m "not cpu_offload"` to skip these tests
    """

    @property
    def model_split_percents(self) -> list[float]:
        """List of percentages for splitting model across devices during offloading tests."""
        return [0.5, 0.7]

    @require_offload_support
    @torch.no_grad()
    def test_cpu_offload(self, tmp_path, atol=1e-5, rtol=0):
        config = self.get_init_dict()
        inputs_dict = self.get_dummy_inputs()
        model = self.model_class(**config).eval()

        model = model.to(torch_device)

        torch.manual_seed(0)
        base_output = model(**inputs_dict)

        model_size = compute_module_sizes(model)[""]
        # We test several splits of sizes to make sure it works
        max_gpu_sizes = [int(p * model_size) for p in self.model_split_percents]
        model.cpu().save_pretrained(str(tmp_path))

        for max_size in max_gpu_sizes:
            max_memory = {0: max_size, "cpu": model_size * 2}
            new_model = self.model_class.from_pretrained(str(tmp_path), device_map="auto", max_memory=max_memory)
            # Making sure part of the model will actually end up offloaded
            assert set(new_model.hf_device_map.values()) == {0, "cpu"}, "Model should be split between GPU and CPU"

            check_device_map_is_respected(new_model, new_model.hf_device_map)
            torch.manual_seed(0)
            new_output = new_model(**inputs_dict)

            assert_tensors_close(
                base_output[0], new_output[0], atol=atol, rtol=rtol, msg="Output should match with CPU offloading"
            )

    @require_offload_support
    @torch.no_grad()
    def test_disk_offload_without_safetensors(self, tmp_path, atol=1e-5, rtol=0):
        config = self.get_init_dict()
        inputs_dict = self.get_dummy_inputs()
        model = self.model_class(**config).eval()

        model = model.to(torch_device)

        torch.manual_seed(0)
        base_output = model(**inputs_dict)

        model_size = compute_module_sizes(model)[""]
        max_size = int(self.model_split_percents[0] * model_size)
        # Force disk offload by setting very small CPU memory
        max_memory = {0: max_size, "cpu": int(0.1 * max_size)}

        model.cpu().save_pretrained(str(tmp_path), safe_serialization=False)
        # This errors out because it's missing an offload folder
        with pytest.raises(ValueError):
            new_model = self.model_class.from_pretrained(str(tmp_path), device_map="auto", max_memory=max_memory)

        new_model = self.model_class.from_pretrained(
            str(tmp_path), device_map="auto", max_memory=max_memory, offload_folder=str(tmp_path)
        )

        check_device_map_is_respected(new_model, new_model.hf_device_map)
        torch.manual_seed(0)
        new_output = new_model(**inputs_dict)

        assert_tensors_close(
            base_output[0], new_output[0], atol=atol, rtol=rtol, msg="Output should match with disk offloading"
        )

    @require_offload_support
    @torch.no_grad()
    def test_disk_offload_with_safetensors(self, tmp_path, atol=1e-5, rtol=0):
        config = self.get_init_dict()
        inputs_dict = self.get_dummy_inputs()
        model = self.model_class(**config).eval()

        model = model.to(torch_device)

        torch.manual_seed(0)
        base_output = model(**inputs_dict)

        model_size = compute_module_sizes(model)[""]
        model.cpu().save_pretrained(str(tmp_path))

        max_size = int(self.model_split_percents[0] * model_size)
        max_memory = {0: max_size, "cpu": max_size}
        new_model = self.model_class.from_pretrained(
            str(tmp_path), device_map="auto", offload_folder=str(tmp_path), max_memory=max_memory
        )

        check_device_map_is_respected(new_model, new_model.hf_device_map)
        torch.manual_seed(0)
        new_output = new_model(**inputs_dict)

        assert_tensors_close(
            base_output[0],
            new_output[0],
            atol=atol,
            rtol=rtol,
            msg="Output should match with disk offloading (safetensors)",
        )


@is_group_offload
class GroupOffloadTesterMixin:
    """
    Mixin class for testing group offloading functionality.

    Expected from config mixin:
        - model_class: The model class to test

    Expected methods from config mixin:
        - get_init_dict(): Returns dict of arguments to initialize the model
        - get_dummy_inputs(): Returns dict of inputs to pass to the model forward pass

    Pytest mark: group_offload
        Use `pytest -m "not group_offload"` to skip these tests
    """

    @require_group_offload_support
    @pytest.mark.parametrize("record_stream", [False, True])
    def test_group_offloading(self, record_stream, atol=1e-5, rtol=0):
        init_dict = self.get_init_dict()
        inputs_dict = self.get_dummy_inputs()
        torch.manual_seed(0)

        @torch.no_grad()
        def run_forward(model):
            assert all(
                module._diffusers_hook.get_hook("group_offloading") is not None
                for module in model.modules()
                if hasattr(module, "_diffusers_hook")
            ), "Group offloading hook should be set"
            model.eval()
            return model(**inputs_dict)[0]

        model = self.model_class(**init_dict)

        model.to(torch_device)
        output_without_group_offloading = run_forward(model)

        torch.manual_seed(0)
        model = self.model_class(**init_dict)
        model.enable_group_offload(torch_device, offload_type="block_level", num_blocks_per_group=1)
        output_with_group_offloading1 = run_forward(model)

        torch.manual_seed(0)
        model = self.model_class(**init_dict)
        model.enable_group_offload(torch_device, offload_type="block_level", num_blocks_per_group=1, non_blocking=True)
        output_with_group_offloading2 = run_forward(model)

        torch.manual_seed(0)
        model = self.model_class(**init_dict)
        model.enable_group_offload(torch_device, offload_type="leaf_level")
        output_with_group_offloading3 = run_forward(model)

        torch.manual_seed(0)
        model = self.model_class(**init_dict)
        model.enable_group_offload(
            torch_device, offload_type="leaf_level", use_stream=True, record_stream=record_stream
        )
        output_with_group_offloading4 = run_forward(model)

        assert_tensors_close(
            output_without_group_offloading,
            output_with_group_offloading1,
            atol=atol,
            rtol=rtol,
            msg="Output should match with block-level offloading",
        )
        assert_tensors_close(
            output_without_group_offloading,
            output_with_group_offloading2,
            atol=atol,
            rtol=rtol,
            msg="Output should match with non-blocking block-level offloading",
        )
        assert_tensors_close(
            output_without_group_offloading,
            output_with_group_offloading3,
            atol=atol,
            rtol=rtol,
            msg="Output should match with leaf-level offloading",
        )
        assert_tensors_close(
            output_without_group_offloading,
            output_with_group_offloading4,
            atol=atol,
            rtol=rtol,
            msg="Output should match with leaf-level offloading with stream",
        )

    @require_group_offload_support
    @pytest.mark.parametrize("record_stream", [False, True])
    @pytest.mark.parametrize("offload_type", ["block_level", "leaf_level"])
    @torch.no_grad()
    def test_group_offloading_with_layerwise_casting(self, record_stream, offload_type):
        torch.manual_seed(0)
        init_dict = self.get_init_dict()
        inputs_dict = self.get_dummy_inputs()
        model = self.model_class(**init_dict)

        model.to(torch_device)
        model.eval()
        _ = model(**inputs_dict)[0]

        torch.manual_seed(0)
        init_dict = self.get_init_dict()
        inputs_dict = self.get_dummy_inputs()
        storage_dtype, compute_dtype = torch.float16, torch.float32
        inputs_dict = cast_inputs_to_dtype(inputs_dict, torch.float32, compute_dtype)
        model = self.model_class(**init_dict)
        model.eval()
        additional_kwargs = {} if offload_type == "leaf_level" else {"num_blocks_per_group": 1}
        model.enable_group_offload(
            torch_device, offload_type=offload_type, use_stream=True, record_stream=record_stream, **additional_kwargs
        )
        model.enable_layerwise_casting(storage_dtype=storage_dtype, compute_dtype=compute_dtype)
        _ = model(**inputs_dict)[0]

    @require_group_offload_support
    @pytest.mark.parametrize("record_stream", [False, True])
    @pytest.mark.parametrize("offload_type", ["block_level", "leaf_level"])
    @torch.no_grad()
    @torch.inference_mode()
    def test_group_offloading_with_disk(self, tmp_path, record_stream, offload_type, atol=1e-5, rtol=0):
        def _has_generator_arg(model):
            sig = inspect.signature(model.forward)
            params = sig.parameters
            return "generator" in params

        def _run_forward(model, inputs_dict):
            accepts_generator = _has_generator_arg(model)
            if accepts_generator:
                inputs_dict["generator"] = torch.manual_seed(0)
            torch.manual_seed(0)
            return model(**inputs_dict)[0]

        init_dict = self.get_init_dict()
        inputs_dict = self.get_dummy_inputs()
        torch.manual_seed(0)
        model = self.model_class(**init_dict)

        model.eval()
        model.to(torch_device)
        output_without_group_offloading = _run_forward(model, inputs_dict)

        torch.manual_seed(0)
        model = self.model_class(**init_dict)
        model.eval()

        num_blocks_per_group = None if offload_type == "leaf_level" else 1
        additional_kwargs = {} if offload_type == "leaf_level" else {"num_blocks_per_group": num_blocks_per_group}
        tmpdir = str(tmp_path)
        model.enable_group_offload(
            torch_device,
            offload_type=offload_type,
            offload_to_disk_path=tmpdir,
            use_stream=True,
            record_stream=record_stream,
            **additional_kwargs,
        )
        has_safetensors = glob.glob(f"{tmpdir}/*.safetensors")
        assert has_safetensors, "No safetensors found in the directory."

        # For "leaf-level", there is a prefetching hook which makes this check a bit non-deterministic
        # in nature. So, skip it.
        if offload_type != "leaf_level":
            is_correct, extra_files, missing_files = _check_safetensors_serialization(
                module=model,
                offload_to_disk_path=tmpdir,
                offload_type=offload_type,
                num_blocks_per_group=num_blocks_per_group,
            )
            if not is_correct:
                if extra_files:
                    raise ValueError(f"Found extra files: {', '.join(extra_files)}")
                elif missing_files:
                    raise ValueError(f"Following files are missing: {', '.join(missing_files)}")

        output_with_group_offloading = _run_forward(model, inputs_dict)
        assert_tensors_close(
            output_without_group_offloading,
            output_with_group_offloading,
            atol=atol,
            rtol=rtol,
            msg="Output should match with disk-based group offloading",
        )


class LayerwiseCastingTesterMixin:
    """
    Mixin class for testing layerwise dtype casting for memory optimization.

    Expected from config mixin:
        - model_class: The model class to test

    Expected methods from config mixin:
        - get_init_dict(): Returns dict of arguments to initialize the model
        - get_dummy_inputs(): Returns dict of inputs to pass to the model forward pass
    """

    @torch.no_grad()
    def test_layerwise_casting_memory(self):
        MB_TOLERANCE = 0.2
        LEAST_COMPUTE_CAPABILITY = 8.0

        def reset_memory_stats():
            gc.collect()
            backend_synchronize(torch_device)
            backend_empty_cache(torch_device)
            backend_reset_peak_memory_stats(torch_device)

        def get_memory_usage(storage_dtype, compute_dtype):
            torch.manual_seed(0)
            config = self.get_init_dict()
            inputs_dict = self.get_dummy_inputs()
            inputs_dict = cast_inputs_to_dtype(inputs_dict, torch.float32, compute_dtype)
            model = self.model_class(**config).eval()
            model = model.to(torch_device, dtype=compute_dtype)
            model.enable_layerwise_casting(storage_dtype=storage_dtype, compute_dtype=compute_dtype)

            reset_memory_stats()
            model(**inputs_dict)
            model_memory_footprint = model.get_memory_footprint()
            peak_inference_memory_allocated_mb = backend_max_memory_allocated(torch_device) / 1024**2

            return model_memory_footprint, peak_inference_memory_allocated_mb

        fp32_memory_footprint, fp32_max_memory = get_memory_usage(torch.float32, torch.float32)
        fp8_e4m3_fp32_memory_footprint, fp8_e4m3_fp32_max_memory = get_memory_usage(torch.float8_e4m3fn, torch.float32)
        fp8_e4m3_bf16_memory_footprint, fp8_e4m3_bf16_max_memory = get_memory_usage(
            torch.float8_e4m3fn, torch.bfloat16
        )

        compute_capability = get_torch_cuda_device_capability() if torch_device == "cuda" else None
        assert fp8_e4m3_bf16_memory_footprint < fp8_e4m3_fp32_memory_footprint < fp32_memory_footprint, (
            "Memory footprint should decrease with lower precision storage"
        )

        # NOTE: the following assertion would fail on our CI (running Tesla T4) due to bf16 using more memory than fp32.
        # On other devices, such as DGX (Ampere) and Audace (Ada), the test passes. So, we conditionally check it.
        if compute_capability and compute_capability >= LEAST_COMPUTE_CAPABILITY:
            assert fp8_e4m3_bf16_max_memory < fp8_e4m3_fp32_max_memory, (
                "Peak memory should be lower with bf16 compute on newer GPUs"
            )

        # On this dummy test case with a small model, sometimes fp8_e4m3_fp32 max memory usage is higher than fp32 by a few
        # bytes. This only happens for some models, so we allow a small tolerance.
        # For any real model being tested, the order would be fp8_e4m3_bf16 < fp8_e4m3_fp32 < fp32.
        assert (
            fp8_e4m3_fp32_max_memory < fp32_max_memory
            or abs(fp8_e4m3_fp32_max_memory - fp32_max_memory) < MB_TOLERANCE
        ), "Peak memory should be lower or within tolerance with fp8 storage"

    def test_layerwise_casting_training(self):
        def test_fn(storage_dtype, compute_dtype):
            if torch.device(torch_device).type == "cpu" and compute_dtype == torch.bfloat16:
                pytest.skip("Skipping test because CPU doesn't go well with bfloat16.")

            model = self.model_class(**self.get_init_dict())
            model = model.to(torch_device, dtype=compute_dtype)
            model.enable_layerwise_casting(storage_dtype=storage_dtype, compute_dtype=compute_dtype)
            model.train()

            inputs_dict = self.get_dummy_inputs()
            inputs_dict = cast_inputs_to_dtype(inputs_dict, torch.float32, compute_dtype)
            with torch.amp.autocast(device_type=torch.device(torch_device).type):
                output = model(**inputs_dict, return_dict=False)[0]

                input_tensor = inputs_dict[self.main_input_name]
                noise = torch.randn((input_tensor.shape[0],) + self.output_shape).to(torch_device)
                noise = cast_inputs_to_dtype(noise, torch.float32, compute_dtype)
                loss = torch.nn.functional.mse_loss(output, noise)

            loss.backward()

        test_fn(torch.float16, torch.float32)
        test_fn(torch.float8_e4m3fn, torch.float32)
        test_fn(torch.float8_e5m2, torch.float32)
        test_fn(torch.float8_e4m3fn, torch.bfloat16)


@is_memory
@require_accelerator
class MemoryTesterMixin(CPUOffloadTesterMixin, GroupOffloadTesterMixin, LayerwiseCastingTesterMixin):
    """
    Combined mixin class for all memory optimization tests including CPU/disk offloading,
    group offloading, and layerwise dtype casting.

    This mixin inherits from:
        - CPUOffloadTesterMixin: CPU and disk offloading tests
        - GroupOffloadTesterMixin: Group offloading tests (block-level and leaf-level)
        - LayerwiseCastingTesterMixin: Layerwise dtype casting tests

    Expected from config mixin:
        - model_class: The model class to test

    Optional properties:
        - model_split_percents: List of percentages for splitting model across devices (default: [0.5, 0.7])

    Expected methods from config mixin:
        - get_init_dict(): Returns dict of arguments to initialize the model
        - get_dummy_inputs(): Returns dict of inputs to pass to the model forward pass

    Pytest mark: memory
        Use `pytest -m "not memory"` to skip these tests
    """
