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

import json
import os
import tempfile
from collections import defaultdict

import pytest
import torch
import torch.nn as nn
from accelerate.utils.modeling import _get_proper_dtype, compute_module_sizes, dtype_byte_size

from diffusers.utils import SAFE_WEIGHTS_INDEX_NAME, _add_variant, logging
from diffusers.utils.testing_utils import require_accelerator, require_torch_multi_accelerator

from ...testing_utils import CaptureLogger, torch_device


def named_persistent_module_tensors(
    module: nn.Module,
    recurse: bool = False,
):
    """
    A helper function that gathers all the tensors (parameters + persistent buffers) of a given module.

    Args:
        module (`torch.nn.Module`):
            The module we want the tensors on.
        recurse (`bool`, *optional`, defaults to `False`):
            Whether or not to go look in every submodule or just return the direct parameters and buffers.
    """
    yield from module.named_parameters(recurse=recurse)

    for named_buffer in module.named_buffers(recurse=recurse):
        name, _ = named_buffer
        # Get parent by splitting on dots and traversing the model
        parent = module
        if "." in name:
            parent_name = name.rsplit(".", 1)[0]
            for part in parent_name.split("."):
                parent = getattr(parent, part)
            name = name.split(".")[-1]
        if name not in parent._non_persistent_buffers_set:
            yield named_buffer


def compute_module_persistent_sizes(
    model: nn.Module,
    dtype: str | torch.device | None = None,
    special_dtypes: dict[str, str | torch.device] | None = None,
):
    """
    Compute the size of each submodule of a given model (parameters + persistent buffers).
    """
    if dtype is not None:
        dtype = _get_proper_dtype(dtype)
        dtype_size = dtype_byte_size(dtype)
    if special_dtypes is not None:
        special_dtypes = {key: _get_proper_dtype(dtyp) for key, dtyp in special_dtypes.items()}
        special_dtypes_size = {key: dtype_byte_size(dtyp) for key, dtyp in special_dtypes.items()}
    module_sizes = defaultdict(int)

    module_list = []

    module_list = named_persistent_module_tensors(model, recurse=True)

    for name, tensor in module_list:
        if special_dtypes is not None and name in special_dtypes:
            size = tensor.numel() * special_dtypes_size[name]
        elif dtype is None:
            size = tensor.numel() * dtype_byte_size(tensor.dtype)
        elif str(tensor.dtype).startswith(("torch.uint", "torch.int", "torch.bool")):
            # According to the code in set_module_tensor_to_device, these types won't be converted
            # so use their original size here
            size = tensor.numel() * dtype_byte_size(tensor.dtype)
        else:
            size = tensor.numel() * min(dtype_size, dtype_byte_size(tensor.dtype))
        name_parts = name.split(".")
        for idx in range(len(name_parts) + 1):
            module_sizes[".".join(name_parts[:idx])] += size

    return module_sizes


def calculate_expected_num_shards(index_map_path):
    """
    Calculate expected number of shards from index file.

    Args:
        index_map_path: Path to the sharded checkpoint index file

    Returns:
        int: Expected number of shards
    """
    with open(index_map_path) as f:
        weight_map_dict = json.load(f)["weight_map"]
    first_key = list(weight_map_dict.keys())[0]
    weight_loc = weight_map_dict[first_key]  # e.g., diffusion_pytorch_model-00001-of-00002.safetensors
    expected_num_shards = int(weight_loc.split("-")[-1].split(".")[0])
    return expected_num_shards


def check_device_map_is_respected(model, device_map):
    for param_name, param in model.named_parameters():
        # Find device in device_map
        while len(param_name) > 0 and param_name not in device_map:
            param_name = ".".join(param_name.split(".")[:-1])
        if param_name not in device_map:
            raise ValueError("device map is incomplete, it does not contain any device for `param_name`.")

        param_device = device_map[param_name]
        if param_device in ["cpu", "disk"]:
            assert param.device == torch.device("meta"), f"Expected device 'meta' for {param_name}, got {param.device}"
        else:
            assert param.device == torch.device(param_device), (
                f"Expected device {param_device} for {param_name}, got {param.device}"
            )


class ModelTesterMixin:
    """
    Base mixin class for model testing with common test methods.

    Expected class attributes to be set by subclasses:
        - model_class: The model class to test
        - main_input_name: Name of the main input tensor (e.g., "sample", "hidden_states")
        - base_precision: Default tolerance for floating point comparisons (default: 1e-3)

    Expected methods to be implemented by subclasses:
        - get_init_dict(): Returns dict of arguments to initialize the model
        - get_dummy_inputs(): Returns dict of inputs to pass to the model forward pass
    """

    model_class = None
    base_precision = 1e-3
    model_split_percents = [0.5, 0.7]

    def get_init_dict(self):
        raise NotImplementedError("get_init_dict must be implemented by subclasses. ")

    def get_dummy_inputs(self):
        raise NotImplementedError("get_dummy_inputs must be implemented by subclasses. It should return inputs_dict.")

    def test_from_save_pretrained(self, expected_max_diff=5e-5):
        torch.manual_seed(0)
        model = self.model_class(**self.get_init_dict())
        model.to(torch_device)
        model.eval()

        with tempfile.TemporaryDirectory() as tmpdirname:
            model.save_pretrained(tmpdirname)
            new_model = self.model_class.from_pretrained(tmpdirname)
            new_model.to(torch_device)

        # check if all parameters shape are the same
        for param_name in model.state_dict().keys():
            param_1 = model.state_dict()[param_name]
            param_2 = new_model.state_dict()[param_name]
            assert param_1.shape == param_2.shape, (
                f"Parameter shape mismatch for {param_name}. Original: {param_1.shape}, loaded: {param_2.shape}"
            )

        with torch.no_grad():
            image = model(**self.get_dummy_inputs())

            if isinstance(image, dict):
                image = image.to_tuple()[0]

            new_image = new_model(**self.get_dummy_inputs())

            if isinstance(new_image, dict):
                new_image = new_image.to_tuple()[0]

        max_diff = (image - new_image).abs().max().item()
        assert max_diff <= expected_max_diff, (
            f"Models give different forward passes. Max diff: {max_diff}, expected: {expected_max_diff}"
        )

    def test_from_save_pretrained_variant(self, expected_max_diff=5e-5):
        model = self.model_class(**self.get_init_dict())
        model.to(torch_device)
        model.eval()

        with tempfile.TemporaryDirectory() as tmpdirname:
            model.save_pretrained(tmpdirname, variant="fp16")
            new_model = self.model_class.from_pretrained(tmpdirname, variant="fp16")

            # non-variant cannot be loaded
            with pytest.raises(OSError) as exc_info:
                self.model_class.from_pretrained(tmpdirname)

            # make sure that error message states what keys are missing
            assert "Error no file named diffusion_pytorch_model.bin found in directory" in str(exc_info.value)

            new_model.to(torch_device)

        with torch.no_grad():
            image = model(**self.get_dummy_inputs())
            if isinstance(image, dict):
                image = image.to_tuple()[0]

            new_image = new_model(**self.get_dummy_inputs())

            if isinstance(new_image, dict):
                new_image = new_image.to_tuple()[0]

        max_diff = (image - new_image).abs().max().item()
        assert max_diff <= expected_max_diff, (
            f"Models give different forward passes. Max diff: {max_diff}, expected: {expected_max_diff}"
        )

    def test_from_save_pretrained_dtype(self):
        model = self.model_class(**self.get_init_dict())
        model.to(torch_device)
        model.eval()

        for dtype in [torch.float32, torch.float16, torch.bfloat16]:
            if torch_device == "mps" and dtype == torch.bfloat16:
                continue
            with tempfile.TemporaryDirectory() as tmpdirname:
                model.to(dtype)
                model.save_pretrained(tmpdirname)
                new_model = self.model_class.from_pretrained(tmpdirname, low_cpu_mem_usage=True, torch_dtype=dtype)
                assert new_model.dtype == dtype
                if (
                    hasattr(self.model_class, "_keep_in_fp32_modules")
                    and self.model_class._keep_in_fp32_modules is None
                ):
                    # When loading without accelerate dtype == torch.float32 if _keep_in_fp32_modules is not None
                    new_model = self.model_class.from_pretrained(
                        tmpdirname, low_cpu_mem_usage=False, torch_dtype=dtype
                    )
                    assert new_model.dtype == dtype

    def test_determinism(self, expected_max_diff=1e-5):
        model = self.model_class(**self.get_init_dict())
        model.to(torch_device)
        model.eval()

        with torch.no_grad():
            first = model(**self.get_dummy_inputs())
            if isinstance(first, dict):
                first = first.to_tuple()[0]

            second = model(**self.get_dummy_inputs())
            if isinstance(second, dict):
                second = second.to_tuple()[0]

        # Remove NaN values and compute max difference
        first_flat = first.flatten()
        second_flat = second.flatten()

        # Filter out NaN values
        mask = ~(torch.isnan(first_flat) | torch.isnan(second_flat))
        first_filtered = first_flat[mask]
        second_filtered = second_flat[mask]

        max_diff = torch.abs(first_filtered - second_filtered).max().item()
        assert max_diff <= expected_max_diff, (
            f"Model outputs are not deterministic. Max diff: {max_diff}, expected: {expected_max_diff}"
        )

    def test_output(self, expected_output_shape=None):
        model = self.model_class(**self.get_init_dict())
        model.to(torch_device)
        model.eval()

        inputs_dict = self.get_dummy_inputs()
        with torch.no_grad():
            output = model(**inputs_dict)

            if isinstance(output, dict):
                output = output.to_tuple()[0]

        assert output is not None, "Model output is None"
        assert output[0].shape == expected_output_shape or self.output_shape, (
            f"Output shape does not match expected. Expected {expected_output_shape}, got {output.shape}"
        )

    def test_outputs_equivalence(self):
        def set_nan_tensor_to_zero(t):
            # Temporary fallback until `aten::_index_put_impl_` is implemented in mps
            # Track progress in https://github.com/pytorch/pytorch/issues/77764
            device = t.device
            if device.type == "mps":
                t = t.to("cpu")
            t[t != t] = 0
            return t.to(device)

        def recursive_check(tuple_object, dict_object):
            if isinstance(tuple_object, (list, tuple)):
                for tuple_iterable_value, dict_iterable_value in zip(tuple_object, dict_object.values()):
                    recursive_check(tuple_iterable_value, dict_iterable_value)
            elif isinstance(tuple_object, dict):
                for tuple_iterable_value, dict_iterable_value in zip(tuple_object.values(), dict_object.values()):
                    recursive_check(tuple_iterable_value, dict_iterable_value)
            elif tuple_object is None:
                return
            else:
                assert torch.allclose(
                    set_nan_tensor_to_zero(tuple_object), set_nan_tensor_to_zero(dict_object), atol=1e-5
                ), (
                    "Tuple and dict output are not equal. Difference:"
                    f" {torch.max(torch.abs(tuple_object - dict_object))}. Tuple has `nan`:"
                    f" {torch.isnan(tuple_object).any()} and `inf`: {torch.isinf(tuple_object)}. Dict has"
                    f" `nan`: {torch.isnan(dict_object).any()} and `inf`: {torch.isinf(dict_object)}."
                )

        model = self.model_class(**self.get_init_dict())
        model.to(torch_device)
        model.eval()

        with torch.no_grad():
            outputs_dict = model(**self.get_dummy_inputs())
            outputs_tuple = model(**self.get_dummy_inputs(), return_dict=False)

        recursive_check(outputs_tuple, outputs_dict)

    def test_getattr_is_correct(self):
        init_dict = self.get_init_dict()
        model = self.model_class(**init_dict)

        # save some things to test
        model.dummy_attribute = 5
        model.register_to_config(test_attribute=5)

        logger = logging.get_logger("diffusers.models.modeling_utils")
        # 30 for warning
        logger.setLevel(30)
        with CaptureLogger(logger) as cap_logger:
            assert hasattr(model, "dummy_attribute")
            assert getattr(model, "dummy_attribute") == 5
            assert model.dummy_attribute == 5

        # no warning should be thrown
        assert cap_logger.out == ""

        logger = logging.get_logger("diffusers.models.modeling_utils")
        # 30 for warning
        logger.setLevel(30)
        with CaptureLogger(logger) as cap_logger:
            assert hasattr(model, "save_pretrained")
            fn = model.save_pretrained
            fn_1 = getattr(model, "save_pretrained")

            assert fn == fn_1
        # no warning should be thrown
        assert cap_logger.out == ""

        # warning should be thrown for config attributes accessed directly
        with pytest.warns(FutureWarning):
            assert model.test_attribute == 5

        with pytest.warns(FutureWarning):
            assert getattr(model, "test_attribute") == 5

        with pytest.raises(AttributeError) as error:
            model.does_not_exist

        assert str(error.value) == f"'{type(model).__name__}' object has no attribute 'does_not_exist'"

    @require_accelerator
    @pytest.mark.skipif(
        torch_device not in ["cuda", "xpu"],
        reason="float16 and bfloat16 can only be used with an accelerator",
    )
    def test_keep_in_fp32_modules(self):
        model = self.model_class(**self.get_init_dict())
        fp32_modules = model._keep_in_fp32_modules

        if fp32_modules is None or len(fp32_modules) == 0:
            pytest.skip("Model does not have _keep_in_fp32_modules defined.")

        # Test with float16
        model.to(torch_device)
        model.to(torch.float16)

        for name, param in model.named_parameters():
            if any(module_to_keep_in_fp32 in name.split(".") for module_to_keep_in_fp32 in fp32_modules):
                assert param.dtype == torch.float32, f"Parameter {name} should be float32 but got {param.dtype}"
            else:
                assert param.dtype == torch.float16, f"Parameter {name} should be float16 but got {param.dtype}"

    @require_accelerator
    @pytest.mark.skipif(
        torch_device not in ["cuda", "xpu"],
        reason="float16 and bfloat16 can only be use for inference with an accelerator",
    )
    def test_from_save_pretrained_float16_bfloat16(self):
        model = self.model_class(**self.get_init_dict())
        model.to(torch_device)
        fp32_modules = model._keep_in_fp32_modules

        with tempfile.TemporaryDirectory() as tmp_dir:
            for torch_dtype in [torch.bfloat16, torch.float16]:
                model.to(torch_dtype).save_pretrained(tmp_dir)
                model_loaded = self.model_class.from_pretrained(tmp_dir, torch_dtype=torch_dtype).to(torch_device)

                for name, param in model_loaded.named_parameters():
                    if any(module_to_keep_in_fp32 in name.split(".") for module_to_keep_in_fp32 in fp32_modules):
                        assert param.data.dtype == torch.float32
                    else:
                        assert param.data.dtype == torch_dtype

                with torch.no_grad():
                    output = model(**self.get_dummy_inputs())
                    output_loaded = model_loaded(**self.get_dummy_inputs())

                assert torch.allclose(output, output_loaded, atol=1e-4), (
                    f"Loaded model output differs for {torch_dtype}"
                )

    @require_accelerator
    def test_sharded_checkpoints(self):
        torch.manual_seed(0)
        config = self.get_init_dict()
        inputs_dict = self.get_dummy_inputs()
        model = self.model_class(**config).eval()
        model = model.to(torch_device)

        base_output = model(**inputs_dict)

        model_size = compute_module_persistent_sizes(model)[""]
        max_shard_size = int((model_size * 0.75) / (2**10))  # Convert to KB as these test models are small
        with tempfile.TemporaryDirectory() as tmp_dir:
            model.cpu().save_pretrained(tmp_dir, max_shard_size=f"{max_shard_size}KB")
            assert os.path.exists(os.path.join(tmp_dir, SAFE_WEIGHTS_INDEX_NAME)), "Index file should exist"

            # Check if the right number of shards exists
            expected_num_shards = calculate_expected_num_shards(os.path.join(tmp_dir, SAFE_WEIGHTS_INDEX_NAME))
            actual_num_shards = len([file for file in os.listdir(tmp_dir) if file.endswith(".safetensors")])
            assert actual_num_shards == expected_num_shards, (
                f"Expected {expected_num_shards} shards, got {actual_num_shards}"
            )

            new_model = self.model_class.from_pretrained(tmp_dir).eval()
            new_model = new_model.to(torch_device)

            torch.manual_seed(0)
            inputs_dict_new = self.get_dummy_inputs()
            new_output = new_model(**inputs_dict_new)

            assert torch.allclose(base_output[0], new_output[0], atol=1e-5), (
                "Output should match after sharded save/load"
            )

    @require_accelerator
    def test_sharded_checkpoints_with_variant(self):
        torch.manual_seed(0)
        config = self.get_init_dict()
        inputs_dict = self.get_dummy_inputs()
        model = self.model_class(**config).eval()
        model = model.to(torch_device)

        base_output = model(**inputs_dict)

        model_size = compute_module_persistent_sizes(model)[""]
        max_shard_size = int((model_size * 0.75) / (2**10))  # Convert to KB as these test models are small
        variant = "fp16"
        with tempfile.TemporaryDirectory() as tmp_dir:
            model.cpu().save_pretrained(tmp_dir, max_shard_size=f"{max_shard_size}KB", variant=variant)

            index_filename = _add_variant(SAFE_WEIGHTS_INDEX_NAME, variant)
            assert os.path.exists(os.path.join(tmp_dir, index_filename)), (
                f"Variant index file {index_filename} should exist"
            )

            # Check if the right number of shards exists
            expected_num_shards = calculate_expected_num_shards(os.path.join(tmp_dir, index_filename))
            actual_num_shards = len([file for file in os.listdir(tmp_dir) if file.endswith(".safetensors")])
            assert actual_num_shards == expected_num_shards, (
                f"Expected {expected_num_shards} shards, got {actual_num_shards}"
            )

            new_model = self.model_class.from_pretrained(tmp_dir, variant=variant).eval()
            new_model = new_model.to(torch_device)

            torch.manual_seed(0)
            inputs_dict_new = self.get_dummy_inputs()
            new_output = new_model(**inputs_dict_new)

            assert torch.allclose(base_output[0], new_output[0], atol=1e-5), (
                "Output should match after variant sharded save/load"
            )

    def test_sharded_checkpoints_with_parallel_loading(self):
        import time

        from diffusers.utils import constants

        torch.manual_seed(0)
        config = self.get_init_dict()
        inputs_dict = self.get_dummy_inputs()
        model = self.model_class(**config).eval()
        model = model.to(torch_device)

        base_output = model(**inputs_dict)

        model_size = compute_module_persistent_sizes(model)[""]
        max_shard_size = int((model_size * 0.75) / (2**10))  # Convert to KB as these test models are small

        # Save original values to restore after test
        original_parallel_loading = constants.HF_ENABLE_PARALLEL_LOADING
        original_parallel_workers = getattr(constants, "HF_PARALLEL_WORKERS", None)

        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                model.cpu().save_pretrained(tmp_dir, max_shard_size=f"{max_shard_size}KB")
                assert os.path.exists(os.path.join(tmp_dir, SAFE_WEIGHTS_INDEX_NAME)), "Index file should exist"

                # Check if the right number of shards exists
                expected_num_shards = calculate_expected_num_shards(os.path.join(tmp_dir, SAFE_WEIGHTS_INDEX_NAME))
                actual_num_shards = len([file for file in os.listdir(tmp_dir) if file.endswith(".safetensors")])
                assert actual_num_shards == expected_num_shards, (
                    f"Expected {expected_num_shards} shards, got {actual_num_shards}"
                )

                # Load without parallel loading
                constants.HF_ENABLE_PARALLEL_LOADING = False
                start_time = time.time()
                model_sequential = self.model_class.from_pretrained(tmp_dir).eval()
                sequential_load_time = time.time() - start_time
                model_sequential = model_sequential.to(torch_device)

                torch.manual_seed(0)

                # Load with parallel loading
                constants.HF_ENABLE_PARALLEL_LOADING = True
                constants.DEFAULT_HF_PARALLEL_LOADING_WORKERS = 2

                start_time = time.time()
                model_parallel = self.model_class.from_pretrained(tmp_dir).eval()
                parallel_load_time = time.time() - start_time
                model_parallel = model_parallel.to(torch_device)

                torch.manual_seed(0)
                inputs_dict_parallel = self.get_dummy_inputs()
                output_parallel = model_parallel(**inputs_dict_parallel)

                assert torch.allclose(base_output[0], output_parallel[0], atol=1e-5), (
                    "Output should match with parallel loading"
                )

                # Verify parallel loading is faster or at least not significantly slower
                assert parallel_load_time < sequential_load_time, (
                    f"Parallel loading took {parallel_load_time:.4f}s, sequential took {sequential_load_time:.4f}s"
                )
        finally:
            # Restore original values
            constants.HF_ENABLE_PARALLEL_LOADING = original_parallel_loading
            if original_parallel_workers is not None:
                constants.HF_PARALLEL_WORKERS = original_parallel_workers

    @require_torch_multi_accelerator
    def test_model_parallelism(self):
        if self.model_class._no_split_modules is None:
            pytest.skip("Test not supported for this model as `_no_split_modules` is not set.")

        config = self.get_init_dict()
        inputs_dict = self.get_dummy_inputs()
        model = self.model_class(**config).eval()

        model = model.to(torch_device)

        torch.manual_seed(0)
        base_output = model(**inputs_dict)

        model_size = compute_module_sizes(model)[""]
        max_gpu_sizes = [int(p * model_size) for p in self.model_split_percents]

        with tempfile.TemporaryDirectory() as tmp_dir:
            model.cpu().save_pretrained(tmp_dir)

            for max_size in max_gpu_sizes:
                max_memory = {0: max_size, 1: model_size * 2, "cpu": model_size * 2}
                new_model = self.model_class.from_pretrained(tmp_dir, device_map="auto", max_memory=max_memory)
                # Making sure part of the model will be on GPU 0 and GPU 1
                assert set(new_model.hf_device_map.values()) == {0, 1}, "Model should be split across GPUs"

                check_device_map_is_respected(new_model, new_model.hf_device_map)

                torch.manual_seed(0)
                new_output = new_model(**inputs_dict)

                assert torch.allclose(base_output[0], new_output[0], atol=1e-5), (
                    "Output should match with model parallelism"
                )
