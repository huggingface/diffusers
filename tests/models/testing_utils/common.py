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
from collections import defaultdict
from typing import Any, Dict, Optional, Type

import pytest
import torch
import torch.nn as nn
from accelerate.utils.modeling import _get_proper_dtype, compute_module_sizes, dtype_byte_size

from diffusers.utils import SAFE_WEIGHTS_INDEX_NAME, _add_variant, logging
from diffusers.utils.testing_utils import require_accelerator, require_torch_multi_accelerator

from ...testing_utils import assert_tensors_close, torch_device


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


def cast_inputs_to_dtype(inputs, current_dtype, target_dtype):
    if torch.is_tensor(inputs):
        return inputs.to(target_dtype) if inputs.dtype == current_dtype else inputs
    if isinstance(inputs, dict):
        return {k: cast_inputs_to_dtype(v, current_dtype, target_dtype) for k, v in inputs.items()}
    if isinstance(inputs, list):
        return [cast_inputs_to_dtype(v, current_dtype, target_dtype) for v in inputs]

    return inputs


class BaseModelTesterConfig:
    """
    Base class defining the configuration interface for model testing.

    This class defines the contract that all model test classes must implement.
    It provides a consistent interface for accessing model configuration, initialization
    parameters, and test inputs across all testing mixins.

    Required properties (must be implemented by subclasses):
        - model_class: The model class to test

    Optional properties (can be overridden, have sensible defaults):
        - pretrained_model_name_or_path: Hub repository ID for pretrained model (default: None)
        - pretrained_model_kwargs: Additional kwargs for from_pretrained (default: {})
        - output_shape: Expected output shape for output validation tests (default: None)
        - model_split_percents: Percentages for model parallelism tests (default: [0.5, 0.7])

    Required methods (must be implemented by subclasses):
        - get_init_dict(): Returns dict of arguments to initialize the model
        - get_dummy_inputs(): Returns dict of inputs to pass to the model forward pass

    Example usage:
        class MyModelTestConfig(BaseModelTesterConfig):
            @property
            def model_class(self):
                return MyModel

            @property
            def pretrained_model_name_or_path(self):
                return "org/my-model"

            @property
            def output_shape(self):
                return (1, 3, 32, 32)

            def get_init_dict(self):
                return {"in_channels": 3, "out_channels": 3}

            def get_dummy_inputs(self):
                return {"sample": torch.randn(1, 3, 32, 32, device=torch_device)}

        class TestMyModel(MyModelTestConfig, ModelTesterMixin, QuantizationTesterMixin):
            pass
    """

    # ==================== Required Properties ====================

    @property
    def model_class(self) -> Type[nn.Module]:
        """The model class to test. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement the `model_class` property.")

    # ==================== Optional Properties ====================

    @property
    def pretrained_model_name_or_path(self) -> Optional[str]:
        """Hub repository ID for the pretrained model (used for quantization and hub tests)."""
        return None

    @property
    def pretrained_model_kwargs(self) -> Dict[str, Any]:
        """Additional kwargs to pass to from_pretrained (e.g., subfolder, variant)."""
        return {}

    @property
    def output_shape(self) -> Optional[tuple]:
        """Expected output shape for output validation tests."""
        return None

    @property
    def model_split_percents(self) -> list:
        """Percentages for model parallelism tests."""
        return [0.9]

    # ==================== Required Methods ====================

    def get_init_dict(self) -> Dict[str, Any]:
        """
        Returns dict of arguments to initialize the model.

        Returns:
            Dict[str, Any]: Initialization arguments for the model constructor.

        Example:
            return {
                "in_channels": 3,
                "out_channels": 3,
                "sample_size": 32,
            }
        """
        raise NotImplementedError("Subclasses must implement `get_init_dict()`.")

    def get_dummy_inputs(self) -> Dict[str, Any]:
        """
        Returns dict of inputs to pass to the model forward pass.

        Returns:
            Dict[str, Any]: Input tensors/values for model.forward().

        Example:
            return {
                "sample": torch.randn(1, 3, 32, 32, device=torch_device),
                "timestep": torch.tensor([1], device=torch_device),
            }
        """
        raise NotImplementedError("Subclasses must implement `get_dummy_inputs()`.")


class ModelTesterMixin:
    """
    Base mixin class for model testing with common test methods.

    This mixin expects the test class to also inherit from BaseModelTesterConfig
    (or implement its interface) which provides:
        - model_class: The model class to test
        - get_init_dict(): Returns dict of arguments to initialize the model
        - get_dummy_inputs(): Returns dict of inputs to pass to the model forward pass

    Example:
        class MyModelTestConfig(BaseModelTesterConfig):
            model_class = MyModel
            def get_init_dict(self): ...
            def get_dummy_inputs(self): ...

        class TestMyModel(MyModelTestConfig, ModelTesterMixin):
            pass
    """

    @torch.no_grad()
    def test_from_save_pretrained(self, tmp_path, atol=5e-5, rtol=5e-5):
        torch.manual_seed(0)
        model = self.model_class(**self.get_init_dict())
        model.to(torch_device)
        model.eval()

        model.save_pretrained(tmp_path)
        new_model = self.model_class.from_pretrained(tmp_path)
        new_model.to(torch_device)

        for param_name in model.state_dict().keys():
            param_1 = model.state_dict()[param_name]
            param_2 = new_model.state_dict()[param_name]
            assert param_1.shape == param_2.shape, (
                f"Parameter shape mismatch for {param_name}. Original: {param_1.shape}, loaded: {param_2.shape}"
            )

        image = model(**self.get_dummy_inputs(), return_dict=False)[0]
        new_image = new_model(**self.get_dummy_inputs(), return_dict=False)[0]

        assert_tensors_close(image, new_image, atol=atol, rtol=rtol, msg="Models give different forward passes.")

    @torch.no_grad()
    def test_from_save_pretrained_variant(self, tmp_path, atol=5e-5, rtol=0):
        model = self.model_class(**self.get_init_dict())
        model.to(torch_device)
        model.eval()

        model.save_pretrained(tmp_path, variant="fp16")
        new_model = self.model_class.from_pretrained(tmp_path, variant="fp16")

        with pytest.raises(OSError) as exc_info:
            self.model_class.from_pretrained(tmp_path)

        assert "Error no file named diffusion_pytorch_model.bin found in directory" in str(exc_info.value)

        new_model.to(torch_device)

        image = model(**self.get_dummy_inputs(), return_dict=False)[0]
        new_image = new_model(**self.get_dummy_inputs(), return_dict=False)[0]

        assert_tensors_close(image, new_image, atol=atol, rtol=rtol, msg="Models give different forward passes.")

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16], ids=["fp32", "fp16", "bf16"])
    def test_from_save_pretrained_dtype(self, tmp_path, dtype):
        model = self.model_class(**self.get_init_dict())
        model.to(torch_device)
        model.eval()

        if torch_device == "mps" and dtype == torch.bfloat16:
            pytest.skip(reason=f"{dtype} is not supported on {torch_device}")

        model.to(dtype)
        model.save_pretrained(tmp_path)
        new_model = self.model_class.from_pretrained(tmp_path, low_cpu_mem_usage=True, torch_dtype=dtype)
        assert new_model.dtype == dtype
        if hasattr(self.model_class, "_keep_in_fp32_modules") and self.model_class._keep_in_fp32_modules is None:
            # When loading without accelerate dtype == torch.float32 if _keep_in_fp32_modules is not None
            new_model = self.model_class.from_pretrained(tmp_path, low_cpu_mem_usage=False, torch_dtype=dtype)
            assert new_model.dtype == dtype

    @torch.no_grad()
    def test_determinism(self, atol=1e-5, rtol=0):
        model = self.model_class(**self.get_init_dict())
        model.to(torch_device)
        model.eval()

        first = model(**self.get_dummy_inputs(), return_dict=False)[0]
        second = model(**self.get_dummy_inputs(), return_dict=False)[0]

        first_flat = first.flatten()
        second_flat = second.flatten()
        mask = ~(torch.isnan(first_flat) | torch.isnan(second_flat))
        first_filtered = first_flat[mask]
        second_filtered = second_flat[mask]

        assert_tensors_close(
            first_filtered, second_filtered, atol=atol, rtol=rtol, msg="Model outputs are not deterministic"
        )

    @torch.no_grad()
    def test_output(self, expected_output_shape=None):
        model = self.model_class(**self.get_init_dict())
        model.to(torch_device)
        model.eval()

        inputs_dict = self.get_dummy_inputs()
        output = model(**inputs_dict, return_dict=False)[0]

        assert output is not None, "Model output is None"
        assert output[0].shape == expected_output_shape or self.output_shape, (
            f"Output shape does not match expected. Expected {expected_output_shape}, got {output.shape}"
        )

    @torch.no_grad()
    def test_outputs_equivalence(self, atol=1e-5, rtol=0):
        def set_nan_tensor_to_zero(t):
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
                assert_tensors_close(
                    set_nan_tensor_to_zero(tuple_object),
                    set_nan_tensor_to_zero(dict_object),
                    atol=atol,
                    rtol=rtol,
                    msg="Tuple and dict output are not equal",
                )

        model = self.model_class(**self.get_init_dict())
        model.to(torch_device)
        model.eval()

        outputs_dict = model(**self.get_dummy_inputs())
        outputs_tuple = model(**self.get_dummy_inputs(), return_dict=False)

        recursive_check(outputs_tuple, outputs_dict)

    def test_getattr_is_correct(self, caplog):
        init_dict = self.get_init_dict()
        model = self.model_class(**init_dict)

        # save some things to test
        model.dummy_attribute = 5
        model.register_to_config(test_attribute=5)

        logger_name = "diffusers.models.modeling_utils"
        with caplog.at_level(logging.WARNING, logger=logger_name):
            caplog.clear()
            assert hasattr(model, "dummy_attribute")
            assert getattr(model, "dummy_attribute") == 5
            assert model.dummy_attribute == 5

        # no warning should be thrown
        assert caplog.text == ""

        with caplog.at_level(logging.WARNING, logger=logger_name):
            caplog.clear()
            assert hasattr(model, "save_pretrained")
            fn = model.save_pretrained
            fn_1 = getattr(model, "save_pretrained")

            assert fn == fn_1

        # no warning should be thrown
        assert caplog.text == ""

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
    def test_keep_in_fp32_modules(self, tmp_path):
        model = self.model_class(**self.get_init_dict())
        fp32_modules = model._keep_in_fp32_modules

        if fp32_modules is None or len(fp32_modules) == 0:
            pytest.skip("Model does not have _keep_in_fp32_modules defined.")

        # Save the model and reload with float16 dtype
        # _keep_in_fp32_modules is only enforced during from_pretrained loading
        model.save_pretrained(tmp_path)
        model = self.model_class.from_pretrained(tmp_path, torch_dtype=torch.float16).to(torch_device)

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
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
    @torch.no_grad()
    def test_from_save_pretrained_dtype_inference(self, tmp_path, dtype, atol=1e-4, rtol=0):
        model = self.model_class(**self.get_init_dict())
        model.to(torch_device)
        fp32_modules = model._keep_in_fp32_modules or []

        model.to(dtype).save_pretrained(tmp_path)
        model_loaded = self.model_class.from_pretrained(tmp_path, torch_dtype=dtype).to(torch_device)

        for name, param in model_loaded.named_parameters():
            if fp32_modules and any(
                module_to_keep_in_fp32 in name.split(".") for module_to_keep_in_fp32 in fp32_modules
            ):
                assert param.data.dtype == torch.float32
            else:
                assert param.data.dtype == dtype

        inputs = cast_inputs_to_dtype(self.get_dummy_inputs(), torch.float32, dtype)
        output = model(**inputs, return_dict=False)[0]
        output_loaded = model_loaded(**inputs, return_dict=False)[0]

        assert_tensors_close(
            output, output_loaded, atol=atol, rtol=rtol, msg=f"Loaded model output differs for {dtype}"
        )

    @require_accelerator
    @torch.no_grad()
    def test_sharded_checkpoints(self, tmp_path, atol=1e-5, rtol=0):
        torch.manual_seed(0)
        config = self.get_init_dict()
        inputs_dict = self.get_dummy_inputs()
        model = self.model_class(**config).eval()
        model = model.to(torch_device)

        base_output = model(**inputs_dict, return_dict=False)[0]

        model_size = compute_module_persistent_sizes(model)[""]
        max_shard_size = int((model_size * 0.75) / (2**10))  # Convert to KB as these test models are small

        model.cpu().save_pretrained(tmp_path, max_shard_size=f"{max_shard_size}KB")
        assert os.path.exists(os.path.join(tmp_path, SAFE_WEIGHTS_INDEX_NAME)), "Index file should exist"

        # Check if the right number of shards exists
        expected_num_shards = calculate_expected_num_shards(os.path.join(tmp_path, SAFE_WEIGHTS_INDEX_NAME))
        actual_num_shards = len([file for file in os.listdir(tmp_path) if file.endswith(".safetensors")])
        assert actual_num_shards == expected_num_shards, (
            f"Expected {expected_num_shards} shards, got {actual_num_shards}"
        )

        new_model = self.model_class.from_pretrained(tmp_path).eval()
        new_model = new_model.to(torch_device)

        torch.manual_seed(0)
        inputs_dict_new = self.get_dummy_inputs()
        new_output = new_model(**inputs_dict_new, return_dict=False)[0]

        assert_tensors_close(
            base_output, new_output, atol=atol, rtol=rtol, msg="Output should match after sharded save/load"
        )

    @require_accelerator
    @torch.no_grad()
    def test_sharded_checkpoints_with_variant(self, tmp_path, atol=1e-5, rtol=0):
        torch.manual_seed(0)
        config = self.get_init_dict()
        inputs_dict = self.get_dummy_inputs()
        model = self.model_class(**config).eval()
        model = model.to(torch_device)

        base_output = model(**inputs_dict, return_dict=False)[0]

        model_size = compute_module_persistent_sizes(model)[""]
        max_shard_size = int((model_size * 0.75) / (2**10))  # Convert to KB as these test models are small
        variant = "fp16"

        model.cpu().save_pretrained(tmp_path, max_shard_size=f"{max_shard_size}KB", variant=variant)

        index_filename = _add_variant(SAFE_WEIGHTS_INDEX_NAME, variant)
        assert os.path.exists(os.path.join(tmp_path, index_filename)), (
            f"Variant index file {index_filename} should exist"
        )

        # Check if the right number of shards exists
        expected_num_shards = calculate_expected_num_shards(os.path.join(tmp_path, index_filename))
        actual_num_shards = len([file for file in os.listdir(tmp_path) if file.endswith(".safetensors")])
        assert actual_num_shards == expected_num_shards, (
            f"Expected {expected_num_shards} shards, got {actual_num_shards}"
        )

        new_model = self.model_class.from_pretrained(tmp_path, variant=variant).eval()
        new_model = new_model.to(torch_device)

        torch.manual_seed(0)
        inputs_dict_new = self.get_dummy_inputs()
        new_output = new_model(**inputs_dict_new, return_dict=False)[0]

        assert_tensors_close(
            base_output, new_output, atol=atol, rtol=rtol, msg="Output should match after variant sharded save/load"
        )

    @torch.no_grad()
    def test_sharded_checkpoints_with_parallel_loading(self, tmp_path, atol=1e-5, rtol=0):
        from diffusers.utils import constants

        torch.manual_seed(0)
        config = self.get_init_dict()
        inputs_dict = self.get_dummy_inputs()
        model = self.model_class(**config).eval()
        model = model.to(torch_device)

        base_output = model(**inputs_dict, return_dict=False)[0]

        model_size = compute_module_persistent_sizes(model)[""]
        max_shard_size = int((model_size * 0.75) / (2**10))  # Convert to KB as these test models are small

        # Save original values to restore after test
        original_parallel_loading = constants.HF_ENABLE_PARALLEL_LOADING
        original_parallel_workers = getattr(constants, "HF_PARALLEL_WORKERS", None)

        try:
            model.cpu().save_pretrained(tmp_path, max_shard_size=f"{max_shard_size}KB")
            assert os.path.exists(os.path.join(tmp_path, SAFE_WEIGHTS_INDEX_NAME)), "Index file should exist"

            # Check if the right number of shards exists
            expected_num_shards = calculate_expected_num_shards(os.path.join(tmp_path, SAFE_WEIGHTS_INDEX_NAME))
            actual_num_shards = len([file for file in os.listdir(tmp_path) if file.endswith(".safetensors")])
            assert actual_num_shards == expected_num_shards, (
                f"Expected {expected_num_shards} shards, got {actual_num_shards}"
            )

            # Load without parallel loading
            constants.HF_ENABLE_PARALLEL_LOADING = False
            model_sequential = self.model_class.from_pretrained(tmp_path).eval()
            model_sequential = model_sequential.to(torch_device)

            # Load with parallel loading
            constants.HF_ENABLE_PARALLEL_LOADING = True
            constants.DEFAULT_HF_PARALLEL_LOADING_WORKERS = 2

            torch.manual_seed(0)
            model_parallel = self.model_class.from_pretrained(tmp_path).eval()
            model_parallel = model_parallel.to(torch_device)

            torch.manual_seed(0)
            inputs_dict_parallel = self.get_dummy_inputs()
            output_parallel = model_parallel(**inputs_dict_parallel, return_dict=False)[0]

            assert_tensors_close(
                base_output, output_parallel, atol=atol, rtol=rtol, msg="Output should match with parallel loading"
            )

        finally:
            # Restore original values
            constants.HF_ENABLE_PARALLEL_LOADING = original_parallel_loading
            if original_parallel_workers is not None:
                constants.HF_PARALLEL_WORKERS = original_parallel_workers

    @require_torch_multi_accelerator
    @torch.no_grad()
    def test_model_parallelism(self, tmp_path, atol=1e-5, rtol=0):
        if self.model_class._no_split_modules is None:
            pytest.skip("Test not supported for this model as `_no_split_modules` is not set.")

        config = self.get_init_dict()
        inputs_dict = self.get_dummy_inputs()
        model = self.model_class(**config).eval()

        model = model.to(torch_device)

        torch.manual_seed(0)
        base_output = model(**inputs_dict, return_dict=False)[0]

        model_size = compute_module_sizes(model)[""]
        max_gpu_sizes = [int(p * model_size) for p in self.model_split_percents]

        model.cpu().save_pretrained(tmp_path)

        for max_size in max_gpu_sizes:
            max_memory = {0: max_size, 1: model_size * 2, "cpu": model_size * 2}
            new_model = self.model_class.from_pretrained(tmp_path, device_map="auto", max_memory=max_memory)
            # Making sure part of the model will be on GPU 0 and GPU 1
            assert set(new_model.hf_device_map.values()) == {0, 1}, "Model should be split across GPUs"

            check_device_map_is_respected(new_model, new_model.hf_device_map)

            torch.manual_seed(0)
            new_output = new_model(**inputs_dict, return_dict=False)[0]

            assert_tensors_close(
                base_output, new_output, atol=atol, rtol=rtol, msg="Output should match with model parallelism"
            )
