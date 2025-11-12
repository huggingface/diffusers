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

import tempfile
from typing import Dict, List, Tuple

import pytest
import torch

from ...testing_utils import torch_device


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

    def get_init_dict(self):
        raise NotImplementedError("get_init_dict must be implemented by subclasses. ")

    def get_dummy_inputs(self):
        raise NotImplementedError(
            "get_dummy_inputs must be implemented by subclasses. " "It should return inputs_dict."
        )

    def check_device_map_is_respected(self, model, device_map):
        """Helper method to check if device map is correctly applied to model parameters."""
        for param_name, param in model.named_parameters():
            # Find device in device_map
            while len(param_name) > 0 and param_name not in device_map:
                param_name = ".".join(param_name.split(".")[:-1])
            if param_name not in device_map:
                raise ValueError("device map is incomplete, it does not contain any device for `param_name`.")

            param_device = device_map[param_name]
            if param_device in ["cpu", "disk"]:
                assert param.device == torch.device(
                    "meta"
                ), f"Expected device 'meta' for {param_name}, got {param.device}"
            else:
                assert param.device == torch.device(
                    param_device
                ), f"Expected device {param_device} for {param_name}, got {param.device}"

    def test_from_save_pretrained(self, expected_max_diff=5e-5):
        """Test that model can be saved and loaded with save_pretrained/from_pretrained."""
        model = self.model_class(**self.get_init_dict())
        model.to(torch_device)
        model.eval()

        with tempfile.TemporaryDirectory() as tmpdirname:
            model.save_pretrained(tmpdirname)
            new_model = self.model_class.from_pretrained(tmpdirname)
            new_model.to(torch_device)

        with torch.no_grad():
            image = model(**self.get_dummy_inputs())

            if isinstance(image, dict):
                image = image.to_tuple()[0]

            new_image = new_model(**self.get_dummy_inputs())

            if isinstance(new_image, dict):
                new_image = new_image.to_tuple()[0]

        max_diff = (image - new_image).abs().max().item()
        assert (
            max_diff <= expected_max_diff
        ), f"Models give different forward passes. Max diff: {max_diff}, expected: {expected_max_diff}"

    def test_from_save_pretrained_variant(self, expected_max_diff=5e-5):
        """Test save_pretrained/from_pretrained with variant parameter."""
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
        assert (
            max_diff <= expected_max_diff
        ), f"Models give different forward passes. Max diff: {max_diff}, expected: {expected_max_diff}"

    def test_from_save_pretrained_dtype(self):
        """Test save_pretrained/from_pretrained preserves dtype correctly."""
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
                    new_model = self.model_class.from_pretrained(
                        tmpdirname, low_cpu_mem_usage=False, torch_dtype=dtype
                    )
                    assert new_model.dtype == dtype

    def test_determinism(self, expected_max_diff=1e-5):
        """Test that model outputs are deterministic across multiple forward passes."""
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
        assert (
            max_diff <= expected_max_diff
        ), f"Model outputs are not deterministic. Max diff: {max_diff}, expected: {expected_max_diff}"

    def test_output(self, expected_output_shape=None):
        """Test that model produces output with expected shape."""
        model = self.model_class(**self.get_init_dict())
        model.to(torch_device)
        model.eval()

        inputs_dict = self.get_dummy_inputs()
        with torch.no_grad():
            output = model(**inputs_dict)

            if isinstance(output, dict):
                output = output.to_tuple()[0]

        assert output is not None, "Model output is None"
        assert (
            output.shape == expected_output_shape
        ), f"Output shape does not match expected. Expected {expected_output_shape}, got {output.shape}"

    def test_model_from_pretrained(self):
        """Test that model loaded from pretrained matches original model."""
        model = self.model_class(**self.get_init_dict())
        model.to(torch_device)
        model.eval()

        # test if the model can be loaded from the config
        # and has all the expected shape
        with tempfile.TemporaryDirectory() as tmpdirname:
            model.save_pretrained(tmpdirname, safe_serialization=False)
            new_model = self.model_class.from_pretrained(tmpdirname)
            new_model.to(torch_device)
            new_model.eval()

        # check if all parameters shape are the same
        for param_name in model.state_dict().keys():
            param_1 = model.state_dict()[param_name]
            param_2 = new_model.state_dict()[param_name]
            assert (
                param_1.shape == param_2.shape
            ), f"Parameter shape mismatch for {param_name}. Original: {param_1.shape}, loaded: {param_2.shape}"

        with torch.no_grad():
            output_1 = model(**self.get_dummy_inputs())

            if isinstance(output_1, dict):
                output_1 = output_1.to_tuple()[0]

            output_2 = new_model(**self.get_dummy_inputs())

            if isinstance(output_2, dict):
                output_2 = output_2.to_tuple()[0]

        assert (
            output_1.shape == output_2.shape
        ), f"Output shape mismatch. Original: {output_1.shape}, loaded: {output_2.shape}"

    def test_outputs_equivalence(self):
        """Test that dict and tuple outputs are equivalent."""

        def set_nan_tensor_to_zero(t):
            # Temporary fallback until `aten::_index_put_impl_` is implemented in mps
            # Track progress in https://github.com/pytorch/pytorch/issues/77764
            device = t.device
            if device.type == "mps":
                t = t.to("cpu")
            t[t != t] = 0
            return t.to(device)

        def recursive_check(tuple_object, dict_object):
            if isinstance(tuple_object, (List, Tuple)):
                for tuple_iterable_value, dict_iterable_value in zip(tuple_object, dict_object.values()):
                    recursive_check(tuple_iterable_value, dict_iterable_value)
            elif isinstance(tuple_object, Dict):
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

    def test_model_config_to_json_string(self):
        """Test model config can be serialized to JSON string."""
        model = self.model_class(**self.get_init_dict())

        json_string = model.config.to_json_string()
        assert isinstance(json_string, str), "Config to_json_string should return a string"
        assert len(json_string) > 0, "JSON string should not be empty"

    def test_keep_in_fp32_modules(self):
        r"""
        A simple tests to check if the modules under `_keep_in_fp32_modules` are kept in fp32 when we load the model in fp16/bf16
        Also ensures if inference works.
        """
        if not hasattr(self.model_class, "_keep_in_fp32_modules"):
            pytest.skip("Model does not have _keep_in_fp32_modules")

        fp32_modules = self.model_class._keep_in_fp32_modules

        for torch_dtype in [torch.bfloat16, torch.float16]:
            model = self.model_class.from_pretrained(self.pretrained_model_name_or_path, torch_dtype=torch_dtype).to(
                torch_device
            )
            for name, param in model.named_parameters():
                if any(module_to_keep_in_fp32 in name.split(".") for module_to_keep_in_fp32 in fp32_modules):
                    assert param.data == torch.float32
                else:
                    assert param.data == torch_dtype
