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

import pytest
import torch

from ...testing_utils import backend_empty_cache, is_ip_adapter, torch_device


def check_if_ip_adapter_correctly_set(model, processor_cls) -> bool:
    """
    Check if IP Adapter processors are correctly set in the model.

    Args:
        model: The model to check

    Returns:
        bool: True if IP Adapter is correctly set, False otherwise
    """
    for module in model.attn_processors.values():
        if isinstance(module, processor_cls):
            return True
    return False


@is_ip_adapter
class IPAdapterTesterMixin:
    """
    Mixin class for testing IP Adapter functionality on models.

    Expected from config mixin:
        - model_class: The model class to test

    Required properties (must be implemented by subclasses):
        - ip_adapter_processor_cls: The IP Adapter processor class to use

    Required methods (must be implemented by subclasses):
        - create_ip_adapter_state_dict(): Creates IP Adapter state dict for testing
        - modify_inputs_for_ip_adapter(): Modifies inputs to include IP Adapter data

    Expected methods from config mixin:
        - get_init_dict(): Returns dict of arguments to initialize the model
        - get_dummy_inputs(): Returns dict of inputs to pass to the model forward pass

    Pytest mark: ip_adapter
        Use `pytest -m "not ip_adapter"` to skip these tests
    """

    def setup_method(self):
        gc.collect()
        backend_empty_cache(torch_device)

    def teardown_method(self):
        gc.collect()
        backend_empty_cache(torch_device)

    @property
    def ip_adapter_processor_cls(self):
        """IP Adapter processor class to use for testing. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement the `ip_adapter_processor_cls` property.")

    def create_ip_adapter_state_dict(self, model):
        raise NotImplementedError("child class must implement method to create IPAdapter State Dict")

    def modify_inputs_for_ip_adapter(self, model, inputs_dict):
        raise NotImplementedError("child class must implement method to create IPAdapter model inputs")

    @torch.no_grad()
    def test_load_ip_adapter(self):
        init_dict = self.get_init_dict()
        inputs_dict = self.get_dummy_inputs()
        model = self.model_class(**init_dict).to(torch_device)

        torch.manual_seed(0)
        output_no_adapter = model(**inputs_dict, return_dict=False)[0]

        ip_adapter_state_dict = self.create_ip_adapter_state_dict(model)

        model._load_ip_adapter_weights([ip_adapter_state_dict])
        assert check_if_ip_adapter_correctly_set(model, self.ip_adapter_processor_cls), (
            "IP Adapter processors not set correctly"
        )

        inputs_dict_with_adapter = self.modify_inputs_for_ip_adapter(model, inputs_dict.copy())
        outputs_with_adapter = model(**inputs_dict_with_adapter, return_dict=False)[0]

        assert not torch.allclose(output_no_adapter, outputs_with_adapter, atol=1e-4, rtol=1e-4), (
            "Output should differ with IP Adapter enabled"
        )

    @pytest.mark.skip(
        reason="Setting IP Adapter scale is not defined at the model level. Enable this test after refactoring"
    )
    def test_ip_adapter_scale(self):
        init_dict = self.get_init_dict()
        inputs_dict = self.get_dummy_inputs()
        model = self.model_class(**init_dict).to(torch_device)

        ip_adapter_state_dict = self.create_ip_adapter_state_dict(model)
        model._load_ip_adapter_weights([ip_adapter_state_dict])

        inputs_dict_with_adapter = self.modify_inputs_for_ip_adapter(model, inputs_dict.copy())

        # Test scale = 0.0 (no effect)
        model.set_ip_adapter_scale(0.0)
        torch.manual_seed(0)
        output_scale_zero = model(**inputs_dict_with_adapter, return_dict=False)[0]

        # Test scale = 1.0 (full effect)
        model.set_ip_adapter_scale(1.0)
        torch.manual_seed(0)
        output_scale_one = model(**inputs_dict_with_adapter, return_dict=False)[0]

        # Outputs should differ with different scales
        assert not torch.allclose(output_scale_zero, output_scale_one, atol=1e-4, rtol=1e-4), (
            "Output should differ with different IP Adapter scales"
        )

    @pytest.mark.skip(
        reason="Unloading IP Adapter is not defined at the model level. Enable this test after refactoring"
    )
    def test_unload_ip_adapter(self):
        init_dict = self.get_init_dict()
        model = self.model_class(**init_dict).to(torch_device)

        # Save original processors
        original_processors = {k: type(v).__name__ for k, v in model.attn_processors.items()}

        # Create and load IP adapter
        ip_adapter_state_dict = self.create_ip_adapter_state_dict(model)
        model._load_ip_adapter_weights([ip_adapter_state_dict])

        assert check_if_ip_adapter_correctly_set(model, self.ip_adapter_processor_cls), "IP Adapter should be set"

        # Unload IP adapter
        model.unload_ip_adapter()

        assert not check_if_ip_adapter_correctly_set(model, self.ip_adapter_processor_cls), (
            "IP Adapter should be unloaded"
        )

        # Verify processors are restored
        current_processors = {k: type(v).__name__ for k, v in model.attn_processors.items()}
        assert original_processors == current_processors, "Processors should be restored after unload"
