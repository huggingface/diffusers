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

import os
import tempfile

import torch

from diffusers.models.attention_processor import IPAdapterAttnProcessor

from ...testing_utils import is_ip_adapter, torch_device


def create_ip_adapter_state_dict(model):
    """
    Create a dummy IP Adapter state dict for testing.

    Args:
        model: The model to create IP adapter weights for

    Returns:
        dict: IP adapter state dict with to_k_ip and to_v_ip weights
    """
    ip_state_dict = {}
    key_id = 1

    for name in model.attn_processors.keys():
        # Skip self-attention processors
        cross_attention_dim = getattr(model.config, "cross_attention_dim", None)
        if cross_attention_dim is None:
            continue

        # Get hidden size based on model architecture
        hidden_size = getattr(model.config, "hidden_size", cross_attention_dim)

        # Create IP adapter processor to get state dict structure
        sd = IPAdapterAttnProcessor(
            hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, scale=1.0
        ).state_dict()

        ip_state_dict.update(
            {
                f"{key_id}.to_k_ip.weight": sd["to_k_ip.0.weight"],
                f"{key_id}.to_v_ip.weight": sd["to_v_ip.0.weight"],
            }
        )
        key_id += 2

    return {"ip_adapter": ip_state_dict}


def check_if_ip_adapter_correctly_set(model) -> bool:
    """
    Check if IP Adapter processors are correctly set in the model.

    Args:
        model: The model to check

    Returns:
        bool: True if IP Adapter is correctly set, False otherwise
    """
    for module in model.attn_processors.values():
        if isinstance(module, IPAdapterAttnProcessor):
            return True
    return False


@is_ip_adapter
class IPAdapterTesterMixin:
    """
    Mixin class for testing IP Adapter functionality on models.

    Expected class attributes to be set by subclasses:
        - model_class: The model class to test

    Expected methods to be implemented by subclasses:
        - get_init_dict(): Returns dict of arguments to initialize the model
        - get_dummy_inputs(): Returns dict of inputs to pass to the model forward pass

    Pytest mark: ip_adapter
        Use `pytest -m "not ip_adapter"` to skip these tests
    """

    def create_ip_adapter_state_dict(self, model):
        raise NotImplementedError("child class must implement method to create IPAdapter State Dict")

    def test_load_ip_adapter(self):
        init_dict = self.get_init_dict()
        inputs_dict = self.get_dummy_inputs()
        model = self.model_class(**init_dict).to(torch_device)

        torch.manual_seed(0)
        output_no_adapter = model(**inputs_dict, return_dict=False)[0]

        # Create dummy IP adapter state dict
        ip_adapter_state_dict = self.create_ip_adapter_state_dict(model)

        # Load IP adapter
        model._load_ip_adapter_weights([ip_adapter_state_dict])
        assert check_if_ip_adapter_correctly_set(model), "IP Adapter processors not set correctly"

        torch.manual_seed(0)
        # Create dummy image embeds for IP adapter
        cross_attention_dim = getattr(model.config, "cross_attention_dim", 32)
        image_embeds = torch.randn(1, 1, cross_attention_dim).to(torch_device)
        inputs_dict_with_adapter = inputs_dict.copy()
        inputs_dict_with_adapter["image_embeds"] = image_embeds

        outputs_with_adapter = model(**inputs_dict_with_adapter, return_dict=False)[0]

        assert not torch.allclose(
            output_no_adapter, outputs_with_adapter, atol=1e-4, rtol=1e-4
        ), "Output should differ with IP Adapter enabled"

    def test_ip_adapter_scale(self):
        init_dict = self.get_init_dict()
        inputs_dict = self.get_dummy_inputs()
        model = self.model_class(**init_dict).to(torch_device)

        # Create and load dummy IP adapter state dict
        ip_adapter_state_dict = create_ip_adapter_state_dict(model)
        model._load_ip_adapter_weights([ip_adapter_state_dict])

        # Test scale = 0.0 (no effect)
        model.set_ip_adapter_scale(0.0)
        torch.manual_seed(0)
        output_scale_zero = model(**inputs_dict_with_adapter, return_dict=False)[0]

        # Test scale = 1.0 (full effect)
        model.set_ip_adapter_scale(1.0)
        torch.manual_seed(0)
        output_scale_one = model(**inputs_dict_with_adapter, return_dict=False)[0]

        # Outputs should differ with different scales
        assert not torch.allclose(
            output_scale_zero, output_scale_one, atol=1e-4, rtol=1e-4
        ), "Output should differ with different IP Adapter scales"

    def test_unload_ip_adapter(self):
        init_dict = self.get_init_dict()
        model = self.model_class(**init_dict).to(torch_device)

        # Save original processors
        original_processors = {k: type(v).__name__ for k, v in model.attn_processors.items()}

        # Create and load IP adapter
        ip_adapter_state_dict = create_ip_adapter_state_dict(model)
        model._load_ip_adapter_weights([ip_adapter_state_dict])
        assert check_if_ip_adapter_correctly_set(model), "IP Adapter should be set"

        # Unload IP adapter
        model.unload_ip_adapter()
        assert not check_if_ip_adapter_correctly_set(model), "IP Adapter should be unloaded"

        # Verify processors are restored
        current_processors = {k: type(v).__name__ for k, v in model.attn_processors.items()}
        assert original_processors == current_processors, "Processors should be restored after unload"

    def test_ip_adapter_save_load(self):
        init_dict = self.get_init_dict()
        inputs_dict = self.get_dummy_inputs()
        model = self.model_class(**init_dict).to(torch_device)

        # Create and load IP adapter
        ip_adapter_state_dict = self.create_ip_adapter_state_dict()
        model._load_ip_adapter_weights([ip_adapter_state_dict])

        torch.manual_seed(0)
        output_before_save = model(**inputs_dict, return_dict=False)[0]

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save the IP adapter weights
            save_path = os.path.join(tmpdir, "ip_adapter.safetensors")
            import safetensors.torch

            safetensors.torch.save_file(ip_adapter_state_dict["ip_adapter"], save_path)

            # Unload and reload
            model.unload_ip_adapter()
            assert not check_if_ip_adapter_correctly_set(model), "IP Adapter should be unloaded"

            # Reload from saved file
            loaded_state_dict = {"ip_adapter": safetensors.torch.load_file(save_path)}
            model._load_ip_adapter_weights([loaded_state_dict])
            assert check_if_ip_adapter_correctly_set(model), "IP Adapter should be loaded"

            torch.manual_seed(0)
            output_after_load = model(**inputs_dict_with_adapter, return_dict=False)[0]

            # Outputs should match before and after save/load
            assert torch.allclose(
                output_before_save, output_after_load, atol=1e-4, rtol=1e-4
            ), "Output should match before and after save/load"
