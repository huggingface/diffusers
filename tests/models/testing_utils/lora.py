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

import pytest
import safetensors.torch
import torch

from diffusers.utils.testing_utils import check_if_dicts_are_equal

from ...testing_utils import is_lora, require_peft_backend, torch_device


def check_if_lora_correctly_set(model) -> bool:
    """
    Check if LoRA layers are correctly set in the model.

    Args:
        model: The model to check

    Returns:
        bool: True if LoRA is correctly set, False otherwise
    """
    from peft.tuners.tuners_utils import BaseTunerLayer

    for module in model.modules():
        if isinstance(module, BaseTunerLayer):
            return True
    return False


@is_lora
@require_peft_backend
class LoraTesterMixin:
    """
    Mixin class for testing LoRA/PEFT functionality on models.

    Expected class attributes to be set by subclasses:
        - model_class: The model class to test

    Expected methods to be implemented by subclasses:
        - get_init_dict(): Returns dict of arguments to initialize the model
        - get_dummy_inputs(): Returns dict of inputs to pass to the model forward pass

    Pytest mark: lora
        Use `pytest -m "not lora"` to skip these tests
    """

    def setup_method(self):
        from diffusers.loaders.peft import PeftAdapterMixin

        if not issubclass(self.model_class, PeftAdapterMixin):
            pytest.skip(f"PEFT is not supported for this model ({self.model_class.__name__}).")

    def test_save_load_lora_adapter(self, rank=4, lora_alpha=4, use_dora=False):
        from peft import LoraConfig
        from peft.utils import get_peft_model_state_dict

        init_dict = self.get_init_dict()
        inputs_dict = self.get_dummy_inputs()
        model = self.model_class(**init_dict).to(torch_device)

        torch.manual_seed(0)
        output_no_lora = model(**inputs_dict, return_dict=False)[0]

        denoiser_lora_config = LoraConfig(
            r=rank,
            lora_alpha=lora_alpha,
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
            init_lora_weights=False,
            use_dora=use_dora,
        )
        model.add_adapter(denoiser_lora_config)
        assert check_if_lora_correctly_set(model), "LoRA layers not set correctly"

        torch.manual_seed(0)
        outputs_with_lora = model(**inputs_dict, return_dict=False)[0]

        assert not torch.allclose(output_no_lora, outputs_with_lora, atol=1e-4, rtol=1e-4), (
            "Output should differ with LoRA enabled"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_lora_adapter(tmpdir)
            assert os.path.isfile(os.path.join(tmpdir, "pytorch_lora_weights.safetensors")), (
                "LoRA weights file not created"
            )

            state_dict_loaded = safetensors.torch.load_file(os.path.join(tmpdir, "pytorch_lora_weights.safetensors"))

            model.unload_lora()
            assert not check_if_lora_correctly_set(model), "LoRA should be unloaded"

            model.load_lora_adapter(tmpdir, prefix=None, use_safetensors=True)
            state_dict_retrieved = get_peft_model_state_dict(model, adapter_name="default_0")

            for k in state_dict_loaded:
                loaded_v = state_dict_loaded[k]
                retrieved_v = state_dict_retrieved[k].to(loaded_v.device)
                assert torch.allclose(loaded_v, retrieved_v), f"Mismatch in LoRA weight {k}"

            assert check_if_lora_correctly_set(model), "LoRA layers not set correctly after reload"

        torch.manual_seed(0)
        outputs_with_lora_2 = model(**inputs_dict, return_dict=False)[0]

        assert not torch.allclose(output_no_lora, outputs_with_lora_2, atol=1e-4, rtol=1e-4), (
            "Output should differ with LoRA enabled"
        )
        assert torch.allclose(outputs_with_lora, outputs_with_lora_2, atol=1e-4, rtol=1e-4), (
            "Outputs should match before and after save/load"
        )

    def test_lora_wrong_adapter_name_raises_error(self):
        from peft import LoraConfig

        init_dict = self.get_init_dict()
        model = self.model_class(**init_dict).to(torch_device)

        denoiser_lora_config = LoraConfig(
            r=4,
            lora_alpha=4,
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
            init_lora_weights=False,
            use_dora=False,
        )
        model.add_adapter(denoiser_lora_config)
        assert check_if_lora_correctly_set(model), "LoRA layers not set correctly"

        with tempfile.TemporaryDirectory() as tmpdir:
            wrong_name = "foo"
            with pytest.raises(ValueError) as exc_info:
                model.save_lora_adapter(tmpdir, adapter_name=wrong_name)

            assert f"Adapter name {wrong_name} not found in the model." in str(exc_info.value)

    def test_lora_adapter_metadata_is_loaded_correctly(self, rank=4, lora_alpha=4, use_dora=False):
        from peft import LoraConfig

        init_dict = self.get_init_dict()
        model = self.model_class(**init_dict).to(torch_device)

        denoiser_lora_config = LoraConfig(
            r=rank,
            lora_alpha=lora_alpha,
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
            init_lora_weights=False,
            use_dora=use_dora,
        )
        model.add_adapter(denoiser_lora_config)
        metadata = model.peft_config["default"].to_dict()
        assert check_if_lora_correctly_set(model), "LoRA layers not set correctly"

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_lora_adapter(tmpdir)
            model_file = os.path.join(tmpdir, "pytorch_lora_weights.safetensors")
            assert os.path.isfile(model_file), "LoRA weights file not created"

            model.unload_lora()
            assert not check_if_lora_correctly_set(model), "LoRA should be unloaded"

            model.load_lora_adapter(tmpdir, prefix=None, use_safetensors=True)
            parsed_metadata = model.peft_config["default_0"].to_dict()
            check_if_dicts_are_equal(metadata, parsed_metadata)

    def test_lora_adapter_wrong_metadata_raises_error(self):
        from peft import LoraConfig

        from diffusers.loaders.lora_base import LORA_ADAPTER_METADATA_KEY

        init_dict = self.get_init_dict()
        model = self.model_class(**init_dict).to(torch_device)

        denoiser_lora_config = LoraConfig(
            r=4,
            lora_alpha=4,
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
            init_lora_weights=False,
            use_dora=False,
        )
        model.add_adapter(denoiser_lora_config)
        assert check_if_lora_correctly_set(model), "LoRA layers not set correctly"

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_lora_adapter(tmpdir)
            model_file = os.path.join(tmpdir, "pytorch_lora_weights.safetensors")
            assert os.path.isfile(model_file), "LoRA weights file not created"

            # Perturb the metadata in the state dict
            loaded_state_dict = safetensors.torch.load_file(model_file)
            metadata = {"format": "pt"}
            lora_adapter_metadata = denoiser_lora_config.to_dict()
            lora_adapter_metadata.update({"foo": 1, "bar": 2})
            for key, value in lora_adapter_metadata.items():
                if isinstance(value, set):
                    lora_adapter_metadata[key] = list(value)
            metadata[LORA_ADAPTER_METADATA_KEY] = json.dumps(lora_adapter_metadata, indent=2, sort_keys=True)
            safetensors.torch.save_file(loaded_state_dict, model_file, metadata=metadata)

            model.unload_lora()
            assert not check_if_lora_correctly_set(model), "LoRA should be unloaded"

            with pytest.raises(TypeError) as exc_info:
                model.load_lora_adapter(tmpdir, prefix=None, use_safetensors=True)
            assert "`LoraConfig` class could not be instantiated" in str(exc_info.value)
