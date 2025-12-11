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
import tempfile

import pytest
import torch

from diffusers import BitsAndBytesConfig, GGUFQuantizationConfig, NVIDIAModelOptConfig, QuantoConfig, TorchAoConfig
from diffusers.utils.import_utils import (
    is_bitsandbytes_available,
    is_gguf_available,
    is_nvidia_modelopt_available,
    is_optimum_quanto_available,
    is_torchao_available,
    is_torchao_version,
)

from ...testing_utils import (
    backend_empty_cache,
    is_bitsandbytes,
    is_gguf,
    is_modelopt,
    is_quanto,
    is_torchao,
    nightly,
    require_accelerate,
    require_accelerator,
    require_bitsandbytes_version_greater,
    require_gguf_version_greater_or_equal,
    require_modelopt_version_greater_or_equal,
    require_quanto,
    require_torchao_version_greater_or_equal,
    torch_device,
)


if is_nvidia_modelopt_available():
    import modelopt.torch.quantization as mtq

if is_bitsandbytes_available():
    import bitsandbytes as bnb

if is_optimum_quanto_available():
    from optimum.quanto import QLinear

if is_gguf_available():
    pass

if is_torchao_available():
    if is_torchao_version(">=", "0.9.0"):
        pass


@require_accelerator
class QuantizationTesterMixin:
    """
    Base mixin class providing common test implementations for quantization testing.

    Backend-specific mixins should:
    1. Implement _create_quantized_model(config_kwargs)
    2. Implement _verify_if_layer_quantized(name, module, config_kwargs)
    3. Define their config dict (e.g., BNB_CONFIGS, QUANTO_WEIGHT_TYPES, etc.)
    4. Use @pytest.mark.parametrize to create tests that call the common test methods below

    Expected class attributes:
        - model_class: The model class to test
        - pretrained_model_name_or_path: Hub repository ID for the pretrained model
        - pretrained_model_kwargs: (Optional) Dict of kwargs to pass to from_pretrained (e.g., {"subfolder": "transformer"})

    Expected methods in test classes:
        - get_dummy_inputs(): Returns dict of inputs to pass to the model forward pass
    """

    def setup_method(self):
        gc.collect()
        backend_empty_cache(torch_device)

    def teardown_method(self):
        gc.collect()
        backend_empty_cache(torch_device)

    def _create_quantized_model(self, config_kwargs, **extra_kwargs):
        """
        Create a quantized model with the given config kwargs.

        Args:
            config_kwargs: Quantization config parameters
            **extra_kwargs: Additional kwargs to pass to from_pretrained (e.g., device_map, offload_folder)
        """
        raise NotImplementedError("Subclass must implement _create_quantized_model")

    def _verify_if_layer_quantized(self, name, module, config_kwargs):
        raise NotImplementedError("Subclass must implement _verify_if_layer_quantized")

    def _is_module_quantized(self, module):
        """
        Check if a module is quantized. Returns True if quantized, False otherwise.
        Default implementation tries _verify_if_layer_quantized and catches exceptions.
        Subclasses can override for more efficient checking.
        """
        try:
            self._verify_if_layer_quantized("", module, {})
            return True
        except (AssertionError, AttributeError):
            return False

    def _load_unquantized_model(self):
        kwargs = getattr(self, "pretrained_model_kwargs", {})
        return self.model_class.from_pretrained(self.pretrained_model_name_or_path, **kwargs)

    def _test_quantization_num_parameters(self, config_kwargs):
        model = self._load_unquantized_model()
        num_params = model.num_parameters()

        model_quantized = self._create_quantized_model(config_kwargs)
        num_params_quantized = model_quantized.num_parameters()

        assert num_params == num_params_quantized, (
            f"Parameter count mismatch: unquantized={num_params}, quantized={num_params_quantized}"
        )

    def _test_quantization_memory_footprint(self, config_kwargs, expected_memory_reduction=1.2):
        model = self._load_unquantized_model()
        mem = model.get_memory_footprint()

        model_quantized = self._create_quantized_model(config_kwargs)
        mem_quantized = model_quantized.get_memory_footprint()

        ratio = mem / mem_quantized
        assert ratio >= expected_memory_reduction, (
            f"Memory ratio {ratio:.2f} is less than expected ({expected_memory_reduction}x). unquantized={mem}, quantized={mem_quantized}"
        )

    def _test_quantization_inference(self, config_kwargs):
        model_quantized = self._create_quantized_model(config_kwargs)

        with torch.no_grad():
            inputs = self.get_dummy_inputs()
            output = model_quantized(**inputs)

            if isinstance(output, tuple):
                output = output[0]
            assert output is not None, "Model output is None"
            assert not torch.isnan(output).any(), "Model output contains NaN"

    def _test_quantization_dtype_assignment(self, config_kwargs):
        model = self._create_quantized_model(config_kwargs)

        with pytest.raises(ValueError):
            model.to(torch.float16)

        with pytest.raises(ValueError):
            device_0 = f"{torch_device}:0"
            model.to(device=device_0, dtype=torch.float16)

        with pytest.raises(ValueError):
            model.float()

        with pytest.raises(ValueError):
            model.half()

        model.to(torch_device)

    def _test_quantization_lora_inference(self, config_kwargs):
        try:
            from peft import LoraConfig
        except ImportError:
            pytest.skip("peft is not available")

        from diffusers.loaders.peft import PeftAdapterMixin

        if not issubclass(self.model_class, PeftAdapterMixin):
            pytest.skip(f"PEFT is not supported for this model ({self.model_class.__name__})")

        model = self._create_quantized_model(config_kwargs)

        lora_config = LoraConfig(
            r=4,
            lora_alpha=4,
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
            init_lora_weights=False,
        )
        model.add_adapter(lora_config)

        with torch.no_grad():
            inputs = self.get_dummy_inputs()
            output = model(**inputs)

            if isinstance(output, tuple):
                output = output[0]
            assert output is not None, "Model output is None with LoRA"
            assert not torch.isnan(output).any(), "Model output contains NaN with LoRA"

    def _test_quantization_serialization(self, config_kwargs):
        model = self._create_quantized_model(config_kwargs)

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir, safe_serialization=True)

            model_loaded = self.model_class.from_pretrained(tmpdir)

            with torch.no_grad():
                inputs = self.get_dummy_inputs()
                output = model_loaded(**inputs)
                if isinstance(output, tuple):
                    output = output[0]
                assert not torch.isnan(output).any(), "Loaded model output contains NaN"

    def _test_quantized_layers(self, config_kwargs):
        model_fp = self._load_unquantized_model()
        num_linear_layers = sum(1 for module in model_fp.modules() if isinstance(module, torch.nn.Linear))

        model_quantized = self._create_quantized_model(config_kwargs)

        num_fp32_modules = 0
        if hasattr(model_quantized, "_keep_in_fp32_modules") and model_quantized._keep_in_fp32_modules:
            for name, module in model_quantized.named_modules():
                if isinstance(module, torch.nn.Linear):
                    if any(fp32_name in name for fp32_name in model_quantized._keep_in_fp32_modules):
                        num_fp32_modules += 1

        expected_quantized_layers = num_linear_layers - num_fp32_modules

        num_quantized_layers = 0
        for name, module in model_quantized.named_modules():
            if isinstance(module, torch.nn.Linear):
                if hasattr(model_quantized, "_keep_in_fp32_modules") and model_quantized._keep_in_fp32_modules:
                    if any(fp32_name in name for fp32_name in model_quantized._keep_in_fp32_modules):
                        continue
                self._verify_if_layer_quantized(name, module, config_kwargs)
                num_quantized_layers += 1

        assert num_quantized_layers > 0, (
            f"No quantized layers found in model (expected {expected_quantized_layers} linear layers, {num_fp32_modules} kept in FP32)"
        )
        assert num_quantized_layers == expected_quantized_layers, (
            f"Quantized layer count mismatch: expected {expected_quantized_layers}, got {num_quantized_layers} (total linear layers: {num_linear_layers}, FP32 modules: {num_fp32_modules})"
        )

    def _test_quantization_modules_to_not_convert(self, config_kwargs, modules_to_not_convert):
        """
        Test that modules specified in modules_to_not_convert are not quantized.

        Args:
            config_kwargs: Base quantization config kwargs
            modules_to_not_convert: List of module names to exclude from quantization
        """
        # Create config with modules_to_not_convert
        config_kwargs_with_exclusion = config_kwargs.copy()
        config_kwargs_with_exclusion["modules_to_not_convert"] = modules_to_not_convert

        model_with_exclusion = self._create_quantized_model(config_kwargs_with_exclusion)

        # Find a module that should NOT be quantized
        found_excluded = False
        for name, module in model_with_exclusion.named_modules():
            if isinstance(module, torch.nn.Linear):
                # Check if this module is in the exclusion list
                if any(excluded in name for excluded in modules_to_not_convert):
                    found_excluded = True
                    # This module should NOT be quantized
                    assert not self._is_module_quantized(module), (
                        f"Module {name} should not be quantized but was found to be quantized"
                    )

        assert found_excluded, f"No linear layers found in excluded modules: {modules_to_not_convert}"

        # Find a module that SHOULD be quantized (not in exclusion list)
        found_quantized = False
        for name, module in model_with_exclusion.named_modules():
            if isinstance(module, torch.nn.Linear):
                # Check if this module is NOT in the exclusion list
                if not any(excluded in name for excluded in modules_to_not_convert):
                    if self._is_module_quantized(module):
                        found_quantized = True
                        break

        assert found_quantized, "No quantized layers found outside of excluded modules"

        # Compare memory footprint with fully quantized model
        model_fully_quantized = self._create_quantized_model(config_kwargs)

        mem_with_exclusion = model_with_exclusion.get_memory_footprint()
        mem_fully_quantized = model_fully_quantized.get_memory_footprint()

        assert mem_with_exclusion > mem_fully_quantized, (
            f"Model with exclusions should be larger. With exclusion: {mem_with_exclusion}, fully quantized: {mem_fully_quantized}"
        )

    def _test_quantization_device_map(self, config_kwargs):
        """
        Test that quantized models work correctly with device_map="auto".

        Args:
            config_kwargs: Base quantization config kwargs
        """
        model = self._create_quantized_model(config_kwargs, device_map="auto")

        # Verify device map is set
        assert hasattr(model, "hf_device_map"), "Model should have hf_device_map attribute"
        assert model.hf_device_map is not None, "hf_device_map should not be None"

        # Verify inference works
        with torch.no_grad():
            inputs = self.get_dummy_inputs()
            output = model(**inputs)
            if isinstance(output, tuple):
                output = output[0]
            assert output is not None, "Model output is None"
            assert not torch.isnan(output).any(), "Model output contains NaN"


@is_bitsandbytes
@nightly
@require_accelerator
@require_bitsandbytes_version_greater("0.43.2")
@require_accelerate
class BitsAndBytesTesterMixin(QuantizationTesterMixin):
    """
    Mixin class for testing BitsAndBytes quantization on models.

    Expected class attributes:
        - model_class: The model class to test
        - pretrained_model_name_or_path: Hub repository ID for the pretrained model
        - pretrained_model_kwargs: (Optional) Dict of kwargs to pass to from_pretrained (e.g., {"subfolder": "transformer"})

    Expected methods to be implemented by subclasses:
        - get_dummy_inputs(): Returns dict of inputs to pass to the model forward pass

    Optional class attributes:
        - BNB_CONFIGS: Dict of config name -> BitsAndBytesConfig kwargs to test

    Pytest mark: bitsandbytes
        Use `pytest -m "not bitsandbytes"` to skip these tests
    """

    # Standard BnB configs tested for all models
    # Subclasses can override to add or modify configs
    BNB_CONFIGS = {
        "4bit_nf4": {
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": torch.float16,
        },
        "4bit_fp4": {
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "fp4",
            "bnb_4bit_compute_dtype": torch.float16,
        },
        "8bit": {
            "load_in_8bit": True,
        },
    }

    BNB_EXPECTED_MEMORY_REDUCTIONS = {
        "4bit_nf4": 3.0,
        "4bit_fp4": 3.0,
        "8bit": 1.5,
    }

    def _create_quantized_model(self, config_kwargs, **extra_kwargs):
        config = BitsAndBytesConfig(**config_kwargs)
        kwargs = getattr(self, "pretrained_model_kwargs", {}).copy()
        kwargs["quantization_config"] = config
        kwargs.update(extra_kwargs)
        return self.model_class.from_pretrained(self.pretrained_model_name_or_path, **kwargs)

    def _verify_if_layer_quantized(self, name, module, config_kwargs):
        expected_weight_class = bnb.nn.Params4bit if config_kwargs.get("load_in_4bit") else bnb.nn.Int8Params
        assert module.weight.__class__ == expected_weight_class, (
            f"Layer {name} has weight type {module.weight.__class__}, expected {expected_weight_class}"
        )

    @pytest.mark.parametrize("config_name", list(BNB_CONFIGS.keys()))
    def test_bnb_quantization_num_parameters(self, config_name):
        self._test_quantization_num_parameters(self.BNB_CONFIGS[config_name])

    @pytest.mark.parametrize("config_name", list(BNB_CONFIGS.keys()))
    def test_bnb_quantization_memory_footprint(self, config_name):
        expected = self.BNB_EXPECTED_MEMORY_REDUCTIONS.get(config_name, 1.2)
        self._test_quantization_memory_footprint(self.BNB_CONFIGS[config_name], expected_memory_reduction=expected)

    @pytest.mark.parametrize("config_name", list(BNB_CONFIGS.keys()))
    def test_bnb_quantization_inference(self, config_name):
        self._test_quantization_inference(self.BNB_CONFIGS[config_name])

    @pytest.mark.parametrize("config_name", ["4bit_nf4"])
    def test_bnb_quantization_dtype_assignment(self, config_name):
        self._test_quantization_dtype_assignment(self.BNB_CONFIGS[config_name])

    @pytest.mark.parametrize("config_name", ["4bit_nf4"])
    def test_bnb_quantization_lora_inference(self, config_name):
        self._test_quantization_lora_inference(self.BNB_CONFIGS[config_name])

    @pytest.mark.parametrize("config_name", ["4bit_nf4"])
    def test_bnb_quantization_serialization(self, config_name):
        self._test_quantization_serialization(self.BNB_CONFIGS[config_name])

    @pytest.mark.parametrize("config_name", list(BNB_CONFIGS.keys()))
    def test_bnb_quantized_layers(self, config_name):
        self._test_quantized_layers(self.BNB_CONFIGS[config_name])

    @pytest.mark.parametrize("config_name", list(BNB_CONFIGS.keys()))
    def test_bnb_quantization_config_serialization(self, config_name):
        model = self._create_quantized_model(self.BNB_CONFIGS[config_name])

        assert "quantization_config" in model.config, "Missing quantization_config"
        _ = model.config["quantization_config"].to_dict()
        _ = model.config["quantization_config"].to_diff_dict()
        _ = model.config["quantization_config"].to_json_string()

    def test_bnb_original_dtype(self):
        config_name = list(self.BNB_CONFIGS.keys())[0]
        config_kwargs = self.BNB_CONFIGS[config_name]

        model = self._create_quantized_model(config_kwargs)

        assert "_pre_quantization_dtype" in model.config, "Missing _pre_quantization_dtype"
        assert model.config["_pre_quantization_dtype"] in [
            torch.float16,
            torch.float32,
            torch.bfloat16,
        ], f"Unexpected dtype: {model.config['_pre_quantization_dtype']}"

    def test_bnb_keep_modules_in_fp32(self):
        if not hasattr(self.model_class, "_keep_in_fp32_modules"):
            pytest.skip(f"{self.model_class.__name__} does not have _keep_in_fp32_modules")

        config_kwargs = self.BNB_CONFIGS["4bit_nf4"]

        original_fp32_modules = getattr(self.model_class, "_keep_in_fp32_modules", None)
        self.model_class._keep_in_fp32_modules = ["proj_out"]

        try:
            model = self._create_quantized_model(config_kwargs)

            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Linear):
                    if any(fp32_name in name for fp32_name in model._keep_in_fp32_modules):
                        assert module.weight.dtype == torch.float32, (
                            f"Module {name} should be FP32 but is {module.weight.dtype}"
                        )
                    else:
                        assert module.weight.dtype == torch.uint8, (
                            f"Module {name} should be uint8 but is {module.weight.dtype}"
                        )

            with torch.no_grad():
                inputs = self.get_dummy_inputs()
                _ = model(**inputs)
        finally:
            if original_fp32_modules is not None:
                self.model_class._keep_in_fp32_modules = original_fp32_modules

    def test_bnb_modules_to_not_convert(self):
        """Test that modules_to_not_convert parameter works correctly."""
        modules_to_exclude = getattr(self, "modules_to_not_convert_for_test", None)
        if modules_to_exclude is None:
            pytest.skip("modules_to_not_convert_for_test not defined for this model")

        self._test_quantization_modules_to_not_convert(self.BNB_CONFIGS["4bit_nf4"], modules_to_exclude)

    def test_bnb_device_map(self):
        """Test that device_map='auto' works correctly with quantization."""
        self._test_quantization_device_map(self.BNB_CONFIGS["4bit_nf4"])


@is_quanto
@nightly
@require_quanto
@require_accelerate
@require_accelerator
class QuantoTesterMixin(QuantizationTesterMixin):
    """
    Mixin class for testing Quanto quantization on models.

    Expected class attributes:
        - model_class: The model class to test
        - pretrained_model_name_or_path: Hub repository ID for the pretrained model
        - pretrained_model_kwargs: (Optional) Dict of kwargs to pass to from_pretrained (e.g., {"subfolder": "transformer"})

    Expected methods to be implemented by subclasses:
        - get_dummy_inputs(): Returns dict of inputs to pass to the model forward pass

    Optional class attributes:
        - QUANTO_WEIGHT_TYPES: Dict of weight_type_name -> qtype

    Pytest mark: quanto
        Use `pytest -m "not quanto"` to skip these tests
    """

    QUANTO_WEIGHT_TYPES = {
        "float8": {"weights_dtype": "float8"},
        "int8": {"weights_dtype": "int8"},
        "int4": {"weights_dtype": "int4"},
        "int2": {"weights_dtype": "int2"},
    }

    QUANTO_EXPECTED_MEMORY_REDUCTIONS = {
        "float8": 1.5,
        "int8": 1.5,
        "int4": 3.0,
        "int2": 7.0,
    }

    def _create_quantized_model(self, config_kwargs, **extra_kwargs):
        config = QuantoConfig(**config_kwargs)
        kwargs = getattr(self, "pretrained_model_kwargs", {}).copy()
        kwargs["quantization_config"] = config
        kwargs.update(extra_kwargs)
        return self.model_class.from_pretrained(self.pretrained_model_name_or_path, **kwargs)

    def _verify_if_layer_quantized(self, name, module, config_kwargs):
        assert isinstance(module, QLinear), f"Layer {name} is not QLinear, got {type(module)}"

    @pytest.mark.parametrize("weight_type_name", list(QUANTO_WEIGHT_TYPES.keys()))
    def test_quanto_quantization_num_parameters(self, weight_type_name):
        self._test_quantization_num_parameters(self.QUANTO_WEIGHT_TYPES[weight_type_name])

    @pytest.mark.parametrize("weight_type_name", list(QUANTO_WEIGHT_TYPES.keys()))
    def test_quanto_quantization_memory_footprint(self, weight_type_name):
        expected = self.QUANTO_EXPECTED_MEMORY_REDUCTIONS.get(weight_type_name, 1.2)
        self._test_quantization_memory_footprint(
            self.QUANTO_WEIGHT_TYPES[weight_type_name], expected_memory_reduction=expected
        )

    @pytest.mark.parametrize("weight_type_name", list(QUANTO_WEIGHT_TYPES.keys()))
    def test_quanto_quantization_inference(self, weight_type_name):
        self._test_quantization_inference(self.QUANTO_WEIGHT_TYPES[weight_type_name])

    @pytest.mark.parametrize("weight_type_name", ["int8"])
    def test_quanto_quantized_layers(self, weight_type_name):
        self._test_quantized_layers(self.QUANTO_WEIGHT_TYPES[weight_type_name])

    @pytest.mark.parametrize("weight_type_name", ["int8"])
    def test_quanto_quantization_lora_inference(self, weight_type_name):
        self._test_quantization_lora_inference(self.QUANTO_WEIGHT_TYPES[weight_type_name])

    @pytest.mark.parametrize("weight_type_name", ["int8"])
    def test_quanto_quantization_serialization(self, weight_type_name):
        self._test_quantization_serialization(self.QUANTO_WEIGHT_TYPES[weight_type_name])

    def test_quanto_modules_to_not_convert(self):
        """Test that modules_to_not_convert parameter works correctly."""
        modules_to_exclude = getattr(self, "modules_to_not_convert_for_test", None)
        if modules_to_exclude is None:
            pytest.skip("modules_to_not_convert_for_test not defined for this model")

        self._test_quantization_modules_to_not_convert(self.QUANTO_WEIGHT_TYPES["int8"], modules_to_exclude)

    def test_quanto_device_map(self):
        """Test that device_map='auto' works correctly with quantization."""
        self._test_quantization_device_map(self.QUANTO_WEIGHT_TYPES["int8"])


@is_torchao
@require_accelerator
@require_torchao_version_greater_or_equal("0.7.0")
class TorchAoTesterMixin(QuantizationTesterMixin):
    """
    Mixin class for testing TorchAO quantization on models.

    Expected class attributes:
        - model_class: The model class to test
        - pretrained_model_name_or_path: Hub repository ID for the pretrained model
        - pretrained_model_kwargs: (Optional) Dict of kwargs to pass to from_pretrained (e.g., {"subfolder": "transformer"})

    Expected methods to be implemented by subclasses:
        - get_dummy_inputs(): Returns dict of inputs to pass to the model forward pass

    Optional class attributes:
        - TORCHAO_QUANT_TYPES: Dict of quantization type strings to test

    Pytest mark: torchao
        Use `pytest -m "not torchao"` to skip these tests
    """

    TORCHAO_QUANT_TYPES = {
        "int4wo": {"quant_type": "int4_weight_only"},
        "int8wo": {"quant_type": "int8_weight_only"},
        "int8dq": {"quant_type": "int8_dynamic_activation_int8_weight"},
    }

    TORCHAO_EXPECTED_MEMORY_REDUCTIONS = {
        "int4wo": 3.0,
        "int8wo": 1.5,
        "int8dq": 1.5,
    }

    def _create_quantized_model(self, config_kwargs, **extra_kwargs):
        config = TorchAoConfig(**config_kwargs)
        kwargs = getattr(self, "pretrained_model_kwargs", {}).copy()
        kwargs["quantization_config"] = config
        kwargs.update(extra_kwargs)
        return self.model_class.from_pretrained(self.pretrained_model_name_or_path, **kwargs)

    def _verify_if_layer_quantized(self, name, module, config_kwargs):
        assert isinstance(module, torch.nn.Linear), f"Layer {name} is not Linear, got {type(module)}"

    @pytest.mark.parametrize("quant_type", list(TORCHAO_QUANT_TYPES.keys()))
    def test_torchao_quantization_num_parameters(self, quant_type):
        self._test_quantization_num_parameters(self.TORCHAO_QUANT_TYPES[quant_type])

    @pytest.mark.parametrize("quant_type", list(TORCHAO_QUANT_TYPES.keys()))
    def test_torchao_quantization_memory_footprint(self, quant_type):
        expected = self.TORCHAO_EXPECTED_MEMORY_REDUCTIONS.get(quant_type, 1.2)
        self._test_quantization_memory_footprint(
            self.TORCHAO_QUANT_TYPES[quant_type], expected_memory_reduction=expected
        )

    @pytest.mark.parametrize("quant_type", list(TORCHAO_QUANT_TYPES.keys()))
    def test_torchao_quantization_inference(self, quant_type):
        self._test_quantization_inference(self.TORCHAO_QUANT_TYPES[quant_type])

    @pytest.mark.parametrize("quant_type", ["int8wo"])
    def test_torchao_quantized_layers(self, quant_type):
        self._test_quantized_layers(self.TORCHAO_QUANT_TYPES[quant_type])

    @pytest.mark.parametrize("quant_type", ["int8wo"])
    def test_torchao_quantization_lora_inference(self, quant_type):
        self._test_quantization_lora_inference(self.TORCHAO_QUANT_TYPES[quant_type])

    @pytest.mark.parametrize("quant_type", ["int8wo"])
    def test_torchao_quantization_serialization(self, quant_type):
        self._test_quantization_serialization(self.TORCHAO_QUANT_TYPES[quant_type])

    def test_torchao_modules_to_not_convert(self):
        """Test that modules_to_not_convert parameter works correctly."""
        # Get a module name that exists in the model - this needs to be set by test classes
        # For now, use a generic pattern that should work with transformer models
        modules_to_exclude = getattr(self, "modules_to_not_convert_for_test", None)
        if modules_to_exclude is None:
            pytest.skip("modules_to_not_convert_for_test not defined for this model")

        self._test_quantization_modules_to_not_convert(self.TORCHAO_QUANT_TYPES["int8wo"], modules_to_exclude)

    def test_torchao_device_map(self):
        """Test that device_map='auto' works correctly with quantization."""
        self._test_quantization_device_map(self.TORCHAO_QUANT_TYPES["int8wo"])


@is_gguf
@nightly
@require_accelerate
@require_accelerator
@require_gguf_version_greater_or_equal("0.10.0")
class GGUFTesterMixin(QuantizationTesterMixin):
    """
    Mixin class for testing GGUF quantization on models.

    Expected class attributes:
        - model_class: The model class to test
        - gguf_filename: URL or path to the GGUF file

    Expected methods to be implemented by subclasses:
        - get_dummy_inputs(): Returns dict of inputs to pass to the model forward pass

    Pytest mark: gguf
        Use `pytest -m "not gguf"` to skip these tests
    """

    gguf_filename = None

    def _create_quantized_model(self, config_kwargs=None, **extra_kwargs):
        if config_kwargs is None:
            config_kwargs = {"compute_dtype": torch.bfloat16}

        config = GGUFQuantizationConfig(**config_kwargs)
        kwargs = {
            "quantization_config": config,
            "torch_dtype": config_kwargs.get("compute_dtype", torch.bfloat16),
        }
        kwargs.update(extra_kwargs)
        return self.model_class.from_single_file(self.gguf_filename, **kwargs)

    def _verify_if_layer_quantized(self, name, module, config_kwargs=None):
        from diffusers.quantizers.gguf.utils import GGUFParameter

        assert isinstance(module.weight, GGUFParameter), f"{name} weight is not GGUFParameter"
        assert hasattr(module.weight, "quant_type"), f"{name} weight missing quant_type"
        assert module.weight.dtype == torch.uint8, f"{name} weight dtype should be uint8"

    def test_gguf_quantization_inference(self):
        self._test_quantization_inference({"compute_dtype": torch.bfloat16})

    def test_gguf_keep_modules_in_fp32(self):
        if not hasattr(self.model_class, "_keep_in_fp32_modules"):
            pytest.skip(f"{self.model_class.__name__} does not have _keep_in_fp32_modules")

        _keep_in_fp32_modules = self.model_class._keep_in_fp32_modules
        self.model_class._keep_in_fp32_modules = ["proj_out"]

        try:
            model = self._create_quantized_model()

            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Linear):
                    if any(fp32_name in name for fp32_name in model._keep_in_fp32_modules):
                        assert module.weight.dtype == torch.float32, f"Module {name} should be FP32"
        finally:
            self.model_class._keep_in_fp32_modules = _keep_in_fp32_modules

    def test_gguf_quantization_dtype_assignment(self):
        self._test_quantization_dtype_assignment({"compute_dtype": torch.bfloat16})

    def test_gguf_quantization_lora_inference(self):
        self._test_quantization_lora_inference({"compute_dtype": torch.bfloat16})

    def test_gguf_dequantize_model(self):
        from diffusers.quantizers.gguf.utils import GGUFLinear, GGUFParameter

        model = self._create_quantized_model()
        model.dequantize()

        def _check_for_gguf_linear(model):
            has_children = list(model.children())
            if not has_children:
                return

            for name, module in model.named_children():
                if isinstance(module, torch.nn.Linear):
                    assert not isinstance(module, GGUFLinear), f"{name} is still GGUFLinear"
                    assert not isinstance(module.weight, GGUFParameter), f"{name} weight is still GGUFParameter"

        for name, module in model.named_children():
            _check_for_gguf_linear(module)

    def test_gguf_quantized_layers(self):
        self._test_quantized_layers({"compute_dtype": torch.bfloat16})


@is_modelopt
@nightly
@require_accelerator
@require_accelerate
@require_modelopt_version_greater_or_equal("0.33.1")
class ModelOptTesterMixin(QuantizationTesterMixin):
    """
    Mixin class for testing NVIDIA ModelOpt quantization on models.

    Expected class attributes:
        - model_class: The model class to test
        - pretrained_model_name_or_path: Hub repository ID for the pretrained model
        - pretrained_model_kwargs: (Optional) Dict of kwargs to pass to from_pretrained (e.g., {"subfolder": "transformer"})

    Expected methods to be implemented by subclasses:
        - get_dummy_inputs(): Returns dict of inputs to pass to the model forward pass

    Optional class attributes:
        - MODELOPT_CONFIGS: Dict of config name -> NVIDIAModelOptConfig kwargs to test

    Pytest mark: modelopt
        Use `pytest -m "not modelopt"` to skip these tests
    """

    MODELOPT_CONFIGS = {
        "fp8": {"quant_type": "FP8"},
        "int8": {"quant_type": "INT8"},
        "int4": {"quant_type": "INT4"},
    }

    MODELOPT_EXPECTED_MEMORY_REDUCTIONS = {
        "fp8": 1.5,
        "int8": 1.5,
        "int4": 3.0,
    }

    def _create_quantized_model(self, config_kwargs, **extra_kwargs):
        config = NVIDIAModelOptConfig(**config_kwargs)
        kwargs = getattr(self, "pretrained_model_kwargs", {}).copy()
        kwargs["quantization_config"] = config
        kwargs.update(extra_kwargs)
        return self.model_class.from_pretrained(self.pretrained_model_name_or_path, **kwargs)

    def _verify_if_layer_quantized(self, name, module, config_kwargs):
        assert mtq.utils.is_quantized(module), f"Layer {name} does not have weight_quantizer attribute (not quantized)"

    @pytest.mark.parametrize("config_name", ["fp8"])
    def test_modelopt_quantization_num_parameters(self, config_name):
        self._test_quantization_num_parameters(self.MODELOPT_CONFIGS[config_name])

    @pytest.mark.parametrize("config_name", list(MODELOPT_CONFIGS.keys()))
    def test_modelopt_quantization_memory_footprint(self, config_name):
        expected = self.MODELOPT_EXPECTED_MEMORY_REDUCTIONS.get(config_name, 1.2)
        self._test_quantization_memory_footprint(
            self.MODELOPT_CONFIGS[config_name], expected_memory_reduction=expected
        )

    @pytest.mark.parametrize("config_name", list(MODELOPT_CONFIGS.keys()))
    def test_modelopt_quantization_inference(self, config_name):
        self._test_quantization_inference(self.MODELOPT_CONFIGS[config_name])

    @pytest.mark.parametrize("config_name", ["fp8"])
    def test_modelopt_quantization_dtype_assignment(self, config_name):
        self._test_quantization_dtype_assignment(self.MODELOPT_CONFIGS[config_name])

    @pytest.mark.parametrize("config_name", ["fp8"])
    def test_modelopt_quantization_lora_inference(self, config_name):
        self._test_quantization_lora_inference(self.MODELOPT_CONFIGS[config_name])

    @pytest.mark.parametrize("config_name", ["fp8"])
    def test_modelopt_quantization_serialization(self, config_name):
        self._test_quantization_serialization(self.MODELOPT_CONFIGS[config_name])

    @pytest.mark.parametrize("config_name", ["fp8"])
    def test_modelopt_quantized_layers(self, config_name):
        self._test_quantized_layers(self.MODELOPT_CONFIGS[config_name])

    def test_modelopt_modules_to_not_convert(self):
        """Test that modules_to_not_convert parameter works correctly."""
        modules_to_exclude = getattr(self, "modules_to_not_convert_for_test", None)
        if modules_to_exclude is None:
            pytest.skip("modules_to_not_convert_for_test not defined for this model")

        self._test_quantization_modules_to_not_convert(self.MODELOPT_CONFIGS["fp8"], modules_to_exclude)

    def test_modelopt_device_map(self):
        """Test that device_map='auto' works correctly with quantization."""
        self._test_quantization_device_map(self.MODELOPT_CONFIGS["fp8"])
