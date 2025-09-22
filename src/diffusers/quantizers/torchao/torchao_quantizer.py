# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""
Adapted from
https://github.com/huggingface/transformers/blob/3a8eb74668e9c2cc563b2f5c62fac174797063e0/src/transformers/quantizers/quantizer_torchao.py
"""

import importlib
import re
import types
from fnmatch import fnmatch
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from packaging import version

from ...utils import (
    get_module_from_name,
    is_torch_available,
    is_torch_version,
    is_torchao_available,
    is_torchao_version,
    logging,
)
from ..base import DiffusersQuantizer


if TYPE_CHECKING:
    from ...models.modeling_utils import ModelMixin


if is_torch_available():
    import torch
    import torch.nn as nn

    if is_torch_version(">=", "2.5"):
        SUPPORTED_TORCH_DTYPES_FOR_QUANTIZATION = (
            # At the moment, only int8 is supported for integer quantization dtypes.
            # In Torch 2.6, int1-int7 will be introduced, so this can be visited in the future
            # to support more quantization methods, such as intx_weight_only.
            torch.int8,
            torch.float8_e4m3fn,
            torch.float8_e5m2,
            torch.uint1,
            torch.uint2,
            torch.uint3,
            torch.uint4,
            torch.uint5,
            torch.uint6,
            torch.uint7,
        )
    else:
        SUPPORTED_TORCH_DTYPES_FOR_QUANTIZATION = (
            torch.int8,
            torch.float8_e4m3fn,
            torch.float8_e5m2,
        )

if is_torchao_available():
    from torchao.quantization import quantize_


def _update_torch_safe_globals():
    safe_globals = [
        (torch.uint1, "torch.uint1"),
        (torch.uint2, "torch.uint2"),
        (torch.uint3, "torch.uint3"),
        (torch.uint4, "torch.uint4"),
        (torch.uint5, "torch.uint5"),
        (torch.uint6, "torch.uint6"),
        (torch.uint7, "torch.uint7"),
    ]
    try:
        from torchao.dtypes import NF4Tensor
        from torchao.dtypes.floatx.float8_layout import Float8AQTTensorImpl
        from torchao.dtypes.uintx.uint4_layout import UInt4Tensor
        from torchao.dtypes.uintx.uintx_layout import UintxAQTTensorImpl, UintxTensor

        safe_globals.extend([UintxTensor, UInt4Tensor, UintxAQTTensorImpl, Float8AQTTensorImpl, NF4Tensor])

    except (ImportError, ModuleNotFoundError) as e:
        logger.warning(
            "Unable to import `torchao` Tensor objects. This may affect loading checkpoints serialized with `torchao`"
        )
        logger.debug(e)

    finally:
        torch.serialization.add_safe_globals(safe_globals=safe_globals)


if (
    is_torch_available()
    and is_torch_version(">=", "2.6.0")
    and is_torchao_available()
    and is_torchao_version(">=", "0.7.0")
):
    _update_torch_safe_globals()


def fuzzy_match_size(config_name: str) -> Optional[str]:
    """
    Extract the size digit from strings like "4weight", "8weight". Returns the digit as an integer if found, otherwise
    None.
    """
    config_name = config_name.lower()

    str_match = re.search(r"(\d)weight", config_name)

    if str_match:
        return str_match.group(1)

    return None


logger = logging.get_logger(__name__)


def _quantization_type(weight):
    from torchao.dtypes import AffineQuantizedTensor
    from torchao.quantization.linear_activation_quantized_tensor import LinearActivationQuantizedTensor

    if isinstance(weight, AffineQuantizedTensor):
        return f"{weight.__class__.__name__}({weight._quantization_type()})"

    if isinstance(weight, LinearActivationQuantizedTensor):
        return f"{weight.__class__.__name__}(activation={weight.input_quant_func}, weight={_quantization_type(weight.original_weight_tensor)})"


def _linear_extra_repr(self):
    weight = _quantization_type(self.weight)
    if weight is None:
        return f"in_features={self.weight.shape[1]}, out_features={self.weight.shape[0]}, weight=None"
    else:
        return f"in_features={self.weight.shape[1]}, out_features={self.weight.shape[0]}, weight={weight}"


class TorchAoHfQuantizer(DiffusersQuantizer):
    r"""
    Diffusers Quantizer for TorchAO: https://github.com/pytorch/ao/.
    """

    requires_calibration = False
    required_packages = ["torchao"]

    def __init__(self, quantization_config, **kwargs):
        super().__init__(quantization_config, **kwargs)

    def validate_environment(self, *args, **kwargs):
        if not is_torchao_available():
            raise ImportError(
                "Loading a TorchAO quantized model requires the torchao library. Please install with `pip install torchao`"
            )
        torchao_version = version.parse(importlib.metadata.version("torch"))
        if torchao_version < version.parse("0.7.0"):
            raise RuntimeError(
                f"The minimum required version of `torchao` is 0.7.0, but the current version is {torchao_version}. Please upgrade with `pip install -U torchao`."
            )

        self.offload = False

        device_map = kwargs.get("device_map", None)
        if isinstance(device_map, dict):
            if "cpu" in device_map.values() or "disk" in device_map.values():
                if self.pre_quantized:
                    raise ValueError(
                        "You are attempting to perform cpu/disk offload with a pre-quantized torchao model "
                        "This is not supported yet. Please remove the CPU or disk device from the `device_map` argument."
                    )
                else:
                    self.offload = True

        if self.pre_quantized:
            weights_only = kwargs.get("weights_only", None)
            if weights_only:
                torch_version = version.parse(importlib.metadata.version("torch"))
                if torch_version < version.parse("2.5.0"):
                    # TODO(aryan): TorchAO is compatible with Pytorch >= 2.2 for certain quantization types. Try to see if we can support it in future
                    raise RuntimeError(
                        f"In order to use TorchAO pre-quantized model, you need to have torch>=2.5.0. However, the current version is {torch_version}."
                    )

    def update_torch_dtype(self, torch_dtype):
        quant_type = self.quantization_config.quant_type
        if isinstance(quant_type, str) and (quant_type.startswith("int") or quant_type.startswith("uint")):
            if torch_dtype is not None and torch_dtype != torch.bfloat16:
                logger.warning(
                    f"You are trying to set torch_dtype to {torch_dtype} for int4/int8/uintx quantization, but "
                    f"only bfloat16 is supported right now. Please set `torch_dtype=torch.bfloat16`."
                )

        if torch_dtype is None:
            # We need to set the torch_dtype, otherwise we have dtype mismatch when performing the quantized linear op
            logger.warning(
                "Overriding `torch_dtype` with `torch_dtype=torch.bfloat16` due to requirements of `torchao` "
                "to enable model loading in different precisions. Pass your own `torch_dtype` to specify the "
                "dtype of the remaining non-linear layers, or pass torch_dtype=torch.bfloat16, to remove this warning."
            )
            torch_dtype = torch.bfloat16

        return torch_dtype

    def adjust_target_dtype(self, target_dtype: "torch.dtype") -> "torch.dtype":
        quant_type = self.quantization_config.quant_type
        from accelerate.utils import CustomDtype

        if isinstance(quant_type, str):
            if quant_type.startswith("int8"):
                # Note that int4 weights are created by packing into torch.int8, but since there is no torch.int4, we use torch.int8
                return torch.int8
            elif quant_type.startswith("int4"):
                return CustomDtype.INT4
            elif quant_type == "uintx_weight_only":
                return self.quantization_config.quant_type_kwargs.get("dtype", torch.uint8)
            elif quant_type.startswith("uint"):
                return {
                    1: torch.uint1,
                    2: torch.uint2,
                    3: torch.uint3,
                    4: torch.uint4,
                    5: torch.uint5,
                    6: torch.uint6,
                    7: torch.uint7,
                }[int(quant_type[4])]
            elif quant_type.startswith("float") or quant_type.startswith("fp"):
                return torch.bfloat16

        elif is_torchao_version(">", "0.9.0"):
            from torchao.core.config import AOBaseConfig

            quant_type = self.quantization_config.quant_type
            if isinstance(quant_type, AOBaseConfig):
                # Extract size digit using fuzzy match on the class name
                config_name = quant_type.__class__.__name__
                size_digit = fuzzy_match_size(config_name)

                # Map the extracted digit to appropriate dtype
                if size_digit == "4":
                    return CustomDtype.INT4
                else:
                    # Default to int8
                    return torch.int8

        if isinstance(target_dtype, SUPPORTED_TORCH_DTYPES_FOR_QUANTIZATION):
            return target_dtype

        # We need one of the supported dtypes to be selected in order for accelerate to determine
        # the total size of modules/parameters for auto device placement.
        possible_device_maps = ["auto", "balanced", "balanced_low_0", "sequential"]
        raise ValueError(
            f"You have set `device_map` as one of {possible_device_maps} on a TorchAO quantized model but a suitable target dtype "
            f"could not be inferred. The supported target_dtypes are: {SUPPORTED_TORCH_DTYPES_FOR_QUANTIZATION}. If you think the "
            f"dtype you are using should be supported, please open an issue at https://github.com/huggingface/diffusers/issues."
        )

    def adjust_max_memory(self, max_memory: Dict[str, Union[int, str]]) -> Dict[str, Union[int, str]]:
        max_memory = {key: val * 0.9 for key, val in max_memory.items()}
        return max_memory

    def check_if_quantized_param(
        self,
        model: "ModelMixin",
        param_value: "torch.Tensor",
        param_name: str,
        state_dict: Dict[str, Any],
        **kwargs,
    ) -> bool:
        param_device = kwargs.pop("param_device", None)
        # Check if the param_name is not in self.modules_to_not_convert
        if any((key + "." in param_name) or (key == param_name) for key in self.modules_to_not_convert):
            return False
        elif param_device == "cpu" and self.offload:
            # We don't quantize weights that we offload
            return False
        else:
            # We only quantize the weight of nn.Linear
            module, tensor_name = get_module_from_name(model, param_name)
            return isinstance(module, torch.nn.Linear) and (tensor_name == "weight")

    def create_quantized_param(
        self,
        model: "ModelMixin",
        param_value: "torch.Tensor",
        param_name: str,
        target_device: "torch.device",
        state_dict: Dict[str, Any],
        unexpected_keys: List[str],
        **kwargs,
    ):
        r"""
        Each nn.Linear layer that needs to be quantized is processed here. First, we set the value the weight tensor,
        then we move it to the target device. Finally, we quantize the module.
        """
        module, tensor_name = get_module_from_name(model, param_name)

        if self.pre_quantized:
            # If we're loading pre-quantized weights, replace the repr of linear layers for pretty printing info
            # about AffineQuantizedTensor
            module._parameters[tensor_name] = torch.nn.Parameter(param_value.to(device=target_device))
            if isinstance(module, nn.Linear):
                module.extra_repr = types.MethodType(_linear_extra_repr, module)
        else:
            # As we perform quantization here, the repr of linear layers is that of AQT, so we don't have to do it ourselves
            module._parameters[tensor_name] = torch.nn.Parameter(param_value).to(device=target_device)
            quantize_(module, self.quantization_config.get_apply_tensor_subclass())

    def get_cuda_warm_up_factor(self):
        """
        This factor is used in caching_allocator_warmup to determine how many bytes to pre-allocate for CUDA warmup.
        - A factor of 2 means we pre-allocate the full memory footprint of the model.
        - A factor of 4 means we pre-allocate half of that, and so on

        However, when using TorchAO, calculating memory usage with param.numel() * param.element_size() doesn't give
        the correct size for quantized weights (like int4 or int8) That's because TorchAO internally represents
        quantized tensors using subtensors and metadata, and the reported element_size() still corresponds to the
        torch_dtype not the actual bit-width of the quantized data.

        To correct for this:
        - Use a division factor of 8 for int4 weights
        - Use a division factor of 4 for int8 weights
        """
        # Original mapping for non-AOBaseConfig types
        # For the uint types, this is a best guess. Once these types become more used
        # we can look into their nuances.
        if is_torchao_version(">", "0.9.0"):
            from torchao.core.config import AOBaseConfig

            quant_type = self.quantization_config.quant_type
            # For autoquant case, it will be treated in the string implementation below in map_to_target_dtype
            if isinstance(quant_type, AOBaseConfig):
                # Extract size digit using fuzzy match on the class name
                config_name = quant_type.__class__.__name__
                size_digit = fuzzy_match_size(config_name)

                if size_digit == "4":
                    return 8
                else:
                    return 4

        map_to_target_dtype = {"int4_*": 8, "int8_*": 4, "uint*": 8, "float8*": 4}
        quant_type = self.quantization_config.quant_type
        for pattern, target_dtype in map_to_target_dtype.items():
            if fnmatch(quant_type, pattern):
                return target_dtype
        raise ValueError(f"Unsupported quant_type: {quant_type!r}")

    def _process_model_before_weight_loading(
        self,
        model: "ModelMixin",
        device_map,
        keep_in_fp32_modules: List[str] = [],
        **kwargs,
    ):
        self.modules_to_not_convert = self.quantization_config.modules_to_not_convert

        if not isinstance(self.modules_to_not_convert, list):
            self.modules_to_not_convert = [self.modules_to_not_convert]

        self.modules_to_not_convert.extend(keep_in_fp32_modules)

        # Extend `self.modules_to_not_convert` to keys that are supposed to be offloaded to `cpu` or `disk`
        if isinstance(device_map, dict) and len(device_map.keys()) > 1:
            keys_on_cpu = [key for key, value in device_map.items() if value in ["disk", "cpu"]]
            self.modules_to_not_convert.extend(keys_on_cpu)

        # Purge `None`.
        # Unlike `transformers`, we don't know if we should always keep certain modules in FP32
        # in case of diffusion transformer models. For language models and others alike, `lm_head`
        # and tied modules are usually kept in FP32.
        self.modules_to_not_convert = [module for module in self.modules_to_not_convert if module is not None]

        model.config.quantization_config = self.quantization_config

    def _process_model_after_weight_loading(self, model: "ModelMixin"):
        return model

    def is_serializable(self, safe_serialization=None):
        # TODO(aryan): needs to be tested
        if safe_serialization:
            logger.warning(
                "torchao quantized model does not support safe serialization, please set `safe_serialization` to False."
            )
            return False

        _is_torchao_serializable = version.parse(importlib.metadata.version("huggingface_hub")) >= version.parse(
            "0.25.0"
        )

        if not _is_torchao_serializable:
            logger.warning("torchao quantized model is only serializable after huggingface_hub >= 0.25.0 ")

        if self.offload and self.quantization_config.modules_to_not_convert is None:
            logger.warning(
                "The model contains offloaded modules and these modules are not quantized. We don't recommend saving the model as we won't be able to reload them."
                "If you want to specify modules to not quantize, please specify modules_to_not_convert in the quantization_config."
            )
            return False

        return _is_torchao_serializable

    @property
    def is_trainable(self):
        return self.quantization_config.quant_type.startswith("int8")

    @property
    def is_compileable(self) -> bool:
        return True
