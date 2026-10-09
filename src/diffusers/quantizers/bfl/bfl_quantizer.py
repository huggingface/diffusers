# Copyright 2026 The HuggingFace Team. All rights reserved.
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

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ...utils import get_module_from_name, is_accelerate_available, is_accelerate_version, is_torch_available
from ..base import DiffusersQuantizer


if TYPE_CHECKING:
    from ...models.modeling_utils import ModelMixin


if is_torch_available():
    import torch
    import torch.nn as nn

    from .utils import (
        BFLQuantizedParameter,
        FP8RLinear,
        NVFP4Linear,
        _dequantize_bfl_and_restore_linear,
        _dequantize_weight,
        _replace_with_bfl_linear,
    )


class BFLQuantizer(DiffusersQuantizer):
    use_keep_in_fp32_modules = True

    def __init__(self, quantization_config, **kwargs):
        super().__init__(quantization_config, **kwargs)
        self.compute_dtype = quantization_config.compute_dtype
        self.keep_in_fp32_modules = []

    def validate_environment(self, *args, **kwargs):
        if not is_accelerate_available() or is_accelerate_version("<", "0.26.0"):
            raise ImportError(
                "Loading BFL quantized checkpoints requires `accelerate` installed in your environment: `pip install 'accelerate>=0.26.0'`"
            )

    def adjust_target_dtype(self, target_dtype: "torch.dtype") -> "torch.dtype":
        return torch.uint8

    def update_torch_dtype(self, torch_dtype: "torch.dtype") -> "torch.dtype":
        if torch_dtype is None:
            torch_dtype = self.compute_dtype
        return torch_dtype

    def maybe_update_state_dict(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        prefixes = [key[: -len(".weight_scale")] for key in state_dict if key.endswith(".weight_scale")]
        if not prefixes:
            raise ValueError("The checkpoint has no `weight_scale` tensors, so it is not an fp8r or nvfp4 checkpoint.")
        for prefix in prefixes:
            weight_scale = state_dict.pop(f"{prefix}.weight_scale")
            weight_scale_2 = state_dict.pop(f"{prefix}.weight_scale_2", None)
            input_scale = state_dict.pop(f"{prefix}.input_scale", None)
            quant_type = "nvfp4" if weight_scale_2 is not None else "fp8r"
            weight = state_dict[f"{prefix}.weight"]
            expected_dtype = torch.uint8 if quant_type == "nvfp4" else torch.float8_e4m3fn
            if weight.dtype != expected_dtype:
                raise ValueError(
                    f"{prefix}.weight is {weight.dtype}, but the {quant_type} layout expects {expected_dtype}."
                )
            if quant_type == "nvfp4" and input_scale is None:
                raise ValueError(
                    f"{prefix} has `weight_scale_2` but no `input_scale`, which the nvfp4 layout requires."
                )
            state_dict[f"{prefix}.weight"] = BFLQuantizedParameter(
                weight, quant_type, weight_scale, weight_scale_2, input_scale
            )
        return state_dict

    def check_if_quantized_param(
        self,
        model: "ModelMixin",
        param_value: "torch.Tensor",
        param_name: str,
        state_dict: dict[str, Any],
        **kwargs,
    ) -> bool:
        return isinstance(param_value, BFLQuantizedParameter)

    def create_quantized_param(
        self,
        model: "ModelMixin",
        param_value: "BFLQuantizedParameter",
        param_name: str,
        target_device: "torch.device",
        state_dict: dict[str, Any] | None = None,
        unexpected_keys: list[str] | None = None,
        **kwargs,
    ):
        module, tensor_name = get_module_from_name(model, param_name)
        if isinstance(module, (FP8RLinear, NVFP4Linear)):
            module.weight = nn.Parameter(param_value.as_tensor().to(target_device), requires_grad=False)
            module.weight_scale = param_value.weight_scale.to(target_device)
            if isinstance(module, NVFP4Linear):
                module.weight_scale_2 = param_value.weight_scale_2.to(target_device)
                module.input_scale = param_value.input_scale.to(target_device)
            return

        # The module was excluded from quantization (for example by `_keep_in_fp32_modules`), so it stays a
        # regular `nn.Linear` and receives the dequantized weight.
        keep_in_fp32 = any(m in param_name.split(".") for m in self.keep_in_fp32_modules)
        dtype = torch.float32 if keep_in_fp32 else self.compute_dtype
        module._parameters[tensor_name] = nn.Parameter(
            _dequantize_weight(param_value).to(target_device, dtype), requires_grad=False
        )

    def _process_model_before_weight_loading(
        self,
        model: "ModelMixin",
        device_map,
        keep_in_fp32_modules: list[str] = [],
        **kwargs,
    ):
        self.keep_in_fp32_modules = [module for module in keep_in_fp32_modules if module is not None]
        _replace_with_bfl_linear(
            model, self.compute_dtype, kwargs.get("state_dict"), modules_to_not_convert=self.keep_in_fp32_modules
        )

    def _process_model_after_weight_loading(self, model: "ModelMixin", **kwargs):
        return model

    @property
    def is_serializable(self):
        return False

    @property
    def is_trainable(self) -> bool:
        return False

    @property
    def is_compileable(self) -> bool:
        return True

    def _dequantize(self, model):
        return _dequantize_bfl_and_restore_linear(model)
