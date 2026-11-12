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

import importlib.metadata
import json

import torch
from packaging.version import Version
from torch import nn

from ...utils import get_module_from_name, is_accelerate_available, is_comfy_kitchen_available
from ..base import DiffusersQuantizer
from .utils import ConvRotInt8Linear


class ConvRotQuantizer(DiffusersQuantizer):
    use_keep_in_fp32_modules = True

    def __init__(self, quantization_config, **kwargs):
        super().__init__(quantization_config, **kwargs)
        self.compute_dtype = quantization_config.compute_dtype

    def validate_environment(self, *args, **kwargs):
        if not is_accelerate_available():
            raise ImportError("Loading ConvRot checkpoints requires accelerate: `pip install accelerate`.")
        if not is_comfy_kitchen_available():
            raise ImportError("ConvRot inference requires comfy-kitchen: `pip install 'comfy-kitchen>=0.2.41'`.")
        if Version(importlib.metadata.version("comfy-kitchen")) < Version("0.2.41"):
            raise ImportError("ConvRot inference requires comfy-kitchen>=0.2.41.")

    def update_torch_dtype(self, torch_dtype):
        if torch_dtype is not None and torch_dtype != self.compute_dtype:
            raise ValueError("`dtype` must match ConvRotQuantizationConfig.compute_dtype.")
        return self.compute_dtype

    def _process_model_before_weight_loading(
        self, model, device_map, state_dict=None, keep_in_fp32_modules=(), **kwargs
    ):
        if state_dict is None:
            raise ValueError("ConvRot checkpoints must be loaded with from_single_file().")
        marker_keys = [key for key in state_dict if key.endswith(".comfy_quant")]
        if not marker_keys:
            raise ValueError("The checkpoint has no native Comfy INT8 quantization markers.")
        unmarked = [
            key
            for key, value in state_dict.items()
            if key.endswith(".weight")
            and value.dtype == torch.int8
            and f"{key[: -len('.weight')]}.comfy_quant" not in state_dict
        ]
        if unmarked:
            raise ValueError(f"INT8 weights require native Comfy quantization markers: {unmarked}.")
        replacements = []
        for key in marker_keys:
            prefix = key[: -len(".comfy_quant")]
            payload = state_dict[key]
            if payload.dtype != torch.uint8 or payload.ndim != 1:
                raise ValueError(f"{key} must be a one-dimensional uint8 JSON marker.")
            try:
                marker = json.loads(bytes(payload.tolist()))
            except (ValueError, UnicodeDecodeError) as exc:
                raise ValueError(f"{key} is not valid JSON.") from exc
            if not isinstance(marker, dict) or marker.get("format") != "int8_tensorwise":
                raise ValueError(f"{key} must declare format='int8_tensorwise'.")
            if marker.get("full_precision_matrix_mult", False):
                raise ValueError(f"{prefix} requests full_precision_matrix_mult, which is not supported.")
            if not isinstance(marker.get("convrot", False), bool):
                raise ValueError(f"{prefix}.convrot must be a boolean.")
            weight = state_dict.get(f"{prefix}.weight")
            scale = state_dict.get(f"{prefix}.weight_scale")
            if weight is None or scale is None:
                raise ValueError(f"{prefix} requires both weight and weight_scale tensors.")
            if weight.dtype != torch.int8 or weight.ndim != 2:
                raise ValueError(f"{prefix}.weight must be a two-dimensional int8 tensor.")
            rows, columns = weight.shape
            if scale.dtype != torch.float32 or scale.shape not in (
                torch.Size([]),
                torch.Size([1]),
                torch.Size([rows]),
                torch.Size([rows, 1]),
            ):
                raise ValueError(f"{prefix}.weight_scale must be FP32 with a scalar or per-output-row shape.")
            if not torch.isfinite(scale).all() or not (scale > 0).all():
                raise ValueError(f"{prefix}.weight_scale must contain finite positive values.")
            if marker.get("convrot", False):
                group_size = marker.get("convrot_groupsize", 256)
                if (
                    type(group_size) is not int
                    or group_size < 4
                    or group_size & (group_size - 1)
                    or (group_size.bit_length() - 1) % 2
                ):
                    raise ValueError(f"{prefix}.convrot_groupsize must be a power of four, at least 4.")
                if columns % group_size or scale.numel() != rows:
                    raise ValueError(f"{prefix} requires group-aligned input features and per-output-row scales.")
            module, _ = get_module_from_name(model, f"{prefix}.weight")
            if not isinstance(module, nn.Linear) or tuple(weight.shape) != (module.out_features, module.in_features):
                raise ValueError(f"{prefix} does not match a Linear layer with shape {tuple(weight.shape)}.")
            if any(name in prefix.split(".") for name in keep_in_fp32_modules):
                raise ValueError(f"{prefix} is quantized but must be kept in FP32 by the model.")
            replacements.append((prefix, module, scale.shape, marker, key))

        for prefix, module, scale_shape, marker, key in replacements:
            parent, name = get_module_from_name(model, prefix)
            setattr(parent, name, ConvRotInt8Linear(module, scale_shape, marker, self.compute_dtype))
            state_dict.pop(key)

    def check_if_quantized_param(self, model, param_value, param_name, state_dict, **kwargs):
        module, name = get_module_from_name(model, param_name)
        return isinstance(module, ConvRotInt8Linear) and name in ("weight", "weight_scale")

    def create_quantized_param(
        self, model, param_value, param_name, target_device, state_dict, unexpected_keys=None, **kwargs
    ):
        module, name = get_module_from_name(model, param_name)
        if name == "weight":
            module.weight = nn.Parameter(param_value.to(target_device, torch.int8), requires_grad=False)
        else:
            module.weight_scale = state_dict[param_name].to(target_device, torch.float32)

    def _process_model_after_weight_loading(self, model, **kwargs):
        missing = [name for name, tensor in model.state_dict().items() if tensor.is_meta]
        if missing:
            raise ValueError(f"The ConvRot checkpoint is missing model tensors: {missing}.")
        return model

    @property
    def is_serializable(self):
        return False

    @property
    def is_trainable(self):
        return False
