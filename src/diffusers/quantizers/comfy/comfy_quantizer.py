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


if is_comfy_kitchen_available():
    from comfy_kitchen.tensor import get_layout_class

    from .utils import ComfyLinear, ComfyQuantizedTensor


COMFY_FORMATS = {
    "float8_e4m3fn": ("TensorCoreFP8Layout", torch.float8_e4m3fn, 8),
    "float8_e5m2": ("TensorCoreFP8Layout", torch.float8_e5m2, 8),
    "int8_tensorwise": ("TensorWiseINT8Layout", torch.int8, 8),
    "nvfp4": ("TensorCoreNVFP4Layout", torch.uint8, 4),
    "mxfp8": ("TensorCoreMXFP8Layout", torch.float8_e4m3fn, 8),
    "awq_w4a16": ("TensorCoreAWQW4A16Layout", torch.int8, 4),
    "svdquant_w4a4": ("TensorCoreSVDQuantW4A4Layout", torch.int8, 4),
    "convrot_w4a4": ("TensorCoreConvRotW4A4Layout", torch.int8, 4),
    "asym_w4a8_int8": ("AsymW4A8Int8Layout", torch.int8, 4),
    "w6a8_int8": ("AsymW4A8Int8Layout", torch.int8, 6),
}


class ComfyQuantizer(DiffusersQuantizer):
    use_keep_in_fp32_modules = True

    def __init__(self, quantization_config, **kwargs):
        quantization_config.post_init()
        super().__init__(quantization_config, **kwargs)
        self.compute_dtype = quantization_config.compute_dtype
        self.pre_quantized = True
        self.keep_in_fp32_modules = []

    def validate_environment(self, *args, **kwargs):
        if not is_accelerate_available():
            raise ImportError("Loading Comfy checkpoints requires accelerate: `pip install accelerate`.")
        if not is_comfy_kitchen_available():
            raise ImportError("Comfy inference requires comfy-kitchen: `pip install 'comfy-kitchen>=0.2.41'`.")
        if Version(importlib.metadata.version("comfy-kitchen")) < Version("0.2.41"):
            raise ImportError("Comfy inference requires comfy-kitchen>=0.2.41.")

    def update_torch_dtype(self, torch_dtype):
        if torch_dtype is not None and torch_dtype != self.compute_dtype:
            raise ValueError("`dtype` must match ComfyQuantizationConfig.compute_dtype.")
        return self.compute_dtype

    def maybe_update_state_dict(self, state_dict):
        marker_keys = [key for key in state_dict if key.endswith(".comfy_quant")]
        if not marker_keys:
            raise ValueError("The checkpoint has no native Comfy quantization markers.")
        for key in marker_keys:
            prefix = key.removesuffix(".comfy_quant")
            payload = state_dict.pop(key)
            if payload.dtype != torch.uint8 or payload.ndim != 1:
                raise ValueError(f"{key} must be a one-dimensional uint8 JSON marker.")
            try:
                marker = json.loads(bytes(payload.tolist()))
            except (ValueError, UnicodeDecodeError) as exc:
                raise ValueError(f"{key} is not valid JSON.") from exc
            if not isinstance(marker, dict) or marker.get("format") not in COMFY_FORMATS:
                raise ValueError(f"{key} must declare a supported Comfy format: {list(COMFY_FORMATS)}.")
            quant_format = marker["format"]
            layout_name, storage_dtype, bits = COMFY_FORMATS[quant_format]
            if f"{prefix}.weight" not in state_dict:
                raise ValueError(f"{prefix} requires weight.")
            weight = state_dict[f"{prefix}.weight"]
            if weight.dtype != storage_dtype or weight.ndim not in (2, 4):
                raise ValueError(f"{prefix}.weight has an invalid shape or dtype for {quant_format}.")
            if weight.ndim == 4:
                if quant_format != "svdquant_w4a4" or tuple(weight.shape[2:]) != (32, 128):
                    raise ValueError(f"{prefix}.weight has an invalid tile-packed layout.")
                shape = (weight.shape[0] * 128, weight.shape[1] * 64)
            else:
                if weight.shape[1] * 8 % bits:
                    raise ValueError(f"{prefix}.weight has an invalid packed input width.")
                shape = (weight.shape[0], weight.shape[1] * 8 // bits)

            def pop_tensor(suffix, dtype=None):
                name = f"{prefix}.{suffix}"
                if name not in state_dict:
                    raise ValueError(f"{prefix} requires {suffix}.")
                value = state_dict.pop(name)
                return value.view(dtype) if dtype is not None else value

            if not isinstance(marker.get("params", {}), dict):
                raise ValueError(f"{key}.params must be an object.")
            options = {**marker.get("params", {}), **marker}
            params = {"orig_dtype": self.compute_dtype, "orig_shape": shape}
            if quant_format == "nvfp4":
                params.update(
                    scale=pop_tensor("weight_scale_2"), block_scale=pop_tensor("weight_scale", torch.float8_e4m3fn)
                )
            elif quant_format == "mxfp8":
                params["scale"] = pop_tensor("weight_scale", torch.float8_e8m0fnu)
            elif quant_format in ("asym_w4a8_int8", "w6a8_int8"):
                params.update(
                    scale=pop_tensor("weight_s_rel", torch.float8_e4m3fn),
                    s_channel=pop_tensor("weight_s_channel"),
                    codebook=state_dict.pop(f"{prefix}.weight_codebook", None),
                    correction=state_dict.pop(f"{prefix}.weight_correction", None),
                    group_size=options.get("group_size", 16),
                    convrot_groupsize=options.get("convrot_groupsize", 256),
                )
            else:
                params["scale"] = pop_tensor("weight_scale")
            if quant_format == "int8_tensorwise":
                scale = params["scale"]
                rows, columns = shape
                if scale.dtype != torch.float32 or tuple(scale.shape) not in ((), (1,), (rows,), (rows, 1)):
                    raise ValueError(f"{prefix}.weight_scale must be FP32 with a scalar or per-output-row shape.")
                if not torch.isfinite(scale).all() or not (scale > 0).all():
                    raise ValueError(f"{prefix}.weight_scale must contain finite positive values.")
                convrot = options.get("convrot", False)
                group_size = options.get("convrot_groupsize", 256)
                if not isinstance(convrot, bool):
                    raise ValueError(f"{prefix}.convrot must be a boolean.")
                if convrot:
                    if (
                        type(group_size) is not int
                        or group_size < 4
                        or group_size & (group_size - 1)
                        or (group_size.bit_length() - 1) % 2
                    ):
                        raise ValueError(f"{prefix}.convrot_groupsize must be a power of four, at least 4.")
                    if columns % group_size or scale.numel() != rows:
                        raise ValueError(f"{prefix} requires group-aligned input features and per-output-row scales.")
                params.update(convrot=convrot, convrot_groupsize=group_size)
                if scale.numel() > 1:
                    params["scale"] = scale.reshape(rows, 1)
            elif quant_format == "awq_w4a16":
                params.update(zeros=pop_tensor("weight_zero"), group_size=options.get("group_size", 64))
            elif quant_format == "svdquant_w4a4":
                params.update(
                    proj_down=pop_tensor("proj_down"),
                    proj_up=pop_tensor("proj_up"),
                    smooth_factor=pop_tensor("smooth_factor"),
                    act_unsigned=options.get("act_unsigned", False),
                )
            elif quant_format == "convrot_w4a4":
                params.update(
                    convrot_groupsize=options.get("convrot_groupsize", 256),
                    linear_dtype=options.get("linear_dtype", "int4"),
                )
            sidecars = {
                name: state_dict.pop(f"{prefix}.{name}")
                for name in ("input_scale", "pre_quant_scale")
                if f"{prefix}.{name}" in state_dict
            }
            layout = get_layout_class(layout_name)
            state_dict[f"{prefix}.weight"] = ComfyQuantizedTensor(
                weight, layout_name, layout.Params(**params), marker, sidecars
            )
        unmarked = [
            key
            for key, value in state_dict.items()
            if key.endswith(".weight") and value.dtype in {value[1] for value in COMFY_FORMATS.values()}
        ]
        if unmarked:
            raise ValueError(f"Quantized weights require native Comfy quantization markers: {unmarked}.")
        return state_dict

    def _process_model_before_weight_loading(
        self, model, device_map, state_dict=None, keep_in_fp32_modules=(), **kwargs
    ):
        if state_dict is None:
            raise ValueError("Comfy checkpoints must be loaded with from_single_file().")
        self.keep_in_fp32_modules = [name for name in keep_in_fp32_modules if name is not None]
        for prefix, module in list(model.named_modules()):
            weight = state_dict.get(f"{prefix}.weight")
            if not isinstance(weight, ComfyQuantizedTensor):
                continue
            if not isinstance(module, nn.Linear) or tuple(weight.shape) != (module.out_features, module.in_features):
                raise ValueError(f"{prefix} does not match a Linear layer with shape {tuple(weight.shape)}.")
            if any(name in prefix.split(".") for name in self.keep_in_fp32_modules):
                continue
            parent, name = get_module_from_name(model, prefix)
            linear = ComfyLinear(
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
                device="meta",
                dtype=self.compute_dtype,
            )
            linear.bias = module.bias
            setattr(parent, name, linear)

    def check_if_quantized_param(self, model, param_value, param_name, state_dict, **kwargs):
        return isinstance(param_value, ComfyQuantizedTensor)

    def create_quantized_param(
        self, model, param_value, param_name, target_device, state_dict, unexpected_keys=None, **kwargs
    ):
        module, name = get_module_from_name(model, param_name)
        if isinstance(module, ComfyLinear):
            value = param_value.to(target_device)
        else:
            value = param_value.to(torch.float32).dequantize().to(target_device)
        module._parameters[name] = nn.Parameter(value, requires_grad=False)

    def _process_model_after_weight_loading(self, model, **kwargs):
        missing = [name for name, tensor in model.state_dict().items() if tensor.is_meta]
        if missing:
            raise ValueError(f"The Comfy checkpoint is missing model tensors: {missing}.")
        return model

    @property
    def is_serializable(self):
        return False

    @property
    def is_trainable(self):
        return False
