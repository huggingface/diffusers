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

from ...dependency_versions_table import deps
from ...utils import (
    get_module_from_name,
    is_entropack_available,
    is_torch_available,
    logging,
)
from ..base import DiffusersQuantizer


if TYPE_CHECKING:
    from ...models.modeling_utils import ModelMixin


if is_torch_available():
    import torch
    import torch.nn as nn


logger = logging.get_logger(__name__)


class EntroPackQuantizer(DiffusersQuantizer):
    """Diffusers quantizer for EntroPack weight compression: https://github.com/modelscope/entropack."""

    requires_calibration = False
    required_packages = ["entropack"]
    use_keep_in_fp32_modules = True

    def __init__(self, quantization_config, **kwargs):
        super().__init__(quantization_config, **kwargs)
        self.modules_to_not_convert = list(quantization_config.modules_to_not_convert or [])

    def validate_environment(self, *args, **kwargs):
        if not is_entropack_available():
            raise ImportError(
                f"Loading an EntroPack quantized model requires `{deps['entropack']}`. Install it with "
                "`pip install -U 'entropack[cuda12]'` (or `[cuda13]` for CUDA 13)."
            )

        from entropack.backends.cuda import probe

        reason = probe()
        if reason is not None:
            raise RuntimeError(f"The entropack CUDA codec is unavailable: {reason}")

        device_map = kwargs.get("device_map")
        if isinstance(device_map, dict) and "disk" in device_map.values():
            raise ValueError("EntroPack does not support disk offloading; remove the `disk` device from `device_map`.")

    def update_torch_dtype(self, torch_dtype: "torch.dtype | None") -> "torch.dtype":
        if torch_dtype is None:
            logger.info("You did not specify `torch_dtype`; setting it to `torch.bfloat16` for EntroPack.")
            torch_dtype = torch.bfloat16
        return torch_dtype

    def adjust_target_dtype(self, target_dtype: "torch.dtype") -> "torch.dtype":
        return torch.uint8

    def adjust_max_memory(self, max_memory: dict[str, int | str]) -> dict[str, int | str]:
        return {key: value * 0.9 for key, value in max_memory.items()}

    def maybe_update_loaded_keys(self, loaded_keys: list[str], checkpoint_files: list[str]) -> list[str]:
        if not self.pre_quantized:
            return loaded_keys
        suffix = ".entropack_quant_state"
        return [key[: -len(suffix)] if key.endswith(suffix) else key for key in loaded_keys]

    def maybe_update_state_dict(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        if not self.pre_quantized:
            return state_dict
        import entropack

        quant_suffix = ".weight.entropack_quant_state"
        converted = {}
        for key, value in state_dict.items():
            if key.endswith(quant_suffix):
                key = key[: -len(quant_suffix)] + ".weight"
                value = entropack.CompressedTensor.from_state_dict({"entropack_quant_state": value})
            converted[key] = value
        return converted

    def check_if_quantized_param(
        self,
        model: "ModelMixin",
        param_value: "torch.Tensor",
        param_name: str,
        state_dict: dict[str, Any],
        **kwargs,
    ) -> bool:
        import entropack

        module, tensor_name = get_module_from_name(model, param_name)
        if not isinstance(module, entropack.CompressedLinear):
            return False

        if self.pre_quantized:
            if tensor_name == "weight":
                return isinstance(param_value, entropack.CompressedTensor)
            # scales are quantizer-loaded so they keep the float32 dtype stored in the checkpoint
            return tensor_name in module.state_buffer_names

        return tensor_name == "weight"

    def check_quantized_param_shape(self, param_name, current_param, loaded_param):
        current_shape = tuple(current_param.shape)
        loaded_shape = tuple(loaded_param.shape)
        if current_shape != loaded_shape:
            raise ValueError(f"{param_name} has an expected shape of {current_shape}, but got {loaded_shape}.")
        return True

    def create_quantized_param(
        self,
        model: "ModelMixin",
        param_value: "torch.Tensor",
        param_name: str,
        target_device: "torch.device",
        state_dict: dict[str, Any],
        unexpected_keys: list[str] | None = None,
        **kwargs,
    ):
        module, tensor_name = get_module_from_name(model, param_name)
        if tensor_name in module.state_buffer_names:
            # param_value has been converted to the model dtype; state_dict still holds the
            # original float32 scale required by the layer.
            setattr(module, tensor_name, state_dict[param_name].to(device=target_device, dtype=torch.float32))
            return

        if self.pre_quantized:
            module.weight = nn.Parameter(param_value.to(target_device), requires_grad=False)
        else:
            module.compress_weight(param_value.detach().to(target_device))

    def update_missing_keys(self, model, missing_keys: list[str], prefix: str) -> list[str]:
        if self.pre_quantized:
            return missing_keys
        import entropack

        kept = []
        for key in missing_keys:
            module, tensor_name = get_module_from_name(model, key[len(prefix) :])
            if not (isinstance(module, entropack.CompressedLinear) and tensor_name in module.state_buffer_names):
                kept.append(key)
        return kept

    def _process_model_before_weight_loading(
        self,
        model: "ModelMixin",
        keep_in_fp32_modules: list[str] = [],
        **kwargs,
    ):
        self.modules_to_not_convert.extend(module for module in keep_in_fp32_modules if module)
        from .utils import _replace_with_entropack_linear

        _replace_with_entropack_linear(model, self.quantization_config, self.modules_to_not_convert)
        model.config.quantization_config = self.quantization_config
        return model

    def _process_model_after_weight_loading(self, model: "ModelMixin", **kwargs):
        import entropack

        for name, module in model.named_modules():
            if not isinstance(module, entropack.CompressedLinear):
                continue
            if not isinstance(module.weight, entropack.CompressedTensor):
                raise RuntimeError(f"EntroPack layer {name} has no packed weight")
            for buffer_name in module.state_buffer_names:
                if getattr(module, buffer_name).device.type == "meta":
                    raise RuntimeError(f"EntroPack layer {name} did not load {buffer_name}")
        return model

    @property
    def is_serializable(self):
        return True

    @property
    def is_trainable(self) -> bool:
        return True

    def _dequantize(self, model):
        import entropack

        return entropack.dequantize_model(model)
