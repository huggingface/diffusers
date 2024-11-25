# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
https://github.com/huggingface/transformers/blob/52cb4034ada381fe1ffe8d428a1076e5411a8026/src/transformers/quantizers/base.py
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from ..utils import is_torch_available
from .quantization_config import QuantizationConfigMixin


if TYPE_CHECKING:
    from ..models.modeling_utils import ModelMixin

if is_torch_available():
    import torch


class DiffusersQuantizer(ABC):
    """
    Abstract class of the HuggingFace quantizer. Supports for now quantizing HF diffusers models for inference and/or
    quantization. This class is used only for diffusers.models.modeling_utils.ModelMixin.from_pretrained and cannot be
    easily used outside the scope of that method yet.

    Attributes
        quantization_config (`diffusers.quantizers.quantization_config.QuantizationConfigMixin`):
            The quantization config that defines the quantization parameters of your model that you want to quantize.
        modules_to_not_convert (`List[str]`, *optional*):
            The list of module names to not convert when quantizing the model.
        required_packages (`List[str]`, *optional*):
            The list of required pip packages to install prior to using the quantizer
        requires_calibration (`bool`):
            Whether the quantization method requires to calibrate the model before using it.
    """

    requires_calibration = False
    required_packages = None

    def __init__(self, quantization_config: QuantizationConfigMixin, **kwargs):
        self.quantization_config = quantization_config

        # -- Handle extra kwargs below --
        self.modules_to_not_convert = kwargs.pop("modules_to_not_convert", [])
        self.pre_quantized = kwargs.pop("pre_quantized", True)

        if not self.pre_quantized and self.requires_calibration:
            raise ValueError(
                f"The quantization method {quantization_config.quant_method} does require the model to be pre-quantized."
                f" You explicitly passed `pre_quantized=False` meaning your model weights are not quantized. Make sure to "
                f"pass `pre_quantized=True` while knowing what you are doing."
            )

    def update_torch_dtype(self, torch_dtype: "torch.dtype") -> "torch.dtype":
        """
        Some quantization methods require to explicitly set the dtype of the model to a target dtype. You need to
        override this method in case you want to make sure that behavior is preserved

        Args:
            torch_dtype (`torch.dtype`):
                The input dtype that is passed in `from_pretrained`
        """
        return torch_dtype

    def update_device_map(self, device_map: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Override this method if you want to pass a override the existing device map with a new one. E.g. for
        bitsandbytes, since `accelerate` is a hard requirement, if no device_map is passed, the device_map is set to
        `"auto"``

        Args:
            device_map (`Union[dict, str]`, *optional*):
                The device_map that is passed through the `from_pretrained` method.
        """
        return device_map

    def adjust_target_dtype(self, torch_dtype: "torch.dtype") -> "torch.dtype":
        """
        Override this method if you want to adjust the `target_dtype` variable used in `from_pretrained` to compute the
        device_map in case the device_map is a `str`. E.g. for bitsandbytes we force-set `target_dtype` to `torch.int8`
        and for 4-bit we pass a custom enum `accelerate.CustomDtype.int4`.

        Args:
            torch_dtype (`torch.dtype`, *optional*):
                The torch_dtype that is used to compute the device_map.
        """
        return torch_dtype

    def update_missing_keys(self, model, missing_keys: List[str], prefix: str) -> List[str]:
        """
        Override this method if you want to adjust the `missing_keys`.

        Args:
            missing_keys (`List[str]`, *optional*):
                The list of missing keys in the checkpoint compared to the state dict of the model
        """
        return missing_keys

    def get_special_dtypes_update(self, model, torch_dtype: "torch.dtype") -> Dict[str, "torch.dtype"]:
        """
        returns dtypes for modules that are not quantized - used for the computation of the device_map in case one
        passes a str as a device_map. The method will use the `modules_to_not_convert` that is modified in
        `_process_model_before_weight_loading`. `diffusers` models don't have any `modules_to_not_convert` attributes
        yet but this can change soon in the future.

        Args:
            model (`~diffusers.models.modeling_utils.ModelMixin`):
                The model to quantize
            torch_dtype (`torch.dtype`):
                The dtype passed in `from_pretrained` method.
        """

        return {
            name: torch_dtype
            for name, _ in model.named_parameters()
            if any(m in name for m in self.modules_to_not_convert)
        }

    def adjust_max_memory(self, max_memory: Dict[str, Union[int, str]]) -> Dict[str, Union[int, str]]:
        """adjust max_memory argument for infer_auto_device_map() if extra memory is needed for quantization"""
        return max_memory

    def check_if_quantized_param(
        self,
        model: "ModelMixin",
        param_value: "torch.Tensor",
        param_name: str,
        state_dict: Dict[str, Any],
        **kwargs,
    ) -> bool:
        """
        checks if a loaded state_dict component is part of quantized param + some validation; only defined for
        quantization methods that require to create a new parameters for quantization.
        """
        return False

    def create_quantized_param(self, *args, **kwargs) -> "torch.nn.Parameter":
        """
        takes needed components from state_dict and creates quantized param.
        """
        return

    def check_quantized_param_shape(self, *args, **kwargs):
        """
        checks if the quantized param has expected shape.
        """
        return True

    def validate_environment(self, *args, **kwargs):
        """
        This method is used to potentially check for potential conflicts with arguments that are passed in
        `from_pretrained`. You need to define it for all future quantizers that are integrated with diffusers. If no
        explicit check are needed, simply return nothing.
        """
        return

    def preprocess_model(self, model: "ModelMixin", **kwargs):
        """
        Setting model attributes and/or converting model before weights loading. At this point the model should be
        initialized on the meta device so you can freely manipulate the skeleton of the model in order to replace
        modules in-place. Make sure to override the abstract method `_process_model_before_weight_loading`.

        Args:
            model (`~diffusers.models.modeling_utils.ModelMixin`):
                The model to quantize
            kwargs (`dict`, *optional*):
                The keyword arguments that are passed along `_process_model_before_weight_loading`.
        """
        model.is_quantized = True
        model.quantization_method = self.quantization_config.quant_method
        return self._process_model_before_weight_loading(model, **kwargs)

    def postprocess_model(self, model: "ModelMixin", **kwargs):
        """
        Post-process the model post weights loading. Make sure to override the abstract method
        `_process_model_after_weight_loading`.

        Args:
            model (`~diffusers.models.modeling_utils.ModelMixin`):
                The model to quantize
            kwargs (`dict`, *optional*):
                The keyword arguments that are passed along `_process_model_after_weight_loading`.
        """
        return self._process_model_after_weight_loading(model, **kwargs)

    def dequantize(self, model):
        """
        Potentially dequantize the model to retrive the original model, with some loss in accuracy / performance. Note
        not all quantization schemes support this.
        """
        model = self._dequantize(model)

        # Delete quantizer and quantization config
        del model.hf_quantizer

        return model

    def _dequantize(self, model):
        raise NotImplementedError(
            f"{self.quantization_config.quant_method} has no implementation of `dequantize`, please raise an issue on GitHub."
        )

    @abstractmethod
    def _process_model_before_weight_loading(self, model, **kwargs):
        ...

    @abstractmethod
    def _process_model_after_weight_loading(self, model, **kwargs):
        ...

    @property
    @abstractmethod
    def is_serializable(self):
        ...

    @property
    @abstractmethod
    def is_trainable(self):
        ...
