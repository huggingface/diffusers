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
https://github.com/huggingface/transformers/blob/c409cd81777fb27aadc043ed3d8339dbc020fb3b/src/transformers/quantizers/quantizer_bnb_4bit.py
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from ...utils import get_module_from_name
from ..base import DiffusersQuantizer


if TYPE_CHECKING:
    from ...models.modeling_utils import ModelMixin

from ...utils import (
    is_accelerate_available,
    is_accelerate_version,
    is_bitsandbytes_available,
    is_bitsandbytes_version,
    is_torch_available,
    logging,
)


if is_torch_available():
    import torch

logger = logging.get_logger(__name__)


class BnB4BitDiffusersQuantizer(DiffusersQuantizer):
    """
    4-bit quantization from bitsandbytes.py quantization method:
        before loading: converts transformer layers into Linear4bit during loading: load 16bit weight and pass to the
        layer object after: quantizes individual weights in Linear4bit into 4bit at the first .cuda() call saving:
            from state dict, as usual; saves weights and `quant_state` components
        loading:
            need to locate `quant_state` components and pass to Param4bit constructor
    """

    use_keep_in_fp32_modules = True
    requires_calibration = False

    def __init__(self, quantization_config, **kwargs):
        super().__init__(quantization_config, **kwargs)

        if self.quantization_config.llm_int8_skip_modules is not None:
            self.modules_to_not_convert = self.quantization_config.llm_int8_skip_modules

    def validate_environment(self, *args, **kwargs):
        if not torch.cuda.is_available():
            raise RuntimeError("No GPU found. A GPU is needed for quantization.")
        if not is_accelerate_available() or is_accelerate_version("<", "0.26.0"):
            raise ImportError(
                "Using `bitsandbytes` 4-bit quantization requires Accelerate: `pip install 'accelerate>=0.26.0'`"
            )
        if not is_bitsandbytes_available() or is_bitsandbytes_version("<", "0.43.3"):
            raise ImportError(
                "Using `bitsandbytes` 4-bit quantization requires the latest version of bitsandbytes: `pip install -U bitsandbytes`"
            )

        if kwargs.get("from_flax", False):
            raise ValueError(
                "Converting into 4-bit weights from flax weights is currently not supported, please make"
                " sure the weights are in PyTorch format."
            )

        device_map = kwargs.get("device_map", None)
        if (
            device_map is not None
            and isinstance(device_map, dict)
            and not self.quantization_config.llm_int8_enable_fp32_cpu_offload
        ):
            device_map_without_no_convert = {
                key: device_map[key] for key in device_map.keys() if key not in self.modules_to_not_convert
            }
            if "cpu" in device_map_without_no_convert.values() or "disk" in device_map_without_no_convert.values():
                raise ValueError(
                    "Some modules are dispatched on the CPU or the disk. Make sure you have enough GPU RAM to fit the "
                    "quantized model. If you want to dispatch the model on the CPU or the disk while keeping these modules "
                    "in 32-bit, you need to set `load_in_8bit_fp32_cpu_offload=True` and pass a custom `device_map` to "
                    "`from_pretrained`. Check "
                    "https://huggingface.co/docs/transformers/main/en/main_classes/quantization#offload-between-cpu-and-gpu "
                    "for more details. "
                )

    def adjust_target_dtype(self, target_dtype: "torch.dtype") -> "torch.dtype":
        if target_dtype != torch.int8:
            from accelerate.utils import CustomDtype

            logger.info("target_dtype {target_dtype} is replaced by `CustomDtype.INT4` for 4-bit BnB quantization")
            return CustomDtype.INT4
        else:
            raise ValueError(f"Wrong `target_dtype` ({target_dtype}) provided.")

    def check_if_quantized_param(
        self,
        model: "ModelMixin",
        param_value: "torch.Tensor",
        param_name: str,
        state_dict: Dict[str, Any],
        **kwargs,
    ) -> bool:
        import bitsandbytes as bnb

        module, tensor_name = get_module_from_name(model, param_name)
        if isinstance(module._parameters.get(tensor_name, None), bnb.nn.Params4bit):
            # Add here check for loaded components' dtypes once serialization is implemented
            return True
        elif isinstance(module, bnb.nn.Linear4bit) and tensor_name == "bias":
            # bias could be loaded by regular set_module_tensor_to_device() from accelerate,
            # but it would wrongly use uninitialized weight there.
            return True
        else:
            return False

    def create_quantized_param(
        self,
        model: "ModelMixin",
        param_value: "torch.Tensor",
        param_name: str,
        target_device: "torch.device",
        state_dict: Dict[str, Any],
        unexpected_keys: Optional[List[str]] = None,
    ):
        import bitsandbytes as bnb

        module, tensor_name = get_module_from_name(model, param_name)

        if tensor_name not in module._parameters:
            raise ValueError(f"{module} does not have a parameter or a buffer named {tensor_name}.")

        old_value = getattr(module, tensor_name)

        if tensor_name == "bias":
            if param_value is None:
                new_value = old_value.to(target_device)
            else:
                new_value = param_value.to(target_device)

            new_value = torch.nn.Parameter(new_value, requires_grad=old_value.requires_grad)
            module._parameters[tensor_name] = new_value
            return

        if not isinstance(module._parameters[tensor_name], bnb.nn.Params4bit):
            raise ValueError("this function only loads `Linear4bit components`")
        if (
            old_value.device == torch.device("meta")
            and target_device not in ["meta", torch.device("meta")]
            and param_value is None
        ):
            raise ValueError(f"{tensor_name} is on the meta device, we need a `value` to put in on {target_device}.")

        # construct `new_value` for the module._parameters[tensor_name]:
        if self.pre_quantized:
            # 4bit loading. Collecting components for restoring quantized weight
            # This can be expanded to make a universal call for any quantized weight loading

            if not self.is_serializable:
                raise ValueError(
                    "Detected int4 weights but the version of bitsandbytes is not compatible with int4 serialization. "
                    "Make sure to download the latest `bitsandbytes` version. `pip install --upgrade bitsandbytes`."
                )

            if (param_name + ".quant_state.bitsandbytes__fp4" not in state_dict) and (
                param_name + ".quant_state.bitsandbytes__nf4" not in state_dict
            ):
                raise ValueError(
                    f"Supplied state dict for {param_name} does not contain `bitsandbytes__*` and possibly other `quantized_stats` components."
                )

            quantized_stats = {}
            for k, v in state_dict.items():
                # `startswith` to counter for edge cases where `param_name`
                # substring can be present in multiple places in the `state_dict`
                if param_name + "." in k and k.startswith(param_name):
                    quantized_stats[k] = v
                    if unexpected_keys is not None and k in unexpected_keys:
                        unexpected_keys.remove(k)

            new_value = bnb.nn.Params4bit.from_prequantized(
                data=param_value,
                quantized_stats=quantized_stats,
                requires_grad=False,
                device=target_device,
            )
        else:
            new_value = param_value.to("cpu")
            kwargs = old_value.__dict__
            new_value = bnb.nn.Params4bit(new_value, requires_grad=False, **kwargs).to(target_device)

        module._parameters[tensor_name] = new_value

    def check_quantized_param_shape(self, param_name, current_param, loaded_param):
        current_param_shape = current_param.shape
        loaded_param_shape = loaded_param.shape

        n = current_param_shape.numel()
        inferred_shape = (n,) if "bias" in param_name else ((n + 1) // 2, 1)
        if loaded_param_shape != inferred_shape:
            raise ValueError(
                f"Expected the flattened shape of the current param ({param_name}) to be {loaded_param_shape} but is {inferred_shape}."
            )
        else:
            return True

    def adjust_max_memory(self, max_memory: Dict[str, Union[int, str]]) -> Dict[str, Union[int, str]]:
        # need more space for buffers that are created during quantization
        max_memory = {key: val * 0.90 for key, val in max_memory.items()}
        return max_memory

    def update_torch_dtype(self, torch_dtype: "torch.dtype") -> "torch.dtype":
        if torch_dtype is None:
            # We force the `dtype` to be float16, this is a requirement from `bitsandbytes`
            logger.info(
                "Overriding torch_dtype=%s with `torch_dtype=torch.float16` due to "
                "requirements of `bitsandbytes` to enable model loading in 8-bit or 4-bit. "
                "Pass your own torch_dtype to specify the dtype of the remaining non-linear layers or pass"
                " torch_dtype=torch.float16 to remove this warning.",
                torch_dtype,
            )
            torch_dtype = torch.float16
        return torch_dtype

    # (sayakpaul): I think it could be better to disable custom `device_map`s
    # for the first phase of the integration in the interest of simplicity.
    # Commenting this for discussions on the PR.
    # def update_device_map(self, device_map):
    #     if device_map is None:
    #         device_map = {"": torch.cuda.current_device()}
    #         logger.info(
    #             "The device_map was not initialized. "
    #             "Setting device_map to {'':torch.cuda.current_device()}. "
    #             "If you want to use the model for inference, please set device_map ='auto' "
    #         )
    #     return device_map

    def _process_model_before_weight_loading(
        self,
        model: "ModelMixin",
        device_map,
        keep_in_fp32_modules: List[str] = [],
        **kwargs,
    ):
        from .utils import replace_with_bnb_linear

        load_in_8bit_fp32_cpu_offload = self.quantization_config.llm_int8_enable_fp32_cpu_offload

        # We may keep some modules such as the `proj_out` in their original dtype for numerical stability reasons
        self.modules_to_not_convert = self.quantization_config.llm_int8_skip_modules

        if not isinstance(self.modules_to_not_convert, list):
            self.modules_to_not_convert = [self.modules_to_not_convert]

        self.modules_to_not_convert.extend(keep_in_fp32_modules)

        # Extend `self.modules_to_not_convert` to keys that are supposed to be offloaded to `cpu` or `disk`
        if isinstance(device_map, dict) and len(device_map.keys()) > 1:
            keys_on_cpu = [key for key, value in device_map.items() if value in ["disk", "cpu"]]

            if len(keys_on_cpu) > 0 and not load_in_8bit_fp32_cpu_offload:
                raise ValueError(
                    "If you want to offload some keys to `cpu` or `disk`, you need to set "
                    "`llm_int8_enable_fp32_cpu_offload=True`. Note that these modules will not be "
                    " converted to 8-bit but kept in 32-bit."
                )
            self.modules_to_not_convert.extend(keys_on_cpu)

        # Purge `None`.
        # Unlike `transformers`, we don't know if we should always keep certain modules in FP32
        # in case of diffusion transformer models. For language models and others alike, `lm_head`
        # and tied modules are usually kept in FP32.
        self.modules_to_not_convert = [module for module in self.modules_to_not_convert if module is not None]

        model = replace_with_bnb_linear(
            model, modules_to_not_convert=self.modules_to_not_convert, quantization_config=self.quantization_config
        )
        model.config.quantization_config = self.quantization_config

    def _process_model_after_weight_loading(self, model: "ModelMixin", **kwargs):
        model.is_loaded_in_4bit = True
        model.is_4bit_serializable = self.is_serializable
        return model

    @property
    def is_serializable(self):
        # Because we're mandating `bitsandbytes` 0.43.3.
        return True

    @property
    def is_trainable(self) -> bool:
        # Because we're mandating `bitsandbytes` 0.43.3.
        return True

    def _dequantize(self, model):
        from .utils import dequantize_and_replace

        is_model_on_cpu = model.device.type == "cpu"
        if is_model_on_cpu:
            logger.info(
                "Model was found to be on CPU (could happen as a result of `enable_model_cpu_offload()`). So, moving it to GPU. After dequantization, will move the model back to CPU again to preserve the previous device."
            )
            model.to(torch.cuda.current_device())

        model = dequantize_and_replace(
            model, self.modules_to_not_convert, quantization_config=self.quantization_config
        )
        if is_model_on_cpu:
            model.to("cpu")
        return model


class BnB8BitDiffusersQuantizer(DiffusersQuantizer):
    """
    8-bit quantization from bitsandbytes quantization method:
        before loading: converts transformer layers into Linear8bitLt during loading: load 16bit weight and pass to the
        layer object after: quantizes individual weights in Linear8bitLt into 8bit at fitst .cuda() call
    saving:
        from state dict, as usual; saves weights and 'SCB' component
    loading:
        need to locate SCB component and pass to the Linear8bitLt object
    """

    use_keep_in_fp32_modules = True
    requires_calibration = False

    def __init__(self, quantization_config, **kwargs):
        super().__init__(quantization_config, **kwargs)

        if self.quantization_config.llm_int8_skip_modules is not None:
            self.modules_to_not_convert = self.quantization_config.llm_int8_skip_modules

    def validate_environment(self, *args, **kwargs):
        if not torch.cuda.is_available():
            raise RuntimeError("No GPU found. A GPU is needed for quantization.")
        if not is_accelerate_available() or is_accelerate_version("<", "0.26.0"):
            raise ImportError(
                "Using `bitsandbytes` 8-bit quantization requires Accelerate: `pip install 'accelerate>=0.26.0'`"
            )
        if not is_bitsandbytes_available() or is_bitsandbytes_version("<", "0.43.3"):
            raise ImportError(
                "Using `bitsandbytes` 8-bit quantization requires the latest version of bitsandbytes: `pip install -U bitsandbytes`"
            )

        if kwargs.get("from_flax", False):
            raise ValueError(
                "Converting into 8-bit weights from flax weights is currently not supported, please make"
                " sure the weights are in PyTorch format."
            )

        device_map = kwargs.get("device_map", None)
        if (
            device_map is not None
            and isinstance(device_map, dict)
            and not self.quantization_config.llm_int8_enable_fp32_cpu_offload
        ):
            device_map_without_no_convert = {
                key: device_map[key] for key in device_map.keys() if key not in self.modules_to_not_convert
            }
            if "cpu" in device_map_without_no_convert.values() or "disk" in device_map_without_no_convert.values():
                raise ValueError(
                    "Some modules are dispatched on the CPU or the disk. Make sure you have enough GPU RAM to fit the "
                    "quantized model. If you want to dispatch the model on the CPU or the disk while keeping these modules "
                    "in 32-bit, you need to set `load_in_8bit_fp32_cpu_offload=True` and pass a custom `device_map` to "
                    "`from_pretrained`. Check "
                    "https://huggingface.co/docs/transformers/main/en/main_classes/quantization#offload-between-cpu-and-gpu "
                    "for more details. "
                )

    # Copied from diffusers.quantizers.bitsandbytes.bnb_quantizer.BnB4BitDiffusersQuantizer.adjust_max_memory
    def adjust_max_memory(self, max_memory: Dict[str, Union[int, str]]) -> Dict[str, Union[int, str]]:
        # need more space for buffers that are created during quantization
        max_memory = {key: val * 0.90 for key, val in max_memory.items()}
        return max_memory

    # Copied from diffusers.quantizers.bitsandbytes.bnb_quantizer.BnB4BitDiffusersQuantizer.update_torch_dtype
    def update_torch_dtype(self, torch_dtype: "torch.dtype") -> "torch.dtype":
        if torch_dtype is None:
            # We force the `dtype` to be float16, this is a requirement from `bitsandbytes`
            logger.info(
                "Overriding torch_dtype=%s with `torch_dtype=torch.float16` due to "
                "requirements of `bitsandbytes` to enable model loading in 8-bit or 4-bit. "
                "Pass your own torch_dtype to specify the dtype of the remaining non-linear layers or pass"
                " torch_dtype=torch.float16 to remove this warning.",
                torch_dtype,
            )
            torch_dtype = torch.float16
        return torch_dtype

    # # Copied from diffusers.quantizers.bitsandbytes.bnb_quantizer.BnB4BitDiffusersQuantizer.update_device_map
    # def update_device_map(self, device_map):
    #     if device_map is None:
    #         device_map = {"": torch.cuda.current_device()}
    #         logger.info(
    #             "The device_map was not initialized. "
    #             "Setting device_map to {'':torch.cuda.current_device()}. "
    #             "If you want to use the model for inference, please set device_map ='auto' "
    #         )
    #     return device_map

    def adjust_target_dtype(self, target_dtype: "torch.dtype") -> "torch.dtype":
        if target_dtype != torch.int8:
            logger.info("target_dtype {target_dtype} is replaced by `torch.int8` for 8-bit BnB quantization")
        return torch.int8

    def check_if_quantized_param(
        self,
        model: "ModelMixin",
        param_value: "torch.Tensor",
        param_name: str,
        state_dict: Dict[str, Any],
        **kwargs,
    ):
        import bitsandbytes as bnb

        module, tensor_name = get_module_from_name(model, param_name)
        if isinstance(module._parameters.get(tensor_name, None), bnb.nn.Int8Params):
            if self.pre_quantized:
                if param_name.replace("weight", "SCB") not in state_dict.keys():
                    raise ValueError("Missing quantization component `SCB`")
                if param_value.dtype != torch.int8:
                    raise ValueError(
                        f"Incompatible dtype `{param_value.dtype}` when loading 8-bit prequantized weight. Expected `torch.int8`."
                    )
            return True
        return False

    def create_quantized_param(
        self,
        model: "ModelMixin",
        param_value: "torch.Tensor",
        param_name: str,
        target_device: "torch.device",
        state_dict: Dict[str, Any],
        unexpected_keys: Optional[List[str]] = None,
    ):
        import bitsandbytes as bnb

        fp16_statistics_key = param_name.replace("weight", "SCB")
        fp16_weights_format_key = param_name.replace("weight", "weight_format")

        fp16_statistics = state_dict.get(fp16_statistics_key, None)
        fp16_weights_format = state_dict.get(fp16_weights_format_key, None)

        module, tensor_name = get_module_from_name(model, param_name)
        if tensor_name not in module._parameters:
            raise ValueError(f"{module} does not have a parameter or a buffer named {tensor_name}.")

        old_value = getattr(module, tensor_name)

        if not isinstance(module._parameters[tensor_name], bnb.nn.Int8Params):
            raise ValueError(f"Parameter `{tensor_name}` should only be a `bnb.nn.Int8Params` instance.")
        if (
            old_value.device == torch.device("meta")
            and target_device not in ["meta", torch.device("meta")]
            and param_value is None
        ):
            raise ValueError(f"{tensor_name} is on the meta device, we need a `value` to put in on {target_device}.")

        new_value = param_value.to("cpu")
        if self.pre_quantized and not self.is_serializable:
            raise ValueError(
                "Detected int8 weights but the version of bitsandbytes is not compatible with int8 serialization. "
                "Make sure to download the latest `bitsandbytes` version. `pip install --upgrade bitsandbytes`."
            )

        kwargs = old_value.__dict__
        new_value = bnb.nn.Int8Params(new_value, requires_grad=False, **kwargs).to(target_device)

        module._parameters[tensor_name] = new_value
        if fp16_statistics is not None:
            setattr(module.weight, "SCB", fp16_statistics.to(target_device))
            if unexpected_keys is not None:
                unexpected_keys.remove(fp16_statistics_key)

        # We just need to pop the `weight_format` keys from the state dict to remove unneeded
        # messages. The correct format is correctly retrieved during the first forward pass.
        if fp16_weights_format is not None and unexpected_keys is not None:
            unexpected_keys.remove(fp16_weights_format_key)

    # Copied from diffusers.quantizers.bitsandbytes.bnb_quantizer.BnB4BitDiffusersQuantizer._process_model_after_weight_loading with 4bit->8bit
    def _process_model_after_weight_loading(self, model: "ModelMixin", **kwargs):
        model.is_loaded_in_8bit = True
        model.is_8bit_serializable = self.is_serializable
        return model

    # Copied from diffusers.quantizers.bitsandbytes.bnb_quantizer.BnB4BitDiffusersQuantizer._process_model_before_weight_loading
    def _process_model_before_weight_loading(
        self,
        model: "ModelMixin",
        device_map,
        keep_in_fp32_modules: List[str] = [],
        **kwargs,
    ):
        from .utils import replace_with_bnb_linear

        load_in_8bit_fp32_cpu_offload = self.quantization_config.llm_int8_enable_fp32_cpu_offload

        # We may keep some modules such as the `proj_out` in their original dtype for numerical stability reasons
        self.modules_to_not_convert = self.quantization_config.llm_int8_skip_modules

        if not isinstance(self.modules_to_not_convert, list):
            self.modules_to_not_convert = [self.modules_to_not_convert]

        self.modules_to_not_convert.extend(keep_in_fp32_modules)

        # Extend `self.modules_to_not_convert` to keys that are supposed to be offloaded to `cpu` or `disk`
        if isinstance(device_map, dict) and len(device_map.keys()) > 1:
            keys_on_cpu = [key for key, value in device_map.items() if value in ["disk", "cpu"]]

            if len(keys_on_cpu) > 0 and not load_in_8bit_fp32_cpu_offload:
                raise ValueError(
                    "If you want to offload some keys to `cpu` or `disk`, you need to set "
                    "`llm_int8_enable_fp32_cpu_offload=True`. Note that these modules will not be "
                    " converted to 8-bit but kept in 32-bit."
                )
            self.modules_to_not_convert.extend(keys_on_cpu)

        # Purge `None`.
        # Unlike `transformers`, we don't know if we should always keep certain modules in FP32
        # in case of diffusion transformer models. For language models and others alike, `lm_head`
        # and tied modules are usually kept in FP32.
        self.modules_to_not_convert = [module for module in self.modules_to_not_convert if module is not None]

        model = replace_with_bnb_linear(
            model, modules_to_not_convert=self.modules_to_not_convert, quantization_config=self.quantization_config
        )
        model.config.quantization_config = self.quantization_config

    @property
    # Copied from diffusers.quantizers.bitsandbytes.bnb_quantizer.BnB4BitDiffusersQuantizer.is_serializable
    def is_serializable(self):
        # Because we're mandating `bitsandbytes` 0.43.3.
        return True

    @property
    # Copied from diffusers.quantizers.bitsandbytes.bnb_quantizer.BnB4BitDiffusersQuantizer.is_serializable
    def is_trainable(self) -> bool:
        # Because we're mandating `bitsandbytes` 0.43.3.
        return True

    def _dequantize(self, model):
        from .utils import dequantize_and_replace

        model = dequantize_and_replace(
            model, self.modules_to_not_convert, quantization_config=self.quantization_config
        )
        return model
