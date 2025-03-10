#!/usr/bin/env python
# coding=utf-8

# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
https://github.com/huggingface/transformers/blob/52cb4034ada381fe1ffe8d428a1076e5411a8026/src/transformers/utils/quantization_config.py
"""

import copy
import importlib.metadata
import inspect
import json
import os
from dataclasses import dataclass
from enum import Enum
from functools import partial
from typing import Any, Dict, List, Optional, Union

from packaging import version

from ..utils import is_torch_available, is_torchao_available, logging


if is_torch_available():
    import torch

logger = logging.get_logger(__name__)


class QuantizationMethod(str, Enum):
    BITS_AND_BYTES = "bitsandbytes"
    GGUF = "gguf"
    TORCHAO = "torchao"
    QUANTO = "quanto"


if is_torchao_available():
    from torchao.quantization.quant_primitives import MappingType

    class TorchAoJSONEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, MappingType):
                return obj.name
            return super().default(obj)


@dataclass
class QuantizationConfigMixin:
    """
    Mixin class for quantization config
    """

    quant_method: QuantizationMethod
    _exclude_attributes_at_init = []

    @classmethod
    def from_dict(cls, config_dict, return_unused_kwargs=False, **kwargs):
        """
        Instantiates a [`QuantizationConfigMixin`] from a Python dictionary of parameters.

        Args:
            config_dict (`Dict[str, Any]`):
                Dictionary that will be used to instantiate the configuration object.
            return_unused_kwargs (`bool`,*optional*, defaults to `False`):
                Whether or not to return a list of unused keyword arguments. Used for `from_pretrained` method in
                `PreTrainedModel`.
            kwargs (`Dict[str, Any]`):
                Additional parameters from which to initialize the configuration object.

        Returns:
            [`QuantizationConfigMixin`]: The configuration object instantiated from those parameters.
        """

        config = cls(**config_dict)

        to_remove = []
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
                to_remove.append(key)
        for key in to_remove:
            kwargs.pop(key, None)

        if return_unused_kwargs:
            return config, kwargs
        else:
            return config

    def to_json_file(self, json_file_path: Union[str, os.PathLike]):
        """
        Save this instance to a JSON file.

        Args:
            json_file_path (`str` or `os.PathLike`):
                Path to the JSON file in which this configuration instance's parameters will be saved.
            use_diff (`bool`, *optional*, defaults to `True`):
                If set to `True`, only the difference between the config instance and the default
                `QuantizationConfig()` is serialized to JSON file.
        """
        with open(json_file_path, "w", encoding="utf-8") as writer:
            config_dict = self.to_dict()
            json_string = json.dumps(config_dict, indent=2, sort_keys=True) + "\n"

            writer.write(json_string)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes this instance to a Python dictionary. Returns:
            `Dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance.
        """
        return copy.deepcopy(self.__dict__)

    def __iter__(self):
        """allows `dict(obj)` for situations where obj may be a dict or QuantizationConfigMixin"""
        for attr, value in copy.deepcopy(self.__dict__).items():
            yield attr, value

    def __repr__(self):
        return f"{self.__class__.__name__} {self.to_json_string()}"

    def to_json_string(self, use_diff: bool = True) -> str:
        """
        Serializes this instance to a JSON string.

        Args:
            use_diff (`bool`, *optional*, defaults to `True`):
                If set to `True`, only the difference between the config instance and the default `PretrainedConfig()`
                is serialized to JSON string.

        Returns:
            `str`: String containing all the attributes that make up this configuration instance in JSON format.
        """
        if use_diff is True:
            config_dict = self.to_diff_dict()
        else:
            config_dict = self.to_dict()
        return json.dumps(config_dict, indent=2, sort_keys=True) + "\n"

    def update(self, **kwargs):
        """
        Updates attributes of this class instance with attributes from `kwargs` if they match existing attributes,
        returning all the unused kwargs.

        Args:
            kwargs (`Dict[str, Any]`):
                Dictionary of attributes to tentatively update this class.

        Returns:
            `Dict[str, Any]`: Dictionary containing all the key-value pairs that were not used to update the instance.
        """
        to_remove = []
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                to_remove.append(key)

        # Remove all the attributes that were updated, without modifying the input dict
        unused_kwargs = {key: value for key, value in kwargs.items() if key not in to_remove}
        return unused_kwargs


@dataclass
class BitsAndBytesConfig(QuantizationConfigMixin):
    """
    This is a wrapper class about all possible attributes and features that you can play with a model that has been
    loaded using `bitsandbytes`.

    This replaces `load_in_8bit` or `load_in_4bit`therefore both options are mutually exclusive.

    Currently only supports `LLM.int8()`, `FP4`, and `NF4` quantization. If more methods are added to `bitsandbytes`,
    then more arguments will be added to this class.

    Args:
        load_in_8bit (`bool`, *optional*, defaults to `False`):
            This flag is used to enable 8-bit quantization with LLM.int8().
        load_in_4bit (`bool`, *optional*, defaults to `False`):
            This flag is used to enable 4-bit quantization by replacing the Linear layers with FP4/NF4 layers from
            `bitsandbytes`.
        llm_int8_threshold (`float`, *optional*, defaults to 6.0):
            This corresponds to the outlier threshold for outlier detection as described in `LLM.int8() : 8-bit Matrix
            Multiplication for Transformers at Scale` paper: https://arxiv.org/abs/2208.07339 Any hidden states value
            that is above this threshold will be considered an outlier and the operation on those values will be done
            in fp16. Values are usually normally distributed, that is, most values are in the range [-3.5, 3.5], but
            there are some exceptional systematic outliers that are very differently distributed for large models.
            These outliers are often in the interval [-60, -6] or [6, 60]. Int8 quantization works well for values of
            magnitude ~5, but beyond that, there is a significant performance penalty. A good default threshold is 6,
            but a lower threshold might be needed for more unstable models (small models, fine-tuning).
        llm_int8_skip_modules (`List[str]`, *optional*):
            An explicit list of the modules that we do not want to convert in 8-bit. This is useful for models such as
            Jukebox that has several heads in different places and not necessarily at the last position. For example
            for `CausalLM` models, the last `lm_head` is typically kept in its original `dtype`.
        llm_int8_enable_fp32_cpu_offload (`bool`, *optional*, defaults to `False`):
            This flag is used for advanced use cases and users that are aware of this feature. If you want to split
            your model in different parts and run some parts in int8 on GPU and some parts in fp32 on CPU, you can use
            this flag. This is useful for offloading large models such as `google/flan-t5-xxl`. Note that the int8
            operations will not be run on CPU.
        llm_int8_has_fp16_weight (`bool`, *optional*, defaults to `False`):
            This flag runs LLM.int8() with 16-bit main weights. This is useful for fine-tuning as the weights do not
            have to be converted back and forth for the backward pass.
        bnb_4bit_compute_dtype (`torch.dtype` or str, *optional*, defaults to `torch.float32`):
            This sets the computational type which might be different than the input type. For example, inputs might be
            fp32, but computation can be set to bf16 for speedups.
        bnb_4bit_quant_type (`str`,  *optional*, defaults to `"fp4"`):
            This sets the quantization data type in the bnb.nn.Linear4Bit layers. Options are FP4 and NF4 data types
            which are specified by `fp4` or `nf4`.
        bnb_4bit_use_double_quant (`bool`, *optional*, defaults to `False`):
            This flag is used for nested quantization where the quantization constants from the first quantization are
            quantized again.
        bnb_4bit_quant_storage (`torch.dtype` or str, *optional*, defaults to `torch.uint8`):
            This sets the storage type to pack the quanitzed 4-bit prarams.
        kwargs (`Dict[str, Any]`, *optional*):
            Additional parameters from which to initialize the configuration object.
    """

    _exclude_attributes_at_init = ["_load_in_4bit", "_load_in_8bit", "quant_method"]

    def __init__(
        self,
        load_in_8bit=False,
        load_in_4bit=False,
        llm_int8_threshold=6.0,
        llm_int8_skip_modules=None,
        llm_int8_enable_fp32_cpu_offload=False,
        llm_int8_has_fp16_weight=False,
        bnb_4bit_compute_dtype=None,
        bnb_4bit_quant_type="fp4",
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_storage=None,
        **kwargs,
    ):
        self.quant_method = QuantizationMethod.BITS_AND_BYTES

        if load_in_4bit and load_in_8bit:
            raise ValueError("load_in_4bit and load_in_8bit are both True, but only one can be used at the same time")

        self._load_in_8bit = load_in_8bit
        self._load_in_4bit = load_in_4bit
        self.llm_int8_threshold = llm_int8_threshold
        self.llm_int8_skip_modules = llm_int8_skip_modules
        self.llm_int8_enable_fp32_cpu_offload = llm_int8_enable_fp32_cpu_offload
        self.llm_int8_has_fp16_weight = llm_int8_has_fp16_weight
        self.bnb_4bit_quant_type = bnb_4bit_quant_type
        self.bnb_4bit_use_double_quant = bnb_4bit_use_double_quant

        if bnb_4bit_compute_dtype is None:
            self.bnb_4bit_compute_dtype = torch.float32
        elif isinstance(bnb_4bit_compute_dtype, str):
            self.bnb_4bit_compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
        elif isinstance(bnb_4bit_compute_dtype, torch.dtype):
            self.bnb_4bit_compute_dtype = bnb_4bit_compute_dtype
        else:
            raise ValueError("bnb_4bit_compute_dtype must be a string or a torch.dtype")

        if bnb_4bit_quant_storage is None:
            self.bnb_4bit_quant_storage = torch.uint8
        elif isinstance(bnb_4bit_quant_storage, str):
            if bnb_4bit_quant_storage not in ["float16", "float32", "int8", "uint8", "float64", "bfloat16"]:
                raise ValueError(
                    "`bnb_4bit_quant_storage` must be a valid string (one of 'float16', 'float32', 'int8', 'uint8', 'float64', 'bfloat16') "
                )
            self.bnb_4bit_quant_storage = getattr(torch, bnb_4bit_quant_storage)
        elif isinstance(bnb_4bit_quant_storage, torch.dtype):
            self.bnb_4bit_quant_storage = bnb_4bit_quant_storage
        else:
            raise ValueError("bnb_4bit_quant_storage must be a string or a torch.dtype")

        if kwargs and not all(k in self._exclude_attributes_at_init for k in kwargs):
            logger.warning(f"Unused kwargs: {list(kwargs.keys())}. These kwargs are not used in {self.__class__}.")

        self.post_init()

    @property
    def load_in_4bit(self):
        return self._load_in_4bit

    @load_in_4bit.setter
    def load_in_4bit(self, value: bool):
        if not isinstance(value, bool):
            raise TypeError("load_in_4bit must be a boolean")

        if self.load_in_8bit and value:
            raise ValueError("load_in_4bit and load_in_8bit are both True, but only one can be used at the same time")
        self._load_in_4bit = value

    @property
    def load_in_8bit(self):
        return self._load_in_8bit

    @load_in_8bit.setter
    def load_in_8bit(self, value: bool):
        if not isinstance(value, bool):
            raise TypeError("load_in_8bit must be a boolean")

        if self.load_in_4bit and value:
            raise ValueError("load_in_4bit and load_in_8bit are both True, but only one can be used at the same time")
        self._load_in_8bit = value

    def post_init(self):
        r"""
        Safety checker that arguments are correct - also replaces some NoneType arguments with their default values.
        """
        if not isinstance(self.load_in_4bit, bool):
            raise TypeError("load_in_4bit must be a boolean")

        if not isinstance(self.load_in_8bit, bool):
            raise TypeError("load_in_8bit must be a boolean")

        if not isinstance(self.llm_int8_threshold, float):
            raise TypeError("llm_int8_threshold must be a float")

        if self.llm_int8_skip_modules is not None and not isinstance(self.llm_int8_skip_modules, list):
            raise TypeError("llm_int8_skip_modules must be a list of strings")
        if not isinstance(self.llm_int8_enable_fp32_cpu_offload, bool):
            raise TypeError("llm_int8_enable_fp32_cpu_offload must be a boolean")

        if not isinstance(self.llm_int8_has_fp16_weight, bool):
            raise TypeError("llm_int8_has_fp16_weight must be a boolean")

        if self.bnb_4bit_compute_dtype is not None and not isinstance(self.bnb_4bit_compute_dtype, torch.dtype):
            raise TypeError("bnb_4bit_compute_dtype must be torch.dtype")

        if not isinstance(self.bnb_4bit_quant_type, str):
            raise TypeError("bnb_4bit_quant_type must be a string")

        if not isinstance(self.bnb_4bit_use_double_quant, bool):
            raise TypeError("bnb_4bit_use_double_quant must be a boolean")

        if self.load_in_4bit and not version.parse(importlib.metadata.version("bitsandbytes")) >= version.parse(
            "0.39.0"
        ):
            raise ValueError(
                "4 bit quantization requires bitsandbytes>=0.39.0 - please upgrade your bitsandbytes version"
            )

    def is_quantizable(self):
        r"""
        Returns `True` if the model is quantizable, `False` otherwise.
        """
        return self.load_in_8bit or self.load_in_4bit

    def quantization_method(self):
        r"""
        This method returns the quantization method used for the model. If the model is not quantizable, it returns
        `None`.
        """
        if self.load_in_8bit:
            return "llm_int8"
        elif self.load_in_4bit and self.bnb_4bit_quant_type == "fp4":
            return "fp4"
        elif self.load_in_4bit and self.bnb_4bit_quant_type == "nf4":
            return "nf4"
        else:
            return None

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes this instance to a Python dictionary. Returns:
            `Dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance.
        """
        output = copy.deepcopy(self.__dict__)
        output["bnb_4bit_compute_dtype"] = str(output["bnb_4bit_compute_dtype"]).split(".")[1]
        output["bnb_4bit_quant_storage"] = str(output["bnb_4bit_quant_storage"]).split(".")[1]
        output["load_in_4bit"] = self.load_in_4bit
        output["load_in_8bit"] = self.load_in_8bit

        return output

    def __repr__(self):
        config_dict = self.to_dict()
        return f"{self.__class__.__name__} {json.dumps(config_dict, indent=2, sort_keys=True)}\n"

    def to_diff_dict(self) -> Dict[str, Any]:
        """
        Removes all attributes from config which correspond to the default config attributes for better readability and
        serializes to a Python dictionary.

        Returns:
            `Dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        config_dict = self.to_dict()

        # get the default config dict
        default_config_dict = BitsAndBytesConfig().to_dict()

        serializable_config_dict = {}

        # only serialize values that differ from the default config
        for key, value in config_dict.items():
            if value != default_config_dict[key]:
                serializable_config_dict[key] = value

        return serializable_config_dict


@dataclass
class GGUFQuantizationConfig(QuantizationConfigMixin):
    """This is a config class for GGUF Quantization techniques.

    Args:
        compute_dtype: (`torch.dtype`, defaults to `torch.float32`):
            This sets the computational type which might be different than the input type. For example, inputs might be
            fp32, but computation can be set to bf16 for speedups.

    """

    def __init__(self, compute_dtype: Optional["torch.dtype"] = None):
        self.quant_method = QuantizationMethod.GGUF
        self.compute_dtype = compute_dtype
        self.pre_quantized = True

        # TODO: (Dhruv) Add this as an init argument when we can support loading unquantized checkpoints.
        self.modules_to_not_convert = None

        if self.compute_dtype is None:
            self.compute_dtype = torch.float32


@dataclass
class TorchAoConfig(QuantizationConfigMixin):
    """This is a config class for torchao quantization/sparsity techniques.

    Args:
        quant_type (`str`):
            The type of quantization we want to use, currently supporting:
                - **Integer quantization:**
                    - Full function names: `int4_weight_only`, `int8_dynamic_activation_int4_weight`,
                      `int8_weight_only`, `int8_dynamic_activation_int8_weight`
                    - Shorthands: `int4wo`, `int4dq`, `int8wo`, `int8dq`

                - **Floating point 8-bit quantization:**
                    - Full function names: `float8_weight_only`, `float8_dynamic_activation_float8_weight`,
                      `float8_static_activation_float8_weight`
                    - Shorthands: `float8wo`, `float8wo_e5m2`, `float8wo_e4m3`, `float8dq`, `float8dq_e4m3`,
                      `float8_e4m3_tensor`, `float8_e4m3_row`,

                - **Floating point X-bit quantization:**
                    - Full function names: `fpx_weight_only`
                    - Shorthands: `fpX_eAwB`, where `X` is the number of bits (between `1` to `7`), `A` is the number
                      of exponent bits and `B` is the number of mantissa bits. The constraint of `X == A + B + 1` must
                      be satisfied for a given shorthand notation.

                - **Unsigned Integer quantization:**
                    - Full function names: `uintx_weight_only`
                    - Shorthands: `uint1wo`, `uint2wo`, `uint3wo`, `uint4wo`, `uint5wo`, `uint6wo`, `uint7wo`
        modules_to_not_convert (`List[str]`, *optional*, default to `None`):
            The list of modules to not quantize, useful for quantizing models that explicitly require to have some
            modules left in their original precision.
        kwargs (`Dict[str, Any]`, *optional*):
            The keyword arguments for the chosen type of quantization, for example, int4_weight_only quantization
            supports two keyword arguments `group_size` and `inner_k_tiles` currently. More API examples and
            documentation of arguments can be found in
            https://github.com/pytorch/ao/tree/main/torchao/quantization#other-available-quantization-techniques

    Example:
        ```python
        from diffusers import FluxTransformer2DModel, TorchAoConfig

        quantization_config = TorchAoConfig("int8wo")
        transformer = FluxTransformer2DModel.from_pretrained(
            "black-forest-labs/Flux.1-Dev",
            subfolder="transformer",
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
        )
        ```
    """

    def __init__(self, quant_type: str, modules_to_not_convert: Optional[List[str]] = None, **kwargs) -> None:
        self.quant_method = QuantizationMethod.TORCHAO
        self.quant_type = quant_type
        self.modules_to_not_convert = modules_to_not_convert

        # When we load from serialized config, "quant_type_kwargs" will be the key
        if "quant_type_kwargs" in kwargs:
            self.quant_type_kwargs = kwargs["quant_type_kwargs"]
        else:
            self.quant_type_kwargs = kwargs

        TORCHAO_QUANT_TYPE_METHODS = self._get_torchao_quant_type_to_method()
        if self.quant_type not in TORCHAO_QUANT_TYPE_METHODS.keys():
            is_floating_quant_type = self.quant_type.startswith("float") or self.quant_type.startswith("fp")
            if is_floating_quant_type and not self._is_cuda_capability_atleast_8_9():
                raise ValueError(
                    f"Requested quantization type: {self.quant_type} is not supported on GPUs with CUDA capability <= 8.9. You "
                    f"can check the CUDA capability of your GPU using `torch.cuda.get_device_capability()`."
                )

            raise ValueError(
                f"Requested quantization type: {self.quant_type} is not supported or is an incorrect `quant_type` name. If you think the "
                f"provided quantization type should be supported, please open an issue at https://github.com/huggingface/diffusers/issues."
            )

        method = TORCHAO_QUANT_TYPE_METHODS[self.quant_type]
        signature = inspect.signature(method)
        all_kwargs = {
            param.name
            for param in signature.parameters.values()
            if param.kind in [inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD]
        }
        unsupported_kwargs = list(self.quant_type_kwargs.keys() - all_kwargs)

        if len(unsupported_kwargs) > 0:
            raise ValueError(
                f'The quantization method "{quant_type}" does not support the following keyword arguments: '
                f"{unsupported_kwargs}. The following keywords arguments are supported: {all_kwargs}."
            )

    @classmethod
    def _get_torchao_quant_type_to_method(cls):
        r"""
        Returns supported torchao quantization types with all commonly used notations.
        """

        if is_torchao_available():
            # TODO(aryan): Support autoquant and sparsify
            from torchao.quantization import (
                float8_dynamic_activation_float8_weight,
                float8_static_activation_float8_weight,
                float8_weight_only,
                fpx_weight_only,
                int4_weight_only,
                int8_dynamic_activation_int4_weight,
                int8_dynamic_activation_int8_weight,
                int8_weight_only,
                uintx_weight_only,
            )

            # TODO(aryan): Add a note on how to use PerAxis and PerGroup observers
            from torchao.quantization.observer import PerRow, PerTensor

            def generate_float8dq_types(dtype: torch.dtype):
                name = "e5m2" if dtype == torch.float8_e5m2 else "e4m3"
                types = {}

                for granularity_cls in [PerTensor, PerRow]:
                    # Note: Activation and Weights cannot have different granularities
                    granularity_name = "tensor" if granularity_cls is PerTensor else "row"
                    types[f"float8dq_{name}_{granularity_name}"] = partial(
                        float8_dynamic_activation_float8_weight,
                        activation_dtype=dtype,
                        weight_dtype=dtype,
                        granularity=(granularity_cls(), granularity_cls()),
                    )

                return types

            def generate_fpx_quantization_types(bits: int):
                types = {}

                for ebits in range(1, bits):
                    mbits = bits - ebits - 1
                    types[f"fp{bits}_e{ebits}m{mbits}"] = partial(fpx_weight_only, ebits=ebits, mbits=mbits)

                non_sign_bits = bits - 1
                default_ebits = (non_sign_bits + 1) // 2
                default_mbits = non_sign_bits - default_ebits
                types[f"fp{bits}"] = partial(fpx_weight_only, ebits=default_ebits, mbits=default_mbits)

                return types

            INT4_QUANTIZATION_TYPES = {
                # int4 weight + bfloat16/float16 activation
                "int4wo": int4_weight_only,
                "int4_weight_only": int4_weight_only,
                # int4 weight + int8 activation
                "int4dq": int8_dynamic_activation_int4_weight,
                "int8_dynamic_activation_int4_weight": int8_dynamic_activation_int4_weight,
            }

            INT8_QUANTIZATION_TYPES = {
                # int8 weight + bfloat16/float16 activation
                "int8wo": int8_weight_only,
                "int8_weight_only": int8_weight_only,
                # int8 weight + int8 activation
                "int8dq": int8_dynamic_activation_int8_weight,
                "int8_dynamic_activation_int8_weight": int8_dynamic_activation_int8_weight,
            }

            # TODO(aryan): handle torch 2.2/2.3
            FLOATX_QUANTIZATION_TYPES = {
                # float8_e5m2 weight + bfloat16/float16 activation
                "float8wo": partial(float8_weight_only, weight_dtype=torch.float8_e5m2),
                "float8_weight_only": float8_weight_only,
                "float8wo_e5m2": partial(float8_weight_only, weight_dtype=torch.float8_e5m2),
                # float8_e4m3 weight + bfloat16/float16 activation
                "float8wo_e4m3": partial(float8_weight_only, weight_dtype=torch.float8_e4m3fn),
                # float8_e5m2 weight + float8 activation (dynamic)
                "float8dq": float8_dynamic_activation_float8_weight,
                "float8_dynamic_activation_float8_weight": float8_dynamic_activation_float8_weight,
                # ===== Matrix multiplication is not supported in float8_e5m2 so the following errors out.
                # However, changing activation_dtype=torch.float8_e4m3 might work here =====
                # "float8dq_e5m2": partial(
                #     float8_dynamic_activation_float8_weight,
                #     activation_dtype=torch.float8_e5m2,
                #     weight_dtype=torch.float8_e5m2,
                # ),
                # **generate_float8dq_types(torch.float8_e5m2),
                # ===== =====
                # float8_e4m3 weight + float8 activation (dynamic)
                "float8dq_e4m3": partial(
                    float8_dynamic_activation_float8_weight,
                    activation_dtype=torch.float8_e4m3fn,
                    weight_dtype=torch.float8_e4m3fn,
                ),
                **generate_float8dq_types(torch.float8_e4m3fn),
                # float8 weight + float8 activation (static)
                "float8_static_activation_float8_weight": float8_static_activation_float8_weight,
                # For fpx, only x <= 8 is supported by default. Other dtypes can be explored by users directly
                # fpx weight + bfloat16/float16 activation
                **generate_fpx_quantization_types(3),
                **generate_fpx_quantization_types(4),
                **generate_fpx_quantization_types(5),
                **generate_fpx_quantization_types(6),
                **generate_fpx_quantization_types(7),
            }

            UINTX_QUANTIZATION_DTYPES = {
                "uintx_weight_only": uintx_weight_only,
                "uint1wo": partial(uintx_weight_only, dtype=torch.uint1),
                "uint2wo": partial(uintx_weight_only, dtype=torch.uint2),
                "uint3wo": partial(uintx_weight_only, dtype=torch.uint3),
                "uint4wo": partial(uintx_weight_only, dtype=torch.uint4),
                "uint5wo": partial(uintx_weight_only, dtype=torch.uint5),
                "uint6wo": partial(uintx_weight_only, dtype=torch.uint6),
                "uint7wo": partial(uintx_weight_only, dtype=torch.uint7),
                # "uint8wo": partial(uintx_weight_only, dtype=torch.uint8),  # uint8 quantization is not supported
            }

            QUANTIZATION_TYPES = {}
            QUANTIZATION_TYPES.update(INT4_QUANTIZATION_TYPES)
            QUANTIZATION_TYPES.update(INT8_QUANTIZATION_TYPES)
            QUANTIZATION_TYPES.update(UINTX_QUANTIZATION_DTYPES)

            if cls._is_cuda_capability_atleast_8_9():
                QUANTIZATION_TYPES.update(FLOATX_QUANTIZATION_TYPES)

            return QUANTIZATION_TYPES
        else:
            raise ValueError(
                "TorchAoConfig requires torchao to be installed, please install with `pip install torchao`"
            )

    @staticmethod
    def _is_cuda_capability_atleast_8_9() -> bool:
        if not torch.cuda.is_available():
            raise RuntimeError("TorchAO requires a CUDA compatible GPU and installation of PyTorch.")

        major, minor = torch.cuda.get_device_capability()
        if major == 8:
            return minor >= 9
        return major >= 9

    def get_apply_tensor_subclass(self):
        TORCHAO_QUANT_TYPE_METHODS = self._get_torchao_quant_type_to_method()
        return TORCHAO_QUANT_TYPE_METHODS[self.quant_type](**self.quant_type_kwargs)

    def __repr__(self):
        r"""
        Example of how this looks for `TorchAoConfig("uint4wo", group_size=32)`:

        ```
        TorchAoConfig {
            "modules_to_not_convert": null,
            "quant_method": "torchao",
            "quant_type": "uint4wo",
            "quant_type_kwargs": {
                "group_size": 32
            }
        }
        ```
        """
        config_dict = self.to_dict()
        return (
            f"{self.__class__.__name__} {json.dumps(config_dict, indent=2, sort_keys=True, cls=TorchAoJSONEncoder)}\n"
        )


@dataclass
class QuantoConfig(QuantizationConfigMixin):
    """
    This is a wrapper class about all possible attributes and features that you can play with a model that has been
    loaded using `quanto`.

    Args:
        weights_dtype (`str`, *optional*, defaults to `"int8"`):
            The target dtype for the weights after quantization. Supported values are ("float8","int8","int4","int2")
       modules_to_not_convert (`list`, *optional*, default to `None`):
            The list of modules to not quantize, useful for quantizing models that explicitly require to have some
            modules left in their original precision (e.g. Whisper encoder, Llava encoder, Mixtral gate layers).
    """

    def __init__(
        self,
        weights_dtype: str = "int8",
        modules_to_not_convert: Optional[List[str]] = None,
        **kwargs,
    ):
        self.quant_method = QuantizationMethod.QUANTO
        self.weights_dtype = weights_dtype
        self.modules_to_not_convert = modules_to_not_convert

        self.post_init()

    def post_init(self):
        r"""
        Safety checker that arguments are correct
        """
        accepted_weights = ["float8", "int8", "int4", "int2"]
        if self.weights_dtype not in accepted_weights:
            raise ValueError(f"Only support weights in {accepted_weights} but found {self.weights_dtype}")
