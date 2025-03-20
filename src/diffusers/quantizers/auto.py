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
https://github.com/huggingface/transformers/blob/c409cd81777fb27aadc043ed3d8339dbc020fb3b/src/transformers/quantizers/auto.py
"""

import warnings
from typing import Dict, Optional, Union

from .bitsandbytes import BnB4BitDiffusersQuantizer, BnB8BitDiffusersQuantizer
from .gguf import GGUFQuantizer
from .quantization_config import (
    BitsAndBytesConfig,
    GGUFQuantizationConfig,
    QuantizationConfigMixin,
    QuantizationMethod,
    TorchAoConfig,
)
from .torchao import TorchAoHfQuantizer


AUTO_QUANTIZER_MAPPING = {
    "bitsandbytes_4bit": BnB4BitDiffusersQuantizer,
    "bitsandbytes_8bit": BnB8BitDiffusersQuantizer,
    "gguf": GGUFQuantizer,
    "torchao": TorchAoHfQuantizer,
}

AUTO_QUANTIZATION_CONFIG_MAPPING = {
    "bitsandbytes_4bit": BitsAndBytesConfig,
    "bitsandbytes_8bit": BitsAndBytesConfig,
    "gguf": GGUFQuantizationConfig,
    "torchao": TorchAoConfig,
}


class DiffusersAutoQuantizer:
    """
     The auto diffusers quantizer class that takes care of automatically instantiating to the correct
    `DiffusersQuantizer` given the `QuantizationConfig`.
    """

    @classmethod
    def from_dict(cls, quantization_config_dict: Dict):
        quant_method = quantization_config_dict.get("quant_method", None)
        # We need a special care for bnb models to make sure everything is BC ..
        if quantization_config_dict.get("load_in_8bit", False) or quantization_config_dict.get("load_in_4bit", False):
            suffix = "_4bit" if quantization_config_dict.get("load_in_4bit", False) else "_8bit"
            quant_method = QuantizationMethod.BITS_AND_BYTES + suffix
        elif quant_method is None:
            raise ValueError(
                "The model's quantization config from the arguments has no `quant_method` attribute. Make sure that the model has been correctly quantized"
            )

        if quant_method not in AUTO_QUANTIZATION_CONFIG_MAPPING.keys():
            raise ValueError(
                f"Unknown quantization type, got {quant_method} - supported types are:"
                f" {list(AUTO_QUANTIZER_MAPPING.keys())}"
            )

        target_cls = AUTO_QUANTIZATION_CONFIG_MAPPING[quant_method]
        return target_cls.from_dict(quantization_config_dict)

    @classmethod
    def from_config(cls, quantization_config: Union[QuantizationConfigMixin, Dict], **kwargs):
        # Convert it to a QuantizationConfig if the q_config is a dict
        if isinstance(quantization_config, dict):
            quantization_config = cls.from_dict(quantization_config)

        quant_method = quantization_config.quant_method

        # Again, we need a special care for bnb as we have a single quantization config
        # class for both 4-bit and 8-bit quantization
        if quant_method == QuantizationMethod.BITS_AND_BYTES:
            if quantization_config.load_in_8bit:
                quant_method += "_8bit"
            else:
                quant_method += "_4bit"

        if quant_method not in AUTO_QUANTIZER_MAPPING.keys():
            raise ValueError(
                f"Unknown quantization type, got {quant_method} - supported types are:"
                f" {list(AUTO_QUANTIZER_MAPPING.keys())}"
            )

        target_cls = AUTO_QUANTIZER_MAPPING[quant_method]
        return target_cls(quantization_config, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        model_config = cls.load_config(pretrained_model_name_or_path, **kwargs)
        if getattr(model_config, "quantization_config", None) is None:
            raise ValueError(
                f"Did not found a `quantization_config` in {pretrained_model_name_or_path}. Make sure that the model is correctly quantized."
            )
        quantization_config_dict = model_config.quantization_config
        quantization_config = cls.from_dict(quantization_config_dict)
        # Update with potential kwargs that are passed through from_pretrained.
        quantization_config.update(kwargs)

        return cls.from_config(quantization_config)

    @classmethod
    def merge_quantization_configs(
        cls,
        quantization_config: Union[dict, QuantizationConfigMixin],
        quantization_config_from_args: Optional[QuantizationConfigMixin],
    ):
        """
        handles situations where both quantization_config from args and quantization_config from model config are
        present.
        """
        if quantization_config_from_args is not None:
            warning_msg = (
                "You passed `quantization_config` or equivalent parameters to `from_pretrained` but the model you're loading"
                " already has a `quantization_config` attribute. The `quantization_config` from the model will be used."
            )
        else:
            warning_msg = ""

        if isinstance(quantization_config, dict):
            quantization_config = cls.from_dict(quantization_config)

        if warning_msg != "":
            warnings.warn(warning_msg)

        return quantization_config
