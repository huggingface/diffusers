# Copyright 2024 The HuggingFace Team. All rights reserved.
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

import inspect
from typing import Dict, List, Optional

from ..utils import is_transformers_available, logging
from .auto import DiffusersAutoQuantizer
from .base import DiffusersQuantizer


logger = logging.get_logger(__name__)


class PipelineQuantizationConfig:
    """TODO"""

    def __init__(
        self,
        quant_backend: str = None,
        quant_kwargs: Dict[str, str] = None,
        modules_to_quantize: Optional[List[str]] = None,
        quant_mapping: Dict[str,] = None,
    ):
        self.quant_backend = quant_backend
        # Initialize kwargs to be {} to set to the defaults.
        self.quant_kwargs = quant_kwargs or {}
        self.modules_to_quantize = modules_to_quantize
        self.quant_mapping = quant_mapping

        self.post_init()

    def post_init(self):
        quant_mapping = self.quant_mapping
        self.is_granular = True if quant_mapping is not None else False

        self._validate_init_args()

    def _validate_init_args(self):
        if self.quant_backend and self.quant_mapping:
            raise ValueError("Both `quant_backend` and `quant_mapping` cannot be set.")

        if not self.quant_mapping and not self.quant_backend:
            raise ValueError("Must provide a `quant_backend` when not providing a `quant_mapping`.")

        if not self.quant_kwargs and not self.quant_mapping:
            raise ValueError("Both `quant_kwargs` and `quant_mapping` cannot be None.")

        if self.quant_backend is not None:
            self._validate_init_kwargs_in_backends()

        if self.quant_mapping is not None:
            self._validate_quant_mapping_args()

    def _validate_init_kwargs_in_backends(self):
        quant_backend = self.quant_backend

        self._check_backend_availability(quant_backend)

        quant_config_mapping_transformers, quant_config_mapping_diffusers = self._get_quant_config_list()

        if quant_config_mapping_transformers is not None:
            if quant_backend not in quant_config_mapping_transformers:
                raise ValueError(
                    f"`{quant_backend=}` is not available in `transformers`, available ones are: {list(quant_config_mapping_transformers.keys())}."
                )
            init_kwargs_transformers = inspect.signature(quant_config_mapping_transformers[quant_backend].__init__)
            init_kwargs_transformers = {name for name in init_kwargs_transformers.parameters if name != "self"}
        else:
            init_kwargs_transformers = None

        if quant_backend not in quant_config_mapping_diffusers:
            raise ValueError(
                f"`{quant_backend=}` is not available in `diffusers`, available ones are: {list(quant_config_mapping_diffusers.keys())}."
            )
        init_kwargs_diffusers = inspect.signature(quant_config_mapping_diffusers[quant_backend].__init__)
        init_kwargs_diffusers = {name for name in init_kwargs_diffusers.parameters if name != "self"}

        if init_kwargs_transformers != init_kwargs_diffusers:
            raise ValueError(
                "The signatures of the __init__ methods of the quantization config classes in `diffusers` and `transformers` don't match. "
                f"Please provide a `quant_mapping` instead, in the {self.__class__.__name__} class."
            )

    def _validate_quant_mapping_args(self):
        quant_mapping = self.quant_mapping
        quant_config_mapping_transformers, quant_config_mapping_diffusers = self._get_quant_config_list()

        available_configs_transformers = (
            list(quant_config_mapping_transformers.values()) if quant_config_mapping_transformers else None
        )
        available_configs_diffusers = list(quant_config_mapping_diffusers.values())

        for module_name, config in quant_mapping.items():
            if config not in available_configs_diffusers or (
                available_configs_transformers and config not in available_configs_transformers
            ):
                msg = f"Provided config for {module_name=} could not be found. Available ones for `diffusers` are: {available_configs_diffusers}.)"
                if available_configs_transformers is not None:
                    msg += f" Available ones for `diffusers` are: {available_configs_transformers}."
                raise ValueError(msg)

    def _check_backend_availability(self, quant_backend: str):
        quant_config_mapping_transformers, quant_config_mapping_diffusers = self._get_quant_config_list()

        available_backends_transformers = (
            list(quant_config_mapping_transformers.keys()) if quant_config_mapping_transformers else None
        )
        available_backends_diffusers = list(quant_config_mapping_diffusers.keys())

        if (
            available_backends_transformers and quant_backend not in available_backends_transformers
        ) or quant_backend not in quant_config_mapping_diffusers:
            error_message = f"Provided quant_backend={quant_backend} was not found."
            if available_backends_transformers:
                error_message += f"\nAvailable ones (transformers): {available_backends_transformers}."
            error_message += f"\nAvailable ones (diffusers): {available_backends_diffusers}."
            raise ValueError(error_message)

    def _resolve_quant_config(self, is_diffusers: bool = True, module_name: str = None):
        quant_config_mapping_transformers, quant_config_mapping_diffusers = self._get_quant_config_list()

        quant_mapping = self.quant_mapping
        modules_to_quantize = self.modules_to_quantize

        # Granular case
        if self.is_granular and module_name in quant_mapping:
            logger.debug(f"Initializing quantization config class for {module_name}.")
            config = quant_mapping[module_name]
            return config

        # Global config case
        else:
            should_quantize = False
            # Only quantize the modules requested for.
            if modules_to_quantize and module_name in modules_to_quantize:
                should_quantize = True
            # No specification for `modules_to_quantize` means all modules should be quantized.
            elif not self.is_granular and not modules_to_quantize:
                should_quantize = True

            if should_quantize:
                logger.debug(f"Initializing quantization config class for {module_name}.")
                mapping_to_use = quant_config_mapping_diffusers if is_diffusers else quant_config_mapping_transformers
                quant_config_cls = mapping_to_use[self.quant_backend]
                # If `quant_kwargs` is None we default to initializing with the defaults of `quant_config_cls`.
                quant_kwargs = self.quant_kwargs or {}
                return quant_config_cls(**quant_kwargs)

        # Fallback: no applicable configuration found.
        return None

    def _get_quant_config_list(self):
        if is_transformers_available():
            from transformers.quantizers.auto import (
                AUTO_QUANTIZATION_CONFIG_MAPPING as quant_config_mapping_transformers,
            )
        else:
            quant_config_mapping_transformers = None

        from ..quantizers.auto import AUTO_QUANTIZATION_CONFIG_MAPPING as quant_config_mapping_diffusers

        return quant_config_mapping_transformers, quant_config_mapping_diffusers
