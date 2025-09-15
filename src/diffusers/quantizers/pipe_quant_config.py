# Copyright 2025 The HuggingFace Team. All rights reserved.
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
from typing import Dict, List, Optional, Union

from ..utils import is_transformers_available, logging
from .quantization_config import QuantizationConfigMixin as DiffQuantConfigMixin


try:
    from transformers.utils.quantization_config import QuantizationConfigMixin as TransformersQuantConfigMixin
except ImportError:

    class TransformersQuantConfigMixin:
        pass


logger = logging.get_logger(__name__)


class PipelineQuantizationConfig:
    """
    Configuration class to be used when applying quantization on-the-fly to [`~DiffusionPipeline.from_pretrained`].

    Args:
        quant_backend (`str`): Quantization backend to be used. When using this option, we assume that the backend
            is available to both `diffusers` and `transformers`.
        quant_kwargs (`dict`): Params to initialize the quantization backend class.
        components_to_quantize (`list`): Components of a pipeline to be quantized.
        quant_mapping (`dict`): Mapping defining the quantization specs to be used for the pipeline
            components. When using this argument, users are not expected to provide `quant_backend`, `quant_kawargs`,
            and `components_to_quantize`.
    """

    def __init__(
        self,
        quant_backend: str = None,
        quant_kwargs: Dict[str, Union[str, float, int, dict]] = None,
        components_to_quantize: Optional[Union[List[str], str]] = None,
        quant_mapping: Dict[str, Union[DiffQuantConfigMixin, "TransformersQuantConfigMixin"]] = None,
    ):
        self.quant_backend = quant_backend
        # Initialize kwargs to be {} to set to the defaults.
        self.quant_kwargs = quant_kwargs or {}
        if components_to_quantize:
            if isinstance(components_to_quantize, str):
                components_to_quantize = [components_to_quantize]
        self.components_to_quantize = components_to_quantize
        self.quant_mapping = quant_mapping
        self.config_mapping = {}  # book-keeping Example: `{module_name: quant_config}`
        self.post_init()

    def post_init(self):
        quant_mapping = self.quant_mapping
        self.is_granular = True if quant_mapping is not None else False

        self._validate_init_args()

    def _validate_init_args(self):
        if self.quant_backend and self.quant_mapping:
            raise ValueError("Both `quant_backend` and `quant_mapping` cannot be specified at the same time.")

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
            init_kwargs_transformers = inspect.signature(quant_config_mapping_transformers[quant_backend].__init__)
            init_kwargs_transformers = {name for name in init_kwargs_transformers.parameters if name != "self"}
        else:
            init_kwargs_transformers = None

        init_kwargs_diffusers = inspect.signature(quant_config_mapping_diffusers[quant_backend].__init__)
        init_kwargs_diffusers = {name for name in init_kwargs_diffusers.parameters if name != "self"}

        if init_kwargs_transformers != init_kwargs_diffusers:
            raise ValueError(
                "The signatures of the __init__ methods of the quantization config classes in `diffusers` and `transformers` don't match. "
                f"Please provide a `quant_mapping` instead, in the {self.__class__.__name__} class. Refer to [the docs](https://huggingface.co/docs/diffusers/main/en/quantization/overview#pipeline-level-quantization) to learn more about how "
                "this mapping would look like."
            )

    def _validate_quant_mapping_args(self):
        quant_mapping = self.quant_mapping
        transformers_map, diffusers_map = self._get_quant_config_list()

        available_transformers = list(transformers_map.values()) if transformers_map else None
        available_diffusers = list(diffusers_map.values())

        for module_name, config in quant_mapping.items():
            if any(isinstance(config, cfg) for cfg in available_diffusers):
                continue

            if available_transformers and any(isinstance(config, cfg) for cfg in available_transformers):
                continue

            if available_transformers:
                raise ValueError(
                    f"Provided config for module_name={module_name} could not be found. "
                    f"Available diffusers configs: {available_diffusers}; "
                    f"Available transformers configs: {available_transformers}."
                )
            else:
                raise ValueError(
                    f"Provided config for module_name={module_name} could not be found. "
                    f"Available diffusers configs: {available_diffusers}."
                )

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
        components_to_quantize = self.components_to_quantize

        # Granular case
        if self.is_granular and module_name in quant_mapping:
            logger.debug(f"Initializing quantization config class for {module_name}.")
            config = quant_mapping[module_name]
            self.config_mapping.update({module_name: config})
            return config

        # Global config case
        else:
            should_quantize = False
            # Only quantize the modules requested for.
            if components_to_quantize and module_name in components_to_quantize:
                should_quantize = True
            # No specification for `components_to_quantize` means all modules should be quantized.
            elif not self.is_granular and not components_to_quantize:
                should_quantize = True

            if should_quantize:
                logger.debug(f"Initializing quantization config class for {module_name}.")
                mapping_to_use = quant_config_mapping_diffusers if is_diffusers else quant_config_mapping_transformers
                quant_config_cls = mapping_to_use[self.quant_backend]
                quant_kwargs = self.quant_kwargs
                quant_obj = quant_config_cls(**quant_kwargs)
                self.config_mapping.update({module_name: quant_obj})
                return quant_obj

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

    def __repr__(self):
        out = ""
        config_mapping = dict(sorted(self.config_mapping.copy().items()))
        for module_name, config in config_mapping.items():
            out += f"{module_name} {config}"
        return out
