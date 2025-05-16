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
from dataclasses import dataclass
from typing import Dict, Type


@dataclass
class TransformerBlockMetadata:
    return_hidden_states_index: int = None
    return_encoder_hidden_states_index: int = None

    _cls: Type = None
    _cached_parameter_indices: Dict[str, int] = None

    def _get_parameter_from_args_kwargs(self, identifier: str, args=(), kwargs=None):
        kwargs = kwargs or {}
        if identifier in kwargs:
            return kwargs[identifier]
        if self._cached_parameter_indices is not None:
            return args[self._cached_parameter_indices[identifier]]
        if self._cls is None:
            raise ValueError("Model class is not set for metadata.")
        parameters = list(inspect.signature(self._cls.forward).parameters.keys())
        parameters = parameters[1:]  # skip `self`
        self._cached_parameter_indices = {param: i for i, param in enumerate(parameters)}
        if identifier not in self._cached_parameter_indices:
            raise ValueError(f"Parameter '{identifier}' not found in function signature but was requested.")
        index = self._cached_parameter_indices[identifier]
        if index >= len(args):
            raise ValueError(f"Expected {index} arguments but got {len(args)}.")
        return args[index]


def register_transformer_block(metadata: TransformerBlockMetadata):
    def inner(model_class: Type):
        metadata._cls = model_class
        model_class._diffusers_transformer_block_metadata = metadata
        return model_class

    return inner
