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

import torch

from ._common import _ALL_TRANSFORMER_BLOCK_IDENTIFIERS, _ATTENTION_CLASSES, _FEEDFORWARD_CLASSES


def _get_identifiable_transformer_blocks_in_module(module: torch.nn.Module):
    module_list_with_transformer_blocks = []
    for name, submodule in module.named_modules():
        name_endswith_identifier = any(name.endswith(identifier) for identifier in _ALL_TRANSFORMER_BLOCK_IDENTIFIERS)
        is_modulelist = isinstance(submodule, torch.nn.ModuleList)
        if name_endswith_identifier and is_modulelist:
            module_list_with_transformer_blocks.append((name, submodule))
    return module_list_with_transformer_blocks


def _get_identifiable_attention_layers_in_module(module: torch.nn.Module):
    attention_layers = []
    for name, submodule in module.named_modules():
        if isinstance(submodule, _ATTENTION_CLASSES):
            attention_layers.append((name, submodule))
    return attention_layers


def _get_identifiable_feedforward_layers_in_module(module: torch.nn.Module):
    feedforward_layers = []
    for name, submodule in module.named_modules():
        if isinstance(submodule, _FEEDFORWARD_CLASSES):
            feedforward_layers.append((name, submodule))
    return feedforward_layers
