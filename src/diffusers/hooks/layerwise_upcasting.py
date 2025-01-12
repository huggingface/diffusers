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

import re
from typing import List, Type

import torch

from ..utils import get_logger
from .hooks import HookRegistry, ModelHook


logger = get_logger(__name__)  # pylint: disable=invalid-name


# fmt: off
_SUPPORTED_PYTORCH_LAYERS = [
    torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d,
    torch.nn.ConvTranspose1d, torch.nn.ConvTranspose2d, torch.nn.ConvTranspose3d,
    torch.nn.Linear,
]

_DEFAULT_SKIP_MODULES_PATTERN = ["pos_embed", "patch_embed", "norm"]
# fmt: on


class LayerwiseUpcastingHook(ModelHook):
    r"""
    A hook that casts the weights of a module to a high precision dtype for computation, and to a low precision dtype
    for storage. This process may lead to quality loss in the output, but can significantly reduce the memory
    footprint.
    """

    _is_stateful = False

    def __init__(self, storage_dtype: torch.dtype, compute_dtype: torch.dtype, non_blocking: bool) -> None:
        self.storage_dtype = storage_dtype
        self.compute_dtype = compute_dtype
        self.non_blocking = non_blocking

    def initialize_hook(self, module: torch.nn.Module):
        module.to(dtype=self.storage_dtype, non_blocking=self.non_blocking)
        return module

    def pre_forward(self, module: torch.nn.Module, *args, **kwargs):
        module.to(dtype=self.compute_dtype, non_blocking=self.non_blocking)
        return args, kwargs

    def post_forward(self, module: torch.nn.Module, output):
        module.to(dtype=self.storage_dtype, non_blocking=self.non_blocking)
        return output


def apply_layerwise_upcasting(
    module: torch.nn.Module,
    storage_dtype: torch.dtype,
    compute_dtype: torch.dtype,
    skip_modules_pattern: List[str] = _DEFAULT_SKIP_MODULES_PATTERN,
    skip_modules_classes: List[Type[torch.nn.Module]] = [],
    non_blocking: bool = False,
) -> torch.nn.Module:
    r"""
    Applies layerwise upcasting to a given module. The module expected here is a Diffusers ModelMixin but it can be any
    nn.Module using diffusers layers or pytorch primitives.

    Args:
        module (`torch.nn.Module`):
            The module whose leaf modules will be cast to a high precision dtype for computation, and to a low
            precision dtype for storage.
        storage_dtype (`torch.dtype`):
            The dtype to cast the module to before/after the forward pass for storage.
        compute_dtype (`torch.dtype`):
            The dtype to cast the module to during the forward pass for computation.
        skip_modules_pattern (`List[str]`, defaults to `["pos_embed", "patch_embed", "norm"]`):
            A list of patterns to match the names of the modules to skip during the layerwise upcasting process.
        skip_modules_classes (`List[Type[torch.nn.Module]]`, defaults to `[]`):
            A list of module classes to skip during the layerwise upcasting process.
        non_blocking (`bool`, defaults to `False`):
            If `True`, the weight casting operations are non-blocking.
    """
    for name, submodule in module.named_modules():
        if (
            any(re.search(pattern, name) for pattern in skip_modules_pattern)
            or any(isinstance(submodule, module_class) for module_class in skip_modules_classes)
            or not isinstance(submodule, tuple(_SUPPORTED_PYTORCH_LAYERS))
            or len(list(submodule.children())) > 0
        ):
            logger.debug(f'Skipping layerwise upcasting for layer "{name}"')
            continue
        logger.debug(f'Applying layerwise upcasting to layer "{name}"')
        apply_layerwise_upcasting_hook(submodule, storage_dtype, compute_dtype, non_blocking)
    return module


def apply_layerwise_upcasting_hook(
    module: torch.nn.Module, storage_dtype: torch.dtype, compute_dtype: torch.dtype, non_blocking: bool
) -> torch.nn.Module:
    r"""
    Applies a `LayerwiseUpcastingHook` to a given module.

    Args:
        module (`torch.nn.Module`):
            The module to attach the hook to.
        storage_dtype (`torch.dtype`):
            The dtype to cast the module to before the forward pass.
        compute_dtype (`torch.dtype`):
            The dtype to cast the module to during the forward pass.
        non_blocking (`bool`):
            If `True`, the weight casting operations are non-blocking.

    Returns:
        `torch.nn.Module`:
            The same module, with the hook attached (the module is modified in place).
    """
    registry = HookRegistry.check_if_exists_or_initialize(module)
    hook = LayerwiseUpcastingHook(storage_dtype, compute_dtype, non_blocking)
    registry.register_hook(hook, "layerwise_upcasting")
