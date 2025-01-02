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
from enum import Enum
from typing import Any, List, Type

import torch

from ..utils import get_logger
from .attention import FeedForward, LuminaFeedForward
from .embeddings import (
    AttentionPooling,
    CogVideoXPatchEmbed,
    CogView3PlusPatchEmbed,
    GLIGENTextBoundingboxProjection,
    HunyuanDiTAttentionPool,
    LuminaPatchEmbed,
    MochiAttentionPool,
    PixArtAlphaTextProjection,
    TimestepEmbedding,
)
from .hooks import ModelHook, add_hook_to_module


logger = get_logger(__name__)  # pylint: disable=invalid-name


class LayerwiseUpcastingHook(ModelHook):
    r"""
    A hook that cast the input tensors and torch.nn.Module to a pre-specified dtype before the forward pass and cast
    the module back to the original dtype after the forward pass. This is useful when a model is loaded/stored in a
    lower precision dtype but performs computation in a higher precision dtype. This process may lead to quality loss
    in the output, but can significantly reduce the memory footprint.
    """

    _is_stateful = False

    def __init__(self, storage_dtype: torch.dtype, compute_dtype: torch.dtype) -> None:
        self.storage_dtype = storage_dtype
        self.compute_dtype = compute_dtype

    def init_hook(self, module: torch.nn.Module):
        module.to(dtype=self.storage_dtype)
        return module

    def pre_forward(self, module: torch.nn.Module, *args, **kwargs):
        module.to(dtype=self.compute_dtype)
        # How do we account for LongTensor, BoolTensor, etc.?
        # args = tuple(_align_maybe_tensor_dtype(arg, self.compute_dtype) for arg in args)
        # kwargs = {k: _align_maybe_tensor_dtype(v, self.compute_dtype) for k, v in kwargs.items()}
        return args, kwargs

    def post_forward(self, module: torch.nn.Module, output):
        module.to(dtype=self.storage_dtype)
        return output


class LayerwiseUpcastingGranularity(str, Enum):
    r"""
    An enumeration class that defines the granularity of the layerwise upcasting process.

    Granularity can be one of the following:
        - `DIFFUSERS_LAYER`:
            Applies layerwise upcasting to the lower-level diffusers layers of the model. This method is applied to
            only those layers that are a group of linear layers, while excluding precision-critical layers like
            modulation and normalization layers.
        - `PYTORCH_LAYER`:
            Applies layerwise upcasting to lower-level PyTorch primitive layers of the model. This is the most granular
            level of layerwise upcasting. The memory footprint for inference and training is greatly reduced, while
            also ensuring important operations like normalization with learned parameters remain unaffected from the
            downcasting/upcasting process, by default. As not all parameters are casted to lower precision, the memory
            footprint for storing the model may be slightly higher than the alternatives. This method causes the
            highest number of casting operations, which may contribute to a slight increase in the overall computation
            time.

        Note: try and ensure that precision-critical layers like modulation and normalization layers are not casted to
        lower precision, as this may lead to significant quality loss.
    """

    DIFFUSERS_LAYER = "diffusers_layer"
    PYTORCH_LAYER = "pytorch_layer"


# fmt: off
_SUPPORTED_DIFFUSERS_LAYERS = [
    AttentionPooling, MochiAttentionPool, HunyuanDiTAttentionPool,
    CogVideoXPatchEmbed, CogView3PlusPatchEmbed, LuminaPatchEmbed,
    TimestepEmbedding,  GLIGENTextBoundingboxProjection, PixArtAlphaTextProjection,
    FeedForward, LuminaFeedForward,
]

_SUPPORTED_PYTORCH_LAYERS = [
    torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d,
    torch.nn.ConvTranspose1d, torch.nn.ConvTranspose2d, torch.nn.ConvTranspose3d,
    torch.nn.Linear,
]

_DEFAULT_SKIP_MODULES_PATTERN = ["pos_embed", "patch_embed", "norm"]
# fmt: on


def apply_layerwise_upcasting(
    module: torch.nn.Module,
    storage_dtype: torch.dtype,
    compute_dtype: torch.dtype,
    granularity: LayerwiseUpcastingGranularity = LayerwiseUpcastingGranularity.PYTORCH_LAYER,
    skip_modules_pattern: List[str] = _DEFAULT_SKIP_MODULES_PATTERN,
    skip_modules_classes: List[Type[torch.nn.Module]] = [],
) -> torch.nn.Module:
    r"""
    Applies layerwise upcasting to a given module. The module expected here is a Diffusers ModelMixin but it can be any
    nn.Module using diffusers layers or pytorch primitives.

    Args:
        module (`torch.nn.Module`):
            The module to attach the hook to.
        storage_dtype (`torch.dtype`):
            The dtype to cast the module to before the forward pass.
        compute_dtype (`torch.dtype`):
            The dtype to cast the module to during the forward pass.
        granularity (`LayerwiseUpcastingGranularity`, *optional*, defaults to `LayerwiseUpcastingGranularity.PYTORCH_LAYER`):
            The granularity of the layerwise upcasting process.
        skip_modules_pattern (`List[str]`, defaults to `["pos_embed", "patch_embed", "norm"]`):
            A list of patterns to match the names of the modules to skip during the layerwise upcasting process.
        skip_modules_classes (`List[Type[torch.nn.Module]]`, defaults to `[]`):
            A list of module classes to skip during the layerwise upcasting process.
    """
    if granularity == LayerwiseUpcastingGranularity.DIFFUSERS_LAYER:
        return _apply_layerwise_upcasting_diffusers_layer(
            module, storage_dtype, compute_dtype, skip_modules_pattern, skip_modules_classes
        )
    if granularity == LayerwiseUpcastingGranularity.PYTORCH_LAYER:
        return _apply_layerwise_upcasting_pytorch_layer(
            module, storage_dtype, compute_dtype, skip_modules_pattern, skip_modules_classes
        )


def apply_layerwise_upcasting_hook(
    module: torch.nn.Module, storage_dtype: torch.dtype, compute_dtype: torch.dtype
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

    Returns:
        `torch.nn.Module`:
            The same module, with the hook attached (the module is modified in place, so the result can be discarded).
    """
    hook = LayerwiseUpcastingHook(storage_dtype, compute_dtype)
    return add_hook_to_module(module, hook, append=True)


def _apply_layerwise_upcasting_diffusers_layer(
    module: torch.nn.Module,
    storage_dtype: torch.dtype,
    compute_dtype: torch.dtype,
    skip_modules_pattern: List[str] = _DEFAULT_SKIP_MODULES_PATTERN,
    skip_modules_classes: List[Type[torch.nn.Module]] = [],
) -> torch.nn.Module:
    for name, submodule in module.named_modules():
        if (
            any(re.search(pattern, name) for pattern in skip_modules_pattern)
            or any(isinstance(submodule, module_class) for module_class in skip_modules_classes)
            or not isinstance(submodule, tuple(_SUPPORTED_DIFFUSERS_LAYERS))
        ):
            logger.debug(f'Skipping layerwise upcasting for layer "{name}"')
            continue
        logger.debug(f'Applying layerwise upcasting to layer "{name}"')
        apply_layerwise_upcasting_hook(submodule, storage_dtype, compute_dtype)
    return module


def _apply_layerwise_upcasting_pytorch_layer(
    module: torch.nn.Module,
    storage_dtype: torch.dtype,
    compute_dtype: torch.dtype,
    skip_modules_pattern: List[str] = _DEFAULT_SKIP_MODULES_PATTERN,
    skip_modules_classes: List[Type[torch.nn.Module]] = [],
) -> torch.nn.Module:
    for name, submodule in module.named_modules():
        if (
            any(re.search(pattern, name) for pattern in skip_modules_pattern)
            or any(isinstance(submodule, module_class) for module_class in skip_modules_classes)
            or not isinstance(submodule, tuple(_SUPPORTED_PYTORCH_LAYERS))
        ):
            logger.debug(f'Skipping layerwise upcasting for layer "{name}"')
            continue
        logger.debug(f'Applying layerwise upcasting to layer "{name}"')
        apply_layerwise_upcasting_hook(submodule, storage_dtype, compute_dtype)
    return module


def _align_maybe_tensor_dtype(input: Any, dtype: torch.dtype) -> Any:
    r"""
    Aligns the dtype of a tensor or a list of tensors to a given dtype.

    Args:
        input (`Any`):
            The input tensor, list of tensors, or dictionary of tensors to align. If the input is neither of these
            types, it will be returned as is.
        dtype (`torch.dtype`):
            The dtype to align the tensor(s) to.

    Returns:
        `Any`:
            The tensor or list of tensors aligned to the given dtype.
    """
    if isinstance(input, torch.Tensor):
        return input.to(dtype=dtype)
    if isinstance(input, (list, tuple)):
        return [_align_maybe_tensor_dtype(t, dtype) for t in input]
    if isinstance(input, dict):
        return {k: _align_maybe_tensor_dtype(v, dtype) for k, v in input.items()}
    return input
