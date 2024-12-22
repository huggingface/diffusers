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

import functools
import re
from enum import Enum
from typing import Any, Dict, List, Tuple, Type

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


logger = get_logger(__name__)  # pylint: disable=invalid-name


# Reference: https://github.com/huggingface/accelerate/blob/ba7ab93f5e688466ea56908ea3b056fae2f9a023/src/accelerate/hooks.py
class ModelHook:
    r"""
    A hook that contains callbacks to be executed just before and after the forward method of a model. The difference
    with PyTorch existing hooks is that they get passed along the kwargs.
    """

    def init_hook(self, module: torch.nn.Module) -> torch.nn.Module:
        r"""
        Hook that is executed when a model is initialized.

        Args:
            module (`torch.nn.Module`):
                The module attached to this hook.
        """
        return module

    def pre_forward(self, module: torch.nn.Module, *args, **kwargs) -> Tuple[Tuple[Any], Dict[str, Any]]:
        r"""
        Hook that is executed just before the forward method of the model.

        Args:
            module (`torch.nn.Module`):
                The module whose forward pass will be executed just after this event.
            args (`Tuple[Any]`):
                The positional arguments passed to the module.
            kwargs (`Dict[Str, Any]`):
                The keyword arguments passed to the module.
        Returns:
            `Tuple[Tuple[Any], Dict[Str, Any]]`:
                A tuple with the treated `args` and `kwargs`.
        """
        return args, kwargs

    def post_forward(self, module: torch.nn.Module, output: Any) -> Any:
        r"""
        Hook that is executed just after the forward method of the model.

        Args:
            module (`torch.nn.Module`):
                The module whose forward pass been executed just before this event.
            output (`Any`):
                The output of the module.
        Returns:
            `Any`: The processed `output`.
        """
        return output

    def detach_hook(self, module: torch.nn.Module) -> torch.nn.Module:
        r"""
        Hook that is executed when the hook is detached from a module.

        Args:
            module (`torch.nn.Module`):
                The module detached from this hook.
        """
        return module


class SequentialHook(ModelHook):
    r"""A hook that can contain several hooks and iterates through them at each event."""

    def __init__(self, *hooks):
        self.hooks = hooks

    def init_hook(self, module):
        for hook in self.hooks:
            module = hook.init_hook(module)
        return module

    def pre_forward(self, module, *args, **kwargs):
        for hook in self.hooks:
            args, kwargs = hook.pre_forward(module, *args, **kwargs)
        return args, kwargs

    def post_forward(self, module, output):
        for hook in self.hooks:
            output = hook.post_forward(module, output)
        return output

    def detach_hook(self, module):
        for hook in self.hooks:
            module = hook.detach_hook(module)
        return module


class LayerwiseUpcastingHook(ModelHook):
    r"""
    A hook that cast the input tensors and torch.nn.Module to a pre-specified dtype before the forward pass and cast
    the module back to the original dtype after the forward pass. This is useful when a model is loaded/stored in a
    lower precision dtype but performs computation in a higher precision dtype. This process may lead to quality loss
    in the output, but can significantly reduce the memory footprint.
    """

    def __init__(self, storage_dtype: torch.dtype, compute_dtype: torch.dtype) -> None:
        self.storage_dtype = storage_dtype
        self.compute_dtype = compute_dtype

    def init_hook(self, module: torch.nn.Module):
        module.to(dtype=self.storage_dtype)
        return module

    @torch._dynamo.disable(recursive=False)
    def pre_forward(self, module: torch.nn.Module, *args, **kwargs):
        module.to(dtype=self.compute_dtype)
        # How do we account for LongTensor, BoolTensor, etc.?
        # args = tuple(align_maybe_tensor_dtype(arg, self.compute_dtype) for arg in args)
        # kwargs = {k: align_maybe_tensor_dtype(v, self.compute_dtype) for k, v in kwargs.items()}
        return args, kwargs

    @torch._dynamo.disable(recursive=False)
    def post_forward(self, module: torch.nn.Module, output):
        module.to(dtype=self.storage_dtype)
        return output


def add_hook_to_module(module: torch.nn.Module, hook: ModelHook, append: bool = False):
    r"""
    Adds a hook to a given module. This will rewrite the `forward` method of the module to include the hook, to remove
    this behavior and restore the original `forward` method, use `remove_hook_from_module`.

    <Tip warning={true}>

    If the module already contains a hook, this will replace it with the new hook passed by default. To chain two hooks
    together, pass `append=True`, so it chains the current and new hook into an instance of the `SequentialHook` class.

    </Tip>

    Args:
        module (`torch.nn.Module`):
            The module to attach a hook to.
        hook (`ModelHook`):
            The hook to attach.
        append (`bool`, *optional*, defaults to `False`):
            Whether the hook should be chained with an existing one (if module already contains a hook) or not.
    Returns:
        `torch.nn.Module`:
            The same module, with the hook attached (the module is modified in place, so the result can be discarded).
    """
    original_hook = hook

    if append and getattr(module, "_diffusers_hook", None) is not None:
        old_hook = module._diffusers_hook
        remove_hook_from_module(module)
        hook = SequentialHook(old_hook, hook)

    if hasattr(module, "_diffusers_hook") and hasattr(module, "_old_forward"):
        # If we already put some hook on this module, we replace it with the new one.
        old_forward = module._old_forward
    else:
        old_forward = module.forward
        module._old_forward = old_forward

    module = hook.init_hook(module)
    module._diffusers_hook = hook

    if hasattr(original_hook, "new_forward"):
        new_forward = original_hook.new_forward
    else:

        def new_forward(module, *args, **kwargs):
            args, kwargs = module._diffusers_hook.pre_forward(module, *args, **kwargs)
            output = module._old_forward(*args, **kwargs)
            return module._diffusers_hook.post_forward(module, output)

    # Overriding a GraphModuleImpl forward freezes the forward call and later modifications on the graph will fail.
    # Reference: https://pytorch.slack.com/archives/C3PDTEV8E/p1705929610405409
    if "GraphModuleImpl" in str(type(module)):
        module.__class__.forward = functools.update_wrapper(functools.partial(new_forward, module), old_forward)
    else:
        module.forward = functools.update_wrapper(functools.partial(new_forward, module), old_forward)

    return module


def remove_hook_from_module(module: torch.nn.Module, recurse: bool = False) -> torch.nn.Module:
    """
    Removes any hook attached to a module via `add_hook_to_module`.

    Args:
        module (`torch.nn.Module`):
            The module to attach a hook to.
        recurse (`bool`, defaults to `False`):
            Whether to remove the hooks recursively
    Returns:
        `torch.nn.Module`:
            The same module, with the hook detached (the module is modified in place, so the result can be discarded).
    """

    if hasattr(module, "_diffusers_hook"):
        module._diffusers_hook.detach_hook(module)
        delattr(module, "_diffusers_hook")

    if hasattr(module, "_old_forward"):
        # Overriding a GraphModuleImpl forward freezes the forward call and later modifications on the graph will fail.
        # Reference: https://pytorch.slack.com/archives/C3PDTEV8E/p1705929610405409
        if "GraphModuleImpl" in str(type(module)):
            module.__class__.forward = module._old_forward
        else:
            module.forward = module._old_forward
        delattr(module, "_old_forward")

    if recurse:
        for child in module.children():
            remove_hook_from_module(child, recurse)

    return module


def align_maybe_tensor_dtype(input: Any, dtype: torch.dtype) -> Any:
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
        return [align_maybe_tensor_dtype(t, dtype) for t in input]
    if isinstance(input, dict):
        return {k: align_maybe_tensor_dtype(v, dtype) for k, v in input.items()}
    return input


class LayerwiseUpcastingGranualarity(str, Enum):
    r"""
    An enumeration class that defines the granularity of the layerwise upcasting process.

    Granularity can be one of the following:
        - `DIFFUSERS_MODEL`:
            Applies layerwise upcasting to the entire model at the highest diffusers modeling level. This will cast all
            the layers of model to the specified storage dtype. This results in the lowest memory usage for storing the
            model in memory, but may incur significant loss in quality because layers that perform normalization with
            learned parameters (e.g., RMSNorm with elementwise affinity) are cast to a lower dtype, but this is known
            to cause quality issues. This method will not reduce the memory required for the forward pass (which
            comprises of intermediate activations and gradients) of a given modeling component, but may be useful in
            cases like lowering the memory footprint of text encoders in a pipeline.
        - `DIFFUSERS_BLOCK`:
            TODO???
        - `DIFFUSERS_LAYER`:
            Applies layerwise upcasting to the lower-level diffusers layers of the model. This is more granular than
            the `DIFFUSERS_MODEL` level, but less granular than the `PYTORCH_LAYER` level. This method is applied to
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

    DIFFUSERS_MODEL = "diffusers_model"
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

_DEFAULT_PYTORCH_LAYER_SKIP_MODULES_PATTERN = ["pos_embed", "patch_embed", "norm"]
# fmt: on


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


def apply_layerwise_upcasting(
    module: torch.nn.Module,
    storage_dtype: torch.dtype,
    compute_dtype: torch.dtype,
    granularity: LayerwiseUpcastingGranualarity = LayerwiseUpcastingGranualarity.PYTORCH_LAYER,
    skip_modules_pattern: List[str] = [],
    skip_modules_classes: List[Type[torch.nn.Module]] = [],
) -> torch.nn.Module:
    if granularity == LayerwiseUpcastingGranualarity.DIFFUSERS_MODEL:
        return _apply_layerwise_upcasting_diffusers_model(module, storage_dtype, compute_dtype)
    if granularity == LayerwiseUpcastingGranualarity.DIFFUSERS_LAYER:
        return _apply_layerwise_upcasting_diffusers_layer(
            module, storage_dtype, compute_dtype, skip_modules_pattern, skip_modules_classes
        )
    if granularity == LayerwiseUpcastingGranualarity.PYTORCH_LAYER:
        return _apply_layerwise_upcasting_pytorch_layer(
            module, storage_dtype, compute_dtype, skip_modules_pattern, skip_modules_classes
        )


def _apply_layerwise_upcasting_diffusers_model(
    module: torch.nn.Module,
    storage_dtype: torch.dtype,
    compute_dtype: torch.dtype,
) -> torch.nn.Module:
    from .modeling_utils import ModelMixin

    if not isinstance(module, ModelMixin):
        raise ValueError("The input module must be an instance of ModelMixin")

    logger.debug(f'Applying layerwise upcasting to model "{module.__class__.__name__}"')
    apply_layerwise_upcasting_hook(module, storage_dtype, compute_dtype)
    return module


def _apply_layerwise_upcasting_diffusers_layer(
    module: torch.nn.Module,
    storage_dtype: torch.dtype,
    compute_dtype: torch.dtype,
    skip_modules_pattern: List[str] = _DEFAULT_PYTORCH_LAYER_SKIP_MODULES_PATTERN,
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
    skip_modules_pattern: List[str] = _DEFAULT_PYTORCH_LAYER_SKIP_MODULES_PATTERN,
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
