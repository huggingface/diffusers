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

import re
from typing import Optional, Tuple, Type, Union

import torch

from ..utils import get_logger, is_peft_available, is_peft_version
from ._common import _GO_LC_SUPPORTED_PYTORCH_LAYERS
from .hooks import HookRegistry, ModelHook


logger = get_logger(__name__)  # pylint: disable=invalid-name


# fmt: off
_LAYERWISE_CASTING_HOOK = "layerwise_casting"
_PEFT_AUTOCAST_DISABLE_HOOK = "peft_autocast_disable"
DEFAULT_SKIP_MODULES_PATTERN = ("pos_embed", "patch_embed", "norm", "^proj_in$", "^proj_out$")
# fmt: on

_SHOULD_DISABLE_PEFT_INPUT_AUTOCAST = is_peft_available() and is_peft_version(">", "0.14.0")
if _SHOULD_DISABLE_PEFT_INPUT_AUTOCAST:
    from peft.helpers import disable_input_dtype_casting
    from peft.tuners.tuners_utils import BaseTunerLayer


class LayerwiseCastingHook(ModelHook):
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

    def deinitalize_hook(self, module: torch.nn.Module):
        raise NotImplementedError(
            "LayerwiseCastingHook does not support deinitialization. A model once enabled with layerwise casting will "
            "have casted its weights to a lower precision dtype for storage. Casting this back to the original dtype "
            "will lead to precision loss, which might have an impact on the model's generation quality. The model should "
            "be re-initialized and loaded in the original dtype."
        )

    def pre_forward(self, module: torch.nn.Module, *args, **kwargs):
        module.to(dtype=self.compute_dtype, non_blocking=self.non_blocking)
        return args, kwargs

    def post_forward(self, module: torch.nn.Module, output):
        module.to(dtype=self.storage_dtype, non_blocking=self.non_blocking)
        return output


class PeftInputAutocastDisableHook(ModelHook):
    r"""
    A hook that disables the casting of inputs to the module weight dtype during the forward pass. By default, PEFT
    casts the inputs to the weight dtype of the module, which can lead to precision loss.

    The reasons for needing this are:
        - If we don't add PEFT layers' weight names to `skip_modules_pattern` when applying layerwise casting, the
          inputs will be casted to the, possibly lower precision, storage dtype. Reference:
          https://github.com/huggingface/peft/blob/0facdebf6208139cbd8f3586875acb378813dd97/src/peft/tuners/lora/layer.py#L706
        - We can, on our end, use something like accelerate's `send_to_device` but for dtypes. This way, we can ensure
          that the inputs are casted to the computation dtype correctly always. However, there are two goals we are
          hoping to achieve:
            1. Making forward implementations independent of device/dtype casting operations as much as possible.
            2. Performing inference without losing information from casting to different precisions. With the current
               PEFT implementation (as linked in the reference above), and assuming running layerwise casting inference
               with storage_dtype=torch.float8_e4m3fn and compute_dtype=torch.bfloat16, inputs are cast to
               torch.float8_e4m3fn in the lora layer. We will then upcast back to torch.bfloat16 when we continue the
               forward pass in PEFT linear forward or Diffusers layer forward, with a `send_to_dtype` operation from
               LayerwiseCastingHook. This will be a lossy operation and result in poorer generation quality.
    """

    def new_forward(self, module: torch.nn.Module, *args, **kwargs):
        with disable_input_dtype_casting(module):
            return self.fn_ref.original_forward(*args, **kwargs)


def apply_layerwise_casting(
    module: torch.nn.Module,
    storage_dtype: torch.dtype,
    compute_dtype: torch.dtype,
    skip_modules_pattern: Union[str, Tuple[str, ...]] = "auto",
    skip_modules_classes: Optional[Tuple[Type[torch.nn.Module], ...]] = None,
    non_blocking: bool = False,
) -> None:
    r"""
    Applies layerwise casting to a given module. The module expected here is a Diffusers ModelMixin but it can be any
    nn.Module using diffusers layers or pytorch primitives.

    Example:

    ```python
    >>> import torch
    >>> from diffusers import CogVideoXTransformer3DModel

    >>> transformer = CogVideoXTransformer3DModel.from_pretrained(
    ...     model_id, subfolder="transformer", torch_dtype=torch.bfloat16
    ... )

    >>> apply_layerwise_casting(
    ...     transformer,
    ...     storage_dtype=torch.float8_e4m3fn,
    ...     compute_dtype=torch.bfloat16,
    ...     skip_modules_pattern=["patch_embed", "norm", "proj_out"],
    ...     non_blocking=True,
    ... )
    ```

    Args:
        module (`torch.nn.Module`):
            The module whose leaf modules will be cast to a high precision dtype for computation, and to a low
            precision dtype for storage.
        storage_dtype (`torch.dtype`):
            The dtype to cast the module to before/after the forward pass for storage.
        compute_dtype (`torch.dtype`):
            The dtype to cast the module to during the forward pass for computation.
        skip_modules_pattern (`Tuple[str, ...]`, defaults to `"auto"`):
            A list of patterns to match the names of the modules to skip during the layerwise casting process. If set
            to `"auto"`, the default patterns are used. If set to `None`, no modules are skipped. If set to `None`
            alongside `skip_modules_classes` being `None`, the layerwise casting is applied directly to the module
            instead of its internal submodules.
        skip_modules_classes (`Tuple[Type[torch.nn.Module], ...]`, defaults to `None`):
            A list of module classes to skip during the layerwise casting process.
        non_blocking (`bool`, defaults to `False`):
            If `True`, the weight casting operations are non-blocking.
    """
    if skip_modules_pattern == "auto":
        skip_modules_pattern = DEFAULT_SKIP_MODULES_PATTERN

    if skip_modules_classes is None and skip_modules_pattern is None:
        apply_layerwise_casting_hook(module, storage_dtype, compute_dtype, non_blocking)
        return

    _apply_layerwise_casting(
        module,
        storage_dtype,
        compute_dtype,
        skip_modules_pattern,
        skip_modules_classes,
        non_blocking,
    )
    _disable_peft_input_autocast(module)


def _apply_layerwise_casting(
    module: torch.nn.Module,
    storage_dtype: torch.dtype,
    compute_dtype: torch.dtype,
    skip_modules_pattern: Optional[Tuple[str, ...]] = None,
    skip_modules_classes: Optional[Tuple[Type[torch.nn.Module], ...]] = None,
    non_blocking: bool = False,
    _prefix: str = "",
) -> None:
    should_skip = (skip_modules_classes is not None and isinstance(module, skip_modules_classes)) or (
        skip_modules_pattern is not None and any(re.search(pattern, _prefix) for pattern in skip_modules_pattern)
    )
    if should_skip:
        logger.debug(f'Skipping layerwise casting for layer "{_prefix}"')
        return

    if isinstance(module, _GO_LC_SUPPORTED_PYTORCH_LAYERS):
        logger.debug(f'Applying layerwise casting to layer "{_prefix}"')
        apply_layerwise_casting_hook(module, storage_dtype, compute_dtype, non_blocking)
        return

    for name, submodule in module.named_children():
        layer_name = f"{_prefix}.{name}" if _prefix else name
        _apply_layerwise_casting(
            submodule,
            storage_dtype,
            compute_dtype,
            skip_modules_pattern,
            skip_modules_classes,
            non_blocking,
            _prefix=layer_name,
        )


def apply_layerwise_casting_hook(
    module: torch.nn.Module, storage_dtype: torch.dtype, compute_dtype: torch.dtype, non_blocking: bool
) -> None:
    r"""
    Applies a `LayerwiseCastingHook` to a given module.

    Args:
        module (`torch.nn.Module`):
            The module to attach the hook to.
        storage_dtype (`torch.dtype`):
            The dtype to cast the module to before the forward pass.
        compute_dtype (`torch.dtype`):
            The dtype to cast the module to during the forward pass.
        non_blocking (`bool`):
            If `True`, the weight casting operations are non-blocking.
    """
    registry = HookRegistry.check_if_exists_or_initialize(module)
    hook = LayerwiseCastingHook(storage_dtype, compute_dtype, non_blocking)
    registry.register_hook(hook, _LAYERWISE_CASTING_HOOK)


def _is_layerwise_casting_active(module: torch.nn.Module) -> bool:
    for submodule in module.modules():
        if (
            hasattr(submodule, "_diffusers_hook")
            and submodule._diffusers_hook.get_hook(_LAYERWISE_CASTING_HOOK) is not None
        ):
            return True
    return False


def _disable_peft_input_autocast(module: torch.nn.Module) -> None:
    if not _SHOULD_DISABLE_PEFT_INPUT_AUTOCAST:
        return
    for submodule in module.modules():
        if isinstance(submodule, BaseTunerLayer) and _is_layerwise_casting_active(submodule):
            registry = HookRegistry.check_if_exists_or_initialize(submodule)
            hook = PeftInputAutocastDisableHook()
            registry.register_hook(hook, _PEFT_AUTOCAST_DISABLE_HOOK)
