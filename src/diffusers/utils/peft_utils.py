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
"""
PEFT utilities: Utilities related to peft library
"""

import collections
import functools
import importlib

from packaging import version

from . import logging
from .import_utils import is_peft_available, is_torch_available
from .torch_utils import empty_device_cache


logger = logging.get_logger(__name__)

if is_torch_available():
    import torch


def recurse_remove_peft_layers(model):
    r"""
    Recursively replace all instances of `LoraLayer` with corresponding new layers in `model`.
    """
    from peft.tuners.tuners_utils import BaseTunerLayer

    has_base_layer_pattern = False
    for module in model.modules():
        if isinstance(module, BaseTunerLayer):
            has_base_layer_pattern = hasattr(module, "base_layer")
            break

    if has_base_layer_pattern:
        from peft.utils import _get_submodules

        key_list = [key for key, _ in model.named_modules() if "lora" not in key]
        for key in key_list:
            try:
                parent, target, target_name = _get_submodules(model, key)
            except AttributeError:
                continue
            if hasattr(target, "base_layer"):
                setattr(parent, target_name, target.get_base_layer())
    else:
        # This is for backwards compatibility with PEFT <= 0.6.2.
        # TODO can be removed once that PEFT version is no longer supported.
        from peft.tuners.lora import LoraLayer

        for name, module in model.named_children():
            if len(list(module.children())) > 0:
                ## compound module, go inside it
                recurse_remove_peft_layers(module)

            module_replaced = False

            if isinstance(module, LoraLayer) and isinstance(module, torch.nn.Linear):
                new_module = torch.nn.Linear(
                    module.in_features,
                    module.out_features,
                    bias=module.bias is not None,
                ).to(module.weight.device)
                new_module.weight = module.weight
                if module.bias is not None:
                    new_module.bias = module.bias

                module_replaced = True
            elif isinstance(module, LoraLayer) and isinstance(module, torch.nn.Conv2d):
                new_module = torch.nn.Conv2d(
                    module.in_channels,
                    module.out_channels,
                    module.kernel_size,
                    module.stride,
                    module.padding,
                    module.dilation,
                    module.groups,
                ).to(module.weight.device)

                new_module.weight = module.weight
                if module.bias is not None:
                    new_module.bias = module.bias

                module_replaced = True

            if module_replaced:
                setattr(model, name, new_module)
                del module

                empty_device_cache()
    return model


def scale_lora_layers(model, weight):
    """
    Adjust the weightage given to the LoRA layers of the model.

    Args:
        model (`torch.nn.Module`):
            The model to scale.
        weight (`float`):
            The weight to be given to the LoRA layers.
    """
    from peft.tuners.tuners_utils import BaseTunerLayer

    if weight == 1.0:
        return

    for module in model.modules():
        if isinstance(module, BaseTunerLayer):
            module.scale_layer(weight)


def unscale_lora_layers(model, weight: float | None = None):
    """
    Removes the previously passed weight given to the LoRA layers of the model.

    Args:
        model (`torch.nn.Module`):
            The model to scale.
        weight (`float`, *optional*):
            The weight to be given to the LoRA layers. If no scale is passed the scale of the lora layer will be
            re-initialized to the correct value. If 0.0 is passed, we will re-initialize the scale with the correct
            value.
    """
    from peft.tuners.tuners_utils import BaseTunerLayer

    if weight is None or weight == 1.0:
        return

    for module in model.modules():
        if isinstance(module, BaseTunerLayer):
            if weight != 0:
                module.unscale_layer(weight)
            else:
                for adapter_name in module.active_adapters:
                    # if weight == 0 unscale should re-set the scale to the original value.
                    module.set_scale(adapter_name, 1.0)


def get_adapter_name(model):
    from peft.tuners.tuners_utils import BaseTunerLayer

    for module in model.modules():
        if isinstance(module, BaseTunerLayer):
            return f"default_{len(module.r)}"
    return "default_0"


def set_adapter_layers(model, enabled=True):
    from peft.tuners.tuners_utils import BaseTunerLayer

    for module in model.modules():
        if isinstance(module, BaseTunerLayer):
            # The recent version of PEFT needs to call `enable_adapters` instead
            if hasattr(module, "enable_adapters"):
                module.enable_adapters(enabled=enabled)
            else:
                module.disable_adapters = not enabled


def delete_adapter_layers(model, adapter_name):
    from peft.tuners.tuners_utils import BaseTunerLayer

    for module in model.modules():
        if isinstance(module, BaseTunerLayer):
            if hasattr(module, "delete_adapter"):
                module.delete_adapter(adapter_name)
            else:
                raise ValueError(
                    "The version of PEFT you are using is not compatible, please use a version that is greater than 0.6.1"
                )

    # For transformers integration - we need to pop the adapter from the config
    if getattr(model, "_hf_peft_config_loaded", False) and hasattr(model, "peft_config"):
        model.peft_config.pop(adapter_name, None)
        # In case all adapters are deleted, we need to delete the config
        # and make sure to set the flag to False
        if len(model.peft_config) == 0:
            del model.peft_config
            model._hf_peft_config_loaded = None


def set_weights_and_activate_adapters(model, adapter_names, weights):
    from peft.tuners.tuners_utils import BaseTunerLayer

    def get_module_weight(weight_for_adapter, module_name):
        if not isinstance(weight_for_adapter, dict):
            # If weight_for_adapter is a single number, always return it.
            return weight_for_adapter

        for layer_name, weight_ in weight_for_adapter.items():
            if layer_name in module_name:
                return weight_

        parts = module_name.split(".")
        # e.g. key = "down_blocks.1.attentions.0"
        key = f"{parts[0]}.{parts[1]}.attentions.{parts[3]}"
        block_weight = weight_for_adapter.get(key, 1.0)

        return block_weight

    for module_name, module in model.named_modules():
        if isinstance(module, BaseTunerLayer):
            # For backward compatibility with previous PEFT versions, set multiple active adapters
            if hasattr(module, "set_adapter"):
                module.set_adapter(adapter_names)
            else:
                module.active_adapter = adapter_names

            # Set the scaling weight for each adapter for this module
            for adapter_name, weight in zip(adapter_names, weights):
                module.set_scale(adapter_name, get_module_weight(weight, module_name))


def apply_lora_scale(kwargs_name: str = "joint_attention_kwargs"):
    """
    Decorator to automatically handle LoRA layer scaling/unscaling in forward methods.

    This decorator extracts the `lora_scale` from the specified kwargs parameter, applies scaling before the forward
    pass, and ensures unscaling happens after, even if an exception occurs.

    Args:
        kwargs_name (`str`, defaults to `"joint_attention_kwargs"`):
            The name of the keyword argument that contains the LoRA scale. Common values include
            "joint_attention_kwargs", "attention_kwargs", "cross_attention_kwargs", etc.
    """

    def decorator(forward_fn):
        @functools.wraps(forward_fn)
        def wrapper(self, *args, **kwargs):
            from . import USE_PEFT_BACKEND

            lora_scale = 1.0
            attention_kwargs = kwargs.get(kwargs_name)

            if attention_kwargs is not None:
                attention_kwargs = attention_kwargs.copy()
                kwargs[kwargs_name] = attention_kwargs
                lora_scale = attention_kwargs.pop("scale", 1.0)

                if not USE_PEFT_BACKEND and lora_scale != 1.0:
                    logger.warning(
                        f"Passing `scale` via `{kwargs_name}` when not using the PEFT backend is ineffective."
                    )

            # Apply LoRA scaling if using PEFT backend
            if USE_PEFT_BACKEND:
                scale_lora_layers(self, lora_scale)

            try:
                # Execute the forward pass
                result = forward_fn(self, *args, **kwargs)
                return result
            finally:
                # Always unscale, even if forward pass raises an exception
                if USE_PEFT_BACKEND:
                    unscale_lora_layers(self, lora_scale)

        return wrapper

    return decorator


def check_peft_version(min_version: str) -> None:
    r"""
    Checks if the version of PEFT is compatible.

    Args:
        version (`str`):
            The version of PEFT to check against.
    """
    if not is_peft_available():
        raise ValueError("PEFT is not installed. Please install it with `pip install peft`")

    is_peft_version_compatible = version.parse(importlib.metadata.version("peft")) > version.parse(min_version)

    if not is_peft_version_compatible:
        raise ValueError(
            f"The version of PEFT you are using is not compatible, please use a version that is greater"
            f" than {min_version}"
        )
