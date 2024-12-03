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
"""
PEFT utilities: Utilities related to peft library
"""

import collections
import importlib
from typing import Optional

from packaging import version

from .import_utils import is_peft_available, is_torch_available


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

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
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


def unscale_lora_layers(model, weight: Optional[float] = None):
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


def get_peft_kwargs(rank_dict, network_alpha_dict, peft_state_dict, is_unet=True):
    rank_pattern = {}
    alpha_pattern = {}
    r = lora_alpha = list(rank_dict.values())[0]

    if len(set(rank_dict.values())) > 1:
        # get the rank occuring the most number of times
        r = collections.Counter(rank_dict.values()).most_common()[0][0]

        # for modules with rank different from the most occuring rank, add it to the `rank_pattern`
        rank_pattern = dict(filter(lambda x: x[1] != r, rank_dict.items()))
        rank_pattern = {k.split(".lora_B.")[0]: v for k, v in rank_pattern.items()}

    if network_alpha_dict is not None and len(network_alpha_dict) > 0:
        if len(set(network_alpha_dict.values())) > 1:
            # get the alpha occuring the most number of times
            lora_alpha = collections.Counter(network_alpha_dict.values()).most_common()[0][0]

            # for modules with alpha different from the most occuring alpha, add it to the `alpha_pattern`
            alpha_pattern = dict(filter(lambda x: x[1] != lora_alpha, network_alpha_dict.items()))
            if is_unet:
                alpha_pattern = {
                    ".".join(k.split(".lora_A.")[0].split(".")).replace(".alpha", ""): v
                    for k, v in alpha_pattern.items()
                }
            else:
                alpha_pattern = {".".join(k.split(".down.")[0].split(".")[:-1]): v for k, v in alpha_pattern.items()}
        else:
            lora_alpha = set(network_alpha_dict.values()).pop()

    # layer names without the Diffusers specific
    target_modules = list({name.split(".lora")[0] for name in peft_state_dict.keys()})
    use_dora = any("lora_magnitude_vector" in k for k in peft_state_dict)

    lora_config_kwargs = {
        "r": r,
        "lora_alpha": lora_alpha,
        "rank_pattern": rank_pattern,
        "alpha_pattern": alpha_pattern,
        "target_modules": target_modules,
        "use_dora": use_dora,
    }
    return lora_config_kwargs


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

    # iterate over each adapter, make it active and set the corresponding scaling weight
    for adapter_name, weight in zip(adapter_names, weights):
        for module_name, module in model.named_modules():
            if isinstance(module, BaseTunerLayer):
                # For backward compatbility with previous PEFT versions
                if hasattr(module, "set_adapter"):
                    module.set_adapter(adapter_name)
                else:
                    module.active_adapter = adapter_name
                module.set_scale(adapter_name, get_module_weight(weight, module_name))

    # set multiple active adapters
    for module in model.modules():
        if isinstance(module, BaseTunerLayer):
            # For backward compatbility with previous PEFT versions
            if hasattr(module, "set_adapter"):
                module.set_adapter(adapter_names)
            else:
                module.active_adapter = adapter_names


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
