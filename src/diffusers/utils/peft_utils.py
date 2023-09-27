# Copyright 2023 The HuggingFace Team. All rights reserved.
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

from .import_utils import is_torch_available


MIN_PEFT_VERSION = "0.5.0"


def recurse_remove_peft_layers(model):
    if is_torch_available():
        import torch

    r"""
    Recursively replace all instances of `LoraLayer` with corresponding new layers in `model`.
    """
    from peft.tuners.lora import LoraLayer

    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            ## compound module, go inside it
            recurse_remove_peft_layers(module)

        module_replaced = False

        if isinstance(module, LoraLayer) and isinstance(module, torch.nn.Linear):
            new_module = torch.nn.Linear(module.in_features, module.out_features, bias=module.bias is not None).to(
                module.weight.device
            )
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
                module.bias,
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


def scale_peft_layers(model, scale: float = None):
    r"""
    Scale peft layers - Loops over the modules of the model and scale the layers that are of type `BaseTunerLayer`. We
    also store the original scale factor in case we multiply it by zero.

    Args:
        model (`torch.nn.Module`):
            The model to scale.
        scale (`float`, *optional*):
            The scale factor to use.
    """
    from peft.tuners.tuners_utils import BaseTunerLayer

    if scale is not None and scale != 1.0:
        for module in model.modules():
            if isinstance(module, BaseTunerLayer):
                # To deal with previous PEFT versions
                active_adapters = module.active_adapter
                if isinstance(active_adapters, str):
                    active_adapters = [active_adapters]

                for active_adapter in active_adapters:
                    original_scale = module.scaling[active_adapter]

                    # Store the previous scale in case we multiply it by zero
                    if "_hf_peft_original_scales" not in module.scaling:
                        module.scaling["_hf_peft_original_scales"] = {active_adapter: original_scale}
                    else:
                        module.scaling["_hf_peft_original_scales"][active_adapter] = original_scale

                    module.scaling[active_adapter] *= scale


def unscale_peft_layers(model, scale: float = None):
    r"""
    Un-scale peft layers - in case the modules has been zero-ed by a zero factor we retrieve the previous scale and
    restore it. Otherwise, assuming the user uses the same scale factor, we just divide by the scale factor.

    Args:
        model (`torch.nn.Module`):
            The model to unscale.
        scale (`float`, *optional*):
            The scale factor to use. If 0.0 is passed, we retrieve the original scale factor. In order to retrieve the
            original factor the user needs first to call `scale_peft_layers` with the same scale factor.
    """
    from peft.tuners.tuners_utils import BaseTunerLayer

    if scale is not None and scale != 1.0 and scale != 0.0:
        for module in model.modules():
            if isinstance(module, BaseTunerLayer):
                # To deal with previous PEFT versions
                active_adapters = module.active_adapter
                if isinstance(active_adapters, str):
                    active_adapters = [active_adapters]

                for active_adapter in active_adapters:
                    module.scaling[active_adapter] /= scale
    elif scale is not None and scale == 0.0:
        for module in model.modules():
            if isinstance(module, BaseTunerLayer):
                if "_hf_peft_original_scales" not in module.scaling:
                    raise ValueError(
                        "The layer has not been scaled, cannot unscale it - please call first `scale_peft_layers`"
                    )
                # To deal with previous PEFT versions
                active_adapters = module.active_adapter
                if isinstance(active_adapters, str):
                    active_adapters = [active_adapters]

                for active_adapter in active_adapters:
                    original_scale = module.scaling["_hf_peft_original_scales"][active_adapter]
                    module.scaling[active_adapter] = original_scale

                    del module.scaling["_hf_peft_original_scales"][active_adapter]

                # Clean up ..
                if len(module.scaling["_hf_peft_original_scales"]) == 0:
                    del module.scaling["_hf_peft_original_scales"]


def get_peft_kwargs(rank_dict, network_alpha_dict, peft_state_dict):
    rank_pattern = {}
    alpha_pattern = {}
    r = lora_alpha = list(rank_dict.values())[0]
    if len(set(rank_dict.values())) > 1:
        # get the rank occuring the most number of times
        r = collections.Counter(rank_dict.values()).most_common()[0][0]

        # for modules with rank different from the most occuring rank, add it to the `rank_pattern`
        rank_pattern = dict(filter(lambda x: x[1] != r, rank_dict.items()))
        rank_pattern = {k.split(".lora_B.")[0]: v for k, v in rank_pattern.items()}

    if network_alpha_dict is not None and len(set(network_alpha_dict.values())) > 1:
        # get the alpha occuring the most number of times
        lora_alpha = collections.Counter(network_alpha_dict.values()).most_common()[0][0]

        # for modules with alpha different from the most occuring alpha, add it to the `alpha_pattern`
        alpha_pattern = dict(filter(lambda x: x[1] != lora_alpha, network_alpha_dict.items()))
        alpha_pattern = {".".join(k.split(".down.")[0].split(".")[:-1]): v for k, v in alpha_pattern.items()}

    # layer names without the Diffusers specific
    target_modules = list({name.split(".lora")[0] for name in peft_state_dict.keys()})

    lora_config_kwargs = {
        "r": r,
        "lora_alpha": lora_alpha,
        "rank_pattern": rank_pattern,
        "alpha_pattern": alpha_pattern,
        "target_modules": target_modules,
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
                module.enable_adapters(enabled=False)
            else:
                module.disable_adapters = True


def set_weights_and_activate_adapters(model, adapter_names, weights):
    from peft.tuners.tuners_utils import BaseTunerLayer

    # iterate over each adapter, make it active and set the corresponding scaling weight
    for adapter_name, weight in zip(adapter_names, weights):
        for module in model.modules():
            if isinstance(module, BaseTunerLayer):
                # For backward compatbility with previous PEFT versions
                if hasattr(module, "set_adapter"):
                    module.set_adapter(adapter_name)
                else:
                    module.active_adapter = adapter_name
                module.scale_layer(weight)

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