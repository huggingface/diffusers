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

    for module in model.modules():
        if isinstance(module, BaseTunerLayer):
            module.scale_layer(weight)


def unscale_lora_layers(model):
    """
    Removes the previously passed weight given to the LoRA layers of the model.

    Args:
        model (`torch.nn.Module`):
            The model to scale.
        weight (`float`):
            The weight to be given to the LoRA layers.
    """
    from peft.tuners.tuners_utils import BaseTunerLayer

    for module in model.modules():
        if isinstance(module, BaseTunerLayer):
            module.unscale_layer()


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
