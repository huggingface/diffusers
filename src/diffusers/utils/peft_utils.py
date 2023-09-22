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
from .import_utils import is_torch_available


if is_torch_available():
    import torch


def recurse_remove_peft_layers(model):
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


def scale_lora_layers(model, lora_weightage):
    from peft.tuners.tuner_utils import BaseTunerLayer

    for module in model.modules():
        if isinstance(module, BaseTunerLayer):
            module.scale_layer(lora_weightage)


def get_rank_and_alpha_pattern(rank_dict, network_alpha_dict, peft_state_dict):
    rank_pattern = None
    alpha_pattern = None
    r = lora_alpha = list(rank_dict.values())[0]
    if len(set(rank_dict.values())) > 1:
        # get the rank occuring the most number of times
        r = max(set(rank_dict.values()), key=list(rank_dict.values()).count)

        # for modules with rank different from the most occuring rank, add it to the `rank_pattern`
        rank_pattern = dict(filter(lambda x: x[1] != r, rank_dict.items()))
        rank_pattern = {k.split(".lora_B.")[0]: v for k, v in rank_pattern.items()}

    if network_alpha_dict is not None and len(set(network_alpha_dict.values())) > 1:
        # get the alpha occuring the most number of times
        lora_alpha = max(set(network_alpha_dict.values()), key=list(network_alpha_dict.values()).count)

        # for modules with alpha different from the most occuring alpha, add it to the `alpha_pattern`
        alpha_pattern = dict(filter(lambda x: x[1] != lora_alpha, network_alpha_dict.items()))
        alpha_pattern = {".".join(k.split(".down.")[0].split(".")[:-1]): v for k, v in alpha_pattern.items()}

    # layer names without the Diffusers specific
    target_modules = list({name.split(".lora")[0] for name in peft_state_dict.keys()})

    return r, lora_alpha, rank_pattern, alpha_pattern, target_modules


def get_adapter_name(model):
    from peft.tuners.tuner_utils import BaseTunerLayer

    for module in model.modules():
        if isinstance(module, BaseTunerLayer):
            return f"default_{len(module.r)}"
    return "default_0"


def set_adapter_layers(model, enabled=True):
    from peft.tuners.tuner_utils import BaseTunerLayer

    for module in model.modules():
        if isinstance(module, BaseTunerLayer):
            module.disable_adapters = False if enabled else True


def set_weights_and_activate_adapters(model, adapter_names, weights):
    from peft.tuners.tuner_utils import BaseTunerLayer

    # iterate over each adapter, make it active and set the corresponding scaling weight
    for adapter_name, weight in zip(adapter_names, weights):
        for module in model.modules():
            if isinstance(module, BaseTunerLayer):
                module.active_adapter = adapter_name
                module.scale_layer(weight)

    # set multiple active adapters
    for module in model.modules():
        if isinstance(module, BaseTunerLayer):
            module.active_adapter = adapter_names
