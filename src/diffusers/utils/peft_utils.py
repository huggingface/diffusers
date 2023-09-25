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
import importlib
from typing import List

from packaging import version

from .import_utils import is_peft_available, is_torch_available


if is_torch_available():
    import torch

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
                original_scale = module.scaling[module.active_adapter]

                # Store the previous scale in case we multiply it by zero
                if "_hf_peft_original_scales" not in module.scaling:
                    module.scaling["_hf_peft_original_scales"] = {module.active_adapter: original_scale}
                else:
                    module.scaling["_hf_peft_original_scales"][module.active_adapter] = original_scale

                module.scaling[module.active_adapter] *= scale


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
                module.scaling[module.active_adapter] /= scale
    elif scale is not None and scale == 0.0:
        for module in model.modules():
            if isinstance(module, BaseTunerLayer):
                if "_hf_peft_original_scales" not in module.scaling:
                    raise ValueError(
                        "The layer has not been scaled, cannot unscale it - please call first `scale_peft_layers`"
                    )

                original_scale = module.scaling["_hf_peft_original_scales"][module.active_adapter]
                module.scaling[module.active_adapter] = original_scale

                del module.scaling["_hf_peft_original_scales"][module.active_adapter]

                # Clean up ..
                if len(module.scaling["_hf_peft_original_scales"]) == 0:
                    del module.scaling["_hf_peft_original_scales"]


class PeftLayerScaler:
    r"""
    A custom context manager that scale / unscale PEFT layers before and after the forward pass.
    """

    def __init__(self, modules_to_scale: List["torch.nn.Module"], scale: float = None):
        self.modules_to_scale = modules_to_scale
        self.scale = scale

    def __enter__(self, *args, **kwargs):
        if self.scale is not None and self.scale != 1.0:
            for submodule in self.modules_to_scale:
                scale_peft_layers(submodule, self.scale)

    def __exit__(self, *args, **kwargs):
        if self.scale is not None and self.scale != 1.0:
            for submodule in self.modules_to_scale:
                unscale_peft_layers(submodule, self.scale)


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
