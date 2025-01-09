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
from typing import List, Optional, Union

import torch

from .hooks import HookRegistry, ModelHook


_TRANSFORMER_STACK_IDENTIFIERS = [
    "transformer_blocks",
    "single_transformer_blocks",
    "temporal_transformer_blocks",
    "transformer_layers",
    "layers",
    "blocks",
]


class ModuleGroup:
    def __init__(
        self,
        modules: List[torch.nn.Module],
        offload_device: torch.device,
        onload_device: torch.device,
        offload_leader: torch.nn.Module,
        onload_leader: Optional[torch.nn.Module] = None,
    ) -> None:
        self.modules = modules
        self.offload_device = offload_device
        self.onload_device = onload_device
        self.offload_leader = offload_leader
        self.onload_leader = onload_leader


class GroupOffloadingHook(ModelHook):
    r"""
    A hook that offloads groups of torch.nn.Module to the CPU for storage and onloads to accelerator device for
    computation. Each group has one "onload leader" module that is responsible for onloading, and an "offload leader"
    module that is responsible for offloading.

    This implementation assumes the following:
    - For offload_group_patterns="diffusers_block", the leader of a group can be automatically determined. For a custom
      user-provided regex pattern, the module that triggers its forward pass first is considered the leader.
    - The inputs are already on the correct device. This is expected because the hook does not modify the state of
      inputs or outputs at any stage of the forward pass. If an error is raised due to the device of modules and inputs
      not matching during the forward pass for any model in Diffusers, this means that the forward pass of the model is
      not written in the expected. Please open an issue at https://github.com/huggingface/diffusers/issues if you
      encounter such an error.
    """

    def __init__(self, group: ModuleGroup, offload_on_init: bool = True) -> None:
        self.group = group
        self.offload_on_init = offload_on_init

    def initialize_hook(self, module: torch.nn.Module) -> torch.nn.Module:
        if self.offload_on_init:
            self.offload_(module)
        return module

    def onload_(self, module: torch.nn.Module) -> None:
        if self.group.onload_leader is not None and self.group.onload_leader == module:
            for group_module in self.group.modules:
                group_module.to(self.group.onload_device)

    def offload_(self, module: torch.nn.Module) -> None:
        if self.group.offload_leader == module:
            for group_module in self.group.modules:
                group_module.to(self.group.offload_device)

    def pre_forward(self, module: torch.nn.Module, *args, **kwargs):
        if self.group.onload_leader is None:
            self.group.onload_leader = module
        self.onload_(module)
        return args, kwargs

    def post_forward(self, module: torch.nn.Module, output):
        self.offload_(module)
        return output


def apply_group_offloading(
    module: torch.nn.Module,
    offload_group_patterns: Union[str, List[str]] = "diffusers_block",
    num_blocks_per_group: Optional[int] = None,
    offload_device: torch.device = torch.device("cpu"),
    onload_device: torch.device = torch.device("cuda"),
    force_offload: bool = True,
) -> None:
    if offload_group_patterns == "diffusers_block":
        _apply_group_offloading_diffusers_block(
            module, num_blocks_per_group, offload_device, onload_device, force_offload
        )
    else:
        _apply_group_offloading_group_patterns(
            module, offload_group_patterns, offload_device, onload_device, force_offload
        )


def _apply_group_offloading_diffusers_block(
    module: torch.nn.Module,
    num_blocks_per_group: int,
    offload_device: torch.device,
    onload_device: torch.device,
    force_offload: bool,
) -> None:
    if num_blocks_per_group is None:
        raise ValueError("num_blocks_per_group must be provided when using GroupOffloadingType.DIFFUSERS_BLOCK.")

    for transformer_stack_identifier in _TRANSFORMER_STACK_IDENTIFIERS:
        if not hasattr(module, transformer_stack_identifier) or not isinstance(
            getattr(module, transformer_stack_identifier), torch.nn.ModuleList
        ):
            continue

        transformer_stack = getattr(module, transformer_stack_identifier)
        num_blocks = len(transformer_stack)

        for i in range(0, num_blocks, num_blocks_per_group):
            blocks = transformer_stack[i : i + num_blocks_per_group]
            group = ModuleGroup(
                blocks, offload_device, onload_device, offload_leader=blocks[-1], onload_leader=blocks[0]
            )
            should_offload = force_offload or i > 0
            _apply_group_offloading(group, should_offload)


def _apply_group_offloading_group_patterns(
    module: torch.nn.Module,
    offload_group_patterns: List[str],
    offload_device: torch.device,
    onload_device: torch.device,
    force_offload: bool,
) -> None:
    per_group_modules = []
    for i, offload_group_pattern in enumerate(offload_group_patterns):
        group_modules = []
        group_module_names = []
        for name, module in module.named_modules():
            if re.search(offload_group_pattern, name) is not None:
                group_modules.append(module)
                group_module_names.append(name)
        per_group_modules.append(
            {
                "modules": group_modules,
                "module_names": group_module_names,
            }
        )

    # Check if there are any overlapping modules between groups
    for i, group in enumerate(per_group_modules):
        for j, other_group in enumerate(per_group_modules):
            if j <= i:
                continue
            if any(module_name in group["module_names"] for module_name in other_group["module_names"]):
                raise ValueError(
                    f"Overlapping modules between groups {i} and {j}. Please ensure that offloading group patterns are mutually exclusive."
                )

    # Apply offloading to each group
    for group in per_group_modules:
        # TODO: handle offload leader correctly
        group = ModuleGroup(group["modules"], offload_device, onload_device, offload_leader=group["modules"][-1])
        _apply_group_offloading(group, force_offload)


def _apply_group_offloading(group: ModuleGroup, offload_on_init) -> None:
    for module in group.modules:
        hook = GroupOffloadingHook(group, offload_on_init=offload_on_init)
        registry = HookRegistry.check_if_exists_or_initialize(module)
        registry.register_hook(hook, "group_offloading")
