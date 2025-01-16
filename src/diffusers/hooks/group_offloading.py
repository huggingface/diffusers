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
from typing import Dict, List, Optional, Tuple, Union

import torch

from ..utils import get_logger
from .hooks import HookRegistry, ModelHook


logger = get_logger(__name__)  # pylint: disable=invalid-name


class ModuleGroup:
    def __init__(
        self,
        modules: List[torch.nn.Module],
        offload_device: torch.device,
        onload_device: torch.device,
        offload_leader: torch.nn.Module,
        onload_leader: Optional[torch.nn.Module] = None,
        parameters: Optional[List[torch.nn.Parameter]] = None,
        buffers: Optional[List[torch.Tensor]] = None,
    ) -> None:
        self.modules = modules
        self.offload_device = offload_device
        self.onload_device = onload_device
        self.offload_leader = offload_leader
        self.onload_leader = onload_leader
        self.parameters = parameters
        self.buffers = buffers


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

    def __init__(
        self,
        group: ModuleGroup,
        offload_on_init: bool = True,
        non_blocking: bool = False,
        stream: Optional[torch.cuda.Stream] = None,
        next_group: Optional[ModuleGroup] = None,
        cpu_param_dict: Optional[Dict[torch.nn.Parameter, torch.Tensor]] = None,
        onload_self: bool = False,
    ) -> None:
        self.group = group
        self.offload_on_init = offload_on_init
        self.non_blocking = non_blocking
        self.stream = stream
        self.next_group = next_group
        self.cpu_param_dict = cpu_param_dict
        self.onload_self = onload_self

    def initialize_hook(self, module: torch.nn.Module) -> torch.nn.Module:
        if self.offload_on_init:
            self.offload_(module)
        return module

    def pre_forward(self, module: torch.nn.Module, *args, **kwargs):
        if self.group.onload_leader is None:
            self.group.onload_leader = module
        self.onload_(module)
        return args, kwargs

    def post_forward(self, module: torch.nn.Module, output):
        self.offload_(module)
        return output

    def onload_(self, module: torch.nn.Module) -> None:
        if self.group.onload_leader == module:
            if self.stream is not None:
                # Wait for previous Host->Device transfer to complete
                self.stream.synchronize()

                if self.next_group is None:
                    return

                # Start Host->Device transfer for next group
                with torch.cuda.stream(self.stream):
                    for group_module in self.next_group.modules:
                        group_module.to(self.next_group.onload_device, non_blocking=True)

            if self.stream is None or self.onload_self:
                for group_module in self.group.modules:
                    group_module.to(self.group.onload_device, non_blocking=self.non_blocking)
                if self.group.parameters is not None:
                    for param in self.group.parameters:
                        param.data = param.data.to(self.group.onload_device, non_blocking=self.non_blocking)
                if self.group.buffers is not None:
                    for buffer in self.group.buffers:
                        buffer.data = buffer.data.to(self.group.onload_device, non_blocking=self.non_blocking)
                if self.onload_self:
                    torch.cuda.synchronize()

    def offload_(self, module: torch.nn.Module) -> None:
        if self.group.offload_leader == module:
            if self.stream is not None:
                for group_module in self.group.modules:
                    for param in group_module.parameters():
                        param.data = self.cpu_param_dict[param]
            else:
                for group_module in self.group.modules:
                    group_module.to(self.group.offload_device, non_blocking=self.non_blocking)
                if self.group.parameters is not None:
                    for param in self.group.parameters:
                        param.data = param.data.to(self.group.offload_device, non_blocking=self.non_blocking)
                if self.group.buffers is not None:
                    for buffer in self.group.buffers:
                        buffer.data = buffer.data.to(self.group.offload_device, non_blocking=self.non_blocking)

                # TODO: do we need to sync here because of GPU->CPU transfer?
                if self.non_blocking and self.group.offload_device.type == "cpu":
                    torch.cpu.synchronize()


def apply_group_offloading(
    module: torch.nn.Module,
    offload_group_patterns: Union[str, List[str]] = "modulelist_or_sequential",
    num_blocks_per_group: Optional[int] = None,
    offload_device: torch.device = torch.device("cpu"),
    onload_device: torch.device = torch.device("cuda"),
    force_offload: bool = True,
    non_blocking: bool = False,
    cuda_stream: bool = False,
) -> None:
    stream = None
    if cuda_stream:
        stream = torch.cuda.Stream()
    if offload_group_patterns == "modulelist_or_sequential":
        if num_blocks_per_group is None:
            raise ValueError(
                "num_blocks_per_group must be provided when using offload_group_patterns='modulelist_or_sequential'."
            )
        # _apply_group_offloading_diffusers_block(
        #     module,
        #     num_blocks_per_group,
        #     offload_device,
        #     onload_device,
        #     force_offload,
        #     non_blocking,
        #     stream,
        # )
        offload_group_patterns = _get_modulelist_or_sequential_group_patterns(module, num_blocks_per_group)

    _apply_group_offloading_group_patterns(
        module, offload_group_patterns, offload_device, onload_device, force_offload, non_blocking, stream=stream
    )


# def _apply_group_offloading_diffusers_block(
#     module: torch.nn.Module,
#     num_blocks_per_group: int,
#     offload_device: torch.device,
#     onload_device: torch.device,
#     force_offload: bool,
#     non_blocking: bool,
#     stream: Optional[torch.cuda.Stream] = None,
# ) -> None:
#     cpu_param_dict = None
#     if stream is not None:
#         for param in module.parameters():
#             param.data = param.data.cpu().pin_memory()
#         cpu_param_dict = {param: param.data for param in module.parameters()}

#     # Handle device offloading/onloading for unet/transformer stack modules
#     for stack_identifier in _COMMON_STACK_IDENTIFIERS:
#         if not hasattr(module, stack_identifier) or not isinstance(
#             getattr(module, stack_identifier), torch.nn.ModuleList
#         ):
#             continue

#         stack = getattr(module, stack_identifier)
#         num_blocks = len(stack)
#         module_groups = []

#         for i in range(0, num_blocks, num_blocks_per_group):
#             blocks = stack[i : i + num_blocks_per_group]
#             group = ModuleGroup(
#                 blocks, offload_device, onload_device, offload_leader=blocks[-1], onload_leader=blocks[0]
#             )
#             module_groups.append(group)

#         for i, group in enumerate(module_groups):
#             next_group = module_groups[i + 1] if i + 1 < len(module_groups) and stream is not None else None
#             should_offload = force_offload or i > 0
#             _apply_group_offloading(group, should_offload, non_blocking, stream, next_group, cpu_param_dict)

#         if stream is not None:
#             # Start Host->Device transfer for the first group
#             with torch.cuda.stream(stream):
#                 for group_module in module_groups[0].modules:
#                     group_module.to(onload_device, non_blocking=True)
#             if len(module_groups) > 1:
#                 # Assign the first module_group as the next_group for the last module_group
#                 hook_registry = HookRegistry.check_if_exists_or_initialize(module_groups[-1].onload_leader)
#                 hook_registry.hooks["group_offloading"].next_group = module_groups[0]

#     # Handle device offloading/onloading for non-stack modules
#     for name, submodule in module.named_modules():
#         name_split = name.split(".")
#         if not isinstance(submodule, torch.nn.Module) or name == "" or len(name_split) > 1:
#             # We only want the layers that are top-level in the module (encompass all the submodules)
#             # for enabling offloading.
#             continue
#         layer_name = name_split[0]
#         if layer_name in _COMMON_STACK_IDENTIFIERS:
#             continue
#         group = ModuleGroup(
#             [submodule], offload_device, onload_device, offload_leader=submodule, onload_leader=submodule
#         )
#         _apply_group_offloading(group, force_offload, non_blocking)

#     # Always keep parameters and buffers on onload_device
#     for name, param in module.named_parameters(recurse=False):
#         if torch.is_tensor(param.data):
#             param.data = param.data.to(onload_device)
#     for name, buffer in module.named_buffers(recurse=False):
#         if torch.is_tensor(buffer.data):
#             buffer.data = buffer.data.to(onload_device)


def _apply_group_offloading_group_patterns(
    module: torch.nn.Module,
    offload_group_patterns: List[Tuple[str, str, Optional[str]]],
    offload_device: torch.device,
    onload_device: torch.device,
    force_offload: bool,
    non_blocking: bool,
    stream: Optional[torch.cuda.Stream] = None,
) -> None:
    r"""
    This function applies offloading to groups of modules based on the provided regex patterns. Each group of modules
    that match a pattern are offloaded and onloaded together. The order of the patterns in the list is important as it
    determines the order of execution of the forward pass. If the order is not correct, group offloading may almost
    certainly fail with device mismatch errors.

    In the interest of simplicity, this function does not handle complicated cases where one regex pattern matches a
    module, but another regex pattern matches an internal submodule of that module. This would be a difficult case to
    handle and require a more complex checker, which is not implemented here. As a general rule of thumb, make sure to
    provide regex patterns for all models that are at the same level of the computation graph in terms of invocation
    order. For example, either all leaf modules, or all transformer blocks, etc.

    Note that parameters and buffers are always kept on the onload_device. This is because they are usually small
    enough to not have any impact on memory usage. If you require support for offloading parameters and buffers, please
    open an issue at https://github.com/huggingface/diffusers/issues.

    Args:
        module (`torch.nn.Module`):
            The module to which group offloading is applied.
        offload_group_patterns (`List[Tuple[str, str, Optional[str]]]`):
            A list of tuples that determine groups of modules that are offloaded and onloaded together. Each tuple
            contains three elements:
            - A regex pattern that matches the names of the modules in the group.
            - A regex pattern that matches a single layer that is the offload leader of the group.
            - An optional regex pattern that matches a single layer that is the onload leader of the group. This can be
              set to None because it is easier to determine the onload leader based on the forward invocation order,
              which triggers the call to GroupOffloadingHook.
        offload_device (`torch.device`):
            The device to which the group of modules are offloaded. This should typically be the CPU.
        onload_device (`torch.device`):
            The device to which the group of modules are onloaded.
        force_offload (`bool`):
            If True, all module groups are offloaded to the offload_device. If False, only layers that match
            `offload_group_patterns` are offloaded to the offload_device.
        non_blocking (`bool`):
            If True, offloading and onloading is done asynchronously. This can be useful for overlapping computation
            and data transfer.
        stream (`torch.cuda.Stream`, *optional*):
            If provided, offloading and onloading is done asynchronously using the provided stream. This can be useful
            for overlapping computation and data transfer.
    """

    cpu_param_dict = None
    if stream is not None:
        for param in module.parameters():
            param.data = param.data.cpu().pin_memory()
        cpu_param_dict = {param: param.data for param in module.parameters()}

    per_group_modules = [[] for _ in range(len(offload_group_patterns))]
    per_group_offload_leaders = [None] * len(offload_group_patterns)
    per_group_onload_leaders = [None] * len(offload_group_patterns)
    unmatched_group_modules = []

    group_patterns = [pattern[0] for pattern in offload_group_patterns]
    offload_leader_patterns = [pattern[1] for pattern in offload_group_patterns]
    onload_leader_patterns = [pattern[2] for pattern in offload_group_patterns]

    for name, submodule in module.named_modules():
        if name == "" or name.count(".") > 1:
            # We only want the layers that are top-level in the module (encompass all the other submodules)
            # for enabling offloading. This method is specifically targeted for diffusers format models,
            # so we can ignore submodules.
            # TODO(aryan): This is not the case and is just a workaround to make the benchmark code work
            # for now. We need to support the arbitrary nesting of modules here.
            continue

        # Check if the module matches any of the offload group patterns
        num_matches = 0
        for i, pattern in enumerate(group_patterns):
            if re.search(pattern, name) is not None:
                per_group_modules[i].append(submodule)
                num_matches += 1

        # Check if the module matches any of the offload leader patterns
        for i, pattern in enumerate(offload_leader_patterns):
            if re.search(pattern, name) is not None:
                if per_group_offload_leaders[i] is not None:
                    raise ValueError(
                        f"Module {name} matches multiple offload leader patterns. Please ensure that offload leader patterns are mutually exclusive."
                    )
                per_group_offload_leaders[i] = submodule

        # Check if the module matches any of the onload leader patterns
        for i, pattern in enumerate(onload_leader_patterns):
            if pattern is None:
                continue
            if re.search(pattern, name) is not None:
                if per_group_onload_leaders[i] is not None:
                    raise ValueError(
                        f"Module {name} matches multiple onload leader patterns. Please ensure that onload leader patterns are mutually exclusive."
                    )
                per_group_onload_leaders[i] = submodule

        if num_matches == 0:
            unmatched_group_modules.append((name, submodule))
        elif num_matches > 1:
            raise ValueError(
                f"Module {name} matches multiple offload group patterns. Please ensure that offloading group patterns are mutually exclusive."
            )

    # Handle modules that matched patterns
    groups = []
    for i in range(len(per_group_modules)):
        if per_group_offload_leaders[i] is None:
            raise ValueError(
                f"No offload leader found for group {i}. Please ensure that each group has a single offload leader."
            )
        group = ModuleGroup(
            per_group_modules[i],
            offload_device,
            onload_device,
            offload_leader=per_group_offload_leaders[i],
            onload_leader=per_group_onload_leaders[i],
        )
        groups.append(group)

    for i in range(len(groups)):
        next_group = groups[i + 1] if i + 1 < len(groups) and stream is not None else None
        should_offload = force_offload or i > 0
        _apply_group_offloading(
            groups[i], should_offload, non_blocking, stream, next_group, cpu_param_dict, onload_self=False
        )

    # Ignore parameters/buffers if they're already accounted for in unmatched_group_modules (for example, a nn.Linear
    # in the top-level module will also be present in the named_parameters iterator)
    parameters = []
    for name, parameter in module.named_parameters(recurse=False):
        if not any(name.startswith(unmatched_name) for unmatched_name, _ in unmatched_group_modules):
            parameters.append(parameter)

    buffers = []
    for name, buffer in module.named_buffers(recurse=False):
        if not any(name.startswith(unmatched_name) for unmatched_name, _ in unmatched_group_modules):
            buffers.append(buffer)

    ignore_blocks = ["transformer_blocks", "single_transformer_blocks", "temporal_transformer_blocks", "blocks"]
    unmatched_modules = [module for name, module in unmatched_group_modules if name not in ignore_blocks]
    unmatched_group = ModuleGroup(
        unmatched_modules,
        offload_device,
        onload_device,
        offload_leader=module,
        onload_leader=None,
        parameters=parameters,
        buffers=buffers,
    )
    _apply_group_offloading(
        unmatched_group, force_offload, non_blocking, stream, groups[0], cpu_param_dict, onload_self=True
    )


def _apply_group_offloading(
    group: ModuleGroup,
    offload_on_init: bool,
    non_blocking: bool,
    stream: Optional[torch.cuda.Stream] = None,
    next_group: Optional[ModuleGroup] = None,
    cpu_param_dict: Optional[Dict[torch.nn.Parameter, torch.Tensor]] = None,
    onload_self: bool = False,
) -> None:
    for module in group.modules:
        hook = GroupOffloadingHook(
            group, offload_on_init, non_blocking, stream, next_group, cpu_param_dict, onload_self
        )
        registry = HookRegistry.check_if_exists_or_initialize(module)
        registry.register_hook(hook, "group_offloading")


def _get_modulelist_or_sequential_group_patterns(module: torch.nn.Module, num_blocks_per_group: int) -> List[str]:
    r"""
    This function generates group patterns for offloading based on the number of blocks per group. Given a module, it
    will iterate through the submodules and find usages of torch.nn.ModuleList and torch.nn.Sequential. For each group
    of `num_blocks_per_group` consecutive blocks, it will generate a regex pattern that matches the names of these
    blocks. The generated patterns can be used to create ModuleGroup objects which are offloaded and onloaded together.
    """
    group_patterns = []

    # We only want the layers that are top-level in the module (encompass all the other submodules)
    # for enabling offloading. This method is specifically targeted for diffusers format models,
    # so we can ignore everything but the children of this module.
    for name, submodule in module.named_children():
        if not isinstance(submodule, (torch.nn.ModuleList, torch.nn.Sequential)):
            continue
        for i in range(0, len(submodule), num_blocks_per_group):
            num_modules = len(submodule[i : i + num_blocks_per_group])
            pattern = "|".join([rf"{name}\.{i + j}\b" for j in range(num_modules)])
            pattern = f"({pattern})"
            onload_leader_pattern = rf"{name}\.{i}\b"
            offload_leader_pattern = rf"{name}\.{i + num_modules - 1}\b"
            group_patterns.append((pattern, offload_leader_pattern, onload_leader_pattern))

    logger.debug(f"Generated group patterns for apply_groupwise_offloading: {group_patterns}")
    return group_patterns
