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

from contextlib import nullcontext
from typing import Dict, List, Optional

import torch
from accelerate.utils import send_to_device

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
        non_blocking: bool = False,
        stream: Optional[torch.cuda.Stream] = None,
        cpu_param_dict: Optional[Dict[torch.nn.Parameter, torch.Tensor]] = None,
        onload_self: bool = True,
    ) -> None:
        self.modules = modules
        self.offload_device = offload_device
        self.onload_device = onload_device
        self.offload_leader = offload_leader
        self.onload_leader = onload_leader
        self.parameters = parameters
        self.buffers = buffers
        self.non_blocking = non_blocking or stream is not None
        self.stream = stream
        self.cpu_param_dict = cpu_param_dict
        self.onload_self = onload_self

        if self.stream is not None and self.cpu_param_dict is None:
            raise ValueError("cpu_param_dict must be provided when using stream for data transfer.")

    def onload_(self):
        context = nullcontext() if self.stream is None else torch.cuda.stream(self.stream)
        if self.stream is not None:
            # Wait for previous Host->Device transfer to complete
            self.stream.synchronize()

        with context:
            for group_module in self.modules:
                group_module.to(self.onload_device, non_blocking=self.non_blocking)
            if self.parameters is not None:
                for param in self.parameters:
                    param.data = param.data.to(self.onload_device, non_blocking=self.non_blocking)
            if self.buffers is not None:
                for buffer in self.buffers:
                    buffer.data = buffer.data.to(self.onload_device, non_blocking=self.non_blocking)

    def offload_(self):
        if self.stream is not None:
            for group_module in self.modules:
                for param in group_module.parameters():
                    param.data = self.cpu_param_dict[param]
        else:
            for group_module in self.modules:
                group_module.to(self.offload_device, non_blocking=self.non_blocking)
            if self.parameters is not None:
                for param in self.parameters:
                    param.data = param.data.to(self.offload_device, non_blocking=self.non_blocking)
            if self.buffers is not None:
                for buffer in self.buffers:
                    buffer.data = buffer.data.to(self.offload_device, non_blocking=self.non_blocking)

            # TODO: do we need to sync here because of GPU->CPU transfer?
            if self.non_blocking and self.offload_device.type == "cpu":
                torch.cpu.synchronize()


class GroupOffloadingHook(ModelHook):
    r"""
    A hook that offloads groups of torch.nn.Module to the CPU for storage and onloads to accelerator device for
    computation. Each group has one "onload leader" module that is responsible for onloading, and an "offload leader"
    module that is responsible for offloading. If prefetching is enabled, the onload leader of the previous module
    group is responsible for onloading the current module group.
    """

    def __init__(
        self,
        group: ModuleGroup,
        offload_on_init: bool = True,
        next_group: Optional[ModuleGroup] = None,
    ) -> None:
        self.group = group
        self.offload_on_init = offload_on_init
        self.next_group = next_group

    def initialize_hook(self, module: torch.nn.Module) -> torch.nn.Module:
        if self.offload_on_init and self.group.offload_leader == module:
            self.group.offload_()
        return module

    def pre_forward(self, module: torch.nn.Module, *args, **kwargs):
        if self.group.onload_leader is None:
            self.group.onload_leader = module
        if self.group.onload_leader == module:
            if self.group.onload_self:
                self.group.onload_()
            if self.next_group is not None and not self.next_group.onload_self:
                self.next_group.onload_()
        args = send_to_device(args, self.group.onload_device, non_blocking=self.group.non_blocking)
        kwargs = send_to_device(kwargs, self.group.onload_device, non_blocking=self.group.non_blocking)
        return args, kwargs

    def post_forward(self, module: torch.nn.Module, output):
        if self.group.offload_leader == module:
            self.group.offload_()
        return output


def apply_group_offloading(
    module: torch.nn.Module,
    offload_type: str = "block_level",
    num_blocks_per_group: Optional[int] = None,
    offload_device: torch.device = torch.device("cpu"),
    onload_device: torch.device = torch.device("cuda"),
    force_offload: bool = True,
    non_blocking: bool = False,
    use_stream: bool = False,
) -> None:
    stream = None
    if use_stream:
        if torch.cuda.is_available():
            stream = torch.cuda.Stream()
        else:
            raise ValueError("Using streams for data transfer requires a CUDA device.")

    if offload_type == "block_level":
        if num_blocks_per_group is None:
            raise ValueError("num_blocks_per_group must be provided when using offload_group_patterns='block_level'.")

        _apply_group_offloading_block_level(
            module, num_blocks_per_group, offload_device, onload_device, force_offload, non_blocking, stream=stream
        )
    # elif offload_type == "leaf_level":
    #     _apply_group_offloading_leaf_level(
    #         module, offload_device, onload_device, force_offload, non_blocking, stream=stream
    #     )


def _apply_group_offloading_block_level(
    module: torch.nn.Module,
    num_blocks_per_group: int,
    offload_device: torch.device,
    onload_device: torch.device,
    force_offload: bool,
    non_blocking: bool,
    stream: Optional[torch.cuda.Stream] = None,
) -> None:
    r"""
    This function applies offloading to groups of torch.nn.ModuleList or torch.nn.Sequential blocks.

    Args:
        module (`torch.nn.Module`):
            The module to which group offloading is applied.
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

    unmatched_modules = []
    matched_module_groups = []
    for name, submodule in module.named_children():
        if not isinstance(submodule, (torch.nn.ModuleList, torch.nn.Sequential)):
            unmatched_modules.append((name, submodule))
            continue
        for i in range(0, len(submodule), num_blocks_per_group):
            group = ModuleGroup(
                modules=submodule[i : i + num_blocks_per_group],
                offload_device=offload_device,
                onload_device=onload_device,
                offload_leader=submodule[i],
                onload_leader=None,
                non_blocking=non_blocking,
                stream=stream,
                cpu_param_dict=cpu_param_dict,
                onload_self=stream is None,
            )
            matched_module_groups.append(group)

    for i, group in enumerate(matched_module_groups):
        next_group = (
            matched_module_groups[i + 1] if i + 1 < len(matched_module_groups) and stream is not None else None
        )
        should_offload = force_offload or i > 0
        _apply_group_offloading(group, should_offload, next_group)

    parameters = []
    for name, parameter in module.named_parameters(recurse=False):
        if not any(name.startswith(unmatched_name) for unmatched_name, _ in unmatched_modules):
            parameters.append(parameter)

    buffers = []
    for name, buffer in module.named_buffers(recurse=False):
        if not any(name.startswith(unmatched_name) for unmatched_name, _ in unmatched_modules):
            buffers.append(buffer)

    unmatched_modules = [unmatched_module for _, unmatched_module in unmatched_modules]
    unmatched_group = ModuleGroup(
        modules=unmatched_modules,
        offload_device=offload_device,
        onload_device=onload_device,
        offload_leader=module,
        onload_leader=None,
        parameters=parameters,
        buffers=buffers,
        non_blocking=False,
        stream=None,
        cpu_param_dict=cpu_param_dict,
        onload_self=True,
    )
    _apply_group_offloading(unmatched_group, force_offload, matched_module_groups[0])


# def _apply_group_offloading_leaf_level(
#     module: torch.nn.Module,
#     offload_device: torch.device,
#     onload_device: torch.device,
#     force_offload: bool,
#     non_blocking: bool,
#     stream: Optional[torch.cuda.Stream] = None,
# ) -> None:
#     r"""
# This function applies offloading to groups of leaf modules in a torch.nn.Module.

# Args: # module (`torch.nn.Module`): # The module to which group offloading is applied. # offload_device
(`torch.device`): # The device to which the group of modules are offloaded. This should typically be the CPU. #
onload_device (`torch.device`): # The device to which the group of modules are onloaded. # force_offload (`bool`): # If
True, all module groups are offloaded to the offload_device. If False, only layers that match #
`offload_group_patterns` are offloaded to the offload_device. # non_blocking (`bool`): # If True, offloading and
onloading is done asynchronously. This can be useful for overlapping computation # and data transfer. # stream
(`torch.cuda.Stream`, *optional*): # If provided, offloading and onloading is done asynchronously using the provided
stream. This can be useful # for overlapping computation and data transfer. #"""

#     cpu_param_dict = None
#     if stream is not None:
#         for param in module.parameters():
#             param.data = param.data.cpu().pin_memory()
#         cpu_param_dict = {param: param.data for param in module.parameters()}


def _apply_group_offloading(
    group: ModuleGroup,
    offload_on_init: bool,
    next_group: Optional[ModuleGroup] = None,
) -> None:
    for module in group.modules:
        hook = GroupOffloadingHook(group, offload_on_init, next_group)
        registry = HookRegistry.check_if_exists_or_initialize(module)
        registry.register_hook(hook, "group_offloading")
