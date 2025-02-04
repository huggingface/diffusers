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
from typing import Dict, List, Optional, Set, Tuple

import torch

from ..utils import get_logger, is_accelerate_available
from .hooks import HookRegistry, ModelHook


if is_accelerate_available():
    from accelerate.utils import send_to_device


logger = get_logger(__name__)  # pylint: disable=invalid-name


# fmt: off
_GROUP_OFFLOADING = "group_offloading"
_LAYER_EXECUTION_TRACKER = "layer_execution_tracker"
_LAZY_PREFETCH_GROUP_OFFLOADING = "lazy_prefetch_group_offloading"

_SUPPORTED_PYTORCH_LAYERS = (
    torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d,
    torch.nn.ConvTranspose1d, torch.nn.ConvTranspose2d, torch.nn.ConvTranspose3d,
    torch.nn.Linear,
    torch.nn.LayerNorm, torch.nn.GroupNorm,
)
# fmt: on


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
        r"""Onloads the group of modules to the onload_device."""
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
        r"""Offloads the group of modules to the offload_device."""
        if self.stream is not None:
            torch.cuda.current_stream().synchronize()
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


class GroupOffloadingHook(ModelHook):
    r"""
    A hook that offloads groups of torch.nn.Module to the CPU for storage and onloads to accelerator device for
    computation. Each group has one "onload leader" module that is responsible for onloading, and an "offload leader"
    module that is responsible for offloading. If prefetching is enabled, the onload leader of the previous module
    group is responsible for onloading the current module group.
    """

    _is_stateful = False

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
        # If there wasn't an onload_leader assigned, we assume that the submodule that first called its forward
        # method is the onload_leader of the group.
        if self.group.onload_leader is None:
            self.group.onload_leader = module

        # If the current module is the onload_leader of the group, we onload the group if it is supposed
        # to onload itself. In the case of using prefetching with streams, we onload the next group if
        # it is not supposed to onload itself.
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


class LazyPrefetchGroupOffloadingHook(ModelHook):
    r"""
    A hook, used in conjuction with GroupOffloadingHook, that applies lazy prefetching to groups of torch.nn.Module.
    This hook is used to determine the order in which the layers are executed during the forward pass. Once the layer
    invocation order is known, assignments of the next_group attribute for prefetching can be made, which allows
    prefetching groups in the correct order.
    """

    _is_stateful = False

    def __init__(self):
        self.execution_order: List[Tuple[str, torch.nn.Module]] = []
        self._layer_execution_tracker_module_names = set()

    def initialize_hook(self, module):
        # To every submodule that contains a group offloading hook (at this point, no prefetching is enabled for any
        # of the groups), we add a layer execution tracker hook that will be used to determine the order in which the
        # layers are executed during the forward pass.
        for name, submodule in module.named_modules():
            if name == "" or not hasattr(submodule, "_diffusers_hook"):
                continue

            registry = HookRegistry.check_if_exists_or_initialize(submodule)
            group_offloading_hook = registry.get_hook(_GROUP_OFFLOADING)

            if group_offloading_hook is not None:

                def make_execution_order_update_callback(current_name, current_submodule):
                    def callback():
                        logger.debug(f"Adding {current_name} to the execution order")
                        self.execution_order.append((current_name, current_submodule))

                    return callback

                layer_tracker_hook = LayerExecutionTrackerHook(make_execution_order_update_callback(name, submodule))
                registry.register_hook(layer_tracker_hook, _LAYER_EXECUTION_TRACKER)
                self._layer_execution_tracker_module_names.add(name)

        return module

    def post_forward(self, module, output):
        # At this point, for the current modules' submodules, we know the execution order of the layers. We can now
        # remove the layer execution tracker hooks and apply prefetching by setting the next_group attribute for each
        # group offloading hook.
        num_executed = len(self.execution_order)
        execution_order_module_names = {name for name, _ in self.execution_order}

        # It may be possible that some layers were not executed during the forward pass. This can happen if the layer
        # is not used in the forward pass, or if the layer is not executed due to some other reason. In such cases, we
        # may not be able to apply prefetching in the correct order, which can lead to device-mismatch related errors
        # if the missing layers end up being executed in the future.
        if execution_order_module_names != self._layer_execution_tracker_module_names:
            unexecuted_layers = list(self._layer_execution_tracker_module_names - execution_order_module_names)
            logger.warning(
                "It seems like some layers were not executed during the forward pass. This may lead to problems when "
                "applying lazy prefetching with automatic tracing and lead to device-mismatch related errors. Please "
                "make sure that all layers are executed during the forward pass. The following layers were not executed:\n"
                f"{unexecuted_layers=}"
            )

        # Remove the layer execution tracker hooks from the submodules
        base_module_registry = module._diffusers_hook
        registries = [submodule._diffusers_hook for _, submodule in self.execution_order]

        for i in range(num_executed):
            registries[i].remove_hook(_LAYER_EXECUTION_TRACKER, recurse=False)

        # Remove the current lazy prefetch group offloading hook so that it doesn't interfere with the next forward pass
        base_module_registry.remove_hook(_LAZY_PREFETCH_GROUP_OFFLOADING, recurse=False)

        # Apply lazy prefetching by setting required attributes
        group_offloading_hooks = [registry.get_hook(_GROUP_OFFLOADING) for registry in registries]
        if num_executed > 0:
            base_module_group_offloading_hook = base_module_registry.get_hook(_GROUP_OFFLOADING)
            base_module_group_offloading_hook.next_group = group_offloading_hooks[0].group
            base_module_group_offloading_hook.next_group.onload_self = False

        for i in range(num_executed - 1):
            name1, _ = self.execution_order[i]
            name2, _ = self.execution_order[i + 1]
            logger.debug(f"Applying lazy prefetch group offloading from {name1} to {name2}")
            group_offloading_hooks[i].next_group = group_offloading_hooks[i + 1].group
            group_offloading_hooks[i].next_group.onload_self = False

        return output


class LayerExecutionTrackerHook(ModelHook):
    r"""
    A hook that tracks the order in which the layers are executed during the forward pass by calling back to the
    LazyPrefetchGroupOffloadingHook to update the execution order.
    """

    _is_stateful = False

    def __init__(self, execution_order_update_callback):
        self.execution_order_update_callback = execution_order_update_callback

    def pre_forward(self, module, *args, **kwargs):
        self.execution_order_update_callback()
        return args, kwargs


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
    elif offload_type == "leaf_level":
        _apply_group_offloading_leaf_level(
            module, offload_device, onload_device, force_offload, non_blocking, stream=stream
        )
    else:
        raise ValueError(f"Unsupported offload_type: {offload_type}")


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
    This function applies offloading to groups of torch.nn.ModuleList or torch.nn.Sequential blocks. In comparison to
    the "leaf_level" offloading, which is more fine-grained, this offloading is done at the top-level blocks.

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

    # Create a pinned CPU parameter dict for async data transfer if streams are to be used
    cpu_param_dict = None
    if stream is not None:
        for param in module.parameters():
            param.data = param.data.cpu().pin_memory()
        cpu_param_dict = {param: param.data for param in module.parameters()}

    # Create module groups for ModuleList and Sequential blocks
    modules_with_group_offloading = set()
    unmatched_modules = []
    matched_module_groups = []
    for name, submodule in module.named_children():
        if not isinstance(submodule, (torch.nn.ModuleList, torch.nn.Sequential)):
            unmatched_modules.append((name, submodule))
            modules_with_group_offloading.add(name)
            continue

        for i in range(0, len(submodule), num_blocks_per_group):
            current_modules = submodule[i : i + num_blocks_per_group]
            group = ModuleGroup(
                modules=current_modules,
                offload_device=offload_device,
                onload_device=onload_device,
                offload_leader=current_modules[-1],
                onload_leader=current_modules[0],
                non_blocking=non_blocking,
                stream=stream,
                cpu_param_dict=cpu_param_dict,
                onload_self=stream is None,
            )
            matched_module_groups.append(group)
            for j in range(i, i + len(current_modules)):
                modules_with_group_offloading.add(f"{name}.{j}")

    # Apply group offloading hooks to the module groups
    for i, group in enumerate(matched_module_groups):
        next_group = (
            matched_module_groups[i + 1] if i + 1 < len(matched_module_groups) and stream is not None else None
        )
        should_offload = force_offload or i > 0

        for group_module in group.modules:
            _apply_group_offloading_hook(group_module, group, should_offload, next_group)

    # Parameters and Buffers of the top-level module need to be offloaded/onloaded separately
    # when the forward pass of this module is called. This is because the top-level module is not
    # part of any group (as doing so would lead to no VRAM savings).
    parameters = _gather_parameters_with_no_group_offloading_parent(module, modules_with_group_offloading)
    buffers = _gather_buffers_with_no_group_offloading_parent(module, modules_with_group_offloading)
    parameters = [param for _, param in parameters]
    buffers = [buffer for _, buffer in buffers]

    # Create a group for the unmatched submodules of the top-level module so that they are on the correct
    # device when the forward pass is called.
    unmatched_modules = [unmatched_module for _, unmatched_module in unmatched_modules]
    unmatched_group = ModuleGroup(
        modules=unmatched_modules,
        offload_device=offload_device,
        onload_device=onload_device,
        offload_leader=module,
        onload_leader=module,
        parameters=parameters,
        buffers=buffers,
        non_blocking=False,
        stream=None,
        cpu_param_dict=None,
        onload_self=True,
    )
    next_group = matched_module_groups[0] if len(matched_module_groups) > 0 else None
    _apply_group_offloading_hook(module, unmatched_group, force_offload, next_group)


def _apply_group_offloading_leaf_level(
    module: torch.nn.Module,
    offload_device: torch.device,
    onload_device: torch.device,
    force_offload: bool,
    non_blocking: bool,
    stream: Optional[torch.cuda.Stream] = None,
) -> None:
    r"""
    This function applies offloading to groups of leaf modules in a torch.nn.Module. This method has minimal memory
    requirements. However, it can be slower compared to other offloading methods due to the excessive number of device
    synchronizations. When using devices that support streams to overlap data transfer and computation, this method can
    reduce memory usage without any performance degradation.

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

    # Create a pinned CPU parameter dict for async data transfer if streams are to be used
    cpu_param_dict = None
    if stream is not None:
        for param in module.parameters():
            param.data = param.data.cpu().pin_memory()
        cpu_param_dict = {param: param.data for param in module.parameters()}

    # Create module groups for leaf modules and apply group offloading hooks
    modules_with_group_offloading = set()
    for name, submodule in module.named_modules():
        if not isinstance(submodule, _SUPPORTED_PYTORCH_LAYERS):
            continue
        group = ModuleGroup(
            modules=[submodule],
            offload_device=offload_device,
            onload_device=onload_device,
            offload_leader=submodule,
            onload_leader=submodule,
            non_blocking=non_blocking,
            stream=stream,
            cpu_param_dict=cpu_param_dict,
            onload_self=True,
        )
        _apply_group_offloading_hook(submodule, group, True, None)
        modules_with_group_offloading.add(name)

    # Parameters and Buffers at all non-leaf levels need to be offloaded/onloaded separately when the forward pass
    # of the module is called
    module_dict = dict(module.named_modules())
    parameters = _gather_parameters_with_no_group_offloading_parent(module, modules_with_group_offloading)
    buffers = _gather_buffers_with_no_group_offloading_parent(module, modules_with_group_offloading)

    # Find closest module parent for each parameter and buffer, and attach group hooks
    parent_to_parameters = {}
    for name, param in parameters:
        parent_name = _find_parent_module_in_module_dict(name, module_dict)
        if parent_name in parent_to_parameters:
            parent_to_parameters[parent_name].append(param)
        else:
            parent_to_parameters[parent_name] = [param]

    parent_to_buffers = {}
    for name, buffer in buffers:
        parent_name = _find_parent_module_in_module_dict(name, module_dict)
        if parent_name in parent_to_buffers:
            parent_to_buffers[parent_name].append(buffer)
        else:
            parent_to_buffers[parent_name] = [buffer]

    parent_names = set(parent_to_parameters.keys()) | set(parent_to_buffers.keys())
    for name in parent_names:
        parameters = parent_to_parameters.get(name, [])
        buffers = parent_to_buffers.get(name, [])
        parent_module = module_dict[name]
        assert getattr(parent_module, "_diffusers_hook", None) is None
        group = ModuleGroup(
            modules=[],
            offload_device=offload_device,
            onload_device=onload_device,
            offload_leader=parent_module,
            onload_leader=parent_module,
            parameters=parameters,
            buffers=buffers,
            non_blocking=non_blocking,
            stream=stream,
            cpu_param_dict=cpu_param_dict,
            onload_self=True,
        )
        _apply_group_offloading_hook(parent_module, group, True, None)

    # This is a dummy group that will handle lazy prefetching from the top-level module to the first leaf module
    unmatched_group = ModuleGroup(
        modules=[],
        offload_device=offload_device,
        onload_device=onload_device,
        offload_leader=module,
        onload_leader=module,
        parameters=None,
        buffers=None,
        non_blocking=False,
        stream=None,
        cpu_param_dict=None,
        onload_self=True,
    )

    # When using streams, we need to know the layer execution order for applying prefetching (to overlap data transfer
    # and computation). Since we don't know the order beforehand, we apply a lazy prefetching hook that will find the
    # execution order and apply prefetching in the correct order.
    if stream is None:
        _apply_group_offloading_hook(module, unmatched_group, force_offload, None)
    else:
        _apply_lazy_group_offloading_hook(module, unmatched_group, force_offload, None)


def _apply_group_offloading_hook(
    module: torch.nn.Module,
    group: ModuleGroup,
    offload_on_init: bool,
    next_group: Optional[ModuleGroup] = None,
) -> None:
    registry = HookRegistry.check_if_exists_or_initialize(module)

    # We may have already registered a group offloading hook if the module had a torch.nn.Parameter whose parent
    # is the current module. In such cases, we don't want to overwrite the existing group offloading hook.
    if registry.get_hook(_GROUP_OFFLOADING) is None:
        hook = GroupOffloadingHook(group, offload_on_init, next_group)
        registry.register_hook(hook, _GROUP_OFFLOADING)


def _apply_lazy_group_offloading_hook(
    module: torch.nn.Module,
    group: ModuleGroup,
    offload_on_init: bool,
    next_group: Optional[ModuleGroup] = None,
) -> None:
    registry = HookRegistry.check_if_exists_or_initialize(module)

    # We may have already registered a group offloading hook if the module had a torch.nn.Parameter whose parent
    # is the current module. In such cases, we don't want to overwrite the existing group offloading hook.
    if registry.get_hook(_GROUP_OFFLOADING) is None:
        hook = GroupOffloadingHook(group, offload_on_init, next_group)
        registry.register_hook(hook, _GROUP_OFFLOADING)

    lazy_prefetch_hook = LazyPrefetchGroupOffloadingHook()
    registry.register_hook(lazy_prefetch_hook, _LAZY_PREFETCH_GROUP_OFFLOADING)


def _gather_parameters_with_no_group_offloading_parent(
    module: torch.nn.Module, modules_with_group_offloading: Set[str]
) -> List[torch.nn.Parameter]:
    parameters = []
    for name, parameter in module.named_parameters():
        has_parent_with_group_offloading = False
        atoms = name.split(".")

        while len(atoms) > 0:
            parent_name = ".".join(atoms)
            if parent_name in modules_with_group_offloading:
                has_parent_with_group_offloading = True
                break
            atoms.pop()

        if not has_parent_with_group_offloading:
            parameters.append((name, parameter))
    return parameters


def _gather_buffers_with_no_group_offloading_parent(
    module: torch.nn.Module, modules_with_group_offloading: Set[str]
) -> List[torch.Tensor]:
    buffers = []
    for name, buffer in module.named_buffers():
        has_parent_with_group_offloading = False
        atoms = name.split(".")

        while len(atoms) > 0:
            parent_name = ".".join(atoms)
            if parent_name in modules_with_group_offloading:
                has_parent_with_group_offloading = True
                break
            atoms.pop()

        if not has_parent_with_group_offloading:
            buffers.append((name, buffer))
    return buffers


def _find_parent_module_in_module_dict(name: str, module_dict: Dict[str, torch.nn.Module]) -> str:
    atoms = name.split(".")
    while len(atoms) > 0:
        parent_name = ".".join(atoms)
        if parent_name in module_dict:
            return parent_name
        atoms.pop()
    return ""
