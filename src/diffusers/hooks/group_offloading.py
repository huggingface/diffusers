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

import hashlib
import os
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Union

import safetensors.torch
import torch

from ..utils import get_logger, is_accelerate_available
from ._common import _GO_LC_SUPPORTED_PYTORCH_LAYERS
from .hooks import HookRegistry, ModelHook


if is_accelerate_available():
    from accelerate.hooks import AlignDevicesHook, CpuOffload
    from accelerate.utils import send_to_device


logger = get_logger(__name__)  # pylint: disable=invalid-name


# fmt: off
_GROUP_OFFLOADING = "group_offloading"
_LAYER_EXECUTION_TRACKER = "layer_execution_tracker"
_LAZY_PREFETCH_GROUP_OFFLOADING = "lazy_prefetch_group_offloading"
_GROUP_ID_LAZY_LEAF = "lazy_leafs"
# fmt: on


class GroupOffloadingType(str, Enum):
    BLOCK_LEVEL = "block_level"
    LEAF_LEVEL = "leaf_level"


@dataclass
class GroupOffloadingConfig:
    onload_device: torch.device
    offload_device: torch.device
    offload_type: GroupOffloadingType
    non_blocking: bool
    record_stream: bool
    low_cpu_mem_usage: bool
    num_blocks_per_group: Optional[int] = None
    offload_to_disk_path: Optional[str] = None
    stream: Optional[Union[torch.cuda.Stream, torch.Stream]] = None


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
        stream: Union[torch.cuda.Stream, torch.Stream, None] = None,
        record_stream: Optional[bool] = False,
        low_cpu_mem_usage: bool = False,
        onload_self: bool = True,
        offload_to_disk_path: Optional[str] = None,
        group_id: Optional[int] = None,
    ) -> None:
        self.modules = modules
        self.offload_device = offload_device
        self.onload_device = onload_device
        self.offload_leader = offload_leader
        self.onload_leader = onload_leader
        self.parameters = parameters or []
        self.buffers = buffers or []
        self.non_blocking = non_blocking or stream is not None
        self.stream = stream
        self.record_stream = record_stream
        self.onload_self = onload_self
        self.low_cpu_mem_usage = low_cpu_mem_usage

        self.offload_to_disk_path = offload_to_disk_path
        self._is_offloaded_to_disk = False

        if self.offload_to_disk_path is not None:
            # Instead of `group_id or str(id(self))` we do this because `group_id` can be "" as well.
            self.group_id = group_id if group_id is not None else str(id(self))
            short_hash = _compute_group_hash(self.group_id)
            self.safetensors_file_path = os.path.join(self.offload_to_disk_path, f"group_{short_hash}.safetensors")

            all_tensors = []
            for module in self.modules:
                all_tensors.extend(list(module.parameters()))
                all_tensors.extend(list(module.buffers()))
            all_tensors.extend(self.parameters)
            all_tensors.extend(self.buffers)
            all_tensors = list(dict.fromkeys(all_tensors))  # Remove duplicates

            self.tensor_to_key = {tensor: f"tensor_{i}" for i, tensor in enumerate(all_tensors)}
            self.key_to_tensor = {v: k for k, v in self.tensor_to_key.items()}
            self.cpu_param_dict = {}
        else:
            self.cpu_param_dict = self._init_cpu_param_dict()

        self._torch_accelerator_module = (
            getattr(torch, torch.accelerator.current_accelerator().type)
            if hasattr(torch, "accelerator")
            else torch.cuda
        )

    def _init_cpu_param_dict(self):
        cpu_param_dict = {}
        if self.stream is None:
            return cpu_param_dict

        for module in self.modules:
            for param in module.parameters():
                cpu_param_dict[param] = param.data.cpu() if self.low_cpu_mem_usage else param.data.cpu().pin_memory()
            for buffer in module.buffers():
                cpu_param_dict[buffer] = (
                    buffer.data.cpu() if self.low_cpu_mem_usage else buffer.data.cpu().pin_memory()
                )

        for param in self.parameters:
            cpu_param_dict[param] = param.data.cpu() if self.low_cpu_mem_usage else param.data.cpu().pin_memory()

        for buffer in self.buffers:
            cpu_param_dict[buffer] = buffer.data.cpu() if self.low_cpu_mem_usage else buffer.data.cpu().pin_memory()

        return cpu_param_dict

    @contextmanager
    def _pinned_memory_tensors(self):
        try:
            pinned_dict = {
                param: tensor.pin_memory() if not tensor.is_pinned() else tensor
                for param, tensor in self.cpu_param_dict.items()
            }
            yield pinned_dict
        finally:
            pinned_dict = None

    def _transfer_tensor_to_device(self, tensor, source_tensor):
        tensor.data = source_tensor.to(self.onload_device, non_blocking=self.non_blocking)
        if self.record_stream:
            tensor.data.record_stream(self._torch_accelerator_module.current_stream())

    def _process_tensors_from_modules(self, pinned_memory=None):
        for group_module in self.modules:
            for param in group_module.parameters():
                source = pinned_memory[param] if pinned_memory else param.data
                self._transfer_tensor_to_device(param, source)
            for buffer in group_module.buffers():
                source = pinned_memory[buffer] if pinned_memory else buffer.data
                self._transfer_tensor_to_device(buffer, source)

        for param in self.parameters:
            source = pinned_memory[param] if pinned_memory else param.data
            self._transfer_tensor_to_device(param, source)

        for buffer in self.buffers:
            source = pinned_memory[buffer] if pinned_memory else buffer.data
            self._transfer_tensor_to_device(buffer, source)

    def _onload_from_disk(self):
        if self.stream is not None:
            # Wait for previous Host->Device transfer to complete
            self.stream.synchronize()

        context = nullcontext() if self.stream is None else self._torch_accelerator_module.stream(self.stream)
        current_stream = self._torch_accelerator_module.current_stream() if self.record_stream else None

        with context:
            # Load to CPU (if using streams) or directly to target device, pin, and async copy to device
            device = str(self.onload_device) if self.stream is None else "cpu"
            loaded_tensors = safetensors.torch.load_file(self.safetensors_file_path, device=device)

            if self.stream is not None:
                for key, tensor_obj in self.key_to_tensor.items():
                    pinned_tensor = loaded_tensors[key].pin_memory()
                    tensor_obj.data = pinned_tensor.to(self.onload_device, non_blocking=self.non_blocking)
                    if self.record_stream:
                        tensor_obj.data.record_stream(current_stream)
            else:
                onload_device = (
                    self.onload_device.type if isinstance(self.onload_device, torch.device) else self.onload_device
                )
                loaded_tensors = safetensors.torch.load_file(self.safetensors_file_path, device=onload_device)
                for key, tensor_obj in self.key_to_tensor.items():
                    tensor_obj.data = loaded_tensors[key]

    def _onload_from_memory(self):
        if self.stream is not None:
            # Wait for previous Host->Device transfer to complete
            self.stream.synchronize()

        context = nullcontext() if self.stream is None else self._torch_accelerator_module.stream(self.stream)
        with context:
            if self.stream is not None:
                with self._pinned_memory_tensors() as pinned_memory:
                    self._process_tensors_from_modules(pinned_memory)
            else:
                self._process_tensors_from_modules(None)

    def _offload_to_disk(self):
        # TODO: we can potentially optimize this code path by checking if the _all_ the desired
        # safetensor files exist on the disk and if so, skip this step entirely, reducing IO
        # overhead. Currently, we just check if the given `safetensors_file_path` exists and if not
        # we perform a write.
        # Check if the file has been saved in this session or if it already exists on disk.
        if not self._is_offloaded_to_disk and not os.path.exists(self.safetensors_file_path):
            os.makedirs(os.path.dirname(self.safetensors_file_path), exist_ok=True)
            tensors_to_save = {key: tensor.data.to(self.offload_device) for tensor, key in self.tensor_to_key.items()}
            safetensors.torch.save_file(tensors_to_save, self.safetensors_file_path)

        # The group is now considered offloaded to disk for the rest of the session.
        self._is_offloaded_to_disk = True

        # We do this to free up the RAM which is still holding the up tensor data.
        for tensor_obj in self.tensor_to_key.keys():
            tensor_obj.data = torch.empty_like(tensor_obj.data, device=self.offload_device)

    def _offload_to_memory(self):
        if self.stream is not None:
            if not self.record_stream:
                self._torch_accelerator_module.current_stream().synchronize()

            for group_module in self.modules:
                for param in group_module.parameters():
                    param.data = self.cpu_param_dict[param]
            for param in self.parameters:
                param.data = self.cpu_param_dict[param]
            for buffer in self.buffers:
                buffer.data = self.cpu_param_dict[buffer]
        else:
            for group_module in self.modules:
                group_module.to(self.offload_device, non_blocking=False)
            for param in self.parameters:
                param.data = param.data.to(self.offload_device, non_blocking=False)
            for buffer in self.buffers:
                buffer.data = buffer.data.to(self.offload_device, non_blocking=False)

    @torch.compiler.disable()
    def onload_(self):
        r"""Onloads the group of parameters to the onload_device."""
        if self.offload_to_disk_path is not None:
            self._onload_from_disk()
        else:
            self._onload_from_memory()

    @torch.compiler.disable()
    def offload_(self):
        r"""Offloads the group of parameters to the offload_device."""
        if self.offload_to_disk_path:
            self._offload_to_disk()
        else:
            self._offload_to_memory()


class GroupOffloadingHook(ModelHook):
    r"""
    A hook that offloads groups of torch.nn.Module to the CPU for storage and onloads to accelerator device for
    computation. Each group has one "onload leader" module that is responsible for onloading, and an "offload leader"
    module that is responsible for offloading. If prefetching is enabled, the onload leader of the previous module
    group is responsible for onloading the current module group.
    """

    _is_stateful = False

    def __init__(self, group: ModuleGroup, *, config: GroupOffloadingConfig) -> None:
        self.group = group
        self.next_group: Optional[ModuleGroup] = None
        self.config = config

    def initialize_hook(self, module: torch.nn.Module) -> torch.nn.Module:
        if self.group.offload_leader == module:
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

            should_onload_next_group = self.next_group is not None and not self.next_group.onload_self
            if should_onload_next_group:
                self.next_group.onload_()

            should_synchronize = (
                not self.group.onload_self and self.group.stream is not None and not should_onload_next_group
            )
            if should_synchronize:
                # If this group didn't onload itself, it means it was asynchronously onloaded by the
                # previous group. We need to synchronize the side stream to ensure parameters
                # are completely loaded to proceed with forward pass. Without this, uninitialized
                # weights will be used in the computation, leading to incorrect results
                # Also, we should only do this synchronization if we don't already do it from the sync call in
                # self.next_group.onload_, hence the `not should_onload_next_group` check.
                self.group.stream.synchronize()

        args = send_to_device(args, self.group.onload_device, non_blocking=self.group.non_blocking)
        kwargs = send_to_device(kwargs, self.group.onload_device, non_blocking=self.group.non_blocking)
        return args, kwargs

    def post_forward(self, module: torch.nn.Module, output):
        if self.group.offload_leader == module:
            self.group.offload_()
        return output


class LazyPrefetchGroupOffloadingHook(ModelHook):
    r"""
    A hook, used in conjunction with GroupOffloadingHook, that applies lazy prefetching to groups of torch.nn.Module.
    This hook is used to determine the order in which the layers are executed during the forward pass. Once the layer
    invocation order is known, assignments of the next_group attribute for prefetching can be made, which allows
    prefetching groups in the correct order.
    """

    _is_stateful = False

    def __init__(self):
        self.execution_order: List[Tuple[str, torch.nn.Module]] = []
        self._layer_execution_tracker_module_names = set()

    def initialize_hook(self, module):
        def make_execution_order_update_callback(current_name, current_submodule):
            def callback():
                if not torch.compiler.is_compiling():
                    logger.debug(f"Adding {current_name} to the execution order")
                self.execution_order.append((current_name, current_submodule))

            return callback

        # To every submodule that contains a group offloading hook (at this point, no prefetching is enabled for any
        # of the groups), we add a layer execution tracker hook that will be used to determine the order in which the
        # layers are executed during the forward pass.
        for name, submodule in module.named_modules():
            if name == "" or not hasattr(submodule, "_diffusers_hook"):
                continue

            registry = HookRegistry.check_if_exists_or_initialize(submodule)
            group_offloading_hook = registry.get_hook(_GROUP_OFFLOADING)

            if group_offloading_hook is not None:
                # For the first forward pass, we have to load in a blocking manner
                group_offloading_hook.group.non_blocking = False
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
            if not torch.compiler.is_compiling():
                logger.warning(
                    "It seems like some layers were not executed during the forward pass. This may lead to problems when "
                    "applying lazy prefetching with automatic tracing and lead to device-mismatch related errors. Please "
                    "make sure that all layers are executed during the forward pass. The following layers were not executed:\n"
                    f"{unexecuted_layers=}"
                )

        # Remove the layer execution tracker hooks from the submodules
        base_module_registry = module._diffusers_hook
        registries = [submodule._diffusers_hook for _, submodule in self.execution_order]
        group_offloading_hooks = [registry.get_hook(_GROUP_OFFLOADING) for registry in registries]

        for i in range(num_executed):
            registries[i].remove_hook(_LAYER_EXECUTION_TRACKER, recurse=False)

        # Remove the current lazy prefetch group offloading hook so that it doesn't interfere with the next forward pass
        base_module_registry.remove_hook(_LAZY_PREFETCH_GROUP_OFFLOADING, recurse=False)

        # LazyPrefetchGroupOffloadingHook is only used with streams, so we know that non_blocking should be True.
        # We disable non_blocking for the first forward pass, but need to enable it for the subsequent passes to
        # see the benefits of prefetching.
        for hook in group_offloading_hooks:
            hook.group.non_blocking = True

        # Set required attributes for prefetching
        if num_executed > 0:
            base_module_group_offloading_hook = base_module_registry.get_hook(_GROUP_OFFLOADING)
            base_module_group_offloading_hook.next_group = group_offloading_hooks[0].group
            base_module_group_offloading_hook.next_group.onload_self = False

        for i in range(num_executed - 1):
            name1, _ = self.execution_order[i]
            name2, _ = self.execution_order[i + 1]
            if not torch.compiler.is_compiling():
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
    onload_device: Union[str, torch.device],
    offload_device: Union[str, torch.device] = torch.device("cpu"),
    offload_type: Union[str, GroupOffloadingType] = "block_level",
    num_blocks_per_group: Optional[int] = None,
    non_blocking: bool = False,
    use_stream: bool = False,
    record_stream: bool = False,
    low_cpu_mem_usage: bool = False,
    offload_to_disk_path: Optional[str] = None,
) -> None:
    r"""
    Applies group offloading to the internal layers of a torch.nn.Module. To understand what group offloading is, and
    where it is beneficial, we need to first provide some context on how other supported offloading methods work.

    Typically, offloading is done at two levels:
    - Module-level: In Diffusers, this can be enabled using the `ModelMixin::enable_model_cpu_offload()` method. It
      works by offloading each component of a pipeline to the CPU for storage, and onloading to the accelerator device
      when needed for computation. This method is more memory-efficient than keeping all components on the accelerator,
      but the memory requirements are still quite high. For this method to work, one needs memory equivalent to size of
      the model in runtime dtype + size of largest intermediate activation tensors to be able to complete the forward
      pass.
    - Leaf-level: In Diffusers, this can be enabled using the `ModelMixin::enable_sequential_cpu_offload()` method. It
      works by offloading the lowest leaf-level parameters of the computation graph to the CPU for storage, and
      onloading only the leafs to the accelerator device for computation. This uses the lowest amount of accelerator
      memory, but can be slower due to the excessive number of device synchronizations.

    Group offloading is a middle ground between the two methods. It works by offloading groups of internal layers,
    (either `torch.nn.ModuleList` or `torch.nn.Sequential`). This method uses lower memory than module-level
    offloading. It is also faster than leaf-level/sequential offloading, as the number of device synchronizations is
    reduced.

    Another supported feature (for CUDA devices with support for asynchronous data transfer streams) is the ability to
    overlap data transfer and computation to reduce the overall execution time compared to sequential offloading. This
    is enabled using layer prefetching with streams, i.e., the layer that is to be executed next starts onloading to
    the accelerator device while the current layer is being executed - this increases the memory requirements slightly.
    Note that this implementation also supports leaf-level offloading but can be made much faster when using streams.

    Args:
        module (`torch.nn.Module`):
            The module to which group offloading is applied.
        onload_device (`torch.device`):
            The device to which the group of modules are onloaded.
        offload_device (`torch.device`, defaults to `torch.device("cpu")`):
            The device to which the group of modules are offloaded. This should typically be the CPU. Default is CPU.
        offload_type (`str` or `GroupOffloadingType`, defaults to "block_level"):
            The type of offloading to be applied. Can be one of "block_level" or "leaf_level". Default is
            "block_level".
        offload_to_disk_path (`str`, *optional*, defaults to `None`):
            The path to the directory where parameters will be offloaded. Setting this option can be useful in limited
            RAM environment settings where a reasonable speed-memory trade-off is desired.
        num_blocks_per_group (`int`, *optional*):
            The number of blocks per group when using offload_type="block_level". This is required when using
            offload_type="block_level".
        non_blocking (`bool`, defaults to `False`):
            If True, offloading and onloading is done with non-blocking data transfer.
        use_stream (`bool`, defaults to `False`):
            If True, offloading and onloading is done asynchronously using a CUDA stream. This can be useful for
            overlapping computation and data transfer.
        record_stream (`bool`, defaults to `False`): When enabled with `use_stream`, it marks the current tensor
            as having been used by this stream. It is faster at the expense of slightly more memory usage. Refer to the
            [PyTorch official docs](https://pytorch.org/docs/stable/generated/torch.Tensor.record_stream.html) more
            details.
        low_cpu_mem_usage (`bool`, defaults to `False`):
            If True, the CPU memory usage is minimized by pinning tensors on-the-fly instead of pre-pinning them. This
            option only matters when using streamed CPU offloading (i.e. `use_stream=True`). This can be useful when
            the CPU memory is a bottleneck but may counteract the benefits of using streams.

    Example:
        ```python
        >>> from diffusers import CogVideoXTransformer3DModel
        >>> from diffusers.hooks import apply_group_offloading

        >>> transformer = CogVideoXTransformer3DModel.from_pretrained(
        ...     "THUDM/CogVideoX-5b", subfolder="transformer", torch_dtype=torch.bfloat16
        ... )

        >>> apply_group_offloading(
        ...     transformer,
        ...     onload_device=torch.device("cuda"),
        ...     offload_device=torch.device("cpu"),
        ...     offload_type="block_level",
        ...     num_blocks_per_group=2,
        ...     use_stream=True,
        ... )
        ```
    """

    onload_device = torch.device(onload_device) if isinstance(onload_device, str) else onload_device
    offload_device = torch.device(offload_device) if isinstance(offload_device, str) else offload_device
    offload_type = GroupOffloadingType(offload_type)

    stream = None
    if use_stream:
        if torch.cuda.is_available():
            stream = torch.cuda.Stream()
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            stream = torch.Stream()
        else:
            raise ValueError("Using streams for data transfer requires a CUDA device, or an Intel XPU device.")

    if not use_stream and record_stream:
        raise ValueError("`record_stream` cannot be True when `use_stream=False`.")
    if offload_type == GroupOffloadingType.BLOCK_LEVEL and num_blocks_per_group is None:
        raise ValueError("`num_blocks_per_group` must be provided when using `offload_type='block_level'.")

    _raise_error_if_accelerate_model_or_sequential_hook_present(module)

    config = GroupOffloadingConfig(
        onload_device=onload_device,
        offload_device=offload_device,
        offload_type=offload_type,
        num_blocks_per_group=num_blocks_per_group,
        non_blocking=non_blocking,
        stream=stream,
        record_stream=record_stream,
        low_cpu_mem_usage=low_cpu_mem_usage,
        offload_to_disk_path=offload_to_disk_path,
    )
    _apply_group_offloading(module, config)


def _apply_group_offloading(module: torch.nn.Module, config: GroupOffloadingConfig) -> None:
    if config.offload_type == GroupOffloadingType.BLOCK_LEVEL:
        _apply_group_offloading_block_level(module, config)
    elif config.offload_type == GroupOffloadingType.LEAF_LEVEL:
        _apply_group_offloading_leaf_level(module, config)
    else:
        assert False


def _apply_group_offloading_block_level(module: torch.nn.Module, config: GroupOffloadingConfig) -> None:
    r"""
    This function applies offloading to groups of torch.nn.ModuleList or torch.nn.Sequential blocks. In comparison to
    the "leaf_level" offloading, which is more fine-grained, this offloading is done at the top-level blocks.
    """

    if config.stream is not None and config.num_blocks_per_group != 1:
        logger.warning(
            f"Using streams is only supported for num_blocks_per_group=1. Got {config.num_blocks_per_group=}. Setting it to 1."
        )
        config.num_blocks_per_group = 1

    # Create module groups for ModuleList and Sequential blocks
    modules_with_group_offloading = set()
    unmatched_modules = []
    matched_module_groups = []
    for name, submodule in module.named_children():
        if not isinstance(submodule, (torch.nn.ModuleList, torch.nn.Sequential)):
            unmatched_modules.append((name, submodule))
            modules_with_group_offloading.add(name)
            continue

        for i in range(0, len(submodule), config.num_blocks_per_group):
            current_modules = submodule[i : i + config.num_blocks_per_group]
            group_id = f"{name}_{i}_{i + len(current_modules) - 1}"
            group = ModuleGroup(
                modules=current_modules,
                offload_device=config.offload_device,
                onload_device=config.onload_device,
                offload_to_disk_path=config.offload_to_disk_path,
                offload_leader=current_modules[-1],
                onload_leader=current_modules[0],
                non_blocking=config.non_blocking,
                stream=config.stream,
                record_stream=config.record_stream,
                low_cpu_mem_usage=config.low_cpu_mem_usage,
                onload_self=True,
                group_id=group_id,
            )
            matched_module_groups.append(group)
            for j in range(i, i + len(current_modules)):
                modules_with_group_offloading.add(f"{name}.{j}")

    # Apply group offloading hooks to the module groups
    for i, group in enumerate(matched_module_groups):
        for group_module in group.modules:
            _apply_group_offloading_hook(group_module, group, config=config)

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
        offload_device=config.offload_device,
        onload_device=config.onload_device,
        offload_to_disk_path=config.offload_to_disk_path,
        offload_leader=module,
        onload_leader=module,
        parameters=parameters,
        buffers=buffers,
        non_blocking=False,
        stream=None,
        record_stream=False,
        onload_self=True,
        group_id=f"{module.__class__.__name__}_unmatched_group",
    )
    if config.stream is None:
        _apply_group_offloading_hook(module, unmatched_group, config=config)
    else:
        _apply_lazy_group_offloading_hook(module, unmatched_group, config=config)


def _apply_group_offloading_leaf_level(module: torch.nn.Module, config: GroupOffloadingConfig) -> None:
    r"""
    This function applies offloading to groups of leaf modules in a torch.nn.Module. This method has minimal memory
    requirements. However, it can be slower compared to other offloading methods due to the excessive number of device
    synchronizations. When using devices that support streams to overlap data transfer and computation, this method can
    reduce memory usage without any performance degradation.
    """
    # Create module groups for leaf modules and apply group offloading hooks
    modules_with_group_offloading = set()
    for name, submodule in module.named_modules():
        if not isinstance(submodule, _GO_LC_SUPPORTED_PYTORCH_LAYERS):
            continue
        group = ModuleGroup(
            modules=[submodule],
            offload_device=config.offload_device,
            onload_device=config.onload_device,
            offload_to_disk_path=config.offload_to_disk_path,
            offload_leader=submodule,
            onload_leader=submodule,
            non_blocking=config.non_blocking,
            stream=config.stream,
            record_stream=config.record_stream,
            low_cpu_mem_usage=config.low_cpu_mem_usage,
            onload_self=True,
            group_id=name,
        )
        _apply_group_offloading_hook(submodule, group, config=config)
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
        group = ModuleGroup(
            modules=[],
            offload_device=config.offload_device,
            onload_device=config.onload_device,
            offload_leader=parent_module,
            onload_leader=parent_module,
            offload_to_disk_path=config.offload_to_disk_path,
            parameters=parameters,
            buffers=buffers,
            non_blocking=config.non_blocking,
            stream=config.stream,
            record_stream=config.record_stream,
            low_cpu_mem_usage=config.low_cpu_mem_usage,
            onload_self=True,
            group_id=name,
        )
        _apply_group_offloading_hook(parent_module, group, config=config)

    if config.stream is not None:
        # When using streams, we need to know the layer execution order for applying prefetching (to overlap data transfer
        # and computation). Since we don't know the order beforehand, we apply a lazy prefetching hook that will find the
        # execution order and apply prefetching in the correct order.
        unmatched_group = ModuleGroup(
            modules=[],
            offload_device=config.offload_device,
            onload_device=config.onload_device,
            offload_to_disk_path=config.offload_to_disk_path,
            offload_leader=module,
            onload_leader=module,
            parameters=None,
            buffers=None,
            non_blocking=False,
            stream=None,
            record_stream=False,
            low_cpu_mem_usage=config.low_cpu_mem_usage,
            onload_self=True,
            group_id=_GROUP_ID_LAZY_LEAF,
        )
        _apply_lazy_group_offloading_hook(module, unmatched_group, config=config)


def _apply_group_offloading_hook(
    module: torch.nn.Module,
    group: ModuleGroup,
    *,
    config: GroupOffloadingConfig,
) -> None:
    registry = HookRegistry.check_if_exists_or_initialize(module)

    # We may have already registered a group offloading hook if the module had a torch.nn.Parameter whose parent
    # is the current module. In such cases, we don't want to overwrite the existing group offloading hook.
    if registry.get_hook(_GROUP_OFFLOADING) is None:
        hook = GroupOffloadingHook(group, config=config)
        registry.register_hook(hook, _GROUP_OFFLOADING)


def _apply_lazy_group_offloading_hook(
    module: torch.nn.Module,
    group: ModuleGroup,
    *,
    config: GroupOffloadingConfig,
) -> None:
    registry = HookRegistry.check_if_exists_or_initialize(module)

    # We may have already registered a group offloading hook if the module had a torch.nn.Parameter whose parent
    # is the current module. In such cases, we don't want to overwrite the existing group offloading hook.
    if registry.get_hook(_GROUP_OFFLOADING) is None:
        hook = GroupOffloadingHook(group, config=config)
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


def _raise_error_if_accelerate_model_or_sequential_hook_present(module: torch.nn.Module) -> None:
    if not is_accelerate_available():
        return
    for name, submodule in module.named_modules():
        if not hasattr(submodule, "_hf_hook"):
            continue
        if isinstance(submodule._hf_hook, (AlignDevicesHook, CpuOffload)):
            raise ValueError(
                f"Cannot apply group offloading to a module that is already applying an alternative "
                f"offloading strategy from Accelerate. If you want to apply group offloading, please "
                f"disable the existing offloading strategy first. Offending module: {name} ({type(submodule)})"
            )


def _get_top_level_group_offload_hook(module: torch.nn.Module) -> Optional[GroupOffloadingHook]:
    for submodule in module.modules():
        if hasattr(submodule, "_diffusers_hook"):
            group_offloading_hook = submodule._diffusers_hook.get_hook(_GROUP_OFFLOADING)
            if group_offloading_hook is not None:
                return group_offloading_hook
    return None


def _is_group_offload_enabled(module: torch.nn.Module) -> bool:
    top_level_group_offload_hook = _get_top_level_group_offload_hook(module)
    return top_level_group_offload_hook is not None


def _get_group_onload_device(module: torch.nn.Module) -> torch.device:
    top_level_group_offload_hook = _get_top_level_group_offload_hook(module)
    if top_level_group_offload_hook is not None:
        return top_level_group_offload_hook.config.onload_device
    raise ValueError("Group offloading is not enabled for the provided module.")


def _compute_group_hash(group_id):
    hashed_id = hashlib.sha256(group_id.encode("utf-8")).hexdigest()
    # first 16 characters for a reasonably short but unique name
    return hashed_id[:16]


def _maybe_remove_and_reapply_group_offloading(module: torch.nn.Module) -> None:
    r"""
    Removes the group offloading hook from the module and re-applies it. This is useful when the module has been
    modified in-place and the group offloading hook references-to-tensors needs to be updated. The in-place
    modification can happen in a number of ways, for example, fusing QKV or unloading/loading LoRAs on-the-fly.

    In this implementation, we make an assumption that group offloading has only been applied at the top-level module,
    and therefore all submodules have the same onload and offload devices. If this assumption is not true, say in the
    case where user has applied group offloading at multiple levels, this function will not work as expected.

    There is some performance penalty associated with doing this when non-default streams are used, because we need to
    retrace the execution order of the layers with `LazyPrefetchGroupOffloadingHook`.
    """
    top_level_group_offload_hook = _get_top_level_group_offload_hook(module)

    if top_level_group_offload_hook is None:
        return

    registry = HookRegistry.check_if_exists_or_initialize(module)
    registry.remove_hook(_GROUP_OFFLOADING, recurse=True)
    registry.remove_hook(_LAYER_EXECUTION_TRACKER, recurse=True)
    registry.remove_hook(_LAZY_PREFETCH_GROUP_OFFLOADING, recurse=True)

    _apply_group_offloading(module, top_level_group_offload_hook.config)
