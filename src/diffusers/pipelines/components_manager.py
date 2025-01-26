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

from collections import OrderedDict
from itertools import combinations
from typing import List, Optional, Union, Dict, Any

import torch
import time
from dataclasses import dataclass

from ..utils import (
    is_accelerate_available,
    logging,
)
from ..models.modeling_utils import ModelMixin


if is_accelerate_available():
    from accelerate.hooks import ModelHook, add_hook_to_module, remove_hook_from_module
    from accelerate.state import PartialState
    from accelerate.utils import send_to_device
    from accelerate.utils.memory import clear_device_cache
    from accelerate.utils.modeling import convert_file_size_to_int

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# YiYi Notes: copied from modeling_utils.py (decide later where to put this)
def get_memory_footprint(self, return_buffers=True):
    r"""
    Get the memory footprint of a model. This will return the memory footprint of the current model in bytes. Useful to
    benchmark the memory footprint of the current model and design some tests. Solution inspired from the PyTorch
    discussions: https://discuss.pytorch.org/t/gpu-memory-that-model-uses/56822/2

    Arguments:
        return_buffers (`bool`, *optional*, defaults to `True`):
            Whether to return the size of the buffer tensors in the computation of the memory footprint. Buffers are
            tensors that do not require gradients and not registered as parameters. E.g. mean and std in batch norm
            layers. Please see: https://discuss.pytorch.org/t/what-pytorch-means-by-buffers/120266/2
    """
    mem = sum([param.nelement() * param.element_size() for param in self.parameters()])
    if return_buffers:
        mem_bufs = sum([buf.nelement() * buf.element_size() for buf in self.buffers()])
        mem = mem + mem_bufs
    return mem


class CustomOffloadHook(ModelHook):
    """
    A hook that offloads a model on the CPU until its forward pass is called. It ensures the model and its inputs are
    on the given device. Optionally offloads other models to the CPU before the forward pass is called.

    Args:
        execution_device(`str`, `int` or `torch.device`, *optional*):
            The device on which the model should be executed. Will default to the MPS device if it's available, then
            GPU 0 if there is a GPU, and finally to the CPU.
    """

    def __init__(
        self,
        execution_device: Optional[Union[str, int, torch.device]] = None,
        other_hooks: Optional[List["UserCustomOffloadHook"]] = None,
        offload_strategy: Optional["AutoOffloadStrategy"] = None,
    ):
        self.execution_device = execution_device if execution_device is not None else PartialState().default_device
        self.other_hooks = other_hooks
        self.offload_strategy = offload_strategy
        self.model_id = None

    def set_strategy(self, offload_strategy: "AutoOffloadStrategy"):
        self.offload_strategy = offload_strategy

    def add_other_hook(self, hook: "UserCustomOffloadHook"):
        """
        Add a hook to the list of hooks to consider for offloading.
        """
        if self.other_hooks is None:
            self.other_hooks = []
        self.other_hooks.append(hook)

    def init_hook(self, module):
        return module.to("cpu")

    def pre_forward(self, module, *args, **kwargs):
        if module.device != self.execution_device:
            if self.other_hooks is not None:
                hooks_to_offload = [hook for hook in self.other_hooks if hook.model.device == self.execution_device]
                # offload all other hooks
                start_time = time.perf_counter()
                if self.offload_strategy is not None:
                    hooks_to_offload = self.offload_strategy(
                        hooks=hooks_to_offload,
                        model_id=self.model_id,
                        model=module,
                        execution_device=self.execution_device,
                    )
                end_time = time.perf_counter()
                logger.info(
                    f" time taken to apply offload strategy for {self.model_id}: {(end_time - start_time):.2f} seconds"
                )

                for hook in hooks_to_offload:
                    logger.info(
                        f"moving {self.model_id} to {self.execution_device}, offloading {hook.model_id} to cpu"
                    )
                    hook.offload()

                if hooks_to_offload:
                    clear_device_cache()
            module.to(self.execution_device)
        return send_to_device(args, self.execution_device), send_to_device(kwargs, self.execution_device)


class UserCustomOffloadHook:
    """
    A simple hook grouping a model and a `CustomOffloadHook`, which provides easy APIs for to call the init method of
    the hook or remove it entirely.
    """

    def __init__(self, model_id, model, hook):
        self.model_id = model_id
        self.model = model
        self.hook = hook

    def offload(self):
        self.hook.init_hook(self.model)

    def attach(self):
        add_hook_to_module(self.model, self.hook)
        self.hook.model_id = self.model_id

    def remove(self):
        remove_hook_from_module(self.model)
        self.hook.model_id = None

    def add_other_hook(self, hook: "UserCustomOffloadHook"):
        self.hook.add_other_hook(hook)


def custom_offload_with_hook(
    model_id: str,
    model: torch.nn.Module,
    execution_device: Union[str, int, torch.device] = None,
    offload_strategy: Optional["AutoOffloadStrategy"] = None,
):
    hook = CustomOffloadHook(execution_device=execution_device, offload_strategy=offload_strategy)
    user_hook = UserCustomOffloadHook(model_id=model_id, model=model, hook=hook)
    user_hook.attach()
    return user_hook


class AutoOffloadStrategy:
    """
    Offload strategy that should be used with `CustomOffloadHook` to automatically offload models to the CPU based on
    the available memory on the device.
    """

    def __init__(self, memory_reserve_margin="3GB"):
        self.memory_reserve_margin = convert_file_size_to_int(memory_reserve_margin)

    def __call__(self, hooks, model_id, model, execution_device):
        if len(hooks) == 0:
            return []

        current_module_size = get_memory_footprint(model)

        mem_on_device = torch.cuda.mem_get_info(execution_device.index)[0]
        mem_on_device = mem_on_device - self.memory_reserve_margin
        if current_module_size < mem_on_device:
            return []

        min_memory_offload = current_module_size - mem_on_device
        logger.info(f" search for models to offload in order to free up {min_memory_offload / 1024**3:.2f} GB memory")

        # exlucde models that's not currently loaded on the device
        module_sizes = dict(
            sorted(
                {hook.model_id: get_memory_footprint(hook.model) for hook in hooks}.items(),
                key=lambda x: x[1],
                reverse=True,
            )
        )

        def search_best_candidate(module_sizes, min_memory_offload):
            """
            search the optimal combination of models to offload to cpu, given a dictionary of module sizes and a
            minimum memory offload size. the combination of models should add up to the smallest modulesize that is
            larger than `min_memory_offload`
            """
            model_ids = list(module_sizes.keys())
            best_candidate = None
            best_size = float("inf")
            for r in range(1, len(model_ids) + 1):
                for candidate_model_ids in combinations(model_ids, r):
                    candidate_size = sum(
                        module_sizes[candidate_model_id] for candidate_model_id in candidate_model_ids
                    )
                    if candidate_size < min_memory_offload:
                        continue
                    else:
                        if best_candidate is None or candidate_size < best_size:
                            best_candidate = candidate_model_ids
                            best_size = candidate_size

            return best_candidate

        best_offload_model_ids = search_best_candidate(module_sizes, min_memory_offload)

        if best_offload_model_ids is None:
            # if no combination is found, meaning that we cannot meet the memory requirement, offload all models
            logger.warning("no combination of models to offload to cpu is found, offloading all models")
            hooks_to_offload = hooks
        else:
            hooks_to_offload = [hook for hook in hooks if hook.model_id in best_offload_model_ids]

        return hooks_to_offload


class ComponentsManager:
    def __init__(self):
        self.components = OrderedDict()
        self.added_time = OrderedDict()  # Store when components were added
        self.model_hooks = None
        self._auto_offload_enabled = False

    def add(self, name, component):
        if name in self.components:
            logger.warning(f"Overriding existing component '{name}' in ComponentsManager")
        self.components[name] = component
        self.added_time[name] = time.time()
        
        if self._auto_offload_enabled:
            self.enable_auto_cpu_offload(self._auto_offload_device)

    def remove(self, name):
        if name not in self.components:
            logger.warning(f"Component '{name}' not found in ComponentsManager")
            return
            
        self.components.pop(name)
        self.added_time.pop(name)
        
        if self._auto_offload_enabled:
            self.enable_auto_cpu_offload(self._auto_offload_device)

    def get(self, names: Union[str, List[str]]):
        if isinstance(names, str):
            if names not in self.components:
                raise ValueError(f"Component '{names}' not found in ComponentsManager")
            return self.components[names]
        elif isinstance(names, list):
            return {n: self.components[n] for n in names}
        else:
            raise ValueError(f"Invalid type for names: {type(names)}")

    def enable_auto_cpu_offload(self, device, memory_reserve_margin="3GB"):
        for name, component in self.components.items():
            if isinstance(component, torch.nn.Module) and hasattr(component, "_hf_hook"):
                remove_hook_from_module(component, recurse=True)

        self.disable_auto_cpu_offload()
        offload_strategy = AutoOffloadStrategy(memory_reserve_margin=memory_reserve_margin)
        device = torch.device(device)
        if device.index is None:
            device = torch.device(f"{device.type}:{0}")
        all_hooks = []
        for name, component in self.components.items():
            if isinstance(component, torch.nn.Module):
                hook = custom_offload_with_hook(name, component, device, offload_strategy=offload_strategy)
                all_hooks.append(hook)

        for hook in all_hooks:
            other_hooks = [h for h in all_hooks if h is not hook]
            for other_hook in other_hooks:
                if other_hook.hook.execution_device == hook.hook.execution_device:
                    hook.add_other_hook(other_hook)

        self.model_hooks = all_hooks
        self._auto_offload_enabled = True
        self._auto_offload_device = device

    def disable_auto_cpu_offload(self):
        if self.model_hooks is None:
            self._auto_offload_enabled = False
            return

        for hook in self.model_hooks:
            hook.offload()
            hook.remove()
        if self.model_hooks:
            clear_device_cache()
        self.model_hooks = None
        self._auto_offload_enabled = False

    def get_model_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive information about a model component.
        
        Args:
            name: Name of the component to get info for
            
        Returns:
            Dictionary containing model metadata including:
            - model_id: Name of the model
            - class_name: Class name of the model
            - device: Device the model is on
            - dtype: Data type of the model
            - size_gb: Size of the model in GB
            - added_time: Timestamp when model was added
            - active_adapters: List of active adapters (if applicable)
            - attn_proc: List of attention processor types (if applicable)
            Returns None if component is not a torch.nn.Module
        """
        if name not in self.components:
            raise ValueError(f"Component '{name}' not found in ComponentsManager")

        component = self.components[name]
        
        # Only process torch.nn.Module components
        if not isinstance(component, torch.nn.Module):
            return None

        info = {
            "model_id": name,
            "class_name": component.__class__.__name__,
            "device": str(getattr(component, "device", "N/A")),
            "dtype": str(component.dtype) if hasattr(component, "dtype") else None,
            "added_time": self.added_time[name],
            "size_gb": get_memory_footprint(component) / (1024**3),
            "active_adapters": None,  # Default to None
        }

        # Get active adapters if applicable
        if isinstance(component, ModelMixin):
            from peft.tuners.tuners_utils import BaseTunerLayer
            for module in component.modules():
                if isinstance(module, BaseTunerLayer):
                    info["active_adapters"] = module.active_adapters
                    break

        # Get attention processors if applicable
        if hasattr(component, "attn_processors"):
            processors = component.attn_processors
            # Get unique processor types
            processor_types = list(set(str(v.__class__.__name__) for v in processors.values()))
            if processor_types:
                info["attn_proc"] = processor_types

        return info

    def __repr__(self):
        col_widths = {
            "id": max(15, max(len(id) for id in self.components.keys())),
            "class": max(25, max(len(component.__class__.__name__) for component in self.components.values())),
            "device": 10,
            "dtype": 15,
            "size": 10,
        }

        # Create the header lines
        sep_line = "=" * (sum(col_widths.values()) + len(col_widths) * 3 - 1) + "\n"
        dash_line = "-" * (sum(col_widths.values()) + len(col_widths) * 3 - 1) + "\n"

        output = "Components:\n" + sep_line

        # Separate components into models and others
        models = {k: v for k, v in self.components.items() if isinstance(v, torch.nn.Module)}
        others = {k: v for k, v in self.components.items() if not isinstance(v, torch.nn.Module)}

        # Models section
        if models:
            output += "Models:\n" + dash_line
            # Column headers
            output += f"{'Model ID':<{col_widths['id']}} | {'Class':<{col_widths['class']}} | "
            output += f"{'Device':<{col_widths['device']}} | {'Dtype':<{col_widths['dtype']}} | Size (GB) \n"
            output += dash_line

            # Model entries
            for name, component in models.items():
                info = self.get_model_info(name)
                output += f"{name:<{col_widths['id']}} | {info['class_name']:<{col_widths['class']}} | "
                output += (
                    f"{info['device']:<{col_widths['device']}} | "
                    f"{info['dtype']:<{col_widths['dtype']}} | "
                    f"{info['size_gb']:.2f}\n"
                )
            output += dash_line

        # Other components section
        if others:
            if models:  # Add extra newline if we had models section
                output += "\n"
            output += "Other Components:\n" + dash_line
            # Column headers for other components
            output += f"{'Component ID':<{col_widths['id']}} | {'Class':<{col_widths['class']}}\n"
            output += dash_line

            # Other component entries
            for name, component in others.items():
                output += f"{name:<{col_widths['id']}} | {component.__class__.__name__:<{col_widths['class']}}\n"
            output += dash_line

        # Add additional component info
        output += "\nAdditional Component Info:\n" + "=" * 50 + "\n"
        for name in self.components:
            info = self.get_model_info(name)
            if info is not None and (info.get("active_adapters") is not None or info.get("attn_proc")):
                output += f"\n{name}:\n"
                if info.get("active_adapters") is not None:
                    output += f"  Active Adapters: {info['active_adapters']}\n"
                if info.get("attn_proc"):
                    output += f"  Attention Processors: {info['attn_proc']}\n"
                output += f"  Added Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(info['added_time']))}\n"
        
        return output

    def add_from_pretrained(self, pretrained_model_name_or_path, **kwargs):
        from ..pipelines.pipeline_utils import DiffusionPipeline

        pipe = DiffusionPipeline.from_pretrained(pretrained_model_name_or_path, **kwargs)
        for name, component in pipe.components.items():
            if name not in self.components and component is not None:
                self.add(name, component)
            elif name in self.components:
                logger.warning(
                    f"Component '{name}' already exists in ComponentsManager and will not be added. To add it, either:\n"
                    f"1. remove the existing component with remove('{name}')\n"
                    f"2. Use a different name: add('{name}_2', component)"
                )
