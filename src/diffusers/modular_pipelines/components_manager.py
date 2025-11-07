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

import copy
import time
from collections import OrderedDict
from itertools import combinations
from typing import Any, Dict, List, Optional, Union

import torch

from ..hooks import ModelHook
from ..utils import (
    is_accelerate_available,
    logging,
)
from ..utils.torch_utils import get_device


if is_accelerate_available():
    from accelerate.hooks import add_hook_to_module, remove_hook_from_module
    from accelerate.state import PartialState
    from accelerate.utils import send_to_device
    from accelerate.utils.memory import clear_device_cache
    from accelerate.utils.modeling import convert_file_size_to_int

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class CustomOffloadHook(ModelHook):
    """
    A hook that offloads a model on the CPU until its forward pass is called. It ensures the model and its inputs are
    on the given device. Optionally offloads other models to the CPU before the forward pass is called.

    Args:
        execution_device(`str`, `int` or `torch.device`, *optional*):
            The device on which the model should be executed. Will default to the MPS device if it's available, then
            GPU 0 if there is a GPU, and finally to the CPU.
    """

    no_grad = False

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


# this is the class that user can customize to implement their own offload strategy
class AutoOffloadStrategy:
    """
    Offload strategy that should be used with `CustomOffloadHook` to automatically offload models to the CPU based on
    the available memory on the device.
    """

    # YiYi TODO: instead of memory_reserve_margin, we should let user set the maximum_total_models_size to keep on device
    # the actual memory usage would be higher. But it's simpler this way, and can be tested
    def __init__(self, memory_reserve_margin="3GB"):
        self.memory_reserve_margin = convert_file_size_to_int(memory_reserve_margin)

    def __call__(self, hooks, model_id, model, execution_device):
        if len(hooks) == 0:
            return []

        current_module_size = model.get_memory_footprint()

        device_type = execution_device.type
        device_module = getattr(torch, device_type, torch.cuda)
        try:
            mem_on_device = device_module.mem_get_info(execution_device.index)[0]
        except AttributeError:
            raise AttributeError(f"Do not know how to obtain obtain memory info for {str(device_module)}.")

        mem_on_device = mem_on_device - self.memory_reserve_margin
        if current_module_size < mem_on_device:
            return []

        min_memory_offload = current_module_size - mem_on_device
        logger.info(f" search for models to offload in order to free up {min_memory_offload / 1024**3:.2f} GB memory")

        # exlucde models that's not currently loaded on the device
        module_sizes = dict(
            sorted(
                {hook.model_id: hook.model.get_memory_footprint() for hook in hooks}.items(),
                key=lambda x: x[1],
                reverse=True,
            )
        )

        # YiYi/Dhruv TODO: sort smallest to largest, and offload in that order we would tend to keep the larger models on GPU more often
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


# utils for display component info in a readable format
# TODO: move to a different file
def summarize_dict_by_value_and_parts(d: Dict[str, Any]) -> Dict[str, Any]:
    """Summarizes a dictionary by finding common prefixes that share the same value.

    For a dictionary with dot-separated keys like: {
        'down_blocks.1.attentions.1.transformer_blocks.0.attn2.processor': [0.6],
        'down_blocks.1.attentions.1.transformer_blocks.1.attn2.processor': [0.6],
        'up_blocks.1.attentions.0.transformer_blocks.0.attn2.processor': [0.3],
    }

    Returns a dictionary where keys are the shortest common prefixes and values are their shared values: {
        'down_blocks': [0.6], 'up_blocks': [0.3]
    }
    """
    # First group by values - convert lists to tuples to make them hashable
    value_to_keys = {}
    for key, value in d.items():
        value_tuple = tuple(value) if isinstance(value, list) else value
        if value_tuple not in value_to_keys:
            value_to_keys[value_tuple] = []
        value_to_keys[value_tuple].append(key)

    def find_common_prefix(keys: List[str]) -> str:
        """Find the shortest common prefix among a list of dot-separated keys."""
        if not keys:
            return ""
        if len(keys) == 1:
            return keys[0]

        # Split all keys into parts
        key_parts = [k.split(".") for k in keys]

        # Find how many initial parts are common
        common_length = 0
        for parts in zip(*key_parts):
            if len(set(parts)) == 1:  # All parts at this position are the same
                common_length += 1
            else:
                break

        if common_length == 0:
            return ""

        # Return the common prefix
        return ".".join(key_parts[0][:common_length])

    # Create summary by finding common prefixes for each value group
    summary = {}
    for value_tuple, keys in value_to_keys.items():
        prefix = find_common_prefix(keys)
        if prefix:  # Only add if we found a common prefix
            # Convert tuple back to list if it was originally a list
            value = list(value_tuple) if isinstance(d[keys[0]], list) else value_tuple
            summary[prefix] = value
        else:
            summary[""] = value  # Use empty string if no common prefix

    return summary


class ComponentsManager:
    """
    A central registry and management system for model components across multiple pipelines.

    [`ComponentsManager`] provides a unified way to register, track, and reuse model components (like UNet, VAE, text
    encoders, etc.) across different modular pipelines. It includes features for duplicate detection, memory
    management, and component organization.

    > [!WARNING] > This is an experimental feature and is likely to change in the future.

    Example:
        ```python
        from diffusers import ComponentsManager

        # Create a components manager
        cm = ComponentsManager()

        # Add components
        cm.add("unet", unet_model, collection="sdxl")
        cm.add("vae", vae_model, collection="sdxl")

        # Enable auto offloading
        cm.enable_auto_cpu_offload()

        # Retrieve components
        unet = cm.get_one(name="unet", collection="sdxl")
        ```
    """

    _available_info_fields = [
        "model_id",
        "added_time",
        "collection",
        "class_name",
        "size_gb",
        "adapters",
        "has_hook",
        "execution_device",
        "ip_adapter",
    ]

    def __init__(self):
        self.components = OrderedDict()
        # YiYi TODO: can remove once confirm we don't need this in mellon
        self.added_time = OrderedDict()  # Store when components were added
        self.collections = OrderedDict()  # collection_name -> set of component_names
        self.model_hooks = None
        self._auto_offload_enabled = False

    def _lookup_ids(
        self,
        name: Optional[str] = None,
        collection: Optional[str] = None,
        load_id: Optional[str] = None,
        components: Optional[OrderedDict] = None,
    ):
        """
        Lookup component_ids by name, collection, or load_id. Does not support pattern matching. Returns a set of
        component_ids
        """
        if components is None:
            components = self.components

        if name:
            ids_by_name = set()
            for component_id, component in components.items():
                comp_name = self._id_to_name(component_id)
                if comp_name == name:
                    ids_by_name.add(component_id)
        else:
            ids_by_name = set(components.keys())
        if collection:
            ids_by_collection = set()
            for component_id, component in components.items():
                if component_id in self.collections[collection]:
                    ids_by_collection.add(component_id)
        else:
            ids_by_collection = set(components.keys())
        if load_id:
            ids_by_load_id = set()
            for name, component in components.items():
                if hasattr(component, "_diffusers_load_id") and component._diffusers_load_id == load_id:
                    ids_by_load_id.add(name)
        else:
            ids_by_load_id = set(components.keys())

        ids = ids_by_name.intersection(ids_by_collection).intersection(ids_by_load_id)
        return ids

    @staticmethod
    def _id_to_name(component_id: str):
        return "_".join(component_id.split("_")[:-1])

    def add(self, name: str, component: Any, collection: Optional[str] = None):
        """
        Add a component to the ComponentsManager.

        Args:
            name (str): The name of the component
            component (Any): The component to add
            collection (Optional[str]): The collection to add the component to

        Returns:
            str: The unique component ID, which is generated as "{name}_{id(component)}" where
                 id(component) is Python's built-in unique identifier for the object
        """
        component_id = f"{name}_{id(component)}"
        is_new_component = True

        # check for duplicated components
        for comp_id, comp in self.components.items():
            if comp == component:
                comp_name = self._id_to_name(comp_id)
                if comp_name == name:
                    logger.warning(f"ComponentsManager: component '{name}' already exists as '{comp_id}'")
                    component_id = comp_id
                    is_new_component = False
                    break
                else:
                    logger.warning(
                        f"ComponentsManager: adding component '{name}' as '{component_id}', but it is duplicate of '{comp_id}'"
                        f"To remove a duplicate, call `components_manager.remove('<component_id>')`."
                    )

        # check for duplicated load_id and warn (we do not delete for you)
        if hasattr(component, "_diffusers_load_id") and component._diffusers_load_id != "null":
            components_with_same_load_id = self._lookup_ids(load_id=component._diffusers_load_id)
            components_with_same_load_id = [id for id in components_with_same_load_id if id != component_id]

            if components_with_same_load_id:
                existing = ", ".join(components_with_same_load_id)
                logger.warning(
                    f"ComponentsManager: adding component '{component_id}', but it has duplicate load_id '{component._diffusers_load_id}' with existing components: {existing}. "
                    f"To remove a duplicate, call `components_manager.remove('<component_id>')`."
                )

        # add component to components manager
        self.components[component_id] = component
        self.added_time[component_id] = time.time()

        if collection:
            if collection not in self.collections:
                self.collections[collection] = set()
            if component_id not in self.collections[collection]:
                comp_ids_in_collection = self._lookup_ids(name=name, collection=collection)
                for comp_id in comp_ids_in_collection:
                    logger.warning(
                        f"ComponentsManager: removing existing {name} from collection '{collection}': {comp_id}"
                    )
                    # remove existing component from this collection (if it is not in any other collection, will be removed from ComponentsManager)
                    self.remove_from_collection(comp_id, collection)

                self.collections[collection].add(component_id)
                logger.info(
                    f"ComponentsManager: added component '{name}' in collection '{collection}': {component_id}"
                )
        else:
            logger.info(f"ComponentsManager: added component '{name}' as '{component_id}'")

        if self._auto_offload_enabled and is_new_component:
            self.enable_auto_cpu_offload(self._auto_offload_device)

        return component_id

    def remove_from_collection(self, component_id: str, collection: str):
        """
        Remove a component from a collection.
        """
        if collection not in self.collections:
            logger.warning(f"Collection '{collection}' not found in ComponentsManager")
            return
        if component_id not in self.collections[collection]:
            logger.warning(f"Component '{component_id}' not found in collection '{collection}'")
            return
        # remove from the collection
        self.collections[collection].remove(component_id)
        # check if this component is in any other collection
        comp_colls = [coll for coll, comps in self.collections.items() if component_id in comps]
        if not comp_colls:  # only if no other collection contains this component, remove it
            logger.warning(f"ComponentsManager: removing component '{component_id}' from ComponentsManager")
            self.remove(component_id)

    def remove(self, component_id: str = None):
        """
        Remove a component from the ComponentsManager.

        Args:
            component_id (str): The ID of the component to remove
        """
        if component_id not in self.components:
            logger.warning(f"Component '{component_id}' not found in ComponentsManager")
            return

        component = self.components.pop(component_id)
        self.added_time.pop(component_id)

        for collection in self.collections:
            if component_id in self.collections[collection]:
                self.collections[collection].remove(component_id)

        if self._auto_offload_enabled:
            self.enable_auto_cpu_offload(self._auto_offload_device)
        else:
            if isinstance(component, torch.nn.Module):
                component.to("cpu")
            del component
            import gc

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if torch.xpu.is_available():
                torch.xpu.empty_cache()

    # YiYi TODO: rename to search_components for now, may remove this method
    def search_components(
        self,
        names: Optional[str] = None,
        collection: Optional[str] = None,
        load_id: Optional[str] = None,
        return_dict_with_names: bool = True,
    ):
        """
        Search components by name with simple pattern matching. Optionally filter by collection or load_id.

        Args:
            names: Component name(s) or pattern(s)
                Patterns:
                - "unet" : match any component with base name "unet" (e.g., unet_123abc)
                - "!unet" : everything except components with base name "unet"
                - "unet*" : anything with base name starting with "unet"
                - "!unet*" : anything with base name NOT starting with "unet"
                - "*unet*" : anything with base name containing "unet"
                - "!*unet*" : anything with base name NOT containing "unet"
                - "refiner|vae|unet" : anything with base name exactly matching "refiner", "vae", or "unet"
                - "!refiner|vae|unet" : anything with base name NOT exactly matching "refiner", "vae", or "unet"
                - "unet*|vae*" : anything with base name starting with "unet" OR starting with "vae"
            collection: Optional collection to filter by
            load_id: Optional load_id to filter by
            return_dict_with_names:
                                    If True, returns a dictionary with component names as keys, throw an error if
                                    multiple components with the same name are found If False, returns a dictionary
                                    with component IDs as keys

        Returns:
            Dictionary mapping component names to components if return_dict_with_names=True, or a dictionary mapping
            component IDs to components if return_dict_with_names=False
        """

        # select components based on collection and load_id filters
        selected_ids = self._lookup_ids(collection=collection, load_id=load_id)
        components = {k: self.components[k] for k in selected_ids}

        def get_return_dict(components, return_dict_with_names):
            """
            Create a dictionary mapping component names to components if return_dict_with_names=True, or a dictionary
            mapping component IDs to components if return_dict_with_names=False, throw an error if duplicate component
            names are found when return_dict_with_names=True
            """
            if return_dict_with_names:
                dict_to_return = {}
                for comp_id, comp in components.items():
                    comp_name = self._id_to_name(comp_id)
                    if comp_name in dict_to_return:
                        raise ValueError(
                            f"Duplicate component names found in the search results: {comp_name}, please set `return_dict_with_names=False` to return a dictionary with component IDs as keys"
                        )
                    dict_to_return[comp_name] = comp
                return dict_to_return
            else:
                return components

        # if no names are provided, return the filtered components as it is
        if names is None:
            return get_return_dict(components, return_dict_with_names)

        # if names is not a string, raise an error
        elif not isinstance(names, str):
            raise ValueError(f"Invalid type for `names: {type(names)}, only support string")

        # Create mapping from component_id to base_name for components to be used for pattern matching
        base_names = {comp_id: self._id_to_name(comp_id) for comp_id in components.keys()}

        # Helper function to check if a component matches a pattern based on its base name
        def matches_pattern(component_id, pattern, exact_match=False):
            """
            Helper function to check if a component matches a pattern based on its base name.

            Args:
                component_id: The component ID to check
                pattern: The pattern to match against
                exact_match: If True, only exact matches to base_name are considered
            """
            base_name = base_names[component_id]

            # Exact match with base name
            if exact_match:
                return pattern == base_name

            # Prefix match (ends with *)
            elif pattern.endswith("*"):
                prefix = pattern[:-1]
                return base_name.startswith(prefix)

            # Contains match (starts with *)
            elif pattern.startswith("*"):
                search = pattern[1:-1] if pattern.endswith("*") else pattern[1:]
                return search in base_name

            # Exact match (no wildcards)
            else:
                return pattern == base_name

        # Check if this is a "not" pattern
        is_not_pattern = names.startswith("!")
        if is_not_pattern:
            names = names[1:]  # Remove the ! prefix

        # Handle OR patterns (containing |)
        if "|" in names:
            terms = names.split("|")
            matches = {}

            for comp_id, comp in components.items():
                # For OR patterns with exact names (no wildcards), we do exact matching on base names
                exact_match = all(not (term.startswith("*") or term.endswith("*")) for term in terms)

                # Check if any of the terms match this component
                should_include = any(matches_pattern(comp_id, term, exact_match) for term in terms)

                # Flip the decision if this is a NOT pattern
                if is_not_pattern:
                    should_include = not should_include

                if should_include:
                    matches[comp_id] = comp

            log_msg = "NOT " if is_not_pattern else ""
            match_type = "exactly matching" if exact_match else "matching any of patterns"
            logger.info(f"Getting components {log_msg}{match_type} {terms}: {list(matches.keys())}")

        # Try exact match with a base name
        elif any(names == base_name for base_name in base_names.values()):
            # Find all components with this base name
            matches = {
                comp_id: comp
                for comp_id, comp in components.items()
                if (base_names[comp_id] == names) != is_not_pattern
            }

            if is_not_pattern:
                logger.info(f"Getting all components except those with base name '{names}': {list(matches.keys())}")
            else:
                logger.info(f"Getting components with base name '{names}': {list(matches.keys())}")

        # Prefix match (ends with *)
        elif names.endswith("*"):
            prefix = names[:-1]
            matches = {
                comp_id: comp
                for comp_id, comp in components.items()
                if base_names[comp_id].startswith(prefix) != is_not_pattern
            }
            if is_not_pattern:
                logger.info(f"Getting components NOT starting with '{prefix}': {list(matches.keys())}")
            else:
                logger.info(f"Getting components starting with '{prefix}': {list(matches.keys())}")

        # Contains match (starts with *)
        elif names.startswith("*"):
            search = names[1:-1] if names.endswith("*") else names[1:]
            matches = {
                comp_id: comp
                for comp_id, comp in components.items()
                if (search in base_names[comp_id]) != is_not_pattern
            }
            if is_not_pattern:
                logger.info(f"Getting components NOT containing '{search}': {list(matches.keys())}")
            else:
                logger.info(f"Getting components containing '{search}': {list(matches.keys())}")

        # Substring match (no wildcards, but not an exact component name)
        elif any(names in base_name for base_name in base_names.values()):
            matches = {
                comp_id: comp
                for comp_id, comp in components.items()
                if (names in base_names[comp_id]) != is_not_pattern
            }
            if is_not_pattern:
                logger.info(f"Getting components NOT containing '{names}': {list(matches.keys())}")
            else:
                logger.info(f"Getting components containing '{names}': {list(matches.keys())}")

        else:
            raise ValueError(f"Component or pattern '{names}' not found in ComponentsManager")

        if not matches:
            raise ValueError(f"No components found matching pattern '{names}'")

        return get_return_dict(matches, return_dict_with_names)

    def enable_auto_cpu_offload(self, device: Union[str, int, torch.device] = None, memory_reserve_margin="3GB"):
        """
        Enable automatic CPU offloading for all components.

        The algorithm works as follows:
        1. All models start on CPU by default
        2. When a model's forward pass is called, it's moved to the execution device
        3. If there's insufficient memory, other models on the device are moved back to CPU
        4. The system tries to offload the smallest combination of models that frees enough memory
        5. Models stay on the execution device until another model needs memory and forces them off

        Args:
            device (Union[str, int, torch.device]): The execution device where models are moved for forward passes
            memory_reserve_margin (str): The memory reserve margin to use, default is 3GB. This is the amount of
                                        memory to keep free on the device to avoid running out of memory during model
                                        execution (e.g., for intermediate activations, gradients, etc.)
        """
        if not is_accelerate_available():
            raise ImportError("Make sure to install accelerate to use auto_cpu_offload")

        # TODO: add a warning if mem_get_info isn't available on `device`.

        for name, component in self.components.items():
            if isinstance(component, torch.nn.Module) and hasattr(component, "_hf_hook"):
                remove_hook_from_module(component, recurse=True)

        self.disable_auto_cpu_offload()
        offload_strategy = AutoOffloadStrategy(memory_reserve_margin=memory_reserve_margin)
        if device is None:
            device = get_device()
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
        """
        Disable automatic CPU offloading for all components.
        """
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

    # YiYi TODO: (1) add quantization info
    def get_model_info(
        self,
        component_id: str,
        fields: Optional[Union[str, List[str]]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Get comprehensive information about a component.

        Args:
            component_id (str): Name of the component to get info for
            fields (Optional[Union[str, List[str]]]):
                   Field(s) to return. Can be a string for single field or list of fields. If None, uses the
                   available_info_fields setting.

        Returns:
            Dictionary containing requested component metadata. If fields is specified, returns only those fields.
            Otherwise, returns all fields.
        """
        if component_id not in self.components:
            raise ValueError(f"Component '{component_id}' not found in ComponentsManager")

        component = self.components[component_id]

        # Validate fields if specified
        if fields is not None:
            if isinstance(fields, str):
                fields = [fields]
            for field in fields:
                if field not in self._available_info_fields:
                    raise ValueError(f"Field '{field}' not found in available_info_fields")

        # Build complete info dict first
        info = {
            "model_id": component_id,
            "added_time": self.added_time[component_id],
            "collection": ", ".join([coll for coll, comps in self.collections.items() if component_id in comps])
            or None,
        }

        # Additional info for torch.nn.Module components
        if isinstance(component, torch.nn.Module):
            # Check for hook information
            has_hook = hasattr(component, "_hf_hook")
            execution_device = None
            if has_hook and hasattr(component._hf_hook, "execution_device"):
                execution_device = component._hf_hook.execution_device

            info.update(
                {
                    "class_name": component.__class__.__name__,
                    "size_gb": component.get_memory_footprint() / (1024**3),
                    "adapters": None,  # Default to None
                    "has_hook": has_hook,
                    "execution_device": execution_device,
                }
            )

            # Get adapters if applicable
            if hasattr(component, "peft_config"):
                info["adapters"] = list(component.peft_config.keys())

            # Check for IP-Adapter scales
            if hasattr(component, "_load_ip_adapter_weights") and hasattr(component, "attn_processors"):
                processors = copy.deepcopy(component.attn_processors)
                # First check if any processor is an IP-Adapter
                processor_types = [v.__class__.__name__ for v in processors.values()]
                if any("IPAdapter" in ptype for ptype in processor_types):
                    # Then get scales only from IP-Adapter processors
                    scales = {
                        k: v.scale
                        for k, v in processors.items()
                        if hasattr(v, "scale") and "IPAdapter" in v.__class__.__name__
                    }
                    if scales:
                        info["ip_adapter"] = summarize_dict_by_value_and_parts(scales)

        # If fields specified, filter info
        if fields is not None:
            return {k: v for k, v in info.items() if k in fields}
        else:
            return info

    # YiYi TODO: (1) add display fields, allow user to set which fields to display in the comnponents table
    def __repr__(self):
        # Handle empty components case
        if not self.components:
            return "Components:\n" + "=" * 50 + "\nNo components registered.\n" + "=" * 50

        # Extract load_id if available
        def get_load_id(component):
            if hasattr(component, "_diffusers_load_id"):
                return component._diffusers_load_id
            return "N/A"

        # Format device info compactly
        def format_device(component, info):
            if not info["has_hook"]:
                return str(getattr(component, "device", "N/A"))
            else:
                device = str(getattr(component, "device", "N/A"))
                exec_device = str(info["execution_device"] or "N/A")
                return f"{device}({exec_device})"

        # Get max length of load_ids for models
        load_ids = [
            get_load_id(component)
            for component in self.components.values()
            if isinstance(component, torch.nn.Module) and hasattr(component, "_diffusers_load_id")
        ]
        max_load_id_len = max([15] + [len(str(lid)) for lid in load_ids]) if load_ids else 15

        # Get all collections for each component
        component_collections = {}
        for name in self.components.keys():
            component_collections[name] = []
            for coll, comps in self.collections.items():
                if name in comps:
                    component_collections[name].append(coll)
            if not component_collections[name]:
                component_collections[name] = ["N/A"]

        # Find the maximum collection name length
        all_collections = [coll for colls in component_collections.values() for coll in colls]
        max_collection_len = max(10, max(len(str(c)) for c in all_collections)) if all_collections else 10

        col_widths = {
            "id": max(15, max(len(name) for name in self.components.keys())),
            "class": max(25, max(len(component.__class__.__name__) for component in self.components.values())),
            "device": 20,
            "dtype": 15,
            "size": 10,
            "load_id": max_load_id_len,
            "collection": max_collection_len,
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
            output += f"{'Name_ID':<{col_widths['id']}} | {'Class':<{col_widths['class']}} | "
            output += f"{'Device: act(exec)':<{col_widths['device']}} | {'Dtype':<{col_widths['dtype']}} | "
            output += f"{'Size (GB)':<{col_widths['size']}} | {'Load ID':<{col_widths['load_id']}} | Collection\n"
            output += dash_line

            # Model entries
            for name, component in models.items():
                info = self.get_model_info(name)
                device_str = format_device(component, info)
                dtype = str(component.dtype) if hasattr(component, "dtype") else "N/A"
                load_id = get_load_id(component)

                # Print first collection on the main line
                first_collection = component_collections[name][0] if component_collections[name] else "N/A"

                output += f"{name:<{col_widths['id']}} | {info['class_name']:<{col_widths['class']}} | "
                output += f"{device_str:<{col_widths['device']}} | {dtype:<{col_widths['dtype']}} | "
                output += f"{info['size_gb']:<{col_widths['size']}.2f} | {load_id:<{col_widths['load_id']}} | {first_collection}\n"

                # Print additional collections on separate lines if they exist
                for i in range(1, len(component_collections[name])):
                    collection = component_collections[name][i]
                    output += f"{'':<{col_widths['id']}} | {'':<{col_widths['class']}} | "
                    output += f"{'':<{col_widths['device']}} | {'':<{col_widths['dtype']}} | "
                    output += f"{'':<{col_widths['size']}} | {'':<{col_widths['load_id']}} | {collection}\n"

            output += dash_line

        # Other components section
        if others:
            if models:  # Add extra newline if we had models section
                output += "\n"
            output += "Other Components:\n" + dash_line
            # Column headers for other components
            output += f"{'ID':<{col_widths['id']}} | {'Class':<{col_widths['class']}} | Collection\n"
            output += dash_line

            # Other component entries
            for name, component in others.items():
                info = self.get_model_info(name)

                # Print first collection on the main line
                first_collection = component_collections[name][0] if component_collections[name] else "N/A"

                output += f"{name:<{col_widths['id']}} | {component.__class__.__name__:<{col_widths['class']}} | {first_collection}\n"

                # Print additional collections on separate lines if they exist
                for i in range(1, len(component_collections[name])):
                    collection = component_collections[name][i]
                    output += f"{'':<{col_widths['id']}} | {'':<{col_widths['class']}} | {collection}\n"

            output += dash_line

        # Add additional component info
        output += "\nAdditional Component Info:\n" + "=" * 50 + "\n"
        for name in self.components:
            info = self.get_model_info(name)
            if info is not None and (info.get("adapters") is not None or info.get("ip_adapter")):
                output += f"\n{name}:\n"
                if info.get("adapters") is not None:
                    output += f"  Adapters: {info['adapters']}\n"
                if info.get("ip_adapter"):
                    output += "  IP-Adapter: Enabled\n"

        return output

    def get_one(
        self,
        component_id: Optional[str] = None,
        name: Optional[str] = None,
        collection: Optional[str] = None,
        load_id: Optional[str] = None,
    ) -> Any:
        """
        Get a single component by either:
        - searching name (pattern matching), collection, or load_id.
        - passing in a component_id
        Raises an error if multiple components match or none are found.

        Args:
            component_id (Optional[str]): Optional component ID to get
            name (Optional[str]): Component name or pattern
            collection (Optional[str]): Optional collection to filter by
            load_id (Optional[str]): Optional load_id to filter by

        Returns:
            A single component

        Raises:
            ValueError: If no components match or multiple components match
        """

        if component_id is not None and (name is not None or collection is not None or load_id is not None):
            raise ValueError("If searching by component_id, do not pass name, collection, or load_id")

        # search by component_id
        if component_id is not None:
            if component_id not in self.components:
                raise ValueError(f"Component '{component_id}' not found in ComponentsManager")
            return self.components[component_id]
        # search with name/collection/load_id
        results = self.search_components(name, collection, load_id)

        if not results:
            raise ValueError(f"No components found matching '{name}'")

        if len(results) > 1:
            raise ValueError(f"Multiple components found matching '{name}': {list(results.keys())}")

        return next(iter(results.values()))

    def get_ids(self, names: Union[str, List[str]] = None, collection: Optional[str] = None):
        """
        Get component IDs by a list of names, optionally filtered by collection.

        Args:
            names (Union[str, List[str]]): List of component names
            collection (Optional[str]): Optional collection to filter by

        Returns:
            List[str]: List of component IDs
        """
        ids = set()
        if not isinstance(names, list):
            names = [names]
        for name in names:
            ids.update(self._lookup_ids(name=name, collection=collection))
        return list(ids)

    def get_components_by_ids(self, ids: List[str], return_dict_with_names: Optional[bool] = True):
        """
        Get components by a list of IDs.

        Args:
            ids (List[str]):
                List of component IDs
            return_dict_with_names (Optional[bool]):
                Whether to return a dictionary with component names as keys:

        Returns:
            Dict[str, Any]: Dictionary of components.
                - If return_dict_with_names=True, keys are component names.
                - If return_dict_with_names=False, keys are component IDs.

        Raises:
            ValueError: If duplicate component names are found in the search results when return_dict_with_names=True
        """
        components = {id: self.components[id] for id in ids}

        if return_dict_with_names:
            dict_to_return = {}
            for comp_id, comp in components.items():
                comp_name = self._id_to_name(comp_id)
                if comp_name in dict_to_return:
                    raise ValueError(
                        f"Duplicate component names found in the search results: {comp_name}, please set `return_dict_with_names=False` to return a dictionary with component IDs as keys"
                    )
                dict_to_return[comp_name] = comp
            return dict_to_return
        else:
            return components

    def get_components_by_names(self, names: List[str], collection: Optional[str] = None):
        """
        Get components by a list of names, optionally filtered by collection.

        Args:
            names (List[str]): List of component names
            collection (Optional[str]): Optional collection to filter by

        Returns:
            Dict[str, Any]: Dictionary of components with component names as keys

        Raises:
            ValueError: If duplicate component names are found in the search results
        """
        ids = self.get_ids(names, collection)
        return self.get_components_by_ids(ids)
