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

import copy
import time
import uuid
from collections import OrderedDict
from itertools import combinations
from typing import Any, Dict, List, Optional, Union

import torch

from ..utils import (
    is_accelerate_available,
    logging,
)


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
        self.collections = OrderedDict()  # collection_name -> set of component_names
        self.model_hooks = None
        self._auto_offload_enabled = False

    def _lookup_ids(self, name=None, collection=None, load_id=None, components: OrderedDict = None):
        """
        Lookup component_ids by name, collection, or load_id.
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

    def add(self, name, component, collection: Optional[str] = None):
        component_id = f"{name}_{uuid.uuid4()}"

        # check for duplicated components
        for comp_id, comp in self.components.items():
            if comp == component:
                comp_name = self._id_to_name(comp_id)
                if comp_name == name:
                    logger.warning(f"component '{name}' already exists as '{comp_id}'")
                    component_id = comp_id
                    break
                else:
                    logger.warning(
                        f"Adding component '{name}' as '{component_id}', but it is duplicate of '{comp_id}'"
                        f"To remove a duplicate, call `components_manager.remove('<component_id>')`."
                    )

        # check for duplicated load_id and warn (we do not delete for you)
        if hasattr(component, "_diffusers_load_id") and component._diffusers_load_id != "null":
            components_with_same_load_id = self._lookup_ids(load_id=component._diffusers_load_id)
            components_with_same_load_id = [id for id in components_with_same_load_id if id != component_id]

            if components_with_same_load_id:
                existing = ", ".join(components_with_same_load_id)
                logger.warning(
                    f"Adding component '{component_id}', but it has duplicate load_id '{component._diffusers_load_id}' with existing components: {existing}. "
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
                    logger.info(f"Removing existing {name} from collection '{collection}': {comp_id}")
                    self.remove(comp_id)
                self.collections[collection].add(component_id)
                logger.info(f"Added component '{name}' in collection '{collection}': {component_id}")
        else:
            logger.info(f"Added component '{name}' as '{component_id}'")

        if self._auto_offload_enabled:
            self.enable_auto_cpu_offload(self._auto_offload_device)

        return component_id

    def remove(self, component_id: str = None):
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

    def get(
        self,
        names: Union[str, List[str]] = None,
        collection: Optional[str] = None,
        load_id: Optional[str] = None,
        as_name_component_tuples: bool = False,
    ):
        """
        Select components by name with simple pattern matching.

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
            as_name_component_tuples: If True, returns a list of (name, component) tuples using base names
                                     instead of a dictionary with component IDs as keys

        Returns:
            Dictionary mapping component IDs to components or list of (base_name, component) tuples if
            as_name_component_tuples=True
        """

        selected_ids = self._lookup_ids(collection=collection, load_id=load_id)
        components = {k: self.components[k] for k in selected_ids}

        # Helper to extract base name from component_id
        def get_base_name(component_id):
            parts = component_id.split("_")
            # If the last part looks like a UUID, remove it
            if len(parts) > 1 and len(parts[-1]) >= 8 and "-" in parts[-1]:
                return "_".join(parts[:-1])
            return component_id

        if names is None:
            if as_name_component_tuples:
                return [(get_base_name(comp_id), comp) for comp_id, comp in components.items()]
            else:
                return components

        # Create mapping from component_id to base_name for all components
        base_names = {comp_id: get_base_name(comp_id) for comp_id in components.keys()}

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

        if isinstance(names, str):
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
                    logger.info(
                        f"Getting all components except those with base name '{names}': {list(matches.keys())}"
                    )
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

            if as_name_component_tuples:
                return [(base_names[comp_id], comp) for comp_id, comp in matches.items()]
            else:
                return matches

        elif isinstance(names, list):
            results = {}
            for name in names:
                result = self.get(name, collection, load_id, as_name_component_tuples=False)
                results.update(result)

            if as_name_component_tuples:
                return [(base_names[comp_id], comp) for comp_id, comp in results.items()]
            else:
                return results

        else:
            raise ValueError(f"Invalid type for names: {type(names)}")

    def enable_auto_cpu_offload(self, device: Union[str, int, torch.device] = "cuda", memory_reserve_margin="3GB"):
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

    # YiYi TODO: add quantization info
    def get_model_info(
        self, component_id: str, fields: Optional[Union[str, List[str]]] = None
    ) -> Optional[Dict[str, Any]]:
        """Get comprehensive information about a component.

        Args:
            component_id: Name of the component to get info for
            fields: Optional field(s) to return. Can be a string for single field or list of fields.
                   If None, returns all fields.

        Returns:
            Dictionary containing requested component metadata. If fields is specified, returns only those fields. If a
            single field is requested as string, returns just that field's value.
        """
        if component_id not in self.components:
            raise ValueError(f"Component '{component_id}' not found in ComponentsManager")

        component = self.components[component_id]

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
                    "size_gb": get_memory_footprint(component) / (1024**3),
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
            if isinstance(fields, str):
                # Single field requested, return just that value
                return {fields: info.get(fields)}
            else:
                # List of fields requested, return dict with just those fields
                return {k: v for k, v in info.items() if k in fields}

        return info

    def __repr__(self):
        # Helper to get simple name without UUID
        def get_simple_name(name):
            # Extract the base name by splitting on underscore and taking first part
            # This assumes names are in format "name_uuid"
            parts = name.split("_")
            # If we have at least 2 parts and the last part looks like a UUID, remove it
            if len(parts) > 1 and len(parts[-1]) >= 8 and "-" in parts[-1]:
                return "_".join(parts[:-1])
            return name

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

        # Get all simple names to calculate width
        simple_names = [get_simple_name(id) for id in self.components.keys()]

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
            "name": max(15, max(len(name) for name in simple_names)),
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
            output += f"{'Name':<{col_widths['name']}} | {'Class':<{col_widths['class']}} | "
            output += f"{'Device: act(exec)':<{col_widths['device']}} | {'Dtype':<{col_widths['dtype']}} | "
            output += f"{'Size (GB)':<{col_widths['size']}} | {'Load ID':<{col_widths['load_id']}} | Collection\n"
            output += dash_line

            # Model entries
            for name, component in models.items():
                info = self.get_model_info(name)
                simple_name = get_simple_name(name)
                device_str = format_device(component, info)
                dtype = str(component.dtype) if hasattr(component, "dtype") else "N/A"
                load_id = get_load_id(component)

                # Print first collection on the main line
                first_collection = component_collections[name][0] if component_collections[name] else "N/A"

                output += f"{simple_name:<{col_widths['name']}} | {info['class_name']:<{col_widths['class']}} | "
                output += f"{device_str:<{col_widths['device']}} | {dtype:<{col_widths['dtype']}} | "
                output += f"{info['size_gb']:<{col_widths['size']}.2f} | {load_id:<{col_widths['load_id']}} | {first_collection}\n"

                # Print additional collections on separate lines if they exist
                for i in range(1, len(component_collections[name])):
                    collection = component_collections[name][i]
                    output += f"{'':<{col_widths['name']}} | {'':<{col_widths['class']}} | "
                    output += f"{'':<{col_widths['device']}} | {'':<{col_widths['dtype']}} | "
                    output += f"{'':<{col_widths['size']}} | {'':<{col_widths['load_id']}} | {collection}\n"

            output += dash_line

        # Other components section
        if others:
            if models:  # Add extra newline if we had models section
                output += "\n"
            output += "Other Components:\n" + dash_line
            # Column headers for other components
            output += f"{'Name':<{col_widths['name']}} | {'Class':<{col_widths['class']}} | Collection\n"
            output += dash_line

            # Other component entries
            for name, component in others.items():
                info = self.get_model_info(name)
                simple_name = get_simple_name(name)

                # Print first collection on the main line
                first_collection = component_collections[name][0] if component_collections[name] else "N/A"

                output += f"{simple_name:<{col_widths['name']}} | {component.__class__.__name__:<{col_widths['class']}} | {first_collection}\n"

                # Print additional collections on separate lines if they exist
                for i in range(1, len(component_collections[name])):
                    collection = component_collections[name][i]
                    output += f"{'':<{col_widths['name']}} | {'':<{col_widths['class']}} | {collection}\n"

            output += dash_line

        # Add additional component info
        output += "\nAdditional Component Info:\n" + "=" * 50 + "\n"
        for name in self.components:
            info = self.get_model_info(name)
            if info is not None and (info.get("adapters") is not None or info.get("ip_adapter")):
                simple_name = get_simple_name(name)
                output += f"\n{simple_name}:\n"
                if info.get("adapters") is not None:
                    output += f"  Adapters: {info['adapters']}\n"
                if info.get("ip_adapter"):
                    output += "  IP-Adapter: Enabled\n"

        return output

    def from_pretrained(self, pretrained_model_name_or_path, prefix: Optional[str] = None, **kwargs):
        """
        Load components from a pretrained model and add them to the manager.

        Args:
            pretrained_model_name_or_path (str): The path or identifier of the pretrained model
            prefix (str, optional): Prefix to add to all component names loaded from this model.
                                  If provided, components will be named as "{prefix}_{component_name}"
            **kwargs: Additional arguments to pass to DiffusionPipeline.from_pretrained()
        """
        subfolder = kwargs.pop("subfolder", None)
        # YiYi TODO: extend AutoModel to support non-diffusers models
        if subfolder:
            from ..models import AutoModel

            component = AutoModel.from_pretrained(pretrained_model_name_or_path, subfolder=subfolder, **kwargs)
            component_name = f"{prefix}_{subfolder}" if prefix else subfolder
            if component_name not in self.components:
                self.add(component_name, component)
            else:
                logger.warning(
                    f"Component '{component_name}' already exists in ComponentsManager and will not be added. To add it, either:\n"
                    f"1. remove the existing component with remove('{component_name}')\n"
                    f"2. Use a different prefix: add_from_pretrained(..., prefix='{prefix}_2')"
                )
        else:
            from ..pipelines.pipeline_utils import DiffusionPipeline

            pipe = DiffusionPipeline.from_pretrained(pretrained_model_name_or_path, **kwargs)
            for name, component in pipe.components.items():
                if component is None:
                    continue

                # Add prefix if specified
                component_name = f"{prefix}_{name}" if prefix else name

                if component_name not in self.components:
                    self.add(component_name, component)
                else:
                    logger.warning(
                        f"Component '{component_name}' already exists in ComponentsManager and will not be added. To add it, either:\n"
                        f"1. remove the existing component with remove('{component_name}')\n"
                        f"2. Use a different prefix: add_from_pretrained(..., prefix='{prefix}_2')"
                    )

    def get_one(
        self,
        component_id: Optional[str] = None,
        name: Optional[str] = None,
        collection: Optional[str] = None,
        load_id: Optional[str] = None,
    ) -> Any:
        """
        Get a single component by name. Raises an error if multiple components match or none are found.

        Args:
            name: Component name or pattern
            collection: Optional collection to filter by
            load_id: Optional load_id to filter by

        Returns:
            A single component

        Raises:
            ValueError: If no components match or multiple components match
        """

        # if component_id is provided, return the component
        if component_id is not None and (name is not None or collection is not None or load_id is not None):
            raise ValueError(" if component_id is provided, name, collection, and load_id must be None")
        elif component_id is not None:
            if component_id not in self.components:
                raise ValueError(f"Component '{component_id}' not found in ComponentsManager")
            return self.components[component_id]

        results = self.get(name, collection, load_id)

        if not results:
            raise ValueError(f"No components found matching '{name}'")

        if len(results) > 1:
            raise ValueError(f"Multiple components found matching '{name}': {list(results.keys())}")

        return next(iter(results.values()))


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
