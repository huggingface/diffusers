# Copyright 2026 The HuggingFace Team. All rights reserved.
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

from __future__ import annotations

import functools
import time
from collections import OrderedDict, deque
from dataclasses import dataclass
from itertools import combinations
from typing import Any

import torch

from ..hooks import ModelHook
from ..utils import (
    is_accelerate_available,
    logging,
)
from ..utils.torch_utils import get_device
from .components_manager_utils import format_size, format_table, summarize_dict_by_value_and_parts


if is_accelerate_available():
    from accelerate.hooks import add_hook_to_module, remove_hook_from_module
    from accelerate.state import PartialState
    from accelerate.utils import send_to_device
    from accelerate.utils.memory import clear_device_cache
    from accelerate.utils.modeling import convert_file_size_to_int

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# Default number of events an `OffloadRecord` keeps. A generation step produces at most a handful, so this
# holds a long run while staying bounded for a long-lived manager (e.g. in a server).
MAX_RECORDED_OFFLOAD_EVENTS = 10000


def available_device_memory(execution_device: torch.device) -> int:
    """
    The device memory available for new weights right now: what the driver reports free, plus the allocator's
    reusable cache (`mem_get_info` counts the cache as used, but freed tensors in it can be reallocated).
    """
    device_module = getattr(torch, execution_device.type, torch.cuda)
    available_memory = device_module.mem_get_info(execution_device.index)[0]
    if hasattr(device_module, "memory_reserved") and hasattr(device_module, "memory_allocated"):
        available_memory += device_module.memory_reserved(execution_device) - device_module.memory_allocated(
            execution_device
        )
    return available_memory


@dataclass
class OffloadEvent:
    """
    A single thing the offloader did: a model moved onto the execution device (`"onload"`), a model moved back to the
    CPU (`"offload"`), or a forward pass that ran out of device memory (`"oom"`).
    """

    action: str
    model_id: str
    model_size: int | None = None
    # why the model moved: which model needed the room, the OOM retry, attaching a new component, ...
    reason: str | None = None
    # free device memory read just before the onload's eviction decision (`None` if the device cannot report it)
    available_memory: int | None = None
    resident_before: tuple[str, ...] = ()
    offloaded: tuple[str, ...] = ()


class OffloadRecord:
    """
    What the offloader did, in order.

    Every offloading decision appends to this record, so after a run it holds the full sequence of moves, what each
    move cost, and where the forward passes ran out of memory. Printing it shows that sequence as a table followed by
    a summary; [`~OffloadRecord.summary`] returns the same numbers as a dict.
    """

    def __init__(self, maxlen: int = MAX_RECORDED_OFFLOAD_EVENTS):
        self.events: deque[OffloadEvent] = deque(maxlen=maxlen)
        self._recorded = 0

    def add(self, event: OffloadEvent):
        self.events.append(event)
        self._recorded += 1

    def clear(self):
        self.events.clear()
        self._recorded = 0

    @property
    def dropped(self) -> int:
        """How many events fell out of the (bounded) log."""
        return max(self._recorded - len(self.events), 0)

    def summary(self) -> dict[str, Any]:
        onloads = [event for event in self.events if event.action == "onload"]
        offloads = [event for event in self.events if event.action == "offload"]
        resident, peak_co_residency = set(), 0
        for event in self.events:
            if event.action == "onload":
                resident.add(event.model_id)
                peak_co_residency = max(peak_co_residency, len(resident))
            elif event.action == "offload":
                resident.discard(event.model_id)
        return {
            "onloads": len(onloads),
            "offloads": len(offloads),
            "oom_retries": sum(event.action == "oom" for event in self.events),
            "peak_co_residency": peak_co_residency,
            "bytes_onloaded": sum(event.model_size or 0 for event in onloads),
            "bytes_offloaded": sum(event.model_size or 0 for event in offloads),
        }

    def __len__(self):
        return len(self.events)

    def __repr__(self):
        if not self.events:
            return "Offload record: nothing recorded yet"

        # One row per decision: an onload and the evictions it caused (its "needed_by" offload events) share a
        # row. Offloads with other causes (OOM retry, offloading disabled) and OOMs get their own rows.
        sizes = {event.model_id: event.model_size for event in self.events if event.model_size is not None}

        def label(model_id):
            return f"{model_id} ({format_size(sizes.get(model_id))})"

        rows = []
        for event in self.events:
            if event.action == "onload":
                offloaded = ", ".join(label(model_id) for model_id in event.offloaded) or "-"
                rows.append(
                    [label(event.model_id), offloaded, format_size(event.available_memory), event.reason or ""]
                )
            elif event.action == "offload":
                if event.reason is not None and event.reason.startswith("needed_by:"):
                    continue  # shown on the row of the onload that caused it
                rows.append(["-", label(event.model_id), "-", event.reason or ""])
            else:  # oom
                rows.append(["-", "-", "-", f"oom:{event.model_id}"])

        table = format_table(
            ["#", "Onload", "Offloaded", "Available", "Reason"],
            [[str(index), *row] for index, row in enumerate(rows, start=1)],
        )
        lines = [table[0], "-" * len(table[0]), *table[1:]]
        if self.dropped:
            lines.insert(2, f"... {self.dropped} earlier events dropped from the log ...")

        summary = self.summary()
        lines += [
            "-" * len(table[0]),
            f"{summary['onloads']} onloads ({format_size(summary['bytes_onloaded'])}) / "
            f"{summary['offloads']} offloads ({format_size(summary['bytes_offloaded'])}), "
            f"{summary['oom_retries']} OOM retries, peak co-residency {summary['peak_co_residency']}",
        ]
        return "\n".join(lines)


class CustomOffloadHook(ModelHook):
    """
    A hook that offloads a model on the CPU until its forward pass is called. It ensures the model and its inputs are
    on the given device. Optionally offloads other models to the CPU before the forward pass is called.

    Args:
        execution_device(`str`, `int` or `torch.device`, *optional*):
            The device on which the model should be executed. Will default to the MPS device if it's available, then
            GPU 0 if there is a GPU, and finally to the CPU.
        retry_on_oom(`bool`, *optional*, defaults to `True`):
            Whether to recover from a forward pass that runs out of device memory by offloading the other models one
            at a time and retrying. If `False`, the error is raised to the caller.
        record(`OffloadRecord`, *optional*):
            Where to record the moves this hook makes.
    """

    no_grad = False

    def __init__(
        self,
        execution_device: str | int | torch.device | None = None,
        other_hooks: list["UserCustomOffloadHook"] | None = None,
        offload_strategy: "AutoOffloadStrategy" | None = None,
        retry_on_oom: bool = True,
        record: OffloadRecord | None = None,
    ):
        self.execution_device = execution_device if execution_device is not None else PartialState().default_device
        self.other_hooks = other_hooks
        self.offload_strategy = offload_strategy
        self.retry_on_oom = retry_on_oom
        self.record = record if record is not None else OffloadRecord()
        self.model_id = None
        self._wrapped_methods = {}

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
            # read before any eviction: this is the number the eviction decision is based on
            device_module = getattr(torch, self.execution_device.type, torch.cuda)
            available_memory = (
                available_device_memory(self.execution_device) if hasattr(device_module, "mem_get_info") else None
            )
            resident_before, hooks_to_offload = (), []
            if self.other_hooks is not None:
                hooks_to_offload = [hook for hook in self.other_hooks if hook.model.device == self.execution_device]
                resident_before = tuple(hook.model_id for hook in hooks_to_offload)
                # offload all other hooks
                start_time = time.perf_counter()
                if self.offload_strategy is not None:
                    hooks_to_offload = self.offload_strategy(
                        hooks=hooks_to_offload,
                        model_id=self.model_id,
                        model=module,
                        execution_device=self.execution_device,
                    )
                elapsed = time.perf_counter() - start_time
                logger.info(f" time taken to apply offload strategy for {self.model_id}: {elapsed:.2f} seconds")

                for hook in hooks_to_offload:
                    logger.info(
                        f"moving {self.model_id} to {self.execution_device}, offloading {hook.model_id} to cpu"
                    )
                    hook.offload(reason=f"needed_by:{self.model_id}")

                if hooks_to_offload:
                    clear_device_cache()
            module.to(self.execution_device)
            self.record.add(
                OffloadEvent(
                    action="onload",
                    model_id=self.model_id,
                    model_size=module.get_memory_footprint(),
                    available_memory=available_memory,
                    resident_before=resident_before,
                    offloaded=tuple(hook.model_id for hook in hooks_to_offload),
                )
            )
        return send_to_device(args, self.execution_device), send_to_device(kwargs, self.execution_device)

    def resident_other_hooks(self):
        """
        The other managed models that are currently on the execution device.
        """
        return [
            other
            for other in (self.other_hooks or [])
            # compare with a normalized index: model.device reports no index for the default device
            if other.model.device.type == self.execution_device.type
            and (other.model.device.index or 0) == (self.execution_device.index or 0)
        ]

    # YiYi TODO: diffusers' own `ModelHook` (`diffusers.hooks.hooks`) supports a `new_forward` around-hook, which
    # would replace this manual wrapping - we may migrate this class to it in the future.
    def wrap_forward(self, module):
        # `pre_forward`/`post_forward` cannot see an exception the forward pass raises, so wrap the (already hooked)
        # entry points here. `forward` is how most models run; autoencoders enter through `encode`/`decode` instead
        # (decorated with `apply_forward_hook`, which fires `pre_forward` but routes around `forward`). On an OOM,
        # free the smallest resident model and retry, escalating until it fits. The memory readings cannot guide
        # this: the failed forward has already unwound, so its activations are freed and the device looks free again.
        # Inference only: the retried forward re-runs cleanly from its original inputs.
        if not self.retry_on_oom:
            return

        def with_oom_retry(entry_point):
            @functools.wraps(entry_point)
            def run_with_oom_retry(*args, **kwargs):
                # each pass offloads one more model, so this terminates once they all have been. Tracked explicitly
                # rather than by device, so that a model that did not actually move cannot be picked twice.
                offloaded = set()
                while True:
                    try:
                        return entry_point(*args, **kwargs)
                    except torch.OutOfMemoryError as e:
                        if not self.retry_on_oom:
                            raise
                        self.record.add(OffloadEvent(action="oom", model_id=self.model_id))
                        resident = sorted(
                            (hook for hook in self.resident_other_hooks() if id(hook) not in offloaded),
                            key=lambda hook: hook.model.get_memory_footprint(),
                        )
                        if not resident:
                            raise torch.OutOfMemoryError(
                                f"{self.model_id} ran out of device memory ({e}) with every other managed model "
                                "already offloaded, so it does not fit on its own. Consider group offloading "
                                "(`ModelMixin.enable_group_offload`), which offloads a single model in groups of "
                                "internal layers."
                            ) from e
                        smallest = resident[0]
                        logger.warning(
                            f"{self.model_id} ran out of device memory ({e}); offloading {smallest.model_id} and "
                            "retrying. If this happens repeatedly, set a larger `memory_reserve` in "
                            "`enable_auto_cpu_offload`."
                        )
                        smallest.offload(reason=f"oom_retry:{self.model_id}")
                        offloaded.add(id(smallest))
                        clear_device_cache()

            return run_with_oom_retry

        for name in ("forward", "encode", "decode"):
            entry_point = getattr(module, name, None)
            if callable(entry_point):
                self._wrapped_methods[name] = entry_point
                setattr(module, name, with_oom_retry(entry_point))

    def unwrap_forward(self, module):
        for name, entry_point in self._wrapped_methods.items():
            setattr(module, name, entry_point)
        self._wrapped_methods.clear()


class UserCustomOffloadHook:
    """
    A simple hook grouping a model and a `CustomOffloadHook`, which provides easy APIs for to call the init method of
    the hook or remove it entirely.
    """

    def __init__(self, model_id, model, hook):
        self.model_id = model_id
        self.model = model
        self.hook = hook

    def offload(self, reason: str | None = None):
        was_resident = self.model.device.type == self.hook.execution_device.type
        self.hook.init_hook(self.model)
        if was_resident:
            self.hook.record.add(
                OffloadEvent(
                    action="offload",
                    model_id=self.model_id,
                    model_size=self.model.get_memory_footprint(),
                    reason=reason,
                )
            )

    def attach(self):
        add_hook_to_module(self.model, self.hook)
        self.hook.model_id = self.model_id
        self.hook.wrap_forward(self.model)

    def remove(self):
        self.hook.unwrap_forward(self.model)
        remove_hook_from_module(self.model)
        self.hook.model_id = None

    def add_other_hook(self, hook: "UserCustomOffloadHook"):
        self.hook.add_other_hook(hook)


def custom_offload_with_hook(
    model_id: str,
    model: torch.nn.Module,
    execution_device: str | int | torch.device = None,
    offload_strategy: "AutoOffloadStrategy" | None = None,
    retry_on_oom: bool = True,
    record: OffloadRecord | None = None,
):
    hook = CustomOffloadHook(
        execution_device=execution_device,
        offload_strategy=offload_strategy,
        retry_on_oom=retry_on_oom,
        record=record,
    )
    user_hook = UserCustomOffloadHook(model_id=model_id, model=model, hook=hook)
    user_hook.attach()
    return user_hook


# this is the class that user can customize to implement their own offload strategy
class AutoOffloadStrategy:
    """
    Offload strategy that should be used with `CustomOffloadHook` to automatically offload models to the CPU so the
    incoming model fits on the device: at each offload decision, check the memory actually available on the device
    and keep `memory_reserve` of it free.

    The sizes cover the weights managed by this strategy only — the actual memory requirements will include
    activations and any other allocations, so `memory_reserve` covers exactly that headroom; a `memory_reserve` of 0
    packs the device as full as the weights allow, relying on the OOM retry as a backstop.
    """

    def __init__(self, memory_reserve="3GB"):
        self.memory_reserve = convert_file_size_to_int(memory_reserve)

    def __call__(self, hooks, model_id, model, execution_device):
        if len(hooks) == 0:
            return []

        try:
            current_module_size = model.get_memory_footprint()
        except AttributeError:
            raise AttributeError(f"Do not know how to compute memory footprint of `{model.__class__.__name__}.")

        resident_size = sum(hook.model.get_memory_footprint() for hook in hooks)

        available_memory = available_device_memory(execution_device)
        if current_module_size <= available_memory - self.memory_reserve:
            return []

        min_memory_offload = current_module_size - (available_memory - self.memory_reserve)
        if min_memory_offload >= resident_size:
            logger.warning(
                f"fitting {model_id} ({current_module_size / 1024**3:.2f} GB) needs "
                f"{min_memory_offload / 1024**3:.2f} GB but only {resident_size / 1024**3:.2f} GB of managed "
                "weights are resident, offloading all other models. If it still does not fit, consider group "
                "offloading (`ModelMixin.enable_group_offload`)."
            )
            return hooks

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

        # a combination is guaranteed to exist: offloading everything frees `resident_size`, which is more than
        # `min_memory_offload` (the case where it isn't returned early above)
        best_offload_model_ids = search_best_candidate(module_sizes, min_memory_offload)

        return [hook for hook in hooks if hook.model_id in best_offload_model_ids]


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
        "quantization",
    ]

    def __init__(self):
        self.components = OrderedDict()
        self.added_time = OrderedDict()  # Store when components were added
        self.collections = OrderedDict()  # collection_name -> set of component_names
        self.model_hooks = None
        self._auto_offload_enabled = False
        self._offload_strategy = None
        self._offload_retry_on_oom = True
        self._offload_record = OffloadRecord()

    def _lookup_ids(
        self,
        name: str | None = None,
        collection: str | None = None,
        load_id: str | None = None,
        components: OrderedDict | None = None,
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
        if collection and collection not in self.collections:
            return set()
        elif collection and collection in self.collections:
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

    def add(self, name: str, component: Any, collection: str | None = None):
        """
        Add a component to the ComponentsManager.

        Args:
            name (str): The name of the component
            component (Any): The component to add
            collection (str | None): The collection to add the component to

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
        if is_new_component:
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

        if self._auto_offload_enabled and is_new_component and isinstance(component, torch.nn.Module):
            self._attach_offload_hook(component_id, component)

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

        if isinstance(component, torch.nn.Module):
            if self._auto_offload_enabled:
                self._detach_offload_hook(component)
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
        names: str | None = None,
        collection: str | None = None,
        load_id: str | None = None,
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

    def enable_auto_cpu_offload(
        self,
        device: str | int | torch.device = None,
        memory_reserve: str | int = "3GB",
        retry_on_oom: bool = True,
    ):
        """
        Enable automatic CPU offloading for all components.

        The algorithm works as follows:
        1. All models start on CPU by default
        2. When a model's forward pass is called, it's moved to the execution device
        3. If it doesn't fit into the memory currently available on the device minus `memory_reserve`, other models
           on the device are moved back to CPU first
        4. The system tries to offload the smallest combination of models that frees enough memory
        5. Models stay on the execution device until another model needs memory and forces them off
        6. If a forward pass still runs out of device memory, the smallest model on the device is offloaded and the
           forward is retried, escalating one model at a time until it fits (inference only: each retried forward
           re-runs from its original inputs)

        Args:
            device (str | int | torch.device): The execution device where models are moved for forward passes
            memory_reserve (str | int, *optional*, defaults to `"3GB"`):
                The amount of available device memory to keep free when deciding whether an incoming model fits,
                checked at each offloading decision — e.g. `"3GB"` or a number of bytes. The reserve is what covers
                allocations the offloading cannot see, mainly activations, which scale with resolution / batch size /
                sequence length. Set it to `0` to keep as much on the device as possible, relying on the OOM retry.
            retry_on_oom (bool, *optional*, defaults to `True`):
                Whether to recover from a forward pass that runs out of device memory by offloading the models on the
                device one at a time, smallest first, and retrying until it fits. Set it to `False` to raise the error
                instead — the forward passes are then left untouched.

        Every move the offloader makes is recorded in [`~ComponentsManager.offload_record`].
        """
        if not is_accelerate_available():
            raise ImportError("Make sure to install accelerate to use auto_cpu_offload")

        if device is None:
            device = get_device()
        if not isinstance(device, torch.device):
            device = torch.device(device)

        if device.index is None:
            device = torch.device(f"{device.type}:{0}")

        device_module = getattr(torch, device.type, torch.cuda)
        if not hasattr(device_module, "mem_get_info"):
            raise NotImplementedError(
                f"Offloading decisions rely on `mem_get_info()`, which is not implemented for {str(device.type)}."
            )

        for name, component in self.components.items():
            if isinstance(component, torch.nn.Module) and hasattr(component, "_hf_hook"):
                remove_hook_from_module(component, recurse=True)

        self.disable_auto_cpu_offload()
        offload_strategy = AutoOffloadStrategy(memory_reserve=memory_reserve)
        self._offload_record.clear()

        all_hooks = []
        for name, component in self.components.items():
            if isinstance(component, torch.nn.Module):
                hook = custom_offload_with_hook(
                    name,
                    component,
                    device,
                    offload_strategy=offload_strategy,
                    retry_on_oom=retry_on_oom,
                    record=self._offload_record,
                )
                all_hooks.append(hook)

        for hook in all_hooks:
            other_hooks = [h for h in all_hooks if h is not hook]
            for other_hook in other_hooks:
                if other_hook.hook.execution_device == hook.hook.execution_device:
                    hook.add_other_hook(other_hook)

        self.model_hooks = all_hooks
        self._auto_offload_enabled = True
        self._auto_offload_device = device
        self._offload_strategy = offload_strategy
        self._offload_retry_on_oom = retry_on_oom

    @property
    def offload_record(self) -> OffloadRecord:
        """
        What the offloader has done so far: every model moved onto or off the device, in order, and where a forward
        pass ran out of memory. Print it to see the sequence, or read [`~OffloadRecord.summary`] for the numbers
        behind it. Kept across [`~ComponentsManager.disable_auto_cpu_offload`] (it is the post-mortem of the run),
        and cleared when offloading is enabled again.
        """
        return self._offload_record

    def _attach_offload_hook(self, component_id: str, component: torch.nn.Module):
        """
        Attach an offload hook to a newly added component without disturbing the models already managed: the new
        component starts on CPU (like every model under auto offload), everything else stays where it is.
        """
        hook = custom_offload_with_hook(
            component_id,
            component,
            self._auto_offload_device,
            offload_strategy=self._offload_strategy,
            retry_on_oom=self._offload_retry_on_oom,
            record=self._offload_record,
        )
        for other_hook in self.model_hooks:
            if other_hook.hook.execution_device == hook.hook.execution_device:
                hook.add_other_hook(other_hook)
                other_hook.add_other_hook(hook)
        hook.offload(reason="component_added")
        self.model_hooks.append(hook)

    def _detach_offload_hook(self, component: torch.nn.Module):
        """Detach a removed component's offload hook, leaving all other managed models where they are."""
        hook = next(user_hook for user_hook in self.model_hooks if user_hook.model is component)
        hook.remove()
        self.model_hooks.remove(hook)
        for other_hook in self.model_hooks:
            if other_hook.hook.other_hooks and hook in other_hook.hook.other_hooks:
                other_hook.hook.other_hooks.remove(hook)

    def disable_auto_cpu_offload(self):
        """
        Disable automatic CPU offloading for all components.
        """
        if self.model_hooks is None:
            self._auto_offload_enabled = False
            return

        for hook in self.model_hooks:
            hook.offload(reason="offloading_disabled")
            hook.remove()
        if self.model_hooks:
            clear_device_cache()
        self.model_hooks = None
        self._auto_offload_enabled = False
        self._offload_strategy = None
        self._offload_retry_on_oom = True

    def get_model_info(
        self,
        component_id: str,
        fields: str | list[str] | None = None,
    ) -> dict[str, Any]:
        """Get comprehensive information about a component.

        Args:
            component_id (str): Name of the component to get info for
            fields (str | list[str] | None): Field(s) to return, all fields if `None`.

        Returns:
            Dictionary containing the requested component metadata.
        """
        if component_id not in self.components:
            raise ValueError(f"Component '{component_id}' not found in ComponentsManager")
        component = self.components[component_id]

        info = {
            "model_id": component_id,
            "added_time": self.added_time[component_id],
            "collection": ", ".join(coll for coll, comps in self.collections.items() if component_id in comps) or None,
        }

        if isinstance(component, torch.nn.Module):
            hook = getattr(component, "_hf_hook", None)
            info.update(
                {
                    "class_name": component.__class__.__name__,
                    "size_gb": component.get_memory_footprint() / 1024**3,
                    "adapters": list(component.peft_config.keys()) if hasattr(component, "peft_config") else None,
                    "has_hook": hook is not None,
                    "execution_device": getattr(hook, "execution_device", None),
                }
            )

            # IP-Adapter attention processor scales, summarized by shared layer prefix
            if hasattr(component, "_load_ip_adapter_weights") and hasattr(component, "attn_processors"):
                scales = {
                    name: processor.scale
                    for name, processor in component.attn_processors.items()
                    if "IPAdapter" in processor.__class__.__name__ and hasattr(processor, "scale")
                }
                if scales:
                    info["ip_adapter"] = summarize_dict_by_value_and_parts(scales)

            hf_quantizer = getattr(component, "hf_quantizer", None)
            if hf_quantizer is None:
                info["quantization"] = None
            else:
                quant_config = hf_quantizer.quantization_config
                info["quantization"] = (
                    quant_config.to_diff_dict() if hasattr(quant_config, "to_diff_dict") else quant_config.to_dict()
                )

        if fields is None:
            return info
        if isinstance(fields, str):
            fields = [fields]
        for field in fields:
            if field not in self._available_info_fields:
                raise ValueError(f"Field '{field}' not found in available_info_fields")
        return {k: v for k, v in info.items() if k in fields}

    # YiYi TODO: (1) add display fields, allow user to set which fields to display in the components table
    def __repr__(self):
        if not self.components:
            return "Components:\n" + "=" * 50 + "\nNo components registered.\n" + "=" * 50

        infos = {name: self.get_model_info(name) for name in self.components}
        # every collection a component belongs to; the first goes on the component's own row, the rest on
        # continuation rows below it
        component_collections = {
            name: [coll for coll, comps in self.collections.items() if name in comps] or ["N/A"]
            for name in self.components
        }

        def rows_with_collections(name: str, cells: list[str]) -> list[list[str]]:
            first, *rest = component_collections[name]
            return [[*cells, first]] + [[""] * len(cells) + [coll] for coll in rest]

        models = {name: c for name, c in self.components.items() if isinstance(c, torch.nn.Module)}
        others = {name: c for name, c in self.components.items() if not isinstance(c, torch.nn.Module)}

        sections = []
        if models:
            rows = []
            for name, component in models.items():
                info = infos[name]
                device = str(getattr(component, "device", "N/A"))
                if info["has_hook"]:
                    device = f"{device}({info['execution_device'] or 'N/A'})"
                rows += rows_with_collections(
                    name,
                    [
                        name,
                        info["class_name"],
                        device,
                        str(component.dtype) if hasattr(component, "dtype") else "N/A",
                        format_size(component.get_memory_footprint()),
                        str(getattr(component, "_diffusers_load_id", "N/A")),
                    ],
                )
            headers = ["Name_ID", "Class", "Device: act(exec)", "Dtype", "Size", "Load ID", "Collection"]
            sections.append(("Models:", format_table(headers, rows)))
        if others:
            rows = [
                row
                for name, component in others.items()
                for row in rows_with_collections(name, [name, component.__class__.__name__])
            ]
            sections.append(("Other Components:", format_table(["ID", "Class", "Collection"], rows)))

        output = "Components:\n" + "=" * max(len(table[0]) for _, table in sections) + "\n"
        for index, (title, table) in enumerate(sections):
            dash_line = "-" * len(table[0]) + "\n"
            if index:
                output += "\n"
            output += title + "\n" + dash_line + table[0] + "\n" + dash_line
            output += "\n".join(table[1:]) + "\n" + dash_line

        output += "\nAdditional Component Info:\n" + "=" * 50 + "\n"
        for name, info in infos.items():
            if info.get("adapters") is not None or info.get("ip_adapter") or info.get("quantization"):
                output += f"\n{name}:\n"
                if info.get("adapters") is not None:
                    output += f"  Adapters: {info['adapters']}\n"
                if info.get("ip_adapter"):
                    output += "  IP-Adapter: Enabled\n"
                if info.get("quantization"):
                    output += f"  Quantization: {info['quantization']}\n"

        return output

    def get_one(
        self,
        component_id: str | None = None,
        name: str | None = None,
        collection: str | None = None,
        load_id: str | None = None,
    ) -> Any:
        """
        Get a single component by either:
        - searching name (pattern matching), collection, or load_id.
        - passing in a component_id
        Raises an error if multiple components match or none are found.

        Args:
            component_id (str | None): Optional component ID to get
            name (str | None): Component name or pattern
            collection (str | None): Optional collection to filter by
            load_id (str | None): Optional load_id to filter by

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

    def get_ids(self, names: str | list[str] = None, collection: str | None = None):
        """
        Get component IDs by a list of names, optionally filtered by collection.

        Args:
            names (str | list[str]): list of component names
            collection (str | None): Optional collection to filter by

        Returns:
            list[str]: list of component IDs
        """
        ids = set()
        if not isinstance(names, list):
            names = [names]
        for name in names:
            ids.update(self._lookup_ids(name=name, collection=collection))
        return list(ids)

    def get_components_by_ids(self, ids: list[str], return_dict_with_names: bool | None = True):
        """
        Get components by a list of IDs.

        Args:
            ids (list[str]):
                list of component IDs
            return_dict_with_names (bool | None):
                Whether to return a dictionary with component names as keys:

        Returns:
            dict[str, Any]: Dictionary of components.
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

    def get_components_by_names(self, names: list[str], collection: str | None = None):
        """
        Get components by a list of names, optionally filtered by collection.

        Args:
            names (list[str]): list of component names
            collection (str | None): Optional collection to filter by

        Returns:
            dict[str, Any]: Dictionary of components with component names as keys

        Raises:
            ValueError: If duplicate component names are found in the search results
        """
        ids = self.get_ids(names, collection)
        return self.get_components_by_ids(ids)
