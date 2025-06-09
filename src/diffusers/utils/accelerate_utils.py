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
"""
Accelerate utilities: Utilities related to accelerate
"""

from typing import Any, Dict, List, Optional, Union

import torch
from packaging import version

from .import_utils import is_accelerate_available


if is_accelerate_available():
    import accelerate
    from accelerate import init_empty_weights
    from accelerate.utils import (
        compute_module_sizes,
        get_balanced_memory,
        get_max_memory,
        infer_auto_device_map,
    )


def apply_forward_hook(method):
    """
    Decorator that applies a registered CpuOffload hook to an arbitrary function rather than `forward`. This is useful
    for cases where a PyTorch module provides functions other than `forward` that should trigger a move to the
    appropriate acceleration device. This is the case for `encode` and `decode` in [`AutoencoderKL`].

    This decorator looks inside the internal `_hf_hook` property to find a registered offload hook.

    :param method: The method to decorate. This method should be a method of a PyTorch module.
    """
    if not is_accelerate_available():
        return method
    accelerate_version = version.parse(accelerate.__version__).base_version
    if version.parse(accelerate_version) < version.parse("0.17.0"):
        return method

    def wrapper(self, *args, **kwargs):
        if hasattr(self, "_hf_hook") and hasattr(self._hf_hook, "pre_forward"):
            self._hf_hook.pre_forward(self)
        return method(self, *args, **kwargs)

    return wrapper


def validate_device_map(device_map: Optional[Union[str, Dict[str, Union[int, str, torch.device]]]]) -> None:
    """
    Validate device map format, supporting all Accelerate formats.

    Args:
        device_map: Can be:
            - None (no device mapping)
            - str: "auto", "balanced", "balanced_low_0", "sequential"
            - dict: Maps module names to devices, e.g. {"": "cuda:0"}, {"unet": 0, "vae": 1},
                    {"text_encoder": "cpu", "unet": "cuda", "vae": "disk", "safety_checker": "meta"}

    Raises:
        ValueError: If device_map format is invalid
    """
    if device_map is None:
        return

    if isinstance(device_map, str):
        # Accelerate will validate string strategies internally
        # Common strategies: "auto", "balanced", "balanced_low_0", "sequential"
        pass
    elif isinstance(device_map, dict):
        # Validate dict format
        for key, value in device_map.items():
            if not isinstance(key, str):
                raise ValueError(
                    f"device_map keys must be strings (module names), got {type(key)} for key {key}"
                )
            # Accept all formats that Accelerate accepts
            # Including special devices like 'cpu', 'cuda', 'disk', 'meta'
            if not isinstance(value, (int, str, torch.device)):
                raise ValueError(
                    f"device_map values must be int (device index), str (device name like 'cpu', 'cuda', 'meta', 'disk'), or torch.device, "
                    f"got {type(value)} for key '{key}'"
                )
            # Validate integer device indices
            if isinstance(value, int):
                if value < 0:
                    raise ValueError(f"Device index must be non-negative, got {value} for key '{key}'")
                if torch.cuda.is_available() and value >= torch.cuda.device_count():
                    raise ValueError(
                        f"CUDA device index {value} is not available. "
                        f"Available CUDA devices: 0-{torch.cuda.device_count()-1}"
                    )
            # Validate special string devices
            if isinstance(value, str) and value not in ['cpu', 'cuda', 'meta', 'disk', 'mps']:
                # Check if it's a valid device string like 'cuda:0'
                try:
                    device = torch.device(value)
                    # Check if CUDA device actually exists
                    if device.type == "cuda" and device.index is not None:
                        if device.index >= torch.cuda.device_count():
                            raise ValueError(
                                f"CUDA device '{value}' is not available. "
                                f"Available CUDA devices: 0-{torch.cuda.device_count()-1}"
                            )
                    elif device.type == "cuda" and torch.cuda.device_count() == 0:
                        raise ValueError("CUDA requested but no CUDA devices available")
                    elif device.type == "mps" and not torch.backends.mps.is_available():
                        raise ValueError("MPS device requested but MPS is not available on this system")
                except (RuntimeError, ValueError) as e:
                    if isinstance(e, ValueError) and "not available" in str(e):
                        raise
                    raise ValueError(
                        f"Invalid device string '{value}' for key '{key}'. "
                        f"Valid options: 'cpu', 'cuda', 'meta', 'disk', 'mps', or device strings like 'cuda:0'"
                    )
    else:
        raise ValueError(
            f"`device_map` must be None, a string strategy ('auto', 'balanced', etc.), "
            f"or a dict mapping module names to devices, got {type(device_map)}"
        )


class PipelineDeviceMapper:
    """
    Handles device mapping for diffusion pipelines with full Accelerate compatibility.

    This class bridges the gap between pipeline-level device maps and component-level device maps,
    ensuring that components can load directly to their target devices without CPU intermediates.
    """

    def __init__(
        self,
        pipeline_class,
        init_dict: Dict[str, tuple],
        passed_class_obj: Optional[Dict[str, Any]] = None,
        cached_folder: str = "",
        **loading_kwargs
    ):
        self.pipeline_class = pipeline_class
        self.init_dict = init_dict
        self.passed_class_obj = passed_class_obj or {}
        self.cached_folder = cached_folder
        self.loading_kwargs = loading_kwargs

    def resolve_component_device_maps(
        self,
        device_map: Union[str, Dict[str, Union[int, str, torch.device]]],
        max_memory: Optional[Dict[Union[int, str], Union[int, str]]] = None,
        torch_dtype: Optional[torch.dtype] = None,
    ) -> Dict[str, Optional[Union[str, Dict[str, Union[int, str, torch.device]]]]]:
        """
        Resolve pipeline-level device_map to component-specific device_maps.

        This maintains 100% compatibility with Accelerate's device mapping behavior.

        Args:
            device_map: Pipeline-level device mapping
            max_memory: Memory constraints per device
            torch_dtype: Data type for components

        Returns:
            Dictionary mapping component names to their device_maps
        """
        if device_map is None:
            return {}

        # Validate the device_map format
        validate_device_map(device_map)

        # Handle dict device maps
        if isinstance(device_map, dict):
            return self._resolve_dict_device_map(device_map)

        # Handle string strategies that need size calculation
        if isinstance(device_map, str) and device_map in ["auto", "balanced", "balanced_low_0", "sequential"]:
            return self._resolve_auto_device_map(device_map, max_memory, torch_dtype)

        # Handle simple device strings (e.g., "cuda:0")
        if isinstance(device_map, str):
            return self._resolve_simple_device_map(device_map)

        raise ValueError(f"Unexpected device_map type: {type(device_map)}")

    def _resolve_dict_device_map(self, device_map: Dict[str, Union[int, str, torch.device]]) -> Dict:
        """
        Resolve explicit dict device mappings to component-specific maps.

        Handles:
        - {"": "cuda:0"} -> all components on cuda:0
        - {"unet": 0, "vae": 1} -> different components on different devices
        - {"unet.down_blocks": 0, "unet.up_blocks": 1} -> parts of unet on different devices
        """
        component_device_maps = {}

        for component_name in self.init_dict.keys():
            # Skip already instantiated components
            if component_name in self.passed_class_obj:
                continue

            # Collect all device assignments for this component
            component_map = self._extract_component_device_map(component_name, device_map)

            if component_map:
                component_device_maps[component_name] = component_map

        return component_device_maps

    def _extract_component_device_map(self, component_name: str, device_map: Dict) -> Optional[Dict]:
        """
        Extract device assignments for a specific component from the full device_map.

        This handles Accelerate's hierarchical device mapping:
        - Direct assignment: {"unet": 0}
        - Root assignment: {"": 0}
        - Submodule assignment: {"unet.down_blocks": 0}
        """
        component_map = {}

        # Check for root assignment that applies to everything
        if "" in device_map:
            component_map[""] = device_map[""]

        # Check for direct component assignment
        if component_name in device_map:
            component_map[""] = device_map[component_name]

        # Check for submodule assignments
        for key, device in device_map.items():
            if key.startswith(f"{component_name}."):
                # Extract the submodule path relative to the component
                submodule_path = key[len(component_name)+1:]
                component_map[submodule_path] = device

        return component_map if component_map else None

    def _resolve_auto_device_map(
        self,
        strategy: str,
        max_memory: Optional[Dict] = None,
        torch_dtype: Optional[torch.dtype] = None
    ) -> Dict:
        """
        Resolve auto strategies using Accelerate's algorithms.

        This creates a virtual unified model to leverage Accelerate's device mapping logic.
        """
        if not is_accelerate_available():
            raise ImportError("Accelerate is required for auto device mapping strategies")

        # Create a virtual pipeline model for size calculation
        virtual_model = self._create_virtual_pipeline(torch_dtype)

        # Calculate module sizes - this is critical for device mapping!
        module_sizes = compute_module_sizes(virtual_model, dtype=torch_dtype)

        # Get available memory if not specified
        if max_memory is None:
            max_memory = get_max_memory()

        # Determine which modules should not be split
        no_split_modules = self._get_no_split_modules(virtual_model)

        # Use Accelerate's algorithm to compute optimal placement
        if strategy == "balanced_low_0":
            low_zero = True
            strategy_for_accelerate = "balanced"
        else:
            low_zero = False
            strategy_for_accelerate = strategy

        if strategy_for_accelerate != "sequential":
            max_memory = get_balanced_memory(
                virtual_model,
                max_memory=max_memory,
                dtype=torch_dtype,
                low_zero=low_zero,
                no_split_module_classes=no_split_modules,
            )

        # Compute the unified device map
        unified_device_map = infer_auto_device_map(
            virtual_model,
            max_memory=max_memory,
            dtype=torch_dtype,
            no_split_module_classes=no_split_modules,
            verbose=self.loading_kwargs.get("verbose", False),
        )

        # Check if we have valid device placement
        if not unified_device_map:
            total_size = module_sizes.get("", 0)
            raise ValueError(
                f"Failed to compute device map. Model size ({total_size} bytes) may exceed "
                f"available memory. Consider using a smaller model or adjusting max_memory."
            )

        # Parse into component-specific device maps
        return self._parse_unified_device_map(unified_device_map)

    def _create_virtual_pipeline(self, torch_dtype: Optional[torch.dtype]) -> torch.nn.Module:
        """
        Create a virtual model representing the entire pipeline for size calculation.

        Uses meta device to avoid actual memory allocation.
        """
        from ..pipelines.pipeline_loading_utils import _load_empty_model

        class VirtualPipeline(torch.nn.Module):
            """Virtual container for pipeline components."""
            pass

        with init_empty_weights():
            virtual_model = VirtualPipeline()

            # Add each component as a submodule
            for name, (library_name, class_name) in self.init_dict.items():
                if name in self.passed_class_obj:
                    # Use the actual component if provided
                    setattr(virtual_model, name, self.passed_class_obj[name])
                else:
                    # Create empty component for size calculation
                    component_dtype = torch_dtype or torch.float32

                    empty_component = _load_empty_model(
                        library_name=library_name,
                        class_name=class_name,
                        importable_classes=self.loading_kwargs.get("importable_classes", {}),
                        pipelines=self.loading_kwargs.get("pipelines"),
                        is_pipeline_module=self.loading_kwargs.get("is_pipeline_module", False),
                        name=name,
                        torch_dtype=component_dtype,
                        cached_folder=self.cached_folder,
                        **self.loading_kwargs
                    )

                    if empty_component is not None:
                        setattr(virtual_model, name, empty_component)

        return virtual_model

    def _get_no_split_modules(self, virtual_model: torch.nn.Module) -> List[str]:
        """
        Get list of module classes that should not be split across devices.

        By default, pipeline components are treated as atomic units.
        """
        no_split = []

        # Add all top-level component classes
        for name, module in virtual_model.named_children():
            if module is not None:
                no_split.append(type(module).__name__)

        # Add any pipeline-specific no-split modules
        if hasattr(self.pipeline_class, "_no_split_modules"):
            no_split.extend(self.pipeline_class._no_split_modules)

        # Add any model-specific no-split modules
        for name, module in virtual_model.named_children():
            if hasattr(module, "_no_split_modules"):
                no_split.extend(module._no_split_modules)

        return list(set(no_split))

    def _parse_unified_device_map(self, unified_map: Dict[str, Union[int, str, torch.device]]) -> Dict:
        """
        Parse Accelerate's unified device map into component-specific maps.
        """
        component_device_maps = {}

        # Group assignments by component
        for path, device in unified_map.items():
            parts = path.split(".")
            if not parts:
                continue

            component_name = parts[0]

            # Only process pipeline components
            if component_name not in self.init_dict:
                continue

            if component_name not in component_device_maps:
                component_device_maps[component_name] = {}

            if len(parts) == 1:
                # Top-level component assignment
                component_device_maps[component_name][""] = device
            else:
                # Submodule assignment
                submodule_path = ".".join(parts[1:])
                component_device_maps[component_name][submodule_path] = device

        return component_device_maps

    def _resolve_simple_device_map(self, device: str) -> Dict:
        """
        Handle simple device strings like "cuda:0".

        All components go to the specified device.
        """
        component_device_maps = {}

        for component_name in self.init_dict.keys():
            if component_name not in self.passed_class_obj:
                component_device_maps[component_name] = {"": device}

        return component_device_maps
