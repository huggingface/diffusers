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
import inspect


import traceback
import warnings
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Union, Optional
from copy import deepcopy


import torch
from tqdm.auto import tqdm
import re
import os
import importlib

from huggingface_hub.utils import validate_hf_hub_args

from ..configuration_utils import ConfigMixin, FrozenDict
from ..utils import (
    is_accelerate_available,
    logging,
    PushToHubMixin,
)
from ..pipelines.pipeline_loading_utils import simple_get_class_obj, _fetch_class_library_tuple
from .modular_pipeline_utils import (
    ComponentSpec,
    ConfigSpec,
    InputParam,
    OutputParam,
    format_components,
    format_configs,
    format_inputs_short,
    format_intermediates_short,
    make_doc_string,
)
from .components_manager import ComponentsManager
from ..utils.dynamic_modules_utils import get_class_from_dynamic_module, resolve_trust_remote_code

from copy import deepcopy
if is_accelerate_available():
    import accelerate

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


MODULAR_LOADER_MAPPING = OrderedDict(
    [
        ("stable-diffusion-xl", "StableDiffusionXLModularLoader"),
    ]
)


@dataclass
class PipelineState:
    """
    [`PipelineState`] stores the state of a pipeline. It is used to pass data between pipeline blocks.
    """

    inputs: Dict[str, Any] = field(default_factory=dict)
    intermediates: Dict[str, Any] = field(default_factory=dict)
    input_kwargs: Dict[str, list[str, Any]] = field(default_factory=dict)
    intermediate_kwargs: Dict[str, list[str, Any]] = field(default_factory=dict)

    def add_input(self, key: str, value: Any, kwargs_type: str = None):
        """
        Add an input to the pipeline state with optional metadata.
        
        Args:
            key (str): The key for the input
            value (Any): The input value
            kwargs_type (str): The kwargs_type to store with the input
        """
        self.inputs[key] = value
        if kwargs_type is not None:
            if kwargs_type not in self.input_kwargs:
                self.input_kwargs[kwargs_type] = [key]
            else:
                self.input_kwargs[kwargs_type].append(key)

    def add_intermediate(self, key: str, value: Any, kwargs_type: str = None):
        """
        Add an intermediate value to the pipeline state with optional metadata.
        
        Args:
            key (str): The key for the intermediate value
            value (Any): The intermediate value
            kwargs_type (str): The kwargs_type to store with the intermediate value
        """
        self.intermediates[key] = value
        if kwargs_type is not None:
            if kwargs_type not in self.intermediate_kwargs:
                self.intermediate_kwargs[kwargs_type] = [key]
            else:
                self.intermediate_kwargs[kwargs_type].append(key)

    def get_input(self, key: str, default: Any = None) -> Any:
        value = self.inputs.get(key, default)
        if value is not None:
            return deepcopy(value)

    def get_inputs(self, keys: List[str], default: Any = None) -> Dict[str, Any]:
        return {key: self.inputs.get(key, default) for key in keys}
    
    def get_inputs_kwargs(self, kwargs_type: str) -> Dict[str, Any]:
        """
        Get all inputs with matching kwargs_type.
        
        Args:
            kwargs_type (str): The kwargs_type to filter by
            
        Returns:
            Dict[str, Any]: Dictionary of inputs with matching kwargs_type
        """
        input_names = self.input_kwargs.get(kwargs_type, [])
        return self.get_inputs(input_names)

    def get_intermediates_kwargs(self, kwargs_type: str) -> Dict[str, Any]:
        """
        Get all intermediates with matching kwargs_type.
        
        Args:
            kwargs_type (str): The kwargs_type to filter by
            
        Returns:
            Dict[str, Any]: Dictionary of intermediates with matching kwargs_type
        """
        intermediate_names = self.intermediate_kwargs.get(kwargs_type, [])
        return self.get_intermediates(intermediate_names)

    def get_intermediate(self, key: str, default: Any = None) -> Any:
        return self.intermediates.get(key, default)

    def get_intermediates(self, keys: List[str], default: Any = None) -> Dict[str, Any]:
        return {key: self.intermediates.get(key, default) for key in keys}

    def to_dict(self) -> Dict[str, Any]:
        return {**self.__dict__, "inputs": self.inputs, "intermediates": self.intermediates}

    def __repr__(self):
        def format_value(v):
            if hasattr(v, "shape") and hasattr(v, "dtype"):
                return f"Tensor(dtype={v.dtype}, shape={v.shape})"
            elif isinstance(v, list) and len(v) > 0 and hasattr(v[0], "shape") and hasattr(v[0], "dtype"):
                return f"[Tensor(dtype={v[0].dtype}, shape={v[0].shape}), ...]"
            else:
                return repr(v)

        inputs = "\n".join(f"    {k}: {format_value(v)}" for k, v in self.inputs.items())
        intermediates = "\n".join(f"    {k}: {format_value(v)}" for k, v in self.intermediates.items())
        
        # Format input_kwargs and intermediate_kwargs
        input_kwargs_str = "\n".join(f"    {k}: {v}" for k, v in self.input_kwargs.items())
        intermediate_kwargs_str = "\n".join(f"    {k}: {v}" for k, v in self.intermediate_kwargs.items())

        return (
            f"PipelineState(\n"
            f"  inputs={{\n{inputs}\n  }},\n"
            f"  intermediates={{\n{intermediates}\n  }},\n"
            f"  input_kwargs={{\n{input_kwargs_str}\n  }},\n"
            f"  intermediate_kwargs={{\n{intermediate_kwargs_str}\n  }}\n"
            f")"
        )


@dataclass 
class BlockState:
    """
    Container for block state data with attribute access and formatted representation.
    """
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __getitem__(self, key: str):
        # allows block_state["foo"]
        return getattr(self, key, None)
    
    def __setitem__(self, key: str, value: Any):
        # allows block_state["foo"] = "bar"
        setattr(self, key, value)
    
    def as_dict(self):
        """
        Convert BlockState to a dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary containing all attributes of the BlockState
        """
        return {key: value for key, value in self.__dict__.items()}

    def __repr__(self):
        def format_value(v):
            # Handle tensors directly
            if hasattr(v, "shape") and hasattr(v, "dtype"):
                return f"Tensor(dtype={v.dtype}, shape={v.shape})"
            
            # Handle lists of tensors
            elif isinstance(v, list):
                if len(v) > 0 and hasattr(v[0], "shape") and hasattr(v[0], "dtype"):
                    shapes = [t.shape for t in v]
                    return f"List[{len(v)}] of Tensors with shapes {shapes}"
                return repr(v)
            
            # Handle tuples of tensors
            elif isinstance(v, tuple):
                if len(v) > 0 and hasattr(v[0], "shape") and hasattr(v[0], "dtype"):
                    shapes = [t.shape for t in v]
                    return f"Tuple[{len(v)}] of Tensors with shapes {shapes}"
                return repr(v)
            
            # Handle dicts with tensor values
            elif isinstance(v, dict):
                formatted_dict = {}
                for k, val in v.items():
                    if hasattr(val, "shape") and hasattr(val, "dtype"):
                        formatted_dict[k] = f"Tensor(shape={val.shape}, dtype={val.dtype})"
                    elif isinstance(val, list) and len(val) > 0 and hasattr(val[0], "shape") and hasattr(val[0], "dtype"):
                        shapes = [t.shape for t in val]
                        formatted_dict[k] = f"List[{len(val)}] of Tensors with shapes {shapes}"
                    else:
                        formatted_dict[k] = repr(val)
                return formatted_dict
            
            # Default case
            return repr(v)

        attributes = "\n".join(f"    {k}: {format_value(v)}" for k, v in self.__dict__.items())
        return f"BlockState(\n{attributes}\n)"


class ModularPipelineBlocks(ConfigMixin):
    """
    Mixin for all PipelineBlocks: PipelineBlock, AutoPipelineBlocks, SequentialPipelineBlocks
    """

    config_name = "config.json"

    @classmethod
    def _get_signature_keys(cls, obj):
        parameters = inspect.signature(obj.__init__).parameters
        required_parameters = {k: v for k, v in parameters.items() if v.default == inspect._empty}
        optional_parameters = set({k for k, v in parameters.items() if v.default != inspect._empty})
        expected_modules = set(required_parameters.keys()) - {"self"}

        return expected_modules, optional_parameters
    

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        trust_remote_code: Optional[bool] = None,
        **kwargs,
    ):
        hub_kwargs_names = [
            "cache_dir",
            "force_download",
            "local_files_only",
            "proxies",
            "resume_download",
            "revision",
            "subfolder",
            "token",
        ]
        hub_kwargs = {name: kwargs.pop(name) for name in hub_kwargs_names if name in kwargs}

        config = cls.load_config(pretrained_model_name_or_path)
        has_remote_code = "auto_map" in config and cls.__name__ in config["auto_map"]
        trust_remote_code = resolve_trust_remote_code(
            trust_remote_code, pretrained_model_name_or_path, has_remote_code
        )
        if not (has_remote_code and trust_remote_code):
            raise ValueError("TODO")

        class_ref = config["auto_map"][cls.__name__]
        module_file, class_name = class_ref.split(".")
        module_file = module_file + ".py"
        block_cls = get_class_from_dynamic_module(
            pretrained_model_name_or_path,
            module_file=module_file,
            class_name=class_name,
            is_modular=True,
            **hub_kwargs,
            **kwargs,
        )
        expected_kwargs, optional_kwargs = block_cls._get_signature_keys(block_cls)
        block_kwargs = {
            name: kwargs.pop(name) for name in kwargs if name in expected_kwargs or name in optional_kwargs
        }

        return block_cls(**block_kwargs)
    
    def init_pipeline(self, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]] = None, component_manager: Optional[ComponentsManager] = None, collection: Optional[str] = None):
        """
        create a ModularLoader, optionally accept modular_repo to load from hub.
        """
        loader_class_name = MODULAR_LOADER_MAPPING.get(self.model_name, ModularLoader.__name__)
        diffusers_module = importlib.import_module("diffusers")
        loader_class = getattr(diffusers_module, loader_class_name)
        
        # Create deep copies to avoid modifying the original specs
        component_specs = deepcopy(self.expected_components)
        config_specs = deepcopy(self.expected_configs)
        # Create the loader with the updated specs
        specs = component_specs + config_specs
        
        loader = loader_class(specs=specs, pretrained_model_name_or_path=pretrained_model_name_or_path, component_manager=component_manager, collection=collection)
        modular_pipeline = ModularPipeline(blocks=self, loader=loader)
        return modular_pipeline


class PipelineBlock(ModularPipelineBlocks):
    
    model_name = None
    
    @property
    def description(self) -> str:
        """Description of the block. Must be implemented by subclasses."""
        # raise NotImplementedError("description method must be implemented in subclasses")
        return "TODO: add a description"

    @property
    def expected_components(self) -> List[ComponentSpec]:
        return []
    
    @property
    def expected_configs(self) -> List[ConfigSpec]:
        return []

    
    @property
    def inputs(self) -> List[InputParam]:
        """List of input parameters. Must be implemented by subclasses."""
        return []

    @property
    def intermediates_inputs(self) -> List[InputParam]:
        """List of intermediate input parameters. Must be implemented by subclasses."""
        return []

    @property
    def intermediates_outputs(self) -> List[OutputParam]:
        """List of intermediate output parameters. Must be implemented by subclasses."""
        return []

    def _get_outputs(self):
        return self.intermediates_outputs

    # YiYi TODO: is it too easy for user to unintentionally override these properties?
    # Adding outputs attributes here for consistency between PipelineBlock/AutoPipelineBlocks/SequentialPipelineBlocks
    @property
    def outputs(self) -> List[OutputParam]:
        return self._get_outputs()

    def _get_required_inputs(self):
        input_names = []
        for input_param in self.inputs:
            if input_param.required:
                input_names.append(input_param.name)
        return input_names

    @property
    def required_inputs(self) -> List[str]:
        return self._get_required_inputs()
    

    def _get_required_intermediates_inputs(self):
        input_names = []
        for input_param in self.intermediates_inputs:
            if input_param.required:
                input_names.append(input_param.name)
        return input_names

    # YiYi TODO: maybe we do not need this, it is only used in docstring, 
    # intermediate_inputs is by default required, unless you manually handle it inside the block
    @property
    def required_intermediates_inputs(self) -> List[str]:
        return self._get_required_intermediates_inputs()


    def __call__(self, pipeline, state: PipelineState) -> PipelineState:
        raise NotImplementedError("__call__ method must be implemented in subclasses")

    def __repr__(self):
        class_name = self.__class__.__name__
        base_class = self.__class__.__bases__[0].__name__

        # Format description with proper indentation
        desc_lines = self.description.split('\n')
        desc = []
        # First line with "Description:" label
        desc.append(f"  Description: {desc_lines[0]}")
        # Subsequent lines with proper indentation
        if len(desc_lines) > 1:
            desc.extend(f"      {line}" for line in desc_lines[1:])
        desc = '\n'.join(desc) + '\n'

        # Components section - use format_components with add_empty_lines=False
        expected_components = getattr(self, "expected_components", [])
        components_str = format_components(expected_components, indent_level=2, add_empty_lines=False)
        components = "  " + components_str.replace("\n", "\n  ")

        # Configs section - use format_configs with add_empty_lines=False
        expected_configs = getattr(self, "expected_configs", [])
        configs_str = format_configs(expected_configs, indent_level=2, add_empty_lines=False)
        configs = "  " + configs_str.replace("\n", "\n  ")

        # Inputs section
        inputs_str = format_inputs_short(self.inputs)
        inputs = "Inputs:\n    " + inputs_str

        # Intermediates section
        intermediates_str = format_intermediates_short(self.intermediates_inputs, self.required_intermediates_inputs, self.intermediates_outputs)
        intermediates = f"Intermediates:\n{intermediates_str}"

        return (
            f"{class_name}(\n"
            f"  Class: {base_class}\n"
            f"{desc}"
            f"{components}\n"
            f"{configs}\n"
            f"  {inputs}\n"
            f"  {intermediates}\n"
            f")"
        )


    @property
    def doc(self):
        return make_doc_string(
            self.inputs, 
            self.intermediates_inputs, 
            self.outputs, 
            self.description,
            class_name=self.__class__.__name__,
            expected_components=self.expected_components,
            expected_configs=self.expected_configs
        )


    # YiYi TODO: input and inteermediate inputs with same name? should warn?
    def get_block_state(self, state: PipelineState) -> dict:
        """Get all inputs and intermediates in one dictionary"""
        data = {}
        
        # Check inputs
        for input_param in self.inputs:
            if input_param.name:
                value = state.get_input(input_param.name)
                if input_param.required and value is None:
                    raise ValueError(f"Required input '{input_param.name}' is missing")
                elif value is not None or (value is None and input_param.name not in data):
                    data[input_param.name] = value
            elif input_param.kwargs_type:
                # if kwargs_type is provided, get all inputs with matching kwargs_type
                if input_param.kwargs_type not in data:
                    data[input_param.kwargs_type] = {}
                inputs_kwargs = state.get_inputs_kwargs(input_param.kwargs_type)
                if inputs_kwargs:
                    for k, v in inputs_kwargs.items():
                        if v is not None:
                            data[k] = v
                            data[input_param.kwargs_type][k] = v

        # Check intermediates
        for input_param in self.intermediates_inputs:
            if input_param.name:
                value = state.get_intermediate(input_param.name)
                if input_param.required and value is None:
                    raise ValueError(f"Required intermediate input '{input_param.name}' is missing")
                elif value is not None or (value is None and input_param.name not in data):
                    data[input_param.name] = value
            elif input_param.kwargs_type:
                # if kwargs_type is provided, get all intermediates with matching kwargs_type
                if input_param.kwargs_type not in data:
                    data[input_param.kwargs_type] = {}
                intermediates_kwargs = state.get_intermediates_kwargs(input_param.kwargs_type)
                if intermediates_kwargs:
                    for k, v in intermediates_kwargs.items():
                        if v is not None:
                            if k not in data:
                                data[k] = v
                            data[input_param.kwargs_type][k] = v
        return BlockState(**data)
    
    def add_block_state(self, state: PipelineState, block_state: BlockState):
        for output_param in self.intermediates_outputs:
            if not hasattr(block_state, output_param.name):
                raise ValueError(f"Intermediate output '{output_param.name}' is missing in block state")
            param = getattr(block_state, output_param.name)
            state.add_intermediate(output_param.name, param, output_param.kwargs_type)
        
        for input_param in self.intermediates_inputs:
            if hasattr(block_state, input_param.name):
                param = getattr(block_state, input_param.name)
                # Only add if the value is different from what's in the state
                current_value = state.get_intermediate(input_param.name)
                if current_value is not param:  # Using identity comparison to check if object was modified
                    state.add_intermediate(input_param.name, param, input_param.kwargs_type)

        for input_param in self.intermediates_inputs:
            if input_param.name and hasattr(block_state, input_param.name):
                param = getattr(block_state, input_param.name)
                # Only add if the value is different from what's in the state
                current_value = state.get_intermediate(input_param.name)
                if current_value is not param:  # Using identity comparison to check if object was modified
                    state.add_intermediate(input_param.name, param, input_param.kwargs_type)
            elif input_param.kwargs_type:
                # if it is a kwargs type, e.g. "guider_input_fields", it is likely to be a list of parameters
                # we need to first find out which inputs are and loop through them.
                intermediates_kwargs = state.get_intermediates_kwargs(input_param.kwargs_type)
                for param_name, current_value in intermediates_kwargs.items():
                    param = getattr(block_state, param_name)
                    if current_value is not param:  # Using identity comparison to check if object was modified
                        state.add_intermediate(param_name, param, input_param.kwargs_type)

    def save_pretrained(self, save_directory, push_to_hub = False, **kwargs):
        # TODO: factor out this logic.
        cls_name = self.__class__.__name__
        
        full_mod = type(self).__module__ 
        module = full_mod.rsplit(".", 1)[-1].replace("__dynamic__", "")
        parent_module = self.save_pretrained.__func__.__qualname__.split(".", 1)[0]  
        auto_map = {f"{parent_module}": f"{module}.{cls_name}"}
        _component_names = [c.name for c in self.expected_components]
        
        self.register_to_config(auto_map=auto_map, _component_names=_component_names)
        self.save_config(save_directory=save_directory, push_to_hub=push_to_hub, **kwargs)
        config = dict(self.config)
        self._internal_dict = FrozenDict(config)


def combine_inputs(*named_input_lists: List[Tuple[str, List[InputParam]]]) -> List[InputParam]:
    """
    Combines multiple lists of InputParam objects from different blocks. For duplicate inputs, updates only if 
    current default value is None and new default value is not None. Warns if multiple non-None default values 
    exist for the same input.

    Args:
        named_input_lists: List of tuples containing (block_name, input_param_list) pairs
    
    Returns:
        List[InputParam]: Combined list of unique InputParam objects
    """
    combined_dict = {}  # name -> InputParam
    value_sources = {}  # name -> block_name
    
    for block_name, inputs in named_input_lists:
        for input_param in inputs:
            if input_param.name is None and input_param.kwargs_type is not None:
                input_name = "*_" + input_param.kwargs_type
            else:
                input_name = input_param.name
            if input_name in combined_dict:
                current_param = combined_dict[input_name]
                if (current_param.default is not None and 
                    input_param.default is not None and 
                    current_param.default != input_param.default):
                    warnings.warn(
                        f"Multiple different default values found for input '{input_name}': "
                        f"{current_param.default} (from block '{value_sources[input_name]}') and "
                        f"{input_param.default} (from block '{block_name}'). Using {current_param.default}."
                    )
                if current_param.default is None and input_param.default is not None:
                    combined_dict[input_name] = input_param
                    value_sources[input_name] = block_name
            else:
                combined_dict[input_name] = input_param
                value_sources[input_name] = block_name
    
    return list(combined_dict.values())

def combine_outputs(*named_output_lists: List[Tuple[str, List[OutputParam]]]) -> List[OutputParam]:
    """
    Combines multiple lists of OutputParam objects from different blocks. For duplicate outputs,
    keeps the first occurrence of each output name.

    Args:
        named_output_lists: List of tuples containing (block_name, output_param_list) pairs
    
    Returns:
        List[OutputParam]: Combined list of unique OutputParam objects
    """
    combined_dict = {}  # name -> OutputParam
    
    for block_name, outputs in named_output_lists:
        for output_param in outputs:
            if (output_param.name not in combined_dict) or (combined_dict[output_param.name].kwargs_type is None and output_param.kwargs_type is not None):
                combined_dict[output_param.name] = output_param
    
    return list(combined_dict.values())


class AutoPipelineBlocks(ModularPipelineBlocks):
    """
    A class that automatically selects a block to run based on the inputs.

    Attributes:
        block_classes: List of block classes to be used
        block_names: List of prefixes for each block
        block_trigger_inputs: List of input names that trigger specific blocks, with None for default
    """

    block_classes = []
    block_names = []
    block_trigger_inputs = []

    def __init__(self):
        blocks = OrderedDict()
        for block_name, block_cls in zip(self.block_names, self.block_classes):
            blocks[block_name] = block_cls()
        self.blocks = blocks
        if not (len(self.block_classes) == len(self.block_names) == len(self.block_trigger_inputs)):
            raise ValueError(f"In {self.__class__.__name__}, the number of block_classes, block_names, and block_trigger_inputs must be the same.")
        default_blocks = [t for t in self.block_trigger_inputs if t is None]
        # can only have 1 or 0 default block, and has to put in the last 
        # the order of blocksmatters here because the first block with matching trigger will be dispatched
        # e.g. blocks = [inpaint, img2img] and block_trigger_inputs = ["mask", "image"]
        # if both mask and image are provided, it is inpaint; if only image is provided, it is img2img
        if len(default_blocks) > 1 or (
                len(default_blocks) == 1 and self.block_trigger_inputs[-1] is not None
            ):
            raise ValueError(
                f"In {self.__class__.__name__}, exactly one None must be specified as the last element "
                "in block_trigger_inputs."
            )

        # Map trigger inputs to block objects
        self.trigger_to_block_map = dict(zip(self.block_trigger_inputs, self.blocks.values()))
        self.trigger_to_block_name_map = dict(zip(self.block_trigger_inputs, self.blocks.keys()))
        self.block_to_trigger_map = dict(zip(self.blocks.keys(), self.block_trigger_inputs))

    @property
    def model_name(self):
        return next(iter(self.blocks.values())).model_name
    
    @property
    def description(self):
        return ""

    @property
    def expected_components(self):
        expected_components = []
        for block in self.blocks.values():
            for component in block.expected_components:
                if component not in expected_components:
                    expected_components.append(component)
        return expected_components

    @property
    def expected_configs(self):
        expected_configs = []
        for block in self.blocks.values():
            for config in block.expected_configs:
                if config not in expected_configs:
                    expected_configs.append(config)
        return expected_configs


    @property
    def required_inputs(self) -> List[str]:
        if None not in self.block_trigger_inputs:
            return []
        first_block = next(iter(self.blocks.values()))
        required_by_all = set(getattr(first_block, "required_inputs", set()))

        # Intersect with required inputs from all other blocks
        for block in list(self.blocks.values())[1:]:
            block_required = set(getattr(block, "required_inputs", set()))
            required_by_all.intersection_update(block_required)

        return list(required_by_all)
    
    # YiYi TODO: maybe we do not need this, it is only used in docstring, 
    # intermediate_inputs is by default required, unless you manually handle it inside the block
    @property
    def required_intermediates_inputs(self) -> List[str]:
        if None not in self.block_trigger_inputs:
            return []
        first_block = next(iter(self.blocks.values()))
        required_by_all = set(getattr(first_block, "required_intermediates_inputs", set()))

        # Intersect with required inputs from all other blocks
        for block in list(self.blocks.values())[1:]:
            block_required = set(getattr(block, "required_intermediates_inputs", set()))
            required_by_all.intersection_update(block_required)

        return list(required_by_all)


    # YiYi TODO: add test for this
    @property
    def inputs(self) -> List[Tuple[str, Any]]:
        named_inputs = [(name, block.inputs) for name, block in self.blocks.items()]
        combined_inputs = combine_inputs(*named_inputs)
        # mark Required inputs only if that input is required by all the blocks
        for input_param in combined_inputs:
            if input_param.name in self.required_inputs:
                input_param.required = True
            else:
                input_param.required = False
        return combined_inputs


    @property
    def intermediates_inputs(self) -> List[str]:
        named_inputs = [(name, block.intermediates_inputs) for name, block in self.blocks.items()]
        combined_inputs = combine_inputs(*named_inputs)
        # mark Required inputs only if that input is required by all the blocks
        for input_param in combined_inputs:
            if input_param.name in self.required_intermediates_inputs:
                input_param.required = True
            else:
                input_param.required = False
        return combined_inputs

    @property
    def intermediates_outputs(self) -> List[str]:
        named_outputs = [(name, block.intermediates_outputs) for name, block in self.blocks.items()]
        combined_outputs = combine_outputs(*named_outputs)
        return combined_outputs
    
    @property
    def outputs(self) -> List[str]:
        named_outputs = [(name, block.outputs) for name, block in self.blocks.items()]
        combined_outputs = combine_outputs(*named_outputs)
        return combined_outputs

    @torch.no_grad()
    def __call__(self, pipeline, state: PipelineState) -> PipelineState:
        # Find default block first (if any)

        block = self.trigger_to_block_map.get(None)
        for input_name in self.block_trigger_inputs:
            if input_name is not None and state.get_input(input_name) is not None:
                block = self.trigger_to_block_map[input_name]
                break
            elif input_name is not None and state.get_intermediate(input_name) is not None:
                block = self.trigger_to_block_map[input_name]
                break

        if block is None:
            logger.warning(f"skipping auto block: {self.__class__.__name__}")
            return pipeline, state

        try:
            logger.info(f"Running block: {block.__class__.__name__}, trigger: {input_name}")
            return block(pipeline, state)
        except Exception as e:
            error_msg = (
                f"\nError in block: {block.__class__.__name__}\n"
                f"Error details: {str(e)}\n"
                f"Traceback:\n{traceback.format_exc()}"
            )
            logger.error(error_msg)
            raise

    def _get_trigger_inputs(self):
        """
        Returns a set of all unique trigger input values found in the blocks.
        Returns: Set[str] containing all unique block_trigger_inputs values
        """
        def fn_recursive_get_trigger(blocks):
            trigger_values = set()
            
            if blocks is not None:
                for name, block in blocks.items():
                    # Check if current block has trigger inputs(i.e. auto block)
                    if hasattr(block, 'block_trigger_inputs') and block.block_trigger_inputs is not None:
                        # Add all non-None values from the trigger inputs list
                        trigger_values.update(t for t in block.block_trigger_inputs if t is not None)
                    
                    # If block has blocks, recursively check them
                    if hasattr(block, 'blocks'):
                        nested_triggers = fn_recursive_get_trigger(block.blocks)
                        trigger_values.update(nested_triggers)
            
            return trigger_values
        
        trigger_inputs = set(self.block_trigger_inputs)
        trigger_inputs.update(fn_recursive_get_trigger(self.blocks))
        
        return trigger_inputs

    @property
    def trigger_inputs(self):
        return self._get_trigger_inputs()

    def __repr__(self):
        class_name = self.__class__.__name__
        base_class = self.__class__.__bases__[0].__name__
        header = (
            f"{class_name}(\n  Class: {base_class}\n"
            if base_class and base_class != "object"
            else f"{class_name}(\n"
        )

    
        if self.trigger_inputs:
            header += "\n"
            header += "  " + "=" * 100 + "\n"
            header += "  This pipeline contains blocks that are selected at runtime based on inputs.\n"
            header += f"  Trigger Inputs: {self.trigger_inputs}\n"
            # Get first trigger input as example
            example_input = next(t for t in self.trigger_inputs if t is not None)
            header += f"  Use `get_execution_blocks()` with input names to see selected blocks (e.g. `get_execution_blocks('{example_input}')`).\n"
            header += "  " + "=" * 100 + "\n\n"

        # Format description with proper indentation
        desc_lines = self.description.split('\n')
        desc = []
        # First line with "Description:" label
        desc.append(f"  Description: {desc_lines[0]}")
        # Subsequent lines with proper indentation
        if len(desc_lines) > 1:
            desc.extend(f"      {line}" for line in desc_lines[1:])
        desc = '\n'.join(desc) + '\n'

        # Components section - focus only on expected components
        expected_components = getattr(self, "expected_components", [])
        components_str = format_components(expected_components, indent_level=2, add_empty_lines=False)
        
        # Configs section - use format_configs with add_empty_lines=False
        expected_configs = getattr(self, "expected_configs", [])
        configs_str = format_configs(expected_configs, indent_level=2, add_empty_lines=False)

        # Blocks section - moved to the end with simplified format
        blocks_str = "  Blocks:\n"
        for i, (name, block) in enumerate(self.blocks.items()):
            # Get trigger input for this block
            trigger = None
            if hasattr(self, 'block_to_trigger_map'):
                trigger = self.block_to_trigger_map.get(name)
                # Format the trigger info
                if trigger is None:
                    trigger_str = "[default]"
                elif isinstance(trigger, (list, tuple)):
                    trigger_str = f"[trigger: {', '.join(str(t) for t in trigger)}]"
                else:
                    trigger_str = f"[trigger: {trigger}]"
                # For AutoPipelineBlocks, add bullet points
                blocks_str += f"    • {name} {trigger_str} ({block.__class__.__name__})\n"
            else:
                # For SequentialPipelineBlocks, show execution order
                blocks_str += f"    [{i}] {name} ({block.__class__.__name__})\n"
            
            # Add block description
            desc_lines = block.description.split('\n')
            indented_desc = desc_lines[0]
            if len(desc_lines) > 1:
                indented_desc += '\n' + '\n'.join('                   ' + line for line in desc_lines[1:])
            blocks_str += f"       Description: {indented_desc}\n\n"

        # Build the representation with conditional sections
        result = f"{header}\n{desc}"
        
        # Only add components section if it has content
        if components_str.strip():
            result += f"\n\n{components_str}"
        
        # Only add configs section if it has content
        if configs_str.strip():
            result += f"\n\n{configs_str}"
        
        # Always add blocks section
        result += f"\n\n{blocks_str})"
        
        return result


    @property
    def doc(self):
        return make_doc_string(
            self.inputs, 
            self.intermediates_inputs, 
            self.outputs, 
            self.description,
            class_name=self.__class__.__name__,
            expected_components=self.expected_components,
            expected_configs=self.expected_configs
        )


class SequentialPipelineBlocks(ModularPipelineBlocks):
    """
    A class that combines multiple pipeline block classes into one. When called, it will call each block in sequence.
    """
    block_classes = []
    block_names = []

    
    @property
    def description(self):
        return ""
    
    @property
    def model_name(self):
        return next(iter(self.blocks.values())).model_name
    

    @property
    def expected_components(self):
        expected_components = []
        for block in self.blocks.values():
            for component in block.expected_components:
                if component not in expected_components:
                    expected_components.append(component)
        return expected_components

    @property
    def expected_configs(self):
        expected_configs = []
        for block in self.blocks.values():
            for config in block.expected_configs:
                if config not in expected_configs:
                    expected_configs.append(config)
        return expected_configs

    @classmethod
    def from_blocks_dict(cls, blocks_dict: Dict[str, Any]) -> "SequentialPipelineBlocks":
        """Creates a SequentialPipelineBlocks instance from a dictionary of blocks.
        
        Args:
            blocks_dict: Dictionary mapping block names to block classes or instances
            
        Returns:
            A new SequentialPipelineBlocks instance
        """
        instance = cls()
        
        # Create instances if classes are provided
        blocks = {}
        for name, block in blocks_dict.items():
            if inspect.isclass(block):
                blocks[name] = block()
            else:
                blocks[name] = block
                
        instance.block_classes = [block.__class__ for block in blocks.values()]
        instance.block_names = list(blocks.keys())
        instance.blocks = blocks
        return instance
    
    def __init__(self):
        blocks = OrderedDict()
        for block_name, block_cls in zip(self.block_names, self.block_classes):
            blocks[block_name] = block_cls()
        self.blocks = blocks


    @property
    def required_inputs(self) -> List[str]:
        # Get the first block from the dictionary
        first_block = next(iter(self.blocks.values()))
        required_by_any = set(getattr(first_block, "required_inputs", set()))

        # Union with required inputs from all other blocks
        for block in list(self.blocks.values())[1:]:
            block_required = set(getattr(block, "required_inputs", set()))
            required_by_any.update(block_required)
        
        return list(required_by_any)
    
    # YiYi TODO: maybe we do not need this, it is only used in docstring, 
    # intermediate_inputs is by default required, unless you manually handle it inside the block
    @property
    def required_intermediates_inputs(self) -> List[str]:
        required_intermediates_inputs = []
        for input_param in self.intermediates_inputs:
            if input_param.required:
                required_intermediates_inputs.append(input_param.name)
        return required_intermediates_inputs

    # YiYi TODO: add test for this
    @property
    def inputs(self) -> List[Tuple[str, Any]]:
        return self.get_inputs()

    def get_inputs(self):
        named_inputs = [(name, block.inputs) for name, block in self.blocks.items()]
        combined_inputs = combine_inputs(*named_inputs)
        # mark Required inputs only if that input is required any of the blocks
        for input_param in combined_inputs:
            if input_param.name in self.required_inputs:
                input_param.required = True
            else:
                input_param.required = False
        return combined_inputs

    @property
    def intermediates_inputs(self) -> List[str]:
        return self.get_intermediates_inputs()
    
    def get_intermediates_inputs(self):
        inputs = []
        outputs = set()
        added_inputs = set()

        # Go through all blocks in order
        for block in self.blocks.values():
            # Add inputs that aren't in outputs yet
            for inp in block.intermediates_inputs:
                if inp.name not in outputs and inp.name not in added_inputs:
                    inputs.append(inp)
                    added_inputs.add(inp.name)

            # Only add outputs if the block cannot be skipped
            should_add_outputs = True
            if hasattr(block, "block_trigger_inputs") and None not in block.block_trigger_inputs:
                should_add_outputs = False
            
            if should_add_outputs:
                # Add this block's outputs
                block_intermediates_outputs = [out.name for out in block.intermediates_outputs]
                outputs.update(block_intermediates_outputs)
        return inputs

    @property
    def intermediates_outputs(self) -> List[str]:
        named_outputs = []
        for name, block in self.blocks.items():
            inp_names = set([inp.name for inp in block.intermediates_inputs])
            # so we only need to list new variables as intermediates_outputs, but if user wants to list these they modified it's still fine (a.k.a we don't enforce)
            # filter out them here so they do not end up as intermediates_outputs
            if name not in inp_names:
                named_outputs.append((name, block.intermediates_outputs))
        combined_outputs = combine_outputs(*named_outputs)
        return combined_outputs
    
    # YiYi TODO: I think we can remove the outputs property
    @property
    def outputs(self) -> List[str]:
        # return next(reversed(self.blocks.values())).intermediates_outputs
        return self.intermediates_outputs
    @torch.no_grad()
    def __call__(self, pipeline, state: PipelineState) -> PipelineState:
        for block_name, block in self.blocks.items():
            try:
                pipeline, state = block(pipeline, state)
            except Exception as e:
                error_msg = (
                    f"\nError in block: ({block_name}, {block.__class__.__name__})\n"
                    f"Error details: {str(e)}\n"
                    f"Traceback:\n{traceback.format_exc()}"
                )
                logger.error(error_msg)
                raise
        return pipeline, state
    
    def _get_trigger_inputs(self):
        """
        Returns a set of all unique trigger input values found in the blocks.
        Returns: Set[str] containing all unique block_trigger_inputs values
        """
        def fn_recursive_get_trigger(blocks):
            trigger_values = set()
            
            if blocks is not None:
                for name, block in blocks.items():
                    # Check if current block has trigger inputs(i.e. auto block)
                    if hasattr(block, 'block_trigger_inputs') and block.block_trigger_inputs is not None:
                        # Add all non-None values from the trigger inputs list
                        trigger_values.update(t for t in block.block_trigger_inputs if t is not None)
                    
                    # If block has blocks, recursively check them
                    if hasattr(block, 'blocks'):
                        nested_triggers = fn_recursive_get_trigger(block.blocks)
                        trigger_values.update(nested_triggers)
            
            return trigger_values
        
        return fn_recursive_get_trigger(self.blocks)

    @property
    def trigger_inputs(self):
        return self._get_trigger_inputs()

    def _traverse_trigger_blocks(self, trigger_inputs):
        # Convert trigger_inputs to a set for easier manipulation
        active_triggers = set(trigger_inputs)
        def fn_recursive_traverse(block, block_name, active_triggers):
            result_blocks = OrderedDict()
            
            # sequential(include loopsequential) or PipelineBlock
            if not hasattr(block, 'block_trigger_inputs'):
                if hasattr(block, 'blocks'):
                    # sequential or LoopSequentialPipelineBlocks (keep traversing)
                    for sub_block_name, sub_block in block.blocks.items():
                        blocks_to_update = fn_recursive_traverse(sub_block, sub_block_name, active_triggers)
                        blocks_to_update = fn_recursive_traverse(sub_block, sub_block_name, active_triggers)
                        blocks_to_update = {f"{block_name}.{k}": v for k,v in blocks_to_update.items()}
                        result_blocks.update(blocks_to_update)
                else:
                    # PipelineBlock
                    result_blocks[block_name] = block
                    # Add this block's output names to active triggers if defined
                    if hasattr(block, 'outputs'):
                        active_triggers.update(out.name for out in block.outputs)
                return result_blocks
                
            # auto
            else:
                # Find first block_trigger_input that matches any value in our active_triggers
                this_block = None
                matching_trigger = None
                for trigger_input in block.block_trigger_inputs:
                    if trigger_input is not None and trigger_input in active_triggers:
                        this_block = block.trigger_to_block_map[trigger_input]
                        matching_trigger = trigger_input
                        break
                
                # If no matches found, try to get the default (None) block
                if this_block is None and None in block.block_trigger_inputs:
                    this_block = block.trigger_to_block_map[None]
                    matching_trigger = None
                
                if this_block is not None:
                    # sequential/auto (keep traversing)
                    if hasattr(this_block, 'blocks'):
                        result_blocks.update(fn_recursive_traverse(this_block, block_name, active_triggers))
                    else:
                        # PipelineBlock
                        result_blocks[block_name] = this_block
                        # Add this block's output names to active triggers if defined
                        # YiYi TODO: do we need outputs here? can it just be intermediate_outputs? can we get rid of outputs attribute?
                        if hasattr(this_block, 'outputs'):
                            active_triggers.update(out.name for out in this_block.outputs)

            return result_blocks
        
        all_blocks = OrderedDict()
        for block_name, block in self.blocks.items():
            blocks_to_update = fn_recursive_traverse(block, block_name, active_triggers)
            all_blocks.update(blocks_to_update)
        return all_blocks
    
    def get_execution_blocks(self, *trigger_inputs):
        trigger_inputs_all = self.trigger_inputs

        if trigger_inputs is not None:

            if not isinstance(trigger_inputs, (list, tuple, set)):
                trigger_inputs = [trigger_inputs]
            invalid_inputs = [x for x in trigger_inputs if x not in trigger_inputs_all]
            if invalid_inputs:
                logger.warning(
                    f"The following trigger inputs will be ignored as they are not supported: {invalid_inputs}"
                )
                trigger_inputs = [x for x in trigger_inputs if x in trigger_inputs_all]
        
        if trigger_inputs is None:
            if None in trigger_inputs_all:
                trigger_inputs = [None]
            else:
                trigger_inputs = [trigger_inputs_all[0]]
        blocks_triggered = self._traverse_trigger_blocks(trigger_inputs)
        return SequentialPipelineBlocks.from_blocks_dict(blocks_triggered)
    
    def __repr__(self):
        class_name = self.__class__.__name__
        base_class = self.__class__.__bases__[0].__name__
        header = (
            f"{class_name}(\n  Class: {base_class}\n"
            if base_class and base_class != "object"
            else f"{class_name}(\n"
        )

    
        if self.trigger_inputs:
            header += "\n"
            header += "  " + "=" * 100 + "\n"
            header += "  This pipeline contains blocks that are selected at runtime based on inputs.\n"
            header += f"  Trigger Inputs: {self.trigger_inputs}\n"
            # Get first trigger input as example
            example_input = next(t for t in self.trigger_inputs if t is not None)
            header += f"  Use `get_execution_blocks()` with input names to see selected blocks (e.g. `get_execution_blocks('{example_input}')`).\n"
            header += "  " + "=" * 100 + "\n\n"

        # Format description with proper indentation
        desc_lines = self.description.split('\n')
        desc = []
        # First line with "Description:" label
        desc.append(f"  Description: {desc_lines[0]}")
        # Subsequent lines with proper indentation
        if len(desc_lines) > 1:
            desc.extend(f"      {line}" for line in desc_lines[1:])
        desc = '\n'.join(desc) + '\n'

        # Components section - focus only on expected components
        expected_components = getattr(self, "expected_components", [])
        components_str = format_components(expected_components, indent_level=2, add_empty_lines=False)
        
        # Configs section - use format_configs with add_empty_lines=False
        expected_configs = getattr(self, "expected_configs", [])
        configs_str = format_configs(expected_configs, indent_level=2, add_empty_lines=False)

        # Blocks section - moved to the end with simplified format
        blocks_str = "  Blocks:\n"
        for i, (name, block) in enumerate(self.blocks.items()):
            # Get trigger input for this block
            trigger = None
            if hasattr(self, 'block_to_trigger_map'):
                trigger = self.block_to_trigger_map.get(name)
                # Format the trigger info
                if trigger is None:
                    trigger_str = "[default]"
                elif isinstance(trigger, (list, tuple)):
                    trigger_str = f"[trigger: {', '.join(str(t) for t in trigger)}]"
                else:
                    trigger_str = f"[trigger: {trigger}]"
                # For AutoPipelineBlocks, add bullet points
                blocks_str += f"    • {name} {trigger_str} ({block.__class__.__name__})\n"
            else:
                # For SequentialPipelineBlocks, show execution order
                blocks_str += f"    [{i}] {name} ({block.__class__.__name__})\n"
            
            # Add block description
            desc_lines = block.description.split('\n')
            indented_desc = desc_lines[0]
            if len(desc_lines) > 1:
                indented_desc += '\n' + '\n'.join('                   ' + line for line in desc_lines[1:])
            blocks_str += f"       Description: {indented_desc}\n\n"

        # Build the representation with conditional sections
        result = f"{header}\n{desc}"
        
        # Only add components section if it has content
        if components_str.strip():
            result += f"\n\n{components_str}"
        
        # Only add configs section if it has content
        if configs_str.strip():
            result += f"\n\n{configs_str}"
        
        # Always add blocks section
        result += f"\n\n{blocks_str})"
        
        return result


    @property
    def doc(self):
        return make_doc_string(
            self.inputs, 
            self.intermediates_inputs, 
            self.outputs, 
            self.description,
            class_name=self.__class__.__name__,
            expected_components=self.expected_components,
            expected_configs=self.expected_configs
        )

#YiYi TODO: __repr__
class LoopSequentialPipelineBlocks(ModularPipelineBlocks):
    """
    A class that combines multiple pipeline block classes into a For Loop. When called, it will call each block in sequence.
    """

    model_name = None
    block_classes = []
    block_names = []
    
    @property
    def description(self) -> str:
        """Description of the block. Must be implemented by subclasses."""
        raise NotImplementedError("description method must be implemented in subclasses")

    @property
    def loop_expected_components(self) -> List[ComponentSpec]:
        return []
    
    @property
    def loop_expected_configs(self) -> List[ConfigSpec]:
        return []

    @property
    def loop_inputs(self) -> List[InputParam]:
        """List of input parameters. Must be implemented by subclasses."""
        return []

    @property
    def loop_intermediates_inputs(self) -> List[InputParam]:
        """List of intermediate input parameters. Must be implemented by subclasses."""
        return []

    @property
    def loop_intermediates_outputs(self) -> List[OutputParam]:
        """List of intermediate output parameters. Must be implemented by subclasses."""
        return []


    @property
    def loop_required_inputs(self) -> List[str]:
        input_names = []
        for input_param in self.loop_inputs:
            if input_param.required:
                input_names.append(input_param.name)
        return input_names

    @property
    def loop_required_intermediates_inputs(self) -> List[str]:
        input_names = []
        for input_param in self.loop_intermediates_inputs:
            if input_param.required:
                input_names.append(input_param.name)
        return input_names

    # modified from SequentialPipelineBlocks to include loop_expected_components
    @property
    def expected_components(self):
        expected_components = []
        for block in self.blocks.values():
            for component in block.expected_components:
                if component not in expected_components:
                    expected_components.append(component)
        for component in self.loop_expected_components:
            if component not in expected_components:
                expected_components.append(component)
        return expected_components

    # modified from SequentialPipelineBlocks to include loop_expected_configs
    @property
    def expected_configs(self):
        expected_configs = []
        for block in self.blocks.values():
            for config in block.expected_configs:
                if config not in expected_configs:
                    expected_configs.append(config)
        for config in self.loop_expected_configs:
            if config not in expected_configs:
                expected_configs.append(config)
        return expected_configs

    # modified from SequentialPipelineBlocks to include loop_inputs
    def get_inputs(self):
        named_inputs = [(name, block.inputs) for name, block in self.blocks.items()]
        named_inputs.append(("loop", self.loop_inputs))
        combined_inputs = combine_inputs(*named_inputs)
        # mark Required inputs only if that input is required any of the blocks
        for input_param in combined_inputs:
            if input_param.name in self.required_inputs:
                input_param.required = True
            else:
                input_param.required = False
        return combined_inputs

    # Copied from SequentialPipelineBlocks
    @property
    def inputs(self):
        return self.get_inputs()
    
    
    # modified from SequentialPipelineBlocks to include loop_intermediates_inputs
    @property
    def intermediates_inputs(self):
        intermediates = self.get_intermediates_inputs()
        intermediate_names = [input.name for input in intermediates]
        for loop_intermediate_input in self.loop_intermediates_inputs:
            if loop_intermediate_input.name not in intermediate_names:
                intermediates.append(loop_intermediate_input)
        return intermediates


    # Copied from SequentialPipelineBlocks
    def get_intermediates_inputs(self):
        inputs = []
        outputs = set()

        # Go through all blocks in order
        for block in self.blocks.values():
            # Add inputs that aren't in outputs yet
            inputs.extend(input_name for input_name in block.intermediates_inputs if input_name.name not in outputs)

            # Only add outputs if the block cannot be skipped
            should_add_outputs = True
            if hasattr(block, "block_trigger_inputs") and None not in block.block_trigger_inputs:
                should_add_outputs = False
            
            if should_add_outputs:
                # Add this block's outputs
                block_intermediates_outputs = [out.name for out in block.intermediates_outputs]
                outputs.update(block_intermediates_outputs)
        return inputs


    # modified from SequentialPipelineBlocks, if any additionan input required by the loop is required by the block
    @property
    def required_inputs(self) -> List[str]:
        # Get the first block from the dictionary
        first_block = next(iter(self.blocks.values()))
        required_by_any = set(getattr(first_block, "required_inputs", set()))

        required_by_loop = set(getattr(self, "loop_required_inputs", set()))
        required_by_any.update(required_by_loop)

        # Union with required inputs from all other blocks
        for block in list(self.blocks.values())[1:]:
            block_required = set(getattr(block, "required_inputs", set()))
            required_by_any.update(block_required)
        
        return list(required_by_any)

    # YiYi TODO: maybe we do not need this, it is only used in docstring, 
    # intermediate_inputs is by default required, unless you manually handle it inside the block
    @property
    def required_intermediates_inputs(self) -> List[str]:
        required_intermediates_inputs = []
        for input_param in self.intermediates_inputs:
            if input_param.required:
                required_intermediates_inputs.append(input_param.name)
        for input_param in self.loop_intermediates_inputs:
            if input_param.required:
                required_intermediates_inputs.append(input_param.name)
        return required_intermediates_inputs


    # YiYi TODO: this need to be thought about more
    # modified from SequentialPipelineBlocks to include loop_intermediates_outputs
    @property
    def intermediates_outputs(self) -> List[str]:
        named_outputs = [(name, block.intermediates_outputs) for name, block in self.blocks.items()]
        combined_outputs = combine_outputs(*named_outputs)
        for output in self.loop_intermediates_outputs:
            if output.name not in set([output.name for output in combined_outputs]):
                combined_outputs.append(output)
        return combined_outputs
    
    # YiYi TODO: this need to be thought about more
    # copied from SequentialPipelineBlocks
    @property
    def outputs(self) -> List[str]:
        return next(reversed(self.blocks.values())).intermediates_outputs


    def __init__(self):
        blocks = OrderedDict()
        for block_name, block_cls in zip(self.block_names, self.block_classes):
            blocks[block_name] = block_cls()
        self.blocks = blocks
    
    @classmethod
    def from_blocks_dict(cls, blocks_dict: Dict[str, Any]) -> "LoopSequentialPipelineBlocks":
        """Creates a LoopSequentialPipelineBlocks instance from a dictionary of blocks.
        
        Args:
            blocks_dict: Dictionary mapping block names to block instances
            
        Returns:
            A new LoopSequentialPipelineBlocks instance
        """
        instance = cls()
        instance.block_classes = [block.__class__ for block in blocks_dict.values()]
        instance.block_names = list(blocks_dict.keys())
        instance.blocks = blocks_dict
        return instance

    def loop_step(self, components, state: PipelineState, **kwargs):

        for block_name, block in self.blocks.items():
            try:
                components, state = block(components, state, **kwargs)
            except Exception as e:
                error_msg = (
                    f"\nError in block: ({block_name}, {block.__class__.__name__})\n"
                    f"Error details: {str(e)}\n"
                    f"Traceback:\n{traceback.format_exc()}"
                )
                logger.error(error_msg)
                raise
        return components, state
    
    def __call__(self, components, state: PipelineState) -> PipelineState:
        raise NotImplementedError("`__call__` method needs to be implemented by the subclass")
    
    
    def get_block_state(self, state: PipelineState) -> dict:
        """Get all inputs and intermediates in one dictionary"""
        data = {}
        
        # Check inputs
        for input_param in self.inputs:
            if input_param.name:
                value = state.get_input(input_param.name)
                if input_param.required and value is None:
                    raise ValueError(f"Required input '{input_param.name}' is missing")
                elif value is not None or (value is None and input_param.name not in data):
                    data[input_param.name] = value
            elif input_param.kwargs_type:
                # if kwargs_type is provided, get all inputs with matching kwargs_type
                if input_param.kwargs_type not in data:
                    data[input_param.kwargs_type] = {}
                inputs_kwargs = state.get_inputs_kwargs(input_param.kwargs_type)
                if inputs_kwargs:
                    for k, v in inputs_kwargs.items():
                        if v is not None:
                            data[k] = v
                            data[input_param.kwargs_type][k] = v

        # Check intermediates
        for input_param in self.intermediates_inputs:
            if input_param.name:
                value = state.get_intermediate(input_param.name)
                if input_param.required and value is None:
                    raise ValueError(f"Required intermediate input '{input_param.name}' is missing")
                elif value is not None or (value is None and input_param.name not in data):
                    data[input_param.name] = value
            elif input_param.kwargs_type:
                # if kwargs_type is provided, get all intermediates with matching kwargs_type
                if input_param.kwargs_type not in data:
                    data[input_param.kwargs_type] = {}
                intermediates_kwargs = state.get_intermediates_kwargs(input_param.kwargs_type)
                if intermediates_kwargs:
                    for k, v in intermediates_kwargs.items():
                        if v is not None:
                            if k not in data:
                                data[k] = v
                            data[input_param.kwargs_type][k] = v
        return BlockState(**data)
    
    def add_block_state(self, state: PipelineState, block_state: BlockState):
        for output_param in self.intermediates_outputs:
            if not hasattr(block_state, output_param.name):
                raise ValueError(f"Intermediate output '{output_param.name}' is missing in block state")
            param = getattr(block_state, output_param.name)
            state.add_intermediate(output_param.name, param, output_param.kwargs_type)

        for input_param in self.intermediates_inputs:
            if input_param.name and hasattr(block_state, input_param.name):
                param = getattr(block_state, input_param.name)
                # Only add if the value is different from what's in the state
                current_value = state.get_intermediate(input_param.name)
                if current_value is not param:  # Using identity comparison to check if object was modified
                    state.add_intermediate(input_param.name, param, input_param.kwargs_type)
            elif input_param.kwargs_type:
                # if it is a kwargs type, e.g. "guider_input_fields", it is likely to be a list of parameters
                # we need to first find out which inputs are and loop through them.
                intermediates_kwargs = state.get_intermediates_kwargs(input_param.kwargs_type)
                for param_name, current_value in intermediates_kwargs.items():
                    if not hasattr(block_state, param_name):
                        continue
                    param = getattr(block_state, param_name)
                    if current_value is not param:  # Using identity comparison to check if object was modified
                        state.add_intermediate(param_name, param, input_param.kwargs_type)


    @property
    def doc(self):
        return make_doc_string(
            self.inputs, 
            self.intermediates_inputs, 
            self.outputs, 
            self.description,
            class_name=self.__class__.__name__,
            expected_components=self.expected_components,
            expected_configs=self.expected_configs
        )

    # modified from SequentialPipelineBlocks, 
    #(does not need trigger_inputs related part so removed them, 
    # do not need to support auto block for loop blocks)
    def __repr__(self):
        class_name = self.__class__.__name__
        base_class = self.__class__.__bases__[0].__name__
        header = (
            f"{class_name}(\n  Class: {base_class}\n"
            if base_class and base_class != "object"
            else f"{class_name}(\n"
        )

        # Format description with proper indentation
        desc_lines = self.description.split('\n')
        desc = []
        # First line with "Description:" label
        desc.append(f"  Description: {desc_lines[0]}")
        # Subsequent lines with proper indentation
        if len(desc_lines) > 1:
            desc.extend(f"      {line}" for line in desc_lines[1:])
        desc = '\n'.join(desc) + '\n'

        # Components section - focus only on expected components
        expected_components = getattr(self, "expected_components", [])
        components_str = format_components(expected_components, indent_level=2, add_empty_lines=False)
        
        # Configs section - use format_configs with add_empty_lines=False
        expected_configs = getattr(self, "expected_configs", [])
        configs_str = format_configs(expected_configs, indent_level=2, add_empty_lines=False)

        # Blocks section - moved to the end with simplified format
        blocks_str = "  Blocks:\n"
        for i, (name, block) in enumerate(self.blocks.items()):
            
            # For SequentialPipelineBlocks, show execution order
            blocks_str += f"    [{i}] {name} ({block.__class__.__name__})\n"
            
            # Add block description
            desc_lines = block.description.split('\n')
            indented_desc = desc_lines[0]
            if len(desc_lines) > 1:
                indented_desc += '\n' + '\n'.join('                   ' + line for line in desc_lines[1:])
            blocks_str += f"       Description: {indented_desc}\n\n"

        # Build the representation with conditional sections
        result = f"{header}\n{desc}"
        
        # Only add components section if it has content
        if components_str.strip():
            result += f"\n\n{components_str}"
        
        # Only add configs section if it has content
        if configs_str.strip():
            result += f"\n\n{configs_str}"
        
        # Always add blocks section
        result += f"\n\n{blocks_str})"
        
        return result

    @torch.compiler.disable
    def progress_bar(self, iterable=None, total=None):
        if not hasattr(self, "_progress_bar_config"):
            self._progress_bar_config = {}
        elif not isinstance(self._progress_bar_config, dict):
            raise ValueError(
                f"`self._progress_bar_config` should be of type `dict`, but is {type(self._progress_bar_config)}."
            )

        if iterable is not None:
            return tqdm(iterable, **self._progress_bar_config)
        elif total is not None:
            return tqdm(total=total, **self._progress_bar_config)
        else:
            raise ValueError("Either `total` or `iterable` has to be defined.")

    def set_progress_bar_config(self, **kwargs):
        self._progress_bar_config = kwargs


# YiYi TODO: 
# 1. move the modular_repo arg and the logic to fetch info from repo out of __init__ so that __init__ alwasy create an default modular_model_index config
# 2. look into the serialization of modular_model_index.json, make sure the items are properly ordered like model_index.json (currently a mess)
# 3. do we need ConfigSpec? seems pretty unnecessrary for loader, can just add and kwargs to the loader
# 4. add validator for methods where we accpet kwargs to be passed to from_pretrained()
class ModularLoader(ConfigMixin, PushToHubMixin):
    """
    Base class for all Modular pipelines loaders.

    """
    config_name = "modular_model_index.json"


    def register_components(self, **kwargs):
        """
        Register components with their corresponding specifications.
        
        This method is responsible for:
        1. Sets component objects as attributes on the loader (e.g., self.unet = unet)
        2. Updates the modular_model_index.json configuration for serialization
        4. Adds components to the component manager if one is attached
        
        This method is called when:
        - Components are first initialized in __init__: 
           - from_pretrained components not loaded during __init__ so they are registered as None; 
           - non from_pretrained components are created during __init__ and registered as the object itself
        - Components are updated with the `update()` method: e.g. loader.update(unet=unet) or loader.update(guider=guider_spec)
        - (from_pretrained) Components are loaded with the `load()` method: e.g. loader.load(component_names=["unet"])
        
        Args:
            **kwargs: Keyword arguments where keys are component names and values are component objects.
                      E.g., register_components(unet=unet_model, text_encoder=encoder_model)
        
        Notes:
            - Components must be created from ComponentSpec (have _diffusers_load_id attribute)
            - When registering None for a component, it updates the modular_model_index.json config but sets attribute to None
        """
        for name, module in kwargs.items():
            # current component spec
            component_spec = self._component_specs.get(name)
            if component_spec is None:
                logger.warning(f"ModularLoader.register_components: skipping unknown component '{name}'")
                continue
            
            # check if it is the first time registration, i.e. calling from __init__
            is_registered = hasattr(self, name)

            # make sure the component is created from ComponentSpec
            if module is not None and not hasattr(module, "_diffusers_load_id"):
                raise ValueError(f"`ModularLoader` only supports components created from `ComponentSpec`.")

            if module is not None:
                # actual library and class name of the module
                library, class_name = _fetch_class_library_tuple(module) # e.g. ("diffusers", "UNet2DConditionModel")

                # extract the loading spec from the updated component spec that'll be used as part of modular_model_index.json config
                # e.g. {"repo": "stabilityai/stable-diffusion-2-1", 
                #       "type_hint": ("diffusers", "UNet2DConditionModel"), 
                #       "subfolder": "unet",
                #       "variant": None,
                #       "revision": None}
                component_spec_dict = self._component_spec_to_dict(component_spec)
            
            else:
                # if module is None, e.g. self.register_components(unet=None) during __init__
                # we do not update the spec, 
                # but we still need to update the modular_model_index.json config based oncomponent spec
                library, class_name = None, None
                component_spec_dict = self._component_spec_to_dict(component_spec)
            register_dict = {name: (library, class_name, component_spec_dict)}

            # set the component as attribute
            # if it is not set yet, just set it and skip the process to check and warn below
            if not is_registered:
                self.register_to_config(**register_dict)
                setattr(self, name, module)
                if module is not None and module._diffusers_load_id != "null" and self._component_manager is not None:
                    self._component_manager.add(name, module, self._collection)
                continue
            
            current_module = getattr(self, name, None)
            # skip if the component is already registered with the same object
            if current_module is module:
                logger.info(f"ModularLoader.register_components: {name} is already registered with same object, skipping")
                continue

            # warn if unregister
            if current_module is not None and module is None:
                logger.info(
                    f"ModularLoader.register_components: setting '{name}' to None "
                    f"(was {current_module.__class__.__name__})"
                )
            # same type, new instance → replace but send debug log
            elif current_module is not None \
                and module is not None \
                and isinstance(module, current_module.__class__) \
                and current_module != module:
                logger.debug(
                    f"ModularLoader.register_components: replacing existing '{name}' "
                    f"(same type {type(current_module).__name__}, new instance)"
                )

            # update modular_model_index.json config
            self.register_to_config(**register_dict)
            # finally set models
            setattr(self, name, module)
            # add to component manager if one is attached
            if module is not None and module._diffusers_load_id != "null" and self._component_manager is not None:
                self._component_manager.add(name, module, self._collection)


    
    # YiYi TODO: add warning for passing multiple ComponentSpec/ConfigSpec with the same name
    def __init__(self, specs: List[Union[ComponentSpec, ConfigSpec]], pretrained_model_name_or_path: Optional[str] = None, component_manager: Optional[ComponentsManager] = None, collection: Optional[str] = None, **kwargs):
        """
        Initialize the loader with a list of component specs and config specs.
        """
        self._component_manager = component_manager
        self._collection = collection
        self._component_specs = {
            spec.name: deepcopy(spec) for spec in specs if isinstance(spec, ComponentSpec)
        }
        self._config_specs = {
            spec.name: deepcopy(spec) for spec in specs if isinstance(spec, ConfigSpec)
        }

        # update component_specs and config_specs from modular_repo
        if pretrained_model_name_or_path is not None:
            config_dict = self.load_config(pretrained_model_name_or_path, **kwargs)

            for name, value in config_dict.items():
                # only update component_spec for from_pretrained components
                if name in self._component_specs and self._component_specs[name].default_creation_method == "from_pretrained" and isinstance(value, (tuple, list)) and len(value) == 3:
                    library, class_name, component_spec_dict = value
                    component_spec = self._dict_to_component_spec(name, component_spec_dict)
                    self._component_specs[name] = component_spec

                elif name in self._config_specs:
                    self._config_specs[name].default = value
        
        register_components_dict = {}
        for name, component_spec in self._component_specs.items():
            if component_spec.default_creation_method == "from_config":
                component = component_spec.create()
            else:
                component = None
            register_components_dict[name] = component
        self.register_components(**register_components_dict)
        
        default_configs = {}
        for name, config_spec in self._config_specs.items():
            default_configs[name] = config_spec.default
        self.register_to_config(**default_configs)


    @property
    def device(self) -> torch.device:
        r"""
        Returns:
            `torch.device`: The torch device on which the pipeline is located.
        """
        modules = self.components.values()
        modules = [m for m in modules if isinstance(m, torch.nn.Module)]

        for module in modules:
            return module.device

        return torch.device("cpu")

    @property
    # Copied from diffusers.pipelines.pipeline_utils.DiffusionPipeline._execution_device
    def _execution_device(self):
        r"""
        Returns the device on which the pipeline's models will be executed. After calling
        [`~DiffusionPipeline.enable_sequential_cpu_offload`] the execution device can only be inferred from
        Accelerate's module hooks.
        """
        for name, model in self.components.items():
            if not isinstance(model, torch.nn.Module):
                continue

            if not hasattr(model, "_hf_hook"):
                return self.device
            for module in model.modules():
                if (
                    hasattr(module, "_hf_hook")
                    and hasattr(module._hf_hook, "execution_device")
                    and module._hf_hook.execution_device is not None
                ):
                    return torch.device(module._hf_hook.execution_device)
        return self.device
    

    @property
    def dtype(self) -> torch.dtype:
        r"""
        Returns:
            `torch.dtype`: The torch dtype on which the pipeline is located.
        """
        modules = self.components.values()
        modules = [m for m in modules if isinstance(m, torch.nn.Module)]

        for module in modules:
            return module.dtype

        return torch.float32


    @property
    def components(self) -> Dict[str, Any]:
        # return only components we've actually set as attributes on self
        return {
            name: getattr(self, name)
            for name in self._component_specs.keys()
            if hasattr(self, name)
        }

    def update(self, **kwargs):
        """
        Update components and configs after instance creation.
        
        Args:

        """   
        """
        Update components and configuration values after the loader has been instantiated.
        
        This method allows you to:
        1. Replace existing components with new ones (e.g., updating the unet or text_encoder)
        2. Update configuration values (e.g., changing requires_safety_checker flag)
        
        Args:
            **kwargs: Component objects or configuration values to update:
                - Component objects: Must be created using ComponentSpec (e.g., `unet=new_unet, text_encoder=new_encoder`)
                - Configuration values: Simple values to update configuration settings (e.g., `requires_safety_checker=False`)
                - ComponentSpec objects: if passed a ComponentSpec object, only support from_config spec, will call create() method to create it
        
        Raises:
            ValueError: If a component wasn't created using ComponentSpec (doesn't have `_diffusers_load_id` attribute)
            
        Examples:
            ```python
            # Update multiple components at once
            loader.update(
                unet=new_unet_model,
                text_encoder=new_text_encoder
            )
            
            # Update configuration values
            loader.update(
                requires_safety_checker=False,
                guidance_rescale=0.7
            )
            
            # Update both components and configs together
            loader.update(
                unet=new_unet_model,
                requires_safety_checker=False
            )
            # update with ComponentSpec objects
            loader.update(
                guider=ComponentSpec(name="guider", type_hint=ClassifierFreeGuidance, config={"guidance_scale": 5.0}, default_creation_method="from_config")
            )
            ```
        """   

        # extract component_specs_updates & config_specs_updates from `specs`
        passed_component_specs = {k: kwargs.pop(k) for k in self._component_specs if k in kwargs and isinstance(kwargs[k], ComponentSpec)}
        passed_components = {k: kwargs.pop(k) for k in self._component_specs if k in kwargs and not isinstance(kwargs[k], ComponentSpec)}
        passed_config_values = {k: kwargs.pop(k) for k in self._config_specs if k in kwargs}

        for name, component in passed_components.items():
            if not hasattr(component, "_diffusers_load_id"):
                raise ValueError(f"`ModularLoader` only supports components created from `ComponentSpec`.")  

            # YiYi TODO: remove this if we remove support for non config mixin components in `create()` method
            if component._diffusers_load_id == "null" and not isinstance(component, ConfigMixin):
                raise ValueError(
                    f"The passed component '{name}' is not supported in update() method "
                    f"because it is not supported in `ComponentSpec.from_component()`. "
                    f"Please pass a ComponentSpec object instead."
                )
            current_component_spec = self._component_specs[name]
            # warn if type changed
            if current_component_spec.type_hint is not None and not isinstance(component, current_component_spec.type_hint):
                logger.warning(f"ModularLoader.update: adding {name} with new type: {component.__class__.__name__}, previous type: {current_component_spec.type_hint.__name__}")
            # update _component_specs based on the new component
            new_component_spec = ComponentSpec.from_component(name, component)
            self._component_specs[name] = new_component_spec
    
        if len(kwargs) > 0:
            logger.warning(f"Unexpected keyword arguments, will be ignored: {kwargs.keys()}")
        
        created_components = {}
        for name, component_spec in passed_component_specs.items():
            if component_spec.default_creation_method == "from_pretrained":
                raise ValueError(f"ComponentSpec object with default_creation_method == 'from_pretrained' is not supported in update() method")
            created_components[name] = component_spec.create()
            current_component_spec = self._component_specs[name]
            # warn if type changed
            if current_component_spec.type_hint is not None and not isinstance(created_components[name], current_component_spec.type_hint):
                logger.warning(f"ModularLoader.update: adding {name} with new type: {created_components[name].__class__.__name__}, previous type: {current_component_spec.type_hint.__name__}")
            # update _component_specs based on the user passed component_spec
            self._component_specs[name] = component_spec
        self.register_components(**passed_components, **created_components)


        config_to_register = {}
        for name, new_value in passed_config_values.items():

            # e.g. requires_aesthetics_score = False
            self._config_specs[name].default = new_value
            config_to_register[name] = new_value
        self.register_to_config(**config_to_register)


    # YiYi TODO: support map for additional from_pretrained kwargs
    def load(self, component_names: Optional[List[str]] = None, **kwargs):
        """
        Load selectedcomponents from specs.
        
        Args:
            component_names: List of component names to load
            **kwargs: additional kwargs to be passed to `from_pretrained()`.Can be:
             - a single value to be applied to all components to be loaded, e.g. torch_dtype=torch.bfloat16
             - a dict, e.g. torch_dtype={"unet": torch.bfloat16, "default": torch.float32}
             - if potentially override ComponentSpec if passed a different loading field in kwargs, e.g. `repo`, `variant`, `revision`, etc.
        """
        # if not specific name, load all the components with default_creation_method == "from_pretrained"
        if component_names is None:
            component_names = [name for name in self._component_specs.keys() if self._component_specs[name].default_creation_method == "from_pretrained"]
        elif not isinstance(component_names, list):
            component_names = [component_names]

        components_to_load = set([name for name in component_names if name in self._component_specs])
        unknown_component_names = set([name for name in component_names if name not in self._component_specs])
        if len(unknown_component_names) > 0:
            logger.warning(f"Unknown components will be ignored: {unknown_component_names}")
        
        components_to_register = {}
        for name in components_to_load:
            spec = self._component_specs[name]
            component_load_kwargs = {}
            for key, value in kwargs.items():
                if not isinstance(value, dict):
                    # if the value is a single value, apply it to all components
                    component_load_kwargs[key] = value
                else:
                    if name in value:
                        # if it is a dict, check if the component name is in the dict
                        component_load_kwargs[key] = value[name]
                    elif "default" in value:
                        # check if the default is specified
                        component_load_kwargs[key] = value["default"]
            try:
                components_to_register[name] = spec.load(**component_load_kwargs)
            except Exception as e:
                logger.warning(f"Failed to create component '{name}': {e}")
            
        # Register all components at once
        self.register_components(**components_to_register)

    # YiYi TODO: should support to method
    def to(self, *args, **kwargs):
        pass

    # YiYi TODO: 
    # 1. should support save some components too! currently only modular_model_index.json is saved
    # 2. maybe order the json file to make it more readable: configs first, then components
    def save_pretrained(self, save_directory: Union[str, os.PathLike], push_to_hub: bool = False, spec_only: bool = True, **kwargs):

        component_names = list(self._component_specs.keys())
        config_names = list(self._config_specs.keys())
        self.register_to_config(_components_names=component_names, _configs_names=config_names)
        self.save_config(save_directory=save_directory, push_to_hub=push_to_hub, **kwargs)
        config = dict(self.config)
        config.pop("_components_names", None)
        config.pop("_configs_names", None)
        self._internal_dict = FrozenDict(config)

    
    @classmethod
    @validate_hf_hub_args
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], spec_only: bool = True, component_manager: Optional[ComponentsManager] = None, collection: Optional[str] = None, **kwargs):
        
        config_dict = cls.load_config(pretrained_model_name_or_path, **kwargs)
        expected_component = set(config_dict.pop("_components_names"))
        expected_config = set(config_dict.pop("_configs_names"))

        component_specs = []
        config_specs = []
        for name, value in config_dict.items():
            if name in expected_component and isinstance(value, (tuple, list)) and len(value) == 3:
                library, class_name, component_spec_dict = value
                    # only pick up pretrained components from the repo
                if component_spec_dict.get("repo", None) is not None:
                    component_spec = cls._dict_to_component_spec(name, component_spec_dict)
                    component_specs.append(component_spec)

            elif name in expected_config:
                config_specs.append(ConfigSpec(name=name, default=value))

        return cls(component_specs + config_specs, component_manager=component_manager, collection=collection)

    
    @staticmethod
    def _component_spec_to_dict(component_spec: ComponentSpec) -> Any:
        """
        Convert a ComponentSpec into a JSON‐serializable dict for saving in
        `modular_model_index.json`.

        This dict contains:
          - "type_hint": Tuple[str, str]
              The fully‐qualified module path and class name of the component.
          - All loading fields defined by `component_spec.loading_fields()`, typically:
              - "repo": Optional[str]
                  The model repository (e.g., "stabilityai/stable-diffusion-xl").
              - "subfolder": Optional[str]
                  A subfolder within the repo where this component lives.
              - "variant": Optional[str]
                  An optional variant identifier for the model.
              - "revision": Optional[str]
                  A specific git revision (commit hash, tag, or branch).
              - ... any other loading fields defined on the spec.

        Args:
            component_spec (ComponentSpec):
                The spec object describing one pipeline component.

        Returns:
            Dict[str, Any]: A mapping suitable for JSON serialization.

        Example:
            >>> from diffusers.pipelines.modular_pipeline_utils import ComponentSpec
            >>> from diffusers.models.unet import UNet2DConditionModel
            >>> spec = ComponentSpec(
            ...     name="unet",
            ...     type_hint=UNet2DConditionModel,
            ...     config=None,
            ...     repo="path/to/repo",
            ...     subfolder="subfolder",
            ...     variant=None,
            ...     revision=None,
            ...     default_creation_method="from_pretrained",
            ... )
            >>> ModularLoader._component_spec_to_dict(spec)
            {
                "type_hint": ("diffusers.models.unet", "UNet2DConditionModel"),
                "repo": "path/to/repo",
                "subfolder": "subfolder",
                "variant": None,
                "revision": None,
            }
        """
        if component_spec.type_hint is not None:
            lib_name, cls_name = _fetch_class_library_tuple(component_spec.type_hint)
        else:
            lib_name = None
            cls_name = None
        load_spec_dict = {k: getattr(component_spec, k) for k in component_spec.loading_fields()}
        return {
            "type_hint": (lib_name, cls_name),
            **load_spec_dict,
        }

    @staticmethod
    def _dict_to_component_spec(
        name: str,
        spec_dict: Dict[str, Any],
    ) -> ComponentSpec:
        """
        Reconstruct a ComponentSpec from a dict.
        """
        # make a shallow copy so we can pop() safely
        spec_dict = spec_dict.copy()
        # pull out and resolve the stored type_hint
        lib_name, cls_name = spec_dict.pop("type_hint")
        if lib_name is not None and cls_name is not None:
            type_hint = simple_get_class_obj(lib_name, cls_name)
        else:
            type_hint = None

        # re‐assemble the ComponentSpec
        return ComponentSpec(
            name=name,
            type_hint=type_hint,
            **spec_dict,
        )


class ModularPipeline:
    """
    Base class for all Modular pipelines.

    Args:
        blocks: ModularPipelineBlocks, the blocks to be used in the pipeline
        loader: ModularLoader, the loader to be used in the pipeline
    """

    def __init__(self, blocks: ModularPipelineBlocks, loader: ModularLoader):
        self.blocks = blocks
        self.loader = loader

    def __repr__(self):
        blocks_class = self.blocks.__class__.__name__
        loader_class = self.loader.__class__.__name__
        return f"ModularPipeline(blocks={blocks_class}, loader={loader_class})"

    @property
    def default_call_parameters(self) -> Dict[str, Any]:
        params = {}
        for input_param in self.blocks.inputs:
            params[input_param.name] = input_param.default
        return params

    def __call__(self, state: PipelineState = None, output: Union[str, List[str]] = None, **kwargs):
        """
        Run one or more blocks in sequence, optionally you can pass a previous pipeline state.
        """
        if state is None:
            state = PipelineState()


        # Make a copy of the input kwargs
        passed_kwargs = kwargs.copy()


        # Add inputs to state, using defaults if not provided in the kwargs or the state
        # if same input already in the state, will override it if provided in the kwargs

        intermediates_inputs = [inp.name for inp in self.blocks.intermediates_inputs]
        for expected_input_param in self.blocks.inputs:
            name = expected_input_param.name
            default = expected_input_param.default
            kwargs_type = expected_input_param.kwargs_type
            if name in passed_kwargs:
                if name not in intermediates_inputs:
                    state.add_input(name, passed_kwargs.pop(name), kwargs_type)
                else:
                    state.add_input(name, passed_kwargs[name], kwargs_type)
            elif name not in state.inputs:
                state.add_input(name, default, kwargs_type)

        for expected_intermediate_param in self.blocks.intermediates_inputs:
            name = expected_intermediate_param.name
            kwargs_type = expected_intermediate_param.kwargs_type
            if name in passed_kwargs:
                state.add_intermediate(name, passed_kwargs.pop(name), kwargs_type)

        # Warn about unexpected inputs
        if len(passed_kwargs) > 0:
            warnings.warn(f"Unexpected input '{passed_kwargs.keys()}' provided. This input will be ignored.")
        # Run the pipeline
        with torch.no_grad():
            try:
                pipeline, state = self.blocks(self.loader, state)
            except Exception:
                error_msg = f"Error in block: ({self.blocks.__class__.__name__}):\n"
                logger.error(error_msg)
                raise

        if output is None:
            return state


        elif isinstance(output, str):
            return state.get_intermediate(output)

        elif isinstance(output, (list, tuple)):
            return state.get_intermediates(output)
        else:
            raise ValueError(f"Output '{output}' is not a valid output type")

    
    def load_components(self, component_names: Optional[List[str]] = None, **kwargs):
        self.loader.load(component_names=component_names, **kwargs)
    
    def update_components(self, **kwargs):
        self.loader.update(**kwargs)
    
    @classmethod
    @validate_hf_hub_args
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],  trust_remote_code: Optional[bool] = None, component_manager: Optional[ComponentsManager] = None, collection: Optional[str] = None, **kwargs):
        blocks = ModularPipelineBlocks.from_pretrained(pretrained_model_name_or_path, trust_remote_code=trust_remote_code, **kwargs)
        pipeline = blocks.init_pipeline(pretrained_model_name_or_path, component_manager=component_manager, collection=collection, **kwargs)
        return pipeline
    
    def save_pretrained(self, save_directory: Optional[Union[str, os.PathLike]] = None, push_to_hub: bool = False, **kwargs):
        self.blocks.save_pretrained(save_directory=save_directory, push_to_hub=push_to_hub, **kwargs)
        self.loader.save_pretrained(save_directory=save_directory, push_to_hub=push_to_hub, **kwargs)
    
    
    @property
    def doc(self):
        return self.blocks.doc