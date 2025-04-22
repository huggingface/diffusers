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

import traceback
import warnings
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Union, Optional, Type


import torch
from tqdm.auto import tqdm
import re

from ..configuration_utils import ConfigMixin
from ..utils import (
    is_accelerate_available,
    is_accelerate_version,
    logging,
)
from .pipeline_loading_utils import _get_pipeline_class


if is_accelerate_available():
    import accelerate

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


MODULAR_PIPELINE_MAPPING = OrderedDict(
    [
        ("stable-diffusion-xl", "StableDiffusionXLModularPipeline"),
    ]
)


@dataclass
class PipelineState:
    """
    [`PipelineState`] stores the state of a pipeline. It is used to pass data between pipeline blocks.
    """

    inputs: Dict[str, Any] = field(default_factory=dict)
    intermediates: Dict[str, Any] = field(default_factory=dict)

    def add_input(self, key: str, value: Any):
        self.inputs[key] = value

    def add_intermediate(self, key: str, value: Any):
        self.intermediates[key] = value

    def get_input(self, key: str, default: Any = None) -> Any:
        return self.inputs.get(key, default)

    def get_inputs(self, keys: List[str], default: Any = None) -> Dict[str, Any]:
        return {key: self.inputs.get(key, default) for key in keys}

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

        return (
            f"PipelineState(\n"
            f"  inputs={{\n{inputs}\n  }},\n"
            f"  intermediates={{\n{intermediates}\n  }}\n"
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
                if any(hasattr(val, "shape") and hasattr(val, "dtype") for val in v.values()):
                    shapes = {k: val.shape for k, val in v.items() if hasattr(val, "shape")}
                    return f"Dict of Tensors with shapes {shapes}"
                return repr(v)
            
            # Default case
            return repr(v)

        attributes = "\n".join(f"    {k}: {format_value(v)}" for k, v in self.__dict__.items())
        return f"BlockState(\n{attributes}\n)"


@dataclass
class ComponentSpec:
    """Specification for a pipeline component."""
    name: str
    type_hint: Type
    description: Optional[str] = None
    obj: Any = None # you can create a default component if it is a stateless class like scheduler, guider or image processor
    default_class_name: Union[str, List[str], Tuple[str, str]] = None  # Either "class_name" or ["module", "class_name"]
    default_repo: Optional[Union[str, List[str]]] = None # either "repo" or ["repo", "subfolder"]

@dataclass 
class ConfigSpec:
    """Specification for a pipeline configuration parameter."""
    name: str
    default: Any
    description: Optional[str] = None


@dataclass
class InputParam:
    name: str
    type_hint: Any = None
    default: Any = None
    required: bool = False
    description: str = ""

    def __repr__(self):
        return f"<{self.name}: {'required' if self.required else 'optional'}, default={self.default}>"

@dataclass 
class OutputParam:
    name: str
    type_hint: Any = None
    description: str = ""

    def __repr__(self):
        return f"<{self.name}: {self.type_hint.__name__ if hasattr(self.type_hint, '__name__') else str(self.type_hint)}>"

def format_inputs_short(inputs):
    """
    Format input parameters into a string representation, with required params first followed by optional ones.
    
    Args:
        inputs: List of input parameters with 'required' and 'name' attributes, and 'default' for optional params
        
    Returns:
        str: Formatted string of input parameters
    
    Example:
        >>> inputs = [
        ...     InputParam(name="prompt", required=True),
        ...     InputParam(name="image", required=True),
        ...     InputParam(name="guidance_scale", required=False, default=7.5),
        ...     InputParam(name="num_inference_steps", required=False, default=50)
        ... ]
        >>> format_inputs_short(inputs)
        'prompt, image, guidance_scale=7.5, num_inference_steps=50'
    """
    required_inputs = [param for param in inputs if param.required]
    optional_inputs = [param for param in inputs if not param.required]
    
    required_str = ", ".join(param.name for param in required_inputs)
    optional_str = ", ".join(f"{param.name}={param.default}" for param in optional_inputs)
    
    inputs_str = required_str
    if optional_str:
        inputs_str = f"{inputs_str}, {optional_str}" if required_str else optional_str
        
    return inputs_str


def format_intermediates_short(intermediates_inputs: List[InputParam], required_intermediates_inputs: List[str], intermediates_outputs: List[OutputParam]) -> str:
    """
    Formats intermediate inputs and outputs of a block into a string representation.
    
    Args:
        intermediates_inputs: List of intermediate input parameters
        required_intermediates_inputs: List of required intermediate input names
        intermediates_outputs: List of intermediate output parameters
    
    Returns:
        str: Formatted string like:
            Intermediates:
                - inputs: Required(latents), dtype
                - modified: latents  # variables that appear in both inputs and outputs
                - outputs: images    # new outputs only
    """
    # Handle inputs
    input_parts = []
    for inp in intermediates_inputs:
        if inp.name in required_intermediates_inputs:
            input_parts.append(f"Required({inp.name})")
        else:
            input_parts.append(inp.name)
    
    # Handle modified variables (appear in both inputs and outputs)
    inputs_set = {inp.name for inp in intermediates_inputs}
    modified_parts = []
    new_output_parts = []
    
    for out in intermediates_outputs:
        if out.name in inputs_set:
            modified_parts.append(out.name)
        else:
            new_output_parts.append(out.name)
    
    result = []
    if input_parts:
        result.append(f"    - inputs: {', '.join(input_parts)}")
    if modified_parts:
        result.append(f"    - modified: {', '.join(modified_parts)}")
    if new_output_parts:
        result.append(f"    - outputs: {', '.join(new_output_parts)}")
        
    return "\n".join(result) if result else "    (none)"


def format_params(params: List[Union[InputParam, OutputParam]], header: str = "Args", indent_level: int = 4, max_line_length: int = 115) -> str:
    """Format a list of InputParam or OutputParam objects into a readable string representation.

    Args:
        params: List of InputParam or OutputParam objects to format
        header: Header text to use (e.g. "Args" or "Returns")
        indent_level: Number of spaces to indent each parameter line (default: 4)
        max_line_length: Maximum length for each line before wrapping (default: 115)

    Returns:
        A formatted string representing all parameters
    """
    if not params:
        return ""
        
    base_indent = " " * indent_level
    param_indent = " " * (indent_level + 4)
    desc_indent = " " * (indent_level + 8)
    formatted_params = []
    
    def get_type_str(type_hint):
        if hasattr(type_hint, "__origin__") and type_hint.__origin__ is Union:
            types = [t.__name__ if hasattr(t, "__name__") else str(t) for t in type_hint.__args__]
            return f"Union[{', '.join(types)}]"
        return type_hint.__name__ if hasattr(type_hint, "__name__") else str(type_hint)
    
    def wrap_text(text: str, indent: str, max_length: int) -> str:
        """Wrap text while preserving markdown links and maintaining indentation."""
        words = text.split()
        lines = []
        current_line = []
        current_length = 0

        for word in words:
            word_length = len(word) + (1 if current_line else 0)
            
            if current_line and current_length + word_length > max_length:
                lines.append(" ".join(current_line))
                current_line = [word]
                current_length = len(word)
            else:
                current_line.append(word)
                current_length += word_length
        
        if current_line:
            lines.append(" ".join(current_line))
            
        return f"\n{indent}".join(lines)
    
    # Add the header
    formatted_params.append(f"{base_indent}{header}:")
    
    for param in params:
        # Format parameter name and type
        type_str = get_type_str(param.type_hint) if param.type_hint != Any else ""
        param_str = f"{param_indent}{param.name} (`{type_str}`"
        
        # Add optional tag and default value if parameter is an InputParam and optional
        if isinstance(param, InputParam):
            if not param.required:
                param_str += ", *optional*"
                if param.default is not None:
                    param_str += f", defaults to {param.default}"
        param_str += "):"
            
        # Add description on a new line with additional indentation and wrapping
        if param.description:
            desc = re.sub(
                r'\[(.*?)\]\((https?://[^\s\)]+)\)',
                r'[\1](\2)',
                param.description
            )
            wrapped_desc = wrap_text(desc, desc_indent, max_line_length)
            param_str += f"\n{desc_indent}{wrapped_desc}"
            
        formatted_params.append(param_str)
    
    return "\n\n".join(formatted_params)

# Then update the original functions to use this combined version:
def format_input_params(input_params: List[InputParam], indent_level: int = 4, max_line_length: int = 115) -> str:
    return format_params(input_params, "Inputs", indent_level, max_line_length)

def format_output_params(output_params: List[OutputParam], indent_level: int = 4, max_line_length: int = 115) -> str:
    return format_params(output_params, "Outputs", indent_level, max_line_length)


def format_components(components: List[ComponentSpec], indent_level: int = 4, max_line_length: int = 115, add_empty_lines: bool = True) -> str:
    """Format a list of ComponentSpec objects into a readable string representation.

    Args:
        components: List of ComponentSpec objects to format
        indent_level: Number of spaces to indent each component line (default: 4)
        max_line_length: Maximum length for each line before wrapping (default: 115)
        add_empty_lines: Whether to add empty lines between components (default: True)

    Returns:
        A formatted string representing all components
    """
    if not components:
        return ""
        
    base_indent = " " * indent_level
    component_indent = " " * (indent_level + 4)
    formatted_components = []
    
    # Add the header
    formatted_components.append(f"{base_indent}Components:")
    if add_empty_lines:
        formatted_components.append("")
    
    # Add each component with optional empty lines between them
    for i, component in enumerate(components):
        # Get type name, handling special cases
        type_name = component.type_hint.__name__ if hasattr(component.type_hint, "__name__") else str(component.type_hint)
        
        component_desc = f"{component_indent}{component.name} (`{type_name}`)"
        if component.description:
            component_desc += f": {component.description}"
        if component.default_repo:
            if isinstance(component.default_repo, list) and len(component.default_repo) == 2:
                repo_info = component.default_repo[0]
                subfolder = component.default_repo[1]
                if subfolder:
                    repo_info += f", subfolder={subfolder}"
            else:
                repo_info = component.default_repo
            component_desc += f" [{repo_info}]"
        formatted_components.append(component_desc)
        
        # Add an empty line after each component except the last one
        if add_empty_lines and i < len(components) - 1:
            formatted_components.append("")
    
    return "\n".join(formatted_components)


def format_configs(configs: List[ConfigSpec], indent_level: int = 4, max_line_length: int = 115, add_empty_lines: bool = True) -> str:
    """Format a list of ConfigSpec objects into a readable string representation.

    Args:
        configs: List of ConfigSpec objects to format
        indent_level: Number of spaces to indent each config line (default: 4)
        max_line_length: Maximum length for each line before wrapping (default: 115)
        add_empty_lines: Whether to add empty lines between configs (default: True)

    Returns:
        A formatted string representing all configs
    """
    if not configs:
        return ""
        
    base_indent = " " * indent_level
    config_indent = " " * (indent_level + 4)
    formatted_configs = []
    
    # Add the header
    formatted_configs.append(f"{base_indent}Configs:")
    if add_empty_lines:
        formatted_configs.append("")
    
    # Add each config with optional empty lines between them
    for i, config in enumerate(configs):
        config_desc = f"{config_indent}{config.name} (default: {config.default})"
        if config.description:
            config_desc += f": {config.description}"
        formatted_configs.append(config_desc)
        
        # Add an empty line after each config except the last one
        if add_empty_lines and i < len(configs) - 1:
            formatted_configs.append("")
    
    return "\n".join(formatted_configs)


def make_doc_string(inputs, intermediates_inputs, outputs, description="", class_name=None, expected_components=None, expected_configs=None):
    """
    Generates a formatted documentation string describing the pipeline block's parameters and structure.
    
    Args:
        inputs (List[InputParam]): List of input parameters
        intermediates_inputs (List[InputParam]): List of intermediate input parameters
        outputs (List[OutputParam]): List of output parameters
        description (str, *optional*): Description of the block
        class_name (str, *optional*): Name of the class to include in the documentation
        expected_components (List[ComponentSpec], *optional*): List of expected components
        expected_configs (List[ConfigSpec], *optional*): List of expected configurations
    
    Returns:
        str: A formatted string containing information about components, configs, call parameters, 
            intermediate inputs/outputs, and final outputs.
    """
    output = ""

    # Add class name if provided
    if class_name:
        output += f"class {class_name}\n\n"

    # Add description
    if description:
        desc_lines = description.strip().split('\n')
        aligned_desc = '\n'.join('  ' + line for line in desc_lines)
        output += aligned_desc + "\n\n"

    # Add components section if provided
    if expected_components and len(expected_components) > 0:
        components_str = format_components(expected_components, indent_level=2)
        output += components_str + "\n\n"

    # Add configs section if provided
    if expected_configs and len(expected_configs) > 0:
        configs_str = format_configs(expected_configs, indent_level=2)
        output += configs_str + "\n\n"

    # Add inputs section
    output += format_input_params(inputs + intermediates_inputs, indent_level=2)
    
    # Add outputs section
    output += "\n\n"
    output += format_output_params(outputs, indent_level=2)

    return output



class PipelineBlock:
    
    model_name = None
    
    @property
    def description(self) -> str:
        """Description of the block. Must be implemented by subclasses."""
        raise NotImplementedError("description method must be implemented in subclasses")

    @property
    def expected_components(self) -> List[ComponentSpec]:
        return []
    
    @property
    def expected_configs(self) -> List[ConfigSpec]:
        return []

    
    # YiYi TODO: can we combine inputs and intermediates_inputs? the difference is inputs are immutable
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

    # Adding outputs attributes here for consistency between PipelineBlock/AutoPipelineBlocks/SequentialPipelineBlocks
    @property
    def outputs(self) -> List[OutputParam]:
        return self.intermediates_outputs

    @property
    def required_inputs(self) -> List[str]:
        input_names = []
        for input_param in self.inputs:
            if input_param.required:
                input_names.append(input_param.name)
        return input_names

    @property
    def required_intermediates_inputs(self) -> List[str]:
        input_names = []
        for input_param in self.intermediates_inputs:
            if input_param.required:
                input_names.append(input_param.name)
        return input_names


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


    def get_block_state(self, state: PipelineState) -> dict:
        """Get all inputs and intermediates in one dictionary"""
        data = {}
        
        # Check inputs
        for input_param in self.inputs:
            value = state.get_input(input_param.name)
            if input_param.required and value is None:
                raise ValueError(f"Required input '{input_param.name}' is missing")
            data[input_param.name] = value

        # Check intermediates
        for input_param in self.intermediates_inputs:
            value = state.get_intermediate(input_param.name)
            if input_param.required and value is None:
                raise ValueError(f"Required intermediate input '{input_param.name}' is missing")
            data[input_param.name] = value

        return BlockState(**data)
    
    def add_block_state(self, state: PipelineState, block_state: BlockState):
        for output_param in self.intermediates_outputs:
            if not hasattr(block_state, output_param.name):
                raise ValueError(f"Intermediate output '{output_param.name}' is missing in block state")
            state.add_intermediate(output_param.name, getattr(block_state, output_param.name))


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
            if input_param.name in combined_dict:
                current_param = combined_dict[input_param.name]
                if (current_param.default is not None and 
                    input_param.default is not None and 
                    current_param.default != input_param.default):
                    warnings.warn(
                        f"Multiple different default values found for input '{input_param.name}': "
                        f"{current_param.default} (from block '{value_sources[input_param.name]}') and "
                        f"{input_param.default} (from block '{block_name}'). Using {current_param.default}."
                    )
                if current_param.default is None and input_param.default is not None:
                    combined_dict[input_param.name] = input_param
                    value_sources[input_param.name] = block_name
            else:
                combined_dict[input_param.name] = input_param
                value_sources[input_param.name] = block_name
    
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
            if output_param.name not in combined_dict:
                combined_dict[output_param.name] = output_param
    
    return list(combined_dict.values())


class AutoPipelineBlocks:
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
        first_block = next(iter(self.blocks.values()))
        required_by_all = set(getattr(first_block, "required_inputs", set()))

        # Intersect with required inputs from all other blocks
        for block in list(self.blocks.values())[1:]:
            block_required = set(getattr(block, "required_inputs", set()))
            required_by_all.intersection_update(block_required)

        return list(required_by_all)

    @property
    def required_intermediates_inputs(self) -> List[str]:
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

        # Inputs and outputs section - moved up
        inputs_str = format_inputs_short(self.inputs)
        inputs_str = "  Inputs:\n    " + inputs_str
        
        outputs = [out.name for out in self.outputs]
        intermediates_str = format_intermediates_short(self.intermediates_inputs, self.required_intermediates_inputs, self.intermediates_outputs)
        intermediates_str = (
            "  Intermediates:\n"
            f"{intermediates_str}\n" 
            f"    - final outputs: {', '.join(outputs)}"
        )

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

        return (
            f"{header}\n"
            f"{desc}"
            f"{components_str}\n"
            f"{configs_str}\n"
            f"{inputs_str}\n"
            f"{intermediates_str}\n"
            f"{blocks_str}"
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

class SequentialPipelineBlocks:
    """
    A class that combines multiple pipeline block classes into one. When called, it will call each block in sequence.
    """
    block_classes = []
    block_names = []

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

    @classmethod
    def from_blocks_dict(cls, blocks_dict: Dict[str, Any]) -> "SequentialPipelineBlocks":
        """Creates a SequentialPipelineBlocks instance from a dictionary of blocks.
        
        Args:
            blocks_dict: Dictionary mapping block names to block instances
            
        Returns:
            A new SequentialPipelineBlocks instance
        """
        instance = cls()
        instance.block_classes = [block.__class__ for block in blocks_dict.values()]
        instance.block_names = list(blocks_dict.keys())
        instance.blocks = blocks_dict
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

    @property
    def intermediates_outputs(self) -> List[str]:
        named_outputs = [(name, block.intermediates_outputs) for name, block in self.blocks.items()]
        combined_outputs = combine_outputs(*named_outputs)
        return combined_outputs
    
    @property
    def outputs(self) -> List[str]:
        return next(reversed(self.blocks.values())).intermediates_outputs
    
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
            
            # sequential or PipelineBlock
            if not hasattr(block, 'block_trigger_inputs'):
                if hasattr(block, 'blocks'):
                    # sequential
                    for block_name, block in block.blocks.items():
                        blocks_to_update = fn_recursive_traverse(block, block_name, active_triggers)
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
                    # sequential/auto
                    if hasattr(this_block, 'blocks'):
                        result_blocks.update(fn_recursive_traverse(this_block, block_name, active_triggers))
                    else:
                        # PipelineBlock
                        result_blocks[block_name] = this_block
                        # Add this block's output names to active triggers if defined
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

        # Components section - use format_components with add_empty_lines=False
        expected_components = getattr(self, "expected_components", [])
        components_str = format_components(expected_components, indent_level=2, add_empty_lines=False)
        
        # Configs section - use format_configs with add_empty_lines=False
        expected_configs = getattr(self, "expected_configs", [])
        configs_str = format_configs(expected_configs, indent_level=2, add_empty_lines=False)

        # Inputs and outputs section - moved up
        inputs_str = format_inputs_short(self.inputs)
        inputs_str = "  Inputs:\n    " + inputs_str
        
        outputs = [out.name for out in self.outputs]
        intermediates_str = format_intermediates_short(self.intermediates_inputs, self.required_intermediates_inputs, self.intermediates_outputs)
        intermediates_str = (
            "  Intermediates:\n"
            f"{intermediates_str}\n" 
            f"    - final outputs: {', '.join(outputs)}"
        )

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

        return (
            f"{header}\n"
            f"{desc}"
            f"{components_str}\n"
            f"{configs_str}\n"
            f"{inputs_str}\n"
            f"{intermediates_str}\n"
            f"{blocks_str}"
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

class ModularPipeline(ConfigMixin):
    """
    Base class for all Modular pipelines.

    """

    config_name = "model_index.json"
    _exclude_from_cpu_offload = []

    def __init__(self, block):
        self.pipeline_block = block

        for component_spec in self.expected_components:
            if component_spec.obj is not None:
                setattr(self, component_spec.name, component_spec.obj)
            else:
                setattr(self, component_spec.name, None)
        
        default_configs = {}
        for config_spec in self.expected_configs:
            default_configs[config_spec.name] = config_spec.default
        self.register_to_config(**default_configs)


    @classmethod
    def from_block(cls, block):
        modular_pipeline_class_name = MODULAR_PIPELINE_MAPPING[block.model_name]
        modular_pipeline_class = _get_pipeline_class(cls, class_name=modular_pipeline_class_name)

        return modular_pipeline_class(block)

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
            if not isinstance(model, torch.nn.Module) or name in self._exclude_from_cpu_offload:
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

    
    def get_execution_blocks(self, *trigger_inputs):
        return self.pipeline_block.get_execution_blocks(*trigger_inputs)
    
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
    def expected_components(self):
        return self.pipeline_block.expected_components

    @property
    def expected_configs(self):
        return self.pipeline_block.expected_configs

    @property
    def components(self):
        components = {}
        for component_spec in self.expected_components:
            if hasattr(self, component_spec.name):
                components[component_spec.name] = getattr(self, component_spec.name)
        return components

    # Copied from diffusers.pipelines.pipeline_utils.DiffusionPipeline.progress_bar
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

    # Copied from diffusers.pipelines.pipeline_utils.DiffusionPipeline.set_progress_bar_config
    def set_progress_bar_config(self, **kwargs):
        self._progress_bar_config = kwargs

    def __call__(self, state: PipelineState = None, output: Union[str, List[str]] = None, **kwargs):
        """
        Run one or more blocks in sequence, optionally you can pass a previous pipeline state.
        """
        if state is None:
            state = PipelineState()

        # Make a copy of the input kwargs
        input_params = kwargs.copy()

        default_params = self.default_call_parameters

        # Add inputs to state, using defaults if not provided in the kwargs or the state
        # if same input already in the state, will override it if provided in the kwargs

        intermediates_inputs = [inp.name for inp in self.pipeline_block.intermediates_inputs]
        for name, default in default_params.items():
            if name in input_params:
                if name not in intermediates_inputs:
                    state.add_input(name, input_params.pop(name))
                else:
                    state.add_input(name, input_params[name])
            elif name not in state.inputs:
                state.add_input(name, default)

        for name in intermediates_inputs:
            if name in input_params:
                state.add_intermediate(name, input_params.pop(name))

        # Warn about unexpected inputs
        if len(input_params) > 0:
            logger.warning(f"Unexpected input '{input_params.keys()}' provided. This input will be ignored.")
        # Run the pipeline
        with torch.no_grad():
            try:
                pipeline, state = self.pipeline_block(self, state)
            except Exception:
                error_msg = f"Error in block: ({self.pipeline_block.__class__.__name__}):\n"
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

    def update_states(self, **kwargs):
        """
        Update components and configs after instance creation. Auxiliaries (e.g. image_processor) should be defined for
        each pipeline block, does not need to be updated by users. Logs if existing non-None components are being
        overwritten.

        Args:
            kwargs (dict): Keyword arguments to update the states.
        """

        for component in self.expected_components:
            if component.name in kwargs:
                if hasattr(self, component.name) and getattr(self, component.name) is not None:
                    current_component = getattr(self, component.name)
                    new_component = kwargs[component.name]

                    if not isinstance(new_component, current_component.__class__):
                        logger.info(
                            f"Overwriting existing component '{component.name}' "
                            f"(type: {current_component.__class__.__name__}) "
                            f"with type: {new_component.__class__.__name__})"
                        )
                    elif isinstance(current_component, torch.nn.Module):
                        if id(current_component) != id(new_component):
                            logger.info(
                                f"Overwriting existing component '{component.name}' "
                                f"(type: {type(current_component).__name__}) "
                                f"with new value (type: {type(new_component).__name__})"
                            )

                setattr(self, component.name, kwargs.pop(component.name))

        configs_to_add = {}
        for config in self.expected_configs:
            if config.name in kwargs:
                configs_to_add[config.name] = kwargs.pop(config.name)
        self.register_to_config(**configs_to_add)

    @property
    def default_call_parameters(self) -> Dict[str, Any]:
        params = {}
        for input_param in self.pipeline_block.inputs:
            params[input_param.name] = input_param.default
        return params


    # YiYi TODO: try to unify the to method with the one in DiffusionPipeline
    # Modified from diffusers.pipelines.pipeline_utils.DiffusionPipeline.to
    def to(self, *args, **kwargs):
        r"""
        Performs Pipeline dtype and/or device conversion. A torch.dtype and torch.device are inferred from the
        arguments of `self.to(*args, **kwargs).`

        <Tip>

            If the pipeline already has the correct torch.dtype and torch.device, then it is returned as is. Otherwise,
            the returned pipeline is a copy of self with the desired torch.dtype and torch.device.

        </Tip>


        Here are the ways to call `to`:

        - `to(dtype, silence_dtype_warnings=False) → DiffusionPipeline` to return a pipeline with the specified
          [`dtype`](https://pytorch.org/docs/stable/tensor_attributes.html#torch.dtype)
        - `to(device, silence_dtype_warnings=False) → DiffusionPipeline` to return a pipeline with the specified
          [`device`](https://pytorch.org/docs/stable/tensor_attributes.html#torch.device)
        - `to(device=None, dtype=None, silence_dtype_warnings=False) → DiffusionPipeline` to return a pipeline with the
          specified [`device`](https://pytorch.org/docs/stable/tensor_attributes.html#torch.device) and
          [`dtype`](https://pytorch.org/docs/stable/tensor_attributes.html#torch.dtype)

        Arguments:
            dtype (`torch.dtype`, *optional*):
                Returns a pipeline with the specified
                [`dtype`](https://pytorch.org/docs/stable/tensor_attributes.html#torch.dtype)
            device (`torch.Device`, *optional*):
                Returns a pipeline with the specified
                [`device`](https://pytorch.org/docs/stable/tensor_attributes.html#torch.device)
            silence_dtype_warnings (`str`, *optional*, defaults to `False`):
                Whether to omit warnings if the target `dtype` is not compatible with the target `device`.

        Returns:
            [`DiffusionPipeline`]: The pipeline converted to specified `dtype` and/or `dtype`.
        """
        dtype = kwargs.pop("dtype", None)
        device = kwargs.pop("device", None)
        silence_dtype_warnings = kwargs.pop("silence_dtype_warnings", False)

        dtype_arg = None
        device_arg = None
        if len(args) == 1:
            if isinstance(args[0], torch.dtype):
                dtype_arg = args[0]
            else:
                device_arg = torch.device(args[0]) if args[0] is not None else None
        elif len(args) == 2:
            if isinstance(args[0], torch.dtype):
                raise ValueError(
                    "When passing two arguments, make sure the first corresponds to `device` and the second to `dtype`."
                )
            device_arg = torch.device(args[0]) if args[0] is not None else None
            dtype_arg = args[1]
        elif len(args) > 2:
            raise ValueError("Please make sure to pass at most two arguments (`device` and `dtype`) `.to(...)`")

        if dtype is not None and dtype_arg is not None:
            raise ValueError(
                "You have passed `dtype` both as an argument and as a keyword argument. Please only pass one of the two."
            )

        dtype = dtype or dtype_arg

        if device is not None and device_arg is not None:
            raise ValueError(
                "You have passed `device` both as an argument and as a keyword argument. Please only pass one of the two."
            )

        device = device or device_arg

        # throw warning if pipeline is in "offloaded"-mode but user tries to manually set to GPU.
        def module_is_sequentially_offloaded(module):
            if not is_accelerate_available() or is_accelerate_version("<", "0.14.0"):
                return False

            return hasattr(module, "_hf_hook") and (
                isinstance(module._hf_hook, accelerate.hooks.AlignDevicesHook)
                or hasattr(module._hf_hook, "hooks")
                and isinstance(module._hf_hook.hooks[0], accelerate.hooks.AlignDevicesHook)
            )

        def module_is_offloaded(module):
            if not is_accelerate_available() or is_accelerate_version("<", "0.17.0.dev0"):
                return False

            return hasattr(module, "_hf_hook") and isinstance(module._hf_hook, accelerate.hooks.CpuOffload)

        # .to("cuda") would raise an error if the pipeline is sequentially offloaded, so we raise our own to make it clearer
        pipeline_is_sequentially_offloaded = any(
            module_is_sequentially_offloaded(module) for _, module in self.components.items()
        )
        if pipeline_is_sequentially_offloaded and device and torch.device(device).type == "cuda":
            raise ValueError(
                "It seems like you have activated sequential model offloading by calling `enable_sequential_cpu_offload`, but are now attempting to move the pipeline to GPU. This is not compatible with offloading. Please, move your pipeline `.to('cpu')` or consider removing the move altogether if you use sequential offloading."
            )

        is_pipeline_device_mapped = hasattr(self, "hf_device_map") and self.hf_device_map is not None and len(self.hf_device_map) > 1
        if is_pipeline_device_mapped:
            raise ValueError(
                "It seems like you have activated a device mapping strategy on the pipeline which doesn't allow explicit device placement using `to()`. You can call `reset_device_map()` first and then call `to()`."
            )

        # Display a warning in this case (the operation succeeds but the benefits are lost)
        pipeline_is_offloaded = any(module_is_offloaded(module) for _, module in self.components.items())
        if pipeline_is_offloaded and device and torch.device(device).type == "cuda":
            logger.warning(
                f"It seems like you have activated model offloading by calling `enable_model_cpu_offload`, but are now manually moving the pipeline to GPU. It is strongly recommended against doing so as memory gains from offloading are likely to be lost. Offloading automatically takes care of moving the individual components {', '.join(self.components.keys())} to GPU when needed. To make sure offloading works as expected, you should consider moving the pipeline back to CPU: `pipeline.to('cpu')` or removing the move altogether if you use offloading."
            )

        modules = [m for m in self.components.values() if isinstance(m, torch.nn.Module)]

        is_offloaded = pipeline_is_offloaded or pipeline_is_sequentially_offloaded
        for module in modules:
            is_loaded_in_8bit = hasattr(module, "is_loaded_in_8bit") and module.is_loaded_in_8bit

            if is_loaded_in_8bit and dtype is not None:
                logger.warning(
                    f"The module '{module.__class__.__name__}' has been loaded in 8bit and conversion to {dtype} is not yet supported. Module is still in 8bit precision."
                )

            if is_loaded_in_8bit and device is not None:
                logger.warning(
                    f"The module '{module.__class__.__name__}' has been loaded in 8bit and moving it to {dtype} via `.to()` is not yet supported. Module is still on {module.device}."
                )
            else:
                module.to(device, dtype)

            if (
                module.dtype == torch.float16
                and str(device) in ["cpu"]
                and not silence_dtype_warnings
                and not is_offloaded
            ):
                logger.warning(
                    "Pipelines loaded with `dtype=torch.float16` cannot run with `cpu` device. It"
                    " is not recommended to move them to `cpu` as running them will fail. Please make"
                    " sure to use an accelerator to run the pipeline in inference, due to the lack of"
                    " support for`float16` operations on this device in PyTorch. Please, remove the"
                    " `torch_dtype=torch.float16` argument, or use another device for inference."
                )
        return self
