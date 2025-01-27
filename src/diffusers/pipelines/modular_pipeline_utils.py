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
from typing import Any, Dict, List, Tuple, Union


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



@dataclass
class PipelineState:
    """
    [`PipelineState`] stores the state of a pipeline. It is used to pass data between pipeline blocks.
    """

    inputs: Dict[str, Any] = field(default_factory=dict)
    intermediates: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)

    def add_input(self, key: str, value: Any):
        self.inputs[key] = value

    def add_intermediate(self, key: str, value: Any):
        self.intermediates[key] = value

    def add_output(self, key: str, value: Any):
        self.outputs[key] = value

    def get_input(self, key: str, default: Any = None) -> Any:
        return self.inputs.get(key, default)

    def get_intermediate(self, key: str, default: Any = None) -> Any:
        return self.intermediates.get(key, default)

    def get_output(self, key: str, default: Any = None) -> Any:
        if key in self.outputs:
            return self.outputs[key]
        elif key in self.intermediates:
            return self.intermediates[key]
        else:
            return default

    def to_dict(self) -> Dict[str, Any]:
        return {**self.__dict__, "inputs": self.inputs, "intermediates": self.intermediates, "outputs": self.outputs}

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
        outputs = "\n".join(f"    {k}: {format_value(v)}" for k, v in self.outputs.items())

        return (
            f"PipelineState(\n"
            f"  inputs={{\n{inputs}\n  }},\n"
            f"  intermediates={{\n{intermediates}\n  }},\n"
            f"  outputs={{\n{outputs}\n  }}\n"
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
class InputParam:
    name: str
    default: Any = None
    required: bool = False
    description: str = ""
    type_hint: Any = Any

    def __repr__(self):
        return f"<{self.name}: {'required' if self.required else 'optional'}, default={self.default}>"

@dataclass 
class OutputParam:
    name: str
    description: str = ""
    type_hint: Any = Any

    def __repr__(self):
        return f"<{self.name}: {self.type_hint.__name__ if hasattr(self.type_hint, '__name__') else str(self.type_hint)}>"



def format_inputs_short(inputs):
    """
    Format input parameters into a string representation, with required params first followed by optional ones.
    
    Args:
        inputs: List of input parameters with 'required' and 'name' attributes, and 'default' for optional params
        
    Returns:
        str: Formatted string of input parameters
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
        block: Pipeline block with potential intermediates
    
    Returns:
        str: Formatted string like "input1, Required(input2) -> output1, output2"
    """
    # Handle inputs
    input_parts = []

    for inp in intermediates_inputs:
        parts = []
        # Check if input is required
        if inp.name in required_intermediates_inputs:
            parts.append("Required")
        
        # Get base name or modified name
        name = inp.name
        if name in {out.name for out in intermediates_outputs}:
            name = f"*{name}"
        
        # Combine Required() wrapper with possibly starred name
        if parts:
            input_parts.append(f"Required({name})")
        else:
            input_parts.append(name)
    
    # Handle outputs
    output_parts = []
    outputs = [out.name for out in intermediates_outputs]
    # Only show new outputs if we have inputs
    inputs_set = {inp.name for inp in intermediates_inputs}
    outputs = [out for out in outputs if out not in inputs_set]
    output_parts.extend(outputs)
    
    # Combine with arrow notation if both inputs and outputs exist
    if output_parts:
        return f"-> {', '.join(output_parts)}" if not input_parts else f"{', '.join(input_parts)} -> {', '.join(output_parts)}"
    elif input_parts:
        return ', '.join(input_parts)
    return ""


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
    return format_params(input_params, "Args", indent_level, max_line_length)


def format_output_params(output_params: List[OutputParam], indent_level: int = 4, max_line_length: int = 115) -> str:
    return format_params(output_params, "Returns", indent_level, max_line_length)


def make_doc_string(inputs, intermediates_inputs, intermediates_outputs, final_intermediates_outputs=None, description=""):
    """
    Generates a formatted documentation string describing the pipeline block's parameters and structure.
    
    Returns:
        str: A formatted string containing information about call parameters, intermediate inputs/outputs,
            and final intermediate outputs.
    """
    output = ""

    if description:
        desc_lines = description.strip().split('\n')
        aligned_desc = '\n'.join('  ' + line for line in desc_lines)
        output += aligned_desc + "\n\n"

    output += format_input_params(inputs + intermediates_inputs, indent_level=2)
    
    # YiYi TODO: refactor to remove this and `outputs` attribute instead
    if final_intermediates_outputs:
        output += "\n\n"
        output += format_output_params(final_intermediates_outputs, indent_level=2)

        if intermediates_outputs:
            output += "\n\n------------------------\n"
            intermediates_str = format_params(intermediates_outputs, "Intermediates Outputs", indent_level=2)
            output += intermediates_str
    
    elif intermediates_outputs:
        output +="\n\n"
        output += format_output_params(intermediates_outputs, indent_level=2)


    return output


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


