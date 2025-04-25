# Copyright 2023 The HuggingFace Team. All rights reserved.
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

import re
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple, Type, Union

from ..utils.import_utils import is_torch_available
from ..configuration_utils import FrozenDict

if is_torch_available():
    import torch


@dataclass
class ComponentSpec:
    """Specification for a pipeline component."""
    name: str
    type_hint: Type # YiYi Notes: change to component_type?
    description: Optional[str] = None
    config: Optional[FrozenDict[str, Any]] = None  # you can specific default config to create a default component if it is a stateless class like scheduler, guider or image processor
    repo: Optional[Union[str, List[str]]] = None
    subfolder: Optional[str] = None
    
    def create(self, **kwargs) -> Any:
        """
        Create the component based on the config and additional kwargs.
        
        Args:
            **kwargs: Additional arguments to pass to the component's __init__ method
            
        Returns:
            The created component
        """
        if self.config is not None:
            init_kwargs = self.config
        else:
            init_kwargs = {}
        return self.type_hint(**init_kwargs, **kwargs)
    
    def load(self, **kwargs) -> Any:
        return self.to_load_spec().load(**kwargs)
    
    def to_load_spec(self) -> "ComponentLoadSpec":
        """Convert to a ComponentLoadSpec for storage in ComponentsManager."""
        return ComponentLoadSpec.from_component_spec(self)

@dataclass
class ComponentLoadSpec:
    type_hint: type
    repo: Optional[str] = None
    subfolder: Optional[str] = None

    def load(self, **kwargs) -> Any:
        """Load the component from the repository."""
        repo = kwargs.pop("repo", self.repo)
        subfolder = kwargs.pop("subfolder", self.subfolder)

        return self.type_hint.from_pretrained(repo, subfolder=subfolder, **kwargs)
        
    
    @classmethod
    def from_component_spec(cls, component_spec: ComponentSpec):
        return cls(type_hint=component_spec.type_hint, repo=component_spec.repo, subfolder=component_spec.subfolder)


@dataclass 
class ConfigSpec:
    """Specification for a pipeline configuration parameter."""
    name: str
    default: Any
    description: Optional[str] = None
@dataclass
class InputParam:
    """Specification for an input parameter."""
    name: str
    type_hint: Any = None
    default: Any = None
    required: bool = False
    description: str = ""

    def __repr__(self):
        return f"<{self.name}: {'required' if self.required else 'optional'}, default={self.default}>"


@dataclass 
class OutputParam:
    """Specification for an output parameter."""
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


def format_intermediates_short(intermediates_inputs, required_intermediates_inputs, intermediates_outputs):
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


def format_params(params, header="Args", indent_level=4, max_line_length=115):
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
    
    def wrap_text(text, indent, max_length):
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
        if hasattr(param, "required"):
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


def format_input_params(input_params, indent_level=4, max_line_length=115):
    """Format a list of InputParam objects into a readable string representation.

    Args:
        input_params: List of InputParam objects to format
        indent_level: Number of spaces to indent each parameter line (default: 4)
        max_line_length: Maximum length for each line before wrapping (default: 115)

    Returns:
        A formatted string representing all input parameters
    """
    return format_params(input_params, "Inputs", indent_level, max_line_length)


def format_output_params(output_params, indent_level=4, max_line_length=115):
    """Format a list of OutputParam objects into a readable string representation.

    Args:
        output_params: List of OutputParam objects to format
        indent_level: Number of spaces to indent each parameter line (default: 4)
        max_line_length: Maximum length for each line before wrapping (default: 115)

    Returns:
        A formatted string representing all output parameters
    """
    return format_params(output_params, "Outputs", indent_level, max_line_length)


def format_components(components, indent_level=4, max_line_length=115, add_empty_lines=True):
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


def format_configs(configs, indent_level=4, max_line_length=115, add_empty_lines=True):
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
        inputs: List of input parameters
        intermediates_inputs: List of intermediate input parameters
        outputs: List of output parameters
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