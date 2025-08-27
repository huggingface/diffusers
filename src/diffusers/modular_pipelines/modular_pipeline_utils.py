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

import inspect
import re
from collections import OrderedDict
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Literal, Optional, Type, Union

import torch

from ..configuration_utils import ConfigMixin, FrozenDict
from ..utils import is_torch_available, logging


if is_torch_available():
    pass

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class InsertableDict(OrderedDict):
    def insert(self, key, value, index):
        items = list(self.items())

        # Remove key if it already exists to avoid duplicates
        items = [(k, v) for k, v in items if k != key]

        # Insert at the specified index
        items.insert(index, (key, value))

        # Clear and update self
        self.clear()
        self.update(items)

        # Return self for method chaining
        return self

    def __repr__(self):
        if not self:
            return "InsertableDict()"

        items = []
        for i, (key, value) in enumerate(self.items()):
            if isinstance(value, type):
                # For classes, show class name and <class ...>
                obj_repr = f"<class '{value.__module__}.{value.__name__}'>"
            else:
                # For objects (instances) and other types, show class name and module
                obj_repr = f"<obj '{value.__class__.__module__}.{value.__class__.__name__}'>"
            items.append(f"{i}: ({repr(key)}, {obj_repr})")

        return "InsertableDict([\n  " + ",\n  ".join(items) + "\n])"


# YiYi TODO:
# 1. validate the dataclass fields
# 2. improve the docstring and potentially add a validator for load methods, make sure they are valid inputs to pass to from_pretrained()
@dataclass
class ComponentSpec:
    """Specification for a pipeline component.

    A component can be created in two ways:
    1. From scratch using __init__ with a config dict
    2. using `from_pretrained`

    Attributes:
        name: Name of the component
        type_hint: Type of the component (e.g. UNet2DConditionModel)
        description: Optional description of the component
        config: Optional config dict for __init__ creation
        repo: Optional repo path for from_pretrained creation
        subfolder: Optional subfolder in repo
        variant: Optional variant in repo
        revision: Optional revision in repo
        default_creation_method: Preferred creation method - "from_config" or "from_pretrained"
    """

    name: Optional[str] = None
    type_hint: Optional[Type] = None
    description: Optional[str] = None
    config: Optional[FrozenDict] = None
    # YiYi Notes: should we change it to pretrained_model_name_or_path for consistency? a bit long for a field name
    repo: Optional[Union[str, List[str]]] = field(default=None, metadata={"loading": True})
    subfolder: Optional[str] = field(default="", metadata={"loading": True})
    variant: Optional[str] = field(default=None, metadata={"loading": True})
    revision: Optional[str] = field(default=None, metadata={"loading": True})
    default_creation_method: Literal["from_config", "from_pretrained"] = "from_pretrained"

    def __hash__(self):
        """Make ComponentSpec hashable, using load_id as the hash value."""
        return hash((self.name, self.load_id, self.default_creation_method))

    def __eq__(self, other):
        """Compare ComponentSpec objects based on name and load_id."""
        if not isinstance(other, ComponentSpec):
            return False
        return (
            self.name == other.name
            and self.load_id == other.load_id
            and self.default_creation_method == other.default_creation_method
        )

    @classmethod
    def from_component(cls, name: str, component: Any) -> Any:
        """Create a ComponentSpec from a Component.

        Currently supports:
        - Components created with `ComponentSpec.load()` method
        - Components that are ConfigMixin subclasses but not nn.Modules (e.g. schedulers, guiders)

        Args:
            name: Name of the component
            component: Component object to create spec from

        Returns:
            ComponentSpec object

        Raises:
            ValueError: If component is not supported (e.g. nn.Module without load_id, non-ConfigMixin)
        """

        # Check if component was created with ComponentSpec.load()
        if hasattr(component, "_diffusers_load_id") and component._diffusers_load_id != "null":
            # component has a usable load_id -> from_pretrained, no warning needed
            default_creation_method = "from_pretrained"
        else:
            # Component doesn't have a usable load_id, check if it's a nn.Module
            if isinstance(component, torch.nn.Module):
                raise ValueError(
                    "Cannot create ComponentSpec from a nn.Module that was not created with `ComponentSpec.load()` method."
                )
            # ConfigMixin objects without weights (e.g. scheduler & guider) can be recreated with from_config
            elif isinstance(component, ConfigMixin):
                # warn if component was not created with `ComponentSpec`
                if not hasattr(component, "_diffusers_load_id"):
                    logger.warning(
                        "Component was not created using `ComponentSpec`, defaulting to `from_config` creation method"
                    )
                default_creation_method = "from_config"
            else:
                # Not a ConfigMixin and not created with `ComponentSpec.load()` method -> throw error
                raise ValueError(
                    f"Cannot create ComponentSpec from {name}({component.__class__.__name__}). Currently ComponentSpec.from_component() only supports: "
                    f" - components created with `ComponentSpec.load()` method"
                    f" - components that are a subclass of ConfigMixin but not a nn.Module (e.g. guider, scheduler)."
                )

        type_hint = component.__class__

        if isinstance(component, ConfigMixin) and default_creation_method == "from_config":
            config = component.config
        else:
            config = None
        if hasattr(component, "_diffusers_load_id") and component._diffusers_load_id != "null":
            load_spec = cls.decode_load_id(component._diffusers_load_id)
        else:
            load_spec = {}

        return cls(
            name=name, type_hint=type_hint, config=config, default_creation_method=default_creation_method, **load_spec
        )

    @classmethod
    def loading_fields(cls) -> List[str]:
        """
        Return the names of all loading‐related fields (i.e. those whose field.metadata["loading"] is True).
        """
        return [f.name for f in fields(cls) if f.metadata.get("loading", False)]

    @property
    def load_id(self) -> str:
        """
        Unique identifier for this spec's pretrained load, composed of repo|subfolder|variant|revision (no empty
        segments).
        """
        if self.default_creation_method == "from_config":
            return "null"
        parts = [getattr(self, k) for k in self.loading_fields()]
        parts = ["null" if p is None else p for p in parts]
        return "|".join(p for p in parts if p)

    @classmethod
    def decode_load_id(cls, load_id: str) -> Dict[str, Optional[str]]:
        """
        Decode a load_id string back into a dictionary of loading fields and values.

        Args:
            load_id: The load_id string to decode, format: "repo|subfolder|variant|revision"
                     where None values are represented as "null"

        Returns:
            Dict mapping loading field names to their values. e.g. {
                "repo": "path/to/repo", "subfolder": "subfolder", "variant": "variant", "revision": "revision"
            } If a segment value is "null", it's replaced with None. Returns None if load_id is "null" (indicating
            component not created with `load` method).
        """

        # Get all loading fields in order
        loading_fields = cls.loading_fields()
        result = dict.fromkeys(loading_fields)

        if load_id == "null":
            return result

        # Split the load_id
        parts = load_id.split("|")

        # Map parts to loading fields by position
        for i, part in enumerate(parts):
            if i < len(loading_fields):
                # Convert "null" string back to None
                result[loading_fields[i]] = None if part == "null" else part

        return result

    # YiYi TODO: I think we should only support ConfigMixin for this method (after we make guider and image_processors config mixin)
    # otherwise we cannot do spec -> spec.create() -> component -> ComponentSpec.from_component(component)
    # the config info is lost in the process
    # remove error check in from_component spec and ModularPipeline.update_components() if we remove support for non configmixin in `create()` method
    def create(self, config: Optional[Union[FrozenDict, Dict[str, Any]]] = None, **kwargs) -> Any:
        """Create component using from_config with config."""

        if self.type_hint is None or not isinstance(self.type_hint, type):
            raise ValueError("`type_hint` is required when using from_config creation method.")

        config = config or self.config or {}

        if issubclass(self.type_hint, ConfigMixin):
            component = self.type_hint.from_config(config, **kwargs)
        else:
            signature_params = inspect.signature(self.type_hint.__init__).parameters
            init_kwargs = {}
            for k, v in config.items():
                if k in signature_params:
                    init_kwargs[k] = v
            for k, v in kwargs.items():
                if k in signature_params:
                    init_kwargs[k] = v
            component = self.type_hint(**init_kwargs)

        component._diffusers_load_id = "null"
        if hasattr(component, "config"):
            self.config = component.config

        return component

    # YiYi TODO: add guard for type of model, if it is supported by from_pretrained
    def load(self, **kwargs) -> Any:
        """Load component using from_pretrained."""

        # select loading fields from kwargs passed from user: e.g. repo, subfolder, variant, revision, note the list could change
        passed_loading_kwargs = {key: kwargs.pop(key) for key in self.loading_fields() if key in kwargs}
        # merge loading field value in the spec with user passed values to create load_kwargs
        load_kwargs = {key: passed_loading_kwargs.get(key, getattr(self, key)) for key in self.loading_fields()}
        # repo is a required argument for from_pretrained, a.k.a. pretrained_model_name_or_path
        repo = load_kwargs.pop("repo", None)
        if repo is None:
            raise ValueError(
                "`repo` info is required when using `load` method (you can directly set it in `repo` field of the ComponentSpec or pass it as an argument)"
            )

        if self.type_hint is None:
            try:
                from diffusers import AutoModel

                component = AutoModel.from_pretrained(repo, **load_kwargs, **kwargs)
            except Exception as e:
                raise ValueError(f"Unable to load {self.name} without `type_hint`: {e}")
            # update type_hint if AutoModel load successfully
            self.type_hint = component.__class__
        else:
            try:
                component = self.type_hint.from_pretrained(repo, **load_kwargs, **kwargs)
            except Exception as e:
                raise ValueError(f"Unable to load {self.name} using load method: {e}")

        self.repo = repo
        for k, v in load_kwargs.items():
            setattr(self, k, v)
        component._diffusers_load_id = self.load_id

        return component


@dataclass
class ConfigSpec:
    """Specification for a pipeline configuration parameter."""

    name: str
    default: Any
    description: Optional[str] = None


# YiYi Notes: both inputs and intermediate_inputs are InputParam objects
# however some fields are not relevant for intermediate_inputs
# e.g. unlike inputs, required only used in docstring for intermediate_inputs, we do not check if a required intermediate inputs is passed
# default is not used for intermediate_inputs, we only use default from inputs, so it is ignored if it is set for intermediate_inputs
# -> should we use different class for inputs and intermediate_inputs?
@dataclass
class InputParam:
    """Specification for an input parameter."""

    name: str = None
    type_hint: Any = None
    default: Any = None
    required: bool = False
    description: str = ""
    kwargs_type: str = None  # YiYi Notes: remove this feature (maybe)

    def __repr__(self):
        return f"<{self.name}: {'required' if self.required else 'optional'}, default={self.default}>"


@dataclass
class OutputParam:
    """Specification for an output parameter."""

    name: str
    type_hint: Any = None
    description: str = ""
    kwargs_type: str = None  # YiYi notes: remove this feature (maybe)

    def __repr__(self):
        return (
            f"<{self.name}: {self.type_hint.__name__ if hasattr(self.type_hint, '__name__') else str(self.type_hint)}>"
        )


def format_inputs_short(inputs):
    """
    Format input parameters into a string representation, with required params first followed by optional ones.

    Args:
        inputs: List of input parameters with 'required' and 'name' attributes, and 'default' for optional params

    Returns:
        str: Formatted string of input parameters

    Example:
        >>> inputs = [ ... InputParam(name="prompt", required=True), ... InputParam(name="image", required=True), ...
        InputParam(name="guidance_scale", required=False, default=7.5), ... InputParam(name="num_inference_steps",
        required=False, default=50) ... ] >>> format_inputs_short(inputs) 'prompt, image, guidance_scale=7.5,
        num_inference_steps=50'
    """
    required_inputs = [param for param in inputs if param.required]
    optional_inputs = [param for param in inputs if not param.required]

    required_str = ", ".join(param.name for param in required_inputs)
    optional_str = ", ".join(f"{param.name}={param.default}" for param in optional_inputs)

    inputs_str = required_str
    if optional_str:
        inputs_str = f"{inputs_str}, {optional_str}" if required_str else optional_str

    return inputs_str


def format_intermediates_short(intermediate_inputs, required_intermediate_inputs, intermediate_outputs):
    """
    Formats intermediate inputs and outputs of a block into a string representation.

    Args:
        intermediate_inputs: List of intermediate input parameters
        required_intermediate_inputs: List of required intermediate input names
        intermediate_outputs: List of intermediate output parameters

    Returns:
        str: Formatted string like:
            Intermediates:
                - inputs: Required(latents), dtype
                - modified: latents # variables that appear in both inputs and outputs
                - outputs: images # new outputs only
    """
    # Handle inputs
    input_parts = []
    for inp in intermediate_inputs:
        if inp.name in required_intermediate_inputs:
            input_parts.append(f"Required({inp.name})")
        else:
            if inp.name is None and inp.kwargs_type is not None:
                inp_name = "*_" + inp.kwargs_type
            else:
                inp_name = inp.name
            input_parts.append(inp_name)

    # Handle modified variables (appear in both inputs and outputs)
    inputs_set = {inp.name for inp in intermediate_inputs}
    modified_parts = []
    new_output_parts = []

    for out in intermediate_outputs:
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
        # YiYi Notes: remove this line if we remove kwargs_type
        name = f"**{param.kwargs_type}" if param.name is None and param.kwargs_type is not None else param.name
        param_str = f"{param_indent}{name} (`{type_str}`"

        # Add optional tag and default value if parameter is an InputParam and optional
        if hasattr(param, "required"):
            if not param.required:
                param_str += ", *optional*"
                if param.default is not None:
                    param_str += f", defaults to {param.default}"
        param_str += "):"

        # Add description on a new line with additional indentation and wrapping
        if param.description:
            desc = re.sub(r"\[(.*?)\]\((https?://[^\s\)]+)\)", r"[\1](\2)", param.description)
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
        type_name = (
            component.type_hint.__name__ if hasattr(component.type_hint, "__name__") else str(component.type_hint)
        )

        component_desc = f"{component_indent}{component.name} (`{type_name}`)"
        if component.description:
            component_desc += f": {component.description}"

        # Get the loading fields dynamically
        loading_field_values = []
        for field_name in component.loading_fields():
            field_value = getattr(component, field_name)
            if field_value is not None:
                loading_field_values.append(f"{field_name}={field_value}")

        # Add loading field information if available
        if loading_field_values:
            component_desc += f" [{', '.join(loading_field_values)}]"

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


def make_doc_string(
    inputs,
    outputs,
    description="",
    class_name=None,
    expected_components=None,
    expected_configs=None,
):
    """
    Generates a formatted documentation string describing the pipeline block's parameters and structure.

    Args:
        inputs: List of input parameters
        intermediate_inputs: List of intermediate input parameters
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
        desc_lines = description.strip().split("\n")
        aligned_desc = "\n".join("  " + line for line in desc_lines)
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
    output += format_input_params(inputs, indent_level=2)

    # Add outputs section
    output += "\n\n"
    output += format_output_params(outputs, indent_level=2)

    return output
