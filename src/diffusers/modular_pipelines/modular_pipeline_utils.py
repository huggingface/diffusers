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
import warnings
from collections import OrderedDict
from dataclasses import dataclass, field
from types import UnionType
from typing import Any, Literal, Type, Union, get_args, get_origin

import PIL.Image
import torch

from ..configuration_utils import ConfigMixin, FrozenDict
from ..loaders.single_file_utils import _is_single_file_path_or_url
from ..utils import DIFFUSERS_LOAD_ID_FIELDS, is_torch_available, logging


if is_torch_available():
    pass

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# Template for modular pipeline model card description with placeholders
MODULAR_MODEL_CARD_TEMPLATE = """{model_description}

## Example Usage

[TODO]

## Pipeline Architecture

This modular pipeline is composed of the following blocks:

{blocks_description} {trigger_inputs_section}

## Model Components

{components_description} {configs_section}

## Input/Output Specification

### Inputs {inputs_description}

### Outputs {outputs_description}
"""


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
        pretrained_model_name_or_path: Optional pretrained_model_name_or_path path for from_pretrained creation
        subfolder: Optional subfolder in pretrained_model_name_or_path
        variant: Optional variant in pretrained_model_name_or_path
        revision: Optional revision in pretrained_model_name_or_path
        default_creation_method: Preferred creation method - "from_config" or "from_pretrained"
    """

    name: str | None = None
    type_hint: Type | None = None
    description: str | None = None
    config: FrozenDict | None = None
    pretrained_model_name_or_path: str | list[str] | None = field(default=None, metadata={"loading": True})
    subfolder: str | None = field(default="", metadata={"loading": True})
    variant: str | None = field(default=None, metadata={"loading": True})
    revision: str | None = field(default=None, metadata={"loading": True})
    default_creation_method: Literal["from_config", "from_pretrained"] = "from_pretrained"

    # Deprecated
    repo: str | list[str] | None = field(default=None, metadata={"loading": False})

    def __post_init__(self):
        repo_value = self.repo
        if repo_value is not None and self.pretrained_model_name_or_path is None:
            object.__setattr__(self, "pretrained_model_name_or_path", repo_value)

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
    def loading_fields(cls) -> list[str]:
        """
        Return the names of all loading‐related fields (i.e. those whose field.metadata["loading"] is True).
        """
        return DIFFUSERS_LOAD_ID_FIELDS.copy()

    @property
    def load_id(self) -> str:
        """
        Unique identifier for this spec's pretrained load, composed of
        pretrained_model_name_or_path|subfolder|variant|revision (no empty segments).
        """
        if self.default_creation_method == "from_config":
            return "null"
        parts = [getattr(self, k) for k in self.loading_fields()]
        parts = ["null" if p is None else p for p in parts]
        return "|".join(parts)

    @classmethod
    def decode_load_id(cls, load_id: str) -> dict[str, str | None]:
        """
        Decode a load_id string back into a dictionary of loading fields and values.

        Args:
            load_id: The load_id string to decode, format: "pretrained_model_name_or_path|subfolder|variant|revision"
                     where None values are represented as "null"

        Returns:
            Dict mapping loading field names to their values. e.g. {
                "pretrained_model_name_or_path": "path/to/repo", "subfolder": "subfolder", "variant": "variant",
                "revision": "revision"
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
    def create(self, config: FrozenDict | dict[str, Any] | None = None, **kwargs) -> Any:
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
        # select loading fields from kwargs passed from user: e.g. pretrained_model_name_or_path, subfolder, variant, revision, note the list could change
        passed_loading_kwargs = {key: kwargs.pop(key) for key in self.loading_fields() if key in kwargs}
        # merge loading field value in the spec with user passed values to create load_kwargs
        load_kwargs = {key: passed_loading_kwargs.get(key, getattr(self, key)) for key in self.loading_fields()}

        pretrained_model_name_or_path = load_kwargs.pop("pretrained_model_name_or_path", None)
        if pretrained_model_name_or_path is None:
            raise ValueError(
                "`pretrained_model_name_or_path` info is required when using `load` method (you can directly set it in `pretrained_model_name_or_path` field of the ComponentSpec or pass it as an argument)"
            )
        is_single_file = _is_single_file_path_or_url(pretrained_model_name_or_path)
        if is_single_file and self.type_hint is None:
            raise ValueError(
                f"`type_hint` is required when loading a single file model but is missing for component: {self.name}"
            )

        if self.type_hint is None:
            try:
                from diffusers import AutoModel

                component = AutoModel.from_pretrained(pretrained_model_name_or_path, **load_kwargs, **kwargs)
            except Exception as e:
                raise ValueError(f"Unable to load {self.name} without `type_hint`: {e}")
            # update type_hint if AutoModel load successfully
            self.type_hint = component.__class__
        else:
            # determine load method
            load_method = (
                getattr(self.type_hint, "from_single_file")
                if is_single_file
                else getattr(self.type_hint, "from_pretrained")
            )

            try:
                component = load_method(pretrained_model_name_or_path, **load_kwargs, **kwargs)
            except Exception as e:
                raise ValueError(f"Unable to load {self.name} using load method: {e}")

        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        for k, v in load_kwargs.items():
            setattr(self, k, v)
        component._diffusers_load_id = self.load_id

        return component


@dataclass
class ConfigSpec:
    """Specification for a pipeline configuration parameter."""

    name: str
    default: Any
    description: str | None = None


# ======================================================
# InputParam and OutputParam templates
# ======================================================

INPUT_PARAM_TEMPLATES = {
    "prompt": {
        "type_hint": str,
        "required": True,
        "description": "The prompt or prompts to guide image generation.",
    },
    "negative_prompt": {
        "type_hint": str,
        "description": "The prompt or prompts not to guide the image generation.",
    },
    "max_sequence_length": {
        "type_hint": int,
        "default": 512,
        "description": "Maximum sequence length for prompt encoding.",
    },
    "height": {
        "type_hint": int,
        "description": "The height in pixels of the generated image.",
    },
    "width": {
        "type_hint": int,
        "description": "The width in pixels of the generated image.",
    },
    "num_inference_steps": {
        "type_hint": int,
        "default": 50,
        "description": "The number of denoising steps.",
    },
    "num_images_per_prompt": {
        "type_hint": int,
        "default": 1,
        "description": "The number of images to generate per prompt.",
    },
    "generator": {
        "type_hint": torch.Generator,
        "description": "Torch generator for deterministic generation.",
    },
    "sigmas": {
        "type_hint": list[float],
        "description": "Custom sigmas for the denoising process.",
    },
    "strength": {
        "type_hint": float,
        "default": 0.9,
        "description": "Strength for img2img/inpainting.",
    },
    "image": {
        "type_hint": PIL.Image.Image | list[PIL.Image.Image],
        "required": True,
        "description": "Reference image(s) for denoising. Can be a single image or list of images.",
    },
    "latents": {
        "type_hint": torch.Tensor,
        "description": "Pre-generated noisy latents for image generation.",
    },
    "timesteps": {
        "type_hint": torch.Tensor,
        "description": "Timesteps for the denoising process.",
    },
    "output_type": {
        "type_hint": str,
        "default": "pil",
        "description": "Output format: 'pil', 'np', 'pt'.",
    },
    "attention_kwargs": {
        "type_hint": dict[str, Any],
        "description": "Additional kwargs for attention processors.",
    },
    "denoiser_input_fields": {
        "name": None,
        "kwargs_type": "denoiser_input_fields",
        "description": "conditional model inputs for the denoiser: e.g. prompt_embeds, negative_prompt_embeds, etc.",
    },
    # inpainting
    "mask_image": {
        "type_hint": PIL.Image.Image,
        "required": True,
        "description": "Mask image for inpainting.",
    },
    "padding_mask_crop": {
        "type_hint": int,
        "description": "Padding for mask cropping in inpainting.",
    },
    # controlnet
    "control_image": {
        "type_hint": PIL.Image.Image,
        "required": True,
        "description": "Control image for ControlNet conditioning.",
    },
    "control_guidance_start": {
        "type_hint": float,
        "default": 0.0,
        "description": "When to start applying ControlNet.",
    },
    "control_guidance_end": {
        "type_hint": float,
        "default": 1.0,
        "description": "When to stop applying ControlNet.",
    },
    "controlnet_conditioning_scale": {
        "type_hint": float,
        "default": 1.0,
        "description": "Scale for ControlNet conditioning.",
    },
    "layers": {
        "type_hint": int,
        "default": 4,
        "description": "Number of layers to extract from the image",
    },
    # common intermediate inputs
    "prompt_embeds": {
        "type_hint": torch.Tensor,
        "required": True,
        "description": "text embeddings used to guide the image generation. Can be generated from text_encoder step.",
    },
    "prompt_embeds_mask": {
        "type_hint": torch.Tensor,
        "required": True,
        "description": "mask for the text embeddings. Can be generated from text_encoder step.",
    },
    "negative_prompt_embeds": {
        "type_hint": torch.Tensor,
        "description": "negative text embeddings used to guide the image generation. Can be generated from text_encoder step.",
    },
    "negative_prompt_embeds_mask": {
        "type_hint": torch.Tensor,
        "description": "mask for the negative text embeddings. Can be generated from text_encoder step.",
    },
    "image_latents": {
        "type_hint": torch.Tensor,
        "required": True,
        "description": "image latents used to guide the image generation. Can be generated from vae_encoder step.",
    },
    "batch_size": {
        "type_hint": int,
        "default": 1,
        "description": "Number of prompts, the final batch size of model inputs should be batch_size * num_images_per_prompt. Can be generated in input step.",
    },
    "dtype": {
        "type_hint": torch.dtype,
        "default": torch.float32,
        "description": "The dtype of the model inputs, can be generated in input step.",
    },
}

OUTPUT_PARAM_TEMPLATES = {
    "images": {
        "type_hint": list[PIL.Image.Image],
        "description": "Generated images.",
    },
    "videos": {
        "type_hint": list[PIL.Image.Image],
        "description": "The generated videos.",
    },
    "latents": {
        "type_hint": torch.Tensor,
        "description": "Denoised latents.",
    },
    # intermediate outputs
    "prompt_embeds": {
        "type_hint": torch.Tensor,
        "kwargs_type": "denoiser_input_fields",
        "description": "The prompt embeddings.",
    },
    "prompt_embeds_mask": {
        "type_hint": torch.Tensor,
        "kwargs_type": "denoiser_input_fields",
        "description": "The encoder attention mask.",
    },
    "negative_prompt_embeds": {
        "type_hint": torch.Tensor,
        "kwargs_type": "denoiser_input_fields",
        "description": "The negative prompt embeddings.",
    },
    "negative_prompt_embeds_mask": {
        "type_hint": torch.Tensor,
        "kwargs_type": "denoiser_input_fields",
        "description": "The negative prompt embeddings mask.",
    },
    "image_latents": {
        "type_hint": torch.Tensor,
        "description": "The latent representation of the input image.",
    },
}


@dataclass
class InputParam:
    """Specification for an input parameter."""

    name: str = None
    type_hint: Any = None
    default: Any = None
    required: bool = False
    description: str = ""
    kwargs_type: str = None
    metadata: dict[str, Any] = None

    def __repr__(self):
        return f"<{self.name}: {'required' if self.required else 'optional'}, default={self.default}>"

    @classmethod
    def template(cls, template_name: str, note: str = None, **overrides) -> "InputParam":
        """Get template for name if exists, otherwise raise ValueError."""
        if template_name not in INPUT_PARAM_TEMPLATES:
            raise ValueError(f"InputParam template for {template_name} not found")

        template_kwargs = INPUT_PARAM_TEMPLATES[template_name].copy()

        # Determine the actual param name:
        # 1. From overrides if provided
        # 2. From template if present
        # 3. Fall back to template_name
        name = overrides.pop("name", template_kwargs.pop("name", template_name))

        if note and "description" in template_kwargs:
            template_kwargs["description"] = f"{template_kwargs['description']} ({note})"

        template_kwargs.update(overrides)
        return cls(name=name, **template_kwargs)


@dataclass
class OutputParam:
    """Specification for an output parameter."""

    name: str
    type_hint: Any = None
    description: str = ""
    kwargs_type: str = None
    metadata: dict[str, Any] = None

    def __repr__(self):
        return (
            f"<{self.name}: {self.type_hint.__name__ if hasattr(self.type_hint, '__name__') else str(self.type_hint)}>"
        )

    @classmethod
    def template(cls, template_name: str, note: str = None, **overrides) -> "OutputParam":
        """Get template for name if exists, otherwise raise ValueError."""
        if template_name not in OUTPUT_PARAM_TEMPLATES:
            raise ValueError(f"OutputParam template for {template_name} not found")

        template_kwargs = OUTPUT_PARAM_TEMPLATES[template_name].copy()

        # Determine the actual param name:
        # 1. From overrides if provided
        # 2. From template if present
        # 3. Fall back to template_name
        name = overrides.pop("name", template_kwargs.pop("name", template_name))

        if note and "description" in template_kwargs:
            template_kwargs["description"] = f"{template_kwargs['description']} ({note})"

        template_kwargs.update(overrides)
        return cls(name=name, **template_kwargs)


def format_inputs_short(inputs):
    """
    Format input parameters into a string representation, with required params first followed by optional ones.

    Args:
        inputs: list of input parameters with 'required' and 'name' attributes, and 'default' for optional params

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
        intermediate_inputs: list of intermediate input parameters
        required_intermediate_inputs: list of required intermediate input names
        intermediate_outputs: list of intermediate output parameters

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
        params: list of InputParam or OutputParam objects to format
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
        if isinstance(type_hint, UnionType) or get_origin(type_hint) is Union:
            type_strs = [t.__name__ if hasattr(t, "__name__") else str(t) for t in get_args(type_hint)]
            return " | ".join(type_strs)
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
        else:
            param_str += f"\n{desc_indent}TODO: Add description."

        formatted_params.append(param_str)

    return "\n".join(formatted_params)


def format_input_params(input_params, indent_level=4, max_line_length=115):
    """Format a list of InputParam objects into a readable string representation.

    Args:
        input_params: list of InputParam objects to format
        indent_level: Number of spaces to indent each parameter line (default: 4)
        max_line_length: Maximum length for each line before wrapping (default: 115)

    Returns:
        A formatted string representing all input parameters
    """
    return format_params(input_params, "Inputs", indent_level, max_line_length)


def format_output_params(output_params, indent_level=4, max_line_length=115):
    """Format a list of OutputParam objects into a readable string representation.

    Args:
        output_params: list of OutputParam objects to format
        indent_level: Number of spaces to indent each parameter line (default: 4)
        max_line_length: Maximum length for each line before wrapping (default: 115)

    Returns:
        A formatted string representing all output parameters
    """
    return format_params(output_params, "Outputs", indent_level, max_line_length)


def format_components(components, indent_level=4, max_line_length=115, add_empty_lines=True):
    """Format a list of ComponentSpec objects into a readable string representation.

    Args:
        components: list of ComponentSpec objects to format
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
            if field_value:
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
        configs: list of ConfigSpec objects to format
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


def format_workflow(workflow_map):
    """Format a workflow map into a readable string representation.

    Args:
        workflow_map: Dictionary mapping workflow names to trigger inputs

    Returns:
        A formatted string representing all workflows
    """
    if workflow_map is None:
        return ""

    lines = ["Supported workflows:"]
    for workflow_name, trigger_inputs in workflow_map.items():
        required_inputs = [k for k, v in trigger_inputs.items() if v]
        if required_inputs:
            inputs_str = ", ".join(f"`{t}`" for t in required_inputs)
            lines.append(f"  - `{workflow_name}`: requires {inputs_str}")
        else:
            lines.append(f"  - `{workflow_name}`: default (no additional inputs required)")

    return "\n".join(lines)


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
        inputs: list of input parameters
        intermediate_inputs: list of intermediate input parameters
        outputs: list of output parameters
        description (str, *optional*): Description of the block
        class_name (str, *optional*): Name of the class to include in the documentation
        expected_components (list[ComponentSpec], *optional*): list of expected components
        expected_configs (list[ConfigSpec], *optional*): list of expected configurations

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
        aligned_desc = "\n".join("  " + line.rstrip() for line in desc_lines)
        output += aligned_desc + "\n\n"

    # Add components section if provided
    if expected_components and len(expected_components) > 0:
        components_str = format_components(expected_components, indent_level=2, add_empty_lines=False)
        output += components_str + "\n\n"

    # Add configs section if provided
    if expected_configs and len(expected_configs) > 0:
        configs_str = format_configs(expected_configs, indent_level=2, add_empty_lines=False)
        output += configs_str + "\n\n"

    # Add inputs section
    output += format_input_params(inputs, indent_level=2)

    # Add outputs section
    output += "\n\n"
    output += format_output_params(outputs, indent_level=2)

    return output


def combine_inputs(*named_input_lists: list[tuple[str, list[InputParam]]]) -> list[InputParam]:
    """
    Combines multiple lists of InputParam objects from different blocks. For duplicate inputs, updates only if current
    default value is None and new default value is not None. Warns if multiple non-None default values exist for the
    same input.

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
                if (
                    current_param.default is not None
                    and input_param.default is not None
                    and current_param.default != input_param.default
                ):
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


def combine_outputs(*named_output_lists: list[tuple[str, list[OutputParam]]]) -> list[OutputParam]:
    """
    Combines multiple lists of OutputParam objects from different blocks. For duplicate outputs, keeps the first
    occurrence of each output name.

    Args:
        named_output_lists: List of tuples containing (block_name, output_param_list) pairs

    Returns:
        List[OutputParam]: Combined list of unique OutputParam objects
    """
    combined_dict = {}  # name -> OutputParam

    for block_name, outputs in named_output_lists:
        for output_param in outputs:
            if (output_param.name not in combined_dict) or (
                combined_dict[output_param.name].kwargs_type is None and output_param.kwargs_type is not None
            ):
                combined_dict[output_param.name] = output_param

    return list(combined_dict.values())


def generate_modular_model_card_content(blocks) -> dict[str, Any]:
    """
    Generate model card content for a modular pipeline.

    This function creates a comprehensive model card with descriptions of the pipeline's architecture, components,
    configurations, inputs, and outputs.

    Args:
        blocks: The pipeline's blocks object containing all pipeline specifications

    Returns:
        Dict[str, Any]: A dictionary containing formatted content sections:
            - pipeline_name: Name of the pipeline
            - model_description: Overall description with pipeline type
            - blocks_description: Detailed architecture of blocks
            - components_description: List of required components
            - configs_section: Configuration parameters section
            - inputs_description: Input parameters specification
            - outputs_description: Output parameters specification
            - trigger_inputs_section: Conditional execution information
            - tags: List of relevant tags for the model card
    """
    blocks_class_name = blocks.__class__.__name__
    pipeline_name = blocks_class_name.replace("Blocks", " Pipeline")
    description = getattr(blocks, "description", "A modular diffusion pipeline.")

    # generate blocks architecture description
    blocks_desc_parts = []
    sub_blocks = getattr(blocks, "sub_blocks", None) or {}
    if sub_blocks:
        for i, (name, block) in enumerate(sub_blocks.items()):
            block_class = block.__class__.__name__
            block_desc = block.description.split("\n")[0] if getattr(block, "description", "") else ""
            blocks_desc_parts.append(f"{i + 1}. **{name}** (`{block_class}`)")
            if block_desc:
                blocks_desc_parts.append(f"   - {block_desc}")

            # add sub-blocks if any
            if hasattr(block, "sub_blocks") and block.sub_blocks:
                for sub_name, sub_block in block.sub_blocks.items():
                    sub_class = sub_block.__class__.__name__
                    sub_desc = sub_block.description.split("\n")[0] if getattr(sub_block, "description", "") else ""
                    blocks_desc_parts.append(f"   - *{sub_name}*: `{sub_class}`")
                    if sub_desc:
                        blocks_desc_parts.append(f"     - {sub_desc}")

    blocks_description = "\n".join(blocks_desc_parts) if blocks_desc_parts else "No blocks defined."

    components = getattr(blocks, "expected_components", [])
    if components:
        components_str = format_components(components, indent_level=0, add_empty_lines=False)
        # remove the "Components:" header since template has its own
        components_description = components_str.replace("Components:\n", "").strip()
        if components_description:
            # Convert to enumerated list
            lines = [line.strip() for line in components_description.split("\n") if line.strip()]
            enumerated_lines = [f"{i + 1}. {line}" for i, line in enumerate(lines)]
            components_description = "\n".join(enumerated_lines)
        else:
            components_description = "No specific components required."
    else:
        components_description = "No specific components required. Components can be loaded dynamically."

    configs = getattr(blocks, "expected_configs", [])
    configs_section = ""
    if configs:
        configs_str = format_configs(configs, indent_level=0, add_empty_lines=False)
        configs_description = configs_str.replace("Configs:\n", "").strip()
        if configs_description:
            configs_section = f"\n\n## Configuration Parameters\n\n{configs_description}"

    inputs = blocks.inputs
    outputs = blocks.outputs

    # format inputs as markdown list
    inputs_parts = []
    required_inputs = [inp for inp in inputs if inp.required]
    optional_inputs = [inp for inp in inputs if not inp.required]

    if required_inputs:
        inputs_parts.append("**Required:**\n")
        for inp in required_inputs:
            if hasattr(inp.type_hint, "__name__"):
                type_str = inp.type_hint.__name__
            elif inp.type_hint is not None:
                type_str = str(inp.type_hint).replace("typing.", "")
            else:
                type_str = "Any"
            desc = inp.description or "No description provided"
            inputs_parts.append(f"- `{inp.name}` (`{type_str}`): {desc}")

    if optional_inputs:
        if required_inputs:
            inputs_parts.append("")
        inputs_parts.append("**Optional:**\n")
        for inp in optional_inputs:
            if hasattr(inp.type_hint, "__name__"):
                type_str = inp.type_hint.__name__
            elif inp.type_hint is not None:
                type_str = str(inp.type_hint).replace("typing.", "")
            else:
                type_str = "Any"
            desc = inp.description or "No description provided"
            default_str = f", default: `{inp.default}`" if inp.default is not None else ""
            inputs_parts.append(f"- `{inp.name}` (`{type_str}`){default_str}: {desc}")

    inputs_description = "\n".join(inputs_parts) if inputs_parts else "No specific inputs defined."

    # format outputs as markdown list
    outputs_parts = []
    for out in outputs:
        if hasattr(out.type_hint, "__name__"):
            type_str = out.type_hint.__name__
        elif out.type_hint is not None:
            type_str = str(out.type_hint).replace("typing.", "")
        else:
            type_str = "Any"
        desc = out.description or "No description provided"
        outputs_parts.append(f"- `{out.name}` (`{type_str}`): {desc}")

    outputs_description = "\n".join(outputs_parts) if outputs_parts else "Standard pipeline outputs."

    trigger_inputs_section = ""
    if hasattr(blocks, "trigger_inputs") and blocks.trigger_inputs:
        trigger_inputs_list = sorted([t for t in blocks.trigger_inputs if t is not None])
        if trigger_inputs_list:
            trigger_inputs_str = ", ".join(f"`{t}`" for t in trigger_inputs_list)
            trigger_inputs_section = f"""
### Conditional Execution

This pipeline contains blocks that are selected at runtime based on inputs:
- **Trigger Inputs**: {trigger_inputs_str}
"""

    # generate tags based on pipeline characteristics
    tags = ["modular-diffusers", "diffusers"]

    if hasattr(blocks, "model_name") and blocks.model_name:
        tags.append(blocks.model_name)

    if hasattr(blocks, "trigger_inputs") and blocks.trigger_inputs:
        triggers = blocks.trigger_inputs
        if any(t in triggers for t in ["mask", "mask_image"]):
            tags.append("inpainting")
        if any(t in triggers for t in ["image", "image_latents"]):
            tags.append("image-to-image")
        if any(t in triggers for t in ["control_image", "controlnet_cond"]):
            tags.append("controlnet")
        if not any(t in triggers for t in ["image", "mask", "image_latents", "mask_image"]):
            tags.append("text-to-image")
    else:
        tags.append("text-to-image")

    block_count = len(blocks.sub_blocks)
    model_description = f"""This is a modular diffusion pipeline built with 🧨 Diffusers' modular pipeline framework.

**Pipeline Type**: {blocks_class_name}

**Description**: {description}

This pipeline uses a {block_count}-block architecture that can be customized and extended."""

    return {
        "pipeline_name": pipeline_name,
        "model_description": model_description,
        "blocks_description": blocks_description,
        "components_description": components_description,
        "configs_section": configs_section,
        "inputs_description": inputs_description,
        "outputs_description": outputs_description,
        "trigger_inputs_section": trigger_inputs_section,
        "tags": tags,
    }
