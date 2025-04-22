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
from .modular_pipeline_util import (
    format_components,
    format_configs,
    format_input_params,
    format_inputs_short,
    format_intermediates_short,
    format_output_params,
    format_params,
    make_doc_string,
)


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



class ModularPipelineMixin:
    """
    Mixin for all PipelineBlocks: PipelineBlock, AutoPipelineBlocks, SequentialPipelineBlocks
    """


    def __init__(self):
        self.components_manager = None
        self.components_manager_prefix = ""
        self.components_state = None
    
    # YiYi TODO: not sure this is the best method name
    def compile(self, components_manager: ComponentsManager, label: Optional[str] = None):
        self.components_manager = components_manager
        self.components_manager_prefix = "" if label is None else f"{label}_"
        self.components_state = ComponentsState(self.expected_components, self.expected_configs)
        
        components_to_add = self.components_manager.get(f"{self.components_manager_prefix}*")
        self.components_state.update_states(self.expected_components, self.expected_configs, **components_to_add)


    @property
    def default_call_parameters(self) -> Dict[str, Any]:
        params = {}
        for input_param in self.inputs:
            params[input_param.name] = input_param.default
        return params

    def run(self, state: PipelineState = None, output: Union[str, List[str]] = None, **kwargs):
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

        intermediates_inputs = [inp.name for inp in self.intermediates_inputs]
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
                pipeline, state = self(self, state)
            except Exception:
                error_msg = f"Error in block: ({self.__class__.__name__}):\n"
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


class ComponentsState(ConfigMixin):
    """
    Base class for all Modular pipelines.

    """
    config_name = "model_index.json"

    def __init__(self, component_specs, config_specs):

        for component_spec in component_specs:
            if component_spec.obj is not None:
                setattr(self, component_spec.name, component_spec.obj)
            else:
                setattr(self, component_spec.name, None)
        
        default_configs = {}
        for config_spec in config_specs:
            default_configs[config_spec.name] = config_spec.default
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
    def components(self):
        components = {}
        for component_spec in self.expected_components:
            if hasattr(self, component_spec.name):
                components[component_spec.name] = getattr(self, component_spec.name)
        return components

    def update_states(self, expected_components, expected_configs, **kwargs):
        """
        Update components and configs after instance creation. Auxiliaries (e.g. image_processor) should be defined for
        each pipeline block, does not need to be updated by users. Logs if existing non-None components are being
        overwritten.

        Args:
            kwargs (dict): Keyword arguments to update the states.
        """

        for component in expected_components:
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

                setattr(self.components_state, component.name, kwargs.pop(component.name))

        configs_to_add = {}
        for config in expected_configs:
            if config.name in kwargs:
                configs_to_add[config.name] = kwargs.pop(config.name)
        self.register_to_config(**configs_to_add)

    # YiYi TODO: should support to method
    def to(self, *args, **kwargs):
        pass
