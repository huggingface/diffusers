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
import os

from huggingface_hub.utils import validate_hf_hub_args

from ..configuration_utils import ConfigMixin
from ..utils import (
    is_accelerate_available,
    is_accelerate_version,
    logging,
    PushToHubMixin,
)
from .pipeline_loading_utils import _get_pipeline_class
from .modular_pipeline_utils import (
    ComponentSpec,
    ConfigSpec,
    InputParam,
    OutputParam,
    format_components,
    format_configs,
    format_input_params,
    format_inputs_short,
    format_intermediates_short,
    format_output_params,
    format_params,
    make_doc_string,
)

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
    
    # def register_loader(self, global_components_manager: ComponentsManager, label: Optional[str] = None):
    #     self._global_components_manager = global_components_manager
    #     self._label = label
    
    #YiYi TODO: add validation for kwargs?
    def setup_loader(self, **kwargs):
        """
        Set up the components loader with repository information.
        
        Args:
            **kwargs: Configuration for component loading.
                - repo: Default repository to use for all components
                - For individual components, pass a tuple of (repo, subfolder)
                  e.g., text_encoder=("repo_name", "text_encoder")
        
        Examples:
            # Set repo for all components (subfolder will be component name)
            setup_loader(repo="stabilityai/stable-diffusion-xl-base-1.0")
            
            # Set specific repo/subfolder for individual components
            setup_loader(
                unet=("stabilityai/stable-diffusion-xl-base-1.0", "unet"),
                text_encoder=("stabilityai/stable-diffusion-xl-base-1.0", "text_encoder")
            )
            
            # Set default repo and override for specific components
            setup_loader(
                repo="stabilityai/stable-diffusion-xl-base-1.0",
                unet=(""stabilityai/stable-diffusion-xl-refiner-1.0", "unet")
            )
        """

        # Create deep copies to avoid modifying the original specs
        component_specs = deepcopy(self.expected_components)
        config_specs = deepcopy(self.expected_configs)

        expected_component_names = set([c.name for c in component_specs])
        expected_config_names = set([c.name for c in config_specs])
        
        # Check if a default repo is provided
        repo = kwargs.pop("repo", None)
        revision = kwargs.pop("revision", None)
        variant = kwargs.pop("variant", None)

        passed_component_kwargs = {k: kwargs.pop(k) for k in expected_component_names if k in kwargs}
        passed_config_kwargs = {k: kwargs.pop(k) for k in expected_config_names if k in kwargs}
        if len(kwargs) > 0:
            logger.warning(f"Unused keyword arguments: {kwargs.keys()}. This input will be ignored.")
        
        for name, value in passed_component_kwargs.items():
            if not isinstance(value, (tuple, list, str)):
                raise ValueError(f"Invalid value for component '{name}': {value}. Expected a string, tuple or list")
            elif isinstance(value, (tuple, list)) and len(value) > 2:
                raise ValueError(f"Invalid value for component '{name}': {value}. Expected a tuple or list of length 1 or 2.")
        
        for name, value in passed_config_kwargs.items():
            if not isinstance(value, str):
                raise ValueError(f"Invalid value for config '{name}': {value}. Expected a string")

        # First apply default repo to all components if provided
        if repo is not None:
            for component_spec in component_specs:
                # components defined with a config are classes like image_processor or guider, 
                # skip setting loading related attributes for them, they should be initialized with the default config
                if component_spec.config is None:
                    component_spec.repo = repo
                
                    # YiYi TODO: should also accept `revision` and `variant` as a dict here so user can set different values for different components
                    if revision is not None:
                        component_spec.revision = revision
                    if variant is not None:
                        component_spec.variant = variant
            for config_spec in config_specs:
                config_spec.repo = repo
        
        # apply component-specific overrides
        for name, value in passed_component_kwargs.items():
            if not isinstance(value, (tuple, list)):
                value = (value,)
            # Find the matching component spec
            for component_spec in component_specs:
                if component_spec.name == name:
                    # Handle tuple of (repo, subfolder)
                    component_spec.repo = value[0]
                    if len(value) > 1:
                        component_spec.subfolder = value[1]
                    break
        
        # apply config overrides
        for name, value in passed_config_kwargs.items():
            for config_spec in config_specs:
                if config_spec.name == name:
                    config_spec.repo = value
                    break

        # Import components loader (it is model-specific class)
        loader_class_name = MODULAR_LOADER_MAPPING[self.model_name]
        diffusers_module = importlib.import_module(self.__module__.split(".")[0])
        loader_class = getattr(diffusers_module, loader_class_name)
        
        # Create the loader with the updated specs
        self.loader = loader_class(component_specs, config_specs)


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


# YiYi TODO: refactor the _fetch_class_library_tuple in pipeline_loading_utils.py to acceept class (current object)
from .pipeline_loading_utils import LOADABLE_CLASSES
import importlib
def _fetch_class_library_tuple(module_class):
    # import it here to avoid circular import
    diffusers_module = importlib.import_module(__name__.split(".")[0])
    pipelines = getattr(diffusers_module, "pipelines")

    library = module_class.__module__.split(".")[0]

    # check if the module is a pipeline module
    module_path_items = module_class.__module__.split(".")
    pipeline_dir = module_path_items[-2] if len(module_path_items) > 2 else None

    path = module_class.__module__.split(".")
    is_pipeline_module = pipeline_dir in path and hasattr(pipelines, pipeline_dir)

    # if library is not in LOADABLE_CLASSES, then it is a custom module.
    # Or if it's a pipeline module, then the module is inside the pipeline
    # folder so we set the library to module name.
    if is_pipeline_module:
        library = pipeline_dir
    elif library not in LOADABLE_CLASSES:
        library = module_class.__module__

    # retrieve class_name
    class_name = module_class.__name__

    return (library, class_name)


def simple_import_class_obj(library_name, class_name):
    from diffusers import pipelines
    is_pipeline_module = hasattr(pipelines, library_name)

    if is_pipeline_module:
        pipeline_module = getattr(pipelines, library_name)
        class_obj = getattr(pipeline_module, class_name)
    else:
        library = importlib.import_module(library_name)
        class_obj = getattr(library, class_name)

    return class_obj

class ModularLoader(ConfigMixin, PushToHubMixin):
    """
    Base class for all Modular pipelines loaders.

    """
    config_name = "modular_model_index.json"


    def register_components(self, **kwargs):
        """
        Register components with their corresponding specs. 
        This method is called when component changed or its spec changed (in self.component_specs).

        Args:
            **kwargs: Keyword arguments where keys are component names and values are component objects.
        
        """
        for name, module in kwargs.items():

            current_module = getattr(self, name, None)
    
            # update config based on the updated component spec
            component_spec = self.component_specs.get(name)
            if component_spec is None:
                logger.warning(f"register_components: skipping unknown component '{name}'")
                continue

            library, class_name = _fetch_class_library_tuple(component_spec.type_hint)
            load_spec_dict = OrderedDict({
                "repo": component_spec.repo,
                "subfolder": component_spec.subfolder,
            })

            register_dict = {name: (library, class_name, load_spec_dict)}

            # save model index config
            self.register_to_config(**register_dict)

            # set the component as attribute
            # skip if the component is already registered with the same object
            if current_module is module:
                logger.info(f"register_components: {name} is already registered with same object, skipping")
                continue
            
            # it module is not an instance of the expected type, still register it but with a warning
            if module is not None and not isinstance(module, component_spec.type_hint):
                logger.warning(f"register_components: adding {name} with type: {module.__class__.__name__}, expected: {component_spec.type_hint.__name__}")

            # warn if unregister
            if current_module is not None and module is None:
                logger.info(
                    f"register_components: setting '{name}' to None "
                    f"(was {current_module.__class__.__name__})"
                )
            # warn if class mismatch
            elif current_module is not None \
                and module is not None \
                and not isinstance(module, current_module.__class__):
                logger.warning(
                    f"register_components: overwriting component '{name}' "
                    f"(type {current_module.__class__.__name__}) "
                    f"with DIFFERENT type {module.__class__.__name__}"
                )
            # same type, new instance → debug
            elif current_module is not None \
                and module is not None \
                and isinstance(module, current_module.__class__) \
                and current_module != module:
                logger.debug(
                    f"register_components: replacing existing '{name}' "
                    f"(same type {type(current_module).__name__}, new instance)"
                )

            # finally set models
            setattr(self, name, module)


    def __init__(self, component_specs: List[ComponentSpec], config_specs: Optional[List[ConfigSpec]]=None):
        self.component_specs = {
            spec.name: deepcopy(spec) for spec in (component_specs or [])
        }
        self.config_specs = {
            spec.name: deepcopy(spec) for spec in (config_specs  or [])
        }

        register_components_dict = {}
        for component_spec in self.component_specs.values():
            register_components_dict[component_spec.name] = None
        self.register_components(**register_components_dict)
        
        default_configs = {}
        for config_spec in self.config_specs.values():
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
    def components(self) -> Dict[str, Any]:
        # return only components we've actually set as attributes on self
        return {
            name: getattr(self, name)
            for name in self.component_specs.keys()
            if hasattr(self, name)
        }

    def update(self, repo=None, **kwargs):
        """
        Update components and configs after instance creation.
        
        Args:
            repo (str, optional): Default repository to use for all components
            **kwargs:
                Updates, which can be:
                - For components:
                    - A string: Used as the repository name
                    - A tuple: (repo, subfolder) or (repo,)
                    - A ComponentSpec: Replace the existing spec

                    If the component is already loaded, it will be reloaded with updated info;
                    otherwise only the spec is updated.

                - For configs:
                    - Any value: Update the config value

            - Additional loader options:
                Passed through to the underlying component loading methods
                (e.g., from_pretrained), such as torch_dtype, revision, variant, etc.
        """   

        # extract component_updates from `kwargs``:
        # e.g. loader.update(unet=..., vae=...)` -> {"unet": ..., "vae": ...}
        component_updates = {k: kwargs.pop(k) for k in self.component_specs.keys() if k in kwargs}
        # extract config_updates from `kwargs``:
        # e.g. loader.update(requires_aesthetics_score=False) -> {"requires_aesthetics_score": False}
        config_updates = {k: kwargs.pop(k) for k in self.config_specs.keys() if k in kwargs}
        
        # create a dict to contain all the component specs to be updated,
        new_components_specs = {}

        # update global default repo on each component spec
        # e.g loader.update(repo="new_repo") -> {"unet": ComponentSpec(repo="new_repo", ...), "vae": ComponentSpec(repo="new_repo", ...)}
        if repo is not None:
            for spec in self.component_specs.values():
                new_spec = deepcopy(spec)
                new_spec.repo = repo
                new_components_specs[spec.name] = new_spec

        # update component specs with component updates extracted from the `kwargs`
        # YiYi Notes: should we automatically reload?
        for name, new_value in component_updates.items():
            # make a copy of the spec to avoid partial mutation
            new_spec = deepcopy(self.component_specs[name])
            if isinstance(new_value, ComponentSpec):
                # e.g. loader.update(unet = ComponentSpec(type_hint=UNet2DConditionModel, ...))
                new_spec = new_value
            elif isinstance(new_value, str):
                # e.g. loader.update(unet="repo/unet")
                new_spec.repo = new_value
            elif isinstance(new_value, (tuple, list)):
                # e.g. loader.update(unet = ("repo/unet", "subfolder"))
                new_spec.repo = new_value[0]
                if len(new_value) > 1:
                    new_spec.subfolder = new_value[1]
            
            # potentially override the spec if global repo is provided
            new_components_specs[name] = new_spec
                
        # attempt to update the components if it's already loaded
        components_to_register = {}
        for name, new_component_spec in new_components_specs.items():
            if getattr(self, name, None) is not None:
                try:
                    # perform atomic update only if successful load the new component
                    # load, update components_spec and register_components
                    new_component = new_component_spec.load(**kwargs)
                    self.component_specs[name] = new_component_spec
                    components_to_register[name] = new_component
                except Exception as e:
                    logger.warning(f"Failed to update component '{name}': {e}")
            else:
                # only update the spec if the component is not loaded (e.g. self.unet = None)
                self.component_specs[name] = new_component_spec
                components_to_register[name] = None
        
        self.register_components(**components_to_register)

        config_to_register = {}
        for name, new_value in config_updates.items():
            if isinstance(new_value, ConfigSpec):
                # e.g. requires_aesthetics_score = ConfigSpec(name="requires_aesthetics_score", default=False)
                self.config_specs[name] = new_value
                config_to_register[name] = new_value.default
            else:
                # e.g. requires_aesthetics_score = False
                self.config_specs[name].default = new_value
                config_to_register[name] = new_value
        self.register_to_config(**config_to_register)

    def load(self, **kwargs):
        """
        Load components and optionally set config values.
        
        This method has three modes:
        1. `self.load()` - load all components from their specs
        2. `self.load(unet=unet, text_encoder=text_encoder)` - use provided components directly,
           load remaining components from specs
        3. `self.load(...,requires_aesthetics_score=False)` - additinally set config values
        
        Args:
            **kwargs: Can include:
                - Component objects to set directly (e.g., unet=my_unet)
                - config values to set (e.g., requires_aesthetics_score=False)
                - additional kwargs to be passed to `from_pretrained()`, e.g. torch_dtype=torch.bfloat16
        
        Returns:
            self: The loader instance with loaded components
        """
        config_updates = {k: kwargs.pop(k) for k in self.config_specs.keys() if k in kwargs}
        passed_component_obj = {k: kwargs.pop(k) for k in self.component_specs.keys() if k in kwargs}
        
        # 1. Set any config values provided (without updating defaults in specs)
        if config_updates:
            self.register_to_config(**config_updates)
        
        # 2. Process components
        components_to_register = {}
        
        # First register the components provided directly
        for name, component in passed_component_obj.items():
            components_to_register[name] = component
        
        # Then load the remaining components from specs
        remaining_components = set(self.component_specs.keys()) - set(passed_component_obj.keys())
        for name in remaining_components:
            spec = self.component_specs[name]
            try:
                if spec.repo is not None:
                    components_to_register[name] = spec.load(**kwargs)
                elif spec.config is not None:
                    components_to_register[name] = spec.create()
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
    def save_pretrained(self, save_directory: Union[str, os.PathLike], push_to_hub: bool = False, **kwargs):

        component_names = list(self.component_specs.keys())
        config_names = list(self.config_specs.keys())
        self.register_to_config(_components_names=component_names, _configs_names=config_names)
        self.save_config(save_directory=save_directory, push_to_hub=push_to_hub, **kwargs)
    
    @classmethod
    @validate_hf_hub_args
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]] = None, **kwargs):
        
        config_dict = cls.load_config(pretrained_model_name_or_path, **kwargs)
        expected_component = set(config_dict.pop("_components_names"))
        expected_config = set(config_dict.pop("_configs_names"))

        component_specs = []
        config_specs = []
        for name, value in config_dict.items():
            if name in expected_component and isinstance(value, (tuple, list)) and len(value) == 3:
                library_name, class_name, load_spec_dict = value
                type_hint = simple_import_class_obj(library_name, class_name)
                component_specs.append(ComponentSpec(name=name, type_hint=type_hint, **load_spec_dict))

            elif name in expected_config:
                config_specs.append(ConfigSpec(name=name, default=value))
        return cls(component_specs, config_specs=config_specs)


