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


class PipelineBlock:
    # YiYi Notes: do we need this?
    # pipelie block should set the default value for all expected config/components, so maybe we do not need to explicitly set the list
    expected_components = []
    expected_configs = []
    model_name = None

    @property
    def inputs(self) -> Tuple[Tuple[str, Any], ...]:
        # (input_name, default_value)
        return ()

    @property
    def intermediates_inputs(self) -> List[str]:
        return []

    @property
    def intermediates_outputs(self) -> List[str]:
        return []

    def __init__(self):
        self.components: Dict[str, Any] = {}
        self.auxiliaries: Dict[str, Any] = {}
        self.configs: Dict[str, Any] = {}

    def __call__(self, pipeline, state: PipelineState) -> PipelineState:
        raise NotImplementedError("__call__ method must be implemented in subclasses")

    def __repr__(self):
        class_name = self.__class__.__name__
        base_class = self.__class__.__bases__[0].__name__

        # Components section - group into main components and auxiliaries if needed
        expected_components = set(getattr(self, "expected_components", []))
        loaded_components = set(self.components.keys())
        all_components = sorted(expected_components | loaded_components)

        main_components = []
        auxiliary_components = []
        for k in all_components:
            component_str = f"    - {k}={type(self.components[k]).__name__}" if k in loaded_components else f"    - {k}"
            if k in getattr(self, "auxiliary_components", []):
                auxiliary_components.append(component_str)
            else:
                main_components.append(component_str)

        components = "Components:\n" + "\n".join(main_components)
        if auxiliary_components:
            components += "\n  Auxiliaries:\n" + "\n".join(auxiliary_components)

        # Configs section
        expected_configs = set(getattr(self, "expected_configs", []))
        loaded_configs = set(self.configs.keys())
        all_configs = sorted(expected_configs | loaded_configs)
        configs = "Configs:\n" + "\n".join(
            f"    - {k}={self.configs[k]}" if k in loaded_configs else f"    - {k}" 
            for k in all_configs
        )

        # Inputs section
        inputs = "inputs: " + ", ".join(
            f"{name}={default}" if default is not None else name 
            for name, default in self.inputs
        )

        # Intermediates section
        input_set = set(self.intermediates_inputs)
        output_set = set(self.intermediates_outputs)
        
        modified_inputs = [f"{item}*" for item in self.intermediates_inputs]
        new_outputs = [item for item in self.intermediates_outputs if item not in input_set]
        
        intermediates = f"intermediates: {', '.join(modified_inputs)} -> {', '.join(new_outputs)}"

        return (
            f"{class_name}(\n"
            f"  Class: {base_class}\n"
            f"  {components}\n"
            f"  {configs}\n"
            f"  {inputs}\n"
            f"  {intermediates}\n"
            f")"
        )


def combine_inputs(*named_input_lists: List[Tuple[str, List[Tuple[str, Any]]]]) -> List[Tuple[str, Any]]:
    """
    Combines multiple lists of (name, default_value) tuples from different blocks. For duplicate inputs, updates only if 
    current value is None and new value is not None. Warns if multiple non-None default values exist for the same input.

    Args:
        named_input_lists: List of tuples containing (block_name, input_list) pairs
    """
    combined_dict = {}
    value_sources = {}
    
    for block_name, inputs in named_input_lists:
        for name, value in inputs:
            if name in combined_dict:
                current_value = combined_dict[name]
                if current_value is not None and value is not None and current_value != value:
                    warnings.warn(
                        f"Multiple different default values found for input '{name}': "
                        f"{current_value} (from block '{value_sources[name]}') and "
                        f"{value} (from block '{block_name}'). Using {current_value}."
                    )
                if current_value is None and value is not None:
                    combined_dict[name] = value
                    value_sources[name] = block_name
            else:
                combined_dict[name] = value
                value_sources[name] = block_name
    return list(combined_dict.items())


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

    # YiYi TODO: address the case where multiple blocks have the same component/auxiliary/config; give out warning etc
    @property
    def components(self):
        # Combine components from all blocks
        components = {}
        for block_name, block in self.blocks.items():
            for key, value in block.components.items():
                # Only update if:
                # 1. Key doesn't exist yet in components, OR
                # 2. New value is not None
                if key not in components or value is not None:
                    components[key] = value
        return components

    @property
    def auxiliaries(self):
        # Combine auxiliaries from all blocks
        auxiliaries = {}
        for block_name, block in self.blocks.items():
            auxiliaries.update(block.auxiliaries)
        return auxiliaries

    @property
    def configs(self):
        # Combine configs from all blocks
        configs = {}
        for block_name, block in self.blocks.items():
            configs.update(block.configs)
        return configs

    @property
    def inputs(self) -> List[Tuple[str, Any]]:
        named_inputs = [(name, block.inputs) for name, block in self.blocks.items()]
        return combine_inputs(*named_inputs)

    @property
    def intermediates_inputs(self) -> List[str]:
        return list(set().union(*(block.intermediates_inputs for block in self.blocks.values())))

    @property
    def intermediates_outputs(self) -> List[str]:
        return list(set().union(*(block.intermediates_outputs for block in self.blocks.values())))

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

        all_triggers = set(self.trigger_to_block_map.keys())
        
        sections = []
        for trigger in sorted(all_triggers, key=lambda x: str(x)):
            sections.append(f"\n  Trigger Input: {trigger}\n")
            
            block = self.trigger_to_block_map.get(trigger)
            if block is None:
                continue
                
            expected_components = set(getattr(block, "expected_components", []))
            loaded_components = set(k for k, v in self.components.items() 
                                 if v is not None and hasattr(block, k))
            all_components = sorted(expected_components | loaded_components)
            if all_components:
                sections.append("    Components:\n" + "\n".join(
                    f"      - {k}={type(self.components[k]).__name__}" if k in loaded_components 
                    else f"      - {k}" for k in all_components
                ))

            if self.auxiliaries:
                sections.append("    Auxiliaries:\n" + "\n".join(
                    f"      - {k}={type(v).__name__}" 
                    for k, v in self.auxiliaries.items()
                ))

            if self.configs:
                sections.append("    Configs:\n" + "\n".join(
                    f"      - {k}={v}" for k, v in self.configs.items()
                ))

            sections.append(f"    Block: {block.__class__.__name__}")
            
            if hasattr(block, "inputs"):
                inputs_str = ", ".join(
                    name if default is None else f"{name}={default}"
                    for name, default in block.inputs
                )
                if inputs_str:
                    sections.append(f"      inputs: {inputs_str}")

            if hasattr(block, "intermediates_inputs") or hasattr(block, "intermediates_outputs"):
                intermediates_str = ""
                if hasattr(block, "intermediates_inputs"):
                    intermediates_str += f"{', '.join(block.intermediates_inputs)}"
                if hasattr(block, "intermediates_outputs"):
                    if intermediates_str:
                        intermediates_str += " -> "
                    intermediates_str += f"{', '.join(block.intermediates_outputs)}"
                if intermediates_str:
                    sections.append(f"      intermediates: {intermediates_str}")
            
            sections.append("") 

        return (
            f"{class_name}(\n"
            f"  Class: {base_class}\n"
            f"{chr(10).join(sections)}"
            f")"
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

    # YiYi TODO: address the case where multiple blocks have the same component/auxiliary/config; give out warning etc
    @property
    def components(self):
        # Combine components from all blocks
        components = {}
        for block_name, block in self.blocks.items():
            for key, value in block.components.items():
                # Only update if:
                # 1. Key doesn't exist yet in components, OR
                # 2. New value is not None
                if key not in components or value is not None:
                    components[key] = value
        return components

    @property
    def auxiliaries(self):
        # Combine auxiliaries from all blocks
        auxiliaries = {}
        for block_name, block in self.blocks.items():
            auxiliaries.update(block.auxiliaries)
        return auxiliaries

    @property
    def configs(self):
        # Combine configs from all blocks
        configs = {}
        for block_name, block in self.blocks.items():
            configs.update(block.configs)
        return configs

    @property
    def inputs(self) -> List[Tuple[str, Any]]:
        named_inputs = [(name, block.inputs) for name, block in self.blocks.items()]
        return combine_inputs(*named_inputs)

    @property
    def intermediates_inputs(self) -> List[str]:
        inputs = set()
        outputs = set()

        # Go through all blocks in order
        for block in self.blocks.values():
            # Add inputs that aren't in outputs yet
            inputs.update(input_name for input_name in block.intermediates_inputs if input_name not in outputs)
            # Add this block's outputs
            outputs.update(block.intermediates_outputs)

        return list(inputs)

    @property
    def intermediates_outputs(self) -> List[str]:
        return list(set().union(*(block.intermediates_outputs for block in self.blocks.values())))
    
    @property
    def final_intermediates_outputs(self) -> List[str]:
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

        def fn_recursive_traverse(block, block_name, trigger_inputs):
            result_blocks = OrderedDict()
            # sequential or PipelineBlock
            if not hasattr(block, 'block_trigger_inputs'):
                if hasattr(block, 'blocks'):
                    # sequential
                    for block_name, block in block.blocks.items():
                        blocks_to_update = fn_recursive_traverse(block, block_name, trigger_inputs)
                        result_blocks.update(blocks_to_update)
                else:
                    # PipelineBlock
                    result_blocks[block_name] = block
                return result_blocks
                
            # auto
            else:
                # Find first block_trigger_input that matches any value in our trigger_value tuple
                this_block = None
                for trigger_input in block.block_trigger_inputs:
                    if trigger_input is not None and trigger_input in trigger_inputs:
                        this_block = block.trigger_to_block_map[trigger_input]
                        break
                
                # If no matches found, try to get the default (None) block
                if this_block is None and None in block.block_trigger_inputs:
                    this_block = block.trigger_to_block_map[None]
                
                if this_block is not None:
                    # sequential/auto
                    if hasattr(this_block, 'blocks'):
                        result_blocks.update(fn_recursive_traverse(this_block, block_name, trigger_inputs))
                    else:
                        # PipelineBlock
                        result_blocks[block_name] = this_block

            return result_blocks
        
        all_blocks = OrderedDict()
        for block_name, block in self.blocks.items():
            blocks_to_update = fn_recursive_traverse(block, block_name, trigger_inputs)
            all_blocks.update(blocks_to_update)
        return all_blocks
    
    def get_triggered_blocks(self, *trigger_inputs):
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
            header += "\n"  # Add empty line before
            header += "  " + "=" * 100 + "\n"  # Add decorative line
            header += "  This pipeline block contains dynamic blocks that are selected at runtime based on your inputs.\n"
            header += "  You can use `get_triggered_blocks(input1, input2,...)` to see which blocks will be used for your trigger inputs.\n"
            header += "  Use `get_triggered_blocks()` to see blocks will be used for default inputs (when no trigger inputs are provided)\n"
            header += f"  Trigger Inputs: {self.trigger_inputs}\n"
            header += "  " + "=" * 100 + "\n"  # Add decorative line
            header += "\n"  # Add empty line after

        # Components section
        expected_components = set(getattr(self, "expected_components", []))
        loaded_components = set(self.components.keys())
        all_components = sorted(expected_components | loaded_components)
        components_str = "  Components:\n" + "\n".join(
            f"    - {k}={type(self.components[k]).__name__}" if k in loaded_components else f"    - {k}"
            for k in all_components
        )

        # Auxiliaries section
        auxiliaries_str = "  Auxiliaries:\n" + "\n".join(
            f"    - {k}={type(v).__name__}" for k, v in self.auxiliaries.items()
        )

        # Configs section
        expected_configs = set(getattr(self, "expected_configs", []))
        loaded_configs = set(self.configs.keys())
        all_configs = sorted(expected_configs | loaded_configs)
        configs_str = "  Configs:\n" + "\n".join(
            f"    - {k}={self.configs[k]}" if k in loaded_configs else f"    - {k}" for k in all_configs
        )

        blocks_str = "  Blocks:\n"
        for i, (name, block) in enumerate(self.blocks.items()):
            blocks_str += f"    {i}. {name} ({block.__class__.__name__})\n"

            if hasattr(block, "inputs"):
                inputs_str = ", ".join(
                    name if default is None else f"{name}={default}"
                    for name, default in block.inputs
                )
                blocks_str += f"       inputs: {inputs_str}\n"

            if hasattr(block, "intermediates_inputs") or hasattr(block, "intermediates_outputs"):
                intermediates_str = ""
                if hasattr(block, "intermediates_inputs"):
                    inputs_set = set(block.intermediates_inputs)
                    intermediates_str += ", ".join(f"*{inp}" if inp in (getattr(block, "intermediates_outputs", set())) else inp 
                                                for inp in block.intermediates_inputs)

                if hasattr(block, "intermediates_outputs"):
                    if intermediates_str:
                        intermediates_str += " -> "
                    outputs_set = set(block.intermediates_outputs)
                    new_outputs = outputs_set - inputs_set if hasattr(block, "intermediates_inputs") else outputs_set
                    intermediates_str += ", ".join(new_outputs)

                if intermediates_str:
                    blocks_str += f"       intermediates: {intermediates_str}\n"
            blocks_str += "\n"
        
        inputs_str = "  inputs:\n    " + ", ".join(
            f"{name}={default}" if default is not None else f"{name}"
            for name, default in self.inputs
        )
 
        modified_inputs = [f"*{inp}" if inp in self.intermediates_outputs else inp for inp in self.intermediates_inputs]
        new_outputs = [out for out in self.intermediates_outputs if out not in self.intermediates_inputs]

        intermediates_str = (
            "\n  Intermediates:\n"
            f"      - inputs: {', '.join(modified_inputs)}\n"
            f"      - outputs: {', '.join(new_outputs)}\n"
            f"      - final outputs: {', '.join(self.final_intermediates_outputs)}"
        )

        return (
            f"{header}\n"
            f"{components_str}\n"
            f"{auxiliaries_str}\n"
            f"{configs_str}\n"
            f"{blocks_str}\n"
            f"{inputs_str}\n"
            f"{intermediates_str}\n"
            f")"
        )


class ModularPipeline(ConfigMixin):
    """
    Base class for all Modular pipelines.

    """

    config_name = "model_index.json"
    _exclude_from_cpu_offload = []

    def __init__(self, block):
        self.pipeline_block = block

        # add default components from pipeline_block (e.g. guider)
        for key, value in block.components.items():
            setattr(self, key, value)

        # add default configs from pipeline_block (e.g. force_zeros_for_empty_prompt)
        self.register_to_config(**block.configs)

        # add default auxiliaries from pipeline_block (e.g. image_processor)
        for key, value in block.auxiliaries.items():
            setattr(self, key, value)

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
        for name in self.expected_components:
            if hasattr(self, name):
                components[name] = getattr(self, name)
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

        for name, default in default_params.items():
            if name in input_params:
                if name not in self.pipeline_block.intermediates_inputs:
                    state.add_input(name, input_params.pop(name))
                else:
                    state.add_input(name, input_params[name])
            elif name not in state.inputs:
                state.add_input(name, default)

        for name in self.pipeline_block.intermediates_inputs:
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

        if isinstance(output, str):
            return state.get_output(output)
        elif isinstance(output, (list, tuple)):
            outputs = {}
            for output_name in output:
                outputs[output_name] = state.get_output(output_name)
            return outputs
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

        for component_name in self.expected_components:
            if component_name in kwargs:
                if hasattr(self, component_name) and getattr(self, component_name) is not None:
                    current_component = getattr(self, component_name)
                    new_component = kwargs[component_name]

                    if not isinstance(new_component, current_component.__class__):
                        logger.info(
                            f"Overwriting existing component '{component_name}' "
                            f"(type: {current_component.__class__.__name__}) "
                            f"with type: {new_component.__class__.__name__})"
                        )
                    elif isinstance(current_component, torch.nn.Module):
                        if id(current_component) != id(new_component):
                            logger.info(
                                f"Overwriting existing component '{component_name}' "
                                f"(type: {type(current_component).__name__}) "
                                f"with new value (type: {type(new_component).__name__})"
                            )

                setattr(self, component_name, kwargs.pop(component_name))

        configs_to_add = {}
        for config_name in self.expected_configs:
            if config_name in kwargs:
                configs_to_add[config_name] = kwargs.pop(config_name)
        self.register_to_config(**configs_to_add)

    @property
    def default_call_parameters(self) -> Dict[str, Any]:
        params = {}
        for name, default in self.pipeline_block.inputs:
            params[name] = default
        return params

    def __repr__(self):
        output = "ModularPipeline:\n"
        output += "==============================\n\n"
        
        block = self.pipeline_block
        if hasattr(block, "trigger_inputs") and block.trigger_inputs:
            output += "\n"
            output += "  Trigger Inputs:\n"
            output += "  --------------\n"
            output += f"  This pipeline contains dynamic blocks that are selected at runtime based on your inputs.\n"
            output += f"  • Trigger inputs: {block.trigger_inputs}\n"
            output += f"  • Use .pipeline_block.get_triggered_blocks(*inputs) to see which blocks will be used for specific inputs\n"
            output += f"  • Use .pipeline_block.get_triggered_blocks() to see blocks will be used for default inputs (when no trigger inputs are provided)\n"
            output += "\n"

        output += "Pipeline Block:\n"
        output += "--------------\n"
        if hasattr(block, "blocks"):
            output += f"{block.__class__.__name__}\n"
            base_class = block.__class__.__bases__[0].__name__
            output += f" (Class: {base_class})\n" if base_class != "object" else "\n"
            for sub_block_name, sub_block in block.blocks.items():
                if hasattr(block, "block_trigger_inputs"):
                    trigger_input = block.block_to_trigger_map[sub_block_name]
                    trigger_info = f" [trigger: {trigger_input}]" if trigger_input is not None else " [default]"
                    output += f"  • {sub_block_name} ({sub_block.__class__.__name__}){trigger_info}\n"
                else:
                    output += f"  • {sub_block_name} ({sub_block.__class__.__name__})\n"
        else:
            output += f"{block.__class__.__name__}\n"
        output += "\n"

        # List the components registered in the pipeline
        output += "Registered Components:\n"
        output += "----------------------\n"
        for name, component in self.components.items():
            output += f"{name}: {type(component).__name__}"
            if hasattr(component, "dtype") and hasattr(component, "device"):
                output += f" (dtype={component.dtype}, device={component.device})"
            output += "\n"
        output += "\n"

        # List the configs registered in the pipeline
        output += "Registered Configs:\n"
        output += "------------------\n"
        for name, config in self.config.items():
            output += f"{name}: {config!r}\n"
        output += "\n"

        # List the default call parameters
        output += "Call Parameters:\n"
        output += "------------------------\n"
        for name, default in self.default_call_parameters.items():
            output += f"{name}: {default!r}\n"

        output += "\nIntermediate inputs:\n"
        output += "--------------------------\n"
        for name in self.pipeline_block.intermediates_inputs:
            output += f"{name}: \n"


        if hasattr(block, "intermediates_outputs"):
            output += "\nIntermediate outputs:\n"
            output += "--------------------------\n"
            output += f"{', '.join(block.intermediates_outputs)}\n\n"

        # Add final intermediate outputs section at the bottom
        if hasattr(block, "final_intermediates_outputs"):
            output += "Final intermediate outputs:\n"
            output += "--------------------------\n"
            output += f"{', '.join(block.final_intermediates_outputs)}\n"

        return output

    # YiYi TO-DO: try to unify the to method with the one in DiffusionPipeline
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
