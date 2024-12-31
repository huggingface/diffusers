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
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from tqdm.auto import tqdm

from ..configuration_utils import ConfigMixin
from ..utils import (
    is_accelerate_available,
    is_accelerate_version,
    logging,
)
from ..utils.hub_utils import validate_hf_hub_args
from .pipeline_loading_utils import _fetch_class_library_tuple
from .pipeline_utils import DiffusionPipeline


if is_accelerate_available():
    import accelerate

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


MODULAR_PIPELINE_MAPPING = {
    "stable-diffusion-xl": "StableDiffusionXLModularPipeline",
}


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
                return f"Tensor(\n      dtype={v.dtype}, shape={v.shape}\n      {v})"
            elif isinstance(v, list) and len(v) > 0 and hasattr(v[0], "shape") and hasattr(v[0], "dtype"):
                return f"[Tensor(\n      dtype={v[0].dtype}, shape={v[0].shape}\n      {v[0]}), ...]"
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
    expected_components = []
    expected_auxiliaries = []
    expected_configs = []
    _model_cpu_offload_seq = None

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

    @property
    def model_cpu_offload_seq(self):
        """
        adjust the model_cpu_offload_seq to reflect actual components loaded in the block
        """

        model_cpu_offload_seq = []
        block_component_names = [k for k, v in self.components.items() if isinstance(v, torch.nn.Module)]
        if len(block_component_names) == 0:
            return None
        if len(block_component_names) == 1:
            return block_component_names[0]
        else:
            if self._model_cpu_offload_seq is None:
                raise ValueError(
                    f"Block {self.__class__.__name__} has multiple components but no model_cpu_offload_seq specified"
                )
            model_cpu_offload_seq = [m for m in self._model_cpu_offload_seq.split("->") if m in block_component_names]
            remaining = [m for m in block_component_names if m not in model_cpu_offload_seq]
            if remaining:
                logger.warning(
                    f"Block {self.__class__.__name__} has components {remaining} that are not in model_cpu_offload_seq {self._model_cpu_offload_seq}"
                )
            return "->".join(model_cpu_offload_seq)

    def update_states(self, **kwargs):
        """
        Update components and configs after instance creation. Auxiliaries (e.g. image_processor) should be defined for
        each pipeline block, does not need to be updated by users. Logs if existing non-None states are being
        overwritten.

        Args:
            **kwargs: Keyword arguments containing components, or configs to add/update.
            e.g. pipeline_block.update_states(unet=unet1, vae=None)
        """
        # Add expected components
        for component_name in self.expected_components:
            if component_name in kwargs:
                if component_name in self.components and self.components[component_name] is not None:
                    if id(self.components[component_name]) != id(kwargs[component_name]):
                        logger.info(
                            f"Overwriting existing component '{component_name}' "
                            f"(type: {type(self.components[component_name]).__name__}) "
                            f"with new value (type: {type(kwargs[component_name]).__name__})"
                        )
                self.components[component_name] = kwargs.pop(component_name)

        # Add expected configs
        for config_name in self.expected_configs:
            if config_name in kwargs:
                if config_name in self.configs and self.configs[config_name] is not None:
                    if self.configs[config_name] != kwargs[config_name]:
                        logger.info(
                            f"Overwriting existing config '{config_name}' "
                            f"(value: {self.configs[config_name]}) "
                            f"with new value ({kwargs[config_name]})"
                        )
                self.configs[config_name] = kwargs.pop(config_name)

    def __init__(self, **kwargs):
        self.components: Dict[str, Any] = {}
        self.auxiliaries: Dict[str, Any] = {}
        self.configs: Dict[str, Any] = {}

        self.update_states(**kwargs)

    # YiYi notes, does pipeline block need "states"? it is not going to be used on its own
    # TODO: address existing components -> overwrite or not? currently overwrite
    def add_states_from_pipe(self, pipe: DiffusionPipeline, **kwargs):
        """
        add components/auxiliaries/configs from a diffusion pipeline object.

        Args:
            pipe: A `[DiffusionPipeline]` object.
            **kwargs: Additional states to update, these take precedence over pipe values.

        Returns:
            PipelineBlock: An instance loaded with the pipeline's components and configurations.
        """
        states_to_update = {}

        # Get components - prefer kwargs over pipe values
        for component_name in self.expected_components:
            if component_name in kwargs:
                states_to_update[component_name] = kwargs.pop(component_name)
            elif component_name in pipe.components:
                states_to_update[component_name] = pipe.components[component_name]

        # Get configs - prefer kwargs over pipe values
        pipe_config = dict(pipe.config)
        for config_name in self.expected_configs:
            if config_name in kwargs:
                states_to_update[config_name] = kwargs.pop(config_name)
            elif config_name in pipe_config:
                states_to_update[config_name] = pipe_config[config_name]

        # Update all states at once
        self.update_states(**states_to_update)

    @validate_hf_hub_args
    def add_states_from_pretrained(self, pretrained_model_or_path, **kwargs):
        base_pipeline = DiffusionPipeline.from_pretrained(pretrained_model_or_path, **kwargs)
        self.add_states_from_pipe(base_pipeline, **kwargs)

    def __call__(self, pipeline, state: PipelineState) -> PipelineState:
        raise NotImplementedError("__call__ method must be implemented in subclasses")

    def __repr__(self):
        class_name = self.__class__.__name__

        # Components section
        expected_components = set(getattr(self, "expected_components", []))
        loaded_components = set(self.components.keys())
        all_components = sorted(expected_components | loaded_components)
        components = ", ".join(
            f"{k}={type(self.components[k]).__name__}" if k in loaded_components else f"{k}" for k in all_components
        )

        # Auxiliaries section
        expected_auxiliaries = set(getattr(self, "expected_auxiliaries", []))
        loaded_auxiliaries = set(self.auxiliaries.keys())
        all_auxiliaries = sorted(expected_auxiliaries | loaded_auxiliaries)
        auxiliaries = ", ".join(
            f"{k}={type(self.auxiliaries[k]).__name__}" if k in loaded_auxiliaries else f"{k}" for k in all_auxiliaries
        )

        # Configs section
        expected_configs = set(getattr(self, "expected_configs", []))
        loaded_configs = set(self.configs.keys())
        all_configs = sorted(expected_configs | loaded_configs)
        configs = ", ".join(f"{k}={self.configs[k]}" if k in loaded_configs else f"{k}" for k in all_configs)

        # Single block shows itself
        blocks = f"step={self.__class__.__name__}"

        # Other information
        inputs = ", ".join(f"{name}={default}" for name, default in self.inputs)
        intermediates_inputs = ", ".join(self.intermediates_inputs)
        intermediates_outputs = ", ".join(self.intermediates_outputs)

        return (
            f"{class_name}(\n"
            f"  components: {components}\n"
            f"  auxiliaries: {auxiliaries}\n"
            f"  configs: {configs}\n"
            f"  blocks: {blocks}\n"
            f"  inputs: {inputs}\n"
            f"  intermediates_inputs: {intermediates_inputs}\n"
            f"  intermediates_outputs: {intermediates_outputs}\n"
            f")"
        )


def combine_inputs(*input_lists: List[Tuple[str, Any]]) -> List[Tuple[str, Any]]:
    """
    Combines multiple lists of (name, default_value) tuples. For duplicate inputs, updates only if current value is
    None and new value is not None. Warns if multiple non-None default values exist for the same input.
    """
    combined_dict = {}
    for inputs in input_lists:
        for name, value in inputs:
            if name in combined_dict:
                current_value = combined_dict[name]
                if current_value is not None and value is not None and current_value != value:
                    warnings.warn(
                        f"Multiple different default values found for input '{name}': "
                        f"{current_value} and {value}. Using {current_value}."
                    )
                if current_value is None and value is not None:
                    combined_dict[name] = value
            else:
                combined_dict[name] = value
    return list(combined_dict.items())


class MultiPipelineBlocks:
    """
    A class that combines multiple pipeline block classes into one. When used, it has same API and properties as
    PipelineBlock. And it can be used in ModularPipelineBuilder as a single pipeline block.
    """

    block_classes = []
    block_prefixes = []
    _model_cpu_offload_seq = None

    @property
    def expected_components(self):
        expected_components = []
        for block in self.blocks.values():
            for component in block.expected_components:
                if component not in expected_components:
                    expected_components.append(component)
        return expected_components

    @property
    def expected_auxiliaries(self):
        expected_auxiliaries = []
        for block in self.blocks.values():
            for auxiliary in block.expected_auxiliaries:
                if auxiliary not in expected_auxiliaries:
                    expected_auxiliaries.append(auxiliary)
        return expected_auxiliaries

    @property
    def expected_configs(self):
        expected_configs = []
        for block in self.blocks.values():
            for config in block.expected_configs:
                if config not in expected_configs:
                    expected_configs.append(config)
        return expected_configs

    def __init__(self, **kwargs):
        blocks = OrderedDict()
        for block_prefix, block_cls in zip(self.block_prefixes, self.block_classes):
            block_name = f"{block_prefix}_step" if block_prefix != "" else "step"
            blocks[block_name] = block_cls(**kwargs)
        self.blocks = blocks

    # YiYi TODO: address the case where multiple blocks have the same component/auxiliary/config; give out warning etc
    @property
    def components(self):
        # Combine components from all blocks
        components = {}
        for block_name, block in self.blocks.items():
            components.update(block.components)
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
        raise NotImplementedError("inputs property must be implemented in subclasses")

    @property
    def intermediates_inputs(self) -> List[str]:
        raise NotImplementedError("intermediates_inputs property must be implemented in subclasses")

    @property
    def intermediates_outputs(self) -> List[str]:
        raise NotImplementedError("intermediates_outputs property must be implemented in subclasses")

    @property
    def model_cpu_offload_seq(self):
        raise NotImplementedError("model_cpu_offload_seq property must be implemented in subclasses")

    def __call__(self, pipeline, state):
        raise NotImplementedError("__call__ method must be implemented in subclasses")

    def update_states(self, **kwargs):
        """
        Update states for each block with support for block-specific kwargs.

        Args:
            **kwargs: Can include both general kwargs (e.g., 'unet') and
                     block-specific kwargs (e.g., 'img2img_step_unet')

        Example:
            pipeline.update_states(
                img2img_step_unet=unet2, # Only for img2img_step step_unet=unet1, # Only for step vae=vae1 # For any
                block that expects vae
            )
        """
        for block_name, block in self.blocks.items():
            # Prepare block-specific kwargs
            if isinstance(block, PipelineBlock):
                block_kwargs = {}

                # Check for block-specific kwargs first (e.g., 'img2img_unet')
                prefix = f"{block_name.replace('_step', '')}_"
                for key, value in kwargs.items():
                    if key.startswith(prefix):
                        # Remove prefix and add to block kwargs
                        block_kwargs[key[len(prefix) :]] = value

                # For any expected component/auxiliary/config not found with prefix,
                # fall back to general kwargs
                for name in (
                    block.expected_components
                    +
                    # block.expected_auxiliaries +
                    block.expected_configs
                ):
                    if name not in block_kwargs:
                        if name in kwargs:
                            block_kwargs[name] = kwargs[name]
            elif isinstance(block, MultiPipelineBlocks):
                block_kwargs = kwargs
            else:
                raise ValueError(f"Unsupported block type: {type(block).__name__}")

            # Update the block with its specific kwargs
            block.update_states(**block_kwargs)

    def add_states_from_pipe(self, pipe: DiffusionPipeline, **kwargs):
        """
        Load components from pipe with support for block-specific kwargs.

        Args:
            pipe: DiffusionPipeline object
            **kwargs: Can include both general kwargs (e.g., 'unet') and
                     block-specific kwargs (e.g., 'img2img_unet' for 'img2img_step')
        """
        for block_name, block in self.blocks.items():
            # Handle different block types
            if isinstance(block, PipelineBlock):
                block_kwargs = {}

                # Check for block-specific kwargs first (e.g., 'img2img_unet')
                prefix = f"{block_name.replace('_step', '')}_"
                for key, value in kwargs.items():
                    if key.startswith(prefix):
                        # Remove prefix and add to block kwargs
                        block_kwargs[key[len(prefix) :]] = value

                # For any expected component/auxiliary/config not found with prefix,
                # fall back to general kwargs
                for name in (
                    block.expected_components
                    +
                    # block.expected_auxiliaries +
                    block.expected_configs
                ):
                    if name not in block_kwargs:
                        if name in kwargs:
                            block_kwargs[name] = kwargs[name]
            elif isinstance(block, MultiPipelineBlocks):
                block_kwargs = kwargs
            else:
                raise ValueError(f"Unsupported block type: {type(block).__name__}")

            # Load the block with its specific kwargs
            block.add_states_from_pipe(pipe, **block_kwargs)

    def add_states_from_pretrained(self, pretrained_model_or_path, **kwargs):
        base_pipeline = DiffusionPipeline.from_pretrained(pretrained_model_or_path, **kwargs)
        self.add_states_from_pipe(base_pipeline, **kwargs)

    def __repr__(self):
        class_name = self.__class__.__name__

        # Components section
        expected_components = set(getattr(self, "expected_components", []))
        loaded_components = set(self.components.keys())
        all_components = sorted(expected_components | loaded_components)
        components_str = "  Components:\n" + "\n".join(
            f"    - {k}={type(self.components[k]).__name__}" if k in loaded_components else f"    - {k}"
            for k in all_components
        )

        # Auxiliaries section
        expected_auxiliaries = set(getattr(self, "expected_auxiliaries", []))
        loaded_auxiliaries = set(self.auxiliaries.keys())
        all_auxiliaries = sorted(expected_auxiliaries | loaded_auxiliaries)
        auxiliaries_str = "  Auxiliaries:\n" + "\n".join(
            f"    - {k}={type(self.auxiliaries[k]).__name__}" if k in loaded_auxiliaries else f"    - {k}"
            for k in all_auxiliaries
        )

        # Configs section
        expected_configs = set(getattr(self, "expected_configs", []))
        loaded_configs = set(self.configs.keys())
        all_configs = sorted(expected_configs | loaded_configs)
        configs_str = "  Configs:\n" + "\n".join(
            f"    - {k}={self.configs[k]}" if k in loaded_configs else f"    - {k}" for k in all_configs
        )

        # Blocks section
        blocks_str = "  Blocks:\n" + "\n".join(
            f"    - {name}={block.__class__.__name__}" for name, block in self.blocks.items()
        )

        # Other information
        inputs_str = "  Inputs:\n" + "\n".join(f"    - {name}={default}" for name, default in self.inputs)

        intermediates_str = (
            "  Intermediates:\n"
            f"    - inputs: {', '.join(self.intermediates_inputs)}\n"
            f"    - outputs: {', '.join(self.intermediates_outputs)}"
        )

        return (
            f"{class_name}(\n"
            f"{components_str}\n"
            f"{auxiliaries_str}\n"
            f"{configs_str}\n"
            f"{blocks_str}\n"
            f"{inputs_str}\n"
            f"{intermediates_str}\n"
            f")"
        )


class AutoPipelineBlocks(MultiPipelineBlocks):
    """
    A class that automatically selects which block to run based on trigger inputs.

    Attributes:
        block_classes: List of block classes to be used
        block_prefixes: List of prefixes for each block
        block_trigger_inputs: List of input names that trigger specific blocks, with None for default
    """

    block_classes = []
    block_prefixes = []
    block_trigger_inputs = []

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__post_init__()

    def __post_init__(self):
        """
        Create mapping of trigger inputs directly to block objects. Validates that there is at most one default block
        (None trigger).
        """
        # Check for at most one default block
        default_blocks = [t for t in self.block_trigger_inputs if t is None]
        if len(default_blocks) > 1:
            raise ValueError(
                f"Multiple default blocks specified in {self.__class__.__name__}. "
                "Must include at most one None in block_trigger_inputs."
            )

        # Map trigger inputs to block objects
        self.trigger_to_block_map = dict(zip(self.block_trigger_inputs, self.blocks.values()))

    @property
    def inputs(self) -> List[Tuple[str, Any]]:
        return combine_inputs(*(block.inputs for block in self.blocks.values()))

    @property
    def intermediates_inputs(self) -> List[str]:
        return list(set().union(*(block.intermediates_inputs for block in self.blocks.values())))

    @property
    def intermediates_outputs(self) -> List[str]:
        return list(set().union(*(block.intermediates_outputs for block in self.blocks.values())))

    @torch.no_grad()
    def __call__(self, pipeline, state: PipelineState) -> PipelineState:
        # Find default block first (if any)
        default_block = self.trigger_to_block_map.get(None)

        # Check which trigger inputs are present
        active_triggers = [
            input_name
            for input_name in self.block_trigger_inputs
            if input_name is not None and state.get_input(input_name) is not None
        ]

        # If multiple triggers are active, raise error
        if len(active_triggers) > 1:
            trigger_names = [f"'{t}'" for t in active_triggers]
            raise ValueError(
                f"Multiple trigger inputs found ({', '.join(trigger_names)}). "
                f"Only one trigger input can be provided for {self.__class__.__name__}."
            )

        # Get the block to run (use default if no triggers active)
        block = self.trigger_to_block_map.get(active_triggers[0]) if active_triggers else default_block
        if block is None:
            logger.warning(f"No valid block found in {self.__class__.__name__}, skipping.")
            return pipeline, state

        try:
            return block(pipeline, state)
        except Exception as e:
            error_msg = (
                f"\nError in block: {block.__class__.__name__}\n"
                f"Error details: {str(e)}\n"
                f"Traceback:\n{traceback.format_exc()}"
            )
            logger.error(error_msg)
            raise

    @property
    def model_cpu_offload_seq(self):
        default_block = self.trigger_to_block_map.get(None)

        return default_block.model_cpu_offload_seq

    def __repr__(self):
        class_name = self.__class__.__name__

        # Components section
        expected_components = set(getattr(self, "expected_components", []))
        loaded_components = set(self.components.keys())
        all_components = sorted(expected_components | loaded_components)
        components_str = "  Components:\n" + "\n".join(
            f"    - {k}={type(self.components[k]).__name__}" if k in loaded_components else f"    - {k}"
            for k in all_components
        )

        # Auxiliaries section
        expected_auxiliaries = set(getattr(self, "expected_auxiliaries", []))
        loaded_auxiliaries = set(self.auxiliaries.keys())
        all_auxiliaries = sorted(expected_auxiliaries | loaded_auxiliaries)
        auxiliaries_str = "  Auxiliaries:\n" + "\n".join(
            f"    - {k}={type(self.auxiliaries[k]).__name__}" if k in loaded_auxiliaries else f"    - {k}"
            for k in all_auxiliaries
        )

        # Configs section
        expected_configs = set(getattr(self, "expected_configs", []))
        loaded_configs = set(self.configs.keys())
        all_configs = sorted(expected_configs | loaded_configs)
        configs_str = "  Configs:\n" + "\n".join(
            f"    - {k}={self.configs[k]}" if k in loaded_configs else f"    - {k}" for k in all_configs
        )

        # Blocks section with trigger information
        blocks_str = "  Blocks:\n"
        for name, block in self.blocks.items():
            # Find trigger for this block
            trigger = next((t for t, b in self.trigger_to_block_map.items() if b == block), None)
            trigger_str = " (default)" if trigger is None else f" (triggered by: {trigger})"

            blocks_str += f"    {name} ({block.__class__.__name__}){trigger_str}\n"

            # Add inputs information
            if hasattr(block, "inputs"):
                inputs_str = ", ".join(f"{name}={default}" for name, default in block.inputs)
                if inputs_str:
                    blocks_str += f"       inputs: {inputs_str}\n"

            # Add intermediates information
            if hasattr(block, "intermediates_inputs") or hasattr(block, "intermediates_outputs"):
                intermediates_str = ""
                if hasattr(block, "intermediates_inputs"):
                    intermediates_str += f"{', '.join(block.intermediates_inputs)}"

                if hasattr(block, "intermediates_outputs"):
                    if intermediates_str:
                        intermediates_str += " -> "
                    intermediates_str += f"{', '.join(block.intermediates_outputs)}"

                if intermediates_str:
                    blocks_str += f"       intermediates: {intermediates_str}\n"
            blocks_str += "\n"
        intermediates_str = (
            "\n    Intermediates:\n"
            f"      - inputs: {', '.join(self.intermediates_inputs)}\n"
            f"      - outputs: {', '.join(self.intermediates_outputs)}"
        )

        return (
            f"{class_name}(\n"
            f"{components_str}\n"
            f"{auxiliaries_str}\n"
            f"{configs_str}\n"
            f"{blocks_str}\n"
            f"{intermediates_str}\n"
            f")"
        )


class SequentialPipelineBlocks(MultiPipelineBlocks):
    """
    A class that combines multiple pipeline block classes into one. When called, it will call each block in sequence.
    """

    @property
    def inputs(self) -> List[Tuple[str, Any]]:
        return combine_inputs(*(block.inputs for block in self.blocks.values()))

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

    @property
    def model_cpu_offload_seq(self):
        model_cpu_offload_seq = []

        for block_name, block in self.blocks.items():
            block_components = [k for k, v in block.components.items() if isinstance(v, torch.nn.Module)]
            if len(block_components) == 0:
                continue
            if len(block_components) == 1:
                if block_components[0] in model_cpu_offload_seq:
                    model_cpu_offload_seq.remove(block_components[0])
                model_cpu_offload_seq.append(block_components[0])
            else:
                if block.model_cpu_offload_seq is None:
                    raise ValueError(
                        f"Block {block_name}:{block.__class__.__name__} has multiple components {block_components} but no model_cpu_offload_seq specified"
                    )
                for model_str in block.model_cpu_offload_seq.split("->"):
                    if model_str in block_components:
                        # if it is already in the list,remove previous occurence and add to the end
                        if model_str in model_cpu_offload_seq:
                            model_cpu_offload_seq.remove(model_str)
                        model_cpu_offload_seq.append(model_str)
                        block_components.remove(model_str)
                if len(block_components) > 0:
                    logger.warning(
                        f"Block {block_name}:{block.__class__.__name__} has components {block_components} that are not in model_cpu_offload_seq {block.model_cpu_offload_seq}"
                    )

        return "->".join(model_cpu_offload_seq)

    def __repr__(self):
        class_name = self.__class__.__name__

        # Components section
        expected_components = set(getattr(self, "expected_components", []))
        loaded_components = set(self.components.keys())
        all_components = sorted(expected_components | loaded_components)
        components_str = "  Components:\n" + "\n".join(
            f"    - {k}={type(self.components[k]).__name__}" if k in loaded_components else f"    - {k}"
            for k in all_components
        )

        # Auxiliaries section
        expected_auxiliaries = set(getattr(self, "expected_auxiliaries", []))
        loaded_auxiliaries = set(self.auxiliaries.keys())
        all_auxiliaries = sorted(expected_auxiliaries | loaded_auxiliaries)
        auxiliaries_str = "  Auxiliaries:\n" + "\n".join(
            f"    - {k}={type(self.auxiliaries[k]).__name__}" if k in loaded_auxiliaries else f"    - {k}"
            for k in all_auxiliaries
        )

        # Configs section
        expected_configs = set(getattr(self, "expected_configs", []))
        loaded_configs = set(self.configs.keys())
        all_configs = sorted(expected_configs | loaded_configs)
        configs_str = "  Configs:\n" + "\n".join(
            f"    - {k}={self.configs[k]}" if k in loaded_configs else f"    - {k}" for k in all_configs
        )

        # Detailed blocks section with data flow
        blocks_str = "  Blocks:\n"
        for i, (name, block) in enumerate(self.blocks.items()):
            blocks_str += f"    {i}. {name} ({block.__class__.__name__})\n"

            # Add inputs information
            if hasattr(block, "inputs"):
                inputs_str = ", ".join(f"{name}={default}" for name, default in block.inputs)
                blocks_str += f"       inputs: {inputs_str}\n"

            # Add intermediates information
            if hasattr(block, "intermediates_inputs") or hasattr(block, "intermediates_outputs"):
                intermediates_str = ""
                if hasattr(block, "intermediates_inputs"):
                    intermediates_str += f"{', '.join(block.intermediates_inputs)}"

                if hasattr(block, "intermediates_outputs"):
                    if intermediates_str:
                        intermediates_str += " -> "
                    intermediates_str += f"{', '.join(block.intermediates_outputs)}"

                if intermediates_str:
                    blocks_str += f"       intermediates: {intermediates_str}\n"
            blocks_str += "\n"

        intermediates_str = (
            "\n    Intermediates:\n"
            f"      - inputs: {', '.join(self.intermediates_inputs)}\n"
            f"      - outputs: {', '.join(self.intermediates_outputs)}"
        )

        return (
            f"{class_name}(\n"
            f"{components_str}\n"
            f"{auxiliaries_str}\n"
            f"{configs_str}\n"
            f"{blocks_str}\n"
            f"{intermediates_str}\n"
            f")"
        )


class ModularPipelineBuilder(ConfigMixin):
    """
    Base class for all Modular pipelines.

    """

    config_name = "model_index.json"
    model_cpu_offload_seq = None
    hf_device_map = None
    _exclude_from_cpu_offload = []
    default_pipeline_blocks = []

    def __init__(self):
        super().__init__()
        self.register_to_config()
        self.pipeline_blocks = []

    # Copied from diffusers.pipelines.pipeline_utils.DiffusionPipeline.register_modules
    def register_modules(self, **kwargs):
        for name, module in kwargs.items():
            # retrieve library
            if module is None or isinstance(module, (tuple, list)) and module[0] is None:
                register_dict = {name: (None, None)}
            else:
                library, class_name = _fetch_class_library_tuple(module)
                register_dict = {name: (library, class_name)}

            # save model index config
            self.register_to_config(**register_dict)

            # set models
            setattr(self, name, module)

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
        r"""
        The `self.components` property returns all modules needed to initialize the pipeline, as defined by the
        pipeline blocks.

        Returns (`dict`):
            A dictionary containing all the components defined in the pipeline blocks.
        """

        expected_components = set()
        for block in self.pipeline_blocks:
            expected_components.update(block.components.keys())

        components = {}
        for name in expected_components:
            if hasattr(self, name):
                components[name] = getattr(self, name)

        return components

    @property
    def auxiliaries(self) -> Dict[str, Any]:
        r"""
        The `self.auxiliaries` property returns all auxiliaries needed to initialize the pipeline, as defined by the
        pipeline blocks.

        Returns (`dict`):
            A dictionary containing all the auxiliaries defined in the pipeline blocks.
        """
        # First collect all expected auxiliary names from blocks
        expected_auxiliaries = set()
        for block in self.pipeline_blocks:
            expected_auxiliaries.update(block.auxiliaries.keys())

        # Then fetch the actual auxiliaries from the pipeline
        auxiliaries = {}
        for name in expected_auxiliaries:
            if hasattr(self, name):
                auxiliaries[name] = getattr(self, name)

        return auxiliaries

    @property
    def configs(self) -> Dict[str, Any]:
        r"""
        The `self.configs` property returns all configs needed to initialize the pipeline, as defined by the pipeline
        blocks.

        Returns (`dict`):
            A dictionary containing all the configs defined in the pipeline blocks.
        """
        # First collect all expected config names from blocks
        expected_configs = set()
        for block in self.pipeline_blocks:
            expected_configs.update(block.configs.keys())

        # Then fetch the actual configs from the pipeline's config
        configs = {}
        for name in expected_configs:
            if name in self.config:
                configs[name] = self.config[name]

        return configs

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

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("__call__ is not implemented for ModularPipelineBuilder")

    # YiYi Notes: do we need to support multiple blocks?
    def remove_blocks(self, indices: Union[int, List[int]]):
        """
        Remove one or more blocks from the pipeline by their indices and clean up associated components, configs, and
        auxiliaries that are no longer needed by remaining blocks.

        Args:
            indices (Union[int, List[int]]): The index or list of indices of blocks to remove
        """
        # Convert single index to list
        indices = [indices] if isinstance(indices, int) else indices

        # Validate indices
        for idx in indices:
            if not 0 <= idx < len(self.pipeline_blocks):
                raise ValueError(
                    f"Invalid block index {idx}. Index must be between 0 and {len(self.pipeline_blocks) - 1}"
                )

        # Sort indices in descending order to avoid shifting issues when removing
        indices = sorted(indices, reverse=True)

        # Store blocks to be removed
        blocks_to_remove = [self.pipeline_blocks[idx] for idx in indices]

        # Remove blocks from pipeline
        for idx in indices:
            self.pipeline_blocks.pop(idx)

        # Consolidate items to remove from all blocks
        components_to_remove = {k: v for block in blocks_to_remove for k, v in block.components.items()}
        auxiliaries_to_remove = {k: v for block in blocks_to_remove for k, v in block.auxiliaries.items()}
        configs_to_remove = {k: v for block in blocks_to_remove for k, v in block.configs.items()}

        # The properties will now reflect only the remaining blocks
        remaining_components = self.components
        remaining_auxiliaries = self.auxiliaries
        remaining_configs = self.configs

        # Clean up all items that are no longer needed
        for component_name in components_to_remove:
            if component_name not in remaining_components:
                if component_name in self.config:
                    del self.config[component_name]
                if hasattr(self, component_name):
                    delattr(self, component_name)

        for auxiliary_name in auxiliaries_to_remove:
            if auxiliary_name not in remaining_auxiliaries:
                if hasattr(self, auxiliary_name):
                    delattr(self, auxiliary_name)

        for config_name in configs_to_remove:
            if config_name not in remaining_configs:
                if config_name in self.config:
                    del self.config[config_name]

    # YiYi Notes: I left all the functionalities to support adding multiple blocks
    # but I wonder if it is still needed now we have `SequentialBlocks` and user can always combine them into one before adding to the builder
    def add_blocks(self, pipeline_blocks, at: int = -1):
        """Add blocks to the pipeline.

        Args:
            pipeline_blocks: A single PipelineBlock instance or a list of PipelineBlock instances.
            at (int, optional): Index at which to insert the blocks. Defaults to -1 (append at end).
        """
        # Convert single block to list for uniform processing
        if not isinstance(pipeline_blocks, (list, tuple)):
            pipeline_blocks = [pipeline_blocks]

        # Validate insert_at index
        if at != -1 and not 0 <= at <= len(self.pipeline_blocks):
            raise ValueError(f"Invalid at index {at}. Index must be between 0 and {len(self.pipeline_blocks)}")

        # Consolidate all items from blocks
        components_to_add = {}
        configs_to_add = {}
        auxiliaries_to_add = {}

        # Add blocks in order
        for i, block in enumerate(pipeline_blocks):
            # Add block to pipeline at specified position
            if at == -1:
                self.pipeline_blocks.append(block)
            else:
                self.pipeline_blocks.insert(at + i, block)

            # Collect components that don't already exist
            for k, v in block.components.items():
                if not hasattr(self, k) or (getattr(self, k, None) is None and v is not None):
                    components_to_add[k] = v

            # Collect configs and auxiliaries
            configs_to_add.update(block.configs)
            auxiliaries_to_add.update(block.auxiliaries)

        # Process all items in batches
        if components_to_add:
            self.register_modules(**components_to_add)
        if configs_to_add:
            self.register_to_config(**configs_to_add)
        for key, value in auxiliaries_to_add.items():
            setattr(self, key, value)

    def replace_blocks(self, pipeline_blocks, at: int):
        """Replace one or more blocks in the pipeline at the specified index.

        Args:
            pipeline_blocks: A single PipelineBlock instance or a list of PipelineBlock instances
                that will replace existing blocks.
            at (int): Index at which to replace the blocks.
        """
        # Convert single block to list for uniform processing
        if not isinstance(pipeline_blocks, (list, tuple)):
            pipeline_blocks = [pipeline_blocks]

        # Validate replace_at index
        if not 0 <= at < len(self.pipeline_blocks):
            raise ValueError(f"Invalid at index {at}. Index must be between 0 and {len(self.pipeline_blocks) - 1}")

        # Add new blocks first
        self.add_blocks(pipeline_blocks, at=at)

        # Calculate indices to remove
        # We need to remove the original blocks that are now shifted by the length of pipeline_blocks
        indices_to_remove = list(range(at + len(pipeline_blocks), at + len(pipeline_blocks) * 2))

        # Remove the old blocks
        self.remove_blocks(indices_to_remove)

    def run_blocks(self, state: PipelineState = None, output: Union[str, List[str]] = None, **kwargs):
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
                if name not in self.pipeline_blocks[0].intermediates_inputs:
                    state.add_input(name, input_params.pop(name))
                else:
                    state.add_input(name, input_params[name])
            elif name not in state.inputs:
                state.add_input(name, default)

        for name in self.pipeline_blocks[0].intermediates_inputs:
            if name in input_params:
                state.add_intermediate(name, input_params.pop(name))

        # Warn about unexpected inputs
        if len(input_params) > 0:
            logger.warning(f"Unexpected input '{input_params.keys()}' provided. This input will be ignored.")
        # Run the pipeline
        with torch.no_grad():
            for block in self.pipeline_blocks:
                try:
                    pipeline, state = block(self, state)
                except Exception:
                    error_msg = f"Error in block: ({block.__class__.__name__}):\n"
                    logger.error(error_msg)
                    raise
            self.maybe_free_model_hooks()

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

    def run_pipeline(self, **kwargs):
        state = PipelineState()

        # Make a copy of the input kwargs
        input_params = kwargs.copy()

        default_params = self.default_call_parameters

        # Add inputs to state, using defaults if not provided
        for name, default in default_params.items():
            if name in input_params:
                state.add_input(name, input_params.pop(name))
            else:
                state.add_input(name, default)

        # Warn about unexpected inputs
        if len(input_params) > 0:
            logger.warning(f"Unexpected input '{input_params.keys()}' provided. This input will be ignored.")

        # Run the pipeline
        with torch.no_grad():
            for block in self.pipeline_blocks:
                try:
                    pipeline, state = block(self, state)
                except Exception as e:
                    error_msg = (
                        f"\nError in block: ({block.__class__.__name__}):\n"
                        f"Error details: {str(e)}\n"
                        f"Stack trace:\n{traceback.format_exc()}"
                    )
                    logger.error(error_msg)
                    raise
            self.maybe_free_model_hooks()

        return state.get_output("images")

    @property
    def default_call_parameters(self) -> Dict[str, Any]:
        params = {}
        for block in self.pipeline_blocks:
            for name, default in block.inputs:
                if name not in params:
                    params[name] = default
        return params

    def __repr__(self):
        output = "CustomPipeline Configuration:\n"
        output += "==============================\n\n"

        # List the blocks used to build the pipeline
        output += "Pipeline Blocks:\n"
        output += "----------------\n"
        for i, block in enumerate(self.pipeline_blocks):
            if isinstance(block, MultiPipelineBlocks):
                output += f"{i}. {block.__class__.__name__} - (CPU offload seq: {block.model_cpu_offload_seq})\n"
                # Add sub-blocks information
                for sub_block_name, sub_block in block.blocks.items():
                    output += f"    • {sub_block_name} ({sub_block.__class__.__name__}) \n"
            else:
                output += f"{i}. {block.__class__.__name__} - (CPU offload seq: {block.model_cpu_offload_seq})\n"
            output += "\n"

            intermediates_str = ""
            if hasattr(block, "intermediates_inputs"):
                intermediates_str += f"{', '.join(block.intermediates_inputs)}"

            if hasattr(block, "intermediates_outputs"):
                if intermediates_str:
                    intermediates_str += " -> "
                else:
                    intermediates_str += "-> "
                intermediates_str += f"{', '.join(block.intermediates_outputs)}"

            if intermediates_str:
                output += f"   {intermediates_str}\n"

            output += "\n"
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

        # List the auxiliaries registered in the pipeline
        output += "Registered Auxiliaries:\n"
        output += "----------------------\n"
        for name, auxiliary in self.auxiliaries.items():
            output += f"{name}: {type(auxiliary).__name__}\n"
        output += "\n"

        # List the configs registered in the pipeline
        output += "Registered Configs:\n"
        output += "------------------\n"
        for name, config in self.configs.items():
            output += f"{name}: {config!r}\n"
        output += "\n"

        # List the default call parameters
        output += "Default Call Parameters:\n"
        output += "------------------------\n"
        params = self.default_call_parameters
        for name, default in params.items():
            output += f"{name}: {default!r}\n"

        # Add a section for required call parameters:
        # intermediate inputs for the first block
        output += "\nRequired Call Parameters:\n"
        output += "--------------------------\n"
        for name in self.pipeline_blocks[0].intermediates_inputs:
            output += f"{name}: \n"
            params[name] = ""

        output += "\nNote: These are the default values. Actual values may be different when running the pipeline."
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

        is_pipeline_device_mapped = self.hf_device_map is not None and len(self.hf_device_map) > 1
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

    def remove_all_hooks(self):
        for _, model in self.components.items():
            if isinstance(model, torch.nn.Module) and hasattr(model, "_hf_hook"):
                accelerate.hooks.remove_hook_from_module(model, recurse=True)
        self._all_hooks = []

    def find_model_sequence(self):
        pass

    # YiYi notes: assume there is only one pipeline block now (still debating if we want to support multiple pipeline blocks)
    @property
    def model_cpu_offload_seq(self):
        return self.pipeline_blocks[0].model_cpu_offload_seq

    def enable_model_cpu_offload(
        self,
        gpu_id: Optional[int] = None,
        device: Union[torch.device, str] = "cuda",
        model_cpu_offload_seq: Optional[str] = None,
    ):
        r"""
        Offloads all models to CPU using accelerate, reducing memory usage with a low impact on performance. Compared
        to `enable_sequential_cpu_offload`, this method moves one whole model at a time to the GPU when its `forward`
        method is called, and the model remains in GPU until the next model runs. Memory savings are lower than with
        `enable_sequential_cpu_offload`, but performance is much better due to the iterative execution of the `unet`.

        Arguments:
            gpu_id (`int`, *optional*):
                The ID of the accelerator that shall be used in inference. If not specified, it will default to 0.
            device (`torch.Device` or `str`, *optional*, defaults to "cuda"):
                The PyTorch device type of the accelerator that shall be used in inference. If not specified, it will
                default to "cuda".
        """
        _exclude_from_cpu_offload = []  # YiYi Notes: this is not used (keep the variable for now)
        is_pipeline_device_mapped = self.hf_device_map is not None and len(self.hf_device_map) > 1
        if is_pipeline_device_mapped:
            raise ValueError(
                "It seems like you have activated a device mapping strategy on the pipeline so calling `enable_model_cpu_offload() isn't allowed. You can call `reset_device_map()` first and then call `enable_model_cpu_offload()`."
            )

        model_cpu_offload_seq = model_cpu_offload_seq or self.model_cpu_offload_seq
        self._model_cpu_offload_seq_used = model_cpu_offload_seq
        if model_cpu_offload_seq is None:
            raise ValueError(
                "Model CPU offload cannot be enabled because no `model_cpu_offload_seq` class attribute is set or passed."
            )

        if is_accelerate_available() and is_accelerate_version(">=", "0.17.0.dev0"):
            from accelerate import cpu_offload_with_hook
        else:
            raise ImportError("`enable_model_cpu_offload` requires `accelerate v0.17.0` or higher.")

        self.remove_all_hooks()

        torch_device = torch.device(device)
        device_index = torch_device.index

        if gpu_id is not None and device_index is not None:
            raise ValueError(
                f"You have passed both `gpu_id`={gpu_id} and an index as part of the passed device `device`={device}"
                f"Cannot pass both. Please make sure to either not define `gpu_id` or not pass the index as part of the device: `device`={torch_device.type}"
            )

        # _offload_gpu_id should be set to passed gpu_id (or id in passed `device`) or default to previously set id or default to 0
        self._offload_gpu_id = gpu_id or torch_device.index or getattr(self, "_offload_gpu_id", 0)

        device_type = torch_device.type
        device = torch.device(f"{device_type}:{self._offload_gpu_id}")
        self._offload_device = device

        self.to("cpu", silence_dtype_warnings=True)
        device_mod = getattr(torch, device.type, None)
        if hasattr(device_mod, "empty_cache") and device_mod.is_available():
            device_mod.empty_cache()  # otherwise we don't see the memory savings (but they probably exist)

        all_model_components = {k: v for k, v in self.components.items() if isinstance(v, torch.nn.Module)}

        self._all_hooks = []
        hook = None
        for model_str in model_cpu_offload_seq.split("->"):
            model = all_model_components.pop(model_str, None)
            if not isinstance(model, torch.nn.Module):
                continue

            _, hook = cpu_offload_with_hook(model, device, prev_module_hook=hook)
            self._all_hooks.append(hook)

        # CPU offload models that are not in the seq chain unless they are explicitly excluded
        # these models will stay on CPU until maybe_free_model_hooks is called
        # some models cannot be in the seq chain because they are iteratively called, such as controlnet
        for name, model in all_model_components.items():
            if not isinstance(model, torch.nn.Module):
                continue

            if name in _exclude_from_cpu_offload:
                model.to(device)
            else:
                _, hook = cpu_offload_with_hook(model, device)
                self._all_hooks.append(hook)

    def maybe_free_model_hooks(self):
        r"""
        Function that offloads all components, removes all model hooks that were added when using
        `enable_model_cpu_offload` and then applies them again. In case the model has not been offloaded this function
        is a no-op. Make sure to add this function to the end of the `__call__` function of your pipeline so that it
        functions correctly when applying enable_model_cpu_offload.
        """
        if not hasattr(self, "_all_hooks") or len(self._all_hooks) == 0:
            # `enable_model_cpu_offload` has not be called, so silently do nothing
            return

        # make sure the model is in the same state as before calling it
        self.enable_model_cpu_offload(
            device=getattr(self, "_offload_device", "cuda"),
            model_cpu_offload_seq=getattr(self, "_model_cpu_offload_seq_used", None),
        )
