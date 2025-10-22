# Copyright 2025 The HuggingFace Team. All rights reserved.
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
import importlib
import inspect
import os
import traceback
import warnings
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from huggingface_hub import create_repo
from huggingface_hub.utils import validate_hf_hub_args
from tqdm.auto import tqdm
from typing_extensions import Self

from ..configuration_utils import ConfigMixin, FrozenDict
from ..pipelines.pipeline_loading_utils import _fetch_class_library_tuple, simple_get_class_obj
from ..utils import PushToHubMixin, is_accelerate_available, logging
from ..utils.dynamic_modules_utils import get_class_from_dynamic_module, resolve_trust_remote_code
from ..utils.hub_utils import load_or_create_model_card, populate_model_card
from .components_manager import ComponentsManager
from .modular_pipeline_utils import (
    ComponentSpec,
    ConfigSpec,
    InputParam,
    InsertableDict,
    OutputParam,
    format_components,
    format_configs,
    make_doc_string,
)


if is_accelerate_available():
    import accelerate

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# map regular pipeline to modular pipeline class name
MODULAR_PIPELINE_MAPPING = OrderedDict(
    [
        ("stable-diffusion-xl", "StableDiffusionXLModularPipeline"),
        ("wan", "WanModularPipeline"),
        ("flux", "FluxModularPipeline"),
        ("flux-kontext", "FluxKontextModularPipeline"),
        ("qwenimage", "QwenImageModularPipeline"),
        ("qwenimage-edit", "QwenImageEditModularPipeline"),
        ("qwenimage-edit-plus", "QwenImageEditPlusModularPipeline"),
    ]
)


@dataclass
class PipelineState:
    """
    [`PipelineState`] stores the state of a pipeline. It is used to pass data between pipeline blocks.
    """

    values: Dict[str, Any] = field(default_factory=dict)
    kwargs_mapping: Dict[str, List[str]] = field(default_factory=dict)

    def set(self, key: str, value: Any, kwargs_type: str = None):
        """
        Add a value to the pipeline state.

        Args:
            key (str): The key for the value
            value (Any): The value to store
            kwargs_type (str): The kwargs_type with which the value is associated
        """
        self.values[key] = value

        if kwargs_type is not None:
            if kwargs_type not in self.kwargs_mapping:
                self.kwargs_mapping[kwargs_type] = [key]
            else:
                self.kwargs_mapping[kwargs_type].append(key)

    def get(self, keys: Union[str, List[str]], default: Any = None) -> Union[Any, Dict[str, Any]]:
        """
        Get one or multiple values from the pipeline state.

        Args:
            keys (Union[str, List[str]]): Key or list of keys for the values
            default (Any): The default value to return if not found

        Returns:
            Union[Any, Dict[str, Any]]: Single value if keys is str, dictionary of values if keys is list
        """
        if isinstance(keys, str):
            return self.values.get(keys, default)
        return {key: self.values.get(key, default) for key in keys}

    def get_by_kwargs(self, kwargs_type: str) -> Dict[str, Any]:
        """
        Get all values with matching kwargs_type.

        Args:
            kwargs_type (str): The kwargs_type to filter by

        Returns:
            Dict[str, Any]: Dictionary of values with matching kwargs_type
        """
        value_names = self.kwargs_mapping.get(kwargs_type, [])
        return self.get(value_names)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert PipelineState to a dictionary.
        """
        return {**self.__dict__}

    def __getattr__(self, name):
        """
        Allow attribute access to intermediate values. If an attribute is not found in the object, look for it in the
        intermediates dict.
        """
        if name in self.values:
            return self.values[name]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def __repr__(self):
        def format_value(v):
            if hasattr(v, "shape") and hasattr(v, "dtype"):
                return f"Tensor(dtype={v.dtype}, shape={v.shape})"
            elif isinstance(v, list) and len(v) > 0 and hasattr(v[0], "shape") and hasattr(v[0], "dtype"):
                return f"[Tensor(dtype={v[0].dtype}, shape={v[0].shape}), ...]"
            else:
                return repr(v)

        values_str = "\n".join(f"    {k}: {format_value(v)}" for k, v in self.values.items())
        kwargs_mapping_str = "\n".join(f"    {k}: {v}" for k, v in self.kwargs_mapping.items())

        return f"PipelineState(\n  values={{\n{values_str}\n  }},\n  kwargs_mapping={{\n{kwargs_mapping_str}\n  }}\n)"


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
        return dict(self.__dict__.items())

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
                    elif (
                        isinstance(val, list)
                        and len(val) > 0
                        and hasattr(val[0], "shape")
                        and hasattr(val[0], "dtype")
                    ):
                        shapes = [t.shape for t in val]
                        formatted_dict[k] = f"List[{len(val)}] of Tensors with shapes {shapes}"
                    else:
                        formatted_dict[k] = repr(val)
                return formatted_dict

            # Default case
            return repr(v)

        attributes = "\n".join(f"    {k}: {format_value(v)}" for k, v in self.__dict__.items())
        return f"BlockState(\n{attributes}\n)"


class ModularPipelineBlocks(ConfigMixin, PushToHubMixin):
    """
    Base class for all Pipeline Blocks: PipelineBlock, AutoPipelineBlocks, SequentialPipelineBlocks,
    LoopSequentialPipelineBlocks

    [`ModularPipelineBlocks`] provides method to load and save the definition of pipeline blocks.

    > [!WARNING] > This is an experimental feature and is likely to change in the future.
    """

    config_name = "modular_config.json"
    model_name = None

    @classmethod
    def _get_signature_keys(cls, obj):
        parameters = inspect.signature(obj.__init__).parameters
        required_parameters = {k: v for k, v in parameters.items() if v.default == inspect._empty}
        optional_parameters = set({k for k, v in parameters.items() if v.default != inspect._empty})
        expected_modules = set(required_parameters.keys()) - {"self"}

        return expected_modules, optional_parameters

    def __init__(self):
        self.sub_blocks = InsertableDict()

    @property
    def description(self) -> str:
        """Description of the block. Must be implemented by subclasses."""
        return ""

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

    def _get_required_inputs(self):
        input_names = []
        for input_param in self.inputs:
            if input_param.required:
                input_names.append(input_param.name)

        return input_names

    @property
    def required_inputs(self) -> List[InputParam]:
        return self._get_required_inputs()

    @property
    def intermediate_outputs(self) -> List[OutputParam]:
        """List of intermediate output parameters. Must be implemented by subclasses."""
        return []

    def _get_outputs(self):
        return self.intermediate_outputs

    @property
    def outputs(self) -> List[OutputParam]:
        return self._get_outputs()

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        trust_remote_code: bool = False,
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
        if not has_remote_code and trust_remote_code:
            raise ValueError(
                "Selected model repository does not happear to have any custom code or does not have a valid `config.json` file."
            )

        class_ref = config["auto_map"][cls.__name__]
        module_file, class_name = class_ref.split(".")
        module_file = module_file + ".py"
        block_cls = get_class_from_dynamic_module(
            pretrained_model_name_or_path,
            module_file=module_file,
            class_name=class_name,
            **hub_kwargs,
            **kwargs,
        )
        expected_kwargs, optional_kwargs = block_cls._get_signature_keys(block_cls)
        block_kwargs = {
            name: kwargs.pop(name) for name in kwargs if name in expected_kwargs or name in optional_kwargs
        }

        return block_cls(**block_kwargs)

    def save_pretrained(self, save_directory, push_to_hub=False, **kwargs):
        # TODO: factor out this logic.
        cls_name = self.__class__.__name__

        full_mod = type(self).__module__
        module = full_mod.rsplit(".", 1)[-1].replace("__dynamic__", "")
        parent_module = self.save_pretrained.__func__.__qualname__.split(".", 1)[0]
        auto_map = {f"{parent_module}": f"{module}.{cls_name}"}

        self.register_to_config(auto_map=auto_map)
        self.save_config(save_directory=save_directory, push_to_hub=push_to_hub, **kwargs)
        config = dict(self.config)
        self._internal_dict = FrozenDict(config)

    def init_pipeline(
        self,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]] = None,
        components_manager: Optional[ComponentsManager] = None,
        collection: Optional[str] = None,
    ) -> "ModularPipeline":
        """
        create a ModularPipeline, optionally accept modular_repo to load from hub.
        """
        pipeline_class_name = MODULAR_PIPELINE_MAPPING.get(self.model_name, ModularPipeline.__name__)
        diffusers_module = importlib.import_module("diffusers")
        pipeline_class = getattr(diffusers_module, pipeline_class_name)

        modular_pipeline = pipeline_class(
            blocks=deepcopy(self),
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            components_manager=components_manager,
            collection=collection,
        )
        return modular_pipeline

    def get_block_state(self, state: PipelineState) -> dict:
        """Get all inputs and intermediates in one dictionary"""
        data = {}
        state_inputs = self.inputs

        # Check inputs
        for input_param in state_inputs:
            if input_param.name:
                value = state.get(input_param.name)
                if input_param.required and value is None:
                    raise ValueError(f"Required input '{input_param.name}' is missing")
                elif value is not None or (value is None and input_param.name not in data):
                    data[input_param.name] = value

            elif input_param.kwargs_type:
                # if kwargs_type is provided, get all inputs with matching kwargs_type
                if input_param.kwargs_type not in data:
                    data[input_param.kwargs_type] = {}
                inputs_kwargs = state.get_by_kwargs(input_param.kwargs_type)
                if inputs_kwargs:
                    for k, v in inputs_kwargs.items():
                        if v is not None:
                            data[k] = v
                            data[input_param.kwargs_type][k] = v

        return BlockState(**data)

    def set_block_state(self, state: PipelineState, block_state: BlockState):
        for output_param in self.intermediate_outputs:
            if not hasattr(block_state, output_param.name):
                raise ValueError(f"Intermediate output '{output_param.name}' is missing in block state")
            param = getattr(block_state, output_param.name)
            state.set(output_param.name, param, output_param.kwargs_type)

        for input_param in self.inputs:
            if input_param.name and hasattr(block_state, input_param.name):
                param = getattr(block_state, input_param.name)
                # Only add if the value is different from what's in the state
                current_value = state.get(input_param.name)
                if current_value is not param:  # Using identity comparison to check if object was modified
                    state.set(input_param.name, param, input_param.kwargs_type)

            elif input_param.kwargs_type:
                # if it is a kwargs type, e.g. "denoiser_input_fields", it is likely to be a list of parameters
                # we need to first find out which inputs are and loop through them.
                intermediate_kwargs = state.get_by_kwargs(input_param.kwargs_type)
                for param_name, current_value in intermediate_kwargs.items():
                    if param_name is None:
                        continue

                    if not hasattr(block_state, param_name):
                        continue

                    param = getattr(block_state, param_name)
                    if current_value is not param:  # Using identity comparison to check if object was modified
                        state.set(param_name, param, input_param.kwargs_type)

    @staticmethod
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

    @staticmethod
    def combine_outputs(*named_output_lists: List[Tuple[str, List[OutputParam]]]) -> List[OutputParam]:
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

    @property
    def input_names(self) -> List[str]:
        return [input_param.name for input_param in self.inputs]

    @property
    def intermediate_output_names(self) -> List[str]:
        return [output_param.name for output_param in self.intermediate_outputs]

    @property
    def output_names(self) -> List[str]:
        return [output_param.name for output_param in self.outputs]

    @property
    def doc(self):
        return make_doc_string(
            self.inputs,
            self.outputs,
            self.description,
            class_name=self.__class__.__name__,
            expected_components=self.expected_components,
            expected_configs=self.expected_configs,
        )


class AutoPipelineBlocks(ModularPipelineBlocks):
    """
    A Pipeline Blocks that automatically selects a block to run based on the inputs.

    This class inherits from [`ModularPipelineBlocks`]. Check the superclass documentation for the generic methods the
    library implements for all the pipeline blocks (such as loading or saving etc.)

    > [!WARNING] > This is an experimental feature and is likely to change in the future.

    Attributes:
        block_classes: List of block classes to be used
        block_names: List of prefixes for each block
        block_trigger_inputs: List of input names that trigger specific blocks, with None for default
    """

    block_classes = []
    block_names = []
    block_trigger_inputs = []

    def __init__(self):
        sub_blocks = InsertableDict()
        for block_name, block in zip(self.block_names, self.block_classes):
            if inspect.isclass(block):
                sub_blocks[block_name] = block()
            else:
                sub_blocks[block_name] = block
        self.sub_blocks = sub_blocks
        if not (len(self.block_classes) == len(self.block_names) == len(self.block_trigger_inputs)):
            raise ValueError(
                f"In {self.__class__.__name__}, the number of block_classes, block_names, and block_trigger_inputs must be the same."
            )
        default_blocks = [t for t in self.block_trigger_inputs if t is None]
        # can only have 1 or 0 default block, and has to put in the last
        # the order of blocks matters here because the first block with matching trigger will be dispatched
        # e.g. blocks = [inpaint, img2img] and block_trigger_inputs = ["mask", "image"]
        # as long as mask is provided, it is inpaint; if only image is provided, it is img2img
        if len(default_blocks) > 1 or (len(default_blocks) == 1 and self.block_trigger_inputs[-1] is not None):
            raise ValueError(
                f"In {self.__class__.__name__}, exactly one None must be specified as the last element "
                "in block_trigger_inputs."
            )

        # Map trigger inputs to block objects
        self.trigger_to_block_map = dict(zip(self.block_trigger_inputs, self.sub_blocks.values()))
        self.trigger_to_block_name_map = dict(zip(self.block_trigger_inputs, self.sub_blocks.keys()))
        self.block_to_trigger_map = dict(zip(self.sub_blocks.keys(), self.block_trigger_inputs))

    @property
    def model_name(self):
        return next(iter(self.sub_blocks.values())).model_name

    @property
    def description(self):
        return ""

    @property
    def expected_components(self):
        expected_components = []
        for block in self.sub_blocks.values():
            for component in block.expected_components:
                if component not in expected_components:
                    expected_components.append(component)
        return expected_components

    @property
    def expected_configs(self):
        expected_configs = []
        for block in self.sub_blocks.values():
            for config in block.expected_configs:
                if config not in expected_configs:
                    expected_configs.append(config)
        return expected_configs

    @property
    def required_inputs(self) -> List[str]:
        if None not in self.block_trigger_inputs:
            return []
        first_block = next(iter(self.sub_blocks.values()))
        required_by_all = set(getattr(first_block, "required_inputs", set()))

        # Intersect with required inputs from all other blocks
        for block in list(self.sub_blocks.values())[1:]:
            block_required = set(getattr(block, "required_inputs", set()))
            required_by_all.intersection_update(block_required)

        return list(required_by_all)

    # YiYi TODO: add test for this
    @property
    def inputs(self) -> List[Tuple[str, Any]]:
        named_inputs = [(name, block.inputs) for name, block in self.sub_blocks.items()]
        combined_inputs = self.combine_inputs(*named_inputs)
        # mark Required inputs only if that input is required by all the blocks
        for input_param in combined_inputs:
            if input_param.name in self.required_inputs:
                input_param.required = True
            else:
                input_param.required = False
        return combined_inputs

    @property
    def intermediate_outputs(self) -> List[str]:
        named_outputs = [(name, block.intermediate_outputs) for name, block in self.sub_blocks.items()]
        combined_outputs = self.combine_outputs(*named_outputs)
        return combined_outputs

    @property
    def outputs(self) -> List[str]:
        named_outputs = [(name, block.outputs) for name, block in self.sub_blocks.items()]
        combined_outputs = self.combine_outputs(*named_outputs)
        return combined_outputs

    @torch.no_grad()
    def __call__(self, pipeline, state: PipelineState) -> PipelineState:
        # Find default block first (if any)

        block = self.trigger_to_block_map.get(None)
        for input_name in self.block_trigger_inputs:
            if input_name is not None and state.get(input_name) is not None:
                block = self.trigger_to_block_map[input_name]
                break

        if block is None:
            logger.info(f"skipping auto block: {self.__class__.__name__}")
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
        Returns a set of all unique trigger input values found in the blocks. Returns: Set[str] containing all unique
        block_trigger_inputs values
        """

        def fn_recursive_get_trigger(blocks):
            trigger_values = set()

            if blocks is not None:
                for name, block in blocks.items():
                    # Check if current block has trigger inputs(i.e. auto block)
                    if hasattr(block, "block_trigger_inputs") and block.block_trigger_inputs is not None:
                        # Add all non-None values from the trigger inputs list
                        trigger_values.update(t for t in block.block_trigger_inputs if t is not None)

                    # If block has sub_blocks, recursively check them
                    if block.sub_blocks:
                        nested_triggers = fn_recursive_get_trigger(block.sub_blocks)
                        trigger_values.update(nested_triggers)

            return trigger_values

        trigger_inputs = set(self.block_trigger_inputs)
        trigger_inputs.update(fn_recursive_get_trigger(self.sub_blocks))

        return trigger_inputs

    @property
    def trigger_inputs(self):
        return self._get_trigger_inputs()

    def __repr__(self):
        class_name = self.__class__.__name__
        base_class = self.__class__.__bases__[0].__name__
        header = (
            f"{class_name}(\n  Class: {base_class}\n" if base_class and base_class != "object" else f"{class_name}(\n"
        )

        if self.trigger_inputs:
            header += "\n"
            header += "  " + "=" * 100 + "\n"
            header += "  This pipeline contains blocks that are selected at runtime based on inputs.\n"
            header += f"  Trigger Inputs: {[inp for inp in self.trigger_inputs if inp is not None]}\n"
            header += "  " + "=" * 100 + "\n\n"

        # Format description with proper indentation
        desc_lines = self.description.split("\n")
        desc = []
        # First line with "Description:" label
        desc.append(f"  Description: {desc_lines[0]}")
        # Subsequent lines with proper indentation
        if len(desc_lines) > 1:
            desc.extend(f"      {line}" for line in desc_lines[1:])
        desc = "\n".join(desc) + "\n"

        # Components section - focus only on expected components
        expected_components = getattr(self, "expected_components", [])
        components_str = format_components(expected_components, indent_level=2, add_empty_lines=False)

        # Configs section - use format_configs with add_empty_lines=False
        expected_configs = getattr(self, "expected_configs", [])
        configs_str = format_configs(expected_configs, indent_level=2, add_empty_lines=False)

        # Blocks section - moved to the end with simplified format
        blocks_str = "  Sub-Blocks:\n"
        for i, (name, block) in enumerate(self.sub_blocks.items()):
            # Get trigger input for this block
            trigger = None
            if hasattr(self, "block_to_trigger_map"):
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
            desc_lines = block.description.split("\n")
            indented_desc = desc_lines[0]
            if len(desc_lines) > 1:
                indented_desc += "\n" + "\n".join("                   " + line for line in desc_lines[1:])
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
            self.outputs,
            self.description,
            class_name=self.__class__.__name__,
            expected_components=self.expected_components,
            expected_configs=self.expected_configs,
        )


class SequentialPipelineBlocks(ModularPipelineBlocks):
    """
    A Pipeline Blocks that combines multiple pipeline block classes into one. When called, it will call each block in
    sequence.

    This class inherits from [`ModularPipelineBlocks`]. Check the superclass documentation for the generic methods the
    library implements for all the pipeline blocks (such as loading or saving etc.)

    > [!WARNING] > This is an experimental feature and is likely to change in the future.

    Attributes:
        block_classes: List of block classes to be used
        block_names: List of prefixes for each block
    """

    block_classes = []
    block_names = []

    @property
    def description(self):
        return ""

    @property
    def model_name(self):
        return next((block.model_name for block in self.sub_blocks.values() if block.model_name is not None), None)

    @property
    def expected_components(self):
        expected_components = []
        for block in self.sub_blocks.values():
            for component in block.expected_components:
                if component not in expected_components:
                    expected_components.append(component)
        return expected_components

    @property
    def expected_configs(self):
        expected_configs = []
        for block in self.sub_blocks.values():
            for config in block.expected_configs:
                if config not in expected_configs:
                    expected_configs.append(config)
        return expected_configs

    @classmethod
    def from_blocks_dict(
        cls, blocks_dict: Dict[str, Any], description: Optional[str] = None
    ) -> "SequentialPipelineBlocks":
        """Creates a SequentialPipelineBlocks instance from a dictionary of blocks.

        Args:
            blocks_dict: Dictionary mapping block names to block classes or instances

        Returns:
            A new SequentialPipelineBlocks instance
        """
        instance = cls()

        # Create instances if classes are provided
        sub_blocks = InsertableDict()
        for name, block in blocks_dict.items():
            if inspect.isclass(block):
                sub_blocks[name] = block()
            else:
                sub_blocks[name] = block

        instance.block_classes = [block.__class__ for block in sub_blocks.values()]
        instance.block_names = list(sub_blocks.keys())
        instance.sub_blocks = sub_blocks

        if description is not None:
            instance.description = description

        return instance

    def __init__(self):
        sub_blocks = InsertableDict()
        for block_name, block in zip(self.block_names, self.block_classes):
            if inspect.isclass(block):
                sub_blocks[block_name] = block()
            else:
                sub_blocks[block_name] = block
        self.sub_blocks = sub_blocks

    def _get_inputs(self):
        inputs = []
        outputs = set()

        # Go through all blocks in order
        for block in self.sub_blocks.values():
            # Add inputs that aren't in outputs yet
            for inp in block.inputs:
                if inp.name not in outputs and inp.name not in {input.name for input in inputs}:
                    inputs.append(inp)

            # Only add outputs if the block cannot be skipped
            should_add_outputs = True
            if hasattr(block, "block_trigger_inputs") and None not in block.block_trigger_inputs:
                should_add_outputs = False

            if should_add_outputs:
                # Add this block's outputs
                block_intermediate_outputs = [out.name for out in block.intermediate_outputs]
                outputs.update(block_intermediate_outputs)

        return inputs

    # YiYi TODO: add test for this
    @property
    def inputs(self) -> List[Tuple[str, Any]]:
        return self._get_inputs()

    @property
    def required_inputs(self) -> List[str]:
        # Get the first block from the dictionary
        first_block = next(iter(self.sub_blocks.values()))
        required_by_any = set(getattr(first_block, "required_inputs", set()))

        # Union with required inputs from all other blocks
        for block in list(self.sub_blocks.values())[1:]:
            block_required = set(getattr(block, "required_inputs", set()))
            required_by_any.update(block_required)

        return list(required_by_any)

    @property
    def intermediate_outputs(self) -> List[str]:
        named_outputs = []
        for name, block in self.sub_blocks.items():
            inp_names = {inp.name for inp in block.inputs}
            # so we only need to list new variables as intermediate_outputs, but if user wants to list these they modified it's still fine (a.k.a we don't enforce)
            # filter out them here so they do not end up as intermediate_outputs
            if name not in inp_names:
                named_outputs.append((name, block.intermediate_outputs))
        combined_outputs = self.combine_outputs(*named_outputs)
        return combined_outputs

    # YiYi TODO: I think we can remove the outputs property
    @property
    def outputs(self) -> List[str]:
        # return next(reversed(self.sub_blocks.values())).intermediate_outputs
        return self.intermediate_outputs

    @torch.no_grad()
    def __call__(self, pipeline, state: PipelineState) -> PipelineState:
        for block_name, block in self.sub_blocks.items():
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
        Returns a set of all unique trigger input values found in the blocks. Returns: Set[str] containing all unique
        block_trigger_inputs values
        """

        def fn_recursive_get_trigger(blocks):
            trigger_values = set()

            if blocks is not None:
                for name, block in blocks.items():
                    # Check if current block has trigger inputs(i.e. auto block)
                    if hasattr(block, "block_trigger_inputs") and block.block_trigger_inputs is not None:
                        # Add all non-None values from the trigger inputs list
                        trigger_values.update(t for t in block.block_trigger_inputs if t is not None)

                    # If block has sub_blocks, recursively check them
                    if block.sub_blocks:
                        nested_triggers = fn_recursive_get_trigger(block.sub_blocks)
                        trigger_values.update(nested_triggers)

            return trigger_values

        return fn_recursive_get_trigger(self.sub_blocks)

    @property
    def trigger_inputs(self):
        return self._get_trigger_inputs()

    def _traverse_trigger_blocks(self, trigger_inputs):
        # Convert trigger_inputs to a set for easier manipulation
        active_triggers = set(trigger_inputs)

        def fn_recursive_traverse(block, block_name, active_triggers):
            result_blocks = OrderedDict()

            # sequential(include loopsequential) or PipelineBlock
            if not hasattr(block, "block_trigger_inputs"):
                if block.sub_blocks:
                    # sequential or LoopSequentialPipelineBlocks (keep traversing)
                    for sub_block_name, sub_block in block.sub_blocks.items():
                        blocks_to_update = fn_recursive_traverse(sub_block, sub_block_name, active_triggers)
                        blocks_to_update = fn_recursive_traverse(sub_block, sub_block_name, active_triggers)
                        blocks_to_update = {f"{block_name}.{k}": v for k, v in blocks_to_update.items()}
                        result_blocks.update(blocks_to_update)
                else:
                    # PipelineBlock
                    result_blocks[block_name] = block
                    # Add this block's output names to active triggers if defined
                    if hasattr(block, "outputs"):
                        active_triggers.update(out.name for out in block.outputs)
                return result_blocks

            # auto
            else:
                # Find first block_trigger_input that matches any value in our active_triggers
                this_block = None
                for trigger_input in block.block_trigger_inputs:
                    if trigger_input is not None and trigger_input in active_triggers:
                        this_block = block.trigger_to_block_map[trigger_input]
                        break

                # If no matches found, try to get the default (None) block
                if this_block is None and None in block.block_trigger_inputs:
                    this_block = block.trigger_to_block_map[None]

                if this_block is not None:
                    # sequential/auto (keep traversing)
                    if this_block.sub_blocks:
                        result_blocks.update(fn_recursive_traverse(this_block, block_name, active_triggers))
                    else:
                        # PipelineBlock
                        result_blocks[block_name] = this_block
                        # Add this block's output names to active triggers if defined
                        # YiYi TODO: do we need outputs here? can it just be intermediate_outputs? can we get rid of outputs attribute?
                        if hasattr(this_block, "outputs"):
                            active_triggers.update(out.name for out in this_block.outputs)

            return result_blocks

        all_blocks = OrderedDict()
        for block_name, block in self.sub_blocks.items():
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
            f"{class_name}(\n  Class: {base_class}\n" if base_class and base_class != "object" else f"{class_name}(\n"
        )

        if self.trigger_inputs:
            header += "\n"
            header += "  " + "=" * 100 + "\n"
            header += "  This pipeline contains blocks that are selected at runtime based on inputs.\n"
            header += f"  Trigger Inputs: {[inp for inp in self.trigger_inputs if inp is not None]}\n"
            # Get first trigger input as example
            example_input = next(t for t in self.trigger_inputs if t is not None)
            header += f"  Use `get_execution_blocks()` with input names to see selected blocks (e.g. `get_execution_blocks('{example_input}')`).\n"
            header += "  " + "=" * 100 + "\n\n"

        # Format description with proper indentation
        desc_lines = self.description.split("\n")
        desc = []
        # First line with "Description:" label
        desc.append(f"  Description: {desc_lines[0]}")
        # Subsequent lines with proper indentation
        if len(desc_lines) > 1:
            desc.extend(f"      {line}" for line in desc_lines[1:])
        desc = "\n".join(desc) + "\n"

        # Components section - focus only on expected components
        expected_components = getattr(self, "expected_components", [])
        components_str = format_components(expected_components, indent_level=2, add_empty_lines=False)

        # Configs section - use format_configs with add_empty_lines=False
        expected_configs = getattr(self, "expected_configs", [])
        configs_str = format_configs(expected_configs, indent_level=2, add_empty_lines=False)

        # Blocks section - moved to the end with simplified format
        blocks_str = "  Sub-Blocks:\n"
        for i, (name, block) in enumerate(self.sub_blocks.items()):
            # Get trigger input for this block
            trigger = None
            if hasattr(self, "block_to_trigger_map"):
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
            desc_lines = block.description.split("\n")
            indented_desc = desc_lines[0]
            if len(desc_lines) > 1:
                indented_desc += "\n" + "\n".join("                   " + line for line in desc_lines[1:])
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
            self.outputs,
            self.description,
            class_name=self.__class__.__name__,
            expected_components=self.expected_components,
            expected_configs=self.expected_configs,
        )


class LoopSequentialPipelineBlocks(ModularPipelineBlocks):
    """
    A Pipeline blocks that combines multiple pipeline block classes into a For Loop. When called, it will call each
    block in sequence.

    This class inherits from [`ModularPipelineBlocks`]. Check the superclass documentation for the generic methods the
    library implements for all the pipeline blocks (such as loading or saving etc.)

    > [!WARNING] > This is an experimental feature and is likely to change in the future.

    Attributes:
        block_classes: List of block classes to be used
        block_names: List of prefixes for each block
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
    def loop_required_inputs(self) -> List[str]:
        input_names = []
        for input_param in self.loop_inputs:
            if input_param.required:
                input_names.append(input_param.name)
        return input_names

    @property
    def loop_intermediate_outputs(self) -> List[OutputParam]:
        """List of intermediate output parameters. Must be implemented by subclasses."""
        return []

    # modified from SequentialPipelineBlocks to include loop_expected_components
    @property
    def expected_components(self):
        expected_components = []
        for block in self.sub_blocks.values():
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
        for block in self.sub_blocks.values():
            for config in block.expected_configs:
                if config not in expected_configs:
                    expected_configs.append(config)
        for config in self.loop_expected_configs:
            if config not in expected_configs:
                expected_configs.append(config)
        return expected_configs

    def _get_inputs(self):
        inputs = []
        inputs.extend(self.loop_inputs)
        outputs = set()

        for name, block in self.sub_blocks.items():
            # Add inputs that aren't in outputs yet
            for inp in block.inputs:
                if inp.name not in outputs and inp not in inputs:
                    inputs.append(inp)

            # Only add outputs if the block cannot be skipped
            should_add_outputs = True
            if hasattr(block, "block_trigger_inputs") and None not in block.block_trigger_inputs:
                should_add_outputs = False

            if should_add_outputs:
                # Add this block's outputs
                block_intermediate_outputs = [out.name for out in block.intermediate_outputs]
                outputs.update(block_intermediate_outputs)

        for input_param in inputs:
            if input_param.name in self.required_inputs:
                input_param.required = True
            else:
                input_param.required = False

        return inputs

    @property
    # Copied from diffusers.modular_pipelines.modular_pipeline.SequentialPipelineBlocks.inputs
    def inputs(self):
        return self._get_inputs()

    # modified from SequentialPipelineBlocks, if any additionan input required by the loop is required by the block
    @property
    def required_inputs(self) -> List[str]:
        # Get the first block from the dictionary
        first_block = next(iter(self.sub_blocks.values()))
        required_by_any = set(getattr(first_block, "required_inputs", set()))

        required_by_loop = set(getattr(self, "loop_required_inputs", set()))
        required_by_any.update(required_by_loop)

        # Union with required inputs from all other blocks
        for block in list(self.sub_blocks.values())[1:]:
            block_required = set(getattr(block, "required_inputs", set()))
            required_by_any.update(block_required)

        return list(required_by_any)

    # YiYi TODO: this need to be thought about more
    # modified from SequentialPipelineBlocks to include loop_intermediate_outputs
    @property
    def intermediate_outputs(self) -> List[str]:
        named_outputs = [(name, block.intermediate_outputs) for name, block in self.sub_blocks.items()]
        combined_outputs = self.combine_outputs(*named_outputs)
        for output in self.loop_intermediate_outputs:
            if output.name not in {output.name for output in combined_outputs}:
                combined_outputs.append(output)
        return combined_outputs

    # YiYi TODO: this need to be thought about more
    @property
    def outputs(self) -> List[str]:
        return next(reversed(self.sub_blocks.values())).intermediate_outputs

    def __init__(self):
        sub_blocks = InsertableDict()
        for block_name, block in zip(self.block_names, self.block_classes):
            if inspect.isclass(block):
                sub_blocks[block_name] = block()
            else:
                sub_blocks[block_name] = block
        self.sub_blocks = sub_blocks

    @classmethod
    def from_blocks_dict(cls, blocks_dict: Dict[str, Any]) -> "LoopSequentialPipelineBlocks":
        """
        Creates a LoopSequentialPipelineBlocks instance from a dictionary of blocks.

        Args:
            blocks_dict: Dictionary mapping block names to block instances

        Returns:
            A new LoopSequentialPipelineBlocks instance
        """
        instance = cls()

        # Create instances if classes are provided
        sub_blocks = InsertableDict()
        for name, block in blocks_dict.items():
            if inspect.isclass(block):
                sub_blocks[name] = block()
            else:
                sub_blocks[name] = block

        instance.block_classes = [block.__class__ for block in blocks_dict.values()]
        instance.block_names = list(blocks_dict.keys())
        instance.sub_blocks = blocks_dict
        return instance

    def loop_step(self, components, state: PipelineState, **kwargs):
        for block_name, block in self.sub_blocks.items():
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

    @property
    def doc(self):
        return make_doc_string(
            self.inputs,
            self.outputs,
            self.description,
            class_name=self.__class__.__name__,
            expected_components=self.expected_components,
            expected_configs=self.expected_configs,
        )

    # modified from SequentialPipelineBlocks,
    # (does not need trigger_inputs related part so removed them,
    # do not need to support auto block for loop blocks)
    def __repr__(self):
        class_name = self.__class__.__name__
        base_class = self.__class__.__bases__[0].__name__
        header = (
            f"{class_name}(\n  Class: {base_class}\n" if base_class and base_class != "object" else f"{class_name}(\n"
        )

        # Format description with proper indentation
        desc_lines = self.description.split("\n")
        desc = []
        # First line with "Description:" label
        desc.append(f"  Description: {desc_lines[0]}")
        # Subsequent lines with proper indentation
        if len(desc_lines) > 1:
            desc.extend(f"      {line}" for line in desc_lines[1:])
        desc = "\n".join(desc) + "\n"

        # Components section - focus only on expected components
        expected_components = getattr(self, "expected_components", [])
        components_str = format_components(expected_components, indent_level=2, add_empty_lines=False)

        # Configs section - use format_configs with add_empty_lines=False
        expected_configs = getattr(self, "expected_configs", [])
        configs_str = format_configs(expected_configs, indent_level=2, add_empty_lines=False)

        # Blocks section - moved to the end with simplified format
        blocks_str = "  Sub-Blocks:\n"
        for i, (name, block) in enumerate(self.sub_blocks.items()):
            # For SequentialPipelineBlocks, show execution order
            blocks_str += f"    [{i}] {name} ({block.__class__.__name__})\n"

            # Add block description
            desc_lines = block.description.split("\n")
            indented_desc = desc_lines[0]
            if len(desc_lines) > 1:
                indented_desc += "\n" + "\n".join("                   " + line for line in desc_lines[1:])
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
# 1. look into the serialization of modular_model_index.json, make sure the items are properly ordered like model_index.json (currently a mess)
# 2. do we need ConfigSpec? the are basically just key/val kwargs
# 3. imnprove docstring and potentially add validator for methods where we accept kwargs to be passed to from_pretrained/save_pretrained/load_components()
class ModularPipeline(ConfigMixin, PushToHubMixin):
    """
    Base class for all Modular pipelines.

    > [!WARNING] > This is an experimental feature and is likely to change in the future.

    Args:
        blocks: ModularPipelineBlocks, the blocks to be used in the pipeline
    """

    config_name = "modular_model_index.json"
    hf_device_map = None
    default_blocks_name = None

    # YiYi TODO: add warning for passing multiple ComponentSpec/ConfigSpec with the same name
    def __init__(
        self,
        blocks: Optional[ModularPipelineBlocks] = None,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]] = None,
        components_manager: Optional[ComponentsManager] = None,
        collection: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize a ModularPipeline instance.

        This method sets up the pipeline by:
        - creating default pipeline blocks if not provided
        - gather component and config specifications based on the pipeline blocks's requirement (e.g.
           expected_components, expected_configs)
        - update the loading specs of from_pretrained components based on the modular_model_index.json file from
           huggingface hub if `pretrained_model_name_or_path` is provided
        - create defaultfrom_config components and register everything

        Args:
            blocks: `ModularPipelineBlocks` instance. If None, will attempt to load
                   default blocks based on the pipeline class name.
            pretrained_model_name_or_path: Path to a pretrained pipeline configuration. Can be None if the pipeline
                    does not require any additional loading config. If provided, will first try to load component specs
                    (only for from_pretrained components) and config values from `modular_model_index.json`, then
                    fallback to `model_index.json` for compatibility with standard non-modular repositories.
            components_manager:
                Optional ComponentsManager for managing multiple component cross different pipelines and apply
                offloading strategies.
            collection: Optional collection name for organizing components in the ComponentsManager.
            **kwargs: Additional arguments passed to `load_config()` when loading pretrained configuration.

        Examples:
            ```python
            # Initialize with custom blocks
            pipeline = ModularPipeline(blocks=my_custom_blocks)

            # Initialize from pretrained configuration
            pipeline = ModularPipeline(blocks=my_blocks, pretrained_model_name_or_path="my-repo/modular-pipeline")

            # Initialize with components manager
            pipeline = ModularPipeline(
                blocks=my_blocks, components_manager=ComponentsManager(), collection="my_collection"
            )
            ```

        Notes:
            - If blocks is None, the method will try to find default blocks based on the pipeline class name
            - Components with default_creation_method="from_config" are created immediately, its specs are not included
              in config dict and will not be saved in `modular_model_index.json`
            - Components with default_creation_method="from_pretrained" are set to None and can be loaded later with
              `load_components()` (with or without specific component names)
            - The pipeline's config dict is populated with component specs (only for from_pretrained components) and
              config values, which will be saved as `modular_model_index.json` during `save_pretrained`
            - The pipeline's config dict is also used to store the pipeline blocks's class name, which will be saved as
              `_blocks_class_name` in the config dict
        """
        if blocks is None:
            blocks_class_name = self.default_blocks_name
            if blocks_class_name is not None:
                diffusers_module = importlib.import_module("diffusers")
                blocks_class = getattr(diffusers_module, blocks_class_name)
                blocks = blocks_class()
            else:
                logger.warning(f"`blocks` is `None`, no default blocks class found for {self.__class__.__name__}")

        self.blocks = blocks
        self._components_manager = components_manager
        self._collection = collection
        self._component_specs = {spec.name: deepcopy(spec) for spec in self.blocks.expected_components}
        self._config_specs = {spec.name: deepcopy(spec) for spec in self.blocks.expected_configs}

        # update component_specs and config_specs from modular_repo
        if pretrained_model_name_or_path is not None:
            cache_dir = kwargs.pop("cache_dir", None)
            force_download = kwargs.pop("force_download", False)
            proxies = kwargs.pop("proxies", None)
            token = kwargs.pop("token", None)
            local_files_only = kwargs.pop("local_files_only", False)
            revision = kwargs.pop("revision", None)

            load_config_kwargs = {
                "cache_dir": cache_dir,
                "force_download": force_download,
                "proxies": proxies,
                "token": token,
                "local_files_only": local_files_only,
                "revision": revision,
            }
            # try to load modular_model_index.json
            try:
                config_dict = self.load_config(pretrained_model_name_or_path, **load_config_kwargs)
            except EnvironmentError as e:
                logger.debug(f"modular_model_index.json not found: {e}")
                config_dict = None

            # update component_specs and config_specs based on modular_model_index.json
            if config_dict is not None:
                for name, value in config_dict.items():
                    # all the components in modular_model_index.json are from_pretrained components
                    if name in self._component_specs and isinstance(value, (tuple, list)) and len(value) == 3:
                        library, class_name, component_spec_dict = value
                        component_spec = self._dict_to_component_spec(name, component_spec_dict)
                        component_spec.default_creation_method = "from_pretrained"
                        self._component_specs[name] = component_spec

                    elif name in self._config_specs:
                        self._config_specs[name].default = value

            # if modular_model_index.json is not found, try to load model_index.json
            else:
                logger.debug(" loading config from model_index.json")
                try:
                    from diffusers import DiffusionPipeline

                    config_dict = DiffusionPipeline.load_config(pretrained_model_name_or_path, **load_config_kwargs)
                except EnvironmentError as e:
                    logger.debug(f" model_index.json not found in the repo: {e}")
                    config_dict = None

                # update component_specs and config_specs based on model_index.json
                if config_dict is not None:
                    for name, value in config_dict.items():
                        if name in self._component_specs and isinstance(value, (tuple, list)) and len(value) == 2:
                            library, class_name = value
                            component_spec_dict = {
                                "repo": pretrained_model_name_or_path,
                                "subfolder": name,
                                "type_hint": (library, class_name),
                            }
                            component_spec = self._dict_to_component_spec(name, component_spec_dict)
                            component_spec.default_creation_method = "from_pretrained"
                            self._component_specs[name] = component_spec
                        elif name in self._config_specs:
                            self._config_specs[name].default = value

        if len(kwargs) > 0:
            logger.warning(f"Unexpected input '{kwargs.keys()}' provided. This input will be ignored.")

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

        self.register_to_config(_blocks_class_name=self.blocks.__class__.__name__ if self.blocks is not None else None)

    @property
    def default_call_parameters(self) -> Dict[str, Any]:
        """
        Returns:
            - Dictionary mapping input names to their default values
        """
        params = {}
        for input_param in self.blocks.inputs:
            params[input_param.name] = input_param.default
        return params

    @classmethod
    @validate_hf_hub_args
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        trust_remote_code: Optional[bool] = None,
        components_manager: Optional[ComponentsManager] = None,
        collection: Optional[str] = None,
        **kwargs,
    ):
        """
        Load a ModularPipeline from a huggingface hub repo.

        Args:
            pretrained_model_name_or_path (`str` or `os.PathLike`, optional):
                Path to a pretrained pipeline configuration. It will first try to load config from
                `modular_model_index.json`, then fallback to `model_index.json` for compatibility with standard
                non-modular repositories. If the repo does not contain any pipeline config, it will be set to None
                during initialization.
            trust_remote_code (`bool`, optional):
                Whether to trust remote code when loading the pipeline, need to be set to True if you want to create
                pipeline blocks based on the custom code in `pretrained_model_name_or_path`
            components_manager (`ComponentsManager`, optional):
                ComponentsManager instance for managing multiple component cross different pipelines and apply
                offloading strategies.
            collection (`str`, optional):`
                Collection name for organizing components in the ComponentsManager.
        """
        from ..pipelines.pipeline_loading_utils import _get_pipeline_class

        try:
            blocks = ModularPipelineBlocks.from_pretrained(
                pretrained_model_name_or_path, trust_remote_code=trust_remote_code, **kwargs
            )
        except EnvironmentError as e:
            logger.debug(f"EnvironmentError: {e}")
            blocks = None

        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        proxies = kwargs.pop("proxies", None)
        token = kwargs.pop("token", None)
        local_files_only = kwargs.pop("local_files_only", False)
        revision = kwargs.pop("revision", None)

        load_config_kwargs = {
            "cache_dir": cache_dir,
            "force_download": force_download,
            "proxies": proxies,
            "token": token,
            "local_files_only": local_files_only,
            "revision": revision,
        }

        try:
            # try to load modular_model_index.json
            config_dict = cls.load_config(pretrained_model_name_or_path, **load_config_kwargs)
        except EnvironmentError as e:
            logger.debug(f" modular_model_index.json not found in the repo: {e}")
            config_dict = None

        if config_dict is not None:
            pipeline_class = _get_pipeline_class(cls, config=config_dict)
        else:
            try:
                logger.debug(" try to load model_index.json")
                from diffusers import DiffusionPipeline
                from diffusers.pipelines.auto_pipeline import _get_model

                config_dict = DiffusionPipeline.load_config(pretrained_model_name_or_path, **load_config_kwargs)
            except EnvironmentError as e:
                logger.debug(f" model_index.json not found in the repo: {e}")

            if config_dict is not None:
                logger.debug(" try to determine the modular pipeline class from model_index.json")
                standard_pipeline_class = _get_pipeline_class(cls, config=config_dict)
                model_name = _get_model(standard_pipeline_class.__name__)
                pipeline_class_name = MODULAR_PIPELINE_MAPPING.get(model_name, ModularPipeline.__name__)
                diffusers_module = importlib.import_module("diffusers")
                pipeline_class = getattr(diffusers_module, pipeline_class_name)
            else:
                # there is no config for modular pipeline, assuming that the pipeline block does not need any from_pretrained components
                pipeline_class = cls
                pretrained_model_name_or_path = None

        pipeline = pipeline_class(
            blocks=blocks,
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            components_manager=components_manager,
            collection=collection,
            **kwargs,
        )
        return pipeline

    def save_pretrained(self, save_directory: Union[str, os.PathLike], push_to_hub: bool = False, **kwargs):
        """
        Save the pipeline to a directory. It does not save components, you need to save them separately.

        Args:
            save_directory (`str` or `os.PathLike`):
                Path to the directory where the pipeline will be saved.
            push_to_hub (`bool`, optional):
                Whether to push the pipeline to the huggingface hub.
            **kwargs: Additional arguments passed to `save_config()` method
        """
        if push_to_hub:
            commit_message = kwargs.pop("commit_message", None)
            private = kwargs.pop("private", None)
            create_pr = kwargs.pop("create_pr", False)
            token = kwargs.pop("token", None)
            repo_id = kwargs.pop("repo_id", save_directory.split(os.path.sep)[-1])
            repo_id = create_repo(repo_id, exist_ok=True, private=private, token=token).repo_id

            # Create a new empty model card and eventually tag it
            model_card = load_or_create_model_card(repo_id, token=token, is_pipeline=True)
            model_card = populate_model_card(model_card)
            model_card.save(os.path.join(save_directory, "README.md"))

        # YiYi TODO: maybe order the json file to make it more readable: configs first, then components
        self.save_config(save_directory=save_directory)

        if push_to_hub:
            self._upload_folder(
                save_directory,
                repo_id,
                token=token,
                commit_message=commit_message,
                create_pr=create_pr,
            )

    @property
    def doc(self):
        """
        Returns:
            - The docstring of the pipeline blocks
        """
        return self.blocks.doc

    def register_components(self, **kwargs):
        """
        Register components with their corresponding specifications.

        This method is responsible for:
        1. Sets component objects as attributes on the loader (e.g., self.unet = unet)
        2. Updates the config dict, which will be saved as `modular_model_index.json` during `save_pretrained` (only
           for from_pretrained components)
        3. Adds components to the component manager if one is attached (only for from_pretrained components)

        This method is called when:
        - Components are first initialized in __init__:
           - from_pretrained components not loaded during __init__ so they are registered as None;
           - non from_pretrained components are created during __init__ and registered as the object itself
        - Components are updated with the `update_components()` method: e.g. loader.update_components(unet=unet) or
          loader.update_components(guider=guider_spec)
        - (from_pretrained) Components are loaded with the `load_components()` method: e.g.
          loader.load_components(names=["unet"]) or loader.load_components() to load all default components

        Args:
            **kwargs: Keyword arguments where keys are component names and values are component objects.
                      E.g., register_components(unet=unet_model, text_encoder=encoder_model)

        Notes:
            - When registering None for a component, it sets attribute to None but still syncs specs with the config
              dict, which will be saved as `modular_model_index.json` during `save_pretrained`
            - component_specs are updated to match the new component outside of this method, e.g. in
              `update_components()` method
        """
        for name, module in kwargs.items():
            # current component spec
            component_spec = self._component_specs.get(name)
            if component_spec is None:
                logger.warning(f"ModularPipeline.register_components: skipping unknown component '{name}'")
                continue

            # check if it is the first time registration, i.e. calling from __init__
            is_registered = hasattr(self, name)
            is_from_pretrained = component_spec.default_creation_method == "from_pretrained"

            if module is not None:
                # actual library and class name of the module
                library, class_name = _fetch_class_library_tuple(module)  # e.g. ("diffusers", "UNet2DConditionModel")
            else:
                # if module is None, e.g. self.register_components(unet=None) during __init__
                # we do not update the spec,
                # but we still need to update the modular_model_index.json config based on component spec
                library, class_name = None, None

            # extract the loading spec from the updated component spec that'll be used as part of modular_model_index.json config
            # e.g. {"repo": "stabilityai/stable-diffusion-2-1",
            #       "type_hint": ("diffusers", "UNet2DConditionModel"),
            #       "subfolder": "unet",
            #       "variant": None,
            #       "revision": None}
            component_spec_dict = self._component_spec_to_dict(component_spec)

            register_dict = {name: (library, class_name, component_spec_dict)}

            # set the component as attribute
            # if it is not set yet, just set it and skip the process to check and warn below
            if not is_registered:
                if is_from_pretrained:
                    self.register_to_config(**register_dict)
                setattr(self, name, module)
                if module is not None and is_from_pretrained and self._components_manager is not None:
                    self._components_manager.add(name, module, self._collection)
                continue

            current_module = getattr(self, name, None)
            # skip if the component is already registered with the same object
            if current_module is module:
                logger.info(
                    f"ModularPipeline.register_components: {name} is already registered with same object, skipping"
                )
                continue

            # warn if unregister
            if current_module is not None and module is None:
                logger.info(
                    f"ModularPipeline.register_components: setting '{name}' to None "
                    f"(was {current_module.__class__.__name__})"
                )
            # same type, new instance → replace but send debug log
            elif (
                current_module is not None
                and module is not None
                and isinstance(module, current_module.__class__)
                and current_module != module
            ):
                logger.debug(
                    f"ModularPipeline.register_components: replacing existing '{name}' "
                    f"(same type {type(current_module).__name__}, new instance)"
                )

            # update modular_model_index.json config
            if is_from_pretrained:
                self.register_to_config(**register_dict)
            # finally set models
            setattr(self, name, module)
            # add to component manager if one is attached
            if module is not None and is_from_pretrained and self._components_manager is not None:
                self._components_manager.add(name, module, self._collection)

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
    # Modified from diffusers.pipelines.pipeline_utils.DiffusionPipeline._execution_device
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
    def null_component_names(self) -> List[str]:
        """
        Returns:
            - List of names for components that needs to be loaded
        """
        return [name for name in self._component_specs.keys() if hasattr(self, name) and getattr(self, name) is None]

    @property
    def component_names(self) -> List[str]:
        """
        Returns:
            - List of names for all components
        """
        return list(self.components.keys())

    @property
    def pretrained_component_names(self) -> List[str]:
        """
        Returns:
            - List of names for from_pretrained components
        """
        return [
            name
            for name in self._component_specs.keys()
            if self._component_specs[name].default_creation_method == "from_pretrained"
        ]

    @property
    def config_component_names(self) -> List[str]:
        """
        Returns:
            - List of names for from_config components
        """
        return [
            name
            for name in self._component_specs.keys()
            if self._component_specs[name].default_creation_method == "from_config"
        ]

    @property
    def components(self) -> Dict[str, Any]:
        """
        Returns:
            - Dictionary mapping component names to their objects (include both from_pretrained and from_config
              components)
        """
        # return only components we've actually set as attributes on self
        return {name: getattr(self, name) for name in self._component_specs.keys() if hasattr(self, name)}

    def get_component_spec(self, name: str) -> ComponentSpec:
        """
        Returns:
            - a copy of the ComponentSpec object for the given component name
        """
        return deepcopy(self._component_specs[name])

    def update_components(self, **kwargs):
        """
        Update components and configuration values and specs after the pipeline has been instantiated.

        This method allows you to:
        1. Replace existing components with new ones (e.g., updating `self.unet` or `self.text_encoder`)
        2. Update configuration values (e.g., changing `self.requires_safety_checker` flag)

        In addition to updating the components and configuration values as pipeline attributes, the method also
        updates:
        - the corresponding specs in `_component_specs` and `_config_specs`
        - the `config` dict, which will be saved as `modular_model_index.json` during `save_pretrained`

        Args:
            **kwargs: Component objects, ComponentSpec objects, or configuration values to update:
                - Component objects: Only supports components we can extract specs using
                  `ComponentSpec.from_component()` method i.e. components created with ComponentSpec.load() or
                  ConfigMixin subclasses that aren't nn.Modules (e.g., `unet=new_unet, text_encoder=new_encoder`)
                - ComponentSpec objects: Only supports default_creation_method == "from_config", will call create()
                  method to create a new component (e.g., `guider=ComponentSpec(name="guider",
                  type_hint=ClassifierFreeGuidance, config={...}, default_creation_method="from_config")`)
                - Configuration values: Simple values to update configuration settings (e.g.,
                  `requires_safety_checker=False`)

        Raises:
            ValueError: If a component object is not supported in ComponentSpec.from_component() method:
                - nn.Module components without a valid `_diffusers_load_id` attribute
                - Non-ConfigMixin components without a valid `_diffusers_load_id` attribute

        Examples:
            ```python
            # Update multiple components at once
            pipeline.update_components(unet=new_unet_model, text_encoder=new_text_encoder)

            # Update configuration values
            pipeline.update_components(requires_safety_checker=False)

            # Update both components and configs together
            pipeline.update_components(unet=new_unet_model, requires_safety_checker=False)

            # Update with ComponentSpec objects (from_config only)
            pipeline.update_components(
                guider=ComponentSpec(
                    name="guider",
                    type_hint=ClassifierFreeGuidance,
                    config={"guidance_scale": 5.0},
                    default_creation_method="from_config",
                )
            )
            ```

        Notes:
            - Components with trained weights must be created using ComponentSpec.load(). If the component has not been
              shared in huggingface hub and you don't have loading specs, you can upload it using `push_to_hub()`
            - ConfigMixin objects without weights (e.g., schedulers, guiders) can be passed directly
            - ComponentSpec objects with default_creation_method="from_pretrained" are not supported in
              update_components()
        """

        # extract component_specs_updates & config_specs_updates from `specs`
        passed_component_specs = {
            k: kwargs.pop(k) for k in self._component_specs if k in kwargs and isinstance(kwargs[k], ComponentSpec)
        }
        passed_components = {
            k: kwargs.pop(k) for k in self._component_specs if k in kwargs and not isinstance(kwargs[k], ComponentSpec)
        }
        passed_config_values = {k: kwargs.pop(k) for k in self._config_specs if k in kwargs}

        for name, component in passed_components.items():
            current_component_spec = self._component_specs[name]

            # log if type changed
            if current_component_spec.type_hint is not None and not isinstance(
                component, current_component_spec.type_hint
            ):
                logger.info(
                    f"ModularPipeline.update_components: adding {name} with new type: {component.__class__.__name__}, previous type: {current_component_spec.type_hint.__name__}"
                )
            # update _component_specs based on the new component
            if component is None:
                new_component_spec = current_component_spec
                if hasattr(self, name) and getattr(self, name) is not None:
                    logger.warning(f"ModularPipeline.update_components: setting {name} to None (spec unchanged)")
            elif current_component_spec.default_creation_method == "from_pretrained" and not (
                hasattr(component, "_diffusers_load_id") and component._diffusers_load_id is not None
            ):
                logger.warning(
                    f"ModularPipeline.update_components: {name} has no valid _diffusers_load_id. "
                    f"This will result in empty loading spec, use ComponentSpec.load() for proper specs"
                )
                new_component_spec = ComponentSpec(name=name, type_hint=type(component))
            else:
                new_component_spec = ComponentSpec.from_component(name, component)

            if new_component_spec.default_creation_method != current_component_spec.default_creation_method:
                logger.info(
                    f"ModularPipeline.update_components: changing the default_creation_method of {name} from {current_component_spec.default_creation_method} to {new_component_spec.default_creation_method}."
                )

            self._component_specs[name] = new_component_spec

        if len(kwargs) > 0:
            logger.warning(f"Unexpected keyword arguments, will be ignored: {kwargs.keys()}")

        created_components = {}
        for name, component_spec in passed_component_specs.items():
            if component_spec.default_creation_method == "from_pretrained":
                raise ValueError(
                    "ComponentSpec object with default_creation_method == 'from_pretrained' is not supported in update_components() method"
                )
            created_components[name] = component_spec.create()
            current_component_spec = self._component_specs[name]
            # warn if type changed
            if current_component_spec.type_hint is not None and not isinstance(
                created_components[name], current_component_spec.type_hint
            ):
                logger.info(
                    f"ModularPipeline.update_components: adding {name} with new type: {created_components[name].__class__.__name__}, previous type: {current_component_spec.type_hint.__name__}"
                )
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
    def load_components(self, names: Optional[Union[List[str], str]] = None, **kwargs):
        """
        Load selected components from specs.

        Args:
            names: List of component names to load. If None, will load all components with
                   default_creation_method == "from_pretrained". If provided as a list or string, will load only the
                   specified components.
            **kwargs: additional kwargs to be passed to `from_pretrained()`.Can be:
             - a single value to be applied to all components to be loaded, e.g. torch_dtype=torch.bfloat16
             - a dict, e.g. torch_dtype={"unet": torch.bfloat16, "default": torch.float32}
             - if potentially override ComponentSpec if passed a different loading field in kwargs, e.g. `repo`,
               `variant`, `revision`, etc.
        """

        if names is None:
            names = [
                name
                for name in self._component_specs.keys()
                if self._component_specs[name].default_creation_method == "from_pretrained"
            ]
        elif isinstance(names, str):
            names = [names]
        elif not isinstance(names, list):
            raise ValueError(f"Invalid type for names: {type(names)}")

        components_to_load = {name for name in names if name in self._component_specs}
        unknown_names = {name for name in names if name not in self._component_specs}
        if len(unknown_names) > 0:
            logger.warning(f"Unknown components will be ignored: {unknown_names}")

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

    # Copied from diffusers.pipelines.pipeline_utils.DiffusionPipeline._maybe_raise_error_if_group_offload_active
    def _maybe_raise_error_if_group_offload_active(
        self, raise_error: bool = False, module: Optional[torch.nn.Module] = None
    ) -> bool:
        from ..hooks.group_offloading import _is_group_offload_enabled

        components = self.components.values() if module is None else [module]
        components = [component for component in components if isinstance(component, torch.nn.Module)]
        for component in components:
            if _is_group_offload_enabled(component):
                if raise_error:
                    raise ValueError(
                        "You are trying to apply model/sequential CPU offloading to a pipeline that contains components "
                        "with group offloading enabled. This is not supported. Please disable group offloading for "
                        "components of the pipeline to use other offloading methods."
                    )
                return True
        return False

    # Modified from diffusers.pipelines.pipeline_utils.DiffusionPipeline.to
    def to(self, *args, **kwargs) -> Self:
        r"""
        Performs Pipeline dtype and/or device conversion. A torch.dtype and torch.device are inferred from the
        arguments of `self.to(*args, **kwargs).`

        > [!TIP] > If the pipeline already has the correct torch.dtype and torch.device, then it is returned as is.
        Otherwise, > the returned pipeline is a copy of self with the desired torch.dtype and torch.device.


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
        from ..pipelines.pipeline_utils import _check_bnb_status
        from ..utils import is_accelerate_available, is_accelerate_version, is_hpu_available, is_transformers_version

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
        device_type = torch.device(device).type if device is not None else None
        pipeline_has_bnb = any(any((_check_bnb_status(module))) for _, module in self.components.items())

        # throw warning if pipeline is in "offloaded"-mode but user tries to manually set to GPU.
        def module_is_sequentially_offloaded(module):
            if not is_accelerate_available() or is_accelerate_version("<", "0.14.0"):
                return False

            _, _, is_loaded_in_8bit_bnb = _check_bnb_status(module)

            if is_loaded_in_8bit_bnb:
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

        is_pipeline_device_mapped = self.hf_device_map is not None and len(self.hf_device_map) > 1
        if is_pipeline_device_mapped:
            raise ValueError(
                "It seems like you have activated a device mapping strategy on the pipeline which doesn't allow explicit device placement using `to()`. You can call `reset_device_map()` to remove the existing device map from the pipeline."
            )

        if device_type in ["cuda", "xpu"]:
            if pipeline_is_sequentially_offloaded and not pipeline_has_bnb:
                raise ValueError(
                    "It seems like you have activated sequential model offloading by calling `enable_sequential_cpu_offload`, but are now attempting to move the pipeline to GPU. This is not compatible with offloading. Please, move your pipeline `.to('cpu')` or consider removing the move altogether if you use sequential offloading."
                )
            # PR: https://github.com/huggingface/accelerate/pull/3223/
            elif pipeline_has_bnb and is_accelerate_version("<", "1.1.0.dev0"):
                raise ValueError(
                    "You are trying to call `.to('cuda')` on a pipeline that has models quantized with `bitsandbytes`. Your current `accelerate` installation does not support it. Please upgrade the installation."
                )

        # Display a warning in this case (the operation succeeds but the benefits are lost)
        pipeline_is_offloaded = any(module_is_offloaded(module) for _, module in self.components.items())
        if pipeline_is_offloaded and device_type in ["cuda", "xpu"]:
            logger.warning(
                f"It seems like you have activated model offloading by calling `enable_model_cpu_offload`, but are now manually moving the pipeline to GPU. It is strongly recommended against doing so as memory gains from offloading are likely to be lost. Offloading automatically takes care of moving the individual components {', '.join(self.components.keys())} to GPU when needed. To make sure offloading works as expected, you should consider moving the pipeline back to CPU: `pipeline.to('cpu')` or removing the move altogether if you use offloading."
            )

        # Enable generic support for Intel Gaudi accelerator using GPU/HPU migration
        if device_type == "hpu" and kwargs.pop("hpu_migration", True) and is_hpu_available():
            os.environ["PT_HPU_GPU_MIGRATION"] = "1"
            logger.debug("Environment variable set: PT_HPU_GPU_MIGRATION=1")

            import habana_frameworks.torch  # noqa: F401

            # HPU hardware check
            if not (hasattr(torch, "hpu") and torch.hpu.is_available()):
                raise ValueError("You are trying to call `.to('hpu')` but HPU device is unavailable.")

            os.environ["PT_HPU_MAX_COMPOUND_OP_SIZE"] = "1"
            logger.debug("Environment variable set: PT_HPU_MAX_COMPOUND_OP_SIZE=1")

        modules = self.components.values()
        modules = [m for m in modules if isinstance(m, torch.nn.Module)]

        is_offloaded = pipeline_is_offloaded or pipeline_is_sequentially_offloaded
        for module in modules:
            _, is_loaded_in_4bit_bnb, is_loaded_in_8bit_bnb = _check_bnb_status(module)
            is_group_offloaded = self._maybe_raise_error_if_group_offload_active(module=module)

            if (is_loaded_in_4bit_bnb or is_loaded_in_8bit_bnb) and dtype is not None:
                logger.warning(
                    f"The module '{module.__class__.__name__}' has been loaded in `bitsandbytes` {'4bit' if is_loaded_in_4bit_bnb else '8bit'} and conversion to {dtype} is not supported. Module is still in {'4bit' if is_loaded_in_4bit_bnb else '8bit'} precision."
                )

            if is_loaded_in_8bit_bnb and device is not None:
                logger.warning(
                    f"The module '{module.__class__.__name__}' has been loaded in `bitsandbytes` 8bit and moving it to {device} via `.to()` is not supported. Module is still on {module.device}."
                )

            # Note: we also handle this at the ModelMixin level. The reason for doing it here too is that modeling
            # components can be from outside diffusers too, but still have group offloading enabled.
            if (
                self._maybe_raise_error_if_group_offload_active(raise_error=False, module=module)
                and device is not None
            ):
                logger.warning(
                    f"The module '{module.__class__.__name__}' is group offloaded and moving it to {device} via `.to()` is not supported."
                )

            # This can happen for `transformer` models. CPU placement was added in
            # https://github.com/huggingface/transformers/pull/33122. So, we guard this accordingly.
            if is_loaded_in_4bit_bnb and device is not None and is_transformers_version(">", "4.44.0"):
                module.to(device=device)
            elif not is_loaded_in_4bit_bnb and not is_loaded_in_8bit_bnb and not is_group_offloaded:
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

    @staticmethod
    def _component_spec_to_dict(component_spec: ComponentSpec) -> Any:
        """
        Convert a ComponentSpec into a JSON‐serializable dict for saving as an entry in `modular_model_index.json`. If
        the `default_creation_method` is not `from_pretrained`, return None.

        This dict contains:
          - "type_hint": Tuple[str, str]
              Library name and class name of the component. (e.g. ("diffusers", "UNet2DConditionModel"))
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
            >>> from diffusers.pipelines.modular_pipeline_utils import ComponentSpec >>> from diffusers import
            UNet2DConditionModel >>> spec = ComponentSpec(
                ... name="unet", ... type_hint=UNet2DConditionModel, ... config=None, ... repo="path/to/repo", ...
                subfolder="subfolder", ... variant=None, ... revision=None, ...
                default_creation_method="from_pretrained",
            ... ) >>> ModularPipeline._component_spec_to_dict(spec) {
                "type_hint": ("diffusers", "UNet2DConditionModel"), "repo": "path/to/repo", "subfolder": "subfolder",
                "variant": None, "revision": None,
            }
        """
        if component_spec.default_creation_method != "from_pretrained":
            return None

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
        Reconstruct a ComponentSpec from a loading specdict.

        This method converts a dictionary representation back into a ComponentSpec object. The dict should contain:
          - "type_hint": Tuple[str, str]
              Library name and class name of the component. (e.g. ("diffusers", "UNet2DConditionModel"))
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
            name (str):
                The name of the component.
            specdict (Dict[str, Any]):
                A dictionary containing the component specification data.

        Returns:
            ComponentSpec: A reconstructed ComponentSpec object.

        Example:
            >>> spec_dict = { ... "type_hint": ("diffusers", "UNet2DConditionModel"), ... "repo":
            "stabilityai/stable-diffusion-xl", ... "subfolder": "unet", ... "variant": None, ... "revision": None, ...
            } >>> ModularPipeline._dict_to_component_spec("unet", spec_dict) ComponentSpec(
                name="unet", type_hint=UNet2DConditionModel, config=None, repo="stabilityai/stable-diffusion-xl",
                subfolder="unet", variant=None, revision=None, default_creation_method="from_pretrained"
            )
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

    def set_progress_bar_config(self, **kwargs):
        for sub_block_name, sub_block in self.blocks.sub_blocks.items():
            if hasattr(sub_block, "set_progress_bar_config"):
                sub_block.set_progress_bar_config(**kwargs)

    def __call__(self, state: PipelineState = None, output: Union[str, List[str]] = None, **kwargs):
        """
        Execute the pipeline by running the pipeline blocks with the given inputs.

        Args:
            state (`PipelineState`, optional):
                PipelineState instance contains inputs and intermediate values. If None, a new `PipelineState` will be
                created based on the user inputs and the pipeline blocks's requirement.
            output (`str` or `List[str]`, optional):
                Optional specification of what to return:
                   - None: Returns the complete `PipelineState` with all inputs and intermediates (default)
                   - str: Returns a specific intermediate value from the state (e.g. `output="image"`)
                   - List[str]: Returns a dictionary of specific intermediate values (e.g. `output=["image",
                     "latents"]`)


        Examples:
            ```python
            # Get complete pipeline state
            state = pipeline(prompt="A beautiful sunset", num_inference_steps=20)
            print(state.intermediates)  # All intermediate outputs

            # Get specific output
            image = pipeline(prompt="A beautiful sunset", output="image")

            # Get multiple specific outputs
            results = pipeline(prompt="A beautiful sunset", output=["image", "latents"])
            image, latents = results["image"], results["latents"]

            # Continue from previous state
            state = pipeline(prompt="A beautiful sunset")
            new_state = pipeline(state=state, output="image")  # Continue processing
            ```

        Returns:
            - If `output` is None: Complete `PipelineState` containing all inputs and intermediates
            - If `output` is str: The specific intermediate value from the state (e.g. `output="image"`)
            - If `output` is List[str]: Dictionary mapping output names to their values from the state (e.g.
              `output=["image", "latents"]`)
        """
        if state is None:
            state = PipelineState()

        # Make a copy of the input kwargs
        passed_kwargs = kwargs.copy()

        # Add inputs to state, using defaults if not provided in the kwargs or the state
        # if same input already in the state, will override it if provided in the kwargs
        for expected_input_param in self.blocks.inputs:
            name = expected_input_param.name
            default = expected_input_param.default
            kwargs_type = expected_input_param.kwargs_type
            if name in passed_kwargs:
                state.set(name, passed_kwargs.pop(name), kwargs_type)
            elif name not in state.values:
                state.set(name, default, kwargs_type)

        # Warn about unexpected inputs
        if len(passed_kwargs) > 0:
            warnings.warn(f"Unexpected input '{passed_kwargs.keys()}' provided. This input will be ignored.")
        # Run the pipeline
        with torch.no_grad():
            try:
                _, state = self.blocks(self, state)
            except Exception:
                error_msg = f"Error in block: ({self.blocks.__class__.__name__}):\n"
                logger.error(error_msg)
                raise

        if output is None:
            return state

        if isinstance(output, str):
            return state.get(output)

        elif isinstance(output, (list, tuple)):
            return state.get(output)
        else:
            raise ValueError(f"Output '{output}' is not a valid output type")
