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

import inspect
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
from ..utils.hub_utils import validate_hf_hub_args
from .auto_pipeline import _get_model
from .pipeline_loading_utils import _fetch_class_library_tuple, _get_pipeline_class
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
        return self.outputs.get(key, default)

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
    optional_components = []
    required_components = []
    required_auxiliaries = []

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

    def __init__(self, **kwargs):
        self.components: Dict[str, Any] = {}
        self.auxiliaries: Dict[str, Any] = {}
        self.configs: Dict[str, Any] = {}

        # Process kwargs
        for key, value in kwargs.items():
            if key in self.required_components or key in self.optional_components:
                self.components[key] = value
            elif key in self.required_auxiliaries:
                self.auxiliaries[key] = value
            else:
                self.configs[key] = value

    @classmethod
    def from_pipe(cls, pipe: DiffusionPipeline, **kwargs):
        """
        Create a PipelineBlock instance from a diffusion pipeline object.

        Args:
            pipe: A `[DiffusionPipeline]` object.

        Returns:
            PipelineBlock: An instance initialized with the pipeline's components and configurations.
        """
        # add components
        expected_components = set(cls.required_components + cls.optional_components)
        # - components that are passed in kwargs
        components_to_add = {
            component_name: kwargs.pop(component_name)
            for component_name in expected_components
            if component_name in kwargs
        }
        # - components that are in the pipeline
        for component_name, component in pipe.components.items():
            if component_name in expected_components and component_name not in components_to_add:
                components_to_add[component_name] = component

        # add auxiliaries
        # - auxiliaries that are passed in kwargs
        auxiliaries_to_add = {k: kwargs.pop(k) for k in cls.required_auxiliaries if k in kwargs}
        # - auxiliaries that are in the pipeline
        for aux_name in cls.required_auxiliaries:
            if hasattr(pipe, aux_name) and aux_name not in auxiliaries_to_add:
                auxiliaries_to_add[aux_name] = getattr(pipe, aux_name)
        block_kwargs = {**components_to_add, **auxiliaries_to_add}

        # add pipeline configs
        init_params = inspect.signature(cls.__init__).parameters
        # modules info are also registered in the config as tuples, e.g. {'tokenizer': ('transformers', 'CLIPTokenizer')}
        # we need to exclude them for block_kwargs otherwise it will override the actual module
        expected_configs = {
            k
            for k in pipe.config.keys()
            if k in init_params and k not in expected_components and k not in cls.required_auxiliaries
        }

        for config_name in expected_configs:
            if config_name not in block_kwargs:
                if config_name in kwargs:
                    # - configs that are passed in kwargs
                    block_kwargs[config_name] = kwargs.pop(config_name)
                else:
                    # - configs that are in the pipeline
                    block_kwargs[config_name] = pipe.config[config_name]

        # Add any remaining relevant pipeline attributes
        for attr_name in dir(pipe):
            if attr_name not in block_kwargs and attr_name in init_params:
                block_kwargs[attr_name] = getattr(pipe, attr_name)

        return cls(**block_kwargs)

    def __call__(self, pipeline, state: PipelineState) -> PipelineState:
        raise NotImplementedError("__call__ method must be implemented in subclasses")

    def __repr__(self):
        class_name = self.__class__.__name__
        components = ", ".join(f"{k}={type(v).__name__}" for k, v in self.components.items())
        auxiliaries = ", ".join(f"{k}={type(v).__name__}" for k, v in self.auxiliaries.items())
        configs = ", ".join(f"{k}={v}" for k, v in self.configs.items())
        inputs = ", ".join(f"{name}={default}" for name, default in self.inputs)
        intermediates_inputs = ", ".join(self.intermediates_inputs)
        intermediates_outputs = ", ".join(self.intermediates_outputs)

        return (
            f"{class_name}(\n"
            f"  components: {components}\n"
            f"  auxiliaries: {auxiliaries}\n"
            f"  configs: {configs}\n"
            f"  inputs: {inputs}\n"
            f"  intermediates_inputs: {intermediates_inputs}\n"
            f"  intermediates_outputs: {intermediates_outputs}\n"
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

        # Validate all required components and auxiliaries after consolidation
        for block in pipeline_blocks:
            for required_component in block.required_components:
                if (
                    not hasattr(self, required_component)
                    and required_component not in components_to_add
                    or getattr(self, required_component, None) is None
                    and components_to_add.get(required_component) is None
                ):
                    raise ValueError(
                        f"Cannot add block {block.__class__.__name__}: Required component {required_component} not found in pipeline"
                    )

            for required_auxiliary in block.required_auxiliaries:
                if (
                    not hasattr(self, required_auxiliary)
                    and required_auxiliary not in auxiliaries_to_add
                    or getattr(self, required_auxiliary, None) is None
                    and auxiliaries_to_add.get(required_auxiliary) is None
                ):
                    raise ValueError(
                        f"Cannot add block {block.__class__.__name__}: Required auxiliary {required_auxiliary} not found in pipeline"
                    )

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

    @classmethod
    @validate_hf_hub_args
    def from_pretrained(cls, pretrained_model_or_path, **kwargs):
        # (1) create the base pipeline
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

        config = cls.load_config(pretrained_model_or_path, **load_config_kwargs)
        base_pipeline_class_name = config["_class_name"]
        base_pipeline_class = _get_pipeline_class(cls, config)

        kwargs = {**load_config_kwargs, **kwargs}
        base_pipeline = base_pipeline_class.from_pretrained(pretrained_model_or_path, **kwargs)

        # (2) map the base pipeline to pipeline blocks
        modular_pipeline_class_name = MODULAR_PIPELINE_MAPPING[_get_model(base_pipeline_class_name)]
        modular_pipeline_class = _get_pipeline_class(cls, config=None, class_name=modular_pipeline_class_name)

        # (3) create the pipeline blocks
        pipeline_blocks = [
            block_class.from_pipe(base_pipeline) for block_class in modular_pipeline_class.default_pipeline_blocks
        ]

        # (4) create the builder
        builder = modular_pipeline_class()
        builder.add_blocks(pipeline_blocks)

        return builder

    @classmethod
    def from_pipe(cls, pipeline, **kwargs):
        base_pipeline_class_name = pipeline.__class__.__name__
        modular_pipeline_class_name = MODULAR_PIPELINE_MAPPING[_get_model(base_pipeline_class_name)]
        modular_pipeline_class = _get_pipeline_class(cls, config=None, class_name=modular_pipeline_class_name)

        pipeline_blocks = []
        # Create each block, passing only unused items that the block expects
        for block_class in modular_pipeline_class.default_pipeline_blocks:
            expected_components = set(block_class.required_components + block_class.optional_components)
            expected_auxiliaries = set(block_class.required_auxiliaries)

            # Get init parameters to check for expected configs
            init_params = inspect.signature(block_class.__init__).parameters
            expected_configs = {
                k for k in init_params if k not in expected_components and k not in expected_auxiliaries
            }

            block_kwargs = {}

            for key, value in kwargs.items():
                if key in expected_components or key in expected_auxiliaries or key in expected_configs:
                    block_kwargs[key] = value

            # Create the block with filtered kwargs
            block = block_class.from_pipe(pipeline, **block_kwargs)
            pipeline_blocks.append(block)

        # Create and setup the builder
        builder = modular_pipeline_class()
        builder.add_blocks(pipeline_blocks)

        # Warn about unused kwargs
        unused_kwargs = {
            k: v
            for k, v in kwargs.items()
            if not any(
                k in block.components or k in block.auxiliaries or k in block.configs for block in pipeline_blocks
            )
        }
        if unused_kwargs:
            logger.warning(
                f"The following items were passed but not used by any pipeline block: {list(unused_kwargs.keys())}"
            )

        return builder

    def run_blocks(self, state: PipelineState = None, **kwargs):
        """
        Run one or more blocks in sequence, optionally you can pass a previous pipeline state.
        """
        if state is None:
            state = PipelineState()

        # Make a copy of the input kwargs
        input_params = kwargs.copy()

        default_params = self.default_call_parameters

        # user can pass the intermediate of the first block
        for name in self.pipeline_blocks[0].intermediates_inputs:
            if name in input_params:
                state.add_intermediate(name, input_params.pop(name))

        # Add inputs to state, using defaults if not provided in the kwargs or the state
        # if same input already in the state, will override it if provided in the kwargs
        for name, default in default_params.items():
            if name in input_params:
                state.add_input(name, input_params.pop(name))
            elif name not in state.inputs:
                state.add_input(name, default)

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

        return state

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
                except Exception:
                    error_msg = f"Error in block: ({block.__class__.__name__}):\n"
                    logger.error(error_msg)
                    raise

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
            output += f"{i}. {block.__class__.__name__}\n"

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
