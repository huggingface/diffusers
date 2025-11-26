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

import inspect
from dataclasses import dataclass
from typing import Dict, List, Type, Union

import torch


if torch.distributed.is_available():
    import torch.distributed._functional_collectives as funcol

from ..models._modeling_parallel import (
    ContextParallelConfig,
    ContextParallelInput,
    ContextParallelModelPlan,
    ContextParallelOutput,
)
from ..utils import get_logger
from ..utils.torch_utils import unwrap_module
from .hooks import HookRegistry, ModelHook


logger = get_logger(__name__)  # pylint: disable=invalid-name

_CONTEXT_PARALLEL_INPUT_HOOK_TEMPLATE = "cp_input---{}"
_CONTEXT_PARALLEL_OUTPUT_HOOK_TEMPLATE = "cp_output---{}"


# TODO(aryan): consolidate with ._helpers.TransformerBlockMetadata
@dataclass
class ModuleForwardMetadata:
    cached_parameter_indices: Dict[str, int] = None
    _cls: Type = None

    def _get_parameter_from_args_kwargs(self, identifier: str, args=(), kwargs=None):
        kwargs = kwargs or {}

        if identifier in kwargs:
            return kwargs[identifier], True, None

        if self.cached_parameter_indices is not None:
            index = self.cached_parameter_indices.get(identifier, None)
            if index is None:
                raise ValueError(f"Parameter '{identifier}' not found in cached indices.")
            return args[index], False, index

        if self._cls is None:
            raise ValueError("Model class is not set for metadata.")

        parameters = list(inspect.signature(self._cls.forward).parameters.keys())
        parameters = parameters[1:]  # skip `self`
        self.cached_parameter_indices = {param: i for i, param in enumerate(parameters)}

        if identifier not in self.cached_parameter_indices:
            raise ValueError(f"Parameter '{identifier}' not found in function signature but was requested.")

        index = self.cached_parameter_indices[identifier]

        if index >= len(args):
            raise ValueError(f"Expected {index} arguments but got {len(args)}.")

        return args[index], False, index


def apply_context_parallel(
    module: torch.nn.Module,
    parallel_config: ContextParallelConfig,
    plan: Dict[str, ContextParallelModelPlan],
) -> None:
    """Apply context parallel on a model."""
    logger.debug(f"Applying context parallel with CP mesh: {parallel_config._mesh} and plan: {plan}")

    for module_id, cp_model_plan in plan.items():
        submodule = _get_submodule_by_name(module, module_id)
        if not isinstance(submodule, list):
            submodule = [submodule]

        logger.debug(f"Applying ContextParallelHook to {module_id=} identifying a total of {len(submodule)} modules")

        for m in submodule:
            if isinstance(cp_model_plan, dict):
                hook = ContextParallelSplitHook(cp_model_plan, parallel_config)
                hook_name = _CONTEXT_PARALLEL_INPUT_HOOK_TEMPLATE.format(module_id)
            elif isinstance(cp_model_plan, (ContextParallelOutput, list, tuple)):
                if isinstance(cp_model_plan, ContextParallelOutput):
                    cp_model_plan = [cp_model_plan]
                if not all(isinstance(x, ContextParallelOutput) for x in cp_model_plan):
                    raise ValueError(f"Expected all elements of cp_model_plan to be CPOutput, but got {cp_model_plan}")
                hook = ContextParallelGatherHook(cp_model_plan, parallel_config)
                hook_name = _CONTEXT_PARALLEL_OUTPUT_HOOK_TEMPLATE.format(module_id)
            else:
                raise ValueError(f"Unsupported context parallel model plan type: {type(cp_model_plan)}")
            registry = HookRegistry.check_if_exists_or_initialize(m)
            registry.register_hook(hook, hook_name)


def remove_context_parallel(module: torch.nn.Module, plan: Dict[str, ContextParallelModelPlan]) -> None:
    for module_id, cp_model_plan in plan.items():
        submodule = _get_submodule_by_name(module, module_id)
        if not isinstance(submodule, list):
            submodule = [submodule]

        for m in submodule:
            registry = HookRegistry.check_if_exists_or_initialize(m)
            if isinstance(cp_model_plan, dict):
                hook_name = _CONTEXT_PARALLEL_INPUT_HOOK_TEMPLATE.format(module_id)
            elif isinstance(cp_model_plan, (ContextParallelOutput, list, tuple)):
                hook_name = _CONTEXT_PARALLEL_OUTPUT_HOOK_TEMPLATE.format(module_id)
            else:
                raise ValueError(f"Unsupported context parallel model plan type: {type(cp_model_plan)}")
            registry.remove_hook(hook_name)


class ContextParallelSplitHook(ModelHook):
    def __init__(self, metadata: ContextParallelModelPlan, parallel_config: ContextParallelConfig) -> None:
        super().__init__()
        self.metadata = metadata
        self.parallel_config = parallel_config
        self.module_forward_metadata = None

    def initialize_hook(self, module):
        cls = unwrap_module(module).__class__
        self.module_forward_metadata = ModuleForwardMetadata(_cls=cls)
        return module

    def pre_forward(self, module, *args, **kwargs):
        args_list = list(args)

        for name, cpm in self.metadata.items():
            if isinstance(cpm, ContextParallelInput) and cpm.split_output:
                continue

            # Maybe the parameter was passed as a keyword argument
            input_val, is_kwarg, index = self.module_forward_metadata._get_parameter_from_args_kwargs(
                name, args_list, kwargs
            )

            if input_val is None:
                continue

            # The input_val may be a tensor or list/tuple of tensors. In certain cases, user may specify to shard
            # the output instead of input for a particular layer by setting split_output=True
            if isinstance(input_val, torch.Tensor):
                input_val = self._prepare_cp_input(input_val, cpm)
            elif isinstance(input_val, (list, tuple)):
                if len(input_val) != len(cpm):
                    raise ValueError(
                        f"Expected input model plan to have {len(input_val)} elements, but got {len(cpm)}."
                    )
                sharded_input_val = []
                for i, x in enumerate(input_val):
                    if torch.is_tensor(x) and not cpm[i].split_output:
                        x = self._prepare_cp_input(x, cpm[i])
                    sharded_input_val.append(x)
                input_val = sharded_input_val
            else:
                raise ValueError(f"Unsupported input type: {type(input_val)}")

            if is_kwarg:
                kwargs[name] = input_val
            elif index is not None and index < len(args_list):
                args_list[index] = input_val
            else:
                raise ValueError(
                    f"An unexpected error occurred while processing the input '{name}'. Please open an "
                    f"issue at https://github.com/huggingface/diffusers/issues and provide a minimal reproducible "
                    f"example along with the full stack trace."
                )

        return tuple(args_list), kwargs

    def post_forward(self, module, output):
        is_tensor = isinstance(output, torch.Tensor)
        is_tensor_list = isinstance(output, (list, tuple)) and all(isinstance(x, torch.Tensor) for x in output)

        if not is_tensor and not is_tensor_list:
            raise ValueError(f"Expected output to be a tensor or a list/tuple of tensors, but got {type(output)}.")

        output = [output] if is_tensor else list(output)
        for index, cpm in self.metadata.items():
            if not isinstance(cpm, ContextParallelInput) or not cpm.split_output:
                continue
            if index >= len(output):
                raise ValueError(f"Index {index} out of bounds for output of length {len(output)}.")
            current_output = output[index]
            current_output = self._prepare_cp_input(current_output, cpm)
            output[index] = current_output

        return output[0] if is_tensor else tuple(output)

    def _prepare_cp_input(self, x: torch.Tensor, cp_input: ContextParallelInput) -> torch.Tensor:
        if cp_input.expected_dims is not None and x.dim() != cp_input.expected_dims:
            logger.warning_once(
                f"Expected input tensor to have {cp_input.expected_dims} dimensions, but got {x.dim()} dimensions, split will not be applied."
            )
            return x
        else:
            return EquipartitionSharder.shard(x, cp_input.split_dim, self.parallel_config._flattened_mesh)


class ContextParallelGatherHook(ModelHook):
    def __init__(self, metadata: ContextParallelModelPlan, parallel_config: ContextParallelConfig) -> None:
        super().__init__()
        self.metadata = metadata
        self.parallel_config = parallel_config

    def post_forward(self, module, output):
        is_tensor = isinstance(output, torch.Tensor)

        if is_tensor:
            output = [output]
        elif not (isinstance(output, (list, tuple)) and all(isinstance(x, torch.Tensor) for x in output)):
            raise ValueError(f"Expected output to be a tensor or a list/tuple of tensors, but got {type(output)}.")

        output = list(output)

        if len(output) != len(self.metadata):
            raise ValueError(f"Expected output to have {len(self.metadata)} elements, but got {len(output)}.")

        for i, cpm in enumerate(self.metadata):
            if cpm is None:
                continue
            output[i] = EquipartitionSharder.unshard(output[i], cpm.gather_dim, self.parallel_config._flattened_mesh)

        return output[0] if is_tensor else tuple(output)


class AllGatherFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, dim, group):
        ctx.dim = dim
        ctx.group = group
        ctx.world_size = torch.distributed.get_world_size(group)
        ctx.rank = torch.distributed.get_rank(group)
        return funcol.all_gather_tensor(tensor, dim, group=group)

    @staticmethod
    def backward(ctx, grad_output):
        grad_chunks = torch.chunk(grad_output, ctx.world_size, dim=ctx.dim)
        return grad_chunks[ctx.rank], None, None


class EquipartitionSharder:
    @classmethod
    def shard(cls, tensor: torch.Tensor, dim: int, mesh: torch.distributed.device_mesh.DeviceMesh) -> torch.Tensor:
        # NOTE: the following assertion does not have to be true in general. We simply enforce it for now
        # because the alternate case has not yet been tested/required for any model.
        assert tensor.size()[dim] % mesh.size() == 0, (
            "Tensor size along dimension to be sharded must be divisible by mesh size"
        )

        # The following is not fullgraph compatible with Dynamo (fails in DeviceMesh.get_rank)
        # return tensor.chunk(mesh.size(), dim=dim)[mesh.get_rank()]

        return tensor.chunk(mesh.size(), dim=dim)[torch.distributed.get_rank(mesh.get_group())]

    @classmethod
    def unshard(cls, tensor: torch.Tensor, dim: int, mesh: torch.distributed.device_mesh.DeviceMesh) -> torch.Tensor:
        tensor = tensor.contiguous()
        tensor = AllGatherFunction.apply(tensor, dim, mesh.get_group())
        return tensor


def _get_submodule_by_name(model: torch.nn.Module, name: str) -> Union[torch.nn.Module, List[torch.nn.Module]]:
    if name.count("*") > 1:
        raise ValueError("Wildcard '*' can only be used once in the name")
    return _find_submodule_by_name(model, name)


def _find_submodule_by_name(model: torch.nn.Module, name: str) -> Union[torch.nn.Module, List[torch.nn.Module]]:
    if name == "":
        return model
    first_atom, remaining_name = name.split(".", 1) if "." in name else (name, "")
    if first_atom == "*":
        if not isinstance(model, torch.nn.ModuleList):
            raise ValueError("Wildcard '*' can only be used with ModuleList")
        submodules = []
        for submodule in model:
            subsubmodules = _find_submodule_by_name(submodule, remaining_name)
            if not isinstance(subsubmodules, list):
                subsubmodules = [subsubmodules]
            submodules.extend(subsubmodules)
        return submodules
    else:
        if hasattr(model, first_atom):
            submodule = getattr(model, first_atom)
            return _find_submodule_by_name(submodule, remaining_name)
        else:
            raise ValueError(f"'{first_atom}' is not a submodule of '{model.__class__.__name__}'")
