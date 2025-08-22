# 🚨🚨🚨 Experimental parallelism support for Diffusers 🚨🚨🚨
# Experimental changes are subject to change and APIs may break without warning.

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

import contextlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Tuple, Union

import torch

from ..utils import get_logger


if TYPE_CHECKING:
    from ..pipelines.pipeline_utils import DiffusionPipeline
    from .modeling_utils import ModelMixin


logger = get_logger(__name__)  # pylint: disable=invalid-name


# TODO(aryan): add support for the following:
# - Unified Attention
# - More dispatcher attention backends
# - CFG/Data Parallel
# - Tensor Parallel


@dataclass
class ParallelConfig:
    ring_degree: Optional[int] = None
    ulysses_degree: Optional[int] = None

    def __post_init__(self):
        if self.ring_degree is None:
            self.ring_degree = 1
        if self.ulysses_degree is None:
            self.ulysses_degree = 1


@dataclass
class _InternalParallelConfig:
    rank: int
    world_size: int
    ring_degree: int
    ulysses_degree: int
    device: torch.device
    cp_mesh: torch.distributed.device_mesh.DeviceMesh

    # Whether to convert output and LSE to float32 for ring attention numerical stability
    convert_to_fp32: bool = True
    # TODO: support alltoall
    rotate_method: Literal["allgather", "alltoall"] = "allgather"

    _flattened_mesh: torch.distributed.device_mesh.DeviceMesh = None
    _ring_mesh: torch.distributed.device_mesh.DeviceMesh = None
    _ulysses_mesh: torch.distributed.device_mesh.DeviceMesh = None
    _ring_local_rank: int = None
    _ulysses_local_rank: int = None

    def __post_init__(self):
        if self.rotate_method != "allgather":
            raise ValueError(f"Only rotate_method='allgather' is supported for now, but got {self.rotate_method}.")
        if self._flattened_mesh is None:
            self._flattened_mesh = self.cp_mesh._flatten()
        if self._ring_mesh is None:
            self._ring_mesh = self.cp_mesh["ring"]
        if self._ulysses_mesh is None:
            self._ulysses_mesh = self.cp_mesh["ulysses"]
        if self._ring_local_rank is None:
            self._ring_local_rank = self._ring_mesh.get_local_rank()
        if self._ulysses_local_rank is None:
            self._ulysses_local_rank = self._ulysses_mesh.get_local_rank()


@dataclass(frozen=True)
class ContextParallelInput:
    split_dim: int
    expected_dims: Optional[int] = None
    split_output: bool = False

    def __repr__(self):
        return f"ContextParallelInput(split_dim={self.split_dim}, expected_dims={self.expected_dims}, split_output={self.split_output})"


@dataclass(frozen=True)
class ContextParallelOutput:
    gather_dim: int
    expected_dims: Optional[int] = None

    def __repr__(self):
        return f"ContextParallelOutput(gather_dim={self.gather_dim}, expected_dims={self.expected_dims})"


# A dictionary where keys denote the input to be split across context parallel region, and the
# value denotes the sharding configuration.
# If the key is a string, it denotes the name of the parameter in the forward function.
# If the key is an integer, split_output must be set to True, and it denotes the index of the output
# to be split across context parallel region.
ContextParallelInputType = Dict[
    Union[str, int], Union[ContextParallelInput, List[ContextParallelInput], Tuple[ContextParallelInput, ...]]
]

# A dictionary where keys denote the output to be gathered across context parallel region, and the
# value denotes the gathering configuration.
ContextParallelOutputType = Union[
    ContextParallelOutput, List[ContextParallelOutput], Tuple[ContextParallelOutput, ...]
]

# A dictionary where keys denote the module id, and the value denotes how the inputs/outputs of
# the module should be split/gathered across context parallel region.
ContextParallelModelPlan = Dict[str, Union[ContextParallelInputType, ContextParallelOutputType]]


_ENABLE_PARALLELISM_WARN_ONCE = False


@contextlib.contextmanager
def enable_parallelism(model_or_pipeline: Union["DiffusionPipeline", "ModelMixin"]):
    from diffusers import DiffusionPipeline, ModelMixin

    from .attention_dispatch import _AttentionBackendRegistry

    global _ENABLE_PARALLELISM_WARN_ONCE
    if not _ENABLE_PARALLELISM_WARN_ONCE:
        logger.warning(
            "Support for `enable_parallelism` is experimental and the API may be subject to change in the future."
        )
        _ENABLE_PARALLELISM_WARN_ONCE = True

    if isinstance(model_or_pipeline, DiffusionPipeline):
        parallelized_components = [
            (name, component)
            for name, component in model_or_pipeline.components.items()
            if getattr(component, "_internal_parallel_config", None) is not None
        ]
        if len(parallelized_components) > 1:
            raise ValueError(
                "Enabling parallelism on a pipeline is not possible when multiple internal components are parallelized. Please run "
                "different stages of the pipeline separately with `enable_parallelism` on each component manually."
            )
        if len(parallelized_components) == 0:
            raise ValueError(
                "No parallelized components found in the pipeline. Please ensure at least one component is parallelized."
            )
        _, model_or_pipeline = parallelized_components[0]
    elif isinstance(model_or_pipeline, ModelMixin):
        if getattr(model_or_pipeline, "_internal_parallel_config", None) is None:
            raise ValueError(
                "The model is not parallelized. Please ensure the model is parallelized with `.parallelize()` before using this context manager."
            )
    else:
        raise TypeError(
            f"Expected a `DiffusionPipeline` or `ModelMixin` instance, but got {type(model_or_pipeline)}. Please provide a valid model or pipeline."
        )

    old_parallel_config = _AttentionBackendRegistry._parallel_config
    _AttentionBackendRegistry._parallel_config = model_or_pipeline._internal_parallel_config

    yield

    _AttentionBackendRegistry._parallel_config = old_parallel_config
