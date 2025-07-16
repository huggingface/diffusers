# Experimental parallelism support for Diffusers.
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

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple, Union

import torch

from ..utils import get_logger


logger = get_logger(__name__)  # pylint: disable=invalid-name


# TODO(aryan): add support for the following:
# - Unified Attention
# - More dispatcher attention backends
# - CFG/Data Parallel
# - Tensor Parallel


@dataclass
class ParallelConfig:
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
