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

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Tuple, Union

import torch

from ..utils import get_logger


if TYPE_CHECKING:
    pass


logger = get_logger(__name__)  # pylint: disable=invalid-name


# TODO(aryan): add support for the following:
# - Unified Attention
# - More dispatcher attention backends
# - CFG/Data Parallel
# - Tensor Parallel


@dataclass
class ContextParallelConfig:
    """
    Configuration for context parallelism.

    Args:
        ring_degree (`int`, *optional*, defaults to `1`):
            Number of devices to use for ring attention within a context parallel region. Must be a divisor of the
            total number of devices in the context parallel mesh.
        ulysses_degree (`int`, *optional*, defaults to `1`):
            Number of devices to use for ulysses attention within a context parallel region. Must be a divisor of the
            total number of devices in the context parallel mesh.
        convert_to_fp32 (`bool`, *optional*, defaults to `True`):
            Whether to convert output and LSE to float32 for ring attention numerical stability.
        rotate_method (`str`, *optional*, defaults to `"allgather"`):
            Method to use for rotating key/value states across devices in ring attention. Currently, only `"allgather"`
            is supported.

    """

    ring_degree: Optional[int] = None
    ulysses_degree: Optional[int] = None
    convert_to_fp32: bool = True
    # TODO: support alltoall
    rotate_method: Literal["allgather", "alltoall"] = "allgather"

    _rank: int = None
    _world_size: int = None
    _device: torch.device = None
    _mesh: torch.distributed.device_mesh.DeviceMesh = None
    _flattened_mesh: torch.distributed.device_mesh.DeviceMesh = None
    _ring_mesh: torch.distributed.device_mesh.DeviceMesh = None
    _ulysses_mesh: torch.distributed.device_mesh.DeviceMesh = None
    _ring_local_rank: int = None
    _ulysses_local_rank: int = None

    def __post_init__(self):
        if self.ring_degree is None:
            self.ring_degree = 1
        if self.ulysses_degree is None:
            self.ulysses_degree = 1

    def setup(self, rank: int, world_size: int, device: torch.device, mesh: torch.distributed.device_mesh.DeviceMesh):
        self._rank = rank
        self._world_size = world_size
        self._device = device
        self._mesh = mesh
        if self.ring_degree is None:
            self.ring_degree = 1
        if self.ulysses_degree is None:
            self.ulysses_degree = 1
        if self.rotate_method != "allgather":
            raise NotImplementedError(
                f"Only rotate_method='allgather' is supported for now, but got {self.rotate_method}."
            )
        if self._flattened_mesh is None:
            self._flattened_mesh = self._mesh._flatten()
        if self._ring_mesh is None:
            self._ring_mesh = self._mesh["ring"]
        if self._ulysses_mesh is None:
            self._ulysses_mesh = self._mesh["ulysses"]
        if self._ring_local_rank is None:
            self._ring_local_rank = self._ring_mesh.get_local_rank()
        if self._ulysses_local_rank is None:
            self._ulysses_local_rank = self._ulysses_mesh.get_local_rank()


@dataclass
class ParallelConfig:
    """
    Configuration for applying different parallelisms.

    Args:
        context_parallel_config (`ContextParallelConfig`, *optional*):
            Configuration for context parallelism.
    """

    context_parallel_config: Optional[ContextParallelConfig] = None

    _rank: int = None
    _world_size: int = None
    _device: torch.device = None
    _cp_mesh: torch.distributed.device_mesh.DeviceMesh = None

    def setup(
        self,
        rank: int,
        world_size: int,
        device: torch.device,
        *,
        cp_mesh: Optional[torch.distributed.device_mesh.DeviceMesh] = None,
    ):
        self._rank = rank
        self._world_size = world_size
        self._device = device
        self._cp_mesh = cp_mesh
        if self.context_parallel_config is not None:
            self.context_parallel_config.setup(rank, world_size, device, cp_mesh)


@dataclass(frozen=True)
class ContextParallelInput:
    """
    Configuration for splitting an input tensor across context parallel region.

    Args:
        split_dim (`int`):
            The dimension along which to split the tensor.
        expected_dims (`int`, *optional*):
            The expected number of dimensions of the tensor. If provided, a check will be performed to ensure that the
            tensor has the expected number of dimensions before splitting.
        split_output (`bool`, *optional*, defaults to `False`):
            Whether to split the output tensor of the layer along the given `split_dim` instead of the input tensor.
            This is useful for layers whose outputs should be split after it does some preprocessing on the inputs (ex:
            RoPE).
    """

    split_dim: int
    expected_dims: Optional[int] = None
    split_output: bool = False

    def __repr__(self):
        return f"ContextParallelInput(split_dim={self.split_dim}, expected_dims={self.expected_dims}, split_output={self.split_output})"


@dataclass(frozen=True)
class ContextParallelOutput:
    """
    Configuration for gathering an output tensor across context parallel region.

    Args:
        gather_dim (`int`):
            The dimension along which to gather the tensor.
        expected_dims (`int`, *optional*):
            The expected number of dimensions of the tensor. If provided, a check will be performed to ensure that the
            tensor has the expected number of dimensions before gathering.
    """

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


# Example of a ContextParallelModelPlan (QwenImageTransformer2DModel):
#
# Each model should define a _cp_plan attribute that contains information on how to shard/gather
# tensors at different stages of the forward:
#
# ```python
# _cp_plan = {
#     "": {
#         "hidden_states": ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
#         "encoder_hidden_states": ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
#         "encoder_hidden_states_mask": ContextParallelInput(split_dim=1, expected_dims=2, split_output=False),
#     },
#     "pos_embed": {
#         0: ContextParallelInput(split_dim=0, expected_dims=2, split_output=True),
#         1: ContextParallelInput(split_dim=0, expected_dims=2, split_output=True),
#     },
#     "proj_out": ContextParallelOutput(gather_dim=1, expected_dims=3),
# }
# ```
#
# The dictionary is a set of module names mapped to their respective CP plan. The inputs/outputs of layers will be
# split/gathered according to this at the respective module level. Here, the following happens:
# - "":
#     we specify that we want to split the various inputs across the sequence dim in the pre-forward hook (i.e. before
#     the actual forward logic of the QwenImageTransformer2DModel is run, we will splitthe inputs)
# - "pos_embed":
#     we specify that we want to split the outputs of the RoPE layer. Since there are two outputs (imag & text freqs),
#     we can individually specify how they should be split
# - "proj_out":
#     before returning to the user, we gather the entire sequence on each rank in the post-forward hook (after the linear
#     layer forward has run).
#
# ContextParallelInput:
#     specifies how to split the input tensor in the pre-forward or post-forward hook of the layer it is attached to
#
# ContextParallelOutput:
#     specifies how to gather the input tensor in the post-forward hook in the layer it is attached to
