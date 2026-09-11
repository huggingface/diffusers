# Copyright 2026 The HuggingFace Team. All rights reserved.
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
from math import prod
from typing import Protocol

import torch


class Transform(Protocol):
    """A reversible operation on an ordered group of tensors.

    Implementations must preserve dtype/device and must not modify their inputs. Outputs may share storage with inputs.
    Configuration needed by either direction belongs on the transform, so export does not depend on an earlier import.
    """

    def forward(self, tensors: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]: ...

    def inverse(self, tensors: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]: ...


@dataclass(frozen=True)
class Identity:
    """Return the tensors unchanged, including their storage."""

    def forward(self, tensors: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        return tensors

    def inverse(self, tensors: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        return tensors


@dataclass(frozen=True)
class FoldLinearGate:
    """Fold a static output-channel gate into a linear layer, exporting a canonical unit gate.

    Original gate factorization is lost. Diffusers -> original -> Diffusers preserves float32/float16/bfloat16 weights,
    while the other round trip produces equivalent folded parameters. Multiplication follows the source importer's
    float32 arithmetic.
    """

    bias: bool = True
    lossless = False

    def forward(self, tensors):
        if len(tensors) != 2 + int(self.bias):
            raise ValueError("Expected gate, weight, and optional bias.")
        gate, weight, *bias = tensors
        if weight.ndim != 2 or gate.shape != weight.shape[:1]:
            raise ValueError("Gate must be a vector matching the linear output channels.")
        if bias and bias[0].shape != gate.shape:
            raise ValueError("Bias must match the gate shape.")
        if any(t.device != weight.device or t.dtype != weight.dtype for t in tensors):
            raise ValueError("Gate and linear parameters must have the same dtype and device.")
        if not weight.is_floating_point():
            raise ValueError("Gate folding requires floating-point parameters.")
        outputs = [(gate.float().unsqueeze(1) * weight.float()).to(weight.dtype)]
        if bias:
            outputs.append((gate.float() * bias[0].float()).to(weight.dtype))
        return tuple(outputs)

    def inverse(self, tensors):
        if len(tensors) != 1 + int(self.bias) or tensors[0].ndim != 2:
            raise ValueError("Expected linear weight and optional bias.")
        weight = tensors[0]
        if self.bias and (
            tensors[1].shape != weight.shape[:1]
            or tensors[1].dtype != weight.dtype
            or tensors[1].device != weight.device
        ):
            raise ValueError("Linear weight and bias must have matching shape, dtype and device.")
        return (weight.new_ones(weight.shape[0]), *tensors)


@dataclass(frozen=True)
class Split:
    """Split one tensor into explicit sizes along `dim`; concatenate those pieces in the inverse direction."""

    sizes: tuple[int, ...]
    dim: int = 0

    def __post_init__(self):
        object.__setattr__(self, "sizes", tuple(self.sizes))
        if not self.sizes or any(size <= 0 for size in self.sizes):
            raise ValueError("Split sizes must be positive.")

    def forward(self, tensors: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        if len(tensors) != 1:
            raise ValueError(f"Split expects one tensor, got {len(tensors)}.")
        (tensor,) = tensors
        if not -tensor.ndim <= self.dim < tensor.ndim:
            raise ValueError(f"Invalid split dimension {self.dim} for shape {tuple(tensor.shape)}.")
        if tensor.shape[self.dim] != sum(self.sizes):
            raise ValueError(f"Split sizes {self.sizes} do not match dimension {self.dim} of {tuple(tensor.shape)}.")
        return tensor.split(self.sizes, dim=self.dim)

    def inverse(self, tensors: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        if len(tensors) != len(self.sizes):
            raise ValueError(f"Expected {len(self.sizes)} split pieces, got {len(tensors)}.")
        first = tensors[0]
        if not -first.ndim <= self.dim < first.ndim:
            raise ValueError(f"Invalid split dimension {self.dim} for shape {tuple(first.shape)}.")
        dim = self.dim % first.ndim
        for tensor, size in zip(tensors, self.sizes):
            expected_shape = list(first.shape)
            expected_shape[dim] = size
            if tuple(tensor.shape) != tuple(expected_shape):
                raise ValueError(f"Expected split piece shape {tuple(expected_shape)}, got {tuple(tensor.shape)}.")
            if tensor.dtype != first.dtype or tensor.device != first.device:
                raise ValueError(
                    "Split pieces must have the same dtype and device; implicit casting is not supported."
                )
        return (torch.cat(tensors, dim=dim),)


@dataclass(frozen=True)
class ReorderChunks:
    """Permute equal-sized chunks of one tensor. `order[i]` identifies the source chunk at output position `i`."""

    order: tuple[int, ...]
    dim: int = 0

    def __post_init__(self):
        object.__setattr__(self, "order", tuple(self.order))
        if not self.order or sorted(self.order) != list(range(len(self.order))):
            raise ValueError("Chunk order must be a permutation of consecutive indices starting at zero.")

    def forward(self, tensors: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        if len(tensors) != 1:
            raise ValueError(f"ReorderChunks expects one tensor, got {len(tensors)}.")
        (tensor,) = tensors
        if not -tensor.ndim <= self.dim < tensor.ndim:
            raise ValueError(f"Invalid chunk dimension {self.dim} for shape {tuple(tensor.shape)}.")
        if tensor.shape[self.dim] == 0 or tensor.shape[self.dim] % len(self.order):
            raise ValueError(
                f"Cannot divide dimension {self.dim} of {tuple(tensor.shape)} into {len(self.order)} chunks."
            )
        chunks = tensor.chunk(len(self.order), dim=self.dim)
        return (torch.cat(tuple(chunks[i] for i in self.order), dim=self.dim),)

    def inverse(self, tensors: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        inverse_order = tuple(self.order.index(i) for i in range(len(self.order)))
        return ReorderChunks(inverse_order, self.dim).forward(tensors)


@dataclass(frozen=True)
class Reshape:
    """Change a tensor's shape with both shapes specified, including singleton dimensions."""

    original_shape: tuple[int, ...]
    diffusers_shape: tuple[int, ...]

    def __post_init__(self):
        object.__setattr__(self, "original_shape", tuple(self.original_shape))
        object.__setattr__(self, "diffusers_shape", tuple(self.diffusers_shape))
        if any(size < 0 for size in self.original_shape + self.diffusers_shape):
            raise ValueError("Reshape requires explicit, nonnegative dimensions.")
        if prod(self.original_shape) != prod(self.diffusers_shape):
            raise ValueError("Reshape must preserve the number of elements.")

    def forward(self, tensors):
        (tensor,) = tensors
        if tuple(tensor.shape) != self.original_shape:
            raise ValueError(f"Expected shape {self.original_shape}, got {tuple(tensor.shape)}.")
        return (tensor.reshape(self.diffusers_shape),)

    def inverse(self, tensors):
        return Reshape(self.diffusers_shape, self.original_shape).forward(tensors)


@dataclass(frozen=True)
class Squeeze:
    """Remove one declared singleton axis and restore it on export, preserving all other dimensions."""

    dim: int
    ndim: int

    def __post_init__(self):
        if not 0 <= self.dim < self.ndim:
            raise ValueError("The singleton dimension must be within the declared tensor rank.")

    def forward(self, tensors: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        (tensor,) = tensors
        if tensor.ndim != self.ndim or tensor.shape[self.dim] != 1:
            raise ValueError(
                f"Expected rank {self.ndim} with a singleton axis at {self.dim}, got {tuple(tensor.shape)}."
            )
        return (tensor.squeeze(self.dim),)

    def inverse(self, tensors: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        (tensor,) = tensors
        if tensor.ndim != self.ndim - 1:
            raise ValueError(f"Expected rank {self.ndim - 1}, got {tensor.ndim}.")
        return (tensor.unsqueeze(self.dim),)


@dataclass(frozen=True)
class Permute:
    """Permute a tensor's axes; reverse with the inverse axis permutation."""

    dims: tuple[int, ...]

    def __post_init__(self):
        object.__setattr__(self, "dims", tuple(self.dims))
        if sorted(self.dims) != list(range(len(self.dims))):
            raise ValueError("Axes must be a permutation of consecutive indices starting at zero.")

    def forward(self, tensors):
        (tensor,) = tensors
        if tensor.ndim != len(self.dims):
            raise ValueError(f"Expected rank {len(self.dims)}, got {tensor.ndim}.")
        return (tensor.permute(self.dims),)

    def inverse(self, tensors):
        return Permute(tuple(self.dims.index(i) for i in range(len(self.dims)))).forward(tensors)


@dataclass(frozen=True)
class Chain:
    """Compose reversible operations, reversing their order for export."""

    transforms: tuple[Transform, ...]

    def __post_init__(self):
        object.__setattr__(self, "transforms", tuple(self.transforms))

    @property
    def lossless(self):
        return all(getattr(transform, "lossless", True) for transform in self.transforms)

    def forward(self, tensors):
        for transform in self.transforms:
            tensors = transform.forward(tensors)
        return tensors

    def inverse(self, tensors):
        for transform in reversed(self.transforms):
            tensors = transform.inverse(tensors)
        return tensors


@dataclass(frozen=True)
class MergeEqual:
    """Merge equal copies of a buffer; restore all copies on export without losing information."""

    copies: int

    def __post_init__(self):
        if self.copies < 1:
            raise ValueError("The number of copies must be positive.")

    def forward(self, tensors):
        if len(tensors) != self.copies:
            raise ValueError(f"Expected {self.copies} equal tensors, got {len(tensors)}.")
        first = tensors[0]
        for tensor in tensors[1:]:
            if tensor.dtype != first.dtype or tensor.device != first.device or not torch.equal(tensor, first):
                raise ValueError("Cannot merge unequal tensors without losing information.")
        return (first,)

    def inverse(self, tensors):
        (tensor,) = tensors
        return (tensor,) * self.copies


@dataclass(frozen=True)
class Reverse:
    """Use an existing transform in the opposite direction."""

    transform: Transform

    @property
    def lossless(self):
        return getattr(self.transform, "lossless", True)

    def forward(self, tensors):
        return self.transform.inverse(tensors)

    def inverse(self, tensors):
        return self.transform.forward(tensors)


@dataclass(frozen=True, eq=False)
class WithConstants:
    """Attach config-derived buffers to one tensor and verify those buffers before removing them on export.

    Floating buffers use the anchor tensor's dtype; integer and boolean buffers retain their own dtype.
    """

    values: tuple[torch.Tensor, ...]

    def __post_init__(self):
        object.__setattr__(self, "values", tuple(value.detach().clone() for value in self.values))

    def forward(self, tensors):
        (anchor,) = tensors
        return (anchor,) + tuple(
            value.to(device=anchor.device, dtype=anchor.dtype if value.is_floating_point() else value.dtype)
            for value in self.values
        )

    def inverse(self, tensors):
        if len(tensors) != len(self.values) + 1:
            raise ValueError(f"Expected an anchor and {len(self.values)} constant buffers.")
        expected = self.forward((tensors[0],))
        for actual, reference in zip(tensors[1:], expected[1:]):
            if (
                actual.dtype != reference.dtype
                or actual.device != reference.device
                or not torch.equal(actual, reference)
            ):
                raise ValueError(
                    "A config-derived buffer was changed and cannot be represented in the original format."
                )
        return (tensors[0],)
