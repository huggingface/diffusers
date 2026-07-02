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

import torch

from ..models._modeling_parallel import TensorParallelConfig
from ..utils import get_logger


logger = get_logger(__name__)  # pylint: disable=invalid-name


class PackedColwiseParallel:
    """Column-wise sharding for fused projections with heterogeneous block structure.

    ``blocks`` is a list of proportional integers whose sum divides the weight's row count. For example, ``[1, 1]`` for
    a SwiGLU gate+linear projection (two equal halves) or ``[1, 1, 1, 3, 3]`` for a Q+K+V+gate+linear projection with
    ``mlp_ratio=3``. If ``blocks`` is ``None``, the Linear module must carry a ``_tp_packed_col_blocks`` attribute set
    during model ``__init__``.
    """

    def __init__(self, blocks: "list[int] | None" = None):
        self.blocks = blocks


class PackedRowwiseParallel:
    """Row-wise sharding for fused projections with heterogeneous block structure.

    ``blocks`` describes the input-column partition of the fused Linear (e.g. ``[1, 3]`` when the input concatenates an
    attention projection and an MLP projection with ``mlp_ratio=3``). If ``blocks`` is ``None``, the module must carry
    a ``_tp_packed_row_blocks`` attribute.
    """

    def __init__(self, blocks: "list[int] | None" = None):
        self.blocks = blocks


def _blocks_to_block_sizes(total_size: int, blocks: "list[int]") -> "list[int]":
    """Convert proportional block counts to absolute sizes.

    ``blocks`` is a list of positive integers interpreted as proportional weights. Their sum must divide ``total_size``
    evenly. Returns a list of absolute sizes that sum to ``total_size``.
    """
    total = sum(blocks)
    if total_size % total != 0:
        raise ValueError(
            f"Cannot split {total_size} into proportional blocks {blocks}: "
            f"sum({blocks})={total} does not divide {total_size}."
        )
    unit = total_size // total
    return [b * unit for b in blocks]


def _resolve_tp_plan(model: torch.nn.Module, tp_plan: dict) -> list:
    """Group a flat ``_tp_plan`` into per-block ``(submodule, {relative_path: style})`` plans.

    Each glob is split at its single ``*``; the prefix must resolve to a ``ModuleList`` and the suffix is the
    per-element key. Grouping by block lets the caller issue one ``parallelize_module`` call per block, which
    ``RowwiseParallel`` needs to attach its input redistribution at the block boundary.
    """
    grouped: dict[int, tuple] = {}
    order: list[int] = []

    for pattern, style in tp_plan.items():
        if pattern.count("*") > 1:
            raise ValueError(f"Wildcard '*' can only be used once in a `_tp_plan` key, got '{pattern}'.")

        if "*" in pattern:
            prefix, _, suffix = pattern.partition("*")
            container = model
            for atom in prefix.strip(".").split("."):
                container = getattr(container, atom)
            if not isinstance(container, torch.nn.ModuleList):
                raise ValueError(
                    f"`_tp_plan` wildcard '{pattern}' must expand over a `ModuleList`, but "
                    f"'{prefix.strip('.')}' resolved to '{container.__class__.__name__}'."
                )
            relative, blocks = suffix.strip("."), list(container)
        else:
            relative, blocks = pattern, [model]

        for block in blocks:
            key = id(block)
            if key not in grouped:
                grouped[key] = (block, {})
                order.append(key)
            grouped[key][1][relative] = style

    return [grouped[key] for key in order]


def _styles(relative_plan: dict) -> dict:
    """Map a ``{relative_path: style}`` plan to ``parallelize_module`` style instances.

    Values may be plain strings (``"colwise"`` / ``"rowwise"``) or ``PackedColwiseParallel`` /
    ``PackedRowwiseParallel`` marker instances.
    """
    import torch.nn as nn
    from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor
    from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel

    def _make_packed_col(marker: PackedColwiseParallel) -> ColwiseParallel:
        _blocks = marker.blocks

        class _PackedColwiseImpl(ColwiseParallel):
            def _partition_linear_fn(self, name, module, device_mesh):
                blocks = _blocks if _blocks is not None else getattr(module, "_tp_packed_col_blocks")
                rank = device_mesh.get_local_rank()
                tp_size = device_mesh.size()
                for param_name, param in module.named_parameters():
                    if param_name == "weight":
                        full = distribute_tensor(
                            param, device_mesh, [Replicate()], src_data_rank=self.src_data_rank
                        ).to_local()
                        block_sizes = _blocks_to_block_sizes(full.shape[0], blocks)
                        parts, offset = [], 0
                        for bs in block_sizes:
                            chunk = bs // tp_size
                            parts.append(full[offset + rank * chunk : offset + (rank + 1) * chunk].contiguous())
                            offset += bs
                        local = torch.cat(parts, dim=0)
                        dist_param = nn.Parameter(
                            DTensor.from_local(local, device_mesh, [Shard(0)], run_check=False),
                            requires_grad=param.requires_grad,
                        )
                    else:
                        dist_param = nn.Parameter(
                            distribute_tensor(param, device_mesh, [Shard(0)], src_data_rank=self.src_data_rank),
                            requires_grad=param.requires_grad,
                        )
                    module.register_parameter(param_name, dist_param)

        return _PackedColwiseImpl()

    def _make_packed_row(marker: PackedRowwiseParallel) -> RowwiseParallel:
        _blocks = marker.blocks

        class _PackedRowwiseImpl(RowwiseParallel):
            def _partition_linear_fn(self, name, module, device_mesh):
                blocks = _blocks if _blocks is not None else getattr(module, "_tp_packed_row_blocks")
                rank = device_mesh.get_local_rank()
                tp_size = device_mesh.size()
                for param_name, param in module.named_parameters():
                    if param_name == "weight":
                        full = distribute_tensor(
                            param, device_mesh, [Replicate()], src_data_rank=self.src_data_rank
                        ).to_local()
                        block_sizes = _blocks_to_block_sizes(full.shape[1], blocks)
                        parts, offset = [], 0
                        for bs in block_sizes:
                            chunk = bs // tp_size
                            parts.append(full[:, offset + rank * chunk : offset + (rank + 1) * chunk].contiguous())
                            offset += bs
                        local = torch.cat(parts, dim=1)
                        dist_param = nn.Parameter(
                            DTensor.from_local(local, device_mesh, [Shard(1)], run_check=False),
                            requires_grad=param.requires_grad,
                        )
                    else:
                        dist_param = nn.Parameter(
                            distribute_tensor(param, device_mesh, [Replicate()], src_data_rank=self.src_data_rank),
                            requires_grad=param.requires_grad,
                        )
                    module.register_parameter(param_name, dist_param)

        return _PackedRowwiseImpl()

    resolved = {}
    for path, style in relative_plan.items():
        if style == "colwise":
            resolved[path] = ColwiseParallel()
        elif style == "rowwise":
            resolved[path] = RowwiseParallel()
        elif isinstance(style, PackedColwiseParallel):
            resolved[path] = _make_packed_col(style)
        elif isinstance(style, PackedRowwiseParallel):
            resolved[path] = _make_packed_row(style)
        else:
            raise ValueError(
                f"Unsupported tensor-parallel style '{style}' for '{path}'. "
                f"Expected 'colwise', 'rowwise', PackedColwiseParallel, or PackedRowwiseParallel."
            )
    return resolved


def apply_tensor_parallel(
    model: torch.nn.Module,
    config: TensorParallelConfig,
    tp_plan: dict,
    *,
    backend: str = "default",
) -> None:
    """Apply tensor parallel on a model from its flat ``_tp_plan``.

    ``backend="neuron"`` routes to the Neuron pre-shard path (works around the NRT consecutive-reduce-scatter bug and
    applies the Flux2 fused-weight permutations); ``"default"`` uses ``parallelize_module`` directly.
    """
    tp_mesh = config._mesh
    if tp_mesh is None:
        raise ValueError("`config._mesh` is None. Call `config.setup(rank, world_size, device)` before applying TP.")

    groups = _resolve_tp_plan(model, tp_plan)
    logger.debug(f"Applying tensor parallel (backend={backend}) over {len(groups)} module group(s) on mesh {tp_mesh}.")

    if backend == "neuron":
        from .tensor_parallel_neuron import _apply_tp_neuron

        _apply_tp_neuron(model, tp_mesh, groups)
        return

    from torch.distributed.tensor.parallel import parallelize_module

    for submodule, relative_plan in groups:
        parallelize_module(submodule, tp_mesh, _styles(relative_plan))
