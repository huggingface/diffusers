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

from typing import NamedTuple

import torch

from ..models._modeling_parallel import TensorParallelConfig
from ..utils import get_logger


logger = get_logger(__name__)  # pylint: disable=invalid-name

_SUPPORTED_TP_DEVICES = ("cuda", "neuron")


class PackedColwiseParallel:
    """Column-wise sharding for fused projections with heterogeneous block structure.

    `blocks` is a list of proportional integers whose sum divides the weight's row count. For example, `[1, 1]` for a
    SwiGLU gate+linear projection (two equal halves) or `[1, 1, 1, 3, 3]` for a Q+K+V+gate+linear projection with
    `mlp_ratio=3`. If `blocks` is `None`, the Linear module must carry a `_tp_packed_col_blocks` attribute set during
    model `__init__`.
    """

    def __init__(self, blocks: "list[int] | None" = None):
        self.blocks = blocks


class PackedRowwiseParallel:
    """Row-wise sharding for fused projections with heterogeneous block structure.

    `blocks` describes the input-column partition of the fused Linear (e.g. `[1, 3]` when the input concatenates an
    attention projection and an MLP projection with `mlp_ratio=3`). If `blocks` is `None`, the module must carry a
    `_tp_packed_row_blocks` attribute.
    """

    def __init__(self, blocks: "list[int] | None" = None):
        self.blocks = blocks


def _blocks_to_block_sizes(total_size: int, blocks: "list[int]") -> "list[int]":
    """Convert proportional block counts to absolute sizes.

    `blocks` is a list of positive integers interpreted as proportional weights. Their sum must divide `total_size`
    evenly. Returns a list of absolute sizes that sum to `total_size`. For example, `_blocks_to_block_sizes(1152, [1,
    1, 1, 3, 3])` returns `[128, 128, 128, 384, 384]`.
    """
    total = sum(blocks)
    if total_size % total != 0:
        raise ValueError(
            f"Cannot split {total_size} into proportional blocks {blocks}: "
            f"sum({blocks})={total} does not divide {total_size}."
        )
    unit = total_size // total
    return [b * unit for b in blocks]


class TPShardSpec(NamedTuple):
    """How one parameter is laid out across the tensor-parallel ranks.

    `dim` is the dimension sharded across ranks, or `None` when the parameter is replicated on every rank (a rowwise
    bias, which is added after the all-reduce). `block_sizes` partitions `dim` into independently sharded blocks; a
    plain `"colwise"` / `"rowwise"` style has a single block covering the whole dimension, and packed styles have one
    per fused projection.
    """

    dim: "int | None"
    block_sizes: "list[int] | None"


def _local_shard(tensor, dim: int, block_sizes: "list[int]", tp_mesh) -> torch.Tensor:
    """Extract this rank's slice of `tensor` along `dim`.

    `tensor` may be a `torch.Tensor` or a safetensors `PySafeSlice`, so the same arithmetic serves both resharding a
    weight already in memory and reading only one rank's slice off disk. Note that `PySafeSlice` exposes `get_shape()`
    rather than `.ndim`, and that a `dim`-1 slice comes back strided, hence the final `contiguous()` —
    `DTensor.from_local` needs a contiguous local tensor.
    """
    rank = tp_mesh.get_local_rank()
    tp_size = tp_mesh.size()
    ndim = tensor.dim() if isinstance(tensor, torch.Tensor) else len(tensor.get_shape())

    parts, offset = [], 0
    for block_size in block_sizes:
        # An uneven split is rejected rather than silently handed to `Shard`, which pads the tail
        # and would break both the paired colwise/rowwise matmul and `_unshard_gathered`.
        if block_size % tp_size != 0:
            raise ValueError(
                f"Cannot shard a block of size {block_size} across {tp_size} tensor-parallel ranks: "
                f"{block_size} is not divisible by {tp_size}."
            )
        chunk = block_size // tp_size
        index = [slice(None)] * ndim
        index[dim] = slice(offset + rank * chunk, offset + (rank + 1) * chunk)
        parts.append(tensor[tuple(index)])
        offset += block_size

    local = parts[0] if len(parts) == 1 else torch.cat(parts, dim=dim)
    return local.contiguous()


def _unshard_gathered(gathered: torch.Tensor, dim: int, block_sizes: "list[int]", tp_size: int) -> torch.Tensor:
    """Undo `_local_shard`'s block interleaving on an all-gathered tensor.

    `DTensor.full_tensor()` concatenates the local shards rank-major, so a packed weight comes back as `[block0_rank0,
    block1_rank0, block0_rank1, block1_rank1, ...]` and has to be regrouped by block. A single block is already in the
    original order and passes through unchanged.
    """
    if len(block_sizes) == 1:
        return gathered

    local_sizes = [block_size // tp_size for block_size in block_sizes]
    stride = sum(local_sizes)
    parts = []
    for i, local_size in enumerate(local_sizes):
        offset = sum(local_sizes[:i])
        parts.extend(gathered.narrow(dim, rank * stride + offset, local_size) for rank in range(tp_size))
    return torch.cat(parts, dim=dim)


def gather_tp_state_dict(state_dict: dict, specs: "dict[str, TPShardSpec]", config: TensorParallelConfig) -> dict:
    """Reassemble a tensor-parallel `state_dict` into ordinary full tensors.

    Every `DTensor` is all-gathered back to its full shape and, for the packed styles, reordered by `_unshard_gathered`
    — `full_tensor()` alone would leave the fused blocks interleaved by rank. Replicated and unplanned parameters pass
    through untouched.

    `full_tensor()` is a collective, so this must run on **every** rank even though usually only rank 0 goes on to
    write the result.
    """
    from torch.distributed.tensor import DTensor

    tp_size = config._tp_degree
    gathered = {}
    for key, value in state_dict.items():
        if not isinstance(value, DTensor):
            gathered[key] = value
            continue
        # `full_tensor()` is the collective; the reorder after it is plain tensor arithmetic, so keep it off
        # the accelerator — CPU is where this state dict is headed anyway, since it is about to be written.
        full = value.full_tensor().cpu()
        spec = specs[key]
        if spec.dim is not None:
            full = _unshard_gathered(full, spec.dim, spec.block_sizes, tp_size)
        gathered[key] = full.contiguous()
    return gathered


def resolve_tp_shard_specs(model: torch.nn.Module, tp_plan: dict) -> "dict[str, TPShardSpec]":
    """Map every `_tp_plan`-covered parameter name to its `TPShardSpec`.

    Parameters absent from the result are untouched by tensor parallelism. Both `weight` and `bias` of each planned
    module are covered.

    The plan is expanded by `_resolve_tp_plan` so there is a single implementation of the glob rules; the `id(module)
    -> name` map recovers qualified names from the submodules it returns. Going back through `_resolve_tp_plan` also
    handles a model that reuses one block instance in two places, which re-expanding the globs here would get wrong.

    Safe to call on a meta model: only shapes, `bias is not None`, and the packed-block attributes set in the module's
    `__init__` are read.
    """
    names = {id(module): name for name, module in model.named_modules()}
    specs: dict[str, TPShardSpec] = {}

    for block, relative_plan in _resolve_tp_plan(model, tp_plan):
        prefix = names[id(block)]
        for relative_path, style in relative_plan.items():
            submodule = block
            for atom in relative_path.split("."):
                submodule = getattr(submodule, atom)
            path = f"{prefix}.{relative_path}" if prefix else relative_path

            # `_tp_packed_*_blocks` hold absolute sizes rather than proportions; that works because
            # they sum to the full dimension, so `_blocks_to_block_sizes` computes `unit == 1`.
            if style == "colwise":
                weight_spec = TPShardSpec(0, [submodule.weight.shape[0]])
                bias_spec = weight_spec
            elif style == "rowwise":
                weight_spec = TPShardSpec(1, [submodule.weight.shape[1]])
                bias_spec = TPShardSpec(None, None)
            elif isinstance(style, PackedColwiseParallel):
                blocks = style.blocks if style.blocks is not None else submodule._tp_packed_col_blocks
                weight_spec = TPShardSpec(0, _blocks_to_block_sizes(submodule.weight.shape[0], blocks))
                bias_spec = weight_spec
            elif isinstance(style, PackedRowwiseParallel):
                blocks = style.blocks if style.blocks is not None else submodule._tp_packed_row_blocks
                weight_spec = TPShardSpec(1, _blocks_to_block_sizes(submodule.weight.shape[1], blocks))
                bias_spec = TPShardSpec(None, None)
            else:
                raise ValueError(
                    f"Unsupported tensor-parallel style '{style}' for '{path}'. "
                    f"Expected 'colwise', 'rowwise', PackedColwiseParallel, or PackedRowwiseParallel."
                )

            specs[f"{path}.weight"] = weight_spec
            if submodule.bias is not None:
                specs[f"{path}.bias"] = bias_spec

    return specs


def _resolve_tp_plan(model: torch.nn.Module, tp_plan: dict) -> list:
    """Group a flat `_tp_plan` into per-block `(submodule, {relative_path: style})` plans.

    Each glob is split at its single `*`; the prefix must resolve to a `ModuleList` and the suffix is the per-element
    key. Grouping by block lets the caller issue one `parallelize_module` call per block, which `RowwiseParallel` needs
    to attach its input redistribution at the block boundary.

    Example: when `transformer_blocks` is a `ModuleList` of length 2, the input `{"transformer_blocks.*.ff.linear_out":
    "rowwise"}` returns `[(transformer_blocks[0], {"ff.linear_out": "rowwise"}), (transformer_blocks[1],
    {"ff.linear_out": "rowwise"})]`.
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
    """Map a `{relative_path: style}` plan to `parallelize_module` style instances.

    Values may be plain strings (`"colwise"` / `"rowwise"`) or `PackedColwiseParallel` / `PackedRowwiseParallel` marker
    instances. Returns `{relative_path: ColwiseParallel() | RowwiseParallel() | <packed impl>}`, each subclassed to
    reject a sharded dim that is not divisible by the TP degree.
    """
    import torch.nn as nn
    from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor
    from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel

    def _make_packed_col(marker: PackedColwiseParallel) -> ColwiseParallel:
        _blocks = marker.blocks

        class _PackedColwiseImpl(ColwiseParallel):
            def _partition_linear_fn(self, name, module, device_mesh):
                blocks = _blocks if _blocks is not None else module._tp_packed_col_blocks
                # Both weight (`[out, in]`) and bias (`[out]`) are sharded row-wise (dim 0) with the same per-block
                # slicing so each rank's bias rows line up with its weight rows for the packed layout.
                for param_name, param in module.named_parameters():
                    # Replicate before slicing: the broadcast from `src_data_rank` is what makes one rank's
                    # weights authoritative when the model was randomly initialized rather than loaded from a
                    # checkpoint, in which case every rank starts with different values.
                    full = distribute_tensor(
                        param, device_mesh, [Replicate()], src_data_rank=self.src_data_rank
                    ).to_local()
                    local = _local_shard(full, 0, _blocks_to_block_sizes(full.shape[0], blocks), device_mesh)
                    module.register_parameter(
                        param_name,
                        nn.Parameter(
                            DTensor.from_local(local, device_mesh, [Shard(0)], run_check=False),
                            requires_grad=param.requires_grad,
                        ),
                    )

        return _PackedColwiseImpl()

    def _make_packed_row(marker: PackedRowwiseParallel) -> RowwiseParallel:
        _blocks = marker.blocks

        class _PackedRowwiseImpl(RowwiseParallel):
            def _partition_linear_fn(self, name, module, device_mesh):
                blocks = _blocks if _blocks is not None else module._tp_packed_row_blocks
                for param_name, param in module.named_parameters():
                    if param_name == "weight":
                        # See `_make_packed_col`: replicate first so one rank's weights win.
                        full = distribute_tensor(
                            param, device_mesh, [Replicate()], src_data_rank=self.src_data_rank
                        ).to_local()
                        local = _local_shard(full, 1, _blocks_to_block_sizes(full.shape[1], blocks), device_mesh)
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

    # `distribute_tensor` accepts an indivisible shard dim and just gives the trailing ranks a smaller (or empty)
    # slice, so an uneven split does not raise here — it surfaces much later as a shape or numerics error, because
    # the attention head split and the paired colwise/rowwise Linear both assume equal shards. Reject it up front,
    # matching what `_local_shard` already does for the packed styles.
    def _make_checked_col(path: str) -> ColwiseParallel:
        class _CheckedColwiseImpl(ColwiseParallel):
            def _partition_linear_fn(self, name, module, device_mesh):
                tp_size = device_mesh.size()
                out_features = module.weight.shape[0]
                if out_features % tp_size != 0:
                    raise ValueError(
                        f"Cannot colwise-shard '{path}' weight rows ({out_features}) across {tp_size} "
                        f"tensor-parallel ranks: not divisible by {tp_size}."
                    )
                super()._partition_linear_fn(name, module, device_mesh)

        return _CheckedColwiseImpl()

    def _make_checked_row(path: str) -> RowwiseParallel:
        class _CheckedRowwiseImpl(RowwiseParallel):
            def _partition_linear_fn(self, name, module, device_mesh):
                tp_size = device_mesh.size()
                in_features = module.weight.shape[1]
                if in_features % tp_size != 0:
                    raise ValueError(
                        f"Cannot rowwise-shard '{path}' weight columns ({in_features}) across {tp_size} "
                        f"tensor-parallel ranks: not divisible by {tp_size}."
                    )
                super()._partition_linear_fn(name, module, device_mesh)

        return _CheckedRowwiseImpl()

    resolved = {}
    for path, style in relative_plan.items():
        if style == "colwise":
            resolved[path] = _make_checked_col(path)
        elif style == "rowwise":
            resolved[path] = _make_checked_row(path)
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


def _hooks_only_styles(relative_plan: dict) -> dict:
    """Map a `{relative_path: style}` plan to styles that partition nothing.

    Used when the caller has already placed every planned parameter as a sharded `DTensor`. `parallelize_module` then
    runs only to register the forward input/output hooks; `_partition_linear_fn` must not re-partition. Packed and
    plain styles share hook behaviour, so both collapse onto the two styles here.

    Note this is not purely additive: `distribute_module` still replicates any *remaining* plain parameter of the
    targeted module into a `Replicate()` DTensor via a broadcast. Callers should therefore place every planned
    parameter themselves, and must ensure none is left on `meta` — the broadcast would be issued on a meta tensor.
    """
    from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel

    class _NoPartitionColwise(ColwiseParallel):
        def _partition_linear_fn(self, name, module, device_mesh):
            pass  # weight already Shard(0)

    class _NoPartitionRowwise(RowwiseParallel):
        def _partition_linear_fn(self, name, module, device_mesh):
            pass  # weight already Shard(1)

    resolved = {}
    for path, style in relative_plan.items():
        if style == "colwise" or isinstance(style, PackedColwiseParallel):
            resolved[path] = _NoPartitionColwise()
        elif style == "rowwise" or isinstance(style, PackedRowwiseParallel):
            resolved[path] = _NoPartitionRowwise()
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
    weights_already_sharded: bool = False,
) -> None:
    """Apply tensor parallel on a model from its flat `_tp_plan`.

    Set `weights_already_sharded` when the planned parameters are already `DTensor` shards, as they are after a
    streaming `from_pretrained` load; only the forward hooks are then registered. This is passed explicitly rather than
    detected, because a planned parameter missing from the checkpoint would still be a meta tensor and would make
    detection say "not sharded" for a model that is in fact half-sharded.
    """
    if tp_plan is None:
        raise ValueError(
            "`_tp_plan` must be set on the model class to use tensor parallelism. "
            f"'{model.__class__.__name__}' does not define one."
        )

    tp_mesh = config._mesh
    if tp_mesh is None:
        raise ValueError("`config._mesh` is None. Call `config.setup(rank, world_size, device)` before applying TP.")

    num_heads = getattr(model.config, "num_attention_heads", None)
    if num_heads is not None and num_heads % config._tp_degree != 0:
        raise ValueError(f"`tp_degree` ({config._tp_degree}) must divide the number of attention heads ({num_heads}).")

    if tp_mesh.device_type not in _SUPPORTED_TP_DEVICES:
        raise ValueError(
            f"Tensor parallelism is not supported on device type '{tp_mesh.device_type}'. Supported device types are "
            f"{list(_SUPPORTED_TP_DEVICES)}. The device type comes from the `mesh` passed to `TensorParallelConfig`, "
            f"or from the active accelerator when the mesh is built from `tp_degree`."
        )

    backend = "neuron" if tp_mesh.device_type == "neuron" else "default"
    groups = _resolve_tp_plan(model, tp_plan)
    logger.debug(f"Applying tensor parallel (backend={backend}) over {len(groups)} module group(s) on mesh {tp_mesh}.")

    from torch.distributed.tensor.parallel import parallelize_module

    if weights_already_sharded:
        for submodule, relative_plan in groups:
            parallelize_module(submodule, tp_mesh, _hooks_only_styles(relative_plan))
        return

    if backend == "neuron":
        from .tensor_parallel_neuron import _apply_tp_neuron

        _apply_tp_neuron(model, tp_mesh, groups, resolve_tp_shard_specs(model, tp_plan))
        return

    for submodule, relative_plan in groups:
        parallelize_module(submodule, tp_mesh, _styles(relative_plan))
