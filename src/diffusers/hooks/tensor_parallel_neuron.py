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

"""Neuron backend for tensor parallelism, dispatched from ``apply_tensor_parallel`` when the TP mesh is on Neuron.

The difference from the generic path is a workaround for a Neuron NRT bug: consecutive ``reduce_scatter`` collectives
for large weight tensors (≥ 5120×5120) can fail when all layers are distributed in a single ``parallelize_module``
call. The fix is to pre-shard each weight locally on CPU via ``DTensor.from_local`` *before* calling
``parallelize_module``; the latter then sees already-placed DTensors, skips the collective for weights, but still
registers the required input/output hooks for the forward pass.
"""

import torch
import torch.distributed as dist
import torch.nn as nn


def _neuron_styles(relative_plan: dict) -> dict:
    """Map a ``{relative_path: style}`` plan to no-op-partition styles for Neuron.

    Weights (and biases) are pre-sharded in ``_pre_shard_and_tp``, so ``parallelize_module`` runs only to register the
    forward hooks; ``_partition_linear_fn`` must not re-partition. Packed and plain styles share hook behavior, so both
    collapse onto the two no-op styles.
    """
    from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel

    from .tensor_parallel import PackedColwiseParallel, PackedRowwiseParallel

    class _NeuronColwise(ColwiseParallel):
        def _partition_linear_fn(self, name, module, device_mesh):
            pass  # weight already Shard(0) via DTensor.from_local; parallelize_module runs only for the hooks

    class _NeuronRowwise(RowwiseParallel):
        def _partition_linear_fn(self, name, module, device_mesh):
            pass  # weight already Shard(1) via DTensor.from_local; parallelize_module runs only for the hooks

    resolved = {}
    for path, style in relative_plan.items():
        if style == "colwise" or isinstance(style, PackedColwiseParallel):
            resolved[path] = _NeuronColwise()
        elif style == "rowwise" or isinstance(style, PackedRowwiseParallel):
            resolved[path] = _NeuronRowwise()
        else:
            raise ValueError(
                f"Unsupported tensor-parallel style '{style}' for '{path}'. "
                f"Expected 'colwise', 'rowwise', PackedColwiseParallel, or PackedRowwiseParallel."
            )
    return resolved


def _pre_shard_and_tp(
    module: nn.Module,
    tp_mesh: "torch.distributed.device_mesh.DeviceMesh",
    original_plan: dict,
    rank: int,
    tp_size: int,
) -> None:
    """Pre-shard Linear weights via ``DTensor.from_local``, then call ``parallelize_module``.

    Workaround for a Neuron NRT bug where consecutive ``reduce_scatter`` calls for large weight tensors (≥ 5120×5120)
    fail when all layers are distributed in a single ``parallelize_module`` call. Pre-sharding each weight on CPU means
    it is already an on-device DTensor when ``parallelize_module`` runs (via ``_neuron_styles``), so the collective is
    skipped while the forward hooks are still registered.
    """
    from torch.distributed.tensor import DTensor, Replicate, Shard
    from torch.distributed.tensor.parallel import parallelize_module

    from .tensor_parallel import PackedColwiseParallel, PackedRowwiseParallel, _blocks_to_block_sizes

    device = torch.neuron.current_device()

    for path, orig_style in original_plan.items():
        # Resolve nested attribute path (e.g. "attn.to_q" or "attn.to_out.0")
        submod = module
        for part in path.split("."):
            submod = getattr(submod, part)

        if not hasattr(submod, "weight"):
            raise ValueError(f"`_tp_plan` entry '{path}' does not resolve to a module with a `weight` parameter.")

        w = submod.weight.data  # CPU at this point
        b = submod.bias.data if submod.bias is not None else None
        if isinstance(orig_style, PackedColwiseParallel):
            blocks = orig_style.blocks if orig_style.blocks is not None else getattr(submod, "_tp_packed_col_blocks")
            block_sizes = _blocks_to_block_sizes(w.shape[0], blocks)
            parts, bias_parts, offset = [], [], 0
            for bs in block_sizes:
                if bs % tp_size != 0:
                    raise ValueError(
                        f"Cannot shard packed block of size {bs} across {tp_size} tensor-parallel ranks: "
                        f"{bs} is not divisible by {tp_size}."
                    )
                chunk = bs // tp_size
                sl = slice(offset + rank * chunk, offset + (rank + 1) * chunk)
                parts.append(w[sl, :].contiguous())
                if b is not None:
                    bias_parts.append(b[sl].contiguous())
                offset += bs
            shard = torch.cat(parts, dim=0).to(device)
            submod.weight = nn.Parameter(DTensor.from_local(shard, tp_mesh, [Shard(0)]))
            if b is not None:
                bias_shard = torch.cat(bias_parts, dim=0).to(device)
                submod.bias = nn.Parameter(DTensor.from_local(bias_shard, tp_mesh, [Shard(0)]))
        elif isinstance(orig_style, PackedRowwiseParallel):
            blocks = orig_style.blocks if orig_style.blocks is not None else getattr(submod, "_tp_packed_row_blocks")
            block_sizes = _blocks_to_block_sizes(w.shape[1], blocks)
            parts, offset = [], 0
            for bs in block_sizes:
                if bs % tp_size != 0:
                    raise ValueError(
                        f"Cannot shard packed block of size {bs} across {tp_size} tensor-parallel ranks: "
                        f"{bs} is not divisible by {tp_size}."
                    )
                chunk = bs // tp_size
                parts.append(w[:, offset + rank * chunk : offset + (rank + 1) * chunk].contiguous())
                offset += bs
            shard = torch.cat(parts, dim=1).to(device)
            submod.weight = nn.Parameter(DTensor.from_local(shard, tp_mesh, [Shard(1)]))
            if b is not None:  # rowwise bias is added post-reduction → keep it replicated
                submod.bias = nn.Parameter(DTensor.from_local(b.to(device), tp_mesh, [Replicate()]))
        elif orig_style == "colwise":
            if w.shape[0] % tp_size != 0:
                raise ValueError(
                    f"Cannot colwise-shard '{path}' weight rows ({w.shape[0]}) across {tp_size} "
                    f"tensor-parallel ranks: not divisible by {tp_size}."
                )
            rows = w.shape[0] // tp_size
            sl = slice(rank * rows, (rank + 1) * rows)
            submod.weight = nn.Parameter(DTensor.from_local(w[sl, :].contiguous().to(device), tp_mesh, [Shard(0)]))
            if b is not None:
                submod.bias = nn.Parameter(DTensor.from_local(b[sl].contiguous().to(device), tp_mesh, [Shard(0)]))
        elif orig_style == "rowwise":
            if w.shape[1] % tp_size != 0:
                raise ValueError(
                    f"Cannot rowwise-shard '{path}' weight columns ({w.shape[1]}) across {tp_size} "
                    f"tensor-parallel ranks: not divisible by {tp_size}."
                )
            cols = w.shape[1] // tp_size
            shard = w[:, rank * cols : (rank + 1) * cols].contiguous().to(device)
            submod.weight = nn.Parameter(DTensor.from_local(shard, tp_mesh, [Shard(1)]))
            if b is not None:  # rowwise bias is added post-reduction → keep it replicated
                submod.bias = nn.Parameter(DTensor.from_local(b.to(device), tp_mesh, [Replicate()]))

    # parallelize_module is now a no-op for weight distribution (already DTensors)
    # but still registers the input/output hooks required for the forward pass.
    parallelize_module(module, tp_mesh, _neuron_styles(original_plan))


def _apply_tp_neuron(
    model: nn.Module,
    tp_mesh: "torch.distributed.device_mesh.DeviceMesh",
    groups: list,
) -> None:
    """Apply tensor parallelism on Neuron from resolved ``_tp_plan`` groups.

    ``groups`` is produced by ``diffusers.hooks.tensor_parallel._resolve_tp_plan`` — the same source of truth used by
    the generic path, so the two backends shard identical layers. For each ``(block, relative_plan)`` group this
    pre-shards the weights via ``DTensor.from_local`` (Neuron NRT consecutive-reduce-scatter workaround), then calls
    ``parallelize_module`` to register the forward hooks.

    Model weights must be on CPU when this is called.
    """
    rank = dist.get_rank()
    tp_size = tp_mesh.size()

    for block, relative_plan in groups:
        _pre_shard_and_tp(block, tp_mesh, relative_plan, rank, tp_size)
