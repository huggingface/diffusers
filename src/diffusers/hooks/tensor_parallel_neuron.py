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

"""Neuron backend for tensor parallelism, dispatched from ``apply_tensor_parallel(backend="neuron")``.

The difference from the generic path is a workaround for a Neuron NRT bug: consecutive ``reduce_scatter`` collectives
for large weight tensors (≥ 5120×5120) can fail when all layers are distributed in a single ``parallelize_module``
call. The fix is to pre-shard each weight locally on CPU via ``DTensor.from_local`` *before* calling
``parallelize_module``; the latter then sees already-placed DTensors, skips the collective for weights, but still
registers the required input/output hooks for the forward pass.
"""

import torch
import torch.distributed as dist
import torch.nn as nn


def _pre_shard_and_tp(
    module: nn.Module,
    tp_mesh: "torch.distributed.device_mesh.DeviceMesh",
    plan: dict,
    rank: int,
    tp_size: int,
) -> None:
    """Pre-shard Linear weights via ``DTensor.from_local``, then call ``parallelize_module``.

    Workaround for a Neuron NRT bug where consecutive ``reduce_scatter`` calls for large weight tensors (≥ 5120×5120)
    fail when all layers are distributed in a single ``parallelize_module`` call. By pre-sharding each weight on CPU
    before the call, ``distribute_tensor`` inside ``parallelize_module`` sees an already-placed DTensor and skips the
    collective, while the module hooks (input/output specs) are still registered correctly.

    Args:
        module: The block whose Linear sub-modules are being sharded.
        tp_mesh: Device mesh for TP (1-D, size == tp_size).
        plan: ``{relative_path: ColwiseParallel() | RowwiseParallel()}`` dict,
            as expected by ``parallelize_module``.
        rank: Current rank (``dist.get_rank()``).
        tp_size: Total TP degree (``tp_mesh.size()``).
    """
    from torch.distributed.tensor import DTensor, Shard
    from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, parallelize_module

    device = torch.neuron.current_device()

    for path, style in plan.items():
        # Resolve nested attribute path (e.g. "attn.to_q" or "attn.to_out.0")
        submod = module
        for part in path.split("."):
            submod = getattr(submod, part)

        if not hasattr(submod, "weight"):
            continue

        w = submod.weight.data  # CPU at this point
        if isinstance(style, ColwiseParallel):
            rows = w.shape[0] // tp_size
            shard = w[rank * rows : (rank + 1) * rows, :].contiguous().to(device)
            submod.weight = nn.Parameter(DTensor.from_local(shard, tp_mesh, [Shard(0)]))
        elif isinstance(style, RowwiseParallel):
            cols = w.shape[1] // tp_size
            shard = w[:, rank * cols : (rank + 1) * cols].contiguous().to(device)
            submod.weight = nn.Parameter(DTensor.from_local(shard, tp_mesh, [Shard(1)]))

    # parallelize_module is now a no-op for weight distribution (already DTensors)
    # but still registers the input/output hooks required for the forward pass.
    parallelize_module(module, tp_mesh, plan)


def _apply_tp_neuron(
    model: nn.Module,
    tp_mesh: "torch.distributed.device_mesh.DeviceMesh",
    groups: list,
) -> None:
    """Apply tensor parallelism on Neuron from resolved ``_tp_plan`` groups.

    ``groups`` is produced by ``diffusers.hooks.tensor_parallel._resolve_tp_plan`` — the same source of truth used by
    the generic path, so the two backends shard identical layers. For each ``(block, relative_plan)`` group this:
    1. permutes the model's fused weights (via ``model._tp_fused_block_permuters``, the same backend-agnostic permuters
       the generic path uses) so column/row slicing gives each rank a correct chunk,
    2. pre-shards the weights via ``DTensor.from_local`` (Neuron NRT consecutive-reduce-scatter workaround), then calls
       ``parallelize_module`` to register the forward hooks.

    The attention processors derive their per-rank sizes from ``_parallel_config`` at runtime, so no processor swap is
    performed here. Model weights must be on CPU when this is called.
    """
    from .tensor_parallel import _styles

    rank = dist.get_rank()
    tp_size = tp_mesh.size()
    permuters = getattr(model, "_tp_fused_block_permuters", None) or {}

    for block, relative_plan in groups:
        permuter = permuters.get(block.__class__.__name__)
        if permuter is not None:
            permuter(block, tp_size)
        _pre_shard_and_tp(block, tp_mesh, _styles(relative_plan), rank, tp_size)
