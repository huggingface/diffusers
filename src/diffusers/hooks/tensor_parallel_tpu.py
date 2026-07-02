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

"""TPU backend for tensor parallelism, dispatched from ``apply_tensor_parallel(backend="tpu")``.

The structure mirrors the Neuron backend (``tensor_parallel_neuron.py``). The motivation differs: on Neuron the
pre-shard path works around an NRT consecutive-reduce-scatter bug; here it prevents OOM. Without pre-sharding,
``parallelize_module`` calls ``distribute_tensor`` internally, which loads the full weight matrix on every TPU chip
before scattering. For large diffusion models this exhausts HBM. Pre-sharding each weight on CPU via
``DTensor.from_local`` first means each chip only receives its local shard, then ``parallelize_module`` is called
as a no-op for weights (they are already DTensors) but still registers the input/output hooks for the forward pass.
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

    # Each torchrun worker owns one TPU chip (its local device). Use "tpu" without an explicit
    # index — specifying tpu:rank would address chip `rank` from the current process's view,
    # which fails because each worker only has access to its own assigned chip.
    device = torch.device("tpu")

    for path, style in plan.items():
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


def _apply_tp_tpu(
    model: nn.Module,
    tp_mesh: "torch.distributed.device_mesh.DeviceMesh",
    groups: list,
) -> None:
    """Apply tensor parallelism on TPU from resolved ``_tp_plan`` groups.

    ``groups`` is produced by ``diffusers.hooks.tensor_parallel._resolve_tp_plan`` — the same source of truth used by
    the generic path, so the two backends shard identical layers. For each ``(block, relative_plan)`` group this:
    1. permutes the model's fused weights (via ``model._tp_fused_block_permuters``, the same backend-agnostic permuters
       the generic path uses) so column/row slicing gives each rank a correct chunk,
    2. pre-shards the weights via ``DTensor.from_local`` (avoids full-weight materialisation on TPU HBM), then calls
       ``parallelize_module`` to register the forward hooks.

    Model weights must be on CPU when this is called.
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
