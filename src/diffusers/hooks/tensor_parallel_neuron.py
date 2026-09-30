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

"""Neuron backend for tensor parallelism, dispatched from `apply_tensor_parallel` when the TP mesh is on Neuron.

The difference from the generic path is a workaround for a Neuron NRT bug: consecutive `reduce_scatter` collectives for
large weight tensors (≥ 5120×5120) can fail when all layers are distributed in a single `parallelize_module` call. The
fix is to pre-shard each weight locally on CPU via `DTensor.from_local` *before* calling `parallelize_module`; the
latter then sees already-placed DTensors and skips the collective for weights, while still registering the required
input/output hooks for the forward pass.

Only needed for a model that is already in memory. `from_pretrained` with a tensor-parallel `parallel_config` streams
each rank's slice straight off disk into its DTensor, which issues no weight collectives at all and so cannot hit the
bug in the first place.
"""

import torch
import torch.nn as nn

from .tensor_parallel import TPShardSpec, _pre_shard_and_parallelize


def _apply_tp_neuron(
    model: nn.Module,
    tp_mesh: "torch.distributed.device_mesh.DeviceMesh",
    groups: list,
    specs: "dict[str, TPShardSpec]",
) -> None:
    """Pre-shard the planned parameters via `DTensor.from_local`, then register the forward hooks.

    `groups` and `specs` both come from the model's `_tp_plan` via `diffusers.hooks.tensor_parallel._resolve_tp_plan` /
    `resolve_tp_shard_specs`, the same source of truth the generic path uses, so the two backends shard identical
    layers.

    Model weights must be on CPU when this is called.
    """
    _pre_shard_and_parallelize(model, tp_mesh, groups, specs, torch.neuron.current_device())
