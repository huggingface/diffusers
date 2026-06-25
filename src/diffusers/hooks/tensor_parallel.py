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
    """Map a ``{relative_path: style_str}`` plan to ``parallelize_module`` style instances."""
    from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel

    mapping = {"colwise": ColwiseParallel, "rowwise": RowwiseParallel}
    resolved = {}
    for path, style in relative_plan.items():
        if style not in mapping:
            raise ValueError(
                f"Unsupported tensor-parallel style '{style}' for '{path}'. Expected one of {list(mapping)}."
            )
        resolved[path] = mapping[style]()
    return resolved


def apply_tensor_parallel(
    model: torch.nn.Module,
    config: TensorParallelConfig,
    tp_plan: dict,
    *,
    backend: str = "default",
) -> None:
    """Apply tensor parallel on a model from its flat ``_tp_plan``.

    ``backend="neuron"`` routes to the Neuron pre-shard path (works around the NRT consecutive-reduce-scatter bug
    and applies the Flux2 fused-weight permutations); ``"default"`` uses ``parallelize_module`` directly.
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

    # Some models fuse projections into single Linear layers (e.g. Flux2's SwiGLU FFN and fused QKV+MLP). Their
    # weights must be re-ordered before contiguous sharding so each rank gets a correct paired slice.
    permuters = getattr(model, "_tp_fused_block_permuters", None) or {}
    tp_size = tp_mesh.size()

    for submodule, relative_plan in groups:
        permuter = permuters.get(submodule.__class__.__name__)
        if permuter is not None:
            permuter(submodule, tp_size)
        parallelize_module(submodule, tp_mesh, _styles(relative_plan))
