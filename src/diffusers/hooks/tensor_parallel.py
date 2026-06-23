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

import torch

from ..models._modeling_parallel import TensorParallelConfig
from ..utils import get_logger


logger = get_logger(__name__)  # pylint: disable=invalid-name


def _get_module(model: torch.nn.Module, path: str) -> torch.nn.Module:
    """Resolve a dotted (wildcard-free) module path relative to ``model``."""
    submodule = model
    if path:
        for atom in path.split("."):
            if not hasattr(submodule, atom):
                raise ValueError(f"'{atom}' is not a submodule of '{submodule.__class__.__name__}'")
            submodule = getattr(submodule, atom)
    return submodule


def _resolve_tp_plan(model: torch.nn.Module, tp_plan: dict) -> list:
    """Group a flat ``_tp_plan`` into per-module ``parallelize_module`` plans.

    ``tp_plan`` maps module-name globs (relative to ``model``) to a style string, e.g.
    ``{"transformer_blocks.*.attn.to_q": "colwise"}``. Each glob is split at its single ``*``:
    the prefix must resolve to a ``ModuleList`` and the suffix becomes the per-element relative
    key. Entries are grouped by the repeated block instance so the caller issues one
    ``parallelize_module`` call per block (required so ``RowwiseParallel`` attaches its input
    redistribution at the block boundary). Keys without a ``*`` are grouped under the model itself.

    Returns:
        A list of ``(submodule, {relative_path: style_str})`` tuples, in plan order.
    """
    grouped: dict[int, tuple] = {}
    order: list[int] = []

    for pattern, style in tp_plan.items():
        if pattern.count("*") > 1:
            raise ValueError(f"Wildcard '*' can only be used once in a `_tp_plan` key, got '{pattern}'.")

        if "*" in pattern:
            prefix, _, suffix = pattern.partition("*")
            container = _get_module(model, prefix.strip("."))
            if not isinstance(container, torch.nn.ModuleList):
                raise ValueError(
                    f"`_tp_plan` wildcard '{pattern}' must expand over a `ModuleList`, but "
                    f"'{prefix.strip('.')}' resolved to '{container.__class__.__name__}'."
                )
            relative = suffix.strip(".")
            blocks = list(container)
        else:
            relative = pattern
            blocks = [model]

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
    """Apply tensor parallelism to a model from its flat ``_tp_plan``.

    This is model-agnostic: it only relies on ``tp_plan`` (a flat mapping of module-name globs to
    ``"colwise"``/``"rowwise"`` styles) and on the device mesh stored on ``config``. The attention
    processors derive their per-rank head/inner sizes from ``config`` at runtime, so no processor
    swap is needed.

    Args:
        model (`torch.nn.Module`):
            The model to shard (e.g. a ``Flux2Transformer2DModel``).
        config (`TensorParallelConfig`):
            TP configuration. ``config.setup()`` must have been called so that ``config._mesh`` is
            populated.
        tp_plan (`dict`):
            The model's ``_tp_plan`` (see :class:`~diffusers.models.transformers.Flux2Transformer2DModel`).
        backend (`str`, *optional*, defaults to `"default"`):
            ``"default"`` uses ``torch.distributed.tensor.parallel.parallelize_module`` directly.
            ``"neuron"`` routes to the Neuron pre-shard path, which works around the Neuron NRT
            consecutive-reduce-scatter bug and applies the Flux2 fused-weight permutations.
    """
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        raise RuntimeError("apply_tensor_parallel requires an initialised torch.distributed process group.")

    tp_mesh = config._mesh
    if tp_mesh is None:
        raise ValueError("`config._mesh` is None. Call `config.setup(rank, world_size, device)` before applying TP.")

    groups = _resolve_tp_plan(model, tp_plan)
    logger.debug(f"Applying tensor parallel (backend={backend}) over {len(groups)} module group(s) on mesh {tp_mesh}.")

    if backend == "neuron":
        from ..models.transformers.transformer_flux2_neuron_tp import _apply_tp_neuron

        _apply_tp_neuron(model, tp_mesh, groups)
        return

    try:
        from torch.distributed.tensor.parallel import parallelize_module
    except ImportError as e:
        raise ImportError(
            "apply_tensor_parallel requires PyTorch >= 2.3 with distributed tensor parallel support."
        ) from e

    # Some models fuse projections into single Linear layers (e.g. Flux2's SwiGLU FFN and fused
    # QKV+MLP). Their weights must be re-ordered before contiguous sharding so each rank gets a
    # correct paired slice.
    permuters = getattr(model, "_tp_fused_block_permuters", None) or {}
    tp_size = tp_mesh.size()

    for submodule, relative_plan in groups:
        permuter = permuters.get(submodule.__class__.__name__)
        if permuter is not None:
            permuter(submodule, tp_size)
        parallelize_module(submodule, tp_mesh, _styles(relative_plan))
