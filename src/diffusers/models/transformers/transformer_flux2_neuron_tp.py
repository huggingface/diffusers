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

"""Neuron-specific Tensor Parallelism utilities for Flux2 and Qwen3.

This module provides the functions needed to apply tensor parallelism on AWS
Neuron hardware.  The key difference from the generic ``apply_tensor_parallel``
path is a workaround for a Neuron NRT bug: consecutive ``reduce_scatter``
collectives for large weight tensors (≥ 5120×5120) can fail when all layers
are distributed in a single ``parallelize_module`` call.  The fix is to
pre-shard each weight locally on CPU via ``DTensor.from_local`` *before*
calling ``parallelize_module``; the latter then sees already-placed DTensors,
skips the collective for weights, but still registers the required
input/output hooks for the forward pass.

Entry points:
    ``apply_tp_flux2_transformer_neuron(model, tp_mesh)``
        Apply TP to a ``Flux2Transformer2DModel``.  Includes the weight
        permutations required by Flux2's SwiGLU FFN and fused QKV+MLP
        projections.

    ``apply_tp_qwen3_neuron(model, tp_mesh)``
        Apply TP to a ``Qwen3ForCausalLM`` text encoder.  The sharding plan is
        derived from ``model.config.base_model_tp_plan`` — the same plan used
        by ``from_pretrained(tp_plan="auto")`` in transformers — so it stays in
        sync automatically if the plan changes upstream.
"""

import torch
import torch.distributed as dist
import torch.nn as nn


def _permute_swiglu_for_tp(weight: torch.Tensor, tp_size: int) -> torch.Tensor:
    """Interleave gate/linear chunks of a SwiGLU FFN weight for column-wise TP.

    ``ff.linear_in`` in Flux2 double-stream blocks stores
    ``[gate_0…gate_N, linear_0…linear_N]`` (two halves concatenated).
    After ``ColwiseParallel``, rank *r* takes rows
    ``[r*chunk : (r+1)*chunk]`` from the full weight — but that would give
    rank *r* only gate rows, not the paired gate+linear rows it needs.
    This function re-orders to ``[gate_0, linear_0, gate_1, linear_1, …]``
    so that slicing is consistent.
    """
    with torch.no_grad():
        total = weight.shape[0]
        inner = total // 2
        chunk = inner // tp_size
        gate = weight[:inner]
        linear = weight[inner:]
        parts = []
        for i in range(tp_size):
            parts.append(gate[i * chunk : (i + 1) * chunk])
            parts.append(linear[i * chunk : (i + 1) * chunk])
        return torch.cat(parts, dim=0)


def _permute_qkv_mlp_for_tp(
    weight: torch.Tensor,
    tp_size: int,
    inner_dim: int,
    mlp_hidden_dim: int,
) -> torch.Tensor:
    """Interleave Q/K/V/gate/linear chunks of the fused ``to_qkv_mlp_proj`` weight.

    ``to_qkv_mlp_proj`` in single-stream blocks concatenates
    ``[Q, K, V, mlp_gate, mlp_linear]`` along the output dimension.
    Re-order so that rank *r* receives a contiguous slice containing its
    proportional share of each component.
    """
    with torch.no_grad():
        q = weight[:inner_dim]
        k = weight[inner_dim : 2 * inner_dim]
        v = weight[2 * inner_dim : 3 * inner_dim]
        mlp_gate = weight[3 * inner_dim : 3 * inner_dim + mlp_hidden_dim]
        mlp_lin = weight[3 * inner_dim + mlp_hidden_dim :]

        qkv_chunk = inner_dim // tp_size
        mlp_chunk = mlp_hidden_dim // tp_size

        parts = []
        for i in range(tp_size):
            parts += [
                q[i * qkv_chunk : (i + 1) * qkv_chunk],
                k[i * qkv_chunk : (i + 1) * qkv_chunk],
                v[i * qkv_chunk : (i + 1) * qkv_chunk],
                mlp_gate[i * mlp_chunk : (i + 1) * mlp_chunk],
                mlp_lin[i * mlp_chunk : (i + 1) * mlp_chunk],
            ]
        return torch.cat(parts, dim=0)


def _permute_out_for_tp(
    weight: torch.Tensor,
    tp_size: int,
    attn_dim: int,
    mlp_dim: int,
) -> torch.Tensor:
    """Interleave attn/mlp output columns of the fused ``to_out`` weight.

    ``to_out`` in single-stream blocks accepts ``[attn_out, mlp_out]``
    concatenated along the input (column) dimension.  Re-order columns so
    that rank *r* receives a contiguous slice of paired attn+mlp columns.
    """
    with torch.no_grad():
        attn_part = weight[:, :attn_dim]
        mlp_part = weight[:, attn_dim:]

        attn_chunk = attn_dim // tp_size
        mlp_chunk = mlp_dim // tp_size

        parts = []
        for i in range(tp_size):
            parts.append(attn_part[:, i * attn_chunk : (i + 1) * attn_chunk])
            parts.append(mlp_part[:, i * mlp_chunk : (i + 1) * mlp_chunk])
        return torch.cat(parts, dim=1)


def _pre_shard_and_tp(
    module: nn.Module,
    tp_mesh: "torch.distributed.device_mesh.DeviceMesh",
    plan: dict,
    rank: int,
    tp_size: int,
) -> None:
    """Pre-shard Linear weights via ``DTensor.from_local``, then call ``parallelize_module``.

    Workaround for a Neuron NRT bug where consecutive ``reduce_scatter`` calls
    for large weight tensors (≥ 5120×5120) fail when all layers are distributed
    in a single ``parallelize_module`` call.  By pre-sharding each weight on CPU
    before the call, ``distribute_tensor`` inside ``parallelize_module`` sees an
    already-placed DTensor and skips the collective, while the module hooks
    (input/output specs) are still registered correctly.

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


def apply_tp_flux2_transformer_neuron(
    model: "Flux2Transformer2DModel",
    tp_mesh: "torch.distributed.device_mesh.DeviceMesh",
) -> "Flux2Transformer2DModel":
    """Apply tensor parallelism to a ``Flux2Transformer2DModel`` on Neuron.

    Steps for each block type:
    1. Permute fused weights so that column-wise slicing gives each rank a
       correct paired chunk (SwiGLU gate+linear, or Q/K/V/MLP).
    2. Pre-shard weights via ``DTensor.from_local`` (Neuron NRT workaround).
    3. Call ``parallelize_module`` to register input/output hooks.
    4. Replace the attention processor with the TP-aware variant.

    The model weights must still be on CPU when this function is called.
    Move the model to the Neuron device *after* this call::

        apply_tp_flux2_transformer_neuron(pipe.transformer, tp_mesh)
        pipe.transformer = pipe.transformer.to(device)

    Args:
        model: ``Flux2Transformer2DModel`` with weights on CPU.
        tp_mesh: 1-D Neuron device mesh of size ``tp_size``.

    Returns:
        The same ``model`` instance, modified in-place.
    """
    from .transformer_flux2 import Flux2AttnProcessorTP, Flux2ParallelSelfAttnProcessorTP

    rank = dist.get_rank()
    tp_size = tp_mesh.size()

    double_plan = model._get_tp_double_block_plan()
    single_plan = model._get_tp_single_block_plan()

    # ── Double-stream blocks (cross-attn + FFN) ────────────────────────────
    for block in model.transformer_blocks:
        # Permute SwiGLU weights before sharding
        block.ff.linear_in.weight.data = _permute_swiglu_for_tp(
            block.ff.linear_in.weight.data, tp_size
        )
        block.ff_context.linear_in.weight.data = _permute_swiglu_for_tp(
            block.ff_context.linear_in.weight.data, tp_size
        )
        _pre_shard_and_tp(block, tp_mesh, double_plan, rank, tp_size)
        block.attn.set_processor(Flux2AttnProcessorTP(tp_size))

    # ── Single-stream blocks (parallel self-attn + fused MLP) ──────────────
    for block in model.single_transformer_blocks:
        attn = block.attn
        inner_dim = attn.inner_dim
        mlp_hidden = attn.mlp_hidden_dim

        attn.to_qkv_mlp_proj.weight.data = _permute_qkv_mlp_for_tp(
            attn.to_qkv_mlp_proj.weight.data, tp_size, inner_dim, mlp_hidden
        )
        attn.to_out.weight.data = _permute_out_for_tp(
            attn.to_out.weight.data, tp_size, inner_dim, mlp_hidden
        )
        _pre_shard_and_tp(block, tp_mesh, single_plan, rank, tp_size)
        block.attn.set_processor(Flux2ParallelSelfAttnProcessorTP(tp_size))

    return model


def apply_tp_qwen3_neuron(
    model: "Qwen3ForCausalLM",
    tp_mesh: "torch.distributed.device_mesh.DeviceMesh",
) -> "Qwen3ForCausalLM":
    """Apply tensor parallelism to a ``Qwen3ForCausalLM`` text encoder on Neuron.

    The sharding plan is derived from ``model.config.base_model_tp_plan`` —
    the same plan used by ``from_pretrained(tp_plan="auto")`` in transformers —
    so it stays in sync automatically if the plan changes upstream.

    ``"replicated_with_grad_allreduce"`` entries (Q/K norm layers) are skipped:
    those layers require gradient all-reduce in training but need no weight
    sharding for inference.

    Qwen3's separate ``gate_proj`` / ``up_proj`` projections require no weight
    permutations (unlike Flux2's fused SwiGLU).

    The model weights must still be on CPU when this function is called::

        apply_tp_qwen3_neuron(pipe.text_encoder, tp_mesh)
        pipe.text_encoder = pipe.text_encoder.to(device)

    **Primary path**: try ``Qwen3ForCausalLM.from_pretrained(model_id, tp_plan="auto")``
    first — transformers' native TP may work on Neuron directly since its hook
    mechanism does not use DTensor reduce_scatter.  Fall back to this function if
    the NRT bug is triggered.

    Args:
        model: ``Qwen3ForCausalLM`` with weights on CPU.
        tp_mesh: 1-D Neuron device mesh of size ``tp_size``.

    Returns:
        The same ``model`` instance, modified in-place.
    """
    from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel

    rank = dist.get_rank()
    tp_size = tp_mesh.size()

    style_map = {
        "colwise": ColwiseParallel(),
        "colwise_gather_output": ColwiseParallel(),  # lm_head — same for inference
        "rowwise": RowwiseParallel(),
        # "replicated_with_grad_allreduce" → skipped (q_norm/k_norm, inference only)
    }

    # config.base_model_tp_plan example:
    # {"layers.*.self_attn.q_proj": "colwise", "layers.*.self_attn.o_proj": "rowwise", ...}
    per_layer_plan = {
        path.split("*.")[1]: style_map[style]
        for path, style in model.config.base_model_tp_plan.items()
        if "*." in path and style in style_map
    }

    if not per_layer_plan:
        raise ValueError(
            "Could not extract a per-layer TP plan from `model.config.base_model_tp_plan`. "
            f"Got: {model.config.base_model_tp_plan}"
        )

    for layer in model.model.layers:
        _pre_shard_and_tp(layer, tp_mesh, per_layer_plan, rank, tp_size)

    return model
