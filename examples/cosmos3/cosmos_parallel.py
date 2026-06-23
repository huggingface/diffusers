# Copyright 2025 The NVIDIA Team and The HuggingFace Team. All rights reserved.
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
"""Multi-GPU helpers for Cosmos 3, implemented entirely outside the model.

This module holds the implementation; the runnable entry point is
``inference_cosmos3.py`` (pass ``--tp-degree`` / ``--cp-degree`` and launch with
``torchrun`` to use any of these across all the pipeline's modalities). Two orthogonal
sharding axes are provided, composable on a 2-D ``(tp, cp)`` device mesh:

  * Context parallelism (CP / Ulysses) — ``enable_cosmos3_context_parallel``. Shards the
    *sequence* across GPUs; attention runs with two all-to-all collectives per layer
    (gather seq / scatter heads -> local attention -> gather heads / scatter seq).
    Replicates the weights, so it cuts latency but not weight memory.
  * Tensor parallelism (TP) — ``enable_cosmos3_tensor_parallel``. Shards the attention
    and MLP *weight* matrices (Megatron-style), so a checkpoint that doesn't fit one GPU
    (Cosmos3-Super, ~120 GB) loads across several. Attention stays dense; pair it with
    ``enable_cosmos3_flash_attention`` (or with CP) so GQA uses the flash kernel.

The model carries no parallelism logic — it exposes small no-op seams:
``transformer._cp_shard_fn`` / ``_cp_gather_fn`` (sequence shard/gather around the
decoder stack) and ``Cosmos3AttnProcessor._run_attention`` (an override seam for the
attention core). The helpers below wire those up.

Why Cosmos 3 needs a custom CP path (not diffusers' declarative ``_cp_plan``):
  1. grouped-query attention — K/V heads must be repeated to match the query heads;
  2. separate understanding (causal) / generation (full) streams, with the generation
     stream attending to ``cat(und, gen)``;
  3. ragged per-stream lengths — each pathway is padded independently and the padded
     generation keys are masked.

GQA + the flash kernel: SDPA's flash/cuDNN kernels reject ``enable_gqa`` and the native
kernel falls back to math (which materializes the full ``[S, S]`` scores and OOMs on long
sequences). Both attention paths here instead expand the KV heads up to the query-head
count and call SDPA with ``enable_gqa=False``, so it dispatches to flash (O(S) memory).
"""

import torch

from diffusers.models.attention_dispatch import AttentionBackendName, dispatch_attention_fn
from diffusers.models.transformers.transformer_cosmos3 import Cosmos3AttnProcessor


try:  # torch >= 2.4
    from torch.distributed.tensor import DTensor, Replicate, Shard
except ImportError:  # pragma: no cover - older torch
    from torch.distributed._tensor import DTensor, Replicate, Shard


def _repeat_kv_heads(x, repeats):
    """Repeat KV heads (for GQA): ``[seq, num_kv_heads, d] -> [seq, num_kv_heads * repeats, d]``.

    Each KV head is repeated ``repeats`` times contiguously, matching GQA grouping
    (query head i pairs with KV group i // repeats).
    """
    if repeats == 1:
        return x
    seq_len, num_kv_heads, head_dim = x.shape
    x = x[:, :, None, :].expand(seq_len, num_kv_heads, repeats, head_dim)
    return x.reshape(seq_len, num_kv_heads * repeats, head_dim)


# =============================================================================
# Context parallelism (Ulysses)
# =============================================================================
# --- Collective primitives (all-to-all / all-gather via DTensor redistribute) ---
def _cp_all_to_all(local_input, scatter_dim, gather_dim, cp_mesh):
    """All-to-all via DTensor redistribute: gather ``gather_dim``, scatter ``scatter_dim``."""
    dt = DTensor.from_local(local_input, cp_mesh, [Shard(gather_dim)], run_check=False)
    return dt.redistribute(cp_mesh, [Shard(scatter_dim)]).to_local()


def _cp_all_gather(local_input, gather_dim, cp_mesh):
    """All-gather via DTensor redistribute: ``Shard(gather_dim) -> Replicate()``."""
    dt = DTensor.from_local(local_input, cp_mesh, [Shard(gather_dim)], run_check=False)
    return dt.redistribute(cp_mesh, [Replicate()]).to_local()


def _cp_gather_seq_scatter_heads(x, cp_mesh):
    """``[seq/cp, h, d] -> [seq, h/cp, d]``."""
    return _cp_all_to_all(x, scatter_dim=1, gather_dim=0, cp_mesh=cp_mesh)


def _cp_gather_heads_scatter_seq(x, cp_mesh):
    """``[seq, h/cp, d] -> [seq/cp, h, d]``."""
    return _cp_all_to_all(x, scatter_dim=0, gather_dim=1, cp_mesh=cp_mesh)


def _cp_pad_dim0(x, target_len):
    """Zero-pad ``x`` along dim 0 up to ``target_len`` (no-op if already there)."""
    pad = target_len - x.shape[0]
    if pad <= 0:
        return x
    return torch.cat([x, x.new_zeros((pad, *x.shape[1:]))], dim=0)


def _cp_shard_dim0(x, cp_mesh):
    """Keep this rank's contiguous shard along dim 0 (dim 0 must be divisible)."""
    world = cp_mesh.size()
    if world == 1:
        return x
    rank = cp_mesh.get_local_rank()
    shard = x.shape[0] // world
    return x.narrow(0, rank * shard, shard).contiguous()


def _cp_gather_dim0(x, cp_mesh):
    """Reassemble the full sequence (dim 0) from per-rank shards, in rank order."""
    if cp_mesh.size() == 1:
        return x
    return _cp_all_gather(x, gather_dim=0, cp_mesh=cp_mesh)


# --- Sharding / gathering of the dual-pathway packed sequence ---
def shard_cosmos3_sequence(und_seq, gen_seq, rotary_emb, cp_mesh):
    """Pad each pathway to a multiple of the CP world size, then shard the und/gen
    hidden states and their rotary embeddings along the sequence dim, independently
    per pathway. ``rotary_emb`` is ``(cos_und, sin_und, cos_gen, sin_gen)``.

    Returns ``(und_seq, gen_seq, rotary_emb, meta)`` where ``meta`` records the real
    and padded per-pathway lengths so attention can mask padded keys and the caller
    can slice the padding off after gathering.
    """
    world = cp_mesh.size()
    cos_und, sin_und, cos_gen, sin_gen = rotary_emb
    und_real, gen_real = und_seq.shape[0], gen_seq.shape[0]
    und_padded = ((und_real + world - 1) // world) * world
    gen_padded = ((gen_real + world - 1) // world) * world
    meta = {"und_real": und_real, "gen_real": gen_real, "und_padded": und_padded, "gen_padded": gen_padded}

    und_seq = _cp_shard_dim0(_cp_pad_dim0(und_seq, und_padded), cp_mesh)
    gen_seq = _cp_shard_dim0(_cp_pad_dim0(gen_seq, gen_padded), cp_mesh)
    rotary_emb = (
        _cp_shard_dim0(_cp_pad_dim0(cos_und, und_padded), cp_mesh),
        _cp_shard_dim0(_cp_pad_dim0(sin_und, und_padded), cp_mesh),
        _cp_shard_dim0(_cp_pad_dim0(cos_gen, gen_padded), cp_mesh),
        _cp_shard_dim0(_cp_pad_dim0(sin_gen, gen_padded), cp_mesh),
    )
    return und_seq, gen_seq, rotary_emb, meta


def build_gen_key_mask(meta, dtype, device):
    """Additive key mask for the generation pathway's full attention. Padded und/gen
    key positions are set to ``-inf`` so real generation queries ignore them. Returns
    ``None`` when no padding was added. Shape ``[1, 1, 1, S_k]``.
    """
    if meta["und_real"] == meta["und_padded"] and meta["gen_real"] == meta["gen_padded"]:
        return None
    s_k = meta["und_padded"] + meta["gen_padded"]
    mask = torch.zeros(s_k, dtype=dtype, device=device)
    neg_inf = torch.finfo(dtype).min
    mask[meta["und_real"] : meta["und_padded"]] = neg_inf
    mask[meta["und_padded"] + meta["gen_real"] :] = neg_inf
    return mask.view(1, 1, 1, s_k)


def gather_and_unpad(und_out, gen_out, meta, cp_mesh):
    """Gather each pathway to its full padded length, then slice off the padding."""
    und_out = _cp_gather_dim0(und_out, cp_mesh)[: meta["und_real"]]
    gen_out = _cp_gather_dim0(gen_out, cp_mesh)[: meta["gen_real"]]
    return und_out, gen_out


def cosmos3_cp_attention(cp_mesh, q_und, k_und, v_und, q_gen, k_gen, v_gen, gen_key_mask=None):
    """Ulysses context-parallel attention for the dual-pathway packed sequence.

    All inputs are *sequence-sharded*: q ``[S/cp, H, D]``, k/v ``[S/cp, Hkv, D]``
    (no batch dim). Returns ``(causal_out, full_out)`` flattened to ``[S/cp, H*D]``
    and ``[S_gen/cp, H*D]``. The understanding pathway self-attends causally; the
    generation pathway attends to the concatenation of und+gen keys/values.
    """
    world = cp_mesh.size()
    q_heads = q_und.shape[1]
    kv_heads = k_und.shape[1]
    if q_heads % world != 0:
        raise ValueError(f"Query heads ({q_heads}) must be divisible by CP world size ({world}).")

    # GQA, step 1: repeat KV heads up to a multiple of the world size so the head
    # scatter in the all-to-all is valid (each rank must receive an equal share).
    kv_head_repeats = max(world // kv_heads, 1)
    repeated_kv_heads = kv_heads * kv_head_repeats
    if repeated_kv_heads % world != 0:
        raise ValueError(f"Repeated KV heads ({repeated_kv_heads}) must be divisible by CP world size ({world}).")
    if kv_head_repeats > 1:
        k_und = _repeat_kv_heads(k_und, kv_head_repeats)
        v_und = _repeat_kv_heads(v_und, kv_head_repeats)
        k_gen = _repeat_kv_heads(k_gen, kv_head_repeats)
        v_gen = _repeat_kv_heads(v_gen, kv_head_repeats)

    # all-to-all #1: gather sequence, scatter heads -> [S, H/cp, D]
    q_und = _cp_gather_seq_scatter_heads(q_und, cp_mesh)
    k_und = _cp_gather_seq_scatter_heads(k_und, cp_mesh)
    v_und = _cp_gather_seq_scatter_heads(v_und, cp_mesh)
    q_gen = _cp_gather_seq_scatter_heads(q_gen, cp_mesh)
    k_gen = _cp_gather_seq_scatter_heads(k_gen, cp_mesh)
    v_gen = _cp_gather_seq_scatter_heads(v_gen, cp_mesh)

    # GQA, step 2: locally expand each rank's KV head shard up to its query-head
    # count, so attention runs with equal Q/K/V heads (enable_gqa=False). This lets
    # SDPA dispatch to the flash / memory-efficient kernel (O(S) memory); passing
    # enable_gqa=True instead forces the math fallback, which materializes the full
    # [S, S] score matrix and OOMs on long sequences (CP shards heads across ranks,
    # but each rank still attends over the *full* sequence length). GQA grouping is
    # preserved: local query head i pairs with local KV group i // head_repeats.
    q_local, kv_local = q_und.shape[1], k_und.shape[1]
    if q_local % kv_local != 0:
        raise ValueError(f"Local query heads ({q_local}) must be a multiple of local KV heads ({kv_local}).")
    head_repeats = q_local // kv_local
    if head_repeats > 1:
        k_und = _repeat_kv_heads(k_und, head_repeats)
        v_und = _repeat_kv_heads(v_und, head_repeats)
        k_gen = _repeat_kv_heads(k_gen, head_repeats)
        v_gen = _repeat_kv_heads(v_gen, head_repeats)

    # Understanding pathway: causal self-attention over the full und sequence.
    causal_out = dispatch_attention_fn(
        q_und.unsqueeze(0),
        k_und.unsqueeze(0),
        v_und.unsqueeze(0),
        is_causal=True,
        enable_gqa=False,
        backend=AttentionBackendName.NATIVE,
        parallel_config=None,
    ).squeeze(0)

    # Generation pathway: full attention over cat(und, gen) keys/values.
    all_k = torch.cat([k_und, k_gen], dim=0)
    all_v = torch.cat([v_und, v_gen], dim=0)
    full_out = dispatch_attention_fn(
        q_gen.unsqueeze(0),
        all_k.unsqueeze(0),
        all_v.unsqueeze(0),
        attn_mask=gen_key_mask,
        is_causal=False,
        enable_gqa=False,
        backend=AttentionBackendName.NATIVE,
        parallel_config=None,
    ).squeeze(0)

    # all-to-all #2: gather heads, scatter sequence -> [S/cp, H, D]
    causal_out = _cp_gather_heads_scatter_seq(causal_out, cp_mesh)
    full_out = _cp_gather_heads_scatter_seq(full_out, cp_mesh)
    return causal_out.flatten(-2, -1), full_out.flatten(-2, -1)


class Cosmos3CPAttnProcessor(Cosmos3AttnProcessor):
    """Cosmos 3 attention processor whose attention core runs Ulysses CP.

    It reuses the base processor's projection + rotary code (``__call__``) and overrides
    only ``_run_attention`` to bracket the two pathways with all-to-all collectives. The
    Ulysses mesh is read from ``self.cp_mesh``; the per-call generation key mask is read
    from the attention module (stamped each forward by the shard seam).
    """

    def __init__(self, cp_mesh):
        self.cp_mesh = cp_mesh

    def _run_attention(self, attn, q_und, k_und, v_und, q_gen, k_gen, v_gen):
        return cosmos3_cp_attention(
            self.cp_mesh,
            q_und,
            k_und,
            v_und,
            q_gen,
            k_gen,
            v_gen,
            gen_key_mask=getattr(attn, "_cp_gen_key_mask", None),
        )


def enable_cosmos3_context_parallel(transformer, cp_mesh):
    """Wire Ulysses context parallelism onto a ``Cosmos3OmniTransformer`` instance.

    Sets the CP attention processor on every decoder layer and installs the shard/gather
    seams so the model shards each pathway across ``cp_mesh`` and gathers before decode.
    All CP state lives in the closures below, so the model itself stays CP-free.
    """
    processor = Cosmos3CPAttnProcessor(cp_mesh)
    for layer in transformer.layers:
        layer.self_attn.set_processor(processor)

    state = {"meta": None}

    def shard_fn(und_seq, gen_seq, rotary_emb):
        und_seq, gen_seq, rotary_emb, meta = shard_cosmos3_sequence(und_seq, gen_seq, rotary_emb, cp_mesh)
        state["meta"] = meta
        gen_key_mask = build_gen_key_mask(meta, und_seq.dtype, und_seq.device)
        # The processor reads the mask off the attention module each forward.
        for layer in transformer.layers:
            layer.self_attn._cp_gen_key_mask = gen_key_mask
        return und_seq, gen_seq, rotary_emb

    def gather_fn(und_out, gen_out):
        return gather_and_unpad(und_out, gen_out, state["meta"], cp_mesh)

    transformer._cp_shard_fn = shard_fn
    transformer._cp_gather_fn = gather_fn
    return transformer


# =============================================================================
# Dense flash attention (GQA-safe; for TP without CP)
# =============================================================================
class Cosmos3FlashAttnProcessor(Cosmos3AttnProcessor):
    """Dense attention that expands the GQA KV heads to the query-head count so SDPA
    uses the flash kernel (``enable_gqa=False``) instead of the math fallback, which
    materializes the full ``[S, S]`` score matrix and OOMs on long sequences.

    No collectives: attention is computed locally over the full sequence (the head
    counts on the attention module are the rank-local values set by
    ``enable_cosmos3_tensor_parallel``). Use this with TP-only; CP installs its own
    processor that already handles the flash dispatch.
    """

    def _run_attention(self, attn, q_und, k_und, v_und, q_gen, k_gen, v_gen):
        repeats = attn.num_attention_heads // attn.num_key_value_heads
        k_und, v_und = _repeat_kv_heads(k_und, repeats), _repeat_kv_heads(v_und, repeats)
        k_gen, v_gen = _repeat_kv_heads(k_gen, repeats), _repeat_kv_heads(v_gen, repeats)

        # Understanding pathway: causal self-attention.
        causal_out = dispatch_attention_fn(
            q_und.unsqueeze(0),
            k_und.unsqueeze(0),
            v_und.unsqueeze(0),
            is_causal=True,
            enable_gqa=False,
            backend=AttentionBackendName.NATIVE,
            parallel_config=None,
        ).squeeze(0)

        # Generation pathway: full attention over cat(und, gen) keys/values.
        all_k = torch.cat([k_und, k_gen], dim=0)
        all_v = torch.cat([v_und, v_gen], dim=0)
        full_out = dispatch_attention_fn(
            q_gen.unsqueeze(0),
            all_k.unsqueeze(0),
            all_v.unsqueeze(0),
            is_causal=False,
            enable_gqa=False,
            backend=AttentionBackendName.NATIVE,
            parallel_config=None,
        ).squeeze(0)
        return causal_out.flatten(-2, -1), full_out.flatten(-2, -1)


def enable_cosmos3_flash_attention(transformer):
    """Install the dense flash-attention processor on every decoder layer."""
    processor = Cosmos3FlashAttnProcessor()
    for layer in transformer.layers:
        layer.self_attn.set_processor(processor)
    return transformer


# =============================================================================
# Tensor parallelism (shard the attention + MLP weights)
# =============================================================================
def enable_cosmos3_tensor_parallel(transformer, tp_mesh):
    """Shard every decoder layer's attention and MLP projection weights across
    ``tp_mesh`` (Megatron-style tensor parallelism), so a model whose weights don't
    fit on one GPU (e.g. Cosmos3-Super, ~120 GB) can be served across several.

    Layout per layer:
      * column-parallel (output / head dim sharded): ``to_q/to_k/to_v`` and
        ``add_q_proj/add_k_proj/add_v_proj``, plus both MLPs' ``gate_proj/up_proj``;
      * row-parallel (input dim sharded, all-reduce on output): ``to_out`` and
        ``to_add_out``, plus both MLPs' ``down_proj``.

    Each rank then owns ``num_attention_heads / tp`` query heads and
    ``num_key_value_heads / tp`` KV heads, so the per-layer head counts on every
    attention module are rewritten to their local values — the projection + reshape
    code in ``Cosmos3AttnProcessor`` reads these. The embeddings, final norms, lm_head
    and modality projections stay replicated (they're a small fraction of the weights).

    Memory note: weights are loaded to CPU first, then each layer is moved to its GPU
    and sharded in place, so the full model is never materialized on one device.

    Composes with Ulysses CP: this shards weights only and never touches the attention
    processor, so combine it with ``enable_cosmos3_context_parallel`` (CP installs its
    own processor) on a 2-D ``(tp, cp)`` mesh, or with ``enable_cosmos3_flash_attention``
    for TP without CP.
    """
    from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, parallelize_module

    tp = tp_mesh.size()
    dev = torch.device("cuda", torch.cuda.current_device())
    plan = {
        "self_attn.to_q": ColwiseParallel(),
        "self_attn.to_k": ColwiseParallel(),
        "self_attn.to_v": ColwiseParallel(),
        "self_attn.to_out": RowwiseParallel(),
        "self_attn.add_q_proj": ColwiseParallel(),
        "self_attn.add_k_proj": ColwiseParallel(),
        "self_attn.add_v_proj": ColwiseParallel(),
        "self_attn.to_add_out": RowwiseParallel(),
        "mlp.gate_proj": ColwiseParallel(),
        "mlp.up_proj": ColwiseParallel(),
        "mlp.down_proj": RowwiseParallel(),
        "mlp_moe_gen.gate_proj": ColwiseParallel(),
        "mlp_moe_gen.up_proj": ColwiseParallel(),
        "mlp_moe_gen.down_proj": RowwiseParallel(),
    }
    for layer in transformer.layers:
        attn = layer.self_attn
        if attn.num_attention_heads % tp != 0 or attn.num_key_value_heads % tp != 0:
            raise ValueError(
                f"TP degree {tp} must divide both the query heads ({attn.num_attention_heads}) "
                f"and the KV heads ({attn.num_key_value_heads})."
            )
        layer.to(dev)  # full layer transiently on this rank's GPU, then sharded in place
        parallelize_module(layer, tp_mesh, plan)
        # Projections now emit only this rank's head shard; the processor reshapes
        # with the local counts.
        attn.num_attention_heads //= tp
        attn.num_key_value_heads //= tp
        attn.num_key_value_groups = attn.num_attention_heads // attn.num_key_value_heads
    return transformer
