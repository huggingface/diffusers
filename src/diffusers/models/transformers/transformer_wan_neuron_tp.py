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

"""Neuron-specific Tensor Parallelism utilities for WanTransformer3DModel.

Entry point::

    apply_tp_wan_transformer_neuron(model, tp_mesh)

Apply TP to a ``WanTransformer3DModel`` for AWS Neuron.  The model weights
must still be on CPU when this function is called.  Move to the Neuron device
*after* this call::

    apply_tp_wan_transformer_neuron(transformer, tp_mesh)
    transformer = transformer.to(device)

TP plan per ``WanTransformerBlock``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Self-attention (``attn1``):
  ``to_q``, ``to_k``, ``to_v``  →  ``ColwiseParallel``
  ``to_out.0``                   →  ``RowwiseParallel``

Cross-attention (``attn2``):
  ``to_q``, ``to_k``, ``to_v``  →  ``ColwiseParallel``
  ``to_out.0``                   →  ``RowwiseParallel``

Feed-forward (GELU-approximate):
  ``ffn.net.0.proj``  →  ``ColwiseParallel``  (dim → ffn_dim // tp_size)
  ``ffn.net.2``       →  ``RowwiseParallel``  (ffn_dim // tp_size → dim)

Non-TP'd layers (small relative to 40 blocks; replicated):
  ``patch_embedding``, ``condition_embedder``, ``norm_out``, ``proj_out``,
  ``scale_shift_table``.

norm_q / norm_k correctness
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
``WanAttention`` applies ``RMSNorm(inner_dim=5120)`` to Q and K *before*
splitting into heads.  After ``ColwiseParallel`` on ``to_q``/``to_k`` each
rank holds ``inner_dim // tp_size`` features.  ``WanAttnProcessorTP``
handles this via ``_apply_global_rms_norm``: the RMS is computed globally
across all TP ranks via ``dist.all_reduce(SUM)`` over local sum-of-squares,
so every rank applies the same scale — identical to the non-TP result.
The norm weight is sliced to each rank's portion.

RoPE float64 → bfloat16 cast
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
``WanRotaryPosEmbed`` stores ``freqs_cos`` / ``freqs_sin`` as float64 for
construction precision.  ``WanAttnProcessorTP`` casts to ``hidden_states.dtype``
(bfloat16) before computing RoPE to avoid mixed-dtype XLA ops which produce
NaN on Neuron.

RoPE non-contiguous patch
~~~~~~~~~~~~~~~~~~~~~~~~~~~
``WanRotaryPosEmbed.forward`` uses ``.expand()`` which produces non-contiguous
views; the XLA cat kernel rejects them.
``apply_tp_wan_transformer_neuron`` replaces ``transformer.rope.forward`` with
``_rope_forward_contiguous`` which calls ``.contiguous()`` on each expanded
tensor before the cat.
"""

import torch
import torch.distributed as dist
import torch.nn as nn

from .transformer_flux2_neuron_tp import _pre_shard_and_tp
from .transformer_wan import WanAttnProcessor


class WanAttnProcessorTP(WanAttnProcessor):
    """TP-aware attention processor for ``WanAttention`` on Neuron.

    Differences from ``WanAttnProcessor``:

    1. ``local_heads = attn.heads // tp_size`` — consistent with the sharded
       ``inner_dim`` after ``ColwiseParallel``.
    2. ``_apply_global_rms_norm`` — computes the RMS globally across all TP
       ranks via ``dist.all_reduce(SUM)`` over local sum-of-squares, then
       applies the weight slice for this rank's portion.
    3. RoPE dtype cast — ``freqs_cos``/``freqs_sin`` are cast to
       ``hidden_states.dtype`` (bfloat16) before use to avoid NaN on Neuron.
    4. I2V path (``add_k_proj`` / ``add_v_proj``) is not implemented — T2V only.

    Args:
        tp_size: Total tensor-parallel degree (``tp_mesh.size()``).
        rank: Current rank (``dist.get_rank()``).
    """

    def __init__(self, tp_size: int, rank: int):
        super().__init__()
        self.tp_size = tp_size
        self.rank = rank

    def _apply_global_rms_norm(
        self,
        x: torch.Tensor,
        norm_module: nn.Module,
        local_dim: int,
    ) -> torch.Tensor:
        """RMSNorm with global RMS computed across all TP ranks via all-reduce.

        ``x``: ``[B, S, local_dim]`` — local shard after ColwiseParallel.
        ``norm_module.weight``: ``[inner_dim = local_dim * tp_size]``.

        The global RMS = ``sqrt(sum_all_ranks(local_sum_sq) / inner_dim + eps)``.
        All ranks contribute their local sum-of-squares via
        ``dist.all_reduce(SUM)``, so each rank uses the same scale — identical
        to the non-TP RMSNorm result.
        """
        start = self.rank * local_dim
        end = start + local_dim
        w = norm_module.weight[start:end]
        eps = getattr(norm_module, "eps", 1e-5)
        inner_dim = local_dim * self.tp_size
        # Local sum of squares [B, S, 1]; summed across TP ranks.
        local_sq_sum = x.float().pow(2).sum(dim=-1, keepdim=True)
        dist.all_reduce(local_sq_sum, op=dist.ReduceOp.SUM)
        rms = local_sq_sum.div(inner_dim).add(eps).rsqrt()
        return (x.float() * rms * w.float()).type_as(x)

    def __call__(
        self,
        attn: "WanAttention",  # noqa: F821
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        from torch.distributed.tensor import DTensor

        def _to_local(t: torch.Tensor) -> torch.Tensor:
            return t.to_local() if isinstance(t, DTensor) else t

        local_heads = attn.heads // self.tp_size
        orig_dtype = hidden_states.dtype

        # to_q/to_k/to_v are ColwiseParallel — output is a Shard(-1) DTensor;
        # _to_local extracts the plain local shard [B, S, local_dim].
        if encoder_hidden_states is None:
            # Self-attention
            query = _to_local(attn.to_q(hidden_states))
            key = _to_local(attn.to_k(hidden_states))
            value = _to_local(attn.to_v(hidden_states))
        else:
            # Cross-attention: encoder_hidden_states is replicated on all ranks.
            # Run in float32 so that the (cond - uncond) guidance difference is
            # at full precision; bfloat16 errors here get amplified by CFG scale.
            query = _to_local(attn.to_q(hidden_states)).float()
            key = _to_local(attn.to_k(encoder_hidden_states)).float()
            value = _to_local(attn.to_v(encoder_hidden_states)).float()

        local_dim = query.shape[-1]  # inner_dim // tp_size

        # Global RMSNorm with weight sliced to this rank's portion.
        query = self._apply_global_rms_norm(query, attn.norm_q, local_dim)
        key = self._apply_global_rms_norm(key, attn.norm_k, local_dim)

        # Reshape: [B, S, local_dim] → [B, S, local_heads, head_dim]
        query = query.unflatten(-1, (local_heads, -1))
        key = key.unflatten(-1, (local_heads, -1))
        value = value.unflatten(-1, (local_heads, -1))

        # RoPE only in self-attention (cross-attention has rotary_emb=None).
        # freqs_cos/sin are float64; cast to hidden_states dtype to avoid NaN.
        if rotary_emb is not None:
            freqs_cos, freqs_sin = rotary_emb

            def _apply_rotary_emb(hs, fc, fs):
                x1, x2 = hs.unflatten(-1, (-1, 2)).unbind(-1)
                cos = fc[..., 0::2].to(hs.dtype)
                sin = fs[..., 1::2].to(hs.dtype)
                out = torch.empty_like(hs)
                out[..., 0::2] = x1 * cos - x2 * sin
                out[..., 1::2] = x1 * sin + x2 * cos
                return out

            query = _apply_rotary_emb(query, freqs_cos, freqs_sin)
            key = _apply_rotary_emb(key, freqs_cos, freqs_sin)

        # BSHD-layout attention; parallel_config only for self-attention.
        from ..attention_dispatch import dispatch_attention_fn

        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=self._attention_backend,
            parallel_config=(self._parallel_config if encoder_hidden_states is None else None),
        )
        # [B, S, local_heads, head_dim] → [B, S, local_dim]
        # Cast back to orig_dtype so RowwiseParallel to_out receives expected dtype.
        hidden_states = hidden_states.flatten(2, 3).to(orig_dtype)

        # to_out[0] is RowwiseParallel: local matmul + all-reduce → [B, S, inner_dim].
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)  # Dropout (no-op at eval)
        return hidden_states


def _make_rope_forward_contiguous(rope_mod: nn.Module):
    """Return a patched ``WanRotaryPosEmbed.forward`` that calls ``.contiguous()``
    on expanded tensors before ``torch.cat``.

    ``WanRotaryPosEmbed.forward`` uses ``.expand()`` which produces non-contiguous
    views.  The XLA cat kernel on Neuron rejects non-contiguous inputs.  This
    patch is identical in semantics but inserts ``.contiguous()`` after each
    expand to ensure packed memory layouts.
    """

    def _forward(hidden_states: torch.Tensor):
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = rope_mod.patch_size
        ppf = num_frames // p_t
        pph = height // p_h
        ppw = width // p_w
        split_sizes = [rope_mod.t_dim, rope_mod.h_dim, rope_mod.w_dim]
        freqs_cos_split = rope_mod.freqs_cos.split(split_sizes, dim=1)
        freqs_sin_split = rope_mod.freqs_sin.split(split_sizes, dim=1)
        # .expand() produces non-contiguous views; .contiguous() copies them so
        # that XLA cat sees packed memory layouts.
        freqs_cos_f = freqs_cos_split[0][:ppf].view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1).contiguous()
        freqs_cos_h = freqs_cos_split[1][:pph].view(1, pph, 1, -1).expand(ppf, pph, ppw, -1).contiguous()
        freqs_cos_w = freqs_cos_split[2][:ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1).contiguous()
        freqs_sin_f = freqs_sin_split[0][:ppf].view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1).contiguous()
        freqs_sin_h = freqs_sin_split[1][:pph].view(1, pph, 1, -1).expand(ppf, pph, ppw, -1).contiguous()
        freqs_sin_w = freqs_sin_split[2][:ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1).contiguous()
        freqs_cos_out = torch.cat([freqs_cos_f, freqs_cos_h, freqs_cos_w], dim=-1).reshape(
            1, ppf * pph * ppw, 1, -1
        )
        freqs_sin_out = torch.cat([freqs_sin_f, freqs_sin_h, freqs_sin_w], dim=-1).reshape(
            1, ppf * pph * ppw, 1, -1
        )
        return freqs_cos_out, freqs_sin_out

    return _forward


def apply_tp_wan_transformer_neuron(
    model: "WanTransformer3DModel",  # noqa: F821
    tp_mesh: "torch.distributed.device_mesh.DeviceMesh",
) -> "WanTransformer3DModel":  # noqa: F821
    """Apply tensor parallelism to a ``WanTransformer3DModel`` on Neuron.

    Steps:

    1. Patch ``model.rope.forward`` with ``_make_rope_forward_contiguous`` to
       fix XLA non-contiguous tensor errors in ``WanRotaryPosEmbed``.
    2. For each ``WanTransformerBlock``:

       a. Pre-shard Linear weights via ``DTensor.from_local`` (workaround for
          the Neuron NRT consecutive-reduce-scatter bug on large tensors).
       b. Call ``parallelize_module`` to register input/output hooks.
       c. Replace the attention processor on ``attn1`` and ``attn2`` with
          ``WanAttnProcessorTP``.

    The model weights must still be on CPU when this function is called.
    Move to the Neuron device *after*::

        apply_tp_wan_transformer_neuron(transformer, tp_mesh)
        transformer = transformer.to(device)

    Args:
        model: ``WanTransformer3DModel`` with weights on CPU.
        tp_mesh: 1-D Neuron device mesh of size ``tp_size``.

    Returns:
        The same ``model`` instance, modified in-place.
    """
    from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel

    rank = dist.get_rank()
    tp_size = tp_mesh.size()

    # Patch RoPE forward to add .contiguous() before XLA cat.
    model.rope.forward = _make_rope_forward_contiguous(model.rope)

    # TP plan per WanTransformerBlock.
    # No weight permutations needed: WAN uses separate Q/K/V linears (not fused)
    # and GELU (not SwiGLU), so column-wise slicing is always correct.
    block_plan = {
        # Self-attention
        "attn1.to_q": ColwiseParallel(),
        "attn1.to_k": ColwiseParallel(),
        "attn1.to_v": ColwiseParallel(),
        "attn1.to_out.0": RowwiseParallel(),
        # Cross-attention (encoder_hidden_states replicated on all ranks)
        "attn2.to_q": ColwiseParallel(),
        "attn2.to_k": ColwiseParallel(),
        "attn2.to_v": ColwiseParallel(),
        "attn2.to_out.0": RowwiseParallel(),
        # Feed-forward: net[0] is GELU (has .proj), net[2] is output Linear
        "ffn.net.0.proj": ColwiseParallel(),
        "ffn.net.2": RowwiseParallel(),
    }

    processor = WanAttnProcessorTP(tp_size=tp_size, rank=rank)

    for block in model.blocks:
        _pre_shard_and_tp(block, tp_mesh, block_plan, rank, tp_size)
        block.attn1.set_processor(processor)
        block.attn2.set_processor(processor)

    return model
