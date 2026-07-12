#!/usr/bin/env python3
# Copyright 2025 Stability AI and The HuggingFace Team. All rights reserved.
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
"""
Convert a Stable Audio 3 checkpoint (stable-audio-tools format) to HuggingFace
Diffusers components.

The script handles the complete SA3-Medium architecture conversion:

  ┌──────────────────────────────────────────────────────────┐
  │ Source (stabilityai/stable-audio-3-medium checkpoint)    │
  │  pretransform.model.*  →  VAE (AutoencoderSAME)          │
  │  conditioner.conditioners.seconds_total.*                 │
  │                        →  DurationEmbedder               │
  │  conditioner.conditioners.prompt.model.*                  │
  │                        →  TextEncoder (T5Gemma)           │
  │  model.*               →  DiT (StableAudio3DiTModel)     │
  └──────────────────────────────────────────────────────────┘

Key conversion details:
  • WNConv1d: weight_g / weight_v kept as-is (both sides use weight_norm).
  • Differential self-attention QKV reorder:
      ref  [q | k | v | q2 | k2]  →  diffusers  [q1 | q2 | k1 | k2 | v]
  • SAME norm renames: pre_norm→norm_attn, ff_norm→norm_ff,
      cross_attend_norm→norm2, pre_norm(DiT)→norm1.
  • SAME FF renames: ff.ff.0.proj.* → ff.proj_in.*  |  ff.ff.2.* → ff.proj_out.*
  • Bottleneck renames: scaling_factor → scale.
  • DiT weights live under the "model.model." prefix; the full architecture
    (AdaLN to_scale_shift_gate + global_cond_embedder, memory_tokens,
    rotary_pos_emb, per-block to_local_embed for inpainting) is converted.

Usage:
    python scripts/convert_stable_audio_3_to_diffusers.py \\
        --checkpoint_path stabilityai/stable-audio-3-medium \\
        --model_config_path /path/to/model_config.json \\
        --output_dir /path/to/output \\
        [--text_encoder_repo google/t5gemma-b-b-ul2] \\
        [--dtype bfloat16]
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Optional

import torch
from safetensors.torch import load_file


# Ensure UTF-8 stdout/stderr for Unicode output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


# ──────────────────────────────────────────────────────────────────────────────
# Tensor-level helpers
# ──────────────────────────────────────────────────────────────────────────────


def _reorder_differential_qkv(weight: torch.Tensor) -> torch.Tensor:
    """Reorder fused self-attention QKV weight from reference layout to diffusers layout.

    Reference layout (stable-audio-tools):
        to_qkv output chunks: [q | k | v | q2 | k2]
    Diffusers layout (StableAudio3SelfAttention):
        to_qkv output chunks: [q1 | q2 | k1 | k2 | v]
    """
    D = weight.shape[0] // 5
    q, k, v, q2, k2 = weight.split(D, dim=0)
    return torch.cat([q, q2, k, k2, v], dim=0)


# ──────────────────────────────────────────────────────────────────────────────
# Key-transformation helpers
# ──────────────────────────────────────────────────────────────────────────────

# Inside a TransformerResamplingBlock's per-layer transformer blocks:
#   ref key                     →   our key
_TRB_BLOCK_RENAMES = [
    ("pre_norm.", "norm_attn."),
    ("self_attn.to_qkv.", "attn.to_qkv."),  # fused; QKV reorder applied separately
    ("self_attn.to_out.", "attn.to_out."),
    ("self_attn.q_norm.", "attn.q_norm."),
    ("self_attn.k_norm.", "attn.k_norm."),
    ("ff_norm.", "norm_ff."),
    ("ff.ff.0.proj.", "ff.proj_in."),  # GLU → proj_in
    ("ff.ff.2.", "ff.proj_out."),  # linear_out → proj_out
]

# Inside a DiT block (StableAudio3DiTBlock). The diffusers block keeps the reference norm and
# attention names, so most entries are identity; only the SwiGLU feed-forward is renamed.
#   ref key                       →   our key
_DIT_BLOCK_RENAMES = [
    ("ff.ff.0.proj.", "ff.proj_in."),
    ("ff.ff.2.", "ff.proj_out."),
]


def _apply_renames(local_key: str, renames) -> str:
    for old, new in renames:
        if old in local_key:
            local_key = local_key.replace(old, new, 1)
            break
    return local_key


def _is_differential_qkv_key(key: str) -> bool:
    """Return True if this key is a fused QKV weight that needs reordering."""
    return "self_attn.to_qkv.weight" in key or "attn.to_qkv.weight" in key


# ──────────────────────────────────────────────────────────────────────────────
# VAE converter
# ──────────────────────────────────────────────────────────────────────────────


def _infer_trb_depth(ref_sd: dict, base: str) -> int:
    """Count how many TRB blocks exist under `base` (e.g. 'encoder.layers')."""
    depth = 0
    while f"{base}.{depth}.new_tokens" in ref_sd:
        depth += 1
    return depth


def _convert_trb_block(
    ref_sd: dict,
    ref_prefix: str,  # e.g. "encoder.layers.0."
    our_prefix: str,  # e.g. "encoder.blocks.0."
    differential: bool,
    out: dict,
    skipped: list,
):
    """Copy all keys inside one TransformerResamplingBlock."""
    for key, val in ref_sd.items():
        if not key.startswith(ref_prefix):
            continue
        local = key[len(ref_prefix) :]  # strip leading prefix

        # --- mapping (WNConv1d): weight_g, weight_v, bias copied as-is ---
        if local.startswith("mapping."):
            out[our_prefix + local] = val
            continue

        # --- new_tokens ---
        if local == "new_tokens":
            out[our_prefix + local] = val
            continue

        # --- transformers.{j}.* ---
        if local.startswith("transformers."):
            new_local = _apply_renames(local, _TRB_BLOCK_RENAMES)
            new_key = our_prefix + new_local
            if differential and _is_differential_qkv_key(new_key):
                val = _reorder_differential_qkv(val)
            out[new_key] = val
            continue

        skipped.append(key)


def convert_vae(ref_sd: dict, differential: bool = True) -> dict:
    """Build diffusers AutoencoderSAME state dict from reference checkpoint."""
    out = {}
    skipped = []

    ae_prefix = "pretransform.model."

    enc_base = ae_prefix + "encoder.layers"
    dec_base = ae_prefix + "decoder.layers"

    enc_depth = _infer_trb_depth(ref_sd, enc_base)
    dec_depth = _infer_trb_depth(ref_sd, dec_base + ".0") if f"{dec_base}.0.new_tokens" in ref_sd else 0
    # Decoder TRBs start after [Transpose, Linear, Transpose] → indices 3, 4, …
    if dec_depth == 0:
        # Count by checking higher indices
        for check_idx in range(3, 20):
            if any(k.startswith(f"{dec_base}.{check_idx}.") for k in ref_sd):
                dec_depth = check_idx - 2  # rough: layers.3 → 1 TRB etc.
                break

    if enc_depth == 0:
        print("WARNING: could not detect encoder TRB depth; defaulting to 1.")
        enc_depth = 1

    # ---- ENCODER TRBs ----
    # ref: encoder.layers.{i}  (i = 0..enc_depth-1) → our: encoder.blocks.{i}
    for i in range(enc_depth):
        _convert_trb_block(
            ref_sd,
            f"{enc_base}.{i}.",
            f"encoder.blocks.{i}.",
            differential,
            out,
            skipped,
        )

    # ---- ENCODER projection (Linear after all TRBs) ----
    # Comes after enc_depth TRBs and one Transpose → index enc_depth+1
    # But Transpose has no params, so the Linear shows as layers.{enc_depth+1}
    # (enc_depth TRBs, then 2 Transposes and 1 Linear: layers[enc_depth] = Transpose,
    # layers[enc_depth+1] = Linear)
    enc_linear_idx = enc_depth + 1
    enc_linear_prefix = f"{enc_base}.{enc_linear_idx}."
    for key, val in ref_sd.items():
        if key.startswith(enc_linear_prefix):
            suffix = key[len(enc_linear_prefix) :]
            out[f"encoder.proj.{suffix}"] = val

    # ---- DECODER projection (Linear before all TRBs) ----
    # Decoder layers: [Transpose, Linear, Transpose, TRB_0, TRB_1, ...]
    # Linear is always at index 1
    dec_linear_prefix = f"{dec_base}.1."
    for key, val in ref_sd.items():
        if key.startswith(dec_linear_prefix):
            suffix = key[len(dec_linear_prefix) :]
            out[f"decoder.proj.{suffix}"] = val

    # ---- DECODER TRBs ----
    # Decoder TRBs start at index 3 in reference.
    # Reference builds them in reverse order (channel_dims[depth]→[depth-1], …).
    # Our SAMEDecoder.blocks also reverses: blocks[0] = largest stride, same order.
    for i in range(enc_depth):  # same depth as encoder
        ref_idx = 3 + i
        _convert_trb_block(
            ref_sd,
            f"{dec_base}.{ref_idx}.",
            f"decoder.blocks.{i}.",
            differential,
            out,
            skipped,
        )

    # ---- BOTTLENECK ----
    bn_prefix = ae_prefix + "bottleneck."
    bottleneck_renames = {
        "scaling_factor": "scale",
        "bias": "bias",
        "running_std": "running_std",
    }
    for ref_suffix, our_suffix in bottleneck_renames.items():
        full_key = bn_prefix + ref_suffix
        if full_key in ref_sd:
            out[f"bottleneck.{our_suffix}"] = ref_sd[full_key]

    if skipped:
        print(f"  VAE: skipped {len(skipped)} keys (pretransform.* patcher, noise_scaling_factor, …)")

    return out


# ──────────────────────────────────────────────────────────────────────────────
# Duration embedder converter
# ──────────────────────────────────────────────────────────────────────────────


def convert_duration_embedder(ref_sd: dict, min_freq: float = 0.5, max_freq: float = 10000.0) -> dict:
    """Build diffusers StableAudio3DurationEmbedder state dict."""
    out = {}
    base = "conditioner.conditioners.seconds_total.embedder.embedding."
    # index 0 = ExpoFourierFeatures (no learnable params)
    # index 1 = nn.Linear
    for suffix in ("weight", "bias"):
        src = f"{base}1.{suffix}"
        if src in ref_sd:
            out[f"linear.{suffix}"] = ref_sd[src]
        else:
            print(f"  DurationEmbedder: key not found: {src}")

    # Compute the freqs buffer (matches StableAudio3DurationEmbedder.__init__)
    if "linear.weight" in out:
        fourier_dim = out["linear.weight"].shape[1]
        half = fourier_dim // 2
        ramp = torch.linspace(0.0, 1.0, half)
        log_min = math.log(min_freq)
        log_max = math.log(max_freq)
        freqs = torch.exp(ramp * (log_max - log_min) + log_min)
        out["freqs"] = freqs

    return out


# ──────────────────────────────────────────────────────────────────────────────
# DiT converter
# ──────────────────────────────────────────────────────────────────────────────


def convert_dit(ref_sd: dict, differential: bool = True) -> dict:
    """Build diffusers StableAudio3DiTModel state dict."""
    out = {}

    # The DiT weights live under the "model.model." prefix in the checkpoint.
    p = "model.model."

    # Top-level keys (outside the inner transformer).
    top_renames = {
        f"{p}to_timestep_embed.0.weight": "to_timestep_embed.0.weight",
        f"{p}to_timestep_embed.0.bias": "to_timestep_embed.0.bias",
        f"{p}to_timestep_embed.2.weight": "to_timestep_embed.2.weight",
        f"{p}to_timestep_embed.2.bias": "to_timestep_embed.2.bias",
        f"{p}to_cond_embed.0.weight": "to_cond_embed.0.weight",
        f"{p}to_cond_embed.2.weight": "to_cond_embed.2.weight",
        f"{p}to_global_embed.0.weight": "to_global_embed.0.weight",
        f"{p}to_global_embed.2.weight": "to_global_embed.2.weight",
        f"{p}preprocess_conv.weight": "preprocess_conv.weight",
        f"{p}postprocess_conv.weight": "postprocess_conv.weight",
        # Inner transformer module-level keys.
        f"{p}transformer.project_in.weight": "proj_in.weight",
        f"{p}transformer.project_out.weight": "proj_out.weight",
        f"{p}transformer.memory_tokens": "memory_tokens",
        f"{p}transformer.rotary_pos_emb.inv_freq": "rotary_pos_emb.inv_freq",
        f"{p}transformer.global_cond_embedder.0.weight": "global_cond_embedder.0.weight",
        f"{p}transformer.global_cond_embedder.0.bias": "global_cond_embedder.0.bias",
        f"{p}transformer.global_cond_embedder.2.weight": "global_cond_embedder.2.weight",
        f"{p}transformer.global_cond_embedder.2.bias": "global_cond_embedder.2.bias",
    }

    for ref_key, our_key in top_renames.items():
        if ref_key in ref_sd:
            out[our_key] = ref_sd[ref_key]
        else:
            print(f"  DiT: top-level key not found: {ref_key}")

    # Per-block keys: model.model.transformer.layers.{i}.*
    block_prefix = f"{p}transformer.layers."
    block_indices = set()
    for k in ref_sd:
        if k.startswith(block_prefix):
            idx_str = k[len(block_prefix) :].split(".")[0]
            if idx_str.isdigit():
                block_indices.add(int(idx_str))

    for i in sorted(block_indices):
        ref_blk = f"{block_prefix}{i}."
        for key, val in ref_sd.items():
            if not key.startswith(ref_blk):
                continue
            local = key[len(ref_blk) :]

            new_local = _apply_renames(local, _DIT_BLOCK_RENAMES)
            new_key = f"transformer_blocks.{i}.{new_local}"

            # Self-attention fused QKV needs reordering ([q|k|v|q2|k2] → [q1|q2|k1|k2|v]).
            # Differential cross-attention to_q ([q|q2]) and to_kv ([k|k2|v]) already match the
            # diffusers layout, so no cross-attention reorder is required.
            if differential and _is_differential_qkv_key(new_key):
                val = _reorder_differential_qkv(val)

            out[new_key] = val

    # The learned text-padding embedding lives on the conditioner in the reference checkpoint,
    # but the diffusers DiT owns it: it replaces padded cross-attention positions with this vector
    # and then attends to the full context (the reference disables the cross-attention mask).
    pad_key = "conditioner.conditioners.prompt.padding_embedding"
    if pad_key in ref_sd:
        out["prompt_padding_embedding"] = ref_sd[pad_key]
    else:
        print(f"  DiT: padding embedding not found: {pad_key}")

    return out


# ──────────────────────────────────────────────────────────────────────────────
# Text encoder extractor
# ──────────────────────────────────────────────────────────────────────────────


def extract_text_encoder(ref_sd: dict) -> dict:
    """Strip the conditioner prefix and return the T5Gemma model state dict."""
    prefix = "conditioner.conditioners.prompt.model."
    te_sd = {}
    for key, val in ref_sd.items():
        if key.startswith(prefix):
            te_sd[key[len(prefix) :]] = val
    return te_sd


# ──────────────────────────────────────────────────────────────────────────────
# Config inference from checkpoint
# ──────────────────────────────────────────────────────────────────────────────


def _infer_vae_config(ref_sd: dict, model_config: Optional[dict] = None) -> dict:
    """
    Infer AutoencoderSAME config from checkpoint tensor shapes.

    Some hyper-parameters (the TRB stride, the sliding-window width and the sinusoidal-FFN layer counts) are NOT
    recoverable from the weights alone — with ``variable_stride`` the encoder/decoder ``new_tokens`` collapse to a
    single shared token, so the stride is invisible. When ``model_config`` (the parsed ``model_config.json``) is
    provided, those values are read directly from it; otherwise production SAME-L/S defaults are used.
    """
    # Bottleneck scale shape: (1, latent_dim, 1)
    latent_dim = ref_sd.get(
        "pretransform.model.bottleneck.scaling_factor",
        ref_sd.get("pretransform.model.bottleneck.scale", torch.zeros(1, 256, 1)),
    ).shape[1]

    # Encoder linear weight: (latent_dim, channel_dims[-1])
    # We detect enc_depth to find the linear key
    enc_base = "pretransform.model.encoder.layers"
    enc_depth = _infer_trb_depth(ref_sd, enc_base)
    if enc_depth == 0:
        enc_depth = 1
    enc_linear_key = f"{enc_base}.{enc_depth + 1}.weight"
    if enc_linear_key in ref_sd:
        enc_final_ch = ref_sd[enc_linear_key].shape[1]  # (latent_dim, enc_final_ch)
    else:
        enc_final_ch = 768  # SAME-S default

    # TRB 0 new_tokens shape (encoder): (1, 1, out_channels)
    trb0_nt = ref_sd.get(f"{enc_base}.0.new_tokens")
    enc_out_ch = trb0_nt.shape[2] if trb0_nt is not None else enc_final_ch

    # TRB 0 mapping.weight_v shape (encoder): (out_ch, in_ch, kernel)
    mapping_wv = ref_sd.get(f"{enc_base}.0.mapping.weight_v")
    if mapping_wv is not None:
        patched_in = mapping_wv.shape[1]  # = audio_channels * patch_size
    else:
        patched_in = 512  # default: 2ch * 256 patch

    # Infer transformer depth per TRB:
    trb_trans_depth = sum(
        1
        for k in ref_sd
        if k.startswith(f"{enc_base}.0.transformers.")
        and k.endswith(".new_tokens") is False
        and ".pre_norm.alpha" in k
    )
    if trb_trans_depth == 0:
        trb_trans_depth = 6  # default

    # Dim heads inferred from q_norm shape: (dim_heads,)
    q_norm_key = f"{enc_base}.0.transformers.0.self_attn.q_norm.gamma"
    dim_heads = ref_sd[q_norm_key].shape[0] if q_norm_key in ref_sd else 64

    # channels base: choose so that enc_out_ch = channels * c_mults[0]
    # We use c_mults = [6] for both SAME-S/L
    c_mults = [6]
    enc_channels_base = enc_out_ch // c_mults[0]

    # ── Weight-invisible hyper-parameters ────────────────────────────────────
    # These come from model_config.json when available (see docstring).
    audio_channels = 2
    sliding_window = 1
    encoder_sinusoidal_blocks = [0] * enc_depth
    decoder_sinusoidal_blocks = [0] * enc_depth

    if model_config is not None:
        ae_cfg = model_config["model"]["pretransform"]["config"]
        enc_cfg = ae_cfg["encoder"]["config"]
        dec_cfg = ae_cfg["decoder"]["config"]
        strides = list(enc_cfg["strides"])
        transformer_depths = list(enc_cfg["transformer_depths"])
        c_mults = list(enc_cfg["c_mults"])
        enc_channels_base = enc_cfg["channels"]
        latent_dim = enc_cfg.get("latent_dim", latent_dim)
        dim_heads = enc_cfg.get("dim_heads", dim_heads)
        audio_channels = ae_cfg.get("io_channels", audio_channels)
        downsampling_ratio = ae_cfg["downsampling_ratio"]
        patch_size = downsampling_ratio // int(math.prod(strides))
        # sliding_window in the reference is a per-side list like [1, 1]; take the (symmetric) half-width.
        sw = enc_cfg.get("sliding_window") or [sliding_window]
        sliding_window = sw[0]
        encoder_sinusoidal_blocks = list(enc_cfg.get("sinusoidal_blocks", encoder_sinusoidal_blocks))
        decoder_sinusoidal_blocks = list(dec_cfg.get("sinusoidal_blocks", decoder_sinusoidal_blocks))
    else:
        # Stride is NOT recoverable from weights under variable_stride; assume the production value of 16.
        strides = [16] * enc_depth
        transformer_depths = [trb_trans_depth] * enc_depth
        patch_size = patched_in // audio_channels

    # chunk_size retained for back-compat; unused by the band-mask attention.
    chunk_size = 32

    return {
        "audio_channels": audio_channels,
        "patch_size": patch_size,
        "encoder_channels": enc_channels_base,
        "encoder_c_mults": c_mults,
        "encoder_strides": strides,
        "encoder_transformer_depths": transformer_depths,
        "latent_dim": latent_dim,
        "use_differential_attention": True,
        "dim_heads": dim_heads,
        "encoder_chunk_size": chunk_size,
        "ff_mult": 3,
        "sliding_window": sliding_window,
        "encoder_sinusoidal_blocks": encoder_sinusoidal_blocks,
        "decoder_sinusoidal_blocks": decoder_sinusoidal_blocks,
        "sampling_rate": 44100,
    }


def _infer_dit_config(ref_sd: dict) -> dict:
    """Infer StableAudio3DiTModel config from checkpoint tensor shapes."""
    p = "model.model."

    # embed_dim from to_timestep_embed.0.weight shape (embed_dim, features_dim)
    ts_w = ref_sd.get(f"{p}to_timestep_embed.0.weight")
    embed_dim = ts_w.shape[0] if ts_w is not None else 1536
    timestep_features_dim = ts_w.shape[1] if ts_w is not None else 256

    # depth: count transformer blocks (RMSNorm → pre_norm.gamma)
    depth = 0
    while f"{p}transformer.layers.{depth}.pre_norm.gamma" in ref_sd:
        depth += 1
    if depth == 0:
        depth = 24

    # num_heads: from self_attn.q_norm.gamma shape (dim_heads,) and embed_dim
    q_norm_key = f"{p}transformer.layers.0.self_attn.q_norm.gamma"
    dim_heads = ref_sd[q_norm_key].shape[0] if q_norm_key in ref_sd else 64
    num_heads = embed_dim // dim_heads

    # cond_token_dim / global_cond_dim from the projection in-weights (embed_dim, *)
    cond_w = ref_sd.get(f"{p}to_cond_embed.0.weight")
    cond_token_dim = cond_w.shape[1] if cond_w is not None else 768
    glob_w = ref_sd.get(f"{p}to_global_embed.0.weight")
    global_cond_dim = glob_w.shape[1] if glob_w is not None else 768

    # io_channels from preprocess_conv.weight (io_ch, io_ch, 1)
    pc_w = ref_sd.get(f"{p}preprocess_conv.weight")
    io_channels = pc_w.shape[0] if pc_w is not None else 256

    # ff_mult from ff.ff.0.proj.weight (inner*2, embed_dim)
    ff_w = ref_sd.get(f"{p}transformer.layers.0.ff.ff.0.proj.weight")
    ff_mult = (ff_w.shape[0] // (2 * embed_dim)) if ff_w is not None else 4

    # local_add_cond_dim from to_local_embed.0.weight (embed_dim, local_add_cond_dim)
    loc_w = ref_sd.get(f"{p}transformer.layers.0.to_local_embed.0.weight")
    local_add_cond_dim = loc_w.shape[1] if loc_w is not None else 257

    # num_memory_tokens from memory_tokens (num_memory_tokens, embed_dim)
    mem = ref_sd.get(f"{p}transformer.memory_tokens")
    num_memory_tokens = mem.shape[0] if mem is not None else 64

    # differential: self_attn.to_qkv rows = embed_dim*5 (differential) vs *3 (standard)
    qkv_w = ref_sd.get(f"{p}transformer.layers.0.self_attn.to_qkv.weight")
    use_differential = qkv_w is not None and qkv_w.shape[0] == embed_dim * 5

    return {
        "io_channels": io_channels,
        "patch_size": 1,
        "embed_dim": embed_dim,
        "depth": depth,
        "num_heads": num_heads,
        "cond_token_dim": cond_token_dim,
        "global_cond_dim": global_cond_dim,
        "local_add_cond_dim": local_add_cond_dim,
        "timestep_features_dim": timestep_features_dim,
        "ff_mult": ff_mult,
        "num_memory_tokens": num_memory_tokens,
        "use_differential_attention": use_differential,
    }


def _infer_duration_embedder_config(ref_sd: dict) -> dict:
    """Infer StableAudio3DurationEmbedder config from checkpoint shapes."""
    w = ref_sd.get("conditioner.conditioners.seconds_total.embedder.embedding.1.weight")
    if w is not None:
        output_dim = w.shape[0]
        fourier_dim = w.shape[1]
    else:
        output_dim, fourier_dim = 768, 256
    return {
        "output_dim": output_dim,
        "fourier_dim": fourier_dim,
        "min_val": 0.0,
        "max_val": 384.0,
        "min_freq": 0.5,
        "max_freq": 10000.0,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main conversion
# ──────────────────────────────────────────────────────────────────────────────


def convert(args):
    # ── Load checkpoint ──────────────────────────────────────────────────────
    checkpoint_path = args.checkpoint_path
    hub_model_config_path = None
    if not Path(checkpoint_path).exists():
        # Try to download from HF Hub
        try:
            from huggingface_hub import hf_hub_download

            print(f"Downloading checkpoint from HF Hub: {checkpoint_path}")
            repo_id = checkpoint_path
            checkpoint_path = hf_hub_download(
                repo_id=repo_id,
                filename="model.safetensors",
            )
            # Also grab model_config.json so we can pick the right scheduler.
            try:
                hub_model_config_path = hf_hub_download(repo_id=repo_id, filename="model_config.json")
            except Exception:
                hub_model_config_path = None
        except Exception as exc:
            print(f"Could not download checkpoint: {exc}")
            sys.exit(1)

    print(f"Loading checkpoint: {checkpoint_path}")
    ref_sd = load_file(checkpoint_path, device="cpu")
    print(f"  Loaded {len(ref_sd)} tensors.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dtype = getattr(torch, args.dtype)

    import diffusers
    from diffusers import (
        AutoencoderSAME,
        PingPongScheduler,
        StableAudio3DiTModel,
        StableAudio3DurationEmbedder,
        StableAudio3EulerScheduler,
        StableAudio3Pipeline,
    )

    # ── Parse model_config.json if provided ─────────────────────────────────
    model_config = None
    model_config_path = args.model_config_path or hub_model_config_path
    if model_config_path and Path(model_config_path).exists():
        with open(model_config_path) as f:
            model_config = json.load(f)
        print(f"Loaded model config: {model_config_path}")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 1. VAE
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print("\n── Converting VAE ──────────────────────────────────────────────────")
    vae_cfg = _infer_vae_config(ref_sd, model_config)
    print(f"  Inferred VAE config: {vae_cfg}")

    vae_sd = convert_vae(ref_sd, differential=vae_cfg["use_differential_attention"])

    vae = AutoencoderSAME(**vae_cfg)
    missing, unexpected = vae.load_state_dict(vae_sd, strict=False)
    if missing:
        print(f"  VAE missing keys: {missing[:10]}")
    if unexpected:
        print(f"  VAE unexpected keys: {unexpected[:10]}")
    print(f"  VAE: loaded {len(vae_sd)} keys, {len(missing)} missing, {len(unexpected)} unexpected")

    vae = vae.to(dtype)
    vae.save_pretrained(output_dir / "vae")
    print(f"  Saved → {output_dir / 'vae'}")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 2. Duration embedder
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print("\n── Converting DurationEmbedder ─────────────────────────────────────")
    dur_cfg = _infer_duration_embedder_config(ref_sd)
    print(f"  Inferred config: {dur_cfg}")

    dur_sd = convert_duration_embedder(ref_sd)
    dur_emb = StableAudio3DurationEmbedder(**dur_cfg)
    dur_missing, dur_unexpected = dur_emb.load_state_dict(dur_sd, strict=False)
    if dur_missing:
        print(f"  DurationEmbedder missing keys: {dur_missing}")

    dur_emb = dur_emb.to(dtype)
    dur_emb.save_pretrained(output_dir / "duration_embedder")
    print(f"  Saved → {output_dir / 'duration_embedder'}")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 3. Text encoder + tokenizer
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print("\n── Converting TextEncoder ──────────────────────────────────────────")
    te_sd = extract_text_encoder(ref_sd)

    text_encoder_repo = args.text_encoder_repo
    # Concrete class names for model_index.json. diffusers cannot load the
    # abstract "AutoTokenizer"/"AutoModel" entries, so we record the resolved
    # concrete classes (e.g. "GemmaTokenizerFast", "T5GemmaEncoderModel").
    tokenizer_cls_name = "AutoTokenizer"
    text_encoder_cls_name = "T5GemmaEncoderModel"
    try:
        from transformers import AutoConfig, AutoTokenizer, T5GemmaEncoderModel

        print(f"  Loading T5Gemma from: {text_encoder_repo}")
        tokenizer = AutoTokenizer.from_pretrained(text_encoder_repo)
        tokenizer_cls_name = type(tokenizer).__name__
        te_config = AutoConfig.from_pretrained(text_encoder_repo)
        te_config.is_encoder_decoder = False
        text_encoder = T5GemmaEncoderModel.from_pretrained(text_encoder_repo, config=te_config)

        if te_sd:
            print(f"  Applying {len(te_sd)} weights extracted from SA3 checkpoint …")
            te_missing, te_unexpected = text_encoder.load_state_dict(te_sd, strict=False)
            if te_missing:
                print(f"    TE missing: {te_missing[:5]} …")
            if te_unexpected:
                print(f"    TE unexpected: {te_unexpected[:5]} …")
        else:
            print(
                "  No text-encoder weights found in SA3 checkpoint (expected if frozen). Using base T5Gemma weights."
            )

        text_encoder = text_encoder.to(dtype)
        text_encoder_cls_name = type(text_encoder).__name__
        text_encoder.save_pretrained(output_dir / "text_encoder")
        tokenizer.save_pretrained(output_dir / "tokenizer")
        print(f"  Saved → {output_dir / 'text_encoder'}, {output_dir / 'tokenizer'}")
    except ImportError:
        print("  WARNING: transformers not installed; skipping text_encoder & tokenizer.")
    except Exception as exc:
        print(f"  WARNING: text encoder conversion failed: {exc}")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 4. DiT
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print("\n── Converting DiT ──────────────────────────────────────────────────")
    dit_cfg = _infer_dit_config(ref_sd)
    print(f"  Inferred DiT config: {dit_cfg}")

    dit_sd = convert_dit(ref_sd, differential=dit_cfg["use_differential_attention"])

    transformer = StableAudio3DiTModel(**dit_cfg)
    dit_missing, dit_unexpected = transformer.load_state_dict(dit_sd, strict=False)
    if dit_missing:
        print(f"  DiT missing keys ({len(dit_missing)}): {dit_missing[:5]} …")
    if dit_unexpected:
        print(f"  DiT unexpected keys ({len(dit_unexpected)}): {dit_unexpected[:5]} …")
    print(f"  DiT: loaded {len(dit_sd)} keys ({len(dit_missing)} missing, {len(dit_unexpected)} unexpected)")

    transformer = transformer.to(dtype)
    transformer.save_pretrained(output_dir / "transformer")
    print(f"  Saved → {output_dir / 'transformer'}")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 5. Scheduler
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print("\n── Saving scheduler ────────────────────────────────────────────────")
    # The base model is a (non-distilled) rectified_flow model that samples with deterministic
    # Euler over many steps; the distilled model (rf_denoiser) uses the 8-step ping-pong sampler.
    diffusion_objective = None
    if model_config is not None:
        diffusion_objective = (
            model_config.get("model", {}).get("diffusion", {}).get("diffusion_objective")
            or model_config.get("model", {}).get("diffusion_objective")
            or model_config.get("diffusion_objective")
        )

    if diffusion_objective == "rf_denoiser":
        scheduler = PingPongScheduler(
            num_inference_steps=8,
            logsnr_min=-6.2,
            logsnr_max=2.0,
        )
        scheduler_cls_name = "PingPongScheduler"
    else:
        # Default (and explicit "rectified_flow"): base model -> deterministic Euler sampler.
        if diffusion_objective is None:
            print("  diffusion_objective not found in model_config; defaulting to Euler (base model).")
        scheduler = StableAudio3EulerScheduler(
            num_inference_steps=100,
            logsnr_min=-6.2,
            logsnr_max=2.0,
        )
        scheduler_cls_name = "StableAudio3EulerScheduler"
    scheduler.save_pretrained(output_dir / "scheduler")
    print(f"  Saved {scheduler_cls_name} → {output_dir / 'scheduler'}")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 6. model_index.json
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print("\n── Writing model_index.json ─────────────────────────────────────────")
    model_index = {
        "_class_name": "StableAudio3Pipeline",
        "_diffusers_version": diffusers.__version__,
        "vae": ["diffusers", "AutoencoderSAME"],
        "text_encoder": ["transformers", text_encoder_cls_name],
        "tokenizer": ["transformers", tokenizer_cls_name],
        "duration_embedder": ["diffusers", "StableAudio3DurationEmbedder"],
        "transformer": ["diffusers", "StableAudio3DiTModel"],
        "scheduler": ["diffusers", scheduler_cls_name],
    }
    with open(output_dir / "model_index.json", "w") as f:
        json.dump(model_index, f, indent=2)
    print(f"  Saved → {output_dir / 'model_index.json'}")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 7. Round-trip sanity check
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    if args.skip_sanity_check:
        print("\n── Skipping round-trip sanity check (--skip_sanity_check) ─────────")
    else:
        print("\n── Round-trip sanity check ──────────────────────────────────────────")
        try:
            pipeline = StableAudio3Pipeline.from_pretrained(str(output_dir))
            print("  ✓ Pipeline loaded successfully.")

            # Quick VAE encode/decode check
            dummy = torch.zeros(1, 2, 44100)
            with torch.no_grad():
                lat = pipeline.vae.encode(dummy).latents
                rec = pipeline.vae.decode(lat).sample
            print(f"  ✓ VAE encode→decode: input {dummy.shape} → latent {lat.shape} → output {rec.shape}")

            # Quick duration embedder check
            with torch.no_grad():
                d = pipeline.duration_embedder(torch.tensor([10.0]))
            print(f"  ✓ DurationEmbedder output shape: {d.shape}")

        except Exception as exc:
            print(f"  WARNING: sanity check failed: {exc}")

    print(f"\n✓ Conversion complete. Output at: {output_dir}")
    print("\nThe full DiT architecture is converted (AdaLN, memory tokens, RoPE, inpaint).")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(description="Convert Stable Audio 3 checkpoint to HuggingFace Diffusers.")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to model.safetensors, OR HF repo id (e.g. stabilityai/stable-audio-3-medium).",
    )
    parser.add_argument(
        "--model_config_path",
        type=str,
        default=None,
        help="Optional path to model_config.json from the SA3 repo.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where the converted pipeline will be saved.",
    )
    parser.add_argument(
        "--text_encoder_repo",
        type=str,
        default="google/t5gemma-b-b-ul2",
        help="HF model id or local path for the T5Gemma text encoder.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float16", "bfloat16"],
        help="Output dtype for saved weights.",
    )
    parser.add_argument(
        "--skip_sanity_check",
        action="store_true",
        help="Skip the round-trip load/forward-pass sanity check.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    convert(args)
