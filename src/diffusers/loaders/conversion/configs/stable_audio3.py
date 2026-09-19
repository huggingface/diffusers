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

"""Original configuration helpers and model presets for the stable_audio3 assembly recipe."""

import math
from typing import Optional

import torch


def _infer_trb_depth(ref_sd: dict, base: str) -> int:
    """Count how many TRB blocks exist under `base` (e.g. 'encoder.layers')."""
    depth = 0
    while f"{base}.{depth}.new_tokens" in ref_sd:
        depth += 1
    return depth


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


__all__ = ["_infer_dit_config", "_infer_duration_embedder_config", "_infer_vae_config"]
