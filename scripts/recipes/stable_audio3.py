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
    python scripts/recipes/stable_audio3.py \\
        --checkpoint_path stabilityai/stable-audio-3-medium \\
        --model_config_path /path/to/model_config.json \\
        --output_dir /path/to/output \\
        [--text_encoder_repo google/t5gemma-b-b-ul2] \\
        [--dtype bfloat16]
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

from diffusers.loaders.conversion.checkpoint import convert_component_checkpoint
from diffusers.loaders.conversion.configs.stable_audio3 import (
    _infer_dit_config,
    _infer_duration_embedder_config,
    _infer_vae_config,
)


# Ensure UTF-8 stdout/stderr for Unicode output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


# ──────────────────────────────────────────────────────────────────────────────
# Tensor-level helpers
# ──────────────────────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────────────────────────
# Key-transformation helpers
# ──────────────────────────────────────────────────────────────────────────────

# Inside a TransformerResamplingBlock's per-layer transformer blocks:
#   ref key                     →   our key

# Inside a DiT block (StableAudio3DiTBlock). The diffusers block keeps the reference norm and
# attention names, so most entries are identity; only the SwiGLU feed-forward is renamed.
#   ref key                       →   our key


# ──────────────────────────────────────────────────────────────────────────────
# VAE converter
# ──────────────────────────────────────────────────────────────────────────────


def convert_vae(ref_sd, config):
    prefix = "pretransform.model."
    state = {key.removeprefix(prefix): value for key, value in ref_sd.items() if key.startswith(prefix)}
    return convert_component_checkpoint(state, config, "AutoencoderSAME")


# ──────────────────────────────────────────────────────────────────────────────
# Duration embedder converter
# ──────────────────────────────────────────────────────────────────────────────


def convert_duration_embedder(ref_sd):
    state = {key: value for key, value in ref_sd.items() if key.startswith("conditioner.conditioners.seconds_total.")}
    state.pop("conditioner.conditioners.seconds_total.embedder.embedding.0.weights", None)
    return convert_component_checkpoint(state, _infer_duration_embedder_config(ref_sd), "StableAudio3DurationEmbedder")


# ──────────────────────────────────────────────────────────────────────────────
# DiT converter
# ──────────────────────────────────────────────────────────────────────────────


def convert_dit(ref_sd, config):
    state = {
        key: value
        for key, value in ref_sd.items()
        if key.startswith("model.model.") or key == "conditioner.conditioners.prompt.padding_embedding"
    }
    return convert_component_checkpoint(state, config, "StableAudio3DiTModel")


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
        FlowMatchEulerDiscreteScheduler,
        StableAudio3DiTModel,
        StableAudio3DurationEmbedder,
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
    print("\n-- Converting VAE --------------------------------------------------")
    vae_cfg = _infer_vae_config(ref_sd, model_config)
    print(f"  Inferred VAE config: {vae_cfg}")

    vae_sd = convert_vae(ref_sd, config=vae_cfg)

    vae = AutoencoderSAME(**vae_cfg)
    missing, unexpected = vae.load_state_dict(vae_sd, strict=False)
    if missing:
        print(f"  VAE missing keys: {missing[:10]}")
    if unexpected:
        print(f"  VAE unexpected keys: {unexpected[:10]}")
    print(f"  VAE: loaded {len(vae_sd)} keys, {len(missing)} missing, {len(unexpected)} unexpected")

    vae = vae.to(dtype)
    vae.save_pretrained(output_dir / "vae")
    print(f"  Saved -> {output_dir / 'vae'}")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 2. Duration embedder
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print("\n-- Converting DurationEmbedder -------------------------------------")
    dur_cfg = _infer_duration_embedder_config(ref_sd)
    print(f"  Inferred config: {dur_cfg}")

    dur_sd = convert_duration_embedder(ref_sd)
    dur_emb = StableAudio3DurationEmbedder(**dur_cfg)
    dur_missing, dur_unexpected = dur_emb.load_state_dict(dur_sd, strict=False)
    if dur_missing:
        print(f"  DurationEmbedder missing keys: {dur_missing}")

    dur_emb = dur_emb.to(dtype)
    dur_emb.save_pretrained(output_dir / "duration_embedder")
    print(f"  Saved -> {output_dir / 'duration_embedder'}")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 3. Text encoder + tokenizer
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print("\n-- Converting TextEncoder ------------------------------------------")
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
            print(f"  Applying {len(te_sd)} weights extracted from SA3 checkpoint ...")
            te_missing, te_unexpected = text_encoder.load_state_dict(te_sd, strict=False)
            if te_missing:
                print(f"    TE missing: {te_missing[:5]} ...")
            if te_unexpected:
                print(f"    TE unexpected: {te_unexpected[:5]} ...")
        else:
            print(
                "  No text-encoder weights found in SA3 checkpoint (expected if frozen). Using base T5Gemma weights."
            )

        text_encoder = text_encoder.to(dtype)
        text_encoder_cls_name = type(text_encoder).__name__
        text_encoder.save_pretrained(output_dir / "text_encoder")
        tokenizer.save_pretrained(output_dir / "tokenizer")
        print(f"  Saved -> {output_dir / 'text_encoder'}, {output_dir / 'tokenizer'}")
    except ImportError:
        print("  WARNING: transformers not installed; skipping text_encoder & tokenizer.")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 4. DiT
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print("\n-- Converting DiT --------------------------------------------------")
    dit_cfg = _infer_dit_config(ref_sd)
    print(f"  Inferred DiT config: {dit_cfg}")

    dit_sd = convert_dit(ref_sd, config=dit_cfg)

    transformer = StableAudio3DiTModel(**dit_cfg)
    dit_missing, dit_unexpected = transformer.load_state_dict(dit_sd, strict=False)
    if dit_missing:
        print(f"  DiT missing keys ({len(dit_missing)}): {dit_missing[:5]} ...")
    if dit_unexpected:
        print(f"  DiT unexpected keys ({len(dit_unexpected)}): {dit_unexpected[:5]} ...")
    print(f"  DiT: loaded {len(dit_sd)} keys ({len(dit_missing)} missing, {len(dit_unexpected)} unexpected)")

    transformer = transformer.to(dtype)
    transformer.save_pretrained(output_dir / "transformer")
    print(f"  Saved -> {output_dir / 'transformer'}")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 5. Scheduler
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print("\n-- Saving scheduler ------------------------------------------------")
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
        # Distilled model: stochastic ping-pong sampling, 8 steps by default.
        scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1, shift=1.0, stochastic_sampling=True)
    else:
        # Default (and explicit "rectified_flow"): base model -> deterministic Euler sampler, ~100 steps.
        if diffusion_objective is None:
            print("  diffusion_objective not found in model_config; defaulting to Euler (base model).")
        scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1, shift=1.0, stochastic_sampling=False)
    scheduler_cls_name = "FlowMatchEulerDiscreteScheduler"
    scheduler.save_pretrained(output_dir / "scheduler")
    print(f"  Saved {scheduler_cls_name} -> {output_dir / 'scheduler'}")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 6. model_index.json
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print("\n-- Writing model_index.json -----------------------------------------")
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
    print(f"  Saved -> {output_dir / 'model_index.json'}")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 7. Round-trip sanity check
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    if args.skip_sanity_check:
        print("\n-- Skipping round-trip sanity check (--skip_sanity_check) ---------")
    else:
        print("\n-- Round-trip sanity check ------------------------------------------")
        try:
            pipeline = StableAudio3Pipeline.from_pretrained(str(output_dir))
            print("  [DONE] Pipeline loaded successfully.")

            # Quick VAE encode/decode check
            dummy = torch.zeros(1, 2, 44100)
            with torch.no_grad():
                lat = pipeline.vae.encode(dummy).latents
                rec = pipeline.vae.decode(lat).sample
            print(f"  [DONE] VAE encode->decode: input {dummy.shape} -> latent {lat.shape} -> output {rec.shape}")

            # Quick duration embedder check
            with torch.no_grad():
                d = pipeline.duration_embedder(torch.tensor([10.0]))
            print(f"  [DONE] DurationEmbedder output shape: {d.shape}")

        except Exception as exc:
            print(f"  WARNING: sanity check failed: {exc}")

    print(f"\n[DONE] Conversion complete. Output at: {output_dir}")
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
