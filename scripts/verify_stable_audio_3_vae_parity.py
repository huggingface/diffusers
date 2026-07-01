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
Forward-parity check for the Stable Audio 3 SAME autoencoder.

Loads the *same* trained weights into the reference ``AudioAutoencoder``
(``stable_audio_3``) and the diffusers ``AutoencoderSAME`` (via the conversion
script), feeds both identical audio, and compares ``encode`` and ``decode``
numerically. Together with ``verify_stable_audio_3_dit_parity.py`` this gives
deterministic, bit-comparable parity for every learned component of the model
(the full pipeline cannot be compared waveform-for-waveform because the
ping-pong sampler's RNG streams differ across codebases).

The check is staged so a mismatch is localized:
  1. encode parity  — isolates patch + encoder + bottleneck
  2. decode parity  — feeds the *reference* latents to both decoders (isolates the decoder)
  3. round-trip     — full encode→decode reconstruction

The ungated base checkpoint ``stabilityai/stable-audio-3-medium-base`` is used by
default (same SAME-L autoencoder as the gated ``-medium``).

Setup:
  - The reference package must be importable, e.g. ``pip install -e /path/to/stable-audio-3``
    or pass its location via ``--reference_repo``.

Usage:
    python scripts/verify_stable_audio_3_vae_parity.py \\
        --checkpoint_path stabilityai/stable-audio-3-medium-base \\
        [--reference_repo /Users/.../stable-audio-3] \\
        [--atol 1e-4] [--num_latent_frames 16]

Exit code is non-zero if parity fails, so it can gate CI.
"""

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path


# The reference autoencoder's attention uses `flex_attention`, which is lowered via `torch.compile`.
# On CPU that lowering is unsupported and floods stderr with a (non-fatal) InductorError before torch
# falls back to eager execution. Disabling dynamo forces the eager path directly — the numerics and the
# parity result are identical either way. Set before importing torch so it takes effect.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import torch  # noqa: E402


def _load_convert_module(repo_root: Path):
    """Import the conversion script as a module (it has no package import path)."""
    path = repo_root / "scripts" / "convert_stable_audio_3_to_diffusers.py"
    spec = importlib.util.spec_from_file_location("sa3_convert", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _resolve_checkpoint(checkpoint_path: str) -> tuple:
    """Return (safetensors_path, model_config_path), downloading from the Hub if needed."""
    p = Path(checkpoint_path)
    if p.is_dir():
        return p / "model.safetensors", p / "model_config.json"
    if p.is_file():
        return p, p.parent / "model_config.json"

    from huggingface_hub import hf_hub_download

    print(f"Downloading checkpoint + config from HF Hub: {checkpoint_path}")
    sd_path = hf_hub_download(repo_id=checkpoint_path, filename="model.safetensors")
    cfg_path = hf_hub_download(repo_id=checkpoint_path, filename="model_config.json")
    return Path(sd_path), Path(cfg_path)


def _diff(a: torch.Tensor, b: torch.Tensor) -> tuple:
    """Return (max_abs, max_relative) between two tensors."""
    max_abs = (a.float() - b.float()).abs().max().item()
    rel = max_abs / a.float().abs().max().clamp(min=1e-8).item()
    return max_abs, rel


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    dtype = getattr(torch, args.dtype)

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    if args.reference_repo:
        sys.path.insert(0, args.reference_repo)

    from safetensors.torch import load_file

    from diffusers import AutoencoderSAME

    try:
        from stable_audio_3.factory import create_autoencoder_from_config
    except ImportError as exc:
        print(
            f"ERROR: could not import the reference package `stable_audio_3` ({exc}).\n"
            "Install it (`pip install -e /path/to/stable-audio-3`) or pass --reference_repo.",
        )
        sys.exit(2)

    convert = _load_convert_module(repo_root)

    # ── Load checkpoint + config ─────────────────────────────────────────────
    sd_path, cfg_path = _resolve_checkpoint(args.checkpoint_path)
    print(f"Loading weights: {sd_path}")
    ref_sd = load_file(str(sd_path), device="cpu")

    with open(cfg_path) as f:
        model_config = json.load(f)
    sample_rate = model_config.get("sample_rate", 44100)
    autoencoder_config = model_config["model"]["pretransform"]["config"]

    # ── Build the reference autoencoder, load weights (strip "pretransform.model.") ──
    ref_ae = create_autoencoder_from_config(autoencoder_config, sample_rate)
    prefix = "pretransform.model."
    ref_ae_sd = {k[len(prefix) :]: v for k, v in ref_sd.items() if k.startswith(prefix)}
    missing, unexpected = ref_ae.load_state_dict(ref_ae_sd, strict=False)
    print(f"Reference AE: loaded {len(ref_ae_sd)} tensors ({len(missing)} missing, {len(unexpected)} unexpected)")
    # The reference TRB adds `randn * mask_noise` to its learnable output token on *every* forward — even in eval.
    # The reference SoftNormBottleneck.decode likewise adds `randn * running_std * 1e-3` when `noise_regularize` is set.
    # Disable both so the comparison is deterministic (the diffusers port intentionally omits this inference-time noise).
    for module in ref_ae.modules():
        if hasattr(module, "mask_noise"):
            module.mask_noise = 0.0
        if hasattr(module, "noise_regularize"):
            module.noise_regularize = False
    ref_ae = ref_ae.to(dtype).eval()

    # ── Build the diffusers autoencoder via the conversion script and load ───
    cfg = convert._infer_vae_config(ref_sd, model_config)
    print(f"Inferred diffusers config: {cfg}")
    diff_sd = convert.convert_vae(ref_sd, differential=cfg["use_differential_attention"])
    diff_ae = AutoencoderSAME(**cfg)
    missing_d, unexpected_d = diff_ae.load_state_dict(diff_sd, strict=False)
    print(f"Diffusers AE: loaded {len(diff_sd)} tensors ({len(missing_d)} missing, {len(unexpected_d)} unexpected)")
    if missing_d:
        print(f"  missing: {missing_d[:8]}")
    if unexpected_d:
        print(f"  unexpected: {unexpected_d[:8]}")
    diff_ae = diff_ae.to(dtype).eval()

    # ── Identical input audio (length = multiple of the downsampling ratio) ──
    dsr = diff_ae.downsampling_ratio
    n_samples = args.num_latent_frames * dsr
    audio = torch.randn(args.batch_size, cfg["audio_channels"], n_samples, dtype=dtype)
    print(f"\nInput audio: {tuple(audio.shape)}  (downsampling_ratio={dsr})")

    results = []
    with torch.no_grad():
        # 1. encode parity — patch + encoder + bottleneck
        ref_lat = ref_ae.encode(audio)
        diff_lat = diff_ae.encode(audio).latents
        if ref_lat.shape != diff_lat.shape:
            print(f"\n✗ encode SHAPE DIFF ref{tuple(ref_lat.shape)} vs diff{tuple(diff_lat.shape)}")
            sys.exit(1)
        enc_abs, enc_rel = _diff(ref_lat, diff_lat)
        results.append(("encode (latents)", enc_abs, enc_rel))

        # 2. decode parity — feed the *reference* latents to both decoders to isolate the decoder
        ref_rec = ref_ae.decode(ref_lat)
        diff_rec = diff_ae.decode(ref_lat).sample
        m = min(ref_rec.shape[-1], diff_rec.shape[-1])
        dec_abs, dec_rel = _diff(ref_rec[..., :m], diff_rec[..., :m])
        results.append(("decode (same latents)", dec_abs, dec_rel))

        # 3. full round-trip reconstruction
        diff_rt = diff_ae.decode(diff_lat).sample
        rt_abs, rt_rel = _diff(ref_rec[..., :m], diff_rt[..., :m])
        results.append(("round-trip recon", rt_abs, rt_rel))

    # ── Report ───────────────────────────────────────────────────────────────
    print("\n── Parity (staged; first non-negligible row localizes a mismatch) ──")
    for name, max_abs, rel in results:
        print(f"  {name:<24} max|Δ| = {max_abs:.3e}   max relative = {rel:.3e}")

    # Encode/decode-of-same-latents must match tightly; round-trip inherits the encode error.
    ok = results[0][1] < args.atol and results[1][1] < args.atol
    print(f"\n{'✓ PARITY PASS' if ok else '✗ PARITY FAIL'} (atol={args.atol} on encode & decode)")
    if not ok:
        print("  A large `encode` diff points at patch/encoder/bottleneck; a large `decode`")
        print("  diff (with encode passing) points at the decoder TRB stack.")
    sys.exit(0 if ok else 1)


def parse_args():
    parser = argparse.ArgumentParser(description="Forward-parity check for the Stable Audio 3 SAME autoencoder.")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="stabilityai/stable-audio-3-medium-base",
        help="Path to model.safetensors / a directory / an HF repo id (ungated base by default).",
    )
    parser.add_argument(
        "--reference_repo",
        type=str,
        default=None,
        help="Path to the stable-audio-3 repo root (prepended to sys.path if `stable_audio_3` is not installed).",
    )
    parser.add_argument("--atol", type=float, default=1e-4, help="Max relative diff allowed for encode/decode.")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_latent_frames", type=int, default=16, help="Audio length in latent frames (×ratio).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--dtype",
        type=str,
        default="float64",
        choices=["float64", "float32", "float16", "bfloat16"],
        help=(
            "Compute dtype. Defaults to float64: the decoder's trailing sinusoidal FFN layers have gain ~pi and "
            "amplify float32 rounding noise well past 1e-4, so float32 is not a meaningful parity threshold there."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
