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
Forward-parity check for the Stable Audio 3 DiT.

Loads the *same* trained weights into the reference ``DiffusionTransformer``
(``stable_audio_3``) and the diffusers ``StableAudio3DiTModel`` (via the conversion
script), feeds both identical inputs, and compares the outputs numerically.

This is the test that turns "structurally correct" (weights load, shapes match)
into "actually correct" (the two implementations compute the same function). It
catches bugs that a structural/shape test cannot: a wrong QKV reorder, an RMSNorm
eps mismatch, an AdaLN chunk-order or gate-formula error, etc.

The ungated base checkpoint ``stabilityai/stable-audio-3-medium-base`` is used by
default (same DiT structure as the gated ``-medium``).

Setup:
  - The reference package must be importable, e.g.:
        pip install -e /path/to/stable-audio-3
    or pass its location via ``--reference_repo`` (prepended to sys.path).

Usage:
    python scripts/verify_stable_audio_3_dit_parity.py \\
        --checkpoint_path stabilityai/stable-audio-3-medium-base \\
        [--reference_repo /Users/.../stable-audio-3] \\
        [--atol 1e-4] [--seq_len 64] [--dtype float32]

Exit code is non-zero if parity fails, so it can gate CI.
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch


# Tap points: (reference module path on DiffusionTransformer, diffusers module path on the model).
# Outputs of these submodules are captured by forward hooks and compared so a mismatch is localized.
_STAGE_TAPS = [
    ("to_timestep_embed", "to_timestep_embed"),
    ("to_cond_embed", "to_cond_embed"),
    ("to_global_embed", "to_global_embed"),
    ("transformer.global_cond_embedder", "global_cond_embedder"),
    ("transformer.layers.0", "transformer_blocks.0"),
    ("transformer.project_out", "proj_out"),
]


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


def _module_by_path(root: torch.nn.Module, dotted: str) -> torch.nn.Module:
    obj = root
    for part in dotted.split("."):
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    return obj


def _register_hooks(model: torch.nn.Module, paths, store: dict, side: str):
    handles = []
    for path in paths:
        try:
            module = _module_by_path(model, path)
        except (AttributeError, IndexError):
            continue

        def _hook(_m, _inp, out, _key=path):
            store[(side, _key)] = out[0] if isinstance(out, tuple) else out

        handles.append(module.register_forward_hook(_hook))
    return handles


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    dtype = getattr(torch, args.dtype)

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    if args.reference_repo:
        sys.path.insert(0, args.reference_repo)

    from safetensors.torch import load_file

    from diffusers import StableAudio3DiTModel

    try:
        from stable_audio_3.models.diffusion import DiTWrapper
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
    diffusion_config = model_config["model"]["diffusion"]
    diffusion_model_config = diffusion_config["config"]
    diffusion_objective = diffusion_config.get("diffusion_objective", "rectified_flow")

    # ── Build the reference DiT and load weights (strip the "model.model." prefix) ──
    ref_wrapper = DiTWrapper(diffusion_objective=diffusion_objective, **diffusion_model_config)
    ref_dit = ref_wrapper.model  # DiffusionTransformer
    prefix = "model.model."
    ref_dit_sd = {k[len(prefix) :]: v for k, v in ref_sd.items() if k.startswith(prefix)}
    missing, unexpected = ref_dit.load_state_dict(ref_dit_sd, strict=False)
    print(f"Reference DiT: loaded {len(ref_dit_sd)} tensors ({len(missing)} missing, {len(unexpected)} unexpected)")
    ref_dit = ref_dit.to(dtype).eval()

    # ── Build the diffusers DiT via the conversion script and load (strict) ──
    cfg = convert._infer_dit_config(ref_sd)
    print(f"Inferred diffusers config: {cfg}")
    diff_sd, _ = convert.convert_dit(ref_sd, differential=cfg["use_differential_attention"])
    diff_dit = StableAudio3DiTModel(**cfg)
    diff_dit.load_state_dict(diff_sd, strict=True)
    diff_dit = diff_dit.to(dtype).eval()

    # ── Identical inputs ─────────────────────────────────────────────────────
    B, T = args.batch_size, args.seq_len
    io_channels = cfg["io_channels"]
    cond_token_dim = cfg["cond_token_dim"]
    global_cond_dim = cfg["global_cond_dim"]

    hidden_states = torch.randn(B, io_channels, T, dtype=dtype)
    timestep = torch.rand(B)  # kept float32, as the reference does
    context = torch.randn(B, args.context_len, cond_token_dim, dtype=dtype)
    global_cond = torch.randn(B, global_cond_dim, dtype=dtype)

    # ── Run both with intermediate taps ──────────────────────────────────────
    acts: dict = {}
    h_ref = _register_hooks(ref_dit, [r for r, _ in _STAGE_TAPS], acts, "ref")
    h_diff = _register_hooks(diff_dit, [d for _, d in _STAGE_TAPS], acts, "diff")

    with torch.no_grad():
        # Reference: call _forward directly (no CFG, no mask — the reference disables the
        # cross-attn mask anyway). to_cond_embed / to_global_embed are applied inside.
        ref_out = ref_dit._forward(
            hidden_states, timestep, cross_attn_cond=context, global_embed=global_cond, local_add_cond=None
        )
        diff_out = diff_dit(
            hidden_states,
            timestep,
            encoder_hidden_states=context,
            global_hidden_states=global_cond,
            return_dict=False,
        )[0]

    for handle in h_ref + h_diff:
        handle.remove()

    # ── Report staged diffs ──────────────────────────────────────────────────
    print("\n── Intermediate activation diffs (localizes any mismatch) ──")
    for ref_path, diff_path in _STAGE_TAPS:
        a, b = acts.get(("ref", ref_path)), acts.get(("diff", diff_path))
        if a is None or b is None:
            print(f"  {diff_path:<28} (tap unavailable)")
            continue
        if a.shape != b.shape:
            print(f"  {diff_path:<28} SHAPE DIFF ref{tuple(a.shape)} vs diff{tuple(b.shape)}")
            continue
        print(f"  {diff_path:<28} max|Δ| = {(a.float() - b.float()).abs().max().item():.3e}")

    # ── Final output diff ────────────────────────────────────────────────────
    max_abs = (ref_out.float() - diff_out.float()).abs().max().item()
    rel = max_abs / ref_out.float().abs().max().clamp(min=1e-8).item()
    print("\n── Final output ──")
    print(f"  ref {tuple(ref_out.shape)}  diff {tuple(diff_out.shape)}")
    print(f"  max|Δ| = {max_abs:.3e}   max relative = {rel:.3e}")

    ok = torch.allclose(ref_out.float(), diff_out.float(), atol=args.atol, rtol=args.rtol)
    print(f"\n{'✓ PARITY PASS' if ok else '✗ PARITY FAIL'} (atol={args.atol}, rtol={args.rtol})")
    if not ok:
        print("  The first non-negligible stage diff above points at the offending component.")
    sys.exit(0 if ok else 1)


def parse_args():
    parser = argparse.ArgumentParser(description="Forward-parity check for the Stable Audio 3 DiT.")
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
    parser.add_argument("--atol", type=float, default=1e-4, help="Absolute tolerance for allclose.")
    parser.add_argument("--rtol", type=float, default=1e-3, help="Relative tolerance for allclose.")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=64, help="Latent length (time frames).")
    parser.add_argument("--context_len", type=int, default=10, help="Cross-attention context length.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"])
    return parser.parse_args()


if __name__ == "__main__":
    main()
