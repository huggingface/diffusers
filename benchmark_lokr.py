"""Benchmark: Three-tier LoKR quality comparison on Flux2 Klein 9B.

Tier 1 - Fuse-first (lossless): Fuse model QKV, map BFL LoKR directly. Exact.
Tier 2 - Kronecker split (default): Split fused QKV via Van Loan re-factorization. Slight loss.
Tier 3 - SVD to LoRA (fully lossy): Convert entire LoKR to LoRA via peft.convert_to_lora.

Tiers 1+2 only apply to BFL-format LoKR (fused QKV). LyCORIS and diffusers-native
formats already have separate Q/K/V and only run the default path.

Uses bf16 with CPU offload.

Usage:
    python benchmark_lokr.py
    python benchmark_lokr.py --lokr-path "puttmorbidly233/lora" --lokr-name "klein_snofs_v1_2.safetensors"
    python benchmark_lokr.py --prompt "a portrait in besch art style" --ranks 32 64 128
    python benchmark_lokr.py --tiers 1 2     # skip SVD tier
    python benchmark_lokr.py --tiers 2 3     # skip fuse-first tier
    python benchmark_lokr.py --weight-space  # weight-space error analysis only (no image generation)
"""

import argparse
import gc
import os
import time

import torch

from diffusers import Flux2KleinPipeline


MODEL_ID = "black-forest-labs/FLUX.2-klein-9B"
DEFAULT_LOKR_PATH = "gattaplayer/besch-flux2-klein-9b-lokr-lion-3e-6-bs2-ga2-v02"
OUTPUT_DIR = "benchmark_output"


def load_pipeline(no_offload=False):
    """Load Flux2 Klein 9B in bf16."""
    pipe = Flux2KleinPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
    if no_offload:
        pipe = pipe.to("cuda")
    else:
        pipe.enable_model_cpu_offload()
    return pipe


def generate(pipe, prompt, seed, num_steps=4, guidance_scale=1.0):
    """Generate a single image with fixed seed for reproducibility."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    image = pipe(
        prompt=prompt,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        generator=generator,
        height=1024,
        width=1024,
    ).images[0]
    return image


# ---------------------------------------------------------------------------
# Weight-space error analysis
# ---------------------------------------------------------------------------


def load_raw_state_dict(lokr_path, lokr_name):
    """Download/load a LoKR checkpoint and return the raw state dict."""
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    if os.path.isfile(lokr_path):
        return load_file(lokr_path)

    if os.path.isdir(lokr_path):
        path = os.path.join(lokr_path, lokr_name) if lokr_name else lokr_path
        return load_file(path)

    # HF repo
    path = hf_hub_download(lokr_path, filename=lokr_name or "pytorch_lora_weights.safetensors")
    return load_file(path)


def _materialize_lokr_delta(state_dict, module_path):
    """Materialize the full delta weight from LoKR factors for a single module."""
    w1_key = f"{module_path}.lokr_w1"
    w2_key = f"{module_path}.lokr_w2"
    w1a_key = f"{module_path}.lokr_w1_a"
    w1b_key = f"{module_path}.lokr_w1_b"
    w2a_key = f"{module_path}.lokr_w2_a"
    w2b_key = f"{module_path}.lokr_w2_b"

    # w1: full or decomposed
    if w1_key in state_dict:
        w1 = state_dict[w1_key].float()
    elif w1a_key in state_dict and w1b_key in state_dict:
        w1 = state_dict[w1a_key].float() @ state_dict[w1b_key].float()
    else:
        return None

    # w2: full or decomposed
    if w2_key in state_dict:
        w2 = state_dict[w2_key].float()
    elif w2a_key in state_dict and w2b_key in state_dict:
        w2 = state_dict[w2a_key].float() @ state_dict[w2b_key].float()
    else:
        return None

    return torch.kron(w1, w2)


def _print_error_table(title, results):
    """Print a formatted error table and aggregate stats."""
    print(f"\n  {title}\n")
    print(f"  {'Module':<60} {'Rel Error %':>12} {'Abs Error':>12} {'Orig Norm':>12}")
    print(f"  {'-' * 60} {'-' * 12} {'-' * 12} {'-' * 12}")

    errors = []
    for name, rel_err, abs_err, orig_norm in results:
        errors.append(rel_err)
        print(f"  {name:<60} {rel_err:>11.6f}% {abs_err:>12.6f} {orig_norm:>12.4f}")

    if errors:
        print(f"\n  Aggregate over {len(errors)} modules:")
        print(f"    Mean relative error: {sum(errors) / len(errors):.6f}%")
        print(f"    Max  relative error: {max(errors):.6f}%")
        print(f"    Min  relative error: {min(errors):.6f}%")


def weight_space_kronecker(lokr_path, lokr_name):
    """Compare tier 1 (lossless) vs tier 2 (Kronecker split) in weight space.

    No model loading needed - operates on checkpoint state dicts only.
    Only meaningful for BFL-format LoKR (fused QKV).
    """
    from diffusers.loaders.lora_conversion_utils import _convert_non_diffusers_flux2_lokr_to_diffusers

    raw_sd = load_raw_state_dict(lokr_path, lokr_name)

    is_bfl = any(k.startswith("diffusion_model.") for k in raw_sd)
    if not is_bfl:
        print("  Checkpoint is not BFL format - no fused QKV to compare.")
        print("  Tiers 1 and 2 produce identical results for this format.")
        return

    sd_fused = _convert_non_diffusers_flux2_lokr_to_diffusers(dict(raw_sd), fuse_qkv=True)
    sd_split = _convert_non_diffusers_flux2_lokr_to_diffusers(dict(raw_sd), fuse_qkv=False)

    # Find all fused QKV modules
    qkv_modules = []
    for key in sd_fused:
        if ".to_qkv.lokr_w1" in key or ".to_added_qkv.lokr_w1" in key:
            qkv_modules.append(key.rsplit(".lokr_w1", 1)[0])

    print(f"\n  Found {len(qkv_modules)} fused QKV modules to compare")

    results = []
    for module_path in sorted(qkv_modules):
        delta_exact = _materialize_lokr_delta(sd_fused, module_path)
        if delta_exact is None:
            continue

        # Determine split target keys
        if ".to_qkv" in module_path:
            base = module_path.replace(".attn.to_qkv", "")
            proj_keys = [f"{base}.attn.to_q", f"{base}.attn.to_k", f"{base}.attn.to_v"]
        else:
            base = module_path.replace(".attn.to_added_qkv", "")
            proj_keys = [f"{base}.attn.add_q_proj", f"{base}.attn.add_k_proj", f"{base}.attn.add_v_proj"]

        chunks = []
        for proj in proj_keys:
            delta = _materialize_lokr_delta(sd_split, proj)
            if delta is None:
                break
            chunks.append(delta)

        if len(chunks) != 3:
            continue

        delta_recon = torch.cat(chunks, dim=0)
        orig_norm = delta_exact.norm().item()
        abs_err = (delta_exact - delta_recon).norm().item()
        rel_err = abs_err / orig_norm if orig_norm > 0 else 0.0

        short_name = module_path.replace("transformer.", "")
        results.append((short_name, rel_err, abs_err, orig_norm))

    _print_error_table("Tier 1 (lossless) vs Tier 2 (Kronecker split) - QKV modules only", results)


def weight_space_svd(lokr_path, lokr_name, ranks, no_offload=False):
    """Compare tier 1 (lossless) vs tier 3 (SVD to LoRA) in weight space.

    Requires loading the full model to run peft.convert_to_lora.
    Compares materialized LoKR deltas against LoRA deltas for ALL modules.
    """
    from peft import convert_to_lora

    # Build reference deltas from the converted state dict (tier 2 / default path)
    # For non-QKV modules tier 2 is identical to tier 1, so this is ground truth.
    raw_sd = load_raw_state_dict(lokr_path, lokr_name)
    is_bfl = any(k.startswith("diffusion_model.") for k in raw_sd)

    if is_bfl:
        from diffusers.loaders.lora_conversion_utils import _convert_non_diffusers_flux2_lokr_to_diffusers

        sd_ref = _convert_non_diffusers_flux2_lokr_to_diffusers(dict(raw_sd), fuse_qkv=False)
    else:
        # For non-BFL, just use the default conversion as reference (already lossless)
        from diffusers.loaders.lora_conversion_utils import (
            _convert_diffusers_flux2_lokr_to_peft,
            _convert_lycoris_flux2_lokr_to_diffusers,
        )

        if any(k.startswith("lycoris_") for k in raw_sd):
            sd_ref = _convert_lycoris_flux2_lokr_to_diffusers(dict(raw_sd))
        else:
            sd_ref = _convert_diffusers_flux2_lokr_to_peft(dict(raw_sd))

    # Find all LoKR modules and materialize their deltas
    ref_deltas = {}
    lokr_modules = set()
    for key in sd_ref:
        if ".lokr_w1" in key and ".lokr_w1_" not in key:
            module_path = key.rsplit(".lokr_w1", 1)[0]
            lokr_modules.add(module_path)
        elif ".lokr_w1_a" in key:
            module_path = key.rsplit(".lokr_w1_a", 1)[0]
            lokr_modules.add(module_path)

    for module_path in lokr_modules:
        delta = _materialize_lokr_delta(sd_ref, module_path)
        if delta is not None:
            ref_deltas[module_path] = delta

    print(f"\n  Materialized {len(ref_deltas)} reference LoKR deltas")

    # Load model and LoKR adapter
    print("\n  Loading model for SVD conversion...")
    pipe = load_pipeline(no_offload=no_offload)
    kwargs = {"weight_name": lokr_name} if lokr_name else {}
    pipe.load_lora_weights(lokr_path, **kwargs)
    adapter_name = next(iter(pipe.transformer.peft_config.keys()))

    for rank in ranks:
        print(f"\n  Converting to LoRA rank={rank}...")
        t0 = time.time()
        lora_config, lora_sd = convert_to_lora(pipe.transformer, rank, adapter_name=adapter_name, progressbar=True)
        print(f"  Converted in {time.time() - t0:.1f}s")
        print(f"  LoRA config: alpha={lora_config.lora_alpha}, r={lora_config.r}")

        # Also print the LoKR config for reference
        lokr_cfg = pipe.transformer.peft_config.get(adapter_name)
        if lokr_cfg:
            alpha = getattr(lokr_cfg, "alpha", getattr(lokr_cfg, "lora_alpha", "?"))
            print(f"  Adapter config: {type(lokr_cfg).__name__}, alpha={alpha}, r={lokr_cfg.r}")

        # Compare each module: LoKR delta vs LoRA delta (lora_B @ lora_A)
        results = []
        for module_path in sorted(ref_deltas.keys()):
            delta_ref = ref_deltas[module_path]

            # Map module_path to LoRA key format: transformer.X.Y -> base_model.model.X.Y
            lora_module = module_path.replace("transformer.", "")
            lora_a_key = f"base_model.model.{lora_module}.lora_A.weight"
            lora_b_key = f"base_model.model.{lora_module}.lora_B.weight"

            if lora_a_key not in lora_sd or lora_b_key not in lora_sd:
                # Try without base_model.model prefix
                lora_a_key = f"{lora_module}.lora_A.weight"
                lora_b_key = f"{lora_module}.lora_B.weight"

            if lora_a_key not in lora_sd or lora_b_key not in lora_sd:
                continue

            lora_a = lora_sd[lora_a_key].float().cpu()
            lora_b = lora_sd[lora_b_key].float().cpu()
            delta_lora = lora_b @ lora_a

            orig_norm = delta_ref.norm().item()
            abs_err = (delta_ref.cpu() - delta_lora).norm().item()
            rel_err = abs_err / orig_norm if orig_norm > 0 else 0.0

            short_name = module_path.replace("transformer.", "")
            results.append((short_name, rel_err, abs_err, orig_norm))

        _print_error_table(f"Tier 1 (lossless) vs Tier 3 (SVD rank={rank}) - all modules", results)

    pipe.unload_lora_weights()
    del pipe
    gc.collect()
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Image generation benchmarks
# ---------------------------------------------------------------------------


def benchmark_baseline(pipe, prompt, seed):
    """Baseline: No adapter."""
    print("\n=== Baseline: No adapter ===")
    t0 = time.time()
    image = generate(pipe, prompt, seed)
    print(f"  Generated in {time.time() - t0:.1f}s")
    return image


def benchmark_tier1_fuse_first(pipe, prompt, seed, lokr_path, lokr_name):
    """Tier 1: Fuse model QKV, then load BFL LoKR directly (lossless)."""
    print("\n=== Tier 1: Fuse-first LoKR (lossless) ===")
    t0 = time.time()
    kwargs = {"weight_name": lokr_name} if lokr_name else {}
    pipe.load_lora_weights(lokr_path, fuse_qkv=True, **kwargs)
    print(f"  Loaded in {time.time() - t0:.1f}s")

    t0 = time.time()
    image = generate(pipe, prompt, seed)
    print(f"  Generated in {time.time() - t0:.1f}s")

    pipe.unload_lora_weights()
    return image


def benchmark_tier2_kronecker_split(pipe, prompt, seed, lokr_path, lokr_name):
    """Tier 2: Split fused QKV via Kronecker re-factorization (default path)."""
    print("\n=== Tier 2: Kronecker split LoKR (default) ===")
    t0 = time.time()
    kwargs = {"weight_name": lokr_name} if lokr_name else {}
    pipe.load_lora_weights(lokr_path, **kwargs)
    print(f"  Loaded in {time.time() - t0:.1f}s")

    t0 = time.time()
    image = generate(pipe, prompt, seed)
    print(f"  Generated in {time.time() - t0:.1f}s")

    pipe.unload_lora_weights()
    return image


def benchmark_tier3_svd(pipe, prompt, seed, rank, lokr_path, lokr_name):
    """Tier 3: Convert LoKR to LoRA via SVD (fully lossy)."""
    from peft import convert_to_lora, inject_adapter_in_model, set_peft_model_state_dict

    print(f"\n=== Tier 3: SVD to LoRA (rank={rank}) ===")
    t0 = time.time()
    kwargs = {"weight_name": lokr_name} if lokr_name else {}
    pipe.load_lora_weights(lokr_path, **kwargs)
    load_time = time.time() - t0

    adapter_name = next(iter(pipe.transformer.peft_config.keys()))
    print(f"  Adapter name: {adapter_name}")

    t0 = time.time()
    lokr_cfg = pipe.transformer.peft_config.get(adapter_name)
    if lokr_cfg:
        alpha = getattr(lokr_cfg, "alpha", getattr(lokr_cfg, "lora_alpha", "?"))
        print(f"  Adapter config: {type(lokr_cfg).__name__}, alpha={alpha}, r={lokr_cfg.r}")

    t0 = time.time()
    lora_config, lora_sd = convert_to_lora(pipe.transformer, rank, adapter_name=adapter_name, progressbar=True)
    convert_time = time.time() - t0
    print(f"  Loaded LoKR in {load_time:.1f}s, converted to LoRA in {convert_time:.1f}s")
    print(f"  LoRA config: alpha={lora_config.lora_alpha}, r={lora_config.r}")

    pipe.transformer.delete_adapters(adapter_name)
    inject_adapter_in_model(lora_config, pipe.transformer, adapter_name=adapter_name)
    set_peft_model_state_dict(pipe.transformer, lora_sd, adapter_name=adapter_name)

    t0 = time.time()
    image = generate(pipe, prompt, seed)
    print(f"  Generated in {time.time() - t0:.1f}s")

    pipe.unload_lora_weights()
    return image


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Benchmark LoKR quality tiers")
    parser.add_argument("--prompt", default="a portrait painting in besch art style")
    parser.add_argument("--lokr-path", default=DEFAULT_LOKR_PATH, help="HF repo or local path to LoKR checkpoint")
    parser.add_argument("--lokr-name", default=None, help="Filename within HF repo (if multi-file)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--tiers", type=int, nargs="+", default=[1, 2, 3], help="Tiers to run (1=fuse, 2=kronecker, 3=svd)"
    )
    parser.add_argument("--ranks", type=int, nargs="+", default=[32, 64, 128], help="SVD ranks for tier 3")
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--no-offload", action="store_true", help="Keep model on GPU instead of CPU offload")
    parser.add_argument("--weight-space", action="store_true", help="Run weight-space error analysis only (no images)")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Model: {MODEL_ID}")
    print(f"LoKR:  {args.lokr_path}" + (f" ({args.lokr_name})" if args.lokr_name else ""))

    # Weight-space analysis
    if args.weight_space:
        print("\n=== Weight-space error: Tier 1 (lossless) vs Tier 2 (Kronecker split) ===")
        weight_space_kronecker(args.lokr_path, args.lokr_name)

        if args.ranks:
            print("\n=== Weight-space error: Tier 1 (lossless) vs Tier 3 (SVD to LoRA) ===")
            weight_space_svd(args.lokr_path, args.lokr_name, args.ranks, no_offload=args.no_offload)

        return

    print(f"Prompt: {args.prompt}")
    print(f"Seed: {args.seed}")
    print(f"Tiers: {args.tiers}")
    if 3 in args.tiers:
        print(f"SVD ranks: {args.ranks}")

    mode = "on GPU" if args.no_offload else "with CPU offload"
    print(f"\nLoading pipeline (bf16, {mode})...")
    pipe = load_pipeline(no_offload=args.no_offload)

    # Baseline
    if not args.skip_baseline:
        img = benchmark_baseline(pipe, args.prompt, args.seed)
        path = os.path.join(OUTPUT_DIR, "baseline.png")
        img.save(path)
        print(f"  Saved: {path}")

    # Tier 1: Fuse-first (lossless, BFL format only - identical to tier 2 for other formats)
    if 1 in args.tiers:
        print("\n  Note: Tier 1 only differs from tier 2 for BFL-format LoKR (fused QKV).")
        img = benchmark_tier1_fuse_first(pipe, args.prompt, args.seed, args.lokr_path, args.lokr_name)
        path = os.path.join(OUTPUT_DIR, "tier1_fuse_lossless.png")
        img.save(path)
        print(f"  Saved: {path}")
        gc.collect()
        torch.cuda.empty_cache()

    # Tier 2: Kronecker split (default)
    if 2 in args.tiers:
        img = benchmark_tier2_kronecker_split(pipe, args.prompt, args.seed, args.lokr_path, args.lokr_name)
        path = os.path.join(OUTPUT_DIR, "tier2_kronecker.png")
        img.save(path)
        print(f"  Saved: {path}")
        gc.collect()
        torch.cuda.empty_cache()

    # Tier 3: SVD to LoRA at various ranks
    if 3 in args.tiers:
        for rank in args.ranks:
            img = benchmark_tier3_svd(pipe, args.prompt, args.seed, rank, args.lokr_path, args.lokr_name)
            path = os.path.join(OUTPUT_DIR, f"tier3_svd_rank{rank}.png")
            img.save(path)
            print(f"  Saved: {path}")
            gc.collect()
            torch.cuda.empty_cache()

    print(f"\nAll results saved to {OUTPUT_DIR}/")
    print("Compare: baseline.png vs tier1_fuse_lossless.png vs tier2_kronecker.png vs tier3_svd_rank*.png")


if __name__ == "__main__":
    main()
