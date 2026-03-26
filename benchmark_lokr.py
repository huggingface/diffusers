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


def weight_space_analysis(lokr_path, lokr_name):
    """Compare tier 1 (lossless) vs tier 2 (Kronecker split) in weight space.

    For each fused QKV module, materializes the exact delta from the fuse-first path
    and the reconstructed delta from the Kronecker split path, then reports the
    relative Frobenius norm error.
    """
    from diffusers.loaders.lora_conversion_utils import _convert_non_diffusers_flux2_lokr_to_diffusers

    raw_sd = load_raw_state_dict(lokr_path, lokr_name)

    is_bfl = any(k.startswith("diffusion_model.") for k in raw_sd)
    if not is_bfl:
        print("  Checkpoint is not BFL format - no fused QKV to compare.")
        print("  Tiers 1 and 2 produce identical results for this format.")
        return

    # Convert both ways from the same raw state dict
    sd_fused = _convert_non_diffusers_flux2_lokr_to_diffusers(dict(raw_sd), fuse_qkv=True)
    sd_split = _convert_non_diffusers_flux2_lokr_to_diffusers(dict(raw_sd), fuse_qkv=False)

    # Find all fused QKV modules (to_qkv and to_added_qkv)
    qkv_modules = {}
    for key in sd_fused:
        if ".to_qkv.lokr_w1" in key or ".to_added_qkv.lokr_w1" in key:
            module_path = key.rsplit(".lokr_w1", 1)[0]
            qkv_modules[module_path] = key

    print(f"\n  Found {len(qkv_modules)} fused QKV modules to compare\n")
    print(f"  {'Module':<65} {'Rel Error':>12} {'Abs Error':>12} {'Orig Norm':>12}")
    print(f"  {'-' * 65} {'-' * 12} {'-' * 12} {'-' * 12}")

    errors = []
    for module_path in sorted(qkv_modules.keys()):
        # Materialize exact delta from fused path
        w1_f = sd_fused[f"{module_path}.lokr_w1"].float()
        w2_f = sd_fused[f"{module_path}.lokr_w2"].float()
        delta_exact = torch.kron(w1_f, w2_f)

        # Determine split target keys
        if ".to_qkv" in module_path:
            base = module_path.replace(".attn.to_qkv", "")
            proj_keys = [f"{base}.attn.to_q", f"{base}.attn.to_k", f"{base}.attn.to_v"]
        else:
            base = module_path.replace(".attn.to_added_qkv", "")
            proj_keys = [f"{base}.attn.add_q_proj", f"{base}.attn.add_k_proj", f"{base}.attn.add_v_proj"]

        # Materialize reconstructed delta from split path
        chunks = []
        for proj in proj_keys:
            w1_key = f"{proj}.lokr_w1"
            w2_key = f"{proj}.lokr_w2"
            if w1_key not in sd_split:
                break
            w1_s = sd_split[w1_key].float()
            w2_s = sd_split[w2_key].float()
            chunks.append(torch.kron(w1_s, w2_s))

        if len(chunks) != 3:
            print(f"  {module_path:<65} {'SKIP':>12}")
            continue

        delta_recon = torch.cat(chunks, dim=0)

        orig_norm = delta_exact.norm().item()
        abs_err = (delta_exact - delta_recon).norm().item()
        rel_err = abs_err / orig_norm if orig_norm > 0 else 0.0

        errors.append(rel_err)

        short_name = module_path.replace("transformer.", "")
        print(f"  {short_name:<65} {rel_err:>11.6f}% {abs_err:>12.6f} {orig_norm:>12.4f}")

    if errors:
        print(f"\n  Aggregate over {len(errors)} QKV modules:")
        print(f"    Mean relative error: {sum(errors) / len(errors):.6f}%")
        print(f"    Max  relative error: {max(errors):.6f}%")
        print(f"    Min  relative error: {min(errors):.6f}%")


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
    lora_config, lora_sd = convert_to_lora(pipe.transformer, rank, adapter_name=adapter_name, progressbar=True)
    convert_time = time.time() - t0
    print(f"  Loaded LoKR in {load_time:.1f}s, converted to LoRA in {convert_time:.1f}s")

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

    # Weight-space analysis (no model needed)
    if args.weight_space:
        print("\n=== Weight-space error: Tier 1 (lossless) vs Tier 2 (Kronecker split) ===")
        weight_space_analysis(args.lokr_path, args.lokr_name)
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
