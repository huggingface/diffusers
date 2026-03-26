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
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Model: {MODEL_ID}")
    print(f"LoKR:  {args.lokr_path}" + (f" ({args.lokr_name})" if args.lokr_name else ""))
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
