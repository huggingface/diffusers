"""Benchmark: Lossless LoKR vs Lossy LoRA-via-SVD on Flux2 Klein 9B.

Generates images using both conversion paths for visual comparison.
Uses bf16 with CPU offload.

Usage:
    python benchmark_lokr.py
    python benchmark_lokr.py --lokr-path "puttmorbidly233/lora" --lokr-name "klein_snofs_v1_2.safetensors"
    python benchmark_lokr.py --prompt "a portrait in besch art style" --ranks 32 64 128
"""

import argparse
import gc
import os
import time

import torch
from diffusers import Flux2KleinPipeline
from peft import convert_to_lora

MODEL_ID = "black-forest-labs/FLUX.2-klein-9B"
DEFAULT_LOKR_PATH = "gattaplayer/besch-flux2-klein-9b-lokr-lion-3e-6-bs2-ga2-v02"
OUTPUT_DIR = "benchmark_output"


def load_pipeline():
    """Load Flux2 Klein 9B in bf16 with model CPU offload."""
    pipe = Flux2KleinPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
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


def benchmark_lossless(pipe, prompt, seed, lokr_path, lokr_name):
    """Path A: Load LoKR natively (lossless)."""
    print("\n=== Path A: Lossless LoKR ===")
    t0 = time.time()
    kwargs = {"weight_name": lokr_name} if lokr_name else {}
    pipe.load_lora_weights(lokr_path, **kwargs)
    print(f"  Loaded in {time.time() - t0:.1f}s")

    t0 = time.time()
    image = generate(pipe, prompt, seed)
    print(f"  Generated in {time.time() - t0:.1f}s")

    pipe.unload_lora_weights()
    return image


def benchmark_lossy(pipe, prompt, seed, rank, lokr_path, lokr_name):
    """Path B: Load LoKR, convert to LoRA via SVD (lossy)."""
    print(f"\n=== Path B: Lossy LoRA via SVD (rank={rank}) ===")
    t0 = time.time()
    kwargs = {"weight_name": lokr_name} if lokr_name else {}
    pipe.load_lora_weights(lokr_path, **kwargs)
    load_time = time.time() - t0

    # Detect the actual adapter name assigned by peft
    adapter_name = next(iter(pipe.transformer.peft_config.keys()))
    print(f"  Adapter name: {adapter_name}")

    t0 = time.time()
    lora_config, lora_sd = convert_to_lora(pipe.transformer, rank, adapter_name=adapter_name, progressbar=True)
    convert_time = time.time() - t0
    print(f"  Loaded LoKR in {load_time:.1f}s, converted to LoRA in {convert_time:.1f}s")

    # Replace LoKR adapter with converted LoRA
    from peft import inject_adapter_in_model, set_peft_model_state_dict

    pipe.transformer.delete_adapters(adapter_name)
    inject_adapter_in_model(lora_config, pipe.transformer, adapter_name=adapter_name)
    set_peft_model_state_dict(pipe.transformer, lora_sd, adapter_name=adapter_name)

    t0 = time.time()
    image = generate(pipe, prompt, seed)
    print(f"  Generated in {time.time() - t0:.1f}s")

    pipe.unload_lora_weights()
    return image


def benchmark_baseline(pipe, prompt, seed):
    """Baseline: No adapter."""
    print("\n=== Baseline: No adapter ===")
    t0 = time.time()
    image = generate(pipe, prompt, seed)
    print(f"  Generated in {time.time() - t0:.1f}s")
    return image


def main():
    parser = argparse.ArgumentParser(description="Benchmark LoKR vs LoRA-via-SVD")
    parser.add_argument("--prompt", default="a portrait painting in besch art style")
    parser.add_argument("--lokr-path", default=DEFAULT_LOKR_PATH, help="HF repo or local path to LoKR checkpoint")
    parser.add_argument("--lokr-name", default=None, help="Filename within HF repo (if multi-file)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ranks", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--skip-lossy", action="store_true")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Model: {MODEL_ID}")
    print(f"LoKR:  {args.lokr_path}" + (f" ({args.lokr_name})" if args.lokr_name else ""))
    print(f"Prompt: {args.prompt}")
    print(f"Seed: {args.seed}")
    if not args.skip_lossy:
        print(f"SVD ranks to test: {args.ranks}")

    print("\nLoading pipeline (bf16, model CPU offload)...")
    pipe = load_pipeline()

    # Baseline
    if not args.skip_baseline:
        img = benchmark_baseline(pipe, args.prompt, args.seed)
        path = os.path.join(OUTPUT_DIR, "baseline.png")
        img.save(path)
        print(f"  Saved: {path}")

    # Path A: Lossless LoKR
    img = benchmark_lossless(pipe, args.prompt, args.seed, args.lokr_path, args.lokr_name)
    path = os.path.join(OUTPUT_DIR, "lokr_lossless.png")
    img.save(path)
    print(f"  Saved: {path}")

    gc.collect()
    torch.cuda.empty_cache()

    # Path B: Lossy LoRA via SVD at various ranks
    if not args.skip_lossy:
        for rank in args.ranks:
            img = benchmark_lossy(pipe, args.prompt, args.seed, rank, args.lokr_path, args.lokr_name)
            path = os.path.join(OUTPUT_DIR, f"lora_svd_rank{rank}.png")
            img.save(path)
            print(f"  Saved: {path}")

            gc.collect()
            torch.cuda.empty_cache()

    print(f"\nAll results saved to {OUTPUT_DIR}/")
    print("Compare: baseline.png vs lokr_lossless.png vs lora_svd_rank*.png")


if __name__ == "__main__":
    main()
