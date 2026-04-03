"""
Micro-benchmark: modulate_index tensor creation before vs after caching fix.

This script demonstrates the overhead of recreating the modulate_index tensor
from a Python list comprehension on every forward pass (old behaviour) vs
returning a cached tensor (new behaviour).

Run on any machine — no GPU or model weights required:
    python examples/profiling/benchmark_modulate_index.py

For GPU results, the improvement is larger because torch.tensor() on GPU
additionally triggers cudaMemcpyAsync + cudaStreamSynchronize.
"""

import time
from math import prod

import torch


# ── Helpers ──────────────────────────────────────────────────────────────────

def build_modulate_index(img_shapes, device):
    """Original implementation: rebuilt every forward pass."""
    return torch.tensor(
        [[0] * prod(sample[0]) + [1] * sum([prod(s) for s in sample[1:]]) for sample in img_shapes],
        device=device,
        dtype=torch.int,
    )


def build_modulate_index_cached(img_shapes, cache, device):
    """Fixed implementation: built once, then looked up from cache."""
    cache_key = (tuple(tuple(s) for s in img_shapes), device)
    if cache_key not in cache:
        cache[cache_key] = torch.tensor(
            [[0] * prod(sample[0]) + [1] * sum([prod(s) for s in sample[1:]]) for sample in img_shapes],
            device=device,
            dtype=torch.int,
        )
    return cache[cache_key]


def timeit(fn, n=1000):
    # Warmup
    for _ in range(10):
        fn()
    torch.cuda.synchronize() if torch.cuda.is_available() else None

    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed = (time.perf_counter() - t0) / n * 1e6  # µs per call
    return elapsed


# ── Benchmark ────────────────────────────────────────────────────────────────

def run(device_str: str, num_inference_steps: int = 20, n_trials: int = 1000):
    device = torch.device(device_str)

    # Realistic img_shapes for QwenImage at 1024x1024 with zero_cond_t=True.
    # sample[0] = primary layer patches, sample[1:] = condition patches.
    # patch_size=2 → 1024//2 = 512 tokens per side → 512*512 = 262144 tokens
    # (simplified for demo — actual numbers depend on model config)
    patch_h, patch_w = 64, 64  # 128x128 latent / 2 patch size
    img_shapes = [
        [(1, patch_h, patch_w), (1, patch_h // 2, patch_w // 2)],  # batch item 1
    ]

    cache: dict = {}

    # --- Pre-cache (first call, same cost as uncached) ---
    _ = build_modulate_index_cached(img_shapes, cache, device)

    # --- Benchmark ---
    uncached_us = timeit(lambda: build_modulate_index(img_shapes, device), n=n_trials)
    cached_us   = timeit(lambda: build_modulate_index_cached(img_shapes, cache, device), n=n_trials)

    speedup = uncached_us / cached_us
    total_uncached_ms = uncached_us * num_inference_steps / 1e3
    total_cached_ms   = cached_us   * num_inference_steps / 1e3
    # First call is shared — only steps 2..N benefit
    saved_ms = uncached_us * (num_inference_steps - 1) / 1e3

    print(f"\n{'='*60}")
    print(f"  Device            : {device_str}")
    print(f"  img_shapes        : {img_shapes}")
    print(f"  num_inference_steps: {num_inference_steps}")
    print(f"{'='*60}")
    print(f"  Per-call (uncached): {uncached_us:.2f} µs")
    print(f"  Per-call (cached)  : {cached_us:.2f} µs")
    print(f"  Speedup per call   : {speedup:.1f}x")
    print(f"{'─'*60}")
    print(f"  Total over {num_inference_steps} steps (uncached): {total_uncached_ms:.3f} ms")
    print(f"  Total over {num_inference_steps} steps (cached)  : {total_cached_ms:.3f} ms")
    print(f"  CPU overhead saved : {saved_ms:.3f} ms")
    print(f"{'='*60}")

    # Verify outputs are identical
    out_uncached = build_modulate_index(img_shapes, device)
    out_cached   = build_modulate_index_cached(img_shapes, cache, device)
    assert torch.equal(out_uncached, out_cached), "BUG: cached and uncached tensors differ!"
    print("  ✅  Output tensors are identical (correctness verified)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    run("cpu", num_inference_steps=20)
    if torch.cuda.is_available():
        run("cuda", num_inference_steps=20)
    else:
        print("(No CUDA device found — run on a GPU machine for full DtoH sync numbers)\n")
