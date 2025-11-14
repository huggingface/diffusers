"""
Performance benchmark for QwenImage attention mask implementation.

This benchmark measures:
1. Latency impact of mask processing
2. Memory overhead
3. Throughput comparison
4. CFG batching performance

Run with: python benchmark_qwen_mask_performance.py
"""

import gc
import time
from typing import Dict

import pandas as pd
import torch
import torch.utils.benchmark as benchmark

from diffusers import QwenImageTransformer2DModel


def flush():
    """Clean up GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()


def get_model():
    """Create a QwenImage model for benchmarking."""
    model = QwenImageTransformer2DModel(
        patch_size=2,
        in_channels=16,
        out_channels=4,
        num_layers=2,
        attention_head_dim=16,
        num_attention_heads=3,
        joint_attention_dim=16,
        guidance_embeds=False,
        axes_dims_rope=(8, 4, 4),  # Match small model dimensions
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32

    model = model.to(device).to(dtype).eval()
    return model, device, dtype


def create_inputs_no_mask(batch_size, device, dtype, height=512, width=512, text_seq_len=256):
    """Create inputs without mask (baseline)."""
    vae_scale_factor = 16
    patch_size = 2

    latent_height = height // vae_scale_factor // patch_size
    latent_width = width // vae_scale_factor // patch_size
    num_latent_pixels = latent_height * latent_width

    hidden_states = torch.randn(batch_size, num_latent_pixels, 16, device=device, dtype=dtype)
    encoder_hidden_states = torch.randn(batch_size, text_seq_len, 16, device=device, dtype=dtype)
    timestep = torch.tensor([1.0], device=device, dtype=dtype).expand(batch_size)

    img_shapes = [(1, latent_height, latent_width)] * batch_size
    txt_seq_lens = [text_seq_len] * batch_size

    return {
        "hidden_states": hidden_states,
        "encoder_hidden_states": encoder_hidden_states,
        "timestep": timestep,
        "img_shapes": img_shapes,
        "txt_seq_lens": txt_seq_lens,
    }


def create_inputs_with_mask_full(batch_size, device, dtype, height=512, width=512, text_seq_len=256):
    """Create inputs with all-ones mask (no actual padding)."""
    inputs = create_inputs_no_mask(batch_size, device, dtype, height, width, text_seq_len)
    inputs["encoder_hidden_states_mask"] = torch.ones(
        batch_size, text_seq_len, dtype=torch.long, device=device
    )
    return inputs


def create_inputs_with_padding(batch_size, device, dtype, height=512, width=512, text_seq_len=256):
    """Create inputs with variable-length sequences (realistic CFG scenario)."""
    vae_scale_factor = 16
    patch_size = 2

    latent_height = height // vae_scale_factor // patch_size
    latent_width = width // vae_scale_factor // patch_size
    num_latent_pixels = latent_height * latent_width

    hidden_states = torch.randn(batch_size, num_latent_pixels, 16, device=device, dtype=dtype)
    encoder_hidden_states = torch.randn(batch_size, text_seq_len, 16, device=device, dtype=dtype)

    # Variable lengths: first is full, second is ~10% (simulates CFG with empty unconditional)
    actual_lengths = [text_seq_len, max(1, text_seq_len // 10)]
    encoder_hidden_states_mask = torch.zeros(batch_size, text_seq_len, dtype=torch.long, device=device)
    for i, length in enumerate(actual_lengths):
        encoder_hidden_states_mask[i, :length] = 1

    # Zero out padding
    mask_expanded = encoder_hidden_states_mask.unsqueeze(-1).to(dtype)
    encoder_hidden_states = encoder_hidden_states * mask_expanded

    timestep = torch.tensor([1.0], device=device, dtype=dtype).expand(batch_size)

    img_shapes = [(1, latent_height, latent_width)] * batch_size

    return {
        "hidden_states": hidden_states,
        "encoder_hidden_states": encoder_hidden_states,
        "encoder_hidden_states_mask": encoder_hidden_states_mask,
        "timestep": timestep,
        "img_shapes": img_shapes,
        "txt_seq_lens": actual_lengths,
    }


def measure_latency(model, inputs, num_warmup=5, num_runs=100):
    """Measure average latency with proper warmup."""
    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(**inputs)

    # Measure
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = model(**inputs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append(end - start)

    return {
        "mean_ms": sum(times) / len(times) * 1000,
        "std_ms": (sum((t - sum(times)/len(times))**2 for t in times) / len(times)) ** 0.5 * 1000,
        "min_ms": min(times) * 1000,
        "max_ms": max(times) * 1000,
    }


def measure_memory(model, inputs):
    """Measure peak memory usage."""
    flush()

    if not torch.cuda.is_available():
        return {"peak_memory_mb": 0}

    with torch.no_grad():
        # Warmup
        _ = model(**inputs)

    flush()

    with torch.no_grad():
        _ = model(**inputs)
        peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB

    return {"peak_memory_mb": peak_memory}


def benchmark_throughput(model, inputs, duration_seconds=10):
    """Measure throughput (iterations per second)."""
    num_iterations = 0
    start_time = time.perf_counter()

    with torch.no_grad():
        while time.perf_counter() - start_time < duration_seconds:
            _ = model(**inputs)
            num_iterations += 1
            if torch.cuda.is_available():
                torch.cuda.synchronize()

    elapsed = time.perf_counter() - start_time
    return {"iterations_per_sec": num_iterations / elapsed}


def run_benchmark_suite():
    """Run complete benchmark suite."""
    print("="*80)
    print("QwenImage Attention Mask Performance Benchmark")
    print("="*80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"CUDA Version: {torch.version.cuda}")
    print()

    results = []

    # Configuration (smaller for faster benchmarking)
    batch_size = 2
    height = 256  # Smaller resolution for faster benchmarking
    width = 256
    text_seq_len = 64  # Shorter sequences for faster benchmarking

    scenarios = [
        ("Baseline (no mask)", lambda m, d, dt: create_inputs_no_mask(batch_size, d, dt, height, width, text_seq_len)),
        ("Mask all-ones (no padding)", lambda m, d, dt: create_inputs_with_mask_full(batch_size, d, dt, height, width, text_seq_len)),
        ("Mask with padding (CFG)", lambda m, d, dt: create_inputs_with_padding(batch_size, d, dt, height, width, text_seq_len)),
    ]

    for scenario_name, input_fn in scenarios:
        print(f"\nBenchmarking: {scenario_name}")
        print("-" * 80)

        flush()
        model, device, dtype = get_model()
        inputs = input_fn(model, device, dtype)

        # Latency
        print("  Measuring latency...")
        latency = measure_latency(model, inputs, num_warmup=5, num_runs=50)

        # Memory
        print("  Measuring memory...")
        memory = measure_memory(model, inputs)

        # Throughput
        print("  Measuring throughput...")
        throughput = benchmark_throughput(model, inputs, duration_seconds=10)

        result = {
            "Scenario": scenario_name,
            "Batch Size": batch_size,
            "Latency (ms)": f"{latency['mean_ms']:.2f} ± {latency['std_ms']:.2f}",
            "Latency Mean (ms)": latency['mean_ms'],
            "Latency Std (ms)": latency['std_ms'],
            "Min Latency (ms)": latency['min_ms'],
            "Max Latency (ms)": latency['max_ms'],
            "Peak Memory (MB)": memory['peak_memory_mb'],
            "Throughput (iter/s)": throughput['iterations_per_sec'],
        }

        results.append(result)
        print(f"  Mean latency: {latency['mean_ms']:.2f} ms (± {latency['std_ms']:.2f})")
        print(f"  Peak memory: {memory['peak_memory_mb']:.1f} MB")
        print(f"  Throughput: {throughput['iterations_per_sec']:.2f} iter/s")

        del model
        flush()

    # Create DataFrame and save
    df = pd.DataFrame(results)

    print("\n" + "="*80)
    print("BENCHMARK RESULTS SUMMARY")
    print("="*80)
    print(df[["Scenario", "Latency (ms)", "Peak Memory (MB)", "Throughput (iter/s)"]].to_string(index=False))

    # Calculate overhead
    if len(results) >= 2:
        baseline_latency = results[0]['Latency Mean (ms)']
        mask_no_padding_latency = results[1]['Latency Mean (ms)']
        mask_with_padding_latency = results[2]['Latency Mean (ms)']

        overhead_no_padding = ((mask_no_padding_latency / baseline_latency) - 1) * 100
        overhead_with_padding = ((mask_with_padding_latency / baseline_latency) - 1) * 100

        print("\n" + "="*80)
        print("PERFORMANCE OVERHEAD ANALYSIS")
        print("="*80)
        print(f"Mask overhead (no padding): {overhead_no_padding:+.2f}%")
        print(f"Mask overhead (with padding): {overhead_with_padding:+.2f}%")

        if abs(overhead_no_padding) < 5:
            print("Negligible overhead for mask processing")
        elif overhead_no_padding < 0:
            print("Actually faster with mask (optimization opportunity)")
        else:
            print(f"WARNING: {overhead_no_padding:.1f}% overhead when using masks")

    # Save to CSV
    csv_filename = "qwen_mask_benchmark_results.csv"
    df.to_csv(csv_filename, index=False)
    print(f"\nResults saved to: {csv_filename}")

    return df


if __name__ == "__main__":
    df = run_benchmark_suite()
