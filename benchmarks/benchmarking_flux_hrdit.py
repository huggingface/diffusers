"""Benchmark HRDiT training-free high-resolution generation against naive single-pass generation.

HRDiT (https://arxiv.org/abs/2608.07003) generates high-resolution images on off-the-shelf FLUX.1-dev
without fine-tuning, via NTK-aware RoPE scaling, Spatial Position Alignment (SPA) and a structure-guided
progressive 1024 -> 2048 -> 4096 ladder. This script times (and measures peak memory of) the end-to-end
call for a naive single-pass FLUX.1-dev baseline and for the HRDiT pipeline at the same target resolution.

Run on a GPU machine with the FLUX.1-dev checkpoint available:

    python benchmarks/benchmarking_flux_hrdit.py --height 4096 --width 4096
"""

import argparse
from pathlib import Path

import torch
from benchmarking_utils import benchmark_fn, flush

from diffusers import FluxPipeline
from diffusers.utils.testing_utils import torch_device


CKPT_ID = "black-forest-labs/FLUX.1-dev"
CUSTOM_PIPELINE_PATH = str(Path(__file__).resolve().parents[1] / "examples" / "community" / "pipeline_flux_hrdit.py")
RESULT_FILENAME = "flux_hrdit.csv"
PROMPT = "a photo of a mountain lake at dawn"


def load_pipeline():
    return FluxPipeline.from_pretrained(
        CKPT_ID,
        dtype=torch.bfloat16,
        custom_pipeline=CUSTOM_PIPELINE_PATH,
    ).to(torch_device)


def _peak_memory_gib():
    return torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else float("nan")


def run_benchmarks(height, width, num_inference_steps):
    hrdit_pipe = load_pipeline()
    # Stock single-pass FLUX.1-dev baseline, sharing the loaded components.
    naive_pipe = FluxPipeline(**hrdit_pipe.components)

    settings = {
        # Naive: generate straight at the target resolution in one pass (stock FluxPipeline).
        "naive": (naive_pipe, {"height": height, "width": width, "num_inference_steps": num_inference_steps}),
        # HRDiT: NTK RoPE + SPA + structure-guided progressive ladder up to the target resolution.
        "hrdit": (hrdit_pipe, {"height": height, "width": width, "num_inference_steps": num_inference_steps}),
    }

    results = []
    for name, (pipe, kwargs) in settings.items():
        flush()
        latency = benchmark_fn(pipe, PROMPT, **kwargs)
        max_memory = _peak_memory_gib()
        results.append((name, latency, max_memory))
        print(f"{name:>6}: {latency:.3f}s, peak memory {max_memory:.2f} GiB")

    with open(RESULT_FILENAME, "w") as f:
        f.write("setting,latency_s,peak_memory_gib\n")
        for name, latency, max_memory in results:
            f.write(f"{name},{latency},{max_memory}\n")
    print(f"Results saved to {RESULT_FILENAME}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--height", type=int, default=4096)
    parser.add_argument("--width", type=int, default=4096)
    parser.add_argument("--num_inference_steps", type=int, default=30)
    args = parser.parse_args()

    run_benchmarks(args.height, args.width, args.num_inference_steps)
