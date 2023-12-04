import argparse
import csv
import gc
import os
from dataclasses import dataclass
from typing import Dict, List, Union

import torch
import torch.utils.benchmark as benchmark


GITHUB_SHA = os.getenv("GITHUB_SHA", None)
BENCHMARK_FIELDS = [
    "pipeline_cls",
    "ckpt_id",
    "batch_size",
    "num_inference_steps",
    "model_cpu_offload",
    "run_compile",
    "time (secs)",
    "memory (gbs)",
    "actual_gpu_memory (gbs)",
    "github_sha",
]

PROMPT = "ghibli style, a fantasy landscape with castles"
BASE_PATH = os.getenv("BASE_PATH", ".")
TOTAL_GPU_MEMORY = float(os.getenv("TOTAL_GPU_MEMORY", torch.cuda.get_device_properties(0).total_memory / (1024**3)))

REPO_ID = "diffusers/benchmarks"
FINAL_CSV_FILE = "collated_results.csv"


@dataclass
class BenchmarkInfo:
    time: float
    memory: float


def flush():
    """Wipes off memory."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()


def bytes_to_giga_bytes(bytes):
    return f"{(bytes / 1024 / 1024 / 1024):.3f}"


def benchmark_fn(f, *args, **kwargs):
    t0 = benchmark.Timer(
        stmt="f(*args, **kwargs)",
        globals={"args": args, "kwargs": kwargs, "f": f},
        num_threads=torch.get_num_threads(),
    )
    return f"{(t0.blocked_autorange().mean):.3f}"


def generate_csv_dict(
    pipeline_cls: str, ckpt: str, args: argparse.Namespace, benchmark_info: BenchmarkInfo
) -> Dict[str, Union[str, bool, float]]:
    """Packs benchmarking data into a dictionary for latter serialization."""
    data_dict = {
        "pipeline_cls": pipeline_cls,
        "ckpt_id": ckpt,
        "batch_size": args.batch_size,
        "num_inference_steps": args.num_inference_steps,
        "model_cpu_offload": args.model_cpu_offload,
        "run_compile": args.run_compile,
        "time (secs)": benchmark_info.time,
        "memory (gbs)": benchmark_info.memory,
        "actual_gpu_memory (gbs)": f"{(TOTAL_GPU_MEMORY):.3f}",
        "github_sha": GITHUB_SHA,
    }
    return data_dict


def write_to_csv(file_name: str, data_dict: Dict[str, Union[str, bool, float]]):
    """Serializes a dictionary into a CSV file."""
    with open(file_name, mode="w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=BENCHMARK_FIELDS)
        writer.writeheader()
        writer.writerow(data_dict)


def collate_csv(input_files: List[str], output_file: str):
    """Collates multiple identically structured CSVs into a single CSV file."""
    with open(output_file, mode="w", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=BENCHMARK_FIELDS)
        writer.writeheader()

        for file in input_files:
            with open(file, mode="r") as infile:
                reader = csv.DictReader(infile)
                for row in reader:
                    writer.writerow(row)
