import gc  
import torch 
import torch.utils.benchmark as benchmark
from dataclasses import dataclass
import argparse

@dataclass
class BenchmarkInfo:
    time: float 
    memory: float


def flush():
    gc.collect()
    torch.cuda.empty_cache()

def bytes_to_giga_bytes(bytes):
    return bytes / 1024 / 1024 / 1024


# Adapted from
# https://pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html
def benchmark_fn(f, *args, **kwargs):
    t0 = benchmark.Timer(
        stmt="f(*args, **kwargs)", globals={"args": args, "kwargs": kwargs, "f": f}
    )
    return f"{(t0.blocked_autorange().mean):.3f}" 

def generate_markdown_table(pipeline_name: str, args: argparse.Namespace, benchmark_info: BenchmarkInfo) -> str:
    headers = ["**Parameter**", "**Value**"]
    data = [
        ["Batch Size", args.batch_size],
        ["Number of Inference Steps", args.num_inference_steps],
        ["Run Compile", args.run_compile],
        ["Time (seconds)", benchmark_info.time],
        ["Memory (GBs)", benchmark_info.memory]
    ]

    # Formatting the table.
    markdown_table = f"## {pipeline_name}\n\n"
    markdown_table += "| " + " | ".join(headers) + " |\n"
    markdown_table += "|-" + "-|-".join(['' for _ in headers]) + "-|\n"
    for row in data:
        markdown_table += "| " + " | ".join(str(item) for item in row) + " |\n"

    return markdown_table