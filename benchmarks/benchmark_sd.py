import argparse
import os
import sys

import torch

from diffusers import DiffusionPipeline


sys.path.append(".")
from benchmark_utils import (  # noqa: E402
    BASE_PATH,
    PROMPT,
    BenchmarkInfo,
    benchmark_fn,
    bytes_to_giga_bytes,
    generate_csv_dict,
    write_to_csv,
)


CKPT = "CompVis/stable-diffusion-v1-4"


def load_pipeline(run_compile=False):
    pipe = DiffusionPipeline.from_pretrained(CKPT, torch_dtype=torch.float16, use_safetensors=True)
    pipe = pipe.to("cuda")

    if run_compile:
        pipe.unet.to(memory_format=torch.channels_last)
        print("Run torch compile")
        pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)

    pipe.set_progress_bar_config(disable=True)
    return pipe


def run_inference(pipe, args):
    _ = pipe(
        prompt=PROMPT,
        num_inference_steps=args.num_inference_steps,
        num_images_per_prompt=args.batch_size,
    )


def main(args) -> dict:
    pipeline = load_pipeline(run_compile=args.run_compile)

    time = benchmark_fn(run_inference, pipeline, args)  # in seconds.
    memory = bytes_to_giga_bytes(torch.cuda.max_memory_allocated())  # in GBs.
    benchmark_info = BenchmarkInfo(time=time, memory=memory)

    csv_dict = generate_csv_dict(
        pipeline_cls=str(pipeline.__class__.__name__), ckpt=CKPT, args=args, benchmark_info=benchmark_info
    )
    return csv_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--model_cpu_offload", action="store_true")
    parser.add_argument("--run_compile", action="store_true")
    args = parser.parse_args()
    csv_dict = main(args)

    name = (
        CKPT.replace("/", "_")
        + f"-bs@{args.batch_size}-steps@{args.num_inference_steps}-mco@{args.model_cpu_offload}-compile@{args.run_compile}.csv"
    )
    filepath = os.path.join(BASE_PATH, name)
    write_to_csv(filepath, csv_dict)
