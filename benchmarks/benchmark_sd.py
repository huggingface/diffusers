import argparse
import os
import torch
from diffusers import DiffusionPipeline
from .benchmark_utils import benchmark_fn, bytes_to_giga_bytes, BenchmarkInfo, generate_markdown_table

CKPT = "CompVis/stable-diffusion-v1-4"
PROMPT = "ghibli style, a fantasy landscape with castles"
BASE_PATH = "benchmark_outputs"


def load_pipeline(run_compile=False, with_tensorrt=False):
    pipe = DiffusionPipeline.from_pretrained(
        CKPT, torch_dtype=torch.float16, use_safetensors=True
    )
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

def main(args):
    pipeline = load_pipeline(
        run_compile=args.run_compile, with_tensorrt=args.with_tensorrt
    )
    
    time = benchmark_fn(run_inference, pipeline, args) # in seconds.
    memory = bytes_to_giga_bytes(torch.cuda.max_memory_allocated()) # in GBs.
    benchmark_info = BenchmarkInfo(time=time, memory=memory)
    
    markdown_report = ""
    markdown_report = generate_markdown_table(pipeline_name=CKPT, args=args, benchmark_info=benchmark_info, markdown_report=markdown_report)  
    return markdown_report

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--run_compile", action="store_true")
    args = parser.parse_args()
    markdown_report = main(args)

    name = CKPT + f"-batch_sze@{args.batch_size}-num_inference_steps@{args.num_inference_steps}--run_compile@{args.run_compile}"
    filepath = os.path.join(BASE_PATH, name)
    with open(filepath, "w") as f:
        f.write(markdown_report)
    
    