import torch 
from fa3_processor import FA3AttnProcessor
from diffusers import DiffusionPipeline 
import argparse
import torch.utils.benchmark as benchmark
import gc
import json

def flush():
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

def load_pipeline(args):
    pipeline = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
    ).to("cuda")
    if args.fa3:
        pipeline.unet.set_attn_processor(FA3AttnProcessor())
        pipeline.vae.set_attn_processor(FA3AttnProcessor())

    pipeline.set_progress_bar_config(disable=True)
    return pipeline 

def run_pipeline(pipeline, args):
    _ = pipeline(
        prompt="a cat with tiger-like looks", 
        num_images_per_prompt=args.batch_size, 
        num_inference_steps=25, 
        guidance_scale=7.5
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fa3", default=0, type=int)
    parser.add_argument("--batch_size", default=1, type=int)
    args = parser.parse_args()

    flush()

    pipeline = load_pipeline(args)

    for _ in range(3):
        run_pipeline(pipeline, args)

    time = benchmark_fn(run_pipeline, pipeline, args)
    memory = bytes_to_giga_bytes(torch.cuda.max_memory_allocated()) 
    data_dict = dict(time=time, memory=memory)

    filename_prefix = f"fa3@{args.fa3}-bs@{args.batch_size}"
    with open(f"{filename_prefix}.json", "w") as f:
        json.dump(data_dict, f)

    image = pipeline(
        prompt="a cat with tiger-like looks", 
        num_images_per_prompt=args.batch_size, 
        num_inference_steps=25, 
        guidance_scale=7.5
    ).images[0]
    image.save(f"{filename_prefix}.png")