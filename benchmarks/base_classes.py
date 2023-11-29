import os
import sys

import torch

from diffusers import DiffusionPipeline


sys.path.append(".")

from benchmarks.utils import (  # noqa: E402
    BASE_PATH,
    PROMPT,
    BenchmarkInfo,
    benchmark_fn,
    bytes_to_giga_bytes,
    generate_csv_dict,
    write_to_csv,
)


class TextToImagePipeline:
    def __init__(self, args):
        pipe = DiffusionPipeline.from_pretrained(args.ckpt, torch_dtype=torch.float16, use_safetensors=True)
        pipe = pipe.to("cuda")

        if args.run_compile:
            pipe.unet.to(memory_format=torch.channels_last)
            print("Run torch compile")
            pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)

        pipe.set_progress_bar_config(disable=True)
        self.pipe = pipe

    def run_inference(self, pipe, args):
        _ = pipe(
            prompt=PROMPT,
            num_inference_steps=args.num_inference_steps,
            num_images_per_prompt=args.batch_size,
        )

    def __call__(self, args):
        time = benchmark_fn(self.run_inference, self.pipe, args)  # in seconds.
        memory = bytes_to_giga_bytes(torch.cuda.max_memory_allocated())  # in GBs.
        benchmark_info = BenchmarkInfo(time=time, memory=memory)

        csv_dict = generate_csv_dict(
            pipeline_cls=str(self.pipe.__class__.__name__), ckpt=args.ckpt, args=args, benchmark_info=benchmark_info
        )
        name = (
            args.ckpt.replace("/", "_")
            + f"-bs@{args.batch_size}-steps@{args.num_inference_steps}-mco@{args.model_cpu_offload}-compile@{args.run_compile}.csv"
        )
        filepath = os.path.join(BASE_PATH, name)
        write_to_csv(filepath, csv_dict)
        print(f"Logs written to: {filepath}")
