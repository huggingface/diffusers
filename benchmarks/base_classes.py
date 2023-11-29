import os
import sys

import torch

from diffusers import AutoPipelineForImage2Image, AutoPipelineForText2Image
from diffusers.utils import load_image


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


RESOLUTION_MAPPING = {
    "runwayml/stable-diffusion-v1-5": (512, 512),
    "stabilityai/stable-diffusion-2-1": (768, 768),
    "stabilityai/stable-diffusion-xl-refiner-1.0": (1024, 1024),
}


class TextToImagePipeline:
    pipeline_class = AutoPipelineForText2Image

    def __init__(self, args):
        pipe = self.pipeline_class.from_pretrained(args.ckpt, torch_dtype=torch.float16, use_safetensors=True)
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

    def benchmark(self, args):
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


class ImageToImagePipeline(TextToImagePipeline):
    pipeline_class = AutoPipelineForImage2Image
    url = "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0f/1665_Girl_with_a_Pearl_Earring.jpg/800px-1665_Girl_with_a_Pearl_Earring.jpg"
    image = load_image(url).convert("RGB")

    def run_inference(self, pipe, args):
        self.image = self.image.resize(RESOLUTION_MAPPING[args.ckpt])

        _ = pipe(
            prompt=PROMPT,
            image=self.image,
            num_inference_steps=args.num_inference_steps,
            num_images_per_prompt=args.batch_size,
        )
