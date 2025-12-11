import torch
from diffusers import (
    StableDiffusionXLAdapterPipeline,
    AutoencoderKL,
    EulerAncestralDiscreteScheduler,
    GatedMultiAdapter
)
from diffusers.utils import load_image
from diffusers import StableDiffusionXLAdapterPipeline, T2IAdapter, EulerAncestralDiscreteScheduler, AutoencoderKL,MultiAdapter,GatedMultiAdapter
from diffusers.utils import load_image, make_image_grid
from controlnet_aux.midas import MidasDetector
import torch
from controlnet_aux.canny import CannyDetector

device = "cuda"

from diffusers import StableDiffusionXLAdapterPipeline, T2IAdapter, EulerAncestralDiscreteScheduler, AutoencoderKL, MultiAdapter
from diffusers.utils import load_image
from controlnet_aux.lineart import LineartDetector

from tqdm import tqdm
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import os
from PIL import Image
from datasets import load_dataset
# --------------------------
# Load HF Dataset
# --------------------------
ds = load_dataset("hengyiqun/10623-project-coco17", split="val[4500:]")

# -----------------------------
# 1. Load trained GatedMultiAdapter
# -----------------------------
adapters = GatedMultiAdapter.from_pretrained(
    "/home/ubuntu/gate-your-sketch-training_output/sdxl_GMA_withFiLM_t1.0_res512_lr1e-7_bs1x4_seed42_step2000/t2iadapter",
    torch_dtype=torch.float16,
)

# load euler_a scheduler
model_id = 'stabilityai/stable-diffusion-xl-base-1.0'
euler_a = EulerAncestralDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
vae=AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
pipe = StableDiffusionXLAdapterPipeline.from_pretrained(
model_id, vae=vae, adapter=adapters, scheduler=euler_a, torch_dtype=torch.float16, variant="fp16",
).to(device)


import io

output_dir = "/home/ubuntu/GMA_inference_outputs"

os.makedirs(output_dir, exist_ok = True)
os.makedirs(output_dir+'/t5', exist_ok = True)
os.makedirs(output_dir+'/t10', exist_ok = True)
os.makedirs(output_dir+'/t15', exist_ok = True)
os.makedirs(output_dir+'/t30', exist_ok = True)

# --------------------------
# Inference Loop
# --------------------------
negative_prompt = "anime, cartoon, graphic, text, painting, crayon, graphite, abstract, glitch, deformed, mutated, ugly, disfigured"
for idx, item in tqdm(enumerate(ds), total=len(ds)):
    if idx <= 46:
        continue
    # item = {'image_id': 521601, 'text': "A small doughnut inside a cup that's sitting on a table.", 'image': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=640x480 at 0x799D7244B260>, 'sketch': <PIL.PngImagePlugin.PngImageFile image mode=RGB size=1344x1024 at 0x799D724480B0>, 'depth': <PIL.PngImagePlugin.PngImageFile image mode=L size=1408x1024 at 0x799D724486E0>}

    sketch_img = item["sketch"].convert("RGB")
    depth_img = item["depth"].convert("RGB")
    prompt = item["text"]

    # gen = pipe(
    #     prompt=prompt,
    #     negative_prompt=negative_prompt,
    #     image=[sketch_img, depth_img], # first: sketch, secound: depth
    #     num_inference_steps=10,
    #     guidance_scale=7.5,
    # ).images[0]

    # gen.save(f"{output_dir}/t10/{4500+idx:05d}.png")

    # gen = pipe(
    #     prompt=prompt,
    #     negative_prompt=negative_prompt,
    #     image=[sketch_img, depth_img], # first: sketch, secound: depth
    #     num_inference_steps=15,
    #     guidance_scale=7.5,
    # ).images[0]

    # gen.save(f"{output_dir}/t15/{4500+idx:05d}.png")

    # gen = pipe(
    #     prompt=prompt,
    #     negative_prompt=negative_prompt,
    #     image=[sketch_img, depth_img], # first: sketch, secound: depth
    #     num_inference_steps=5,
    #     guidance_scale=7.5,
    # ).images[0]
    # gen.save(f"{output_dir}/t5/{4500+idx:05d}.png")

    gen = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=[sketch_img, depth_img], # first: sketch, secound: depth
        num_inference_steps=30,
        guidance_scale=7.5,
    ).images[0]

    gen.save(f"{output_dir}/t30/{4500+idx:05d}.png")

    

