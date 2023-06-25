#!/usr/bin/env python3
from diffusers import DiffusionPipeline, EulerDiscreteScheduler, StableDiffusionPipeline, KDPM2DiscreteScheduler, StableDiffusionImg2ImgPipeline, HeunDiscreteScheduler, KDPM2AncestralDiscreteScheduler, DDIMScheduler
import time
import os
from huggingface_hub import HfApi
# from compel import Compel
import torch
import sys
from pathlib import Path
import requests
from PIL import Image
from io import BytesIO

path = sys.argv[1]

api = HfApi()
start_time = time.time()
pipe = DiffusionPipeline.from_pretrained(path, torch_dtype=torch.float16)
pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
# pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
# pipe = StableDiffusionImg2ImgPipeline.from_pretrained(path, torch_dtype=torch.float16, safety_checker=None

# compel = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder)


pipe = pipe.to("cuda")

prompt = "An astronaut riding a green horse on Mars"

# rompts = ["a cat playing with a ball++ in the forest", "a cat playing with a ball++ in the forest", "a cat playing with a ball-- in the forest"]

# prompt_embeds = torch.cat([compel.build_conditioning_tensor(prompt) for prompt in prompts])

# generator = [torch.Generator(device="cuda").manual_seed(0) for _ in range(prompt_embeds.shape[0])]
#
# url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"
# 
# response = requests.get(url)
# image = Image.open(BytesIO(response.content)).convert("RGB")
# image.thumbnail((768, 768))
#

# pipe.unet.set_default_attn_processor()
image = pipe(prompt=prompt).images[0]

file_name = f"aaa"
path = os.path.join(Path.home(), "images", f"{file_name}.png")
image.save(path)

api.upload_file(
    path_or_fileobj=path,
    path_in_repo=path.split("/")[-1],
    repo_id="patrickvonplaten/images",
    repo_type="dataset",
)
print(f"https://huggingface.co/datasets/patrickvonplaten/images/blob/main/{file_name}.png")
