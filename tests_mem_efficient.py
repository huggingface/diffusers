''' Run a simple benchmark to test

#  To Run without the Memory Efficient Attention
python test.py

# To Run with the Memory Efficient Attention
USE_MEMORY_EFFICIENT_ATTENTION=1 python test.py

'''

import torch
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
   "CompVis/stable-diffusion-v1-4", revision="fp16", torch_dtype=torch.float16, use_auth_token=True
).to("cuda")

with torch.inference_mode(), torch.autocast("cuda"):
   image = pipe("a big dog standing on the eiffel tower")