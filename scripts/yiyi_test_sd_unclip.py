from io import BytesIO

import requests
import torch
from PIL import Image

from diffusers import StableUnCLIPImg2ImgPipeline


pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1-unclip",
    torch_dtype=torch.float16,
    variant="fp16",
)
pipe = pipe.to("cuda")

url = "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/stable_unclip/tarsila_do_amaral.png"

response = requests.get(url)
init_image = Image.open(BytesIO(response.content)).convert("RGB")

images = pipe(init_image).images
images[0].save("fantasy_landscape.png")
