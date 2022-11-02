import os
from io import BytesIO

import torch

import requests
from diffusers import DiffusionPipeline
from PIL import Image


has_cuda = torch.cuda.is_available()
device = torch.device("cpu" if not has_cuda else "cuda")
pipe = DiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    safety_checker=None,
    use_auth_token=True,
    # custom_pipeline="imagic_stable_diffusion",
    custom_pipeline="/home/mark/open_source/diffusers/examples/community/clip_telephone",
).to(device)

generator = torch.Generator("cuda").manual_seed(0)
seed = 0
url = "https://www.dropbox.com/s/02iu769q7hm4o4s/painted-turtle-rebecca-wang.jpeg?dl=1"
response = requests.get(url)
init_image = Image.open(BytesIO(response.content)).convert("RGB")
init_image = init_image.resize((512, 512))

res = pipe(image=init_image, guidance_scale=7.5, num_inference_steps=50, generator=generator)
prompt = res.prompt

if not os.path.exists("./images"):
    os.makedirs("./images")

for i in range(0, 10):
    res = pipe(prompt=prompt, guidance_scale=7.5, num_inference_steps=50, generator=generator)
    prompt = res.prompt
    print("prompt for {i} is {prompt}".format(i=i, prompt=prompt))
    image = res.images[0]
    image.save("./images/clip_telephone_{i}.png".format(i=i))
