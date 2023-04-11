import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
from PIL import Image

model_id = "timbrooks/instruct-pix2pix"

pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker=None)
pipe.to("cuda")
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
# `image` is an RGB PIL.Image
image = Image.open("/home/tylerfeldman321/instruct-pix2pix/imgs/example.jpg")
images = pipe("turn him into cyborg", image=image).images
result = images[0]

result.save("result.jpg")
