"""
Pass path to image as the first argument, and the Path to the mask as the second,
the Prompt third, and the path to save fourth.
"""
import sys
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
from PIL import Image

model_id = 'timbrooks/instruct-pix2pix'

pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker=None)
pipe.to('cuda')
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
img = Image.open(sys.argv[1])
mask = Image.open(sys.argv[2])
images = pipe(sys.argv[3], image=img, mask=mask, mask_guidance_scale=0.2).images
result = images[0]

result.save(sys.argv[4])
