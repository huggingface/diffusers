"""
Pass path to image as the first argument, and the Path to the mask as the second,
the Prompt third, and the path to save fourth.
"""
import sys
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
from PIL import Image
import numpy as np

model_id = 'timbrooks/instruct-pix2pix'

pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker=None)
pipe.to('cuda')
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
img = Image.open(sys.argv[1])
mask_init = Image.open(sys.argv[2])
mask_scale = float(sys.argv[5])
mask_guidance_scale = float(sys.argv[6])
mask_frequency = int(sys.argv[7])
mask_numpy = np.array(mask_init)
mask_int = mask_numpy.astype(int)
mask = mask_int / mask_int.max()
images = pipe(sys.argv[3], image=img, mask=mask, mask_guidance_scale=mask_scale,
              guidance_scale=mask_guidance_scale, mask_enforcement_frequency=mask_frequency).images
result = images[0]

result.save(sys.argv[4])
