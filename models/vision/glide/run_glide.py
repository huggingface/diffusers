import torch
from .modeling_glide import GLIDE
from diffusers import UNetGLIDEModel, GaussianDDPMScheduler

generator = torch.Generator()
generator = generator.manual_seed(0)

# 1. Load models

scheduler = GaussianDDPMScheduler.from_config("fusing/glide-base")
model = UNetGLIDEModel.from_pretrained("fusing/glide-base")

pipeline = GLIDE(model, scheduler)

img = pipeline(generator)

print(img)
