#!/usr/bin/env python3
from diffusers import UNetModel, GaussianDiffusion
from modeling_ddpm import DDPM
import tempfile

unet = UNetModel.from_pretrained("fusing/ddpm_dummy")
sampler = GaussianDiffusion.from_config("fusing/ddpm_dummy")

# compose Diffusion Pipeline
ddpm = DDPM(unet, sampler)
# generate / sample
image = ddpm()
print(image)


# save and load with 0 extra code (handled by general `DiffusionPipeline` class)
with tempfile.TemporaryDirectory() as tmpdirname:
    ddpm.save_pretrained(tmpdirname)
    print("Model saved")
    ddpm_new = DDPM.from_pretrained(tmpdirname)
    print("Model loaded")
    print(ddpm_new)
