#!/usr/bin/env python3
import tempfile
import sys

from diffusers import GaussianDDPMScheduler, UNetModel
from modeling_ddpm import DDPM

model_id = sys.argv[1]
folder = sys.argv[2]
save = bool(int(sys.argv[3]))

unet = UNetModel.from_pretrained(model_id)
sampler = GaussianDDPMScheduler.from_config(model_id)

# compose Diffusion Pipeline
if save:
    ddpm = DDPM(unet, sampler)
    ddpm.save_pretrained(folder)

image = ddpm()

import PIL.Image
import numpy as np
image_processed = image.cpu().permute(0, 2, 3, 1)
image_processed = (image_processed + 1.0) * 127.5
image_processed = image_processed.numpy().astype(np.uint8)
image_pil = PIL.Image.fromarray(image_processed[0])
image_pil.save("test.png")

import ipdb; ipdb.set_trace()
