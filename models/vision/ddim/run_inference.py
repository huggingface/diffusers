#!/usr/bin/env python3
# !pip install diffusers
import numpy as np

import PIL.Image
from modeling_ddim import DDIM


model_id = "fusing/ddpm-cifar10"
model_id = "fusing/ddpm-lsun-bedroom"

# load model and scheduler
ddpm = DDIM.from_pretrained(model_id)

# run pipeline in inference (sample random noise and denoise)
image = ddpm()

# process image to PIL
image_processed = image.cpu().permute(0, 2, 3, 1)
image_processed = (image_processed + 1.0) * 127.5
image_processed = image_processed.numpy().astype(np.uint8)
image_pil = PIL.Image.fromarray(image_processed[0])

# save image
image_pil.save("/home/patrick/images/show.png")
