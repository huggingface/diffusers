#!/usr/bin/env python3
from diffusers import PNDM, UNetModel, PNDMScheduler
import PIL.Image
import numpy as np
import torch

model_id = "fusing/ddim-celeba-hq"

model = UNetModel.from_pretrained(model_id)
scheduler = PNDMScheduler()

# load model and scheduler
ddpm = PNDM(unet=model, noise_scheduler=scheduler)

# run pipeline in inference (sample random noise and denoise)
image = ddpm()

# process image to PIL
image_processed = image.cpu().permute(0, 2, 3, 1)
image_processed = (image_processed + 1.0) / 2
image_processed = torch.clamp(image_processed, 0.0, 1.0)
image_processed = image_processed * 255
image_processed = image_processed.numpy().astype(np.uint8)
image_pil = PIL.Image.fromarray(image_processed[0])

# save image
image_pil.save("/home/patrick/images/test.png")
