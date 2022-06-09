#!/usr/bin/env python3
import os
import pathlib

import numpy as np

import PIL.Image
from modeling_ddim import DDIM


model_ids = ["ddim-celeba-hq", "ddim-lsun-church", "ddim-lsun-bedroom"]

for model_id in model_ids:
    path = os.path.join("/home/patrick/images/hf", model_id)
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)

    ddpm = DDIM.from_pretrained("fusing/" + model_id)
    image = ddpm(batch_size=4)

    image_processed = image.cpu().permute(0, 2, 3, 1)
    image_processed = (image_processed + 1.0) * 127.5
    image_processed = image_processed.numpy().astype(np.uint8)

    for i in range(image_processed.shape[0]):
        image_pil = PIL.Image.fromarray(image_processed[i])
        image_pil.save(os.path.join(path, f"image_{i}.png"))
