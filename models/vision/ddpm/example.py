#!/usr/bin/env python3
import tempfile
import sys
import os
import pathlib
from modeling_ddpm import DDPM
import PIL.Image
import numpy as np

model_ids = ["ddpm-lsun-cat", "ddpm-lsun-cat-ema", "ddpm-lsun-church-ema", "ddpm-lsun-church", "ddpm-lsun-bedroom", "ddpm-lsun-bedroom-ema", "ddpm-cifar10-ema", "ddpm-lsun-cifar10", "ddpm-lsun-celeba-hq", "ddpm-lsun-celeba-hq-ema"]

for model_id in model_ids:

    path = os.path.join("/home/patrick/images/hf", model_id)
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)

    ddpm = DDPM.from_pretrained("fusing/" + model_id)
    image = ddpm(batch_size=4)

    image_processed = image.cpu().permute(0, 2, 3, 1)
    image_processed = (image_processed + 1.0) * 127.5
    image_processed = image_processed.numpy().astype(np.uint8)

    for i in range(image_processed.shape[0]):
        image_pil = PIL.Image.fromarray(image_processed[i])
        image_pil.save(os.path.join(path, f"image_{i}.png"))
