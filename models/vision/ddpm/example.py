#!/usr/bin/env python3
import tempfile
import sys

from modeling_ddpm import DDPM

model_id = sys.argv[1]

ddpm = DDPM.from_pretrained(model_id)
image = ddpm()

import PIL.Image
import numpy as np
image_processed = image.cpu().permute(0, 2, 3, 1)
image_processed = (image_processed + 1.0) * 127.5
image_processed = image_processed.numpy().astype(np.uint8)
image_pil = PIL.Image.fromarray(image_processed[0])
image_pil.save("test.png")

import ipdb; ipdb.set_trace()
